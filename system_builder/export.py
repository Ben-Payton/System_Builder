"""Write portable, architecture-independent outputs for pip-only OpenMM scripts."""
from __future__ import annotations

import datetime
import json
import logging
import platform as _platform
from importlib import metadata
from pathlib import Path

import numpy as np
import openmm
from openmm import app, unit
from openmm.app import Topology

import system_builder

from system_builder.config import BuildConfig

log = logging.getLogger(__name__)


_METHODS = {
    "PME": app.PME,
    "NOCUTOFF": app.NoCutoff,
    "CUTOFFPERIODIC": app.CutoffPeriodic,
    "CUTOFFNONPERIODIC": app.CutoffNonPeriodic,
    "LJPME": app.LJPME,
    "EWALD": app.Ewald,
}
_CONSTRAINTS = {None: None, "HBONDS": app.HBonds, "ALLBONDS": app.AllBonds, "HANGLES": app.HAngles}


def write_xml(system: openmm.System, path: Path) -> Path:
    """XmlSerializer.serialize(system) -> path."""
    path = Path(path)
    path.write_text(openmm.XmlSerializer.serialize(system))
    return path


def write_pdb(topology: Topology, positions: unit.Quantity, path: Path) -> Path:
    """PDB with CRYST1 box and CONECT records.

    keepIds=True keeps our residue/atom names; CONECT records are written by
    OpenMM for bonds in residues that are not standard PDB residues.
    """
    path = Path(path)
    with open(path, "w") as f:
        app.PDBFile.writeFile(topology, positions, f, keepIds=True)
    return path


# Constrained bonds (e.g. HBonds, rigid water) have no HarmonicBondForce entry in the
# OpenMM system, so parmed leaves them untyped and cannot write a prmtop. Give them
# an equilibrium length from the constraint and a typical force constant. This only
# matters if the user later loads the prmtop *without* constraints.
_K_XH = 282000.0  # generic X-H, approx. GAFF c3-hc
_K_OTHER = 250000.0


_WATER_K = 553.0  # kcal/mol/A^2, Amber TIP3P OW-HW and HW-HW bonds (as tleap writes them)
_WATER_ANGLE_K = 100.0  # kcal/mol/rad^2
_WATER_THETA = 104.52  # degrees


def _fill_constrained_bond_types(struct, system: openmm.System) -> None:
    """Make constraint-only terms explicit so the prmtop reproduces the XML system.

    OpenMM represents rigid water (and HBonds-constrained bonds) by constraints
    only: no bond/angle force. An Amber prmtop needs real terms, so:
    - rigid 3-atom O/H/H residues (all three pairs constrained) get Amber TIP3P
      terms as tleap writes them: O-H and H-H bonds (k=553 kcal/mol/A^2) and an
      H-O-H angle (k=100 kcal/mol/rad^2, 104.52 deg). AmberPrmtopFile.createSystem
      then rebuilds the rigid water with rigidWater=True, or flexible water
      with rigidWater=False.
    - other constrained bonds get the constraint length and a typical force constant.
    - a constraint between atoms that are not bonded raises ValueError.
    """
    import parmed

    dist = {}
    for i in range(system.getNumConstraints()):
        a, b, d = system.getConstraintParameters(i)
        dist[frozenset((a, b))] = d.value_in_unit(unit.nanometer)

    handled = set()
    for res in struct.residues:
        atoms = res.atoms
        if len(atoms) != 3:
            continue
        o = [a for a in atoms if a.atomic_number == 8]
        h = [a for a in atoms if a.atomic_number == 1]
        if len(o) != 1 or len(h) != 2:
            continue
        pairs = [frozenset((o[0].idx, h[0].idx)), frozenset((o[0].idx, h[1].idx)),
                 frozenset((h[0].idx, h[1].idx))]
        if not all(p in dist for p in pairs):
            continue
        r_oh = 0.5 * (dist[pairs[0]] + dist[pairs[1]]) * 10.0
        r_hh = dist[pairs[2]] * 10.0
        theta = 2.0 * np.degrees(np.arcsin(r_hh / (2.0 * r_oh)))
        bt_oh = parmed.BondType(_WATER_K, r_oh)
        bt_hh = parmed.BondType(_WATER_K, r_hh)
        at = parmed.AngleType(_WATER_ANGLE_K, theta)
        struct.bond_types.extend([bt_oh, bt_hh])
        struct.angle_types.append(at)
        for bond in list(o[0].bonds):
            bond.type = bt_oh
        struct.bonds.append(parmed.Bond(h[0], h[1], type=bt_hh))
        struct.angles.append(parmed.Angle(h[0], o[0], h[1], type=at))
        handled.update(pairs)

    cache = {}
    for bond in struct.bonds:
        if bond.type is not None:
            continue
        a1, a2 = bond.atom1, bond.atom2
        r0 = dist.get(frozenset((a1.idx, a2.idx)))
        if r0 is None:
            raise ValueError(f"bond {a1.idx}-{a2.idx} has neither a bond force nor a constraint")
        k = _K_XH if 1 in {a1.atomic_number, a2.atomic_number} else _K_OTHER
        key = (round(k), round(r0, 6))
        if key not in cache:
            cache[key] = parmed.BondType(k / (2 * 4.184 * 100.0), r0 * 10.0)
            struct.bond_types.append(cache[key])
        bond.type = cache[key]
        handled.add(frozenset((a1.idx, a2.idx)))

    struct.bond_types.claim()
    struct.angle_types.claim()
    bonded = {frozenset((b.atom1.idx, b.atom2.idx)) for b in struct.bonds}
    for pair in dist:
        if pair not in bonded:
            i, j = sorted(pair)
            raise ValueError(
                f"constraint between non-bonded atoms {i}-{j} cannot be represented in an Amber prmtop"
            )


def write_amber(
    system: openmm.System, topology: Topology, positions: unit.Quantity, prmtop: Path, inpcrd: Path
) -> tuple[Path, Path]:
    """Amber prmtop/inpcrd via parmed.openmm.load_topology(...).save()."""
    import parmed
    from parmed.openmm import load_topology

    prmtop, inpcrd = Path(prmtop), Path(inpcrd)
    struct = load_topology(topology, system, xyz=positions)
    n_res = topology.getNumResidues()
    if len(struct.residues) != n_res:
        raise ValueError(
            f"parmed produced {len(struct.residues)} residues but the topology has {n_res}; "
            "residues must have unique ids (residues with identical chain/id/name get merged)"
        )
    _fill_constrained_bond_types(struct, system)
    struct.save(str(prmtop), format="amber", overwrite=True)
    struct.save(str(inpcrd), format="rst7", overwrite=True)
    return prmtop, inpcrd


def _energy(system: openmm.System, positions, box) -> float:
    ctx = openmm.Context(
        system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
    )
    if box is not None:
        ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(positions)
    e = ctx.getState(getEnergy=True).getPotentialEnergy()
    del ctx
    return e.value_in_unit(unit.kilojoule_per_mole)


def _constraints(system: openmm.System) -> dict:
    out = {}
    for i in range(system.getNumConstraints()):
        a, b, d = system.getConstraintParameters(i)
        out[frozenset((a, b))] = d.value_in_unit(unit.nanometer)
    return out


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def check_energy_consistency(
    system: openmm.System, prmtop: Path, inpcrd: Path, cfg: BuildConfig, rtol: float = 1e-3
) -> dict:
    """Compare the XML system against a system rebuilt from the prmtop.

    Compares potential energy at the exported geometry and at a deterministically
    perturbed, constraint-satisfying geometry (catches missing bond/angle terms that are zero at
    equilibrium), plus the number and lengths of constraints.

    Returns {"xml_kj_mol", "prmtop_kj_mol", "rel_diff", "perturbed_xml_kj_mol",
    "perturbed_prmtop_kj_mol", "perturbed_rel_diff", "n_constraints_xml",
    "n_constraints_prmtop", "constraint_distance_maxdiff_nm", "ok"}.
    """
    prm = app.AmberPrmtopFile(str(prmtop))
    crd = app.AmberInpcrdFile(str(inpcrd))
    method = _METHODS[cfg.nonbonded_method.upper()]
    constraints = _CONSTRAINTS[cfg.constraints.upper() if cfg.constraints else None]
    kwargs = {}
    if method is not app.NoCutoff:
        kwargs["nonbondedCutoff"] = cfg.cutoff_nm * unit.nanometer
    sys2 = prm.createSystem(nonbondedMethod=method, constraints=constraints, **kwargs)
    # Round-trip the xml system so we test the file content, not the live object.
    sys1 = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    box = crd.boxVectors
    pos = crd.positions.value_in_unit(unit.nanometer)
    rng = np.random.default_rng(12345)
    pert = (np.array(pos) + rng.normal(scale=0.01, size=(len(pos), 3))) * unit.nanometer
    # Project onto the constraints so rigid terms (e.g. the kept H-O-H angle of
    # constrained water) are not probed at unphysical geometries.
    pctx = openmm.Context(sys1, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    if box is not None:
        pctx.setPeriodicBoxVectors(*box)
    pctx.setPositions(pert)
    pctx.applyConstraints(1e-8)
    pert = pctx.getState(getPositions=True).getPositions()
    del pctx
    e1, e2 = _energy(sys1, crd.positions, box), _energy(sys2, crd.positions, box)
    p1, p2 = _energy(sys1, pert, box), _energy(sys2, pert, box)
    c1, c2 = _constraints(sys1), _constraints(sys2)
    if set(c1) == set(c2):
        maxdiff = max((abs(c1[k] - c2[k]) for k in c1), default=0.0)
    else:
        maxdiff = float("inf")
    rel, prel = _rel(e1, e2), _rel(p1, p2)
    ok = rel <= rtol and prel <= rtol and set(c1) == set(c2) and maxdiff < 1e-4
    return {
        "xml_kj_mol": e1, "prmtop_kj_mol": e2, "rel_diff": rel,
        "perturbed_xml_kj_mol": p1, "perturbed_prmtop_kj_mol": p2, "perturbed_rel_diff": prel,
        "n_constraints_xml": len(c1), "n_constraints_prmtop": len(c2),
        "constraint_distance_maxdiff_nm": maxdiff, "ok": bool(ok),
    }


def _versions() -> dict:
    out = {"system_builder": system_builder.__version__, "openmm": openmm.__version__,
           "python": _platform.python_version()}
    for key, dist in [
        ("openff-toolkit", "openff-toolkit"),
        ("openff-interchange", "openff-interchange"),
        ("openmmforcefields", "openmmforcefields"),
        ("parmed", "ParmEd"),
    ]:
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[key] = None
    out["ambertools"] = _ambertools_version()
    return out


def _ambertools_version():
    import sys

    try:
        for f in (Path(sys.prefix) / "conda-meta").glob("ambertools-*.json"):
            return json.loads(f.read_text()).get("version")
    except Exception:  # noqa: BLE001 - best effort only
        pass
    return None


def _net_charge(system: openmm.System) -> float:
    total = 0.0
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            for i in range(force.getNumParticles()):
                total += force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
    return total


def write_metadata(path: Path, cfg: BuildConfig, system: openmm.System, topology: Topology, extra: dict | None = None) -> Path:
    """metadata.json: config echo, package versions, atom count, box, net charge, warnings."""
    path = Path(path)
    box = system.getDefaultPeriodicBoxVectors()
    charge = _net_charge(system)
    warns = []
    if abs(charge) > 1e-3:
        warns.append(f"system is not neutral: net charge {charge:+.4f} e")
    meta = {
        "config": cfg.raw,
        "versions": _versions(),
        "n_atoms": topology.getNumAtoms(),
        "n_residues": topology.getNumResidues(),
        "box_vectors_nm": [[float(c) for c in b.value_in_unit(unit.nanometer)] for b in box],
        "net_charge": charge,
        "warnings": warns,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    extra = dict(extra or {})
    meta["warnings"] += list(extra.pop("warnings", []))
    meta.update(extra)
    path.write_text(json.dumps(meta, indent=2))
    return path


def export_all(
    system: openmm.System,
    topology: Topology,
    positions: unit.Quantity,
    cfg: BuildConfig,
    out_dir: Path,
) -> dict[str, Path]:
    """Write system.xml, system.pdb, system.prmtop, system.inpcrd, metadata.json into out_dir.

    Returns {"xml": ..., "pdb": ..., "prmtop": ..., "inpcrd": ..., "metadata": ...}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "xml": write_xml(system, out_dir / "system.xml"),
        "pdb": write_pdb(topology, positions, out_dir / "system.pdb"),
    }
    paths["prmtop"], paths["inpcrd"] = write_amber(
        system, topology, positions, out_dir / "system.prmtop", out_dir / "system.inpcrd"
    )
    extra: dict = {"warnings": []}
    try:
        res = check_energy_consistency(system, paths["prmtop"], paths["inpcrd"], cfg)
        extra["energy_check"] = res
        if not res["ok"]:
            msg = (f"XML and prmtop energies differ: {res['xml_kj_mol']:.3f} vs "
                   f"{res['prmtop_kj_mol']:.3f} kJ/mol (rel diff {res['rel_diff']:.2e})")
            log.warning(msg)
            extra["warnings"].append(msg)
    except Exception as exc:  # noqa: BLE001 - never fail the export over the check
        msg = f"energy consistency check failed to run: {exc}"
        log.warning(msg)
        extra["energy_check"] = {"ok": False, "error": str(exc)}
        extra["warnings"].append(msg)
    paths["metadata"] = write_metadata(out_dir / "metadata.json", cfg, system, topology, extra)
    return paths
