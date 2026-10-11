"""Write portable, architecture-independent outputs for pip-only OpenMM scripts."""
from __future__ import annotations

import datetime
import json
import logging
import platform as _platform
from importlib import metadata
from pathlib import Path

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
_K_WATER_OH = 462750.4  # kJ/mol/nm^2 (OpenMM convention, E = k/2 (r-r0)^2), amber14 TIP3P
_K_XH = 282000.0  # generic X-H, approx. GAFF c3-hc
_K_OTHER = 250000.0


def _fill_constrained_bond_types(struct, system: openmm.System) -> None:
    import parmed

    dist = {}
    for i in range(system.getNumConstraints()):
        a, b, d = system.getConstraintParameters(i)
        dist[frozenset((a, b))] = d.value_in_unit(unit.nanometer)
    cache = {}
    for bond in struct.bonds:
        if bond.type is not None:
            continue
        a1, a2 = bond.atom1, bond.atom2
        r0 = dist.get(frozenset((a1.idx, a2.idx)))
        if r0 is None:
            raise ValueError(f"bond {a1.idx}-{a2.idx} has neither a bond force nor a constraint")
        elems = {a1.atomic_number, a2.atomic_number}
        if elems == {1, 8} and a1.residue.name in ("HOH", "WAT", "TIP3"):
            k = _K_WATER_OH
        elif 1 in elems:
            k = _K_XH
        else:
            k = _K_OTHER
        key = (round(k), round(r0, 6))
        if key not in cache:
            cache[key] = parmed.BondType(k / (2 * 4.184 * 100.0), r0 * 10.0)
            struct.bond_types.append(cache[key])
        bond.type = cache[key]


def write_amber(
    system: openmm.System, topology: Topology, positions: unit.Quantity, prmtop: Path, inpcrd: Path
) -> tuple[Path, Path]:
    """Amber prmtop/inpcrd via parmed.openmm.load_topology(...).save()."""
    import parmed
    from parmed.openmm import load_topology

    prmtop, inpcrd = Path(prmtop), Path(inpcrd)
    struct = load_topology(topology, system, xyz=positions)
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


def check_energy_consistency(
    system: openmm.System, prmtop: Path, inpcrd: Path, cfg: BuildConfig, rtol: float = 1e-3
) -> dict:
    """Compare potential energy of the XML system vs a system rebuilt from prmtop.

    Returns {"xml_kj_mol": float, "prmtop_kj_mol": float, "rel_diff": float, "ok": bool}.
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
    e1 = _energy(sys1, crd.positions, box)
    e2 = _energy(sys2, crd.positions, box)
    rel = abs(e1 - e2) / max(abs(e1), abs(e2), 1e-12)
    return {"xml_kj_mol": e1, "prmtop_kj_mol": e2, "rel_diff": rel, "ok": bool(rel <= rtol)}


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
