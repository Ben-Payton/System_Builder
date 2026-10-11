"""Build an OpenMM System: GAFF2/AM1-BCC small molecules and ions, Amber TIP3P water."""
from __future__ import annotations

import warnings
from pathlib import Path

import openmm
from openmm import unit
from openmm import app
from openmm.app import Topology

from system_builder.config import BuildConfig
from system_builder.molecules import PreparedComponent


def build_system(
    topology: Topology,
    components: list[PreparedComponent],
    cfg: BuildConfig,
    cache_dir: Path | None = None,
) -> openmm.System:
    """Parameterize the packed topology.

    - water components -> amber14/tip3p.xml
    - monatomic ions -> Joung-Cheatham TIP3P ion params from amber14/tip3p.xml (logs a warning)
    - everything else -> openmmforcefields GAFFTemplateGenerator(cfg.small_molecule_ff),
      AM1-BCC charges unless the molecule carries user-supplied partial charges.
      Template cache written to cache_dir/gaff_cache.json if cache_dir is given.
    - nonbonded method/cutoff/constraints from cfg
    - components with charge_scale != 1 have their NonbondedForce particle
      charges and exception chargeProds scaled after creation.
    """
    from openmmforcefields.generators import GAFFTemplateGenerator

    forcefield = app.ForceField("amber14/tip3p.xml")

    if any(pc.is_monatomic for pc in components):
        warnings.warn(
            "Monatomic ions are parameterized with Joung-Cheatham (TIP3P) parameters "
            "from amber14/tip3p.xml, not GAFF.",
            UserWarning,
            stacklevel=2,
        )

    gaff_mols = [pc.molecule for pc in components if not pc.is_water and not pc.is_monatomic]
    if gaff_mols:
        cache = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache = str(cache_dir / "gaff_cache.json")
        gen = GAFFTemplateGenerator(
            molecules=gaff_mols, forcefield=cfg.small_molecule_ff, cache=cache
        )
        forcefield.registerTemplateGenerator(gen.generator)

    methods = {
        "PME": app.PME,
        "LJPME": app.LJPME,
        "CUTOFFPERIODIC": app.CutoffPeriodic,
        "CUTOFFNONPERIODIC": app.CutoffNonPeriodic,
        "NOCUTOFF": app.NoCutoff,
    }
    constraints = {None: None, "none": None, "hbonds": app.HBonds, "allbonds": app.AllBonds,
                   "hangles": app.HAngles}
    try:
        method = methods[str(cfg.nonbonded_method).replace("-", "").replace("_", "").upper()]
    except KeyError:
        raise ValueError(f"Unknown nonbonded method {cfg.nonbonded_method!r}") from None
    ckey = cfg.constraints.lower() if isinstance(cfg.constraints, str) else None
    if ckey not in constraints:
        raise ValueError(f"Unknown constraints {cfg.constraints!r}")

    system = forcefield.createSystem(
        topology,
        nonbondedMethod=method,
        nonbondedCutoff=cfg.cutoff_nm * unit.nanometer,
        constraints=constraints[ckey],
        rigidWater=True,
    )

    _scale_charges(system, topology, components)
    return system


def _scale_charges(system, topology, components) -> None:
    """Scale NonbondedForce charges (and exception chargeProds) of scaled components."""
    scale_by_res = {pc.resname: pc.charge_scale for pc in components if pc.charge_scale != 1.0}
    if not scale_by_res:
        return
    scales = [1.0] * system.getNumParticles()
    for res in topology.residues():
        if res.name in scale_by_res:
            for atom in res.atoms():
                scales[atom.index] = scale_by_res[res.name]
    for force in system.getForces():
        if not isinstance(force, openmm.NonbondedForce):
            continue
        for i, sc in enumerate(scales):
            if sc != 1.0:
                q, sigma, eps = force.getParticleParameters(i)
                force.setParticleParameters(i, q * sc, sigma, eps)
        for k in range(force.getNumExceptions()):
            i, j, qq, sigma, eps = force.getExceptionParameters(k)
            f = scales[i] * scales[j]
            if f != 1.0:
                force.setExceptionParameters(k, i, j, qq * f, sigma, eps)


def net_charge(system: openmm.System) -> float:
    """Sum of NonbondedForce particle charges in elementary charges."""
    total = 0.0
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            for i in range(force.getNumParticles()):
                total += force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
    return total
