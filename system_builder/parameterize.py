"""Build an OpenMM System: GAFF2/AM1-BCC small molecules and ions, Amber TIP3P water."""
from __future__ import annotations

from pathlib import Path

import openmm
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
    raise NotImplementedError


def net_charge(system: openmm.System) -> float:
    """Sum of NonbondedForce particle charges in elementary charges."""
    raise NotImplementedError
