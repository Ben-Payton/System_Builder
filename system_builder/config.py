"""Load and validate YAML build configurations.

Example config::

    components:
      - name: ethanol
        smiles: CCO
        count: 10
      - name: water
        smiles: O
        count: 50
    box:
      density_g_cm3: 0.9      # or size_nm: 3.0 (cubic edge length)
    forcefield:
      small_molecule: gaff-2.11
      water: tip3p
    nonbonded:
      method: PME
      cutoff_nm: 1.0
    constraints: HBonds
    tolerance_A: 2.0
    seed: 42
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Component:
    """One molecular species in the box."""

    name: str
    count: int
    smiles: Optional[str] = None
    file: Optional[Path] = None  # SDF/mol2 with user-supplied partial charges
    charge_scale: float = 1.0  # multiply partial charges (e.g. 0.8 for ionic liquids)
    residue_name: Optional[str] = None  # <=3 chars; derived from name if omitted


@dataclass
class BuildConfig:
    components: list[Component]
    box_size_nm: Optional[float] = None  # cubic edge; exactly one of size/density is set
    density_g_cm3: Optional[float] = None
    small_molecule_ff: str = "gaff-2.11"
    water_model: str = "tip3p"
    nonbonded_method: str = "PME"
    cutoff_nm: float = 1.0
    constraints: Optional[str] = "HBonds"  # None, HBonds, AllBonds, HAngles
    tolerance_A: float = 2.0
    seed: Optional[int] = None
    source_path: Optional[Path] = None  # config file this was loaded from
    raw: dict = field(default_factory=dict)  # the parsed YAML, echoed into metadata


def load_config(path: str | Path) -> BuildConfig:
    """Parse a YAML file into a validated BuildConfig.

    Raises ValueError with a clear message on invalid input: no components,
    count < 1, neither/both of smiles and file, neither/both of box size and
    density, non-positive charge_scale, unknown keys, etc. Relative ``file``
    paths are resolved against the config file's directory.
    """
    raise NotImplementedError


def config_from_dict(data: dict, base_dir: Path | None = None) -> BuildConfig:
    """Same validation as load_config, from an already-parsed mapping."""
    raise NotImplementedError
