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

import yaml


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


_TOP_KEYS = {"components", "box", "forcefield", "nonbonded", "constraints", "tolerance_A", "seed"}
_COMP_KEYS = {"name", "smiles", "file", "count", "charge_scale", "residue_name"}
_BOX_KEYS = {"size_nm", "density_g_cm3"}
_FF_KEYS = {"small_molecule", "water"}
_NB_KEYS = {"method", "cutoff_nm"}
_NB_METHODS = ("NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic", "Ewald", "PME", "LJPME")
_CONSTRAINTS = ("HBonds", "AllBonds", "HAngles")
_WATER_MODELS = ("tip3p",)


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _mapping(value, where: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"'{where}' must be a mapping, got {type(value).__name__}")
    return value


def _check_keys(d: dict, allowed: set, where: str) -> None:
    unknown = sorted(str(k) for k in d if k not in allowed)
    if unknown:
        raise ValueError(
            f"unknown key(s) {unknown} in {where}; allowed: {sorted(allowed)}"
        )


def _positive(value, where: str) -> float:
    if not _is_num(value) or value <= 0:
        raise ValueError(f"{where} must be a positive number, got {value!r}")
    return float(value)


def _parse_component(entry, index: int, base_dir: Path | None) -> Component:
    if not isinstance(entry, dict):
        raise ValueError(f"components[{index}] must be a mapping, got {entry!r}")
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"components[{index}]: 'name' is required and must be a non-empty string")
    where = f"component '{name}'"
    _check_keys(entry, _COMP_KEYS, where)

    count = entry.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"{where}: 'count' must be an integer >= 1, got {count!r}")

    smiles, file = entry.get("smiles"), entry.get("file")
    if (smiles is None) == (file is None):
        raise ValueError(f"{where}: exactly one of 'smiles' or 'file' must be given")
    if smiles is not None and (not isinstance(smiles, str) or not smiles.strip()):
        raise ValueError(f"{where}: 'smiles' must be a non-empty string")
    path = None
    if file is not None:
        path = Path(str(file)).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        if not path.is_file():
            raise ValueError(f"{where}: file '{file}' does not exist (looked for {path})")

    charge_scale = entry.get("charge_scale", 1.0)
    if not _is_num(charge_scale) or charge_scale <= 0:
        raise ValueError(f"{where}: 'charge_scale' must be > 0, got {charge_scale!r}")

    resname = entry.get("residue_name")
    if resname is not None and (not isinstance(resname, str) or not 1 <= len(resname) <= 3):
        raise ValueError(f"{where}: 'residue_name' must be 1-3 characters, got {resname!r}")

    return Component(
        name=name, count=count, smiles=smiles, file=path,
        charge_scale=float(charge_scale), residue_name=resname,
    )


def load_config(path: str | Path) -> BuildConfig:
    """Parse a YAML file into a validated BuildConfig.

    Raises ValueError with a clear message on invalid input: no components,
    count < 1, neither/both of smiles and file, neither/both of box size and
    density, non-positive charge_scale, unknown keys, etc. Relative ``file``
    paths are resolved against the config file's directory.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"could not parse YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    cfg = config_from_dict(data, base_dir=path.parent)
    cfg.source_path = path
    return cfg


def config_from_dict(data: dict, base_dir: Path | None = None) -> BuildConfig:
    """Same validation as load_config, from an already-parsed mapping."""
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    _check_keys(data, _TOP_KEYS, "top level")

    entries = data.get("components")
    if not isinstance(entries, list) or not entries:
        raise ValueError("'components' must be a non-empty list")
    components = [_parse_component(e, i, base_dir) for i, e in enumerate(entries)]
    seen = set()
    for c in components:
        if c.name in seen:
            raise ValueError(f"duplicate component name '{c.name}'")
        seen.add(c.name)

    if "box" not in data or data["box"] is None:
        raise ValueError("'box' section is required (size_nm or density_g_cm3)")
    box = _mapping(data["box"], "box")
    _check_keys(box, _BOX_KEYS, "box")
    has_size, has_dens = box.get("size_nm") is not None, box.get("density_g_cm3") is not None
    if has_size == has_dens:
        raise ValueError("box: exactly one of 'size_nm' or 'density_g_cm3' must be given")
    size = _positive(box["size_nm"], "box.size_nm") if has_size else None
    dens = _positive(box["density_g_cm3"], "box.density_g_cm3") if has_dens else None

    ff = _mapping(data.get("forcefield"), "forcefield")
    _check_keys(ff, _FF_KEYS, "forcefield")
    small = ff.get("small_molecule", "gaff-2.11")
    if not isinstance(small, str) or not small:
        raise ValueError(f"forcefield.small_molecule must be a string, got {small!r}")
    water = ff.get("water", "tip3p")
    if water not in _WATER_MODELS:
        raise ValueError(
            f"forcefield.water {water!r} is not supported; supported: {list(_WATER_MODELS)}"
        )

    nb = _mapping(data.get("nonbonded"), "nonbonded")
    _check_keys(nb, _NB_KEYS, "nonbonded")
    method = nb.get("method", "PME")
    if method not in _NB_METHODS:
        raise ValueError(
            f"nonbonded.method {method!r} is not supported; choose from {list(_NB_METHODS)}"
        )
    cutoff = _positive(nb.get("cutoff_nm", 1.0), "nonbonded.cutoff_nm")

    constraints = data.get("constraints", "HBonds")
    if constraints is not None and constraints not in _CONSTRAINTS:
        raise ValueError(
            f"constraints {constraints!r} is not supported; choose null or one of "
            f"{list(_CONSTRAINTS)}"
        )

    tol = _positive(data.get("tolerance_A", 2.0), "tolerance_A")
    seed = data.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError(f"seed must be an integer or null, got {seed!r}")

    return BuildConfig(
        components=components, box_size_nm=size, density_g_cm3=dens,
        small_molecule_ff=small, water_model=water, nonbonded_method=method,
        cutoff_nm=cutoff, constraints=constraints, tolerance_A=tol, seed=seed,
        raw=data,
    )
