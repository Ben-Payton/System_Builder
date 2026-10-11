"""Write portable, architecture-independent outputs for pip-only OpenMM scripts."""
from __future__ import annotations

from pathlib import Path

import openmm
from openmm import unit
from openmm.app import Topology

from system_builder.config import BuildConfig


def write_xml(system: openmm.System, path: Path) -> Path:
    """XmlSerializer.serialize(system) -> path."""
    raise NotImplementedError


def write_pdb(topology: Topology, positions: unit.Quantity, path: Path) -> Path:
    """PDB with CRYST1 box and CONECT records."""
    raise NotImplementedError


def write_amber(
    system: openmm.System, topology: Topology, positions: unit.Quantity, prmtop: Path, inpcrd: Path
) -> tuple[Path, Path]:
    """Amber prmtop/inpcrd via parmed.openmm.load_topology(...).save()."""
    raise NotImplementedError


def check_energy_consistency(
    system: openmm.System, prmtop: Path, inpcrd: Path, cfg: BuildConfig, rtol: float = 1e-3
) -> dict:
    """Compare potential energy of the XML system vs a system rebuilt from prmtop.

    Returns {"xml_kj_mol": float, "prmtop_kj_mol": float, "rel_diff": float, "ok": bool}.
    """
    raise NotImplementedError


def write_metadata(path: Path, cfg: BuildConfig, system: openmm.System, topology: Topology, extra: dict | None = None) -> Path:
    """metadata.json: config echo, package versions, atom count, box, net charge, warnings."""
    raise NotImplementedError


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
    raise NotImplementedError
