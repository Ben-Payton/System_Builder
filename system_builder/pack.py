"""Pack components into a periodic box with packmol (via openff-interchange)."""
from __future__ import annotations

from openmm import Vec3, unit
from openmm.app import Topology

from system_builder.config import BuildConfig
from system_builder.molecules import PreparedComponent


def pack_system(
    components: list[PreparedComponent], cfg: BuildConfig
) -> tuple[Topology, unit.Quantity, tuple[Vec3, Vec3, Vec3]]:
    """Pack ``count`` copies of each component.

    Uses openff.interchange.components._packmol.pack_box with either a cubic
    box of cfg.box_size_nm or cfg.density_g_cm3. Residue names from
    PreparedComponent.resname must appear in the returned OpenMM topology,
    and the topology's periodic box vectors must be set.

    Returns (openmm Topology, positions in nm as Quantity[list[Vec3]], box vectors in nm).
    """
    import numpy as np
    from openff.interchange.components._packmol import pack_box
    from openff.units import unit as ounit

    kwargs = {}
    if cfg.box_size_nm is not None:
        kwargs["box_vectors"] = np.eye(3) * cfg.box_size_nm * ounit.nanometer
    elif cfg.density_g_cm3 is not None:
        kwargs["target_density"] = cfg.density_g_cm3 * ounit.gram / ounit.milliliter
        kwargs["box_shape"] = np.eye(3)  # cubic box
    else:
        raise ValueError("Either box_size_nm or density_g_cm3 must be set")

    # NOTE: pack_box has no seed argument; packmol output is not reproducible via cfg.seed.
    off_top = pack_box(
        molecules=[pc.molecule for pc in components],
        number_of_copies=[pc.count for pc in components],
        tolerance=cfg.tolerance_A * ounit.angstrom,
        **kwargs,
    )
    top = off_top.to_openmm()

    # Ensure residue names (one residue per molecule, in component order).
    expected = [pc.resname for pc in components for _ in range(pc.count)]
    residues = list(top.residues())
    if len(residues) == len(expected):
        for res, name in zip(residues, expected):
            res.name = name
    for i, res in enumerate(residues):
        res.id = str(i + 1)

    box_nm = np.asarray(off_top.box_vectors.m_as("nanometer"))
    box = tuple(Vec3(*map(float, row)) for row in box_nm)
    top.setPeriodicBoxVectors(unit.Quantity(list(box), unit.nanometer))
    pos_nm = np.asarray(off_top.get_positions().m_as("nanometer"))
    positions = unit.Quantity([Vec3(*map(float, r)) for r in pos_nm], unit.nanometer)
    return top, positions, box
