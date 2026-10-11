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
    raise NotImplementedError
