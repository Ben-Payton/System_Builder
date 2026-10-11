"""Turn config components into OpenFF Molecules ready for packing and parameterization."""
from __future__ import annotations

from dataclasses import dataclass

from openff.toolkit import Molecule

from system_builder.config import BuildConfig, Component


@dataclass
class PreparedComponent:
    component: Component
    molecule: Molecule  # has >=1 conformer; partial_charges set only if user-supplied
    count: int
    resname: str  # unique across components, <=3 chars, used as residue name in topology
    is_water: bool  # parameterized with the water model (TIP3P), not GAFF
    is_monatomic: bool  # single-atom ion; parameterized with JC ion params, not GAFF
    charge_scale: float

    @property
    def net_charge(self) -> float:
        """Formal net charge of one molecule in elementary charges (unscaled)."""
        return float(self.molecule.total_charge.m_as("elementary_charge"))


def is_water(molecule: Molecule) -> bool:
    """True if the molecule is H2O (by graph, not by name)."""
    raise NotImplementedError


def prepare_component(component: Component) -> Molecule:
    """Build one Molecule from SMILES or file, with one 3D conformer.

    Every atom's residue name is set later by prepare_molecules.
    """
    raise NotImplementedError


def prepare_molecules(cfg: BuildConfig) -> list[PreparedComponent]:
    """Prepare all components in config order.

    Assigns unique residue names, sets them on each Molecule's atoms
    (atom.metadata["residue_name"]) so they survive into the OpenMM topology,
    and flags water / monatomic species.
    """
    raise NotImplementedError
