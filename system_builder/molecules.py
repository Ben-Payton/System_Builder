"""Turn config components into OpenFF Molecules ready for packing and parameterization."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

import numpy as np

from openff.toolkit import Molecule
from openff.units import unit

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


_WATER = Molecule.from_smiles("O")


def is_water(molecule: Molecule) -> bool:
    """True if the molecule is H2O (by graph, not by name)."""
    if molecule.n_atoms != 3 or molecule.n_bonds != 2:
        return False
    return Molecule.are_isomorphic(molecule, _WATER, return_atom_map=False)[0]


def _assign_atom_names(molecule: Molecule) -> None:
    """Unique PDB-friendly names: element symbol + per-element index (C1, H12, ...)."""
    counts: Counter = Counter()
    for atom in molecule.atoms:
        counts[atom.symbol] += 1
        name = f"{atom.symbol}{counts[atom.symbol]}"
        if len(name) > 4:
            raise ValueError(f"molecule too large to give unique 4-character atom names ({name})")
        atom.name = name


def prepare_component(component: Component) -> Molecule:
    """Build one Molecule from SMILES or file, with one 3D conformer.

    Every atom's residue name is set later by prepare_molecules.
    """
    try:
        if component.smiles is not None:
            mol = Molecule.from_smiles(component.smiles, allow_undefined_stereo=True)
        elif component.file is not None:
            mol = Molecule.from_file(str(component.file), allow_undefined_stereo=True)
            if isinstance(mol, list):
                mol = mol[0]
        else:
            raise ValueError("neither smiles nor file given")
    except Exception as exc:
        raise ValueError(f"component '{component.name}': could not build molecule: {exc}") from exc

    if mol.n_conformers > 0:
        first = mol.conformers[0]
        mol._conformers = None
        mol.add_conformer(first)
    elif mol.n_atoms == 1:
        mol.add_conformer(np.zeros((1, 3)) * unit.angstrom)
    else:
        try:
            mol.generate_conformers(n_conformers=1)
        except Exception as exc:
            raise ValueError(
                f"component '{component.name}': conformer generation failed: {exc}"
            ) from exc
    _assign_atom_names(mol)
    return mol


def _resolve_resnames(cfg: BuildConfig, flags: list[tuple[bool, bool]], mols: list[Molecule]) -> list[str]:
    explicit: dict[str, str] = {}
    for c in cfg.components:
        if c.residue_name is not None:
            if c.residue_name in explicit:
                raise ValueError(
                    f"residue_name '{c.residue_name}' is used by both component "
                    f"'{explicit[c.residue_name]}' and '{c.name}'"
                )
            explicit[c.residue_name] = c.name
    used = set(explicit)
    names: list[str | None] = [c.residue_name for c in cfg.components]
    counter = 0
    for i, c in enumerate(cfg.components):
        if names[i] is not None:
            continue
        water, mono = flags[i]
        if water:
            cand = "HOH"
        elif mono:
            cand = mols[i].atoms[0].symbol.upper()
        else:
            cand = c.name[:3].upper()
        while cand in used:
            counter += 1
            cand = f"E{counter:02d}"
        used.add(cand)
        names[i] = cand
    return names  # type: ignore[return-value]


def prepare_molecules(cfg: BuildConfig) -> list[PreparedComponent]:
    """Prepare all components in config order.

    Assigns unique residue names, sets them on each Molecule's atoms
    (atom.metadata["residue_name"]) so they survive into the OpenMM topology,
    and flags water / monatomic species.
    """
    mols = [prepare_component(c) for c in cfg.components]
    flags = [(is_water(m), m.n_atoms == 1) for m in mols]
    resnames = _resolve_resnames(cfg, flags, mols)
    out = []
    for c, m, (water, mono), resname in zip(cfg.components, mols, flags, resnames):
        for atom in m.atoms:
            atom.metadata["residue_name"] = resname
        out.append(
            PreparedComponent(
                component=c, molecule=m, count=c.count, resname=resname,
                is_water=water, is_monatomic=mono, charge_scale=c.charge_scale,
            )
        )
    return out
