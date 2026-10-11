import numpy as np
import pytest
from openff.toolkit import Molecule
from openmm import unit

from system_builder.config import BuildConfig, Component
from system_builder.molecules import PreparedComponent
from system_builder.pack import pack_system


def make_pc(name, smiles, count, resname, water=False, mono=False, scale=1.0):
    mol = Molecule.from_smiles(smiles, hydrogens_are_explicit=False)
    mol.generate_conformers(n_conformers=1)
    counts = {}
    for atom in mol.atoms:
        sym = atom.symbol
        counts[sym] = counts.get(sym, 0) + 1
        atom.name = f"{sym}{counts[sym]}"
        atom.metadata["residue_name"] = resname
    comp = Component(name=name, count=count, smiles=smiles, charge_scale=scale)
    return PreparedComponent(comp, mol, count, resname, water, mono, scale)


def tiny_components():
    return [make_pc("ethanol", "CCO", 5, "ETH"), make_pc("water", "O", 20, "HOH", water=True)]


def cfg_for(comps, **kw):
    return BuildConfig(components=[c.component for c in comps], **kw)


@pytest.fixture(scope="module")
def packed_box():
    comps = tiny_components()
    cfg = cfg_for(comps, box_size_nm=2.0, tolerance_A=2.0, seed=1)
    return comps, cfg, pack_system(comps, cfg)


def test_atom_count(packed_box):
    comps, cfg, (top, pos, box) = packed_box
    expected = sum(c.count * c.molecule.n_atoms for c in comps)
    assert top.getNumAtoms() == expected
    assert len(pos) == expected


def test_residue_names_and_counts(packed_box):
    comps, cfg, (top, pos, box) = packed_box
    names = [r.name for r in top.residues()]
    assert names.count("ETH") == 5
    assert names.count("HOH") == 20
    assert len(names) == 25


def test_atom_names_survive(packed_box):
    comps, cfg, (top, pos, box) = packed_box
    first = next(top.residues())
    assert [a.name for a in first.atoms()] == [a.name for a in comps[0].molecule.atoms]


def test_box_set(packed_box):
    comps, cfg, (top, pos, box) = packed_box
    bv = top.getPeriodicBoxVectors()
    assert bv is not None
    assert bv[0][0].value_in_unit(unit.nanometer) == pytest.approx(2.0)
    assert box[1][1].value_in_unit(unit.nanometer) == pytest.approx(2.0) or box[1][1] == pytest.approx(2.0)


def test_positions_in_box(packed_box):
    comps, cfg, (top, pos, box) = packed_box
    xyz = np.array(pos.value_in_unit(unit.nanometer))
    assert xyz.min() > -0.2 and xyz.max() < 2.2


def test_min_intermolecular_distance(packed_box):
    comps, cfg, (top, pos, box) = packed_box
    xyz = np.array(pos.value_in_unit(unit.nanometer))
    res_of = np.array([a.residue.index for a in top.atoms()])
    L = 2.0
    d = xyz[:, None, :] - xyz[None, :, :]
    d -= L * np.round(d / L)
    dist = np.linalg.norm(d, axis=-1)
    mask = res_of[:, None] != res_of[None, :]
    # tolerance 2 A; allow slack for packmol convergence
    assert dist[mask].min() > 0.15


def test_density_mode():
    comps = tiny_components()
    cfg = cfg_for(comps, density_g_cm3=0.9, tolerance_A=2.0, seed=1)
    top, pos, box = pack_system(comps, cfg)
    mass = 0.0
    for c in comps:
        mass += c.count * sum(a.mass.m_as("dalton") for a in c.molecule.atoms)
    vol = box[0][0] * box[1][1] * box[2][2]
    vol = vol.value_in_unit(unit.nanometer) if hasattr(vol, "value_in_unit") else vol
    density = mass * 1.66054 / vol  # dalton/nm^3 -> g/cm^3
    assert density == pytest.approx(0.9, rel=0.05)
