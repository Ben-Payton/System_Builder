import numpy as np
import pytest
from openff.toolkit import Molecule
from openff.units import unit

from system_builder.config import BuildConfig, Component, config_from_dict
from system_builder.molecules import (
    is_water,
    prepare_component,
    prepare_molecules,
)


def comp(name="x", smiles="CCO", count=1, **kw):
    return Component(name=name, count=count, smiles=smiles, **kw)


def cfg_of(*components):
    return BuildConfig(components=list(components), box_size_nm=3.0)


def test_is_water():
    assert is_water(Molecule.from_smiles("O"))
    assert is_water(Molecule.from_smiles("[H][O][H]"))
    assert not is_water(Molecule.from_smiles("CO"))
    assert not is_water(Molecule.from_smiles("[OH-]"))
    assert not is_water(Molecule.from_smiles("OO"))
    assert not is_water(Molecule.from_smiles("[Na+]"))


def test_prepare_component_smiles_one_conformer():
    mol = prepare_component(comp(smiles="CCO"))
    assert mol.n_atoms == 9
    assert mol.n_conformers == 1
    assert mol.partial_charges is None


def test_unique_pdb_friendly_atom_names():
    mol = prepare_component(comp(smiles="CCO"))
    names = [a.name for a in mol.atoms]
    assert len(set(names)) == len(names)
    assert all(1 <= len(n) <= 4 and n.isalnum() for n in names)
    assert {"C1", "C2", "O1", "H1"} <= set(names)


def test_atom_names_big_molecule_stay_unique_and_short():
    mol = prepare_component(comp(smiles="C" * 12))
    names = [a.name for a in mol.atoms]
    assert len(set(names)) == len(names) == 38
    assert max(len(n) for n in names) <= 4


@pytest.mark.parametrize(
    "smiles,charge,n",
    [("[Na+]", 1, 1), ("[Cl-]", -1, 1), ("CCn1cc[n+](C)c1", 1, 19), ("F[B-](F)(F)F", -1, 5)],
)
def test_formal_charge_and_ions(smiles, charge, n):
    mol = prepare_component(comp(smiles=smiles))
    assert mol.n_atoms == n
    assert mol.total_charge.m_as("elementary_charge") == charge
    assert mol.n_conformers == 1


def test_monatomic_conformer_at_origin():
    mol = prepare_component(comp(smiles="[Na+]"))
    assert np.allclose(mol.conformers[0].m_as("angstrom"), 0.0)


def test_undefined_stereo_allowed():
    assert prepare_component(comp(smiles="CC(F)Cl")).n_conformers == 1


def test_file_with_user_charges(tmp_path):
    ref = Molecule.from_smiles("CCO")
    ref.generate_conformers(n_conformers=1)
    charges = np.linspace(-0.2, 0.2, ref.n_atoms)
    charges -= charges.mean()
    ref.partial_charges = charges * unit.elementary_charge
    path = tmp_path / "eth.sdf"
    ref.to_file(str(path), file_format="sdf")
    mol = prepare_component(Component(name="eth", count=1, file=path))
    assert mol.n_conformers == 1
    assert mol.partial_charges is not None
    assert np.allclose(mol.partial_charges.m_as("elementary_charge"), charges, atol=1e-3)
    names = [a.name for a in mol.atoms]
    assert len(set(names)) == len(names)


def test_file_without_charges(tmp_path):
    ref = Molecule.from_smiles("CCO")
    ref.generate_conformers(n_conformers=1)
    path = tmp_path / "eth.sdf"
    ref.to_file(str(path), file_format="sdf")
    mol = prepare_component(Component(name="eth", count=1, file=path))
    assert mol.partial_charges is None
    assert mol.n_conformers == 1


def test_prepare_molecules_tiny(tiny_config_dict):
    prepared = prepare_molecules(config_from_dict(tiny_config_dict))
    assert [p.component.name for p in prepared] == ["ethanol", "water"]
    eth, wat = prepared
    assert (eth.count, wat.count) == (10, 50)
    assert wat.is_water and not eth.is_water
    assert wat.resname == "HOH" and eth.resname == "ETH"
    assert not eth.is_monatomic and not wat.is_monatomic
    assert eth.net_charge == 0.0
    for p in prepared:
        assert all(a.metadata["residue_name"] == p.resname for a in p.molecule.atoms)


def test_ion_resnames_and_flags():
    prepared = prepare_molecules(
        cfg_of(comp("sodium", "[Na+]"), comp("chloride", "[Cl-]", charge_scale=0.8))
    )
    assert [p.resname for p in prepared] == ["NA", "CL"]
    assert all(p.is_monatomic and not p.is_water for p in prepared)
    assert prepared[0].net_charge == 1.0 and prepared[1].charge_scale == 0.8


def test_explicit_residue_name_wins():
    (p,) = prepare_molecules(cfg_of(comp("ethanol", residue_name="EOH")))
    assert p.resname == "EOH"
    assert p.molecule.atoms[0].metadata["residue_name"] == "EOH"


def test_derived_collisions_are_deduped_deterministically():
    def run():
        return [
            p.resname
            for p in prepare_molecules(
                cfg_of(comp("ethanol"), comp("ethane", "CC"), comp("ethyl", "CCC"))
            )
        ]

    first = run()
    assert first == run()
    assert first[0] == "ETH"
    assert len(set(first)) == 3 and all(len(r) <= 3 for r in first)


def test_derived_does_not_steal_explicit_name():
    res = [p.resname for p in prepare_molecules(cfg_of(comp("ethanol"), comp("ethane", "CC", residue_name="ETH")))]
    assert res[1] == "ETH" and res[0] != "ETH"


def test_explicit_collision_raises():
    with pytest.raises(ValueError, match="ABC"):
        prepare_molecules(
            cfg_of(comp("a", residue_name="ABC"), comp("b", "CC", residue_name="ABC"))
        )


def test_two_water_components_get_distinct_resnames():
    prepared = prepare_molecules(cfg_of(comp("water", "O"), comp("water2", "O")))
    assert prepared[0].resname == "HOH"
    assert prepared[1].resname != "HOH"
    assert all(p.is_water for p in prepared)


def test_invalid_smiles_raises_value_error():
    with pytest.raises(ValueError, match="bad"):
        prepare_component(comp("bad", smiles="C(C"))
