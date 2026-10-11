import numpy as np
import openmm
import pytest
from openff.toolkit import Molecule
from openmm import unit
from openmm.app import HBonds

from system_builder.config import BuildConfig, Component
from system_builder.molecules import PreparedComponent
from system_builder.pack import pack_system
from system_builder.parameterize import build_system, net_charge


def make_pc(name, smiles, count, resname, water=False, mono=False, scale=1.0, charges=None):
    mol = Molecule.from_smiles(smiles)
    mol.generate_conformers(n_conformers=1)
    seen = {}
    for atom in mol.atoms:
        seen[atom.symbol] = seen.get(atom.symbol, 0) + 1
        atom.name = f"{atom.symbol}{seen[atom.symbol]}"
        atom.metadata["residue_name"] = resname
    if charges is not None:
        mol.partial_charges = np.array(charges) * unit.elementary_charge
    comp = Component(name=name, count=count, smiles=smiles, charge_scale=scale)
    return PreparedComponent(comp, mol, count, resname, water, mono, scale)


def build(comps, box=2.0, **kw):
    cfg = BuildConfig(components=[c.component for c in comps], box_size_nm=box,
                      cutoff_nm=0.8, seed=1, **kw)
    top, pos, bv = pack_system(comps, cfg)
    return top, build_system(top, comps, cfg, None), cfg


def charges_by_residue(top, system, resname):
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    out = []
    for res in top.residues():
        if res.name == resname:
            out.append([nb.getParticleParameters(a.index)[0].value_in_unit(unit.elementary_charge)
                        for a in res.atoms()])
    return out


@pytest.fixture(scope="module")
def ethanol_water():
    comps = [make_pc("ethanol", "CCO", 2, "ETH"), make_pc("water", "O", 8, "HOH", water=True)]
    top, system, cfg = build(comps)
    return comps, top, system


def test_ethanol_water_builds(ethanol_water):
    comps, top, system = ethanol_water
    assert system.getNumParticles() == top.getNumAtoms()
    assert abs(net_charge(system)) < 1e-4


def test_water_is_tip3p(ethanol_water):
    comps, top, system = ethanol_water
    for q in charges_by_residue(top, system, "HOH"):
        assert q[0] == pytest.approx(-0.834)
        assert q[1] == pytest.approx(0.417)


def test_ethanol_neutral(ethanol_water):
    comps, top, system = ethanol_water
    for q in charges_by_residue(top, system, "ETH"):
        assert sum(q) == pytest.approx(0.0, abs=1e-4)


def test_energy_finite_and_constraints(ethanol_water):
    comps, top, system = ethanol_water
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))
    cfg_pos = None  # positions from fresh pack
    pc = comps
    cfg = BuildConfig(components=[c.component for c in pc], box_size_nm=2.0, seed=1)
    _, pos, _ = pack_system(pc, cfg)
    ctx.setPositions(pos)
    e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert np.isfinite(e)
    # HBonds: water 3 constraints each + ethanol 6 H each
    assert system.getNumConstraints() == 8 * 3 + 2 * 6


def test_nacl_in_water():
    comps = [make_pc("na", "[Na+]", 2, "NA", mono=True), make_pc("cl", "[Cl-]", 2, "CL", mono=True),
             make_pc("water", "O", 8, "HOH", water=True)]
    with pytest.warns(UserWarning, match="Joung"):
        top, system, _ = build(comps)
    assert [q[0] for q in charges_by_residue(top, system, "NA")] == [pytest.approx(1.0)] * 2
    assert [q[0] for q in charges_by_residue(top, system, "CL")] == [pytest.approx(-1.0)] * 2
    assert abs(net_charge(system)) < 1e-4


@pytest.fixture(scope="module")
def ionic_liquid():
    def comps(scale):
        return [make_pc("emim", "CCn1cc[n+](C)c1", 2, "EMI", scale=scale),
                make_pc("bf4", "F[B-](F)(F)F", 2, "BF4", scale=scale)]
    unscaled = build(comps(1.0))
    scaled = build(comps(0.8))
    return unscaled, scaled


def test_ionic_liquid_unscaled(ionic_liquid):
    (top, system, _), _ = ionic_liquid
    for q in charges_by_residue(top, system, "EMI"):
        assert sum(q) == pytest.approx(1.0, abs=1e-4)
    for q in charges_by_residue(top, system, "BF4"):
        assert sum(q) == pytest.approx(-1.0, abs=1e-4)
    assert net_charge(system) == pytest.approx(0.0, abs=1e-4)


def test_ionic_liquid_scaled(ionic_liquid):
    (top, system, _), (top2, system2, _) = ionic_liquid
    for q in charges_by_residue(top2, system2, "EMI"):
        assert sum(q) == pytest.approx(0.8, abs=1e-4)
    for q in charges_by_residue(top2, system2, "BF4"):
        assert sum(q) == pytest.approx(-0.8, abs=1e-4)
    assert net_charge(system2) == pytest.approx(0.0, abs=1e-4)


def test_scaled_exceptions():
    (top, s1, _), (_, s2, _) = [build([make_pc("emim", "CCn1cc[n+](C)c1", 1, "EMI", scale=sc),
                                       make_pc("bf4", "F[B-](F)(F)F", 1, "BF4", scale=sc)])
                                for sc in (1.0, 0.8)]
    nb1 = [f for f in s1.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    nb2 = [f for f in s2.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    assert nb1.getNumExceptions() == nb2.getNumExceptions()
    checked = 0
    for i in range(nb1.getNumExceptions()):
        a = nb1.getExceptionParameters(i)
        b = nb2.getExceptionParameters(i)
        assert b[3] == a[3] and b[4] == a[4]  # sigma/epsilon untouched
        if abs(a[2].value_in_unit(unit.elementary_charge**2)) > 1e-6:
            assert b[2]._value == pytest.approx(0.64 * a[2]._value, rel=1e-5)
            checked += 1
    assert checked > 0
    # LJ untouched
    for i in range(nb1.getNumParticles()):
        assert nb1.getParticleParameters(i)[1:] == nb2.getParticleParameters(i)[1:]


def test_user_partial_charges_used():
    # ethanol with user-supplied charges (sum 0)
    q = [-0.1, 0.1, -0.5] + [0.05] * 5 + [0.4]
    q[-1] = -sum(q[:-1])  # force neutrality
    comps = [make_pc("ethanol", "CCO", 1, "ETH", charges=q)]
    top, system, _ = build(comps, box=2.0)
    got = charges_by_residue(top, system, "ETH")[0]
    # order of atoms in mol == order in topology
    assert got == pytest.approx(q, abs=1e-4)
