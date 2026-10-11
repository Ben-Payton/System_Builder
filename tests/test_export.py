import importlib.util
import json
from pathlib import Path

import numpy as np
import openmm
import pytest
from openmm import app, unit

from system_builder import export
from system_builder.config import BuildConfig, Component

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def water_system():
    ff = app.ForceField("amber14/tip3p.xml")
    mod = app.Modeller(app.Topology(), [] * unit.nanometer)
    mod.addSolvent(
        ff,
        boxSize=openmm.Vec3(1.6, 1.6, 1.6) * unit.nanometer,
        positiveIon="Na+",
        negativeIon="Cl-",
        ionicStrength=0.3 * unit.molar,
    )
    system = ff.createSystem(
        mod.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.8 * unit.nanometer,
        constraints=app.HBonds,
    )
    cfg = BuildConfig(
        components=[Component(name="water", count=1, smiles="O")],
        box_size_nm=1.6,
        cutoff_nm=0.8,
        constraints="HBonds",
        raw={"components": [{"name": "water", "smiles": "O", "count": 1}]},
    )
    return system, mod.topology, mod.positions, cfg


def test_write_xml(water_system, tmp_path):
    system, *_ = water_system
    p = export.write_xml(system, tmp_path / "s.xml")
    s2 = openmm.XmlSerializer.deserialize(p.read_text())
    assert s2.getNumParticles() == system.getNumParticles()


def test_write_pdb_roundtrip(water_system, tmp_path):
    system, top, pos, _ = water_system
    p = export.write_pdb(top, pos, tmp_path / "s.pdb")
    pdb = app.PDBFile(str(p))
    assert pdb.topology.getNumAtoms() == top.getNumAtoms()
    assert pdb.topology.getNumBonds() == top.getNumBonds()
    a = np.array([[v.x, v.y, v.z] for v in top.getPeriodicBoxVectors()])
    b = np.array([[v.x, v.y, v.z] for v in pdb.topology.getPeriodicBoxVectors()])
    assert np.allclose(a, b, atol=1e-3)


def test_write_pdb_conect_nonstandard(tmp_path):
    top = app.Topology()
    ch = top.addChain()
    res = top.addResidue("EOH", ch)
    C, O, H = (app.Element.getBySymbol(x) for x in "COH")
    a = [top.addAtom("C1", C, res), top.addAtom("O1", O, res), top.addAtom("HO", H, res)]
    top.addBond(a[0], a[1])
    top.addBond(a[1], a[2])
    top.setPeriodicBoxVectors([openmm.Vec3(2, 0, 0), openmm.Vec3(0, 2, 0), openmm.Vec3(0, 0, 2)])
    pos = [openmm.Vec3(0, 0, 0), openmm.Vec3(0.14, 0, 0), openmm.Vec3(0.2, 0.08, 0)] * unit.nanometer
    p = export.write_pdb(top, pos, tmp_path / "e.pdb")
    assert "CONECT" in p.read_text()
    pdb = app.PDBFile(str(p))
    assert pdb.topology.getNumBonds() == 2
    assert next(pdb.topology.residues()).name == "EOH"


def test_write_amber(water_system, tmp_path):
    system, top, pos, _ = water_system
    pr, ic = export.write_amber(system, top, pos, tmp_path / "s.prmtop", tmp_path / "s.inpcrd")
    prm = app.AmberPrmtopFile(str(pr))
    crd = app.AmberInpcrdFile(str(ic))
    assert prm.topology.getNumAtoms() == top.getNumAtoms()
    assert crd.boxVectors is not None


def test_energy_consistency(water_system, tmp_path):
    system, top, pos, cfg = water_system
    pr, ic = export.write_amber(system, top, pos, tmp_path / "s.prmtop", tmp_path / "s.inpcrd")
    res = export.check_energy_consistency(system, pr, ic, cfg)
    assert set(res) >= {"xml_kj_mol", "prmtop_kj_mol", "rel_diff", "ok"}
    assert res["ok"], res


def test_metadata(water_system, tmp_path):
    system, top, pos, cfg = water_system
    p = export.write_metadata(tmp_path / "m.json", cfg, system, top, extra={"foo": 1})
    m = json.loads(p.read_text())
    assert m["n_atoms"] == top.getNumAtoms()
    assert m["n_residues"] == top.getNumResidues()
    assert m["config"] == cfg.raw
    assert abs(m["net_charge"]) < 1e-6
    assert len(m["box_vectors_nm"]) == 3
    assert "openmm" in m["versions"] and "system_builder" in m["versions"]
    assert m["foo"] == 1 and m["warnings"] == []
    assert "timestamp" in m


def test_metadata_warns_non_neutral(tmp_path):
    ff = app.ForceField("amber14/tip3p.xml")
    mod = app.Modeller(app.Topology(), [] * unit.nanometer)
    mod.addSolvent(ff, boxSize=openmm.Vec3(1.5, 1.5, 1.5) * unit.nanometer, neutralize=False)
    mod.add(*_na(ff))
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME)
    cfg = BuildConfig(components=[], box_size_nm=1.5)
    m = json.loads(export.write_metadata(tmp_path / "m.json", cfg, system, mod.topology).read_text())
    assert any("neutral" in w for w in m["warnings"])


def _na(ff):
    t = app.Topology()
    c = t.addChain()
    r = t.addResidue("NA", c)
    t.addAtom("NA", app.Element.getBySymbol("Na"), r)
    return t, [openmm.Vec3(0.1, 0.1, 0.1)] * unit.nanometer


def test_export_all(water_system, tmp_path):
    system, top, pos, cfg = water_system
    paths = export.export_all(system, top, pos, cfg, tmp_path / "out")
    assert set(paths) == {"xml", "pdb", "prmtop", "inpcrd", "metadata"}
    assert all(p.exists() for p in paths.values())
    m = json.loads(paths["metadata"].read_text())
    assert m["energy_check"]["ok"]


def _load_run_simulation():
    spec = importlib.util.spec_from_file_location("run_simulation", ROOT / "examples" / "run_simulation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("fmt", ["xml", "amber"])
def test_run_simulation(water_system, tmp_path, fmt):
    system, top, pos, cfg = water_system
    out = tmp_path / "out"
    export.export_all(system, top, pos, cfg, out)
    rs = _load_run_simulation()
    sim = rs.run(out, fmt=fmt, steps=20, platform="CPU", cutoff_nm=0.7, report_interval=10, workdir=tmp_path, minimize_iterations=50)
    assert sim.currentStep == 20
    assert (tmp_path / "trajectory.dcd").exists()


def _prmtop_system(paths_dir, **kw):
    prm = app.AmberPrmtopFile(str(paths_dir / "s.prmtop"))
    return prm.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=0.8 * unit.nanometer, **kw)


def test_amber_water_is_rigid_like_xml(water_system, tmp_path):
    """Regression: rigid TIP3P (constraints only) must stay rigid in the prmtop."""
    system, top, pos, cfg = water_system
    export.write_amber(system, top, pos, tmp_path / "s.prmtop", tmp_path / "s.inpcrd")
    sys2 = _prmtop_system(tmp_path, constraints=app.HBonds, rigidWater=True)
    assert sys2.getNumConstraints() == system.getNumConstraints()
    n_water = sum(1 for r in top.residues() if r.name == "HOH")
    assert system.getNumConstraints() == 3 * n_water
    # flexible alternative: sane O-H / H-O-H parameters, no constraints
    flex = _prmtop_system(tmp_path, constraints=None, rigidWater=False)
    assert flex.getNumConstraints() == 0
    for f in flex.getForces():
        if isinstance(f, openmm.HarmonicAngleForce):
            _, _, _, theta, k = f.getAngleParameters(0)
            assert abs(theta.value_in_unit(unit.radian) - np.radians(104.52)) < 1e-3
            assert abs(k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2) - 100 * 4.184 * 2) < 1


def test_check_reports_constraints_and_perturbed(water_system, tmp_path):
    system, top, pos, cfg = water_system
    pr, ic = export.write_amber(system, top, pos, tmp_path / "s.prmtop", tmp_path / "s.inpcrd")
    res = export.check_energy_consistency(system, pr, ic, cfg)
    assert res["n_constraints_xml"] == res["n_constraints_prmtop"] > 0
    assert res["constraint_distance_maxdiff_nm"] < 1e-4
    assert res["perturbed_rel_diff"] < 1e-3
    assert res["ok"]


def test_check_detects_missing_rigid_water(water_system, tmp_path):
    """A prmtop whose water is not rigid must fail the check (energies at start match)."""
    import copy

    system, top, pos, cfg = water_system
    pr, ic = export.write_amber(system, top, pos, tmp_path / "s.prmtop", tmp_path / "s.inpcrd")
    sys_flex = copy.deepcopy(system)
    # XML system with one extra constraint the prmtop cannot have
    sys_flex.addConstraint(0, 3, 0.5)
    res = export.check_energy_consistency(sys_flex, pr, ic, cfg)
    assert not res["ok"]
    assert res["n_constraints_xml"] != res["n_constraints_prmtop"]


def test_write_amber_unbonded_constraint_errors(water_system, tmp_path):
    import copy

    system, top, pos, _ = water_system
    bad = copy.deepcopy(system)
    bad.addConstraint(0, 3, 0.5)  # atoms in different waters
    with pytest.raises(ValueError, match="constraint"):
        export.write_amber(bad, top, pos, tmp_path / "b.prmtop", tmp_path / "b.inpcrd")


def test_write_amber_residue_count_guard(tmp_path):
    ff = app.ForceField("amber14/tip3p.xml")
    mod = app.Modeller(app.Topology(), [] * unit.nanometer)
    mod.addSolvent(ff, boxSize=openmm.Vec3(1.2, 1.2, 1.2) * unit.nanometer)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.5 * unit.nanometer,
                             constraints=app.HBonds)
    for r in mod.topology.residues():
        r.id = "0"
    with pytest.raises(ValueError, match="residue"):
        export.write_amber(system, mod.topology, mod.positions, tmp_path / "a.prmtop", tmp_path / "a.inpcrd")
