import copy

import pytest
import yaml

from system_builder.config import config_from_dict, load_config


def test_load_tiny(tiny_config_file, tiny_config_dict):
    cfg = load_config(tiny_config_file)
    assert [c.name for c in cfg.components] == ["ethanol", "water"]
    assert cfg.components[0].count == 10
    assert cfg.components[0].smiles == "CCO"
    assert cfg.box_size_nm == 2.5 and cfg.density_g_cm3 is None
    assert cfg.cutoff_nm == 0.9 and cfg.constraints == "HBonds"
    assert cfg.seed == 1 and cfg.tolerance_A == 2.0
    assert cfg.source_path == tiny_config_file
    assert cfg.raw == tiny_config_dict


def test_defaults():
    cfg = config_from_dict({"components": [{"name": "w", "smiles": "O", "count": 3}],
                            "box": {"density_g_cm3": 0.9}})
    assert cfg.small_molecule_ff == "gaff-2.11"
    assert cfg.water_model == "tip3p"
    assert cfg.nonbonded_method == "PME"
    assert cfg.constraints == "HBonds"
    assert cfg.seed is None and cfg.density_g_cm3 == 0.9


def test_constraints_null_and_none_string(tiny_config_dict):
    tiny_config_dict["constraints"] = None
    assert config_from_dict(tiny_config_dict).constraints is None
    tiny_config_dict["constraints"] = "AllBonds"
    assert config_from_dict(tiny_config_dict).constraints == "AllBonds"


def test_relative_file_resolved(tmp_path, tiny_config_dict):
    (tmp_path / "mol.sdf").write_text("x")
    tiny_config_dict["components"][0].pop("smiles")
    tiny_config_dict["components"][0]["file"] = "mol.sdf"
    cfg = config_from_dict(tiny_config_dict, base_dir=tmp_path)
    assert cfg.components[0].file == tmp_path / "mol.sdf"


def test_load_config_resolves_file_against_config_dir(tmp_path, tiny_config_dict):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "mol.sdf").write_text("x")
    tiny_config_dict["components"][0].pop("smiles")
    tiny_config_dict["components"][0]["file"] = "mol.sdf"
    p = sub / "c.yaml"
    p.write_text(yaml.safe_dump(tiny_config_dict))
    assert load_config(p).components[0].file == sub / "mol.sdf"


def _bad(d, mutate, match):
    d = copy.deepcopy(d)
    mutate(d)
    with pytest.raises(ValueError, match=match):
        config_from_dict(d)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda d: d.update(components=[]), "components"),
        (lambda d: d.pop("components"), "components"),
        (lambda d: d["components"][0].update(count=0), "ethanol.*count"),
        (lambda d: d["components"][0].update(count=2.5), "ethanol.*count"),
        (lambda d: d["components"][0].update(count="3"), "ethanol.*count"),
        (lambda d: d["components"][0].update(count=True), "ethanol.*count"),
        (lambda d: d["components"][0].pop("count"), "ethanol.*count"),
        (lambda d: d["components"][0].pop("smiles"), "ethanol.*smiles.*file"),
        (lambda d: d["components"][0].update(file="x.sdf"), "ethanol.*smiles.*file"),
        (lambda d: (d["components"][0].pop("smiles"), d["components"][0].update(file="nope.sdf")),
         "ethanol.*nope.sdf"),
        (lambda d: d["components"][0].update(charge_scale=0), "ethanol.*charge_scale"),
        (lambda d: d["components"][0].update(charge_scale=-1), "ethanol.*charge_scale"),
        (lambda d: d["components"][0].update(residue_name="ABCD"), "ethanol.*residue_name"),
        (lambda d: d["components"][0].update(residue_name=""), "ethanol.*residue_name"),
        (lambda d: d["components"][1].update(name="ethanol"), "duplicate.*ethanol"),
        (lambda d: d["components"][0].pop("name"), "name"),
        (lambda d: d.update(bogus=1), "bogus"),
        (lambda d: d["components"][0].update(smile="CC"), "smile"),
        (lambda d: d["box"].update(density_g_cm3=0.9), "size_nm.*density"),
        (lambda d: d.pop("box"), "box"),
        (lambda d: d.update(box={}), "size_nm.*density"),
        (lambda d: d.update(box={"size_nm": -1}), "size_nm"),
        (lambda d: d.update(box={"density_g_cm3": 0}), "density_g_cm3"),
        (lambda d: d.update(box={"size_nm": 2, "extra": 1}), "extra"),
        (lambda d: d["forcefield"].update(water="spce"), "water"),
        (lambda d: d["forcefield"].update(oops=1), "oops"),
        (lambda d: d["nonbonded"].update(method="Bogus"), "method"),
        (lambda d: d["nonbonded"].update(cutoff_nm=0), "cutoff_nm"),
        (lambda d: d["nonbonded"].update(cutoff_nm=-1.0), "cutoff_nm"),
        (lambda d: d.update(constraints="Everything"), "constraints"),
        (lambda d: d.update(tolerance_A=0), "tolerance_A"),
        (lambda d: d.update(seed="a"), "seed"),
    ],
)
def test_invalid(tiny_config_dict, mutate, match):
    _bad(tiny_config_dict, mutate, match)


def test_nocutoff_method_ok(tiny_config_dict):
    tiny_config_dict["nonbonded"] = {"method": "NoCutoff"}
    assert config_from_dict(tiny_config_dict).nonbonded_method == "NoCutoff"


def test_load_config_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found|exist"):
        load_config(tmp_path / "nope.yaml")


def test_load_config_non_mapping(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(p)
