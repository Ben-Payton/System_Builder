"""Shared fixtures. Keep systems tiny so AM1-BCC and packmol run in seconds."""
import pytest
import yaml


TINY_CONFIG = {
    "components": [
        {"name": "ethanol", "smiles": "CCO", "count": 10},
        {"name": "water", "smiles": "O", "count": 50},
    ],
    "box": {"size_nm": 2.5},
    "forcefield": {"small_molecule": "gaff-2.11", "water": "tip3p"},
    "nonbonded": {"method": "PME", "cutoff_nm": 0.9},
    "constraints": "HBonds",
    "tolerance_A": 2.0,
    "seed": 1,
}


@pytest.fixture
def tiny_config_dict():
    import copy

    return copy.deepcopy(TINY_CONFIG)


@pytest.fixture
def tiny_config_file(tmp_path, tiny_config_dict):
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(tiny_config_dict))
    return path
