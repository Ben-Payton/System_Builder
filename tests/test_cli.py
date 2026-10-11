import pytest

from system_builder import cli


@pytest.fixture
def calls(monkeypatch):
    log = []

    def load_config(path):
        log.append(("load_config", str(path)))
        return "CFG"

    def prepare_molecules(cfg):
        log.append(("prepare", cfg))
        return ["COMP"]

    def pack_system(comps, cfg):
        log.append(("pack", comps, cfg))
        return "TOP", "POS", "BOX"

    def build_system(top, comps, cfg, cache_dir=None):
        log.append(("build", top, comps, cfg, cache_dir))
        return "SYS"

    def export_all(system, top, pos, cfg, out_dir):
        log.append(("export", system, top, pos, cfg, out_dir))
        return {"xml": out_dir / "system.xml"}

    for name, fn in [("load_config", load_config), ("prepare_molecules", prepare_molecules),
                     ("pack_system", pack_system), ("build_system", build_system),
                     ("export_all", export_all)]:
        monkeypatch.setattr(cli, name, fn)
    return log


def test_build_orchestration(calls, tmp_path):
    out = tmp_path / "out"
    paths = cli.build("c.yaml", str(out))
    assert [c[0] for c in calls] == ["load_config", "prepare", "pack", "build", "export"]
    assert calls[3][4] == out
    assert calls[4][5] == out
    assert paths == {"xml": out / "system.xml"}


def test_refuses_nonempty(calls, tmp_path):
    (tmp_path / "x.txt").write_text("hi")
    with pytest.raises(cli.OutputDirError):
        cli.build("c.yaml", str(tmp_path))
    assert calls == []
    cli.build("c.yaml", str(tmp_path), overwrite=True)
    assert calls


def test_empty_existing_dir_ok(calls, tmp_path):
    cli.build("c.yaml", str(tmp_path))
    assert calls


def test_main_ok(calls, tmp_path, capsys):
    rc = cli.main(["build", "c.yaml", "-o", str(tmp_path / "o")])
    assert rc == 0
    assert "system.xml" in capsys.readouterr().out


def test_main_nonempty_exit_code(calls, tmp_path, capsys):
    (tmp_path / "x").write_text("1")
    rc = cli.main(["build", "c.yaml", "-o", str(tmp_path)])
    assert rc == 1
    assert "--overwrite" in capsys.readouterr().err
    assert cli.main(["build", "c.yaml", "-o", str(tmp_path), "--overwrite"]) == 0


def test_main_value_error(monkeypatch, tmp_path, capsys):
    def bad(path):
        raise ValueError("no components")

    monkeypatch.setattr(cli, "load_config", bad)
    rc = cli.main(["build", "c.yaml", "-o", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "no components" in err and "Traceback" not in err


def test_main_value_error_verbose_traceback(monkeypatch, tmp_path, capsys):
    def bad(path):
        raise ValueError("no components")

    monkeypatch.setattr(cli, "load_config", bad)
    rc = cli.main(["build", "c.yaml", "-o", str(tmp_path / "o"), "-v"])
    assert rc == 2
    assert "Traceback" in capsys.readouterr().err


def test_main_unexpected_error(calls, monkeypatch, tmp_path, capsys):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "pack_system", boom)
    assert cli.main(["build", "c.yaml", "-o", str(tmp_path / "o")]) == 1
    assert "kaboom" in capsys.readouterr().err


def test_main_requires_subcommand(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert e.value.code == 2
