"""Command line entry point: ``system-builder build config.yaml -o out/``."""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

from system_builder.config import load_config
from system_builder.export import export_all
from system_builder.molecules import prepare_molecules
from system_builder.pack import pack_system
from system_builder.parameterize import build_system

log = logging.getLogger("system_builder")


class OutputDirError(RuntimeError):
    """The output directory exists, is non-empty, and --overwrite was not given."""


def build(config_path: str, out_dir: str, overwrite: bool = False) -> dict:
    """Run the whole pipeline:

    load_config -> prepare_molecules -> pack_system -> build_system -> export_all.
    Refuses to write into a non-empty out_dir unless overwrite is True.
    Returns the dict of written paths from export_all.
    """
    out = Path(out_dir)
    if out.exists() and not out.is_dir():
        raise OutputDirError(f"{out} exists and is not a directory")
    if out.is_dir() and any(out.iterdir()) and not overwrite:
        raise OutputDirError(f"output directory {out} is not empty; use --overwrite to write into it")

    cfg = load_config(config_path)
    log.info("Preparing molecules")
    components = prepare_molecules(cfg)
    log.info("Packing system")
    topology, positions, _box = pack_system(components, cfg)
    out.mkdir(parents=True, exist_ok=True)
    log.info("Parameterizing system")
    system = build_system(topology, components, cfg, cache_dir=out)
    log.info("Exporting to %s", out)
    return export_all(system, topology, positions, cfg, out)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="system-builder", description="Pack and parameterize OpenMM systems.")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build a system from a YAML config")
    b.add_argument("config", help="YAML configuration file")
    b.add_argument("-o", "--out", required=True, help="output directory")
    b.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty output directory")
    b.add_argument("-v", "--verbose", action="store_true", help="debug logging and full tracebacks")
    return p


def main(argv: list[str] | None = None) -> int:
    """argparse CLI with a ``build`` subcommand. Returns a process exit code."""
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s"
    )
    try:
        paths = build(args.config, args.out, overwrite=args.overwrite)
    except OutputDirError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: invalid input: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"error: build failed: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1
    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
