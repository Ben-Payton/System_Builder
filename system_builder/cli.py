"""Command line entry point: ``system-builder build config.yaml -o out/``."""
from __future__ import annotations


def build(config_path: str, out_dir: str, overwrite: bool = False) -> dict:
    """Run the whole pipeline:

    load_config -> prepare_molecules -> pack_system -> build_system -> export_all.
    Refuses to write into a non-empty out_dir unless overwrite is True.
    Returns the dict of written paths from export_all.
    """
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """argparse CLI with a ``build`` subcommand. Returns a process exit code."""
    raise NotImplementedError
