#!/usr/bin/env python
"""Run an NPT simulation from System_Builder output. Needs only `pip install openmm`.

    python run_simulation.py OUT_DIR [--format xml|amber] [--steps N] [--platform CUDA]

OUT_DIR is the directory written by `system-builder build` (system.xml/.pdb or
system.prmtop/.inpcrd). Writes trajectory.dcd and prints a StateDataReporter log.
"""
import argparse
import sys
from pathlib import Path

import openmm
from openmm import app, unit


def load_xml(out_dir):
    """XML route: cutoff, constraints and PME settings are baked into system.xml."""
    pdb = app.PDBFile(str(out_dir / "system.pdb"))
    system = openmm.XmlSerializer.deserialize((out_dir / "system.xml").read_text())
    return pdb.topology, pdb.positions, system


def load_amber(out_dir, cutoff_nm=1.0):
    """Amber route: nonbonded method, cutoff and constraints are chosen here."""
    prmtop = app.AmberPrmtopFile(str(out_dir / "system.prmtop"))
    inpcrd = app.AmberInpcrdFile(str(out_dir / "system.inpcrd"))
    system = prmtop.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        constraints=app.HBonds,
    )
    if inpcrd.boxVectors is not None:
        system.setDefaultPeriodicBoxVectors(*inpcrd.boxVectors)
    return prmtop.topology, inpcrd.positions, system


def run(
    out_dir,
    fmt="xml",
    steps=10000,
    platform=None,
    temperature_K=300.0,
    pressure_bar=1.0,
    cutoff_nm=1.0,
    report_interval=1000,
    workdir=".",
    seed=None,
    minimize_iterations=0,
):
    """Minimize, then run `steps` of NPT MD. Returns the Simulation."""
    out_dir, workdir = Path(out_dir), Path(workdir)
    if fmt == "xml":
        topology, positions, system = load_xml(out_dir)
    elif fmt == "amber":
        topology, positions, system = load_amber(out_dir, cutoff_nm)
    else:
        raise ValueError(f"unknown format {fmt!r}")

    # Serialized systems carry no barostat; add one for NPT.
    barostat = openmm.MonteCarloBarostat(pressure_bar * unit.bar, temperature_K * unit.kelvin)
    if seed is not None:
        barostat.setRandomNumberSeed(seed)
    system.addForce(barostat)

    integrator = openmm.LangevinMiddleIntegrator(
        temperature_K * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds
    )
    if seed is not None:
        integrator.setRandomNumberSeed(seed)
    plat = openmm.Platform.getPlatformByName(platform) if platform else None
    sim = app.Simulation(topology, system, integrator, *([plat] if plat else []))
    sim.context.setPositions(positions)
    box = topology.getPeriodicBoxVectors()
    if box is not None:
        sim.context.setPeriodicBoxVectors(*box)

    sim.minimizeEnergy(maxIterations=minimize_iterations)
    if seed is None:
        sim.context.setVelocitiesToTemperature(temperature_K * unit.kelvin)
    else:
        sim.context.setVelocitiesToTemperature(temperature_K * unit.kelvin, seed)
    sim.reporters.append(app.DCDReporter(str(workdir / "trajectory.dcd"), report_interval))
    sim.reporters.append(
        app.StateDataReporter(
            sys.stdout, report_interval, step=True, time=True, potentialEnergy=True,
            temperature=True, density=True, speed=True, totalSteps=steps,
        )
    )
    sim.step(steps)
    return sim


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out_dir")
    p.add_argument("--format", choices=["xml", "amber"], default="xml")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--platform", default=None, help="CUDA, OpenCL, CPU, Reference (default: fastest)")
    p.add_argument("--temperature", type=float, default=300.0, help="K")
    p.add_argument("--pressure", type=float, default=1.0, help="bar")
    p.add_argument("--cutoff", type=float, default=1.0, help="nm (amber format only)")
    p.add_argument("--report-interval", type=int, default=1000)
    a = p.parse_args(argv)
    run(a.out_dir, a.format, a.steps, a.platform, a.temperature, a.pressure,
        a.cutoff, a.report_interval)


if __name__ == "__main__":
    main()
