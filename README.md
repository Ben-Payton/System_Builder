# System_Builder

This is a small system builder for OpenMM systems. It packs simple boxes and generates force fields for OpenMM.

It exists because the OpenFF packages are not pip installable. It contains a main script which packs and parameterizes simple systems and exports them so that they can be read in OpenMM scripts. This decouples the simulation from conda: you need conda (with OpenFF) only to *build* a system, and a plain `pip install openmm` to *run* it.

## Install the builder environment

```bash
conda env create -f environment.yml
conda activate system_builder
pip install -e .
```

## Build a system

```bash
system-builder build examples/ethanol_water.yaml -o out/
```

Options: `--overwrite` (allow writing into a non-empty directory; otherwise the build refuses), `-v` (debug logging, full tracebacks). Exit codes: 0 success, 1 build failure or non-empty output directory, 2 invalid configuration.

Example configs are in `examples/`: `ethanol_water.yaml`, `ionic_liquid.yaml` ([EMIM][OTf] with charge scaling), `nacl_water.yaml`.

### Config reference

| Key | Meaning | Default |
|---|---|---|
| `components[].name` | Label for the species | required |
| `components[].smiles` / `file` | SMILES, or SDF/mol2 with user partial charges (exactly one) | required |
| `components[].count` | Number of molecules | required |
| `components[].charge_scale` | Multiplier for partial charges | `1.0` |
| `components[].residue_name` | Residue name, at most 3 characters | derived from `name` |
| `box.size_nm` / `box.density_g_cm3` | Cubic edge length, or target density (exactly one) | required |
| `forcefield.small_molecule` | Small-molecule force field | `gaff-2.11` |
| `forcefield.water` | Water model | `tip3p` |
| `nonbonded.method` | `PME`, `LJPME`, `Ewald`, `CutoffPeriodic`, `CutoffNonPeriodic`, `NoCutoff` | `PME` |
| `nonbonded.cutoff_nm` | Nonbonded cutoff | `1.0` |
| `constraints` | `None`, `HBonds`, `AllBonds`, `HAngles` | `HBonds` |
| `tolerance_A` | Packmol minimum distance | `2.0` |
| `seed` | Recorded in `metadata.json`; packmol placement is currently **not** seeded, so packing is not bit-reproducible | none |

### Force fields

- Organic molecules and molecular ions: GAFF2 (`gaff-2.11`) with AM1-BCC charges (via AmberTools), unless the molecule file carries partial charges.
- Water: Amber TIP3P (`amber14/tip3p.xml`).
- Monatomic ions (`[Na+]`, `[Cl-]`, ...): Joung-Cheatham TIP3P ion parameters from `amber14/tip3p.xml`, not GAFF.
- AM1-BCC (AmberTools sqm) does not support every element, e.g. boron, so BF4- cannot be charged. System_Builder stops with an error naming the component (CLI exit code 2). Supply partial charges yourself in an SDF/mol2 file via `file:` (such components use those charges instead of AM1-BCC):

  ```yaml
  components:
    - name: bf4
      file: bf4_charges.sdf   # relative to the config file; carries partial charges
      count: 100
      residue_name: BF4
  ```
- Charge scaling: `charge_scale` multiplies a component's partial charges (and the corresponding 1-4 exception charge products). Used for ionic liquids (typically 0.7-0.9) to mimic electronic polarization that fixed-charge force fields lack.

## Output files

| File | Content |
|---|---|
| `system.xml` | Serialized OpenMM `System` (cutoff, constraints, PME baked in) |
| `system.pdb` | Topology and positions, with box (CRYST1) and CONECT records |
| `system.prmtop`, `system.inpcrd` | Amber topology/parameters and coordinates with box |
| `metadata.json` | Config echo, package versions, atom/residue counts, box (nm), net charge, energy-check result, warnings, timestamp |
| `gaff_cache.json` | GAFF template cache (reuses AM1-BCC charges on rebuilds) |

After writing, the builder compares the potential energy of `system.xml` against a system rebuilt from the prmtop (same cutoff/method/constraints), at the exported and at a perturbed geometry, and also compares the number and length of constraints. A mismatch beyond rtol 1e-3 is recorded in `metadata.json` and logged as a warning; the export is not aborted.

## Using the output in a pip-only environment

Only `pip install openmm` is needed.

**XML route**

```python
import openmm
from openmm import app, unit

pdb = app.PDBFile("out/system.pdb")
system = openmm.XmlSerializer.deserialize(open("out/system.xml").read())
system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))  # for NPT
integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
sim = app.Simulation(pdb.topology, system, integrator)
sim.context.setPositions(pdb.positions)
sim.minimizeEnergy()
sim.step(10000)
```

**Amber route**

```python
prmtop = app.AmberPrmtopFile("out/system.prmtop")
inpcrd = app.AmberInpcrdFile("out/system.inpcrd")
system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds)
# then add a barostat and continue as above, using prmtop.topology and inpcrd.positions
```

**Complete script:** `examples/run_simulation.py` (imports only `openmm`) does minimization, NPT, and DCD/log output:

```bash
python examples/run_simulation.py out/ --format xml --steps 50000 --platform CUDA
python examples/run_simulation.py out/ --format amber --temperature 350 --pressure 1
```

## Portability notes

- All outputs are plain text and independent of OS and CPU architecture: build on x86 Linux/WSL, run on ARM macOS or a GPU cluster.
- OpenMM XML is versioned. If the runtime OpenMM is older than the one in the build environment, use the Amber route.
- The XML route bakes in cutoff, constraints and the PME setup. The Amber route lets you choose method, cutoff and constraints at load time.
- The serialized system has no barostat; add your own for NPT.
- Constrained bonds (HBonds, rigid water) have no bond force in the OpenMM system, so the prmtop makes them explicit. Water is written as Amber TIP3P as tleap does (O-H and H-H bonds k=553 kcal/mol/A^2, H-O-H angle 100 kcal/mol/rad^2 at 104.52 deg), so `createSystem(constraints=HBonds)` gives rigid water again; other constrained bonds get their constraint length and a typical force constant. This only matters if you load the prmtop without constraints; the XML route is exact.

## Limitations

- AM1-BCC cannot charge molecules containing some elements (e.g. boron); provide charges via `file:` (see above).
- Fixed-charge, non-polarizable force fields only; GAFF2/AM1-BCC quality applies to your molecules.
- Cubic boxes only.
- Packing is not reproducible: the interchange packmol wrapper does not expose a seed.
- AM1-BCC charge generation is slow for large molecules; it runs once per species and is cached.
- Monatomic ions use TIP3P-specific parameters, so combine them with TIP3P water only.
- The energy check covers consistency between the XML and Amber exports, not the physical correctness of the parameters.
