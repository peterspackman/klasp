#!/usr/bin/env python3
"""The claim the Python module is worth anything for.

A configuration built from Python values, a CSP run in this process with no
mpirun anywhere, a landscape read back as data, and an energy that is the
energy klasp has always found for acetic acid in P2_1/c. Plus the thing the log
sink exists for: the run's narration collected in Python while the process's own
stdout stays empty.

Run it directly (it takes an input file), or let ctest run it:

    python3 python/tests/test_klasp.py example-inputs/acetic_acid.json
"""

import json
import math
import os
import sys
import tempfile

import klasp

# Lowest force-field lattice energy (kJ/mol per molecule) on acetic acid's
# P2_1/c landscape, with Tang-Toennies damping range 1.5.
ACETIC_GLOBAL_MIN = -58.99
MIN_TOLERANCE = 0.05

# Space group 14 structures needed to reach that minimum (24 stops short).
TARGET = 100

FAILURES = []


def check(condition, message):
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        FAILURES.append(message)


def basis_path():
    # Only take argv[1] if it is a file: under pytest it is the test path.
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        return sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "example-inputs", "acetic_acid.json")


# Runs both as a script and under pytest; this fixture serves the latter.
try:
    import pytest

    @pytest.fixture(scope="module")
    def workdir():
        with tempfile.TemporaryDirectory(prefix="klasp-pytest-") as d:
            yield d
except ImportError:  # running as a script, with no pytest installed
    pass


def test_config():
    print("\nconfiguration")
    defaults = klasp.default_config()
    check(set(defaults) >= {"generator", "optimizer", "forcefield", "pet",
                            "dedup", "output", "control"},
          f"the default tree has all seven sections: {sorted(defaults)}")

    cfg = klasp.Config()
    check(cfg.optimizer.n_iterations == 150,
          f"optimizer.n_iterations defaults to {cfg.optimizer.n_iterations}")

    cfg.optimizer.n_iterations = 200
    check(cfg.to_dict()["optimizer"]["n_iterations"] == 200,
          "an attribute assignment reaches the dict")

    try:
        cfg.optimizer.n_iterationz = 200
        check(False, "a misspelt setting should raise")
    except AttributeError as exc:
        check("n_iterationz" in str(exc),
              f"a misspelt setting raises: {str(exc)[:60]}...")

    cfg.set("generator.tvp=1.35")
    check(abs(cfg.generator.tvp - 1.35) < 1e-12,
          "--set-style overrides go through klasp's own parser")

    sgs, targets = klasp.parse_space_groups("common5")
    check(len(sgs) == 5 and 14 in sgs,
          f"the preset common5 resolves to {sgs} with targets {targets}")

    round_tripped = klasp.Config(json.loads(cfg.to_json())).to_dict()
    check(round_tripped == cfg.to_dict(), "the tree round-trips through JSON")

    check(klasp.Config().problems() == [], "the defaults have no problems")

    # Wolf electrostatics with streaming is refused, as by `klasp csp`.
    wolf = klasp.Config({"forcefield": {"wolf": True}})
    check(any("forcefield.wolf" in p for p in wolf.problems()),
          f"a contradiction is reported by key: {wolf.problems()}")
    try:
        wolf.validate()
        check(False, "validate() should refuse Wolf under streaming")
    except ValueError as exc:
        check("forcefield.wolf" in str(exc), "validate() raises ValueError for it")

    try:
        klasp.Config({"optimizer": {"sampler": "metropolis"}}).validate()
        check(False, "an unknown sampler name should raise")
    except ValueError as exc:
        check("metropolis" in str(exc),
              f"an unknown enum name raises rather than reading as the first one: "
              f"{str(exc)[:60]}...")


def test_run_and_landscape(workdir):
    print("\nrunning a CSP in this process")
    cfg = klasp.Config()
    cfg.control.basis_paths = [basis_path()]
    cfg.control.space_groups = [14]
    cfg.control.targets = {14: TARGET}
    cfg.output.dir = workdir

    # Captures anything klasp writes to fd 1 during the run; it should be empty.
    stray = os.path.join(workdir, "stray-stdout.txt")
    saved_fd = os.dup(1)
    sink_file = open(stray, "w")
    try:
        sys.stdout.flush()
        os.dup2(sink_file.fileno(), 1)
        with klasp.capture_output() as log:
            result = klasp.run(cfg)
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        sink_file.close()

    stray_bytes = os.path.getsize(stray)
    print(f"  run returned {result}")
    print(f"  narration captured in Python: {len(log.text)} characters, "
          f"{len(log)} messages")
    print(f"  written to the process's stdout during the run: {stray_bytes} bytes")
    print("  --- first lines of the captured narration ---")
    for line in log.text.splitlines()[:6]:
        print("  | " + line)
    print("  ---")

    check(stray_bytes == 0,
          "the run wrote nothing to the process's own stdout")
    check(len(log.text) > 200,
          "the run's narration was captured in Python instead")
    check("space group" in log.text.lower() or "14" in log.text,
          "the narration is the run's, not a placeholder")
    check(result["stored"] >= TARGET,
          f"stored {result['stored']} structures for a target of {TARGET}")
    check(0 < result["unique"] <= result["stored"],
          f"of which {result['unique']} are representatives")
    check(result["stopped_early"] is False, "the run was not cut short")

    print("\nreading the landscape back")
    db = os.path.join(workdir, "csp.db")
    rows = klasp.read_landscape(db)
    check(len(rows) > 0, f"read {len(rows)} unique rows")

    best = rows[0]
    energies = [r["energy"] for r in rows]
    print(f"  lowest energy   {best['energy']:.4f} kJ/mol per molecule")
    print(f"  highest kept    {energies[-1]:.4f}")
    print(f"  best cell       a={best['cell'][0]:.4f} b={best['cell'][1]:.4f} "
          f"c={best['cell'][2]:.4f} "
          f"alpha={best['cell'][3]:.2f} beta={best['cell'][4]:.2f} "
          f"gamma={best['cell'][5]:.2f}")
    print(f"  best density    {best['density']:.4f}")
    print(f"  best seed/sg/stage  {best['seed']} / {best['space_group']} / "
          f"{best['stage']}")

    check(energies == sorted(energies), "rows come back sorted by energy")
    check(all(math.isfinite(e) for e in energies), "every energy is finite")
    check(abs(best["energy"] - ACETIC_GLOBAL_MIN) < MIN_TOLERANCE,
          f"the global minimum is {ACETIC_GLOBAL_MIN} kJ/mol "
          f"(got {best['energy']:.4f})")
    check(best["stage"] == klasp.STAGE_MULTIPOLE,
          "the landscape is ranked on the multipole stage")
    check(len(best["cell"]) == 6 and len(best["translation"]) == 3,
          "cell and rigid-body degrees of freedom came across")
    check("powder" not in best, "descriptors are left out unless asked for")

    with_desc = klasp.read_landscape(db, descriptors=True)
    check(len(with_desc[0]["powder"]) == 1024 and len(with_desc[0]["amd"]) == 100,
          "descriptors=True brings the powder pattern and the AMD vector")

    kept = klasp.read_landscape(db, keep_catastrophes=True, keep_warned=True)
    check(len(kept) >= len(rows),
          f"keeping catastrophes and warned structures reads at least as many rows "
          f"({len(kept)} against {len(rows)})")

    print("\nwhat the database recorded about the run")
    recorded = klasp.effective_config(workdir)
    check(recorded["control"]["space_groups"] == [14]
          and recorded["control"]["targets"] == {14: TARGET},
          f"the recorded configuration is the one that ran "
          f"(space_groups={recorded['control']['space_groups']}, "
          f"targets={recorded['control']['targets']})")
    check(len(klasp.database_meta(db, "config_hash")) > 0,
          f"the database records a resume fingerprint: "
          f"{klasp.database_meta(db, 'config_hash')}")



def test_energy(workdir):
    print("\na single energy")
    structure = os.path.join(os.path.dirname(os.path.abspath(basis_path())),
                             "ethylcarbamate.json")
    with klasp.capture_output() as log:
        result = klasp.energy(structure)
    print(f"  energy() returned {result}")
    check(result["structures"] == 1 and result["failed"] == 0,
          "one structure, none failed")
    check(math.isfinite(result["energy_kj_per_mol"][0]),
          f"the energy is {result['energy_kj_per_mol'][0]:.4f} kJ/mol per molecule")
    check("E (kJ/mol/mol)" in log.text,
          "its table went through the sink rather than to stdout")
    print("  --- the captured table ---")
    for line in log.text.splitlines()[-4:]:
        print("  | " + line)

    with klasp.capture_output():
        untapered = klasp.energy(structure, no_taper=True)
    check(math.isfinite(untapered["energy_kj_per_mol"][0]),
          "an option the binding used to leave out (no_taper) now reaches the command")

    try:
        klasp.energy(structure, relaxx=True)
        check(False, "a misspelt energy option should raise")
    except TypeError as exc:
        check("relaxx" in str(exc),
              f"a misspelt energy option raises: {str(exc)[:60]}...")


def test_structure(workdir):
    print("\none structure, as data")
    inputs = os.path.dirname(os.path.abspath(basis_path()))
    structure = klasp.Structure.from_file(os.path.join(inputs, "ethylcarbamate.json"))
    check(len(structure.molecules) == 2 and structure.cell["a"] > 0,
          f"a structure reads its molecules and cell: {structure!r}")

    with klasp.capture_output():
        single = klasp.evaluate(structure)
    check(math.isfinite(single.energy),
          f"evaluate() gives {single.energy:.4f} kJ/mol per molecule")
    check(abs(single.components["electrostatic"] + single.components["buckingham"]
              - single.energy) < 1e-9,
          f"its terms add up: electrostatic {single.components['electrostatic']:.4f}, "
          f"Buckingham {single.components['buckingham']:.4f}")

    with klasp.capture_output():
        via_energy = klasp.energy(structure)
    check(abs(via_energy["energy_kj_per_mol"][0] - single.energy) < 1e-9,
          "energy() takes a Structure and gives the same single point as evaluate()")

    edited = structure.copy()
    edited.potentials["damping_range"] = 1.0
    with klasp.capture_output():
        at_range_1 = klasp.evaluate(edited)
    check(abs(at_range_1.energy - single.energy) > 1e-6
          and structure.potentials.get("damping_range") != 1.0,
          f"an edited copy changes the energy ({at_range_1.energy:.4f} at damping range 1) "
          f"and leaves the original alone")

    with klasp.capture_output():
        relaxed = klasp.optimize(structure, trajectory=True)
    check(relaxed.converged and relaxed.energy < single.energy,
          f"optimize() converged at {relaxed.energy:.4f} after {relaxed.iterations} iterations")
    check(0 < relaxed.timings["optimize"] <= relaxed.timings["total"],
          f"its timings account for the call: {relaxed.timings['total']:.2f} s in all, "
          f"{relaxed.timings['optimize']:.2f} s optimising")
    frames = relaxed.trajectory
    check(len(frames) >= 2 and abs(frames[0].energy - single.energy) < 1e-6
          and frames[-1].energy == relaxed.energy,
          f"its trajectory runs from the input to the answer in {len(frames)} frames")
    check(len(frames[-1].positions) == relaxed.n_atoms and len(relaxed.symbols) == relaxed.n_atoms,
          f"each frame has every atom: {relaxed.n_atoms} atoms, {relaxed.symbols[:4]}...")

    with klasp.capture_output():
        back = klasp.evaluate(relaxed.structure)
    check(abs(back.energy - relaxed.energy) < 1e-3,
          f"the relaxed structure reads back at its energy ({back.energy:.6f} against "
          f"{relaxed.energy:.6f})")

    xyz = os.path.join(workdir, "ethylcarbamate.traj.xyz")
    relaxed.write_xyz(xyz)
    with open(xyz) as handle:
        lines = handle.read().splitlines()
    check(len(lines) == len(frames) * (relaxed.n_atoms + 2),
          f"write_xyz writes one frame per iteration ({len(lines)} lines)")
    check("_cell_length_a" in relaxed.cif, "the result carries its CIF")

    written = os.path.join(workdir, "ethylcarbamate.relaxed.json")
    relaxed.structure.to_file(written)
    with klasp.capture_output():
        reread = klasp.evaluate(written)
    check(abs(reread.energy - back.energy) < 1e-9,
          "a structure written to a file evaluates the same from the path")

    db = os.path.join(workdir, "csp.db")
    best = klasp.read_landscape(db)[0]
    from_db = klasp.Structure.from_landscape(db, best["seed"], best["space_group"])
    with klasp.capture_output():
        rescored = klasp.evaluate(from_db)
    check(abs(rescored.energy - best["energy"]) < 0.1,
          f"a landscape structure re-evaluates to its stored energy "
          f"({rescored.energy:.4f} against {best['energy']:.4f})")
    check(abs(from_db.cell["a"] - best["cell"][0]) < 1e-6,
          f"and read_landscape's cell is the structure's, in Angstrom (a = {best['cell'][0]:.4f})")

    generated = klasp.generate(basis_path(), 14, count=3)
    check(len(generated) == 3 and all(g.seed is not None for g in generated)
          and all(len(g.to_dict()["symmetry"]) == 4 for g in generated),
          f"generate() draws asymmetric units in memory: seeds {[g.seed for g in generated]}")
    again = klasp.generate(basis_path(), 14, count=1, seed=generated[1].seed)[0]
    check(again.to_dict() == generated[1].to_dict(), "a seed draws the same structure again")
    with klasp.capture_output():
        drawn = klasp.evaluate(generated[0])
    check(math.isfinite(drawn.energy),
          f"a generated structure evaluates: {drawn.energy:.4f} kJ/mol per molecule")
    with klasp.capture_output():
        moved = klasp.optimize(generated[0], iterations=3)
    check(moved.structure.seed == generated[0].seed,
          f"a relaxed structure keeps its seed: {moved.structure.seed}")

    try:
        klasp.optimize(structure, iteratons=10)
        check(False, "a misspelt optimize option should raise")
    except TypeError as exc:
        check("iteratons" in str(exc), f"a misspelt option raises: {str(exc)[:60]}...")
    try:
        klasp.Structure.from_file(basis_path())
        check(False, "a CSP basis should not load as a structure")
    except ValueError as exc:
        check("basis" in str(exc), f"a CSP basis is refused as a structure: {str(exc)[:60]}...")


def main():
    print(f"klasp {klasp.version()}")
    print(f"  backend {klasp.backend}, PET {'on' if klasp.has_pet else 'off'}")

    # Initialize explicitly so Kokkos startup output precedes the watched run.
    started = klasp.initialize()
    print(f"  initialize() started Kokkos: {started}")
    check(klasp.is_initialized(), "Kokkos is running")
    check(klasp.initialize() is False, "a second initialize() is a no-op")

    test_config()
    with tempfile.TemporaryDirectory(prefix="klasp-python-") as workdir:
        test_run_and_landscape(workdir)
        test_energy(workdir)
        test_structure(workdir)

    klasp.finalize()
    klasp.finalize()
    print("\na second finalize() was harmless")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
