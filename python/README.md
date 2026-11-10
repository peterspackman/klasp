# klasp from Python

A Python module for configuring and running a search in-process, and for reading
the results back as data.

```python
import klasp

cfg = klasp.Config()
cfg.control.basis_paths = ["example-inputs/acetic_acid.json"]
cfg.control.space_groups = [14]
cfg.control.targets = {14: 24}
cfg.output.dir = "acetic"

klasp.run(cfg)                                   # one process, no MPI
rows = klasp.read_landscape("acetic/csp.db")     # list of dicts, lowest energy first
print(rows[0]["energy"], rows[0]["cell"])
```

## Installing

```bash
pip install .                                    # from the repository root
pip install . -C cmake.define.KLASP_BACKEND=cuda # choose a backend
```

Or build in a CMake tree with `-DKLASP_BUILD_PYTHON=ON` and add `build/python`
to `PYTHONPATH`.

A built module is specific to the machine, Kokkos backend and Python version it
was built with. `KLASP_ARCH_NATIVE` is on by default, so pass
`-DKLASP_ARCH_NATIVE=OFF` to build for other machines.

## Main functions

| | |
|---|---|
| `Config`, `default_config()` | The run configuration, with the same keys as the command line's `--set` |
| `run(config)` | Run a search in this process |
| `read_landscape(db)` | Read a landscape; same filters as `klasp extract` |
| `energy(paths, relax=...)` | The `klasp energy` command, returning kJ/mol per molecule |
| `Structure` | One structure as editable dicts; `from_file`, `from_landscape` |
| `evaluate(structure)`, `optimize(structure)` | Energy, or relaxation of one structure |
| `generate(basis, space_group, count)` | Trial structures from the generator, in memory |
| `capture_output()`, `set_log_level()` | Collect or silence klasp's log output |

Multi-rank runs still use `mpirun klasp csp`.
