"""klasp from Python.

The point of this module is that a script does not have to build an argv string
to run klasp. A configuration is a tree of plain values, a run is a function
call in this process, and a landscape comes back as data rather than as a
directory of CIF files.

    import klasp

    cfg = klasp.Config()
    cfg.control.basis_paths = ["example-inputs/acetic_acid.json"]
    cfg.control.space_groups = [14]
    cfg.control.targets = {14: 24}
    cfg.output.dir = "/tmp/run"

    result = klasp.run(cfg)                       # no MPI, no mpirun
    rows = klasp.read_landscape("/tmp/run/csp.db")
    print(rows[0]["energy"], rows[0]["cell"])

Two things are worth knowing before the first call.

**Kokkos.** klasp computes on Kokkos Views, and Kokkos has to be started before
anything touches one and stopped once, at the end. ``klasp.initialize()`` does
the first and an ``atexit`` hook does the second, so a script that never
mentions either is correct: the first call that needs Kokkos starts it. Call
``initialize`` yourself only when you want to hand Kokkos arguments, e.g.
``klasp.initialize(["--kokkos-num-threads=8"])`` -- and do it before anything
else, because afterwards it is a no-op. ``finalize()`` is final: Kokkos cannot
be restarted in the same process, so there is no reason to call it early.

**Output.** A run narrates. By default that narration goes to ``sys.stdout``
and ``sys.stderr`` rather than to the process's file descriptors, which means
``contextlib.redirect_stdout`` catches it like any other Python output. To
collect it as a string use ``with klasp.capture_output() as log:``; to silence
it use ``klasp.set_log_level(klasp.ERROR)``.
"""

from __future__ import annotations

import atexit
import contextlib
import copy
import json
import os
import re
import sys
import tempfile
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from . import _klasp

__all__ = [
    "Config",
    "ERROR",
    "Frame",
    "Result",
    "Structure",
    "WARN",
    "INFO",
    "DETAIL",
    "DEBUG",
    "STAGE_GENERATED",
    "STAGE_CHARGES",
    "STAGE_MULTIPOLE",
    "STAGE_FINAL",
    "backend",
    "capture_output",
    "database_meta",
    "default_config",
    "default_generate_options",
    "default_structure_options",
    "effective_config",
    "energy",
    "evaluate",
    "generate",
    "finalize",
    "has_pet",
    "initialize",
    "is_initialized",
    "log_level",
    "log_to",
    "optimize",
    "parse_space_groups",
    "read_landscape",
    "request_stop",
    "run",
    "set_log_level",
    "version",
]

# Log levels, as klasp/log.hpp orders them.
ERROR, WARN, INFO, DETAIL, DEBUG = 0, 1, 2, 3, 4

# Stage numbers, as klasp/dedup.hpp names them. Stage 0 passed to read_landscape
# means "the highest stage present".
STAGE_GENERATED, STAGE_CHARGES, STAGE_MULTIPOLE, STAGE_FINAL = 0, 1, 2, 3

backend = _klasp.backend
has_pet = _klasp.has_pet


def version() -> str:
    """The commit klasp was built from, with branch and build date."""
    return _klasp.__version__


__version__ = _klasp.__version__


# ---------------------------------------------------------------------------
# Kokkos lifetime
# ---------------------------------------------------------------------------

def initialize(args: Optional[Sequence[str]] = None) -> bool:
    """Start Kokkos. True if this call started it, False if it was running.

    Raises RuntimeError after finalize(): Kokkos cannot be restarted.
    """
    return _klasp.initialize(list(args) if args else [])


def finalize() -> None:
    """Stop Kokkos and drop the log sink. Harmless if never started."""
    _klasp.finalize()


def is_initialized() -> bool:
    return _klasp.is_initialized()


def _ensure_initialized() -> None:
    if not _klasp.is_initialized():
        _klasp.initialize([])


# Registered at import so Kokkos is finalized in the right order even without an
# explicit finalize().
atexit.register(_klasp.finalize)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _stream_sink(level: int, text: str) -> None:
    # sys.stdout is looked up per message rather than captured once, so
    # contextlib.redirect_stdout works on a run in progress.
    stream = sys.stderr if level <= WARN else sys.stdout
    if stream is not None:
        stream.write(text)


def log_to(callback: Optional[Callable[[int, str], None]]) -> None:
    """Send klasp's narration to ``callback(level, text)``.

    ``text`` is already formatted and already carries its own newline. None
    restores the default, which writes to ``sys.stdout`` / ``sys.stderr``.
    The callback may be invoked from threads other than the calling one.
    """
    _klasp.set_log_sink(_stream_sink if callback is None else callback)


def set_log_level(level: int) -> None:
    """Drop everything above ``level``. INFO is the default; ERROR is silence
    for everything but a refusal."""
    _klasp.set_log_level(level)


def log_level() -> int:
    return _klasp.log_level()


class _Log:
    """What ``capture_output`` yields: the narration, as it accumulates."""

    def __init__(self) -> None:
        self.records: List[tuple] = []

    def __call__(self, level: int, text: str) -> None:
        self.records.append((level, text))

    @property
    def text(self) -> str:
        return "".join(t for _, t in self.records)

    def at_level(self, level: int) -> str:
        return "".join(t for lv, t in self.records if lv == level)

    def __str__(self) -> str:
        return self.text

    def __len__(self) -> int:
        return len(self.records)


@contextlib.contextmanager
def capture_output(level: int = DETAIL) -> Iterator[_Log]:
    """Collect a run's narration instead of printing it.

    >>> with klasp.capture_output() as log:
    ...     klasp.run(cfg)
    >>> print(log.text)
    """
    log = _Log()
    previous_level = _klasp.log_level()
    _klasp.set_log_level(level)
    _klasp.set_log_sink(log)
    try:
        yield log
    finally:
        _klasp.set_log_sink(_stream_sink)
        _klasp.set_log_level(previous_level)


# Installed at import: klasp's own default writes to file descriptor 1, which a
# Python caller cannot see into.
_klasp.set_log_sink(_stream_sink)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def default_config() -> Dict[str, Any]:
    """The whole configuration tree with every default in place, as a dict."""
    return _tree_from_klasp(json.loads(_klasp.default_config_json()))


_RAW_DEFAULTS = json.loads(_klasp.default_config_json())

# `control.targets` is a std::map<int,int>, which nlohmann renders as a list of
# pairs; it is converted to and from a dict at the boundary, if the defaults show
# that shape.
_TARGETS_ARE_PAIRS = isinstance(_RAW_DEFAULTS["control"]["targets"], list)


def _targets_from_klasp(value: Any) -> Dict[int, int]:
    if isinstance(value, dict):
        return {int(k): int(v) for k, v in value.items()}
    return {int(k): int(v) for k, v in value}


def _targets_to_klasp(value: Any) -> Any:
    if isinstance(value, dict):
        pairs = [[int(k), int(v)] for k, v in sorted(value.items())]
    else:
        pairs = [[int(k), int(v)] for k, v in value]
    return pairs if _TARGETS_ARE_PAIRS else {str(k): v for k, v in pairs}


def _tree_from_klasp(tree: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(tree)
    if "control" in out and "targets" in out["control"]:
        out["control"]["targets"] = _targets_from_klasp(out["control"]["targets"])
    return out


def _tree_to_klasp(tree: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(tree)
    if "control" in out and "targets" in out["control"]:
        out["control"]["targets"] = _targets_to_klasp(out["control"]["targets"])
    return out


_DEFAULTS = _tree_from_klasp(_RAW_DEFAULTS)


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class _Section:
    """One block of the tree -- generator, optimizer, control -- as attributes.

    A name that is not a field of that block raises rather than being silently
    stored, because a configuration key that does nothing is the failure mode
    this whole tree exists to prevent: the old CLI had a --threads that did
    nothing and nobody found out for a while.
    """

    __slots__ = ("_name", "_data")

    def __init__(self, name: str, data: Dict[str, Any]) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_data", data)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(
                f"{self._name} has no setting {key!r}; "
                f"it has {', '.join(sorted(self._data))}"
            ) from None

    def __setattr__(self, key: str, value: Any) -> None:
        if key not in self._data:
            raise AttributeError(
                f"{self._name} has no setting {key!r}; "
                f"it has {', '.join(sorted(self._data))}"
            )
        self._data[key] = value

    def __dir__(self) -> List[str]:
        return sorted(self._data)

    def __repr__(self) -> str:
        return f"<klasp config section {self._name}: {self._data!r}>"

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)


class Config:
    """A run configuration: the same tree klasp's JSON config files describe.

    Attribute access down two levels, a dict in and a dict out:

        cfg = klasp.Config({"control": {"space_groups": [14]}})
        cfg.optimizer.n_iterations = 200
        cfg.to_dict()["optimizer"]["n_iterations"]

    Names are checked against the tree klasp actually has, so a typo raises
    where it is written rather than being dropped on the way into C++.
    """

    __slots__ = ("_data",)

    def __init__(self, patch: Optional[Dict[str, Any]] = None) -> None:
        object.__setattr__(self, "_data", copy.deepcopy(_DEFAULTS))
        if patch:
            self.update(patch)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """A klasp JSON config file, which names only what it changes."""
        with open(path) as handle:
            return cls(json.load(handle))

    @classmethod
    def from_json(cls, text: str) -> "Config":
        return cls(json.loads(text))

    def update(self, patch: Dict[str, Any]) -> "Config":
        """Merge a (possibly partial) tree in, section by section."""
        for section, values in patch.items():
            if section not in self._data:
                raise KeyError(
                    f"no configuration section {section!r}; "
                    f"there are {', '.join(sorted(self._data))}"
                )
            if not isinstance(values, dict):
                raise TypeError(f"section {section!r} must be a dict")
            unknown = set(values) - set(self._data[section])
            if unknown:
                raise KeyError(
                    f"{section} has no setting(s) {', '.join(sorted(unknown))}"
                )
            if section == "control" and "targets" in values:
                values = dict(values)
                values["targets"] = _targets_from_klasp(values["targets"])
            self._data[section] = _deep_merge(self._data[section], values)
        return self

    def set(self, assignment: str) -> "Config":
        """Apply one ``section.key=value``, the way ``--set`` does.

        The parse and the conversion happen in C++, so this accepts exactly what
        the command line accepts and refuses the rest by name.
        """
        patched = _klasp.apply_override(self.to_json(indent=None), assignment)
        object.__setattr__(self, "_data", _tree_from_klasp(json.loads(patched)))
        return self

    def problems(self) -> List[str]:
        """Every reason klasp would refuse to run this configuration, one
        sentence each, naming the key; empty when there are none.

        These are the checks ``klasp csp`` makes -- Wolf electrostatics under
        the streaming optimizer, a sampler with no engine to run on, a model
        named by half a pair of files -- applied by klasp itself. Having no
        inputs or space groups yet is not a problem here: a configuration can
        be a template, and :func:`run` checks that when it starts.

        Raises ValueError for a value klasp cannot read at all, such as an
        enum name it does not know.
        """
        return list(_klasp.config_problems(self.to_json(indent=None)))

    def validate(self) -> "Config":
        """Raise ValueError, listing every problem, if klasp would refuse this
        configuration. See :meth:`problems`."""
        problems = self.problems()
        if problems:
            raise ValueError("this configuration cannot run:\n  "
                             + "\n  ".join(problems))
        return self

    def __getattr__(self, name: str) -> _Section:
        data = object.__getattribute__(self, "_data")
        if name in data:
            return _Section(name, data[name])
        raise AttributeError(
            f"no configuration section {name!r}; there are {', '.join(sorted(data))}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._data:
            if not isinstance(value, dict):
                raise TypeError(
                    f"{name} is a section; assign a dict, or set its fields "
                    f"one at a time"
                )
            self.update({name: value})
            return
        raise AttributeError(f"no configuration section {name!r}")

    def __dir__(self) -> List[str]:
        return sorted(self._data)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """The tree as klasp writes it, so this is a klasp config file."""
        return json.dumps(_tree_to_klasp(self._data), indent=indent)

    def __repr__(self) -> str:
        return f"Config({self.to_json(indent=None)})"


def _config_json(config: Any) -> str:
    if isinstance(config, Config):
        return config.to_json(indent=None)
    if isinstance(config, dict):
        return Config(config).to_json(indent=None)
    if isinstance(config, str):
        return config
    raise TypeError("config must be a klasp.Config, a dict, or a JSON string")


def parse_space_groups(spec: str):
    """A number, a comma list, or a preset name (``common25``, ``fine10``...).

    Returns ``(space_groups, per_group_targets)``; the targets are empty unless
    the preset carries them.
    """
    return _klasp.parse_space_groups(spec)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run(config: Any) -> Dict[str, Any]:
    """Run a CSP in this process and return what it stored.

    No MPI is involved: this is the in-process WorkSource / ResultSink / Comm
    path, which produces the landscape ``mpirun -n 1 klasp csp`` produces, row
    for row. Kokkos is started if it has not been.

    Returns ``{"stored": int, "unique": int, "stopped_early": bool}``. The
    landscape itself is in ``{output.dir}/csp.db``; read it back with
    :func:`read_landscape`.

    Raises ValueError for a configuration ``klasp csp`` would refuse, before
    anything is written, and RuntimeError when the run itself refuses to start
    (a resume that does not match, a landscape it would overwrite).

    Ctrl-C works: klasp does not install signal handlers from here. To ask a
    running search to wrap up from another thread, call :func:`request_stop`.
    """
    _ensure_initialized()
    return _klasp.run_csp(_config_json(config))


def request_stop() -> None:
    """Ask a running CSP to finish the unit in hand and write out."""
    _klasp.request_stop()


# ---------------------------------------------------------------------------
# Reading results
# ---------------------------------------------------------------------------

def read_landscape(db_path: str, **query) -> List[Dict[str, Any]]:
    """A landscape as a list of dicts, lowest energy first.

    Keyword options, whose defaults are klasp's own: ``stage`` (0, the highest
    stage the database has), ``space_group`` (0, every group), ``max_energy``,
    ``unique_only`` (True), ``keep_catastrophes`` and ``keep_warned`` (both
    False), ``catastrophe_threshold``, ``only_missing_stage`` and
    ``descriptors`` (False). The filters are named and defaulted as ``klasp
    extract``'s, so a structure below the catastrophe threshold or carrying a
    warning is dropped unless kept -- the same rows extract would write.

    Each row carries
    ``seed``, ``space_group``, ``stage``, ``n_mols``, ``energy`` (kJ/mol per
    formula unit), ``density``, ``cell`` (a, b, c in Angstrom and the three
    angles in degrees), ``translation`` and ``rotation`` (three numbers per
    independent molecule), and the convergence and warning columns. Pass
    ``descriptors=True`` for the AMD vector and the powder pattern; they are
    left out by default because a landscape is routinely 10^5 rows and the
    pattern alone is 1024 numbers each.

    Needs no Kokkos: this reads a database.
    """
    return _klasp.read_landscape(db_path, **query)


def database_meta(db_path: str, key: str) -> str:
    """One value from a landscape's ``meta`` table: ``config_json``,
    ``basis_json``, ``fingerprint``, and the rest of what a run records."""
    return _klasp.database_meta(db_path, key)


def effective_config(path: str) -> Dict[str, Any]:
    """The configuration a run actually ran with, as a dict.

    Reads ``config.json``, which every run writes beside its database. ``path``
    may be that directory or the database in it.

    Not the database's ``config_json`` meta value: that one is the canonical
    projection the resume fingerprint is taken over, which is deliberately a
    subset -- it covers what changes the answer, not what changes how fast the
    answer arrives. Ask for it by name through :func:`database_meta` if that is
    what you want.
    """
    if os.path.isdir(path):
        config_path = os.path.join(path, "config.json")
    else:
        config_path = os.path.join(os.path.dirname(path), "config.json")
    with open(config_path) as handle:
        return _tree_from_klasp(json.load(handle))


# ---------------------------------------------------------------------------
# A single energy
# ---------------------------------------------------------------------------

def energy(paths, **options) -> Dict[str, Any]:
    """Evaluate (or relax) one or more structures, as ``klasp energy`` does.

    ``paths`` is a structure file, a :class:`Structure`, a structure document
    dict, or a list mixing any of them. Structures and dicts are written to a
    scratch directory for the command, named after their title. For one
    structure's energy as data -- its electrostatic and Buckingham terms, its
    geometry, the relaxed structure -- :func:`evaluate` and :func:`optimize`
    return a :class:`Result` instead. Keyword options are the
    ``energy`` subcommand's, by the same names and with the same defaults:
    ``relax``, ``backend`` (``"ff"`` or ``"pet"``), ``perturb``, ``isolated``,
    ``fixed_cell``, ``pressure_gpa``, ``no_taper``, ``max_order``,
    ``ewald_accuracy``, ``iterations``, ``grad_tol``, ``output_dir``,
    ``write_json``, ``trajectory``, ``summary``, ``symmetry``, ``expand_p1``, ``rigid``, ``device``,
    ``model``, ``model_json``, ``model_weights``. An option it does not have
    raises TypeError rather than being ignored.

    Returns ``{"energy_kj_per_mol": [...], "structures": int, "failed": int}``;
    the energies are kJ/mol per molecule and are in input order. Raises if any
    structure failed, which is what the command does.

    This one still narrates a table as it goes -- the same table the command
    prints -- so wrap it in :func:`capture_output` if you want only the numbers.
    """
    items = list(paths) if isinstance(paths, (list, tuple)) else [paths]
    _ensure_initialized()
    with tempfile.TemporaryDirectory(prefix="klasp-energy-") as scratch:
        files: List[str] = []
        used = set()
        for index, item in enumerate(items):
            if isinstance(item, (str, os.PathLike)):
                files.append(os.fspath(item))
                continue
            structure = item if isinstance(item, Structure) else Structure(item)
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", structure.title or "structure")
            if stem in used:
                stem = f"{stem}_{index}"
            used.add(stem)
            path = os.path.join(scratch, stem + ".json")
            structure.to_file(path)
            files.append(path)
        return _klasp.energy(files, **options)


# ---------------------------------------------------------------------------
# One structure
# ---------------------------------------------------------------------------

class Structure:
    """A crystal structure: a basis and a crystal, as klasp reads them.

    The basis is the molecule types (sites with positions in Angstrom and their
    multipoles), the potentials and the settings; the crystal is the cell and
    where each molecule sits. Both are plain dicts and lists, edited in place::

        s = klasp.Structure.from_file("example-inputs/paracetamol.json")
        s.potentials["damping_range"] = 1.0
        s.cell["a"] *= 1.01
        klasp.evaluate(s).energy

    The flat layout klasp also reads -- cell, molecules and molecule_types at the
    top level -- is rearranged into this one on the way in. A CSP basis with no
    crystal is refused: it has no geometry to evaluate.
    """

    __slots__ = ("_doc",)

    def __init__(self, document: Any) -> None:
        if isinstance(document, Structure):
            document = document.to_dict()
        if not isinstance(document, dict):
            raise TypeError("a Structure is built from a dict; see from_file and from_json")
        self._doc = json.loads(_klasp.normalize_structure(json.dumps(document)))

    @classmethod
    def from_file(cls, path: str) -> "Structure":
        with open(path) as handle:
            return cls(json.load(handle))

    @classmethod
    def from_json(cls, text: str) -> "Structure":
        return cls(json.loads(text))

    @classmethod
    def from_landscape(cls, db_path: str, seed: int, space_group: int,
                       stage: int = 0) -> "Structure":
        """One structure out of a landscape, with the basis the run recorded.

        The geometry is the rigid-molecule one the force field relaxed. Stage 0
        means the highest stage the database holds; a PET stage stores its own
        flexible geometry, which this does not reconstruct.
        """
        return cls(json.loads(_klasp.landscape_structure(db_path, int(seed),
                                                         int(space_group), int(stage))))

    @property
    def basis(self) -> Dict[str, Any]:
        return self._doc["basis"]

    @property
    def crystal(self) -> Dict[str, Any]:
        return self._doc["crystal"]

    @property
    def molecule_types(self) -> List[Dict[str, Any]]:
        return self.basis["molecule_types"]

    @property
    def molecules(self) -> List[Dict[str, Any]]:
        return self.crystal["molecules"]

    @property
    def cell(self) -> Dict[str, float]:
        """a, b, c in Angstrom and alpha, beta, gamma in degrees."""
        return self.crystal["cell"]

    @cell.setter
    def cell(self, value: Dict[str, float]) -> None:
        missing = {"a", "b", "c", "alpha", "beta", "gamma"} - set(value)
        if missing:
            raise KeyError(f"a cell needs {', '.join(sorted(missing))}")
        self.crystal["cell"] = dict(value)

    @property
    def potentials(self) -> Dict[str, Any]:
        return self.basis.setdefault("potentials", {})

    @property
    def settings(self) -> Dict[str, Any]:
        return self.basis.setdefault("settings", {})

    @property
    def title(self) -> Optional[str]:
        return self._doc.get("title")

    @property
    def seed(self) -> Optional[int]:
        """The generator seed this structure was drawn from, when it was drawn."""
        return self._doc.get("seed")

    def copy(self) -> "Structure":
        return Structure(copy.deepcopy(self._doc))

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._doc)

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self._doc, indent=indent)

    def to_file(self, path: str) -> None:
        """Write the document, which klasp and klasp-energy read back."""
        with open(path, "w") as handle:
            handle.write(self.to_json())
            handle.write("\n")

    def to_cif(self) -> str:
        """The structure as explicit atoms in P1, as CIF text."""
        return evaluate(self).cif

    def __repr__(self) -> str:
        c = self.cell
        return (f"<klasp.Structure {self.title or ''!s}: {len(self.molecules)} molecules, "
                f"a={c['a']:.3f} b={c['b']:.3f} c={c['c']:.3f} "
                f"alpha={c['alpha']:.2f} beta={c['beta']:.2f} gamma={c['gamma']:.2f}>")


def _lattice(cell: Sequence[float]) -> List[List[float]]:
    # klasp's cell matrix is row-major with the lattice vectors as its columns;
    # this gives the vectors in turn, which is ASE's convention.
    return [[cell[0], cell[3], cell[6]], [cell[1], cell[4], cell[7]],
            [cell[2], cell[5], cell[8]]]


def _triples(flat: Sequence[float]) -> List[tuple]:
    return [tuple(flat[i:i + 3]) for i in range(0, len(flat), 3)]


class Frame:
    """One optimiser iteration: the accepted point after it, and what it tried.

    ``lattice`` and ``positions`` are in the optimiser's own frame, Angstrom; the
    cell need not be in the orientation a structure document is read back in.
    """

    __slots__ = ("iteration", "energy", "trial_energy", "accepted", "grad_max",
                 "trust_radius", "lattice", "positions")

    def __init__(self, data: Dict[str, Any]) -> None:
        self.iteration = data["iteration"]
        self.energy = data["energy"]
        self.trial_energy = data["trial_energy"]
        self.accepted = data["accepted"]
        self.grad_max = data["grad_max"]
        self.trust_radius = data["trust_radius"]
        self.lattice = _lattice(data["cell"])
        self.positions = _triples(data["positions"])

    def __repr__(self) -> str:
        return (f"<klasp.Frame {self.iteration}: E={self.energy:.6f} "
                f"{'accepted' if self.accepted else 'rejected'} max|grad|={self.grad_max:.2e}>")


class Result:
    """What evaluate() and optimize() return. Energies are kJ/mol per molecule.

    ``structure`` is the final geometry as a Structure, ready to evaluate, edit or
    write. ``components`` splits the lattice energy at that geometry into its
    electrostatic and Buckingham terms, without any pressure term. With
    ``trajectory=True``, ``trajectory`` holds a Frame per optimiser iteration.
    With ``perturb=N``, ``perturbed`` holds the relaxations of N randomly
    displaced copies of the input, which tell a real minimum from a saddle the
    input happened to sit on. ``timings`` is where the wall-clock seconds went:
    ``total``, ``load``, ``initial_energy``, ``setup``, ``optimize`` and
    ``per_iteration``, the optimiser's ``pairs``, ``reciprocal``, ``post``,
    ``expand`` and ``step``, then ``final_structure`` and ``trajectory``.
    """

    def __init__(self, data: Dict[str, Any], perturbed: Sequence["Result"] = ()) -> None:
        self.label = data["label"]
        self.energy = data["energy"]
        self.initial_energy = data["initial_energy"]
        self.components = dict(data["components"])
        self.relaxed = data["relaxed"]
        self.converged = data["converged"]
        self.iterations = data["iterations"]
        self.grad_max = data["grad_max"]
        self.trust_radius = data["trust_radius"]
        self.restarts = data["restarts"]
        self.space_group = data["space_group"]
        self.n_molecules = data["n_molecules"]
        self.n_atoms = data["n_atoms"]
        self.atomic_numbers = list(data["atomic_numbers"])
        self.symbols = list(data["symbols"])
        names = ("a", "b", "c", "alpha", "beta", "gamma")
        self.cell = dict(zip(names, data["cell_params"]))
        self.lattice = _lattice(data["cell"])
        self.positions = _triples(data["positions"])
        self.structure = (Structure(json.loads(data["structure_json"]))
                          if data["structure_json"] else None)
        self.cif = data["cif"]
        self.trajectory = [Frame(f) for f in data["trajectory"]]
        self.perturbed = list(perturbed)
        self.timings = dict(data["timings"])
        self._xyz = data["trajectory_xyz"]

    def write_xyz(self, path: str) -> None:
        """The trajectory as extended XYZ, one frame per iteration."""
        if not self._xyz:
            raise ValueError("no trajectory was recorded; pass trajectory=True to optimize()")
        with open(path, "w") as handle:
            handle.write(self._xyz)

    def write_cif(self, path: str) -> None:
        """The final geometry as explicit atoms in P1."""
        if not self.cif:
            raise ValueError("there is no geometry to write: the energy was not finite")
        with open(path, "w") as handle:
            handle.write(self.cif)

    def __repr__(self) -> str:
        what = "relaxed" if self.relaxed else "single point"
        state = (f", {'converged' if self.converged else 'not converged'} "
                 f"after {self.iterations} iterations" if self.relaxed else "")
        return (f"<klasp.Result {what}: E={self.energy:.6f} kJ/mol{state}, "
                f"{self.timings.get('total', 0.0):.2f} s>")


def default_structure_options() -> Dict[str, Any]:
    """optimize()'s options and their defaults; evaluate() takes the first three
    (max_order, ewald_accuracy, no_taper)."""
    return dict(_klasp.default_structure_options())


def _structure_document(structure: Any) -> str:
    if isinstance(structure, Structure):
        return structure.to_json(indent=None)
    if isinstance(structure, dict):
        return json.dumps(structure)
    if isinstance(structure, (str, os.PathLike)):
        with open(structure) as handle:
            return handle.read()
    raise TypeError("a structure is a klasp.Structure, a dict, or a path to a structure file")


def evaluate(structure: Any, **options) -> Result:
    """The force-field energy of one structure at its geometry.

    ``structure`` is a Structure, a document dict, or a path. Options:
    ``max_order``, ``ewald_accuracy`` and ``no_taper``, as klasp energy takes them.
    An option it does not have raises TypeError.
    """
    _ensure_initialized()
    return Result(_klasp.evaluate_structure(_structure_document(structure), **options))


def optimize(structure: Any, **options) -> Result:
    """Relax one structure under the force field, cell included unless held.

    ``structure`` is a Structure, a document dict, or a path. Options, with the
    defaults :func:`default_structure_options` reports: ``fixed_cell``,
    ``pressure_gpa`` (the energy then includes PV), ``iterations``, ``grad_tol``,
    ``trajectory``, ``perturb``, ``max_order``, ``ewald_accuracy``, ``no_taper``.
    An option it does not have raises TypeError.

    The optimiser runs on the device and does not come back to the host until it
    is done, trajectory or not; the trajectory is kept on the device and handed
    back at the end.
    """
    _ensure_initialized()
    results = _klasp.relax_structure(_structure_document(structure), **options)
    return Result(results[0], perturbed=[Result(r) for r in results[1:]])


# ---------------------------------------------------------------------------
# Generating structures
# ---------------------------------------------------------------------------

def default_generate_options() -> Dict[str, Any]:
    """generate()'s options and their defaults, which are a CSP run's."""
    return dict(_klasp.default_generate_options())


def _basis_documents(basis: Any) -> List[str]:
    items = list(basis) if isinstance(basis, (list, tuple)) else [basis]
    documents = []
    for item in items:
        if isinstance(item, (str, os.PathLike)):
            with open(item) as handle:
                documents.append(handle.read())
        elif isinstance(item, dict):
            documents.append(json.dumps(item))
        else:
            raise TypeError("a basis is a CSP basis file, a basis dict, or a list of them "
                            "for a co-crystal")
    return documents


def generate(basis: Any, space_group: int, count: int = 1, **options) -> List[Structure]:
    """Crystal structures from the CSP generator, in memory: nothing is written.

    ``basis`` is a CSP basis file or dict, or a list of them for a co-crystal.
    Each structure is the asymmetric unit with its space group's operations --
    the form a CSP run relaxes, and several times cheaper to optimise than the
    same crystal expanded -- and ``Structure.seed`` says which draw it is. A
    seed is the same structure here as in ``klasp csp``, so ``seed=`` picks up
    a landscape's sequence wherever you want it.

    Options, with the defaults :func:`default_generate_options` reports:
    ``seed`` (the first seed tried), ``max_seeds`` (how many to try before
    returning fewer than ``count``), ``tvp``, ``collision`` (``"direct"`` or
    ``"fft_pack"``), ``nudge_budget``, ``contract``, and the ``fft_*``
    settings. An option it does not have raises TypeError.
    """
    _ensure_initialized()
    documents = _klasp.generate_structures(_basis_documents(basis), int(space_group),
                                           int(count), **options)
    return [Structure(json.loads(document)) for document in documents]
