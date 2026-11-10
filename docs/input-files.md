# Input files

klasp reads two kinds of JSON document: a **basis file**, describing one rigid
molecule and its force field, used by `klasp csp`; and a **structure file**, a
crystal built from such molecules, used by `klasp energy` and `klasp fdcheck`.

occ writes basis files: `occ dma <wavefunction> --write-csp-input molecule.json`.

## Units and frames

| Quantity | Unit |
|---|---|
| Site `position` | Angstrom, in the molecule's own (body) frame |
| Cell `a`, `b`, `c` / `alpha`, `beta`, `gamma` | Angstrom / degrees |
| `translation` | fractional coordinates |
| `orientation` | angle-axis vector, radians (angle = length, axis = direction) |
| Multipoles | atomic units, e a0^l |
| Buckingham `A` / `rho` / `C6` | eV / Angstrom / eV Angstrom^6 |
| `cutoff`, `spline_min`, `h_bond_extend` | Angstrom |
| `tabulated.r_min`, `tabulated.r_max` | Bohr |
| Energies reported by `klasp energy` | kJ/mol per molecule |

The Cartesian frame of a crystal has `a` along x and `b` in the xy plane.

## Basis file

```json
{
  "basis": {
    "molecule_types": [{
      "name": "hcl",
      "sites": [
        {"element": "H", "type": "H_X", "position": [0.0, 0.0, -1.2],
         "multipoles": {"charge": 0.2, "dipole": [-0.05, 0.0, 0.0]}},
        {"element": "Cl", "type": "Cl_X", "position": [0.0, 0.0, 0.06],
         "multipoles": {"charge": -0.2, "dipole": [0.1, 0.0, 0.0],
                        "quadrupole": [0.3, 0.0, 0.0, 0.0, 0.0]}}
      ]
    }],
    "potentials": {
      "cutoff": 15.0,
      "buckingham": [
        {"types": ["H_X", "H_X"],   "A": 131.42, "rho": 0.2809, "C6": 2.885},
        {"types": ["Cl_X", "Cl_X"], "A": 9000.0, "rho": 0.30,   "C6": 60.0},
        {"types": ["H_X", "Cl_X"],  "A": 1100.0, "rho": 0.29,   "C6": 13.0}
      ]
    },
    "settings": {"ewald_accuracy": 1e-6, "max_interaction_order": 4}
  }
}
```

Everything sits under `basis`; other top-level keys (such as `title`) are
ignored. A basis file holds exactly one entry in `molecule_types`.

### Molecule types and sites

| Key | Type | Required | Meaning |
|---|---|---|---|
| `molecule_types[].name` | string | in structure files | Name that `molecules[].type` refers to |
| `molecule_types[].sites` | array | yes | Sites in a fixed order, used for every output |
| `sites[].position` | [x, y, z] | yes | Body-frame position, Angstrom |
| `sites[].type` | string | no (default `"X"`) | Atom type; keys the Buckingham table |
| `sites[].element` | string | no | Element symbol. Always give it: without it klasp guesses from `type` |
| `sites[].label` | string | no | Free-form label |
| `sites[].multipoles` | object | no (default zero) | See below |

The body origin is the point the molecule rotates about and the point placed
at `translation`. klasp does not recentre the molecule, so put the origin at
its centre of mass (occ does this).

### Multipoles

| Key | Components | Rank |
|---|---|---|
| `charge` | number | 0 |
| `dipole` | 3 | 1 |
| `quadrupole` | 5 | 2 |
| `octupole` | 7 | 3 |
| `hexadecapole` | 9 | 4 |

Components are real spherical multipoles in Stone's convention (as written by
GDMA and occ): Racah normalisation, atomic units, body frame, ordered
Q_l0, Q_l1c, Q_l1s, Q_l2c, Q_l2s, ... So `dipole` is [mu_z, mu_x, mu_y].

The rank of the model is the highest key present on any site (at most 4);
missing lower ranks are zero. Pairs of multipoles interact when
l_A + l_B <= rank. `settings.max_interaction_order` can lower the rank and
`--max-order` overrides it. klasp warns when an array has the wrong length or
the molecule carries a net charge.

### Potentials

| Key | Type | Default | Meaning |
|---|---|---|---|
| `cutoff` | number | 15.0 | Pair cutoff, Angstrom |
| `buckingham` | array | none | Pair parameters; without it there is no repulsion-dispersion term |
| `damping` | boolean | true | Tang-Toennies damping of the C6 term |
| `damping_range` | number > 0 | 1.5 | Damping range factor |
| `tabulated` | boolean or object | off | Evaluate the pair potential from splines |

Each `buckingham` entry has `types` (two type names), `A`, `rho` and `C6`, all
required; an optional `elements` pair is carried along but not used. The pair
energy is

    V(r) = A exp(-r/rho) - f6(range * r / rho) * C6 / r^6

where f6 is the Tang-Toennies function of order 6 (f6 = 1 with damping off).
Pairs are looked up by type name; the order of the two names does not matter,
and a pair listed twice keeps the last entry. Entries naming a type that no
site uses are skipped. A missing cross pair i-j is filled from the i-i and j-j
entries by combining rules (A = sqrt(A_ii A_jj), 1/rho the mean of 1/rho_ii and
1/rho_jj, C6 = sqrt(C6_ii C6_jj)), with a warning; with no i-i or j-j entry the
pair stays zero. At most 31 distinct type names.

**Cutoff and taper.** The Buckingham and higher-multipole real-space terms are
smoothly switched off between `cutoff - spline_min` and `cutoff`
(`spline_min` from `settings`, default 2.0 Angstrom). `--no-taper` disables
this.

**Tabulated.** `true` evaluates the same damped Buckingham form from splines
with the defaults; an object sets `enabled` (true), `n_knots` (1500, at least
4), `r_min` (1.0 Bohr) and `r_max` (Bohr; default 0 means the cutoff).

### Settings

| Key | Type | Default | Meaning |
|---|---|---|---|
| `ewald_accuracy` | number in (0, 1) | 1e-6 | Relative accuracy of the Ewald sums; `--ewald-accuracy` overrides it |
| `max_interaction_order` | integer >= 0 | none | Lowers the multipole rank; never raises it |
| `spline_min` | number | 2.0 | Taper width inside the cutoff, Angstrom |
| `h_bond_extend` | number | 0 | Moves each H site this far along its bond to the nearest heavy atom (within 2 Angstrom); negative shortens. Charges and multipoles move with it |

Other keys (occ writes `use_ewald`, `neighbor_radius`, `gradient_tolerance`,
`max_iterations`) are ignored with a warning.

### Co-crystals

Give one basis file per component: `klasp csp a.json b.json` (or
`control.basis_paths`). The asymmetric unit is one molecule from each file, in
order. Atom types with the same name in two files are the same type, and the
first file's entry wins for a pair listed in both. Cross-component pairs
missing from every file are filled by the combining rules above, so list them
explicitly for production work. `cutoff`, damping, `tabulated` and the other
settings come from the first file; `h_bond_extend` is applied per file, but
keep `settings` the same in every file.

## Structure file

Two layouts are accepted. Flat:

```json
{
  "cell": {"a": 6.0, "b": 6.0, "c": 6.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
  "molecule_types": [{"name": "hcl", "sites": [ ... ]}],
  "molecules": [
    {"type": "hcl", "translation": [0.25, 0.25, 0.25]},
    {"type": "hcl", "translation": [0.75, 0.75, 0.75], "parity": -1}
  ],
  "potentials": { ... },
  "settings": {"ewald_accuracy": 1e-6}
}
```

or a basis file with a `crystal` beside it:

```json
{
  "basis": {"molecule_types": [ ... ], "potentials": { ... }, "settings": { ... }},
  "crystal": {
    "cell": {"a": 6.0, "b": 6.0, "c": 6.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    "molecules": [{"type": "hcl", "translation": [0.25, 0.25, 0.25]}]
  },
  "symmetry": [{"op": "x,y,z", "molecule": 0}, {"op": "-x,-y,-z", "molecule": 0}],
  "space_group": 2
}
```

These two examples describe the same crystal. `molecule_types`, `potentials`
and `settings` are as in the basis file; a structure file may hold several
molecule types. `symmetry`, `space_group`, `reference`, `title` and `seed`
are always top-level keys.

| Key | Type | Required | Meaning |
|---|---|---|---|
| `cell` | object | yes | `a`, `b`, `c`, `alpha`, `beta`, `gamma`, all required |
| `molecules[].type` | string | yes | A `molecule_types` name |
| `molecules[].translation` | 3 numbers | yes | Fractional position of the body origin |
| `molecules[].orientation` | 3 numbers | no (default 0) | Angle-axis rotation of the body frame, radians |
| `molecules[].rotation` | 3x3 numbers | no | Cartesian rotation matrix applied to body positions; replaces `orientation` and `parity`. Determinant -1 places the inverted molecule |
| `molecules[].parity` | 1 or -1 | no (default 1) | -1 places the inverted molecule |
| `symmetry` | array | no | Operations generating the copies, below |
| `space_group` | integer 1-230 | no | Sets the lattice-system constraints on the cell when relaxing |
| `reference.total` | number | no | Reference energy, kJ/mol per molecule; `klasp energy` prints the difference |
| `title`, `seed` | any | no | Not used in the calculation; copied to output documents |

Without `symmetry`, each entry in `molecules` is one molecule in the cell.
With it, `molecules` is the asymmetric unit and each `symmetry` entry places
one copy: `op` is a fractional operation in CIF form (`"-x+1/2, y, -z"`, with
coefficients of +/-1 only) and `molecule` (default 0) the index into
`molecules`. Every molecule must have the same number of copies, and entry k
must copy molecule k for each k < `len(molecules)`; list the identity for
molecules 0, 1, ... first, as klasp's own output does.
`space_group` does not generate operations and is not checked against
`symmetry`. With a `symmetry` list, `parity` is ignored and an improper
`rotation` is refused: mirror the body in `molecule_types` instead.

## Batch documents

`klasp energy` also accepts a structure file whose `cell` and `molecules` are
replaced by a `structures` array. The rest of the document is shared.

```json
{
  "molecule_types": [ ... ], "potentials": { ... }, "settings": { ... },
  "structures": [
    {"name": "s0", "cell": { ... }, "molecules": [ ... ]},
    {"name": "s1", "cell": { ... }, "molecules": [ ... ]}
  ]
}
```

Each entry needs `cell` and `molecules`; `name` (default `<file stem>_<index>`)
names its output row and files. Every entry must have the same number of
molecules and sites. Force-field backend only; at most 512 atoms per
structure.

## Output documents

`klasp energy --write-json <file>` and `klasp energy --relax -o <dir>` write a
basis-and-crystal document with every molecule in the cell listed with an
explicit `rotation`, and the energy in `reference.total`. `klasp extract -f json` writes a flat document
with the asymmetric unit, `space_group` and `symmetry`. Both read back into
`klasp energy`.
