#pragma once
// Read a crystal from a CIF (parsed by gemmi) and return whole molecules: expand
// the asymmetric unit and merge coincident images, bond by covalent radii plus a
// tolerance over neighbouring cells, and unwrap each component across the cell.
// Lengths are Angstrom.

#include <string>
#include <vector>

namespace klasp {

// Lattice vectors are the columns of `matrix` (cart = matrix * frac), as in
// cell_params_to_matrix. Angles in degrees.
struct CifCell {
    double params[6] = {0, 0, 0, 0, 0, 0};  // a, b, c Angstrom; alpha, beta, gamma degrees
    double matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double inverse[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double volume = 0.0;
};

// One whole (unwrapped) molecule from the cell, in Cartesian Angstrom.
struct CifMolecule {
    std::vector<double> positions;   // [n*3] Cartesian Angstrom
    std::vector<int> numbers;        // [n] atomic numbers
    // Asymmetric-unit site and operator that produced each atom.
    std::vector<int> site;           // [n]
    std::vector<int> symop;          // [n]
    double centroid[3] = {0, 0, 0};  // unweighted, over every atom

    // The component bonds onto its own periodic image: a framework or network, not
    // a molecule.
    bool periodic = false;

    int size() const { return static_cast<int>(numbers.size()); }
};

struct CifStructure {
    std::string name;                // the data_ block name
    std::string source;              // the file it was read from
    CifCell cell;
    int space_group = 0;             // ITA number; 0 when gemmi could not name it
    std::string space_group_name;    // H-M, or "unknown"
    int n_symops = 0;                // operators used for the expansion, incl. identity
    int n_sites = 0;                 // asymmetric unit sites kept
    int n_atoms = 0;                 // atoms in the cell after expansion
    std::vector<CifMolecule> molecules;

    // Sites dropped for partial occupancy.
    int n_partial_occupancy = 0;

    // Symmetry images merged into an image of a different site. A large count on a
    // genuine asymmetric unit indicates disorder.
    int n_coincident = 0;
};

struct CifReadOptions {
    // Added to the sum of two covalent radii to decide a bond (Angstrom).
    double bond_tolerance = 0.4;
    // Sites below this occupancy are dropped as minor disorder components.
    double min_occupancy = 0.5;
    // Distance (Angstrom) below which symmetry images are merged as one atom.
    double merge_tolerance = 0.05;
    // Drop hydrogens at read time.
    bool skip_hydrogens = false;
};

// Read every crystal data block of `path`; blocks without a usable cell are
// skipped. Returns empty and sets `error` on parse failure or no crystal block.
std::vector<CifStructure> read_cif(const std::string& path,
                                   const CifReadOptions& opts = {},
                                   std::string* error = nullptr);

// Same, from text in memory; `name` is the reported source.
std::vector<CifStructure> read_cif_string(const std::string& text,
                                          const std::string& name = "<string>",
                                          const CifReadOptions& opts = {},
                                          std::string* error = nullptr);

// Perceive whole molecules from explicit cell contents (bond and unwrap only).
// `cell_params` is a, b, c in Angstrom and angles in degrees; `frac` is
// [n_atoms*3] fractional, row-major.
CifStructure structure_from_cell(const double cell_params[6], const double* frac,
                                 const int* numbers, int n_atoms,
                                 const CifReadOptions& opts = {});

}  // namespace klasp
