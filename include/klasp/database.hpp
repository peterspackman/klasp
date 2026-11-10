#pragma once

#include <string>
#include "klasp/cell.hpp"
#include "klasp/mat3.hpp"
#include "klasp/geometry.hpp"
#include "klasp/constants.hpp"
#include <array>
#include <vector>
#include <memory>

namespace klasp {

// Explicit-atom geometry for a non-rigid structure (e.g. PET-relaxed stage-3).
// Stored in the flexible_structure table, keyed to a parent `structure` row.
struct FlexibleStructure {
    int n_atoms = 0;
    double cell_params[6] = {0, 0, 0, 0, 0, 0};  // a,b,c (Angstrom); alpha,beta,gamma (deg)
    std::vector<int> Z;                          // [n_atoms] atomic numbers
    std::vector<double> frac;                    // [3*n_atoms] fractional coords, row-major
};

// Conversions between CrystalGeometry (Bohr, Cartesian) and the stored row
// (cell lengths in Angstrom, angles in degrees, fractional coordinates).
inline FlexibleStructure to_flexible(const CrystalGeometry& g) {
    FlexibleStructure fs;
    fs.n_atoms = g.n_atoms();
    fs.Z = g.atomic_numbers;

    double cp[6];
    matrix_to_cell_params(g.cell, cp);
    for (int k = 0; k < 6; ++k)
        fs.cell_params[k] = (k < 3) ? cp[k] * double(BOHR_TO_ANGSTROM) : cp[k];

    // frac = cell^-1 * cartesian. A singular cell leaves hinv zero, so the row
    // gets zero coordinates rather than NaNs.
    double hinv[3][3] = {};
    (void)mat3_inv(g.cell, hinv);
    fs.frac.resize(3ull * fs.n_atoms);
    for (int a = 0; a < fs.n_atoms; ++a) {
        const auto& p = g.positions[a];
        for (int r = 0; r < 3; ++r)
            fs.frac[3*a+r] = hinv[r][0]*p[0] + hinv[r][1]*p[1] + hinv[r][2]*p[2];
    }
    return fs;
}

// The inverse, up to orientation: the geometry comes back in the canonical
// orientation of the stored cell parameters.
inline CrystalGeometry from_flexible(const FlexibleStructure& fs) {
    CrystalGeometry g;
    g.atomic_numbers = fs.Z;

    RealType cp[6];
    for (int k = 0; k < 3; ++k) cp[k] = RealType(fs.cell_params[k] * ANGSTROM_TO_BOHR);
    for (int k = 3; k < 6; ++k) cp[k] = RealType(fs.cell_params[k]);
    RealType cm[3][3];
    cell_params_to_matrix(cp, cm);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) g.cell[r][c] = double(cm[r][c]);

    g.positions.resize(fs.n_atoms);
    for (int a = 0; a < fs.n_atoms; ++a) {
        const double* f = &fs.frac[3 * a];
        for (int r = 0; r < 3; ++r)
            g.positions[a][r] = g.cell[r][0]*f[0] + g.cell[r][1]*f[1] + g.cell[r][2]*f[2];
    }
    return g;
}

// SQLite store for a CSP run: one row per (seed, space_group, stage), plus
// descriptors, relaxed geometry and a key/value meta table.
class CSPDatabase {
public:
    CSPDatabase();
    ~CSPDatabase();

    // Open or create database at path. Enables WAL mode for concurrent reads.
    void open(const std::string& path);

    // Insert from a StructureResult (defined in dedup.hpp) and its unique flag.
    void insert_result(const void* result_ptr, bool unique, int duplicate_of = -1,
                       int duplicate_of_sg = -1);

    // Make a unique structure a duplicate of (new_seed, new_sg), and re-point
    // rows that were duplicates of it, so duplicate_of always names a representative.
    void demote_to_duplicate(int old_seed, int space_group, int stage,
                             int new_seed, int new_sg);

    // Set a row's dedup verdict: a representative when duplicate_of is negative,
    // otherwise a duplicate of (duplicate_of, duplicate_of_sg).
    void set_verdict(int seed, int space_group, int stage, int duplicate_of,
                     int duplicate_of_sg);

    // Flush pending inserts (commits current transaction, starts new one).
    void flush();

    // Checkpoint the write-ahead log into the database file. TRUNCATE when
    // truncate is set, PASSIVE otherwise.
    void checkpoint(bool truncate = false);

    void set_meta(const std::string& key, const std::string& value);
    std::string get_meta(const std::string& key);

    // Explicit-atom geometry blob: [int32 atomic number] * n_atoms, then
    // [float64 fractional] * 3n. Cell parameters are separate columns.
    static std::vector<char> encode_flexible_atoms(const FlexibleStructure& fs);
    // False, leaving `out` untouched, if `bytes` does not match n_atoms.
    static bool decode_flexible_atoms(int n_atoms, const void* data, std::size_t bytes,
                                      FlexibleStructure& out);

    void insert_flexible(int seed, int space_group, int stage,
                         const FlexibleStructure& fs);
    // Returns false if no flexible row exists for (seed, space_group, stage).
    bool read_flexible(int seed, int space_group, int stage,
                       FlexibleStructure& out);

    void insert_amd(int seed, int space_group, int stage,
                    const std::vector<double>& amd);

    // Store the X-ray powder pattern (float32 blob) under descriptor name 'powder'.
    void insert_powder(int seed, int space_group, int stage,
                       const float* pattern, int n_bins);

    // Store PDD under descriptor name 'pdd'. Blob layout: int32 n_rows, int32 k,
    // then n_rows*k float64 distances (Angstrom), row-major (row = asym atom,
    // col = neighbour). Row weights are uniform (1/n_rows).
    void insert_pdd(int seed, int space_group, int stage, const double* pdd,
                    int n_rows, int k);

    int count_valid(int space_group, int min_stage);

    // Rows at exactly `stage`, i.e. structures, unlike count_valid's stage >= min_stage.
    int count_at_stage(int space_group, int stage);

    // Rows at `stage` that are unique representatives.
    int count_unique_at_stage(int space_group, int stage);

    // Representatives at `from_stage` with no row at `missing_stage`, over every
    // space group. `klasp rerank --top-up` fills these in.
    int count_unique_missing_stage(int from_stage, int missing_stage);

    void close();

private:
    void insert_descriptor(int seed, int space_group, int stage,
                           const char* name, const std::vector<char>& blob);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace klasp
