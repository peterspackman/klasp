#pragma once

#include "klasp/csp_cif.hpp"
#include "klasp/dedup.hpp"
#include "klasp/geometry.hpp"

#include <array>
#include <string>
#include <vector>

namespace klasp {

struct BasisConfig;  // include/klasp/io.hpp

// Load the CSP basis for a DB either from an explicit file or from the DB's
// embedded `basis_json` meta value (host-only: no Kokkos). Throws on failure.
BasisConfig load_csp_basis(const std::string& basis_path_opt,
                           const std::string& db_path);

// Read a CSP landscape from a database as rigid-body rows (one per unique
// seed+space_group). `stage` selects the source stage (0 = MAX(stage) present).
// Filters mirror `extract`. Rows come back sorted by energy ascending.
// `only_missing_stage` > 0 keeps just the rows with no row of their own at that
// stage.
std::vector<StructureResult> read_landscape(const std::string& db_path, int stage,
                                            int filter_sg, double max_energy,
                                            bool unique_only, bool exclude_catastrophes,
                                            double catastrophe_threshold,
                                            bool exclude_warned = true,
                                            int only_missing_stage = 0);

// Group a landscape by packing similarity (COMPACK): compare the `top` lowest
// energy rows pairwise and report how many distinct packings they hold.
// Tolerances and reported RMSDs are in Angstrom.
struct CompackRunOptions {
    std::string basis_path;      // empty: read the basis the database recorded
    int stage = 0;               // 0 = highest present
    int top = 50;                // rows to compare, lowest energy first
    int n_neighbours = 15;
    double tolerance = 1.0;      // Angstrom
    double radius = 14.0;        // Angstrom
    bool heavy_only = true;
    bool allow_inversion = true;
    bool matrix = false;         // also print the full n-matched matrix
    bool progress = false;       // draw a progress bar on stderr

    // How near an atom must land on a same-element atom for an operation to
    // count as one of the molecule's own symmetries. Too loose invents symmetry
    // (acetic acid's C-OH vs C=O differ by 0.083 A); too tight loses it in
    // distorted rings. Compare against --all-atoms when results look over-merged.
    double self_symmetry_tolerance = 0.05;  // Angstrom

    // --- reading structures (CIF files, and stages that store explicit atoms) ---
    // Added to the sum of two covalent radii to decide a bond.
    double bond_tolerance = 0.4;            // Angstrom
    // Symmetry images closer than this are one atom on a special position.
    double merge_tolerance = 0.05;          // Angstrom
    // Sites below this occupancy are minor components of a disorder model.
    double min_occupancy = 0.5;
    // Residual above which a molecule is called a different compound rather than
    // a poor fit. A flexible molecule in two conformations still fits.
    double max_fit_rmsd = 1.0;              // Angstrom
};
void run_compack(const std::string& db_path, const CompackRunOptions& opts);

// Which of these CIFs are the same packing? The molecule is taken from the
// first data block and every other structure is fitted to it; a structure of a
// different compound is reported and skipped.
void run_compack_cifs(const std::vector<std::string>& paths, const CompackRunOptions& opts);

// Expand a rigid-body CSP structure (per-molecule COM translation + angle-axis
// rotation, space group `sg`) into the full unit cell of explicit atoms. Atom
// order is (op, molecule, site) with op 0 = identity, as in PetSymmetry for
// Z'=1. Cell in Bohr, columns = vectors. Fills both `elements` and
// `atomic_numbers`.
CrystalGeometry expand_rigid_structure(const BasisConfig& basis, int sg,
                                       const double cell_params[6],
                                       const double* translation,
                                       const double* rotation, int n_mols);

// Which database rows to consider; shared by extract and prune.
struct RowFilter {
    bool   keep_catastrophes = false;
    bool   keep_warned = false;
    double catastrophe_threshold = -1000.0;
    double max_energy = 1e30;
    // 0 = the highest stage present, but only when it covers the whole
    // landscape; a partial stage has to be named.
    int    stage = 0;
    int    top_n = 0;                   // 0 = all
};

// Extract structures from CSP database(s) to CIF or JSON files.
struct ExtractOptions {
    std::string db_path;
    std::string basis_path;             // optional: the DB usually embeds it
    std::string format = "cif";
    std::string output_dir = ".";
    RowFilter filter;

    int  filter_sg = 0;                 // 0 = every space group
    bool unique_only = false;
    bool niggli = false;
    bool check_energy = false;

    // Also write one {seed}.cpi Sietronics powder pattern per structure, from
    // the pattern the run stored as the `powder` descriptor.
    bool   dump_powder = false;
    double powder_two_theta_min = 5.0;
    double powder_two_theta_max = 50.0;

    // Compound name for the CIF block keys; empty takes what the run recorded,
    // then the database's own name.
    std::string name_override;
};

struct ExtractResult {
    // False when the command declined (no databases, or no basis); it has
    // printed why.
    bool ok = false;
    int written = 0;              // structures written, in whichever format
    long long rows_read = 0;      // rows the filters were applied to
    int powder_files = 0;         // .cpi files, when --dump-powder asked
    int energy_disagreements = 0; // --check-energy rows outside tolerance
    std::string output_path;      // the CIF/JSON written, or the directory
};

ExtractResult cmd_extract(const ExtractOptions& o);

// The optimisation steps and clustering rule a run recorded in its config_json,
// for the CSP CIF metadata. Empty when the database recorded no settings.
CspCifMetadata csp_metadata_for_run(const std::string& db_path, int read_stage);

// Merge rank databases and deduplicate using the configured dedup pipeline.
// Structures missing a required descriptor are kept as unique.
struct PruneOptions {
    // prune compares across space groups unless told not to (--no-cross-sg).
    PruneOptions() { dedup.cross_sg = true; }

    std::string db_path;
    std::string basis_path;
    std::string output_dir = "pruned";
    DedupConfig dedup;
    RowFilter filter;

    // Proceed even when most structures have no descriptor to compare on, and
    // so are reported unique by default.
    bool allow_missing_descriptors = false;
    std::string name_override;
};

struct PruneResult {
    bool ok = false;
    int considered = 0;           // rows that passed the filters
    int unique = 0;               // representatives after deduplication
    std::string cif_path;
    std::string csv_path;
};

PruneResult cmd_prune(const PruneOptions& o);

// Reclaim space in a finished database. VACUUM recovers free pages;
// drop_descriptors removes the derived AMD/powder descriptors, which prune
// needs. restore recomputes them from the stored geometry. Dropping is refused
// when the database has no basis. dry_run reports and changes nothing.
struct CompactResult {
    bool ok = false;
    int files = 0;
    double mb_before = 0;
    double mb_after = 0;
    bool dry_run = false;
};

struct CompactOptions {
    std::string db_path;
    std::string basis_path;       // empty: the basis the database recorded
    bool drop_descriptors = false;
    bool dry_run = false;
    bool restore = false;
};

CompactResult cmd_compact(const CompactOptions& o);

// Deduplicate the stage-3 structures `klasp rerank` wrote: recompute and store
// each row's AMD, powder pattern and folded PDD from its relaxed geometry, then
// deduplicate in ascending energy with the run's recorded dedup settings.
struct FinalStageDedup {
    int rows = 0;       // stage-3 rows with relaxed geometry
    int unique = 0;
};
FinalStageDedup dedup_final_stage(const std::string& db_path);

// Print an ASCII summary (per-SG table, energy/density histograms, top-N)
// of a finished CSP database.
struct SummaryOptions {
    std::string db_path;
    double catastrophe_threshold = -1000.0;   // kJ/mol
    int    top_n = 10;
    int    n_bins = 20;
    // Structures above this energy are left out of the histograms, though still
    // counted in the per-SG table. 1e29 or more clips at the 95th percentile.
    double max_energy_clip = 1e30;
    int    stage = 0;                          // 0 = the highest present
};

struct SummaryResult {
    bool ok = false;
    long long structures = 0;
    int space_groups = 0;
    // The landscape's global minimum, kJ/mol; meaningless when structures == 0.
    double min_energy = 0;
};

SummaryResult cmd_summary(const SummaryOptions& o);

// How much of a landscape has been seen, and how far the search may be from
// the global minimum.
//
// Basin count and coverage from Boender & Rinnooy Kan:
//     W_hat = w (N - 1) / (N - w - 2),   coverage = (N - w - 1) / (N - 1)
// with stopping rule W_hat < w + 1/2; Chao1 and Good-Turing are printed beside
// them. Each is computed within energy windows above the minimum, since over
// the whole landscape shallow minima dominate.
//
// The endpoint E* is fitted assuming the count of minima below E goes as
// (E - E*)^alpha, alongside Quenouille's jackknife estimates and a Gumbel
// (no finite endpoint) alternative; if the Gumbel fit wins, ignore the endpoint.
//
// `duplicate_of` chains are followed to their root (databases before schema v4
// may chain). energy_lo/energy_hi drop basins outside that range before
// counting, and the number dropped is reported.
struct ConvergedResult {
    bool ok = false;
    long long searches = 0;   // N, over every window
    long long basins = 0;     // w, over every window
    double min_energy = 0;    // lowest energy found, kJ/mol per molecule
    // Tightest window (kJ/mol above the minimum) that satisfies the stopping
    // rule, or -1 if none of them does.
    double converged_window = -1;
    // Point estimate of the true minimum and its bootstrap interval; only
    // meaningful when endpoint_ok.
    bool endpoint_ok = false;
    double endpoint = 0, endpoint_lo = 0, endpoint_hi = 0;
};

struct ConvergedOptions {
    std::string db_path;
    // Windows above the minimum to report, kJ/mol per molecule. The whole
    // landscape is always reported too.
    std::vector<double> windows{1, 2, 5, 10, 20};
    int tail_k = 100;               // lowest N minima the endpoint is fitted to; 0 = 5%
    int n_bootstrap = 400;          // resamples for the endpoint interval; 0 skips it
    double energy_lo = -1e4;        // kJ/mol per molecule
    double energy_hi = 1e4;
    unsigned bootstrap_seed = 0;
    int stage = 0;                  // 0 = the highest present
};

ConvergedResult cmd_converged(const ConvergedOptions& o);

}  // namespace klasp
