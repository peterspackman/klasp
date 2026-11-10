// Every command that runs in one process: what it registers, and what it runs.
//
// Options bind to the fields of the library's options structs; --help defaults
// come from capture_default_str(), not default_val().

#include "commands.hpp"

#include "cli_options.hpp"
#include "klasp/exit_code.hpp"

namespace klasp::cli {

namespace {

const char* const DATABASE_HELP = "Database file (.db) or directory containing csp_rank*.db";
const char* const BASIS_HELP = "Basis JSON (default: the one the database recorded)";
const char* const STAGE_HELP = "Optimisation stage to read (0 = the highest present)";
// extract and prune refuse a partial highest stage rather than read it silently.
const char* const PARTIAL_STAGE_HELP =
    "Optimisation stage to read (0 = the highest present). Refuses when the "
    "highest stage covers fewer structures than another -- which is what a "
    "top-N `klasp rerank` leaves behind.";

}  // namespace

// --- energy ------------------------------------------------------------------
// Evaluate and relax structures.
EnergyCommand::EnergyCommand(CLI::App& parent)
    : LocalCommand(parent.add_subcommand("energy", energy_description())) {
    add_energy_options(app_, options_, inputs_);
}

int EnergyCommand::run() {
    cmd_energy(inputs_, options_);
    return +ExitCode::Ok;
}

// --- fdcheck -----------------------------------------------------------------
FdcheckCommand::FdcheckCommand(CLI::App& parent)
    : LocalCommand(parent.add_subcommand("fdcheck",
          "Finite difference gradient check (exits non-zero if any DOF disagrees)")) {
    app_->add_option("input,--input", options_.input, "Crystal structure JSON")->required();
    app_->add_option("--abs-tol", options_.abs_tol,
        "Absolute floor on |analytic - FD|, for DOFs whose gradient is near zero")
        ->capture_default_str();
    app_->add_option("--rel-tol", options_.rel_tol,
        "Relative bound on |analytic - FD| against the larger of the two")
        ->capture_default_str();
    app_->add_flag("--no-taper", options_.no_taper, "Disable energy tapering");
    app_->add_option("--max-order", options_.max_order,
        "Override multipole rank, 0 to 4. -1 uses whatever rank the input carries.")
        ->check(CLI::Range(-1, 4))
        ->capture_default_str();
    add_ewald_accuracy_option(app_, options_.ewald_accuracy);
}

// Exits non-zero when any DOF fails the check.
int FdcheckCommand::run() {
    return cmd_fdcheck(options_) == 0 ? +ExitCode::Ok : +ExitCode::CheckFailed;
}

#ifdef KLASP_HAS_PET
// --- rerank ------------------------------------------------------------------
// Re-relax stored candidates and write stage-3 rows to the database.
RerankCommand::RerankCommand(CLI::App& parent)
    : LocalCommand(parent.add_subcommand("rerank",
          "Relax the lowest structures of a landscape with PET and store them as stage 3 "
          "(modifies the database)")) {
    auto& o = options_;
    app_->add_option("database", o.db_path, "CSP database")->required();
    app_->add_option("--basis", o.basis_path, BASIS_HELP);
    add_pet_model_options(app_, o.model, o.model_json, o.model_weights);
    app_->add_option("-n,--top", o.top,
        "Relax the N lowest by force-field energy (0 = all)")->capture_default_str();
    app_->add_flag("--top-up", o.top_up,
        "Relax only the unique structures that have no stage-3 energy yet. A csp "
        "run with the stage-3 dedup gate on relaxes one candidate per basin, and "
        "the member it picks is not always the one that ends up flagged unique; "
        "this fills in the rest so every row is ranked the same way. Combine with "
        "--top 0 to take all of them.");
    app_->add_option("--from-stage", o.from_stage,
        "Source stage to read geometry from (0 = max)")->capture_default_str();
    add_switch(app_, "symmetry", o.symmetry,
        "Relax with the space group held fixed rather than freely.");
    app_->add_option("--force-tol", o.force_tol, "Force tolerance (Ha/Bohr)")
        ->capture_default_str();
    app_->add_option("--stress-tol", o.stress_tol, "Stress tolerance (Ha/Bohr^3)")
        ->capture_default_str();
    app_->add_option("--iterations", o.max_iter, "Max L-BFGS iterations")
        ->capture_default_str();
    app_->add_option("--coarse", o.coarse_factor,
        "Coarse-to-fine: a loose pass at (tol x this) over all N, then a tight "
        "pass on the top half (e.g. 10; 0 = a single tight pass)")
        ->capture_default_str();
    app_->add_option("--min-contact", o.min_contact,
        "Drop candidates with any interatomic contact below this (Angstrom) "
        "before relaxing, which rejects force-field catastrophes a weak-force "
        "model cannot relax (0 = off)")
        ->capture_default_str();
}

int RerankCommand::run() {
    cmd_rerank(options_);
    return +ExitCode::Ok;
}
#endif

// --- extract -----------------------------------------------------------------
ExtractCommand::ExtractCommand(CLI::App& parent)
    : LocalCommand(parent.add_subcommand("extract", "Extract structures from CSP database(s)")) {
    auto& o = options_;
    app_->add_option("database", o.db_path, DATABASE_HELP)->required();
    app_->add_option("--basis", o.basis_path, BASIS_HELP);
    app_->add_option("--format,-f", o.format, "Output format: cif, json, or chemiscope")
        ->capture_default_str()
        ->check(CLI::IsMember({"cif", "json", "chemiscope"}));
    app_->add_option("--sg", o.filter_sg, "Filter by space group (0 = all)")
        ->capture_default_str();
    app_->add_option("--top,-n", o.filter.top_n, "Extract only top N by energy (0 = all)")
        ->capture_default_str();
    app_->add_option("--max-energy", o.filter.max_energy,
        "Maximum energy cutoff (kJ/mol); when --unique is set and this is unspecified, "
        "defaults to 0");
    app_->add_flag("--unique", o.unique_only,
        "Only extract unique structures (not duplicates). Implies --max-energy=0 unless "
        "overridden.");
    add_structure_filter_options(app_, o.filter);
    app_->add_flag("--niggli", o.niggli,
        "Niggli-reduce the cell before writing, so a collapsed basis becomes the short "
        "one describing the same lattice. Applies to P1 and P-1 only; other space "
        "groups are written unchanged.");
    app_->add_flag("--check-energy", o.check_energy,
        "Re-evaluate each structure's energy from its geometry and report any that "
        "disagree with the value the run stored.");
    app_->add_option("-o,--output-dir", o.output_dir, "Output directory")
        ->capture_default_str();
    app_->add_option("--name", o.name_override,
        "Compound name for the CIF block keys, overriding what the run recorded");
    app_->add_flag("--dump-powder", o.dump_powder,
        "Also write per-structure .cpi (Sietronics) powder pattern files stored as the "
        "'powder' descriptor");
    app_->add_option("--powder-2theta-min", o.powder_two_theta_min,
        "Powder pattern 2θ min (degrees) used when writing .cpi headers (must match the "
        "run that produced the pattern)")
        ->capture_default_str();
    app_->add_option("--powder-2theta-max", o.powder_two_theta_max,
        "Powder pattern 2θ max (degrees) for .cpi header")
        ->capture_default_str();
    app_->add_option("--stage", o.filter.stage, PARTIAL_STAGE_HELP)->capture_default_str();
}

int ExtractCommand::run() {
    // --unique without an explicit --max-energy keeps bound structures only (E < 0).
    if (options_.unique_only && app_->get_option("--max-energy")->count() == 0)
        options_.filter.max_energy = 0.0;
    cmd_extract(options_);
    return +ExitCode::Ok;
}

// --- prune -------------------------------------------------------------------
PruneCommand::PruneCommand(CLI::App& parent)
    : LocalCommand(parent.add_subcommand("prune",
          "Merge and deduplicate CSP databases (cross-SG)")) {
    auto& o = options_;
    app_->add_option("database", o.db_path, DATABASE_HELP)->required();
    app_->add_option("--basis", o.basis_path, BASIS_HELP);
    app_->add_option("-o,--output-dir", o.output_dir, "Output directory")
        ->capture_default_str();
    app_->add_option("--name", o.name_override,
        "Compound name for the CIF block keys, overriding what the run recorded");
    // Same registration as csp, so offline and online dedup agree. PruneOptions
    // defaults cross_sg on.
    add_dedup_options(app_, o.dedup);
    app_->add_option("--top,-n", o.filter.top_n,
        "Output only top N unique structures (0 = all)")
        ->capture_default_str();
    app_->add_option("--max-energy", o.filter.max_energy, "Maximum energy cutoff (kJ/mol)");
    app_->add_flag("--allow-missing-descriptors", o.allow_missing_descriptors,
        "Proceed even when more than 5% of structures have no descriptor to compare "
        "on. Those are reported unique because nothing could be checked, not because "
        "anything distinguished them.");
    app_->add_option("--stage", o.filter.stage, PARTIAL_STAGE_HELP)->capture_default_str();
    add_structure_filter_options(app_, o.filter);
}

int PruneCommand::run() {
    cmd_prune(options_);
    return +ExitCode::Ok;
}

// --- compack -----------------------------------------------------------------
CompackCommand::CompackCommand(CLI::App& parent)
    : LocalCommand(parent.add_subcommand("compack",
          "Group structures by packing similarity (overlay of a 15-molecule cluster)")) {
    auto& o = options_;
    app_->add_option("input", inputs_,
        "A landscape database (.db), or any number of CIF files. CIFs must hold the "
        "same compound: the molecule is taken from the first and every other structure "
        "is fitted to it.")
        ->required();
    app_->add_option("--basis", o.basis_path, BASIS_HELP);
    app_->add_option("-n,--top", o.top, "Compare this many rows, lowest energy first")
        ->capture_default_str();
    app_->add_option("--stage", o.stage, STAGE_HELP)->capture_default_str();
    app_->add_option("--neighbours", o.n_neighbours,
        "Cluster size: molecules around the central one. COMPACK and Mercury use 15.")
        ->capture_default_str();
    app_->add_option("--tolerance", o.tolerance,
        "Per-molecule RMSD below which a neighbour counts as matched, Angstrom")
        ->capture_default_str();
    app_->add_option("--radius", o.radius, "Neighbour search radius, Angstrom")
        ->capture_default_str();
    app_->add_flag("!--all-atoms", o.heavy_only,
        "Include hydrogens. Off by default: their positions are the least reliable part "
        "of a structure and one rotated methyl would spoil an otherwise perfect match.");
    add_switch(app_, "inversion", o.allow_inversion,
        "Also try the overlay through an inversion. Right for a centrosymmetric "
        "crystal, whose mirrored cluster is its own, and wrong for a chiral one.");
    app_->add_flag("--matrix", o.matrix, "Also print the full matched-neighbour matrix");
    app_->add_flag("--progress", o.progress,
        "Draw a progress bar on stderr. The comparison is quadratic in the number of "
        "structures, so a large batch is worth watching.");
    app_->add_option("--self-symmetry-tolerance", o.self_symmetry_tolerance,
        "How near an atom must land on a same-element atom for an operation to count as "
        "one of the molecule's own symmetries, Angstrom. No value suits every molecule: "
        "too loose invents one (acetic acid's two C-O bonds differ by 0.083 A, so 0.1 "
        "accepts a C2 swapping the oxygens and over-merges), too tight loses a real one "
        "(a packing-distorted benzene stops satisfying its own D6h). Compare against "
        "--all-atoms when a result looks over-merged.")
        ->capture_default_str();
    app_->add_option("--max-fit-rmsd", o.max_fit_rmsd,
        "Residual above which a molecule is called a different compound rather than a "
        "poor fit, Angstrom. Raise it to compare conformers.")
        ->capture_default_str();
    app_->add_option("--bond-tolerance", o.bond_tolerance,
        "Added to the sum of two covalent radii to decide a bond when reading "
        "structures, Angstrom")
        ->capture_default_str();
    app_->add_option("--merge-tolerance", o.merge_tolerance,
        "Symmetry images closer together than this are one atom on a special position, "
        "Angstrom")
        ->capture_default_str();
    app_->add_option("--min-occupancy", o.min_occupancy,
        "Sites below this occupancy are treated as the minor component of a disorder "
        "model and left out")
        ->capture_default_str();
}

int CompackCommand::run() {
    // One .db is a landscape; anything else is CIF files.
    const bool is_db = inputs_.size() == 1 && inputs_[0].size() > 3 &&
                       inputs_[0].compare(inputs_[0].size() - 3, 3, ".db") == 0;
    if (is_db)
        run_compack(inputs_[0], options_);
    else
        run_compack_cifs(inputs_, options_);
    return +ExitCode::Ok;
}

// --- compact -----------------------------------------------------------------
CompactCommand::CompactCommand(CLI::App& parent)
    : LocalCommand(parent.add_subcommand("compact",
          "Reclaim space in a finished run's database")) {
    auto& o = options_;
    app_->add_option("database", o.db_path, DATABASE_HELP)->required();
    app_->add_flag("--drop-descriptors", o.drop_descriptors,
        "Also delete the AMD/powder/PDD descriptors, which is most of the file -- a "
        "finished run goes to a few percent of its size. They are derived from the "
        "stored geometry, so nothing is lost that --restore cannot compute again, but "
        "prune needs them: it will refuse on a stripped database. For archiving a run "
        "you have finished with.");
    app_->add_flag("--dry-run", o.dry_run,
        "Report what is there and what each step would reclaim; write nothing");
    app_->add_flag("--restore", o.restore,
        "Recompute the descriptors from the geometry the database still holds, undoing "
        "--drop-descriptors. Stage-3 rows come from their explicit coordinates, rigid "
        "rows from expanding the DOFs.");
    app_->add_option("--basis", o.basis_path, BASIS_HELP);
}

int CompactCommand::run() {
    cmd_compact(options_);
    return +ExitCode::Ok;
}

// --- summary -----------------------------------------------------------------
SummaryCommand::SummaryCommand(CLI::App& parent)
    : LocalCommand(parent.add_subcommand("summary",
          "Print an ASCII summary of a run (per-SG table, histograms, top-N)")) {
    auto& o = options_;
    app_->add_option("database", o.db_path, DATABASE_HELP)->required();
    app_->add_option("--catastrophe-threshold", o.catastrophe_threshold,
        "Energy below this is catastrophic (kJ/mol)")
        ->capture_default_str();
    app_->add_option("-n,--top", o.top_n, "Show top N lowest-energy unique structures")
        ->capture_default_str();
    app_->add_option("--bins", o.n_bins, "Number of histogram bins for energy + density")
        ->capture_default_str();
    app_->add_option("--max-energy", o.max_energy_clip,
        "Clip energy histogram above this kJ/mol (default: auto p95)");
    app_->add_option("--stage", o.stage, STAGE_HELP)->capture_default_str();
}

int SummaryCommand::run() {
    cmd_summary(options_);
    return +ExitCode::Ok;
}

// --- converged ---------------------------------------------------------------
ConvergedCommand::ConvergedCommand(CLI::App& parent)
    : LocalCommand(parent.add_subcommand("converged",
          "Estimate how completely a landscape has been sampled, per energy window")) {
    auto& o = options_;
    app_->add_option("database", o.db_path, DATABASE_HELP)->required();
    app_->add_option("--window", o.windows,
        "Energy windows above the minimum to report, kJ/mol per molecule. The whole "
        "landscape is always reported too.")
        ->delimiter(',')
        ->capture_default_str();
    app_->add_option("--tail", o.tail_k,
        "Lowest N minima to fit the endpoint to. The fit is asymptotic near the "
        "endpoint, so a small tail is the honest regime and a large one is the precise "
        "but biased one; the printed sweep shows how much the answer moves. 0 uses 5% "
        "of the minima.")
        ->capture_default_str();
    app_->add_option("--bootstrap", o.n_bootstrap,
        "Bootstrap resamples for the endpoint interval, 0 to skip")
        ->capture_default_str();
    app_->add_option("--min-energy", o.energy_lo,
        "Discard basins below this, kJ/mol per molecule. A collapsed structure with a "
        "non-physical energy can absorb a large share of the landscape.")
        ->capture_default_str();
    app_->add_option("--max-energy", o.energy_hi,
        "Discard basins above this, kJ/mol per molecule")
        ->capture_default_str();
    app_->add_option("--seed", o.bootstrap_seed,
        "Seed for the bootstrap, so an interval reproduces")
        ->capture_default_str();
    app_->add_option("--stage", o.stage, STAGE_HELP)->capture_default_str();
}

int ConvergedCommand::run() {
    cmd_converged(options_);
    return +ExitCode::Ok;
}

// --- wyckoff -----------------------------------------------------------------
WyckoffCommand::WyckoffCommand(CLI::App& parent)
    : LocalCommand(parent.add_subcommand("wyckoff",
          "Report the special positions a molecule could occupy in each space group")) {
    auto& o = options_;
    app_->add_option("basis", o.basis_path,
        "Basis JSON describing the molecule")->required();
    app_->add_option("--sg", o.space_groups,
        "Space groups to report, as a list or ranges: 194 or 1-230 or 156,183,194")
        ->capture_default_str();
    app_->add_flag("--all-sites", o.all_sites,
        "List every special position, including the ones this molecule is too "
        "unsymmetric to occupy");
    app_->add_option("--self-symmetry-tolerance", o.self_symmetry_tolerance,
        "How far an atom may sit from where an operation sends it and still count as "
        "mapped, in Angstrom. Loosening this invents symmetry the molecule does not "
        "have -- at 0.1 a C2 appears in acetic acid, because its C-OH and C=O differ by "
        "only 0.083 -- and a molecule put on a site it does not fit gives a structure "
        "that is simply wrong, so this is stricter than compack's use of the same idea.")
        ->capture_default_str();
}

int WyckoffCommand::run() {
    return cmd_wyckoff(options_);
}

// --- all of them -------------------------------------------------------------
Commands::Commands(CLI::App& app)
    : csp(app),
      energy(app),
      fdcheck(app),
      extract(app),
#ifdef KLASP_HAS_PET
      rerank(app),
#endif
      prune(app),
      compack(app),
      compact(app),
      summary(app),
      converged(app),
      wyckoff(app) {
}

LocalCommand* Commands::parsed_local() {
    LocalCommand* const local[] = {
        &energy, &fdcheck, &extract,
#ifdef KLASP_HAS_PET
        &rerank,
#endif
        &prune, &compack, &compact, &summary, &converged, &wyckoff,
    };
    for (LocalCommand* command : local)
        if (command->parsed()) return command;
    return nullptr;
}

}  // namespace klasp::cli
