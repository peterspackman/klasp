#include "klasp/log.hpp"
#include "klasp/db_tool.hpp"

#include "klasp/cif_packing.hpp"
#include "klasp/progress.hpp"
#include "klasp/cif_reader.hpp"
#include "klasp/compack.hpp"
#include "klasp/flex_descriptors.hpp"
#include "klasp/io.hpp"
#include "klasp/csp_cif.hpp"
#include "klasp/version.hpp"
#include "klasp/database.hpp"
#include "klasp/element.hpp"
#include "klasp/dedup.hpp"
#include "klasp/run_state.hpp"
#include "klasp/csp.hpp"         // MAX_INDEP
#include "klasp/descriptors.hpp" // AMD_K
#include "klasp/cell.hpp"
#include "klasp/rotation.hpp"
#include "klasp/sgdata.hpp"
#include "klasp/constants.hpp"
#include "klasp/config.hpp"
#include "klasp/db_open.hpp"
#include "klasp/db_schema.hpp"
#include "klasp/molecule.hpp"
#include "klasp/batch.hpp"

#include <atomic>
#include <mutex>
#include <limits>
#include <random>

#include <sqlite_modern_cpp.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <string>
#include <fstream>
#include <cmath>
#include <set>
#include <algorithm>
#include <filesystem>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>

namespace klasp {

// Allowed disagreement between a stored energy and a fresh single point on the
// same geometry, kJ/mol.
static constexpr double ENERGY_CHECK_TOL = 0.1;

// Host-only basis parser for extract/prune: fills just the fields write_cif and
// write_crystal_json need, with no Workspace or Kokkos Views, so it runs on
// nodes without a GPU.
static BasisConfig load_basis_host_only(const nlohmann::json& basis_json) {
    BasisConfig cfg;
    cfg.basis_json = basis_json;
    auto& mts = basis_json["molecule_types"];
    int n_mols = static_cast<int>(mts.size());
    cfg.n_mols = n_mols;
    cfg.mol_n_sites.resize(n_mols);
    cfg.mol_body_positions.resize(n_mols);
    cfg.mol_elements.resize(n_mols);

    int max_sites = 0;
    for (int m = 0; m < n_mols; ++m) {
        int ns = static_cast<int>(mts[m]["sites"].size());
        cfg.mol_n_sites[m] = ns;
        cfg.mol_body_positions[m].resize(ns);
        cfg.mol_elements[m].resize(ns);
        for (int j = 0; j < ns; ++j) {
            auto& site = mts[m]["sites"][j];
            auto pos = site["position"].get<std::vector<double>>();
            cfg.mol_body_positions[m][j] = {
                pos[0] * ANGSTROM_TO_BOHR,
                pos[1] * ANGSTROM_TO_BOHR,
                pos[2] * ANGSTROM_TO_BOHR};
            std::string el = site.value("element", site.value("type", "C"));
            if (el.size() > 1 && std::islower(el[1])) el = el.substr(0, 2);
            else el = el.substr(0, 1);
            cfg.mol_elements[m][j] = el;
        }
        max_sites = std::max(max_sites, ns);
    }
    cfg.n_sites = max_sites;

    // Same H-bond adjustment load_basis_json applies, per molecule, so extracted
    // H positions match klasp's.
    if (basis_json.contains("settings")) {
        const double h_ext = basis_json["settings"].value("h_bond_extend", 0.0);
        if (h_ext != 0.0)
            for (int m = 0; m < n_mols; ++m)
                apply_h_bond_extend(cfg.mol_body_positions[m], cfg.mol_elements[m],
                                    h_ext * ANGSTROM_TO_BOHR);
    }

    int total = n_mols * max_sites;
    cfg.body_positions.assign(total, {0.0, 0.0, 0.0});
    cfg.elements.assign(total, "X");
    for (int m = 0; m < n_mols; ++m) {
        int off = m * max_sites;
        for (int j = 0; j < cfg.mol_n_sites[m]; ++j) {
            cfg.body_positions[off + j] = cfg.mol_body_positions[m][j];
            cfg.elements[off + j] = cfg.mol_elements[m][j];
        }
    }
    return cfg;
}

static BasisConfig load_basis_host_only_from_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open basis file: " + path);
    nlohmann::json j; f >> j;
    auto& b = j.contains("basis") ? j["basis"] : j;
    return load_basis_host_only(b);
}

// The basis a database recorded, read the way that landscape was run. A basis
// without potentials.damping_range predates it and is read at range 1.
static nlohmann::json recorded_basis(const std::string& text) {
    nlohmann::json basis = nlohmann::json::parse(text);
    if (basis.contains("potentials") && basis["potentials"].is_object()
        && !basis["potentials"].contains("damping_range"))
        basis["potentials"]["damping_range"] = 1.0;
    return basis;
}

BasisConfig load_csp_basis(const std::string& basis_path_opt,
                           const std::string& db_path) {
    if (!basis_path_opt.empty()) return load_basis_host_only_from_file(basis_path_opt);
    CSPDatabase db;
    db.open(db_path);
    std::string basis_json_str = db.get_meta("basis_json");
    db.close();
    if (basis_json_str.empty())
        throw std::runtime_error(
            "No basis JSON embedded in database and no basis file given; "
            "pass a basis JSON explicitly");
    return load_basis_host_only(recorded_basis(basis_json_str));
}

// Turn choose_stage()'s decision (run_state.cpp) into a message or an error. A
// stage covering only part of the landscape (e.g. a rerank's top-K) must be
// named explicitly.
static int resolve_stage(const std::vector<std::string>& db_files, int requested,
                         const char* what) {
    const std::map<int, int> hist = stage_histogram(db_files);
    const StageChoice c = choose_stage(hist, requested);
    if (hist.empty()) return c.stage;

    std::string summary;
    for (const auto& [stage, n] : hist)
        summary += fmt::format("{}{}:{}", summary.empty() ? "" : ", ", stage, n);

    if (c.unknown)
        throw std::runtime_error(fmt::format(
            "{}: no rows at stage {} (rows per stage: {})", what, c.stage, summary));

    if (hist.size() > 1)
        logging::info("  Reading stage {} ({} rows; rows per stage: {})\n",
                   c.stage, c.rows, summary);

    if (c.silently_partial())
        throw std::runtime_error(fmt::format(
            "{}: stage {} holds {} rows but stage {} holds {} -- reading the highest "
            "stage would silently drop {} structures ({:.0f}% of the landscape). "
            "Pass --stage {} for the whole landscape, or --stage {} if the partial "
            "one is really what you want.",
            what, c.stage, c.rows, c.fullest_stage, c.fullest_rows,
            c.fullest_rows - c.rows, 100.0 * (1.0 - c.coverage()),
            c.fullest_stage, c.stage));
    if (!c.requested && c.slightly_short())
        logging::info("  ({} structure{} present at stage {} did not reach stage {}.)\n",
                   c.fullest_rows - c.rows, c.fullest_rows - c.rows == 1 ? "" : "s",
                   c.fullest_stage, c.stage);
    return c.stage;
}

// A stage-3 row's rigid columns are placeholders; only flexible_structure holds
// its geometry. Refuse to read such a stage without that table.
static void require_explicit_geometry(const std::vector<std::string>& db_files, int stage,
                                      const char* what) {
    if (stage < STAGE_FINAL) return;
    for (const auto& file : db_files) {
        sqlite::database db = open_database_readonly(file);
        int rows = 0, has_table = 0, flex = 0;
        db << "SELECT COUNT(*) FROM structure WHERE stage = ?" << stage >> rows;
        db << "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
              "AND name='flexible_structure'" >> has_table;
        if (has_table)
            db << "SELECT COUNT(*) FROM flexible_structure WHERE stage = ?" << stage >> flex;
        if (flex >= rows) continue;
        throw std::runtime_error(fmt::format(
            "{}: {} of the {} stage-{} structures in {} have no relaxed geometry{}. Stage {} "
            "is relaxed with every atom free, so the rigid-body columns of its rows are "
            "placeholders and cannot stand in for it. A copy made with `sqlite3 .dump "
            "structure` loses that table; to make a database smaller, run `klasp compact "
            "--drop-descriptors` on the original instead, which keeps the geometry.",
            what, rows - flex, rows, stage, file,
            has_table ? "" : " (the flexible_structure table is missing)", stage));
    }
}

// Recompute a stored structure's energy from its geometry, kJ/mol per formula
// unit, through the same JSON document and loader a file would use. `ws` is
// reused across structures.
static double recompute_energy(Workspace& ws, const BasisConfig& basis, int space_group,
                               const double cell_params[6], const double* translations,
                               const double* rot_vectors, int n_indep) {
    const nlohmann::json doc = build_crystal_json(basis, space_group, cell_params,
                                                  translations, rot_vectors, n_indep);
    MoleculeData mol{};
    const CrystalConfig cfg = load_structure_json(doc, mol, ws);
    const RealType e = batch_evaluate_energy(mol, ws, cfg.max_rank, cfg.taper_on,
                                             cfg.taper_off, cfg.buck_damping);
    const int z = std::max(1, mol.n_symops / std::max(1, mol.n_indep));
    return double(e) * AU_TO_KJ_PER_MOL / z;
}

std::vector<StructureResult> read_landscape(const std::string& db_path, int stage,
                                            int filter_sg, double max_energy,
                                            bool unique_only, bool exclude_catastrophes,
                                            double catastrophe_threshold,
                                            bool exclude_warned,
                                            int only_missing_stage) {
    std::vector<StructureResult> rows;
    sqlite::database db = open_database_readonly(db_path);
    const std::string stage_sel =
        stage > 0 ? std::to_string(stage) : "(SELECT MAX(stage) FROM structure)";
    // `stage` is selected, not copied from the argument, which is 0 for "highest".
    std::string query =
        "SELECT seed, space_group, stage, energy, density,"
        " a, b, c, alpha, beta, gamma, tx, ty, tz, rx, ry, rz, n_mols, dofs"
        " FROM structure WHERE stage = " + stage_sel;
    if (filter_sg > 0) query += fmt::format(" AND space_group = {}", filter_sg);
    if (max_energy < 1e29) query += fmt::format(" AND energy < {}", max_energy);
    if (exclude_catastrophes) query += fmt::format(" AND energy > {}", catastrophe_threshold);
    if (unique_only) query += " AND unique_structure = 1";
    // Correlated on the outer table by name, so the rest of the query keeps its
    // unqualified column references.
    if (only_missing_stage > 0)
        query += fmt::format(
            " AND NOT EXISTS (SELECT 1 FROM structure AS later WHERE later.stage = {}"
            " AND later.seed = structure.seed"
            " AND later.space_group = structure.space_group)", only_missing_stage);
    query += warned_filter(db, exclude_warned);

    db << query >> [&](int seed, int sg, int row_stage, double energy, double density,
                       double a, double b, double c, double al, double be, double ga,
                       double tx, double ty, double tz, double rx, double ry, double rz,
                       int n_mols, std::unique_ptr<std::string> dofs_json) {
        StructureResult r{};
        r.seed = seed; r.space_group = sg; r.stage = row_stage; r.energy = energy;
        r.density = density; r.n_mols = std::max(1, n_mols);
        r.cell_params[0]=a; r.cell_params[1]=b; r.cell_params[2]=c;
        r.cell_params[3]=al; r.cell_params[4]=be; r.cell_params[5]=ga;
        r.translation[0]=tx; r.translation[1]=ty; r.translation[2]=tz;
        r.rotation[0]=rx; r.rotation[1]=ry; r.rotation[2]=rz;
        if (r.n_mols > 1 && dofs_json) {
            auto jd = nlohmann::json::parse(*dofs_json);
            auto& jt = jd["t"]; auto& jr = jd["r"];
            for (int i = 0; i < r.n_mols * 3 && i < MAX_INDEP * 3; ++i) {
                r.translation[i] = jt[i].get<double>();
                r.rotation[i] = jr[i].get<double>();
            }
        }
        rows.push_back(r);
    };
    std::sort(rows.begin(), rows.end(),
              [](const StructureResult& a, const StructureResult& b) { return a.energy < b.energy; });
    return rows;
}

// The compound name for CIF block keys: the caller's, else the run's recorded
// name, else from the database path. Never empty.
static std::string resolve_structure_name(const std::vector<std::string>& db_files,
                                          const std::string& name_override) {
    if (!name_override.empty()) return name_override;
    if (!db_files.empty()) {
        CSPDatabase db;
        db.open(db_files[0]);
        const std::string recorded = db.get_meta("name");
        db.close();
        if (!recorded.empty()) return recorded;
        // csp.db in a run directory names the run, not the compound, so use the
        // directory.
        const std::filesystem::path p(db_files[0]);
        const std::string stem = p.stem().string();
        if (stem == "csp" && p.has_parent_path() && !p.parent_path().filename().empty())
            return p.parent_path().filename().string();
        return stem;
    }
    return "structure";
}

// Rigid-body -> full-cell explicit atoms (Bohr), in (op, molecule, site) order.
CrystalGeometry expand_rigid_structure(const BasisConfig& basis, int sg,
                                       const double cell_params[6],
                                       const double* tr, const double* rv,
                                       int n_mols) {
    CrystalGeometry out;
    auto& sg_info = get_space_group(sg);

    RealType cp_rt[6];
    for (int k = 0; k < 6; ++k) cp_rt[k] = static_cast<RealType>(cell_params[k]);
    RealType cell[3][3];
    cell_params_to_matrix(cp_rt, cell);
    // Zeroed: mat3_inv leaves its output untouched on a singular stored cell.
    RealType cell_inv[3][3] = {};
    if (!mat3_inv(cell, cell_inv))
        logging::warn("warning: a stored row has a degenerate cell; its expanded "
                      "geometry will be wrong\n");
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out.cell[i][j] = static_cast<double>(cell[i][j]);

    for (int op_idx = 0; op_idx < sg_info.n_symops; ++op_idx) {
        double S_frac[3][3], t_sym[3];
        get_symop(sg_info, op_idx, S_frac, t_sym);
        RealType Sfrac_rt[3][3];
        for (int k = 0; k < 3; ++k)
            for (int l = 0; l < 3; ++l) Sfrac_rt[k][l] = static_cast<RealType>(S_frac[k][l]);
        RealType S_cart[3][3];
        frac_to_cart_symop(cell, cell_inv, Sfrac_rt, S_cart);

        for (int m = 0; m < n_mols; ++m) {
            double frac_com[3];
            for (int k = 0; k < 3; ++k) {
                double f = S_frac[k][0]*tr[m*3+0] + S_frac[k][1]*tr[m*3+1]
                         + S_frac[k][2]*tr[m*3+2] + t_sym[k];
                frac_com[k] = f - std::floor(f);
            }
            double cart_com[3] = {0, 0, 0};
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l) cart_com[k] += cell[k][l] * frac_com[l];

            RealType aa[3] = {static_cast<RealType>(rv[m*3+0]),
                              static_cast<RealType>(rv[m*3+1]),
                              static_cast<RealType>(rv[m*3+2])};
            RealType R_mol[3][3];
            angle_axis_to_matrix(aa, R_mol);
            RealType R_tot[3][3];
            mat3_mul(S_cart, R_mol, R_tot);

            const int body_off = m * basis.n_sites;
            const int ns_m = (m < static_cast<int>(basis.mol_n_sites.size()))
                           ? basis.mol_n_sites[m] : basis.n_sites;
            for (int j = 0; j < ns_m; ++j) {
                double rotated[3] = {0, 0, 0};
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l)
                        rotated[k] += static_cast<double>(R_tot[k][l])
                                    * basis.body_positions[body_off+j][l];
                out.elements.push_back(basis.elements[body_off+j]);
                out.atomic_numbers.push_back(
                    element(basis.elements[body_off+j]).atomic_number);
                out.positions.push_back({rotated[0] + cart_com[0],
                                         rotated[1] + cart_com[1],
                                         rotated[2] + cart_com[2]});
            }
        }
    }
    return out;
}

// The optimisation history and clustering rule of the run that produced a
// landscape, from the `config_json` meta key. No recorded settings yields no
// steps, and no optimisation loop is written.
CspCifMetadata csp_metadata_for_run(const std::string& db_path, int read_stage) {
    CspCifMetadata meta;
    const std::string version = fmt::format("{}{}", GIT_HASH_SHORT, GIT_DIRTY ? "-dirty" : "");

    std::string cfg_str, stage3_model;
    try {
        CSPDatabase db;
        db.open(db_path);
        cfg_str = db.get_meta("config_json");
        stage3_model = db.get_meta("stage3_model");
        db.close();
    } catch (const std::exception&) { return meta; }
    if (cfg_str.empty()) return meta;

    nlohmann::json cfg;
    try { cfg = nlohmann::json::parse(cfg_str); }
    catch (const std::exception&) { return meta; }

    // operator[] would insert a null for a section an older run did not record.
    static const nlohmann::json kNone = nlohmann::json::object();
    const auto section = [&](const char* key) -> const nlohmann::json& {
        const auto it = cfg.find(key);
        return it != cfg.end() && it->is_object() ? *it : kNone;
    };
    const auto& opt = section("optimizer");
    const auto& pet = section("pet");
    const bool multipoles = opt.value("max_rank", 0) > 0;
    // Stage 3 is always the PET relaxation (in the run or by rerank).
    const bool final_is_pet = read_stage >= STAGE_FINAL;

    // Stage 1 (charge-only) and stage 2 are one force-field step.
    CspOptimisationStep ff;
    ff.method = "Force field";
    ff.force_field_name = "klasp";
    ff.force_field_description =
        multipoles
            ? "atom-atom exp-6 repulsion-dispersion with distributed multipole electrostatics"
            : "atom-atom exp-6 repulsion-dispersion with atomic point charges";
    ff.software = "klasp";
    ff.software_version = version;
    ff.final_step = !final_is_pet;
    meta.steps.push_back(std::move(ff));

    if (final_is_pet) {
        CspOptimisationStep ml;
        // Not one of the dictionary's examples; other values are permitted.
        ml.method = "Machine learning potential";
        std::string model = stage3_model;
        if (model.empty()) model = pet.value("model_json", std::string());
        ml.force_field_name = model.empty() ? "PET" : model;
        ml.force_field_description = "PET equivariant transformer potential";
        ml.software = "klasp";
        ml.software_version = version;
        ml.final_step = true;
        meta.steps.push_back(std::move(ml));
    }

    // Record the thresholds, so unique counts can be compared across landscapes.
    if (cfg.contains("dedup")) {
        const auto& d = section("dedup");
        if (d.value("use_powder", true))
            meta.clustering_method = fmt::format(
                "klasp online dedup: energy {:.1f} kJ/mol and density {:.2f} g/cm^3 bins, "
                "AMD L-inf {:.2f} A, powder cosine {:.2f}, powder CDTW {:.1f}",
                d.value("energy_bin", 5.0), d.value("density_bin", 0.05),
                d.value("amd_tol", 1.0), d.value("powder_cos_thresh", 0.8),
                d.value("powder_cdtw_thresh", 10.0));
        else
            meta.clustering_method = fmt::format(
                "klasp online dedup: energy {:.1f} kJ/mol and density {:.2f} g/cm^3 bins, "
                "AMD L-inf {:.2f} A",
                d.value("energy_bin", 5.0), d.value("density_bin", 0.05),
                d.value("amd_tol", 1.0));
    }
    return meta;
}

namespace {

// One open read-only connection per source database, for per-structure
// explicit-geometry lookups. Does not go through CSPDatabase, so nothing is
// migrated; a database without the table reads as having no explicit geometry.
class FlexibleReader {
public:
    // Fills `out` and returns true when this structure has an explicit
    // (non-rigid) geometry stored.
    bool read(const std::string& path, int seed, int sg, int stage,
              FlexibleStructure& out) {
        Source& src = source(path);
        if (!src.has_table) return false;
        bool found = false;
        *src.db << kQuery << seed << sg << stage
            >> [&](int n_atoms, double a, double b, double c, double al, double be,
                   double ga, std::vector<char> blob) {
                   if (!CSPDatabase::decode_flexible_atoms(n_atoms, blob.data(),
                                                           blob.size(), out)) return;
                   out.cell_params[0] = a;  out.cell_params[1] = b;  out.cell_params[2] = c;
                   out.cell_params[3] = al; out.cell_params[4] = be; out.cell_params[5] = ga;
                   found = true;
               };
        return found;
    }

private:
    struct Source {
        std::unique_ptr<sqlite::database> db;
        bool has_table = false;
    };

    static constexpr const char* kQuery =
        "SELECT n_atoms, a, b, c, alpha, beta, gamma, atoms FROM flexible_structure"
        " WHERE seed = ? AND space_group = ? AND stage = ?";

    Source& source(const std::string& path) {
        auto it = sources_.find(path);
        if (it != sources_.end()) return it->second;
        Source src;
        src.db = std::make_unique<sqlite::database>(open_database_readonly(path));
        int n = 0;
        *src.db << "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
                   "AND name='flexible_structure'" >> n;
        src.has_table = n > 0;
        return sources_.emplace(path, std::move(src)).first->second;
    }

    std::map<std::string, Source, std::less<>> sources_;
};

}  // namespace

ExtractResult cmd_extract(const ExtractOptions& o) {
    ExtractResult result;
    // Filters arrive in the CLI's positive sense and are negated once here.
    const std::string& db_path = o.db_path;
    const std::string& basis_path_opt = o.basis_path;
    const std::string& format = o.format;
    const std::string& output_dir = o.output_dir;
    const int filter_sg = o.filter_sg;
    const int top_n = o.filter.top_n;
    const double max_energy = o.filter.max_energy;
    const bool unique_only = o.unique_only;
    const bool exclude_catastrophes = !o.filter.keep_catastrophes;
    const double catastrophe_threshold = o.filter.catastrophe_threshold;
    const bool exclude_warned = !o.filter.keep_warned;
    const bool niggli = o.niggli;
    const bool check_energy = o.check_energy;
    const bool dump_powder = o.dump_powder;
    const double powder_two_theta_min = o.powder_two_theta_min;
    const double powder_two_theta_max = o.powder_two_theta_max;
    const int stage = o.filter.stage;
    const std::string& name_override = o.name_override;
    std::vector<std::string> db_files;
    if (std::filesystem::is_directory(db_path)) {
        for (auto& entry : std::filesystem::directory_iterator(db_path)) {
            auto p = entry.path();
            if (p.extension() == ".db" && p.filename().string().find("csp") != std::string::npos)
                db_files.push_back(p.string());
        }
        std::sort(db_files.begin(), db_files.end());
    } else {
        db_files.push_back(db_path);
    }

    if (db_files.empty()) {
        logging::info("No database files found in {}\n", db_path);
        return result;
    }
    const std::string structure_name = resolve_structure_name(db_files, name_override);

    const int read_stage = resolve_stage(db_files, stage, "extract");
    require_explicit_geometry(db_files, read_stage, "extract");

    // Host-only: no Workspace, no Kokkos Views.
    BasisConfig basis;
    if (!basis_path_opt.empty()) {
        basis = load_basis_host_only_from_file(basis_path_opt);
    } else {
        CSPDatabase first_db;
        first_db.open(db_files[0]);
        std::string basis_json_str = first_db.get_meta("basis_json");
        first_db.close();
        if (basis_json_str.empty()) {
            logging::info("No basis JSON in database and no basis file specified.\n");
            logging::info("Use: klasp extract <db> <basis.json> or re-run CSP to embed basis.\n");
            return result;
        }
        basis = load_basis_host_only(recorded_basis(basis_json_str));
    }

    logging::info("Reading from {} database file{}\n", db_files.size(),
               db_files.size() > 1 ? "s" : "");

    struct Row {
        int seed, sg, n_mols, stage;
        double energy, density;
        double cp[6], tr[MAX_INDEP * 3], rv[MAX_INDEP * 3];
        std::string src_db;  // which DB this row came from (for flexible lookup)
    };
    std::vector<Row> all_rows;
    std::set<std::pair<int,int>> seen;

    for (auto& dbf : db_files) {
        sqlite::database db = open_database_readonly(dbf);
        std::string query = "SELECT seed, space_group, stage, energy, density,"
                            " a, b, c, alpha, beta, gamma, tx, ty, tz, rx, ry, rz,"
                            " n_mols, dofs"
                            " FROM structure WHERE stage = " + std::to_string(read_stage);
        if (filter_sg > 0) query += fmt::format(" AND space_group = {}", filter_sg);
        if (max_energy < 1e29) query += fmt::format(" AND energy < {}", max_energy);
        if (exclude_catastrophes) query += fmt::format(" AND energy > {}", catastrophe_threshold);
        if (unique_only) query += " AND unique_structure = 1";
        query += warned_filter(db, exclude_warned);

        db << query >> [&](int seed, int sg, int row_stage, double energy, double density,
                           double a, double b, double c, double al, double be, double ga,
                           double tx, double ty, double tz, double rx, double ry, double rz,
                           int n_mols, std::unique_ptr<std::string> dofs_json) {
            // `row_stage`, not the --stage request (0 means "highest").
            if (!seen.insert({seed, sg}).second) return;
            Row r{};
            r.seed = seed; r.sg = sg; r.stage = row_stage; r.src_db = dbf;
            r.energy = energy; r.density = density;
            r.n_mols = std::max(1, n_mols);
            r.cp[0]=a; r.cp[1]=b; r.cp[2]=c; r.cp[3]=al; r.cp[4]=be; r.cp[5]=ga;
            r.tr[0]=tx; r.tr[1]=ty; r.tr[2]=tz;
            r.rv[0]=rx; r.rv[1]=ry; r.rv[2]=rz;
            if (r.n_mols > 1 && dofs_json) {
                auto jd = nlohmann::json::parse(*dofs_json);
                auto& jt = jd["t"];
                auto& jr = jd["r"];
                for (int i = 0; i < r.n_mols * 3 && i < MAX_INDEP * 3; ++i) {
                    r.tr[i] = jt[i].get<double>();
                    r.rv[i] = jr[i].get<double>();
                }
            }
            all_rows.push_back(r);
        };
    }

    std::sort(all_rows.begin(), all_rows.end(),
              [](const Row& a, const Row& b) { return a.energy < b.energy; });

    int n_out = static_cast<int>(all_rows.size());
    if (top_n > 0 && top_n < n_out) n_out = top_n;

    std::filesystem::create_directories(output_dir);

    FlexibleReader flex_reader;

    // CSP provenance. all_rows is energy-sorted, so the first is the global
    // minimum that relative energies are measured from.
    const CspCifMetadata csp_run = csp_metadata_for_run(db_files.front(), read_stage);
    const double global_min_energy = all_rows.empty() ? 0.0 : all_rows.front().energy;
    auto csp_for = [&](const Row& r) {
        CspCifMetadata m = csp_run;
        m.have_energy = true;
        m.energy_absolute = r.energy;
        m.energy_relative = r.energy - global_min_energy;
        m.density = r.density;
        m.have_density = r.density > 0;
        return m;
    };

    if (format == "json") {
        // JSON: one file per structure
        for (int i = 0; i < n_out; ++i) {
            auto& r = all_rows[i];
            std::string path = fmt::format("{}/sg{}_seed{}.json", output_dir, r.sg, r.seed);
            write_crystal_json(path, basis, r.sg, r.cp, r.tr, r.rv, r.n_mols, r.energy);
        }
        logging::info("Extracted {} JSON files to {}/\n", n_out, output_dir);
        result.output_path = output_dir;
    } else if (format == "chemiscope") {
        // Single chemiscope JSON for every structure, expanded to the full unit
        // cell.
        nlohmann::json doc;
        doc["meta"] = {
            {"name", "klasp CSP extract"},
            {"description", fmt::format(
                "{} structure{} extracted from CSP database, sorted by energy",
                n_out, n_out == 1 ? "" : "s")}
        };
        auto& structs = doc["structures"] = nlohmann::json::array();
        std::vector<double> p_energy, p_density, p_a, p_b, p_c, p_alpha, p_beta, p_gamma;
        std::vector<int>    p_sg, p_seed;

        for (int i = 0; i < n_out; ++i) {
            auto& r = all_rows[i];
            auto& sg_info = get_space_group(r.sg);

            // Cell matrix (columns = lattice vectors, Bohr)
            RealType cp_rt[6];
            for (int k = 0; k < 6; ++k) cp_rt[k] = static_cast<RealType>(r.cp[k]);
            RealType cell[3][3];
            cell_params_to_matrix(cp_rt, cell);
            RealType cell_inv[3][3] = {};
            (void)mat3_inv(cell, cell_inv);

            std::vector<std::string> names;
            std::vector<double> xs, ys, zs;

            // Use the explicit relaxed geometry when the row has one, as the CIF
            // branch does: stage-3 rigid DOFs are stage-2 placeholders.
            FlexibleStructure cfs;
            const bool cs_flex = flex_reader.read(r.src_db, r.seed, r.sg, r.stage, cfs);
            if (cs_flex) {
                // Stored fractional against a cell in Angstrom; chemiscope wants
                // Cartesian Angstrom, and `cell` above is Bohr.
                RealType fcp[6];
                for (int k = 0; k < 6; ++k) fcp[k] = static_cast<RealType>(cfs.cell_params[k]);
                RealType fcell[3][3];
                cell_params_to_matrix(fcp, fcell);
                for (int a = 0; a < cfs.n_atoms; ++a) {
                    double c3[3] = {0, 0, 0};
                    for (int k = 0; k < 3; ++k)
                        for (int l = 0; l < 3; ++l)
                            c3[k] += static_cast<double>(fcell[k][l]) * cfs.frac[3 * a + l];
                    names.push_back(element_symbol(cfs.Z[a]));
                    xs.push_back(c3[0]); ys.push_back(c3[1]); zs.push_back(c3[2]);
                }
            }
            for (int op_idx = 0; !cs_flex && op_idx < sg_info.n_symops; ++op_idx) {
                double S_frac[3][3], t_sym[3];
                get_symop(sg_info, op_idx, S_frac, t_sym);

                // S_cart = cell @ S_frac @ cell_inv (for rotating body positions)
                RealType Sfrac_rt[3][3];
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l)
                        Sfrac_rt[k][l] = static_cast<RealType>(S_frac[k][l]);
                RealType S_cart[3][3];
                frac_to_cart_symop(cell, cell_inv, Sfrac_rt, S_cart);

                for (int m = 0; m < r.n_mols; ++m) {
                    // Fractional COM after symop: S_frac @ frac_t + t_sym, wrapped
                    // to [0,1). Molecules may poke past cell faces.
                    double frac_com[3];
                    for (int k = 0; k < 3; ++k) {
                        double f = S_frac[k][0]*r.tr[m*3+0]
                                 + S_frac[k][1]*r.tr[m*3+1]
                                 + S_frac[k][2]*r.tr[m*3+2]
                                 + t_sym[k];
                        frac_com[k] = f - std::floor(f);
                    }
                    double cart_com[3] = {0, 0, 0};
                    for (int k = 0; k < 3; ++k)
                        for (int l = 0; l < 3; ++l)
                            cart_com[k] += cell[k][l] * frac_com[l];

                    // R_total = S_cart @ R_mol
                    RealType aa[3] = {
                        static_cast<RealType>(r.rv[m*3+0]),
                        static_cast<RealType>(r.rv[m*3+1]),
                        static_cast<RealType>(r.rv[m*3+2])};
                    RealType R_mol[3][3];
                    angle_axis_to_matrix(aa, R_mol);
                    RealType R_tot[3][3];
                    mat3_mul(S_cart, R_mol, R_tot);

                    int body_off = m * basis.n_sites;
                    int ns_m = (m < static_cast<int>(basis.mol_n_sites.size()))
                             ? basis.mol_n_sites[m] : basis.n_sites;
                    for (int j = 0; j < ns_m; ++j) {
                        double rotated[3] = {0, 0, 0};
                        for (int k = 0; k < 3; ++k)
                            for (int l = 0; l < 3; ++l)
                                rotated[k] += static_cast<double>(R_tot[k][l])
                                            * basis.body_positions[body_off+j][l];
                        names.push_back(basis.elements[body_off+j]);
                        xs.push_back((rotated[0] + cart_com[0]) * BOHR_TO_ANGSTROM);
                        ys.push_back((rotated[1] + cart_com[1]) * BOHR_TO_ANGSTROM);
                        zs.push_back((rotated[2] + cart_com[2]) * BOHR_TO_ANGSTROM);
                    }
                }
            }

            nlohmann::json s;
            s["size"] = static_cast<int>(names.size());
            s["names"] = names;
            s["x"] = xs;
            s["y"] = ys;
            s["z"] = zs;
            // chemiscope cell: row-major 9 floats, lattice vectors as rows
            // (Angstrom), i.e. row k is column k of our matrix.
            double B = static_cast<double>(BOHR_TO_ANGSTROM);
            s["cell"] = {
                cell[0][0]*B, cell[1][0]*B, cell[2][0]*B,
                cell[0][1]*B, cell[1][1]*B, cell[2][1]*B,
                cell[0][2]*B, cell[1][2]*B, cell[2][2]*B
            };
            structs.push_back(std::move(s));

            p_energy.push_back(r.energy);
            p_density.push_back(r.density);
            p_sg.push_back(r.sg);
            p_seed.push_back(r.seed);
            p_a.push_back(r.cp[0] * B);
            p_b.push_back(r.cp[1] * B);
            p_c.push_back(r.cp[2] * B);
            p_alpha.push_back(r.cp[3]);
            p_beta.push_back(r.cp[4]);
            p_gamma.push_back(r.cp[5]);
        }

        auto prop = [](const char* units, const char* desc, auto&& values) {
            nlohmann::json p;
            p["target"] = "structure";
            p["values"] = std::forward<decltype(values)>(values);
            if (units && *units) p["units"] = units;
            if (desc && *desc)   p["description"] = desc;
            return p;
        };
        doc["properties"] = {
            {"energy",      prop("kJ/mol", "Lattice energy", p_energy)},
            {"density",     prop("g/cm^3", "Crystal density", p_density)},
            {"space_group", prop("", "International Tables number", p_sg)},
            {"seed",        prop("", "Generator seed", p_seed)},
            {"a",           prop("Å", "", p_a)},
            {"b",           prop("Å", "", p_b)},
            {"c",           prop("Å", "", p_c)},
            {"alpha",       prop("°", "", p_alpha)},
            {"beta",        prop("°", "", p_beta)},
            {"gamma",       prop("°", "", p_gamma)},
        };

        // Default view: density on x, energy on y.
        doc["settings"] = {
            {"target", "structure"},
            {"map", {
                {"x", {{"property", "density"}}},
                {"y", {{"property", "energy"}}},
                {"color", {{"property", "space_group"}}}
            }}
        };

        std::string path = fmt::format("{}/chemiscope.json", output_dir);
        std::ofstream out(path);
        out << doc.dump() << "\n";
        result.output_path = path;
        logging::info("Extracted {} structures to {} (load at chemiscope.org)\n",
                   n_out, path);
    } else {
        // CIF: one file, one data block per structure. Explicit non-rigid
        // geometry (flexible_structure) is used where present.
        std::string cif_path = fmt::format("{}/structures.cif", output_dir);
        std::ofstream cif_out(cif_path);
        int n_flex = 0, n_reduced = 0, n_unreducible = 0;
        int n_checked = 0, n_disagree = 0;
        double worst_diff = 0; int worst_seed = -1, worst_sg = 0;
        double worst_stored = 0, worst_recomputed = 0;
        std::unique_ptr<Workspace> check_ws;
        if (check_energy)
            check_ws = std::make_unique<Workspace>(
                allocate_workspace(512, 27*27*27, 50000, 32, 4));
        for (int i = 0; i < n_out; ++i) {
            auto& r = all_rows[i];
            FlexibleStructure fs;
            const bool have_flex = flex_reader.read(r.src_db, r.seed, r.sg, r.stage, fs);
            if (have_flex) {
                ++n_flex;
                const CspCifMetadata m = csp_for(r);
                write_cif_explicit(cif_out, fs.cell_params, fs.n_atoms, fs.Z.data(),
                                   fs.frac.data(), r.sg, r.energy,
                                   CspStructureId{structure_name, "QR", r.sg, r.seed, r.stage},
                                   &m);
                cif_out << "\n";
            } else {
                if (check_energy) {
                    const double e = recompute_energy(*check_ws, basis, r.sg, r.cp,
                                                      r.tr, r.rv, r.n_mols);
                    const double d = e - r.energy;
                    ++n_checked;
                    if (std::abs(d) > ENERGY_CHECK_TOL) ++n_disagree;
                    if (std::abs(d) > std::abs(worst_diff)) {
                        worst_diff = d; worst_seed = r.seed; worst_sg = r.sg;
                        worst_stored = r.energy; worst_recomputed = e;
                    }
                }
                const CspCifMetadata m = csp_for(r);
                if (write_cif(cif_out, basis, r.sg, r.cp, r.tr, r.rv, r.n_mols, r.energy,
                              CspStructureId{structure_name, "QR", r.sg, r.seed, r.stage},
                              niggli, &m)) ++n_reduced;
                else if (niggli && !niggli_applicable(r.sg)) ++n_unreducible;
                cif_out << "\n";
            }
        }
        result.output_path = cif_path;
        logging::info("Extracted {} structures to {}{}\n", n_out, cif_path,
                   n_flex > 0 ? fmt::format(" ({} with explicit PET geometry)", n_flex) : "");
        if (niggli)
            logging::info("  Niggli reduced {} cell{}; {} left alone (only P1 and P-1 have "
                       "operations that survive a change of basis)\n",
                       n_reduced, n_reduced == 1 ? "" : "s", n_unreducible);
        if (check_energy && n_checked > 0) {
            logging::info("  Re-evaluated {} structure{}: {} disagree with the stored energy "
                       "by more than {} kJ/mol\n",
                       n_checked, n_checked == 1 ? "" : "s", n_disagree, ENERGY_CHECK_TOL);
            if (worst_seed >= 0 && std::abs(worst_diff) > ENERGY_CHECK_TOL)
                logging::info("    worst: sg {} seed {}, stored {:.3f}, recomputed {:.3f}, "
                           "difference {:+.3f} kJ/mol\n",
                           worst_sg, worst_seed, worst_stored, worst_recomputed, worst_diff);
        }
        result.energy_disagreements = n_disagree;
    }
    logging::info("  {} total across {} DB{}, sorted by energy\n",
               all_rows.size(), db_files.size(), db_files.size() > 1 ? "s" : "");
    result.ok = true;
    result.written = n_out;
    result.rows_read = static_cast<long long>(all_rows.size());

    // Optional: dump stored powder patterns as Sietronics .cpi files.
    if (dump_powder) {
        int written = 0, missing = 0;
        // One connection per input file, held open for the whole dump.
        std::vector<sqlite::database> powder_dbs;
        powder_dbs.reserve(db_files.size());
        for (const auto& dbf : db_files) powder_dbs.push_back(open_database_readonly(dbf));
        for (int i = 0; i < n_out; ++i) {
            auto& r = all_rows[i];
            // Rows are merged across input files; search each until a hit.
            std::vector<float> pattern;
            for (auto& db : powder_dbs) {
                std::unique_ptr<std::vector<char>> blob;
                db << "SELECT value FROM descriptor WHERE seed=? AND space_group=? "
                      "AND stage=? AND name='powder'"
                   << r.seed << r.sg << r.stage
                   >> [&](std::unique_ptr<std::vector<char>> b) { blob = std::move(b); };
                if (blob && !blob->empty()) {
                    pattern.assign(POWDER_N_BINS, 0.0f);
                    if (!decode_powder(blob->data(), blob->size(),
                                       pattern.data(), POWDER_N_BINS)) {
                        pattern.clear();
                        continue;   // unreadable here; another rank DB may have it
                    }
                    break;
                }
            }
            if (pattern.empty()) { ++missing; continue; }

            const int n_bins = static_cast<int>(pattern.size());
            const double step = (powder_two_theta_max - powder_two_theta_min)
                                / static_cast<double>(n_bins);
            std::string cpi_path = fmt::format("{}/seed_{}_sg{}.cpi",
                                                output_dir, r.seed, r.sg);
            std::ofstream out(cpi_path);
            out << "SIETRONICS XRD SCAN\n"
                << fmt::format("{:.4f}\n", powder_two_theta_min)
                << fmt::format("{:.4f}\n", powder_two_theta_max)
                << fmt::format("{:.4f}\n", step)
                << "Cu\n1.54056\n"
                << "klasp CSP powder descriptor\n"
                << "1\n"
                << fmt::format("seed_{} sg{}\n", r.seed, r.sg)
                << "SCANDATA\n";
            for (int j = 0; j < n_bins; ++j)
                out << fmt::format("{:.6e}\n", pattern[j]);
            ++written;
        }
        logging::info("Wrote {} .cpi powder pattern file{} to {} ({} missing)\n",
                   written, written == 1 ? "" : "s", output_dir, missing);
        result.powder_files = written;
    }
    return result;
}

// Above this fraction of uncomparable structures the unique count is not
// meaningful.
static constexpr double kMaxUncomparableFraction = 0.05;

PruneResult cmd_prune(const PruneOptions& o) {
    PruneResult result;
    const std::string& db_path = o.db_path;
    const std::string& basis_path_opt = o.basis_path;
    const std::string& output_dir = o.output_dir;
    DedupConfig dedup_cfg = o.dedup;
    const int top_n = o.filter.top_n;
    const double max_energy = o.filter.max_energy;
    const bool exclude_catastrophes = !o.filter.keep_catastrophes;
    const double catastrophe_threshold = o.filter.catastrophe_threshold;
    const bool exclude_warned = !o.filter.keep_warned;
    const int stage = o.filter.stage;
    const bool allow_missing_descriptors = o.allow_missing_descriptors;
    const std::string& name_override = o.name_override;
    std::vector<std::string> db_files;
    if (std::filesystem::is_directory(db_path)) {
        for (auto& entry : std::filesystem::directory_iterator(db_path)) {
            auto p = entry.path();
            if (p.extension() == ".db" && p.filename().string().find("csp") != std::string::npos)
                db_files.push_back(p.string());
        }
        std::sort(db_files.begin(), db_files.end());
    } else {
        db_files.push_back(db_path);
    }
    if (db_files.empty()) { logging::info("No database files found in {}\n", db_path); return result; }
    const std::string structure_name = resolve_structure_name(db_files, name_override);

    // Host-only: no Workspace, no Kokkos Views.
    BasisConfig basis;
    if (!basis_path_opt.empty()) {
        basis = load_basis_host_only_from_file(basis_path_opt);
    } else {
        CSPDatabase first_db;
        first_db.open(db_files[0]);
        std::string basis_json_str = first_db.get_meta("basis_json");
        first_db.close();
        if (basis_json_str.empty()) {
            logging::info("No basis JSON in database and no basis file specified.\n");
            return result;
        }
        basis = load_basis_host_only(recorded_basis(basis_json_str));
    }

    logging::info("Pruning {} database file{}\n", db_files.size(),
               db_files.size() > 1 ? "s" : "");

    // Shared reader (run_state.hpp); rows come back in ascending energy, so the
    // lowest-energy member of a basin is its representative.
    LandscapeFilter lf;
    lf.stage = resolve_stage(db_files, stage, "prune");
    require_explicit_geometry(db_files, lf.stage, "prune");
    lf.max_energy = max_energy;
    lf.exclude_catastrophes = exclude_catastrophes;
    lf.catastrophe_threshold = catastrophe_threshold;
    lf.exclude_warned = exclude_warned;
    lf.want_amd = dedup_cfg.amd_tol > 0;
    lf.want_powder = dedup_cfg.use_powder;
    lf.want_pdd = dedup_cfg.pdd_tol > 0;
    if (lf.want_pdd) {
        dedup_cfg.pdd_rows = basis.n_asym_atoms();
        dedup_cfg.pdd_k = AMD_K;
        dedup_cfg.pdd_blocks = pdd_symmetry_blocks(basis);
    }
    Landscape land = load_landscape(db_files, lf);
    auto& all_rows = land.rows;

    logging::info("  Loaded {} structures from {} files\n", all_rows.size(), db_files.size());

    // Energy-sorted, so lower-energy structures are kept as representatives.
    StructureDedup dedup(dedup_cfg);
    std::vector<int> unique_indices;
    const bool need_amd    = dedup_cfg.amd_tol > 0;
    const bool need_powder = dedup_cfg.use_powder;
    const bool need_pdd    = dedup_cfg.pdd_tol > 0;
    const size_t pdd_len   = static_cast<size_t>(dedup_cfg.pdd_rows) * dedup_cfg.pdd_k;

    // A structure with no descriptor is kept as unique; count and report them.
    int n_uncomparable = 0;
    std::map<std::string, int> missing;
    for (int i = 0; i < static_cast<int>(all_rows.size()); ++i) {
        const char* lacks = need_amd && !land.has_amd[i]               ? "AMD"
                          : need_powder && !land.has_powder[i]         ? "powder"
                          : need_pdd && land.pdd[i].size() != pdd_len  ? "PDD"
                          : nullptr;
        if (lacks) {
            unique_indices.push_back(i);
            ++n_uncomparable;
            ++missing[lacks];
            continue;
        }
        const auto& pdd = land.pdd[i];
        if (dedup.find_duplicate_or_add(all_rows[i], pdd.empty() ? nullptr : pdd.data(),
                                        static_cast<int>(pdd.size())) < 0) {
            unique_indices.push_back(i);
        }
    }

    if (n_uncomparable) {
        const double frac = double(n_uncomparable) / std::max<std::size_t>(1, all_rows.size());
        std::string which;
        for (const auto& [name, n] : missing)
            which += fmt::format("{}{} {}", which.empty() ? "" : ", ", n, name);
        logging::info("  {} of {} structures ({:.1f}%) lack a descriptor ({}) and were kept as "
                   "unique without being compared to anything.\n",
                   n_uncomparable, all_rows.size(), 100.0 * frac, which);
        if (!allow_missing_descriptors && frac > kMaxUncomparableFraction)
            throw std::runtime_error(fmt::format(
                "prune: {:.1f}% of structures could not be compared for want of a "
                "descriptor, so the unique count below is mostly an artefact. Re-run the "
                "CSP with the descriptor stored, prune with --no-powder-dedup / "
                "--amd-tol 0 / --dedup-pdd 0 to drop the stage that needs it, or pass "
                "--allow-missing-descriptors to proceed anyway.", 100.0 * frac));
    }

    int n_out = static_cast<int>(unique_indices.size());
    if (top_n > 0 && top_n < n_out) n_out = top_n;

    std::map<int, int> sg_total, sg_unique;
    for (auto& r : all_rows) sg_total[r.space_group]++;
    for (int idx : unique_indices) sg_unique[all_rows[idx].space_group]++;

    logging::info("  {} unique / {} total ({:.1f}% duplicates removed) [{}]\n",
               unique_indices.size(), all_rows.size(),
               100.0 * (1.0 - double(unique_indices.size()) / std::max(1.0, double(all_rows.size()))),
               dedup_cfg.cross_sg ? "cross-SG" : "per-SG");
    logging::info("  AMD tol={:.3f} Å, energy bin={:.1f} kJ/mol, density bin={:.3f} g/cm³",
               dedup_cfg.amd_tol, dedup_cfg.energy_bin, dedup_cfg.density_bin);
    if (need_pdd)
        logging::info(", PDD RMS tol={:.1e} Å", dedup_cfg.pdd_tol);
    if (dedup_cfg.use_powder)
        logging::info(", powder cos_thresh={:.2f}, cdtw_thresh={:.1f}",
                   dedup_cfg.powder_cos_thresh, dedup_cfg.powder_cdtw_thresh);
    logging::info("\n");
    logging::info("\n  {:>4s}  {:>8s}  {:>8s}  {:>6s}\n", "SG", "total", "unique", "dup%");
    for (auto& [sg, total] : sg_total) {
        int uniq = sg_unique.count(sg) ? sg_unique[sg] : 0;
        logging::info("  {:>4d}  {:>8d}  {:>8d}  {:>5.1f}%\n",
                   sg, total, uniq, 100.0 * (1.0 - double(uniq) / std::max(1, total)));
    }

    std::filesystem::create_directories(output_dir);

    std::string cif_path = fmt::format("{}/unique_structures.cif", output_dir);
    {
        std::ofstream cif_out(cif_path);
        FlexibleReader flex_reader;
        for (int i = 0; i < n_out; ++i) {
            auto& r = all_rows[unique_indices[i]];
            const CspStructureId id{structure_name, "QR", r.space_group, r.seed, r.stage};
            FlexibleStructure fs;
            bool have_flex = false;
            for (const auto& file : db_files)
                if ((have_flex = flex_reader.read(file, r.seed, r.space_group, r.stage, fs)))
                    break;
            if (have_flex)
                write_cif_explicit(cif_out, fs.cell_params, fs.n_atoms, fs.Z.data(),
                                   fs.frac.data(), r.space_group, r.energy, id);
            else
                write_cif(cif_out, basis, r.space_group, r.cell_params, r.translation,
                          r.rotation, r.n_mols, r.energy, id);
            cif_out << "\n";
        }
    }
    logging::info("\n  Wrote {} unique structures to {}\n", n_out, cif_path);

    std::string csv_path = fmt::format("{}/unique_structures.csv", output_dir);
    {
        std::ofstream csv(csv_path);
        csv << "rank,seed,sg,energy,density,a,b,c,alpha,beta,gamma\n";
        for (int i = 0; i < n_out; ++i) {
            auto& r = all_rows[unique_indices[i]];
            // cell_params are stored in Bohr; written in Angstrom, as in the CIF.
            csv << fmt::format("{},{},{},{:.6f},{:.4f},{:.4f},{:.4f},{:.4f},{:.2f},{:.2f},{:.2f}\n",
                               i + 1, r.seed, r.space_group, r.energy, r.density,
                               r.cell_params[0] * BOHR_TO_ANGSTROM,
                               r.cell_params[1] * BOHR_TO_ANGSTROM,
                               r.cell_params[2] * BOHR_TO_ANGSTROM,
                               r.cell_params[3], r.cell_params[4], r.cell_params[5]);
        }
    }
    logging::info("  Wrote summary to {}\n", csv_path);

    result.ok = true;
    result.considered = static_cast<int>(all_rows.size());
    result.unique = n_out;
    result.cif_path = cif_path;
    result.csv_path = csv_path;
    return result;
}

namespace {

// Bytes each descriptor kind occupies (payload only, so a slight underestimate).
std::string describe_descriptor_usage(const std::string& file) {
    std::string out;
    try {
        sqlite::database db = open_database_readonly(file);
        db << "SELECT name, COUNT(*), SUM(LENGTH(value)) FROM descriptor GROUP BY name"
           >> [&](std::string name, long long rows, long long bytes) {
               out += fmt::format("      {:<8s} {:>8d} rows  {:>8.1f} MB\n",
                                  name, rows, double(bytes) / 1e6);
           };
    } catch (const std::exception&) {
        // No descriptor table, or unreadable: the caller still reports sizes.
    }
    return out;
}

double file_mb(const std::string& f) {
    std::error_code ec;
    const auto n = std::filesystem::file_size(f, ec);
    return ec ? 0.0 : double(n) / 1e6;
}

}  // namespace


namespace {

// Rebuild the descriptors of one database from the geometry it still holds.
// Stage-3 rows use flexible_structure; rigid rows are expanded through the
// space-group operators. For rigid rows this reproduces the stored AMD, since
// each asymmetric-unit atom's orbit contributes identical rows.
int restore_descriptors_for(const std::string& file, const BasisConfig& basis) {
    struct Row { int seed, sg, stage; };
    // One geometry per structure, keyed so the flexible pass can replace the
    // rigid expansion by lookup.
    std::map<std::tuple<int, int, int>, CrystalGeometry> geometry;

    {
        sqlite::database db = open_database_readonly(file);
        // Rigid DOFs for every row, and the explicit geometry where one exists.
        db << "SELECT seed, space_group, stage, a, b, c, alpha, beta, gamma, "
              "tx, ty, tz, rx, ry, rz, n_mols, dofs FROM structure"
           >> [&](int seed, int sg, int stage, double a, double b, double c,
                  double al, double be, double ga, double tx, double ty, double tz,
                  double rx, double ry, double rz, int n_mols,
                  std::unique_ptr<std::string> dofs_json) {
            double cp[6] = {a, b, c, al, be, ga};
            double tr[MAX_INDEP * 3] = {tx, ty, tz};
            double rv[MAX_INDEP * 3] = {rx, ry, rz};
            const int nm = std::max(1, n_mols);
            if (nm > 1 && dofs_json) {
                const auto jd = nlohmann::json::parse(*dofs_json);
                for (int i = 0; i < nm * 3 && i < MAX_INDEP * 3; ++i) {
                    tr[i] = jd["t"][i].get<double>();
                    rv[i] = jd["r"][i].get<double>();
                }
            }
            CrystalGeometry g = expand_rigid_structure(basis, sg, cp, tr, rv, nm);
            if (g.positions.size() < 2) return;
            geometry[{seed, sg, stage}] = std::move(g);
        };
    }

    // Where flexible_structure has a stage-3 geometry, it replaces the expansion.
    {
        sqlite::database db = open_database_readonly(file);
        db << "SELECT seed, space_group, stage, n_atoms, a, b, c, alpha, beta, gamma, atoms "
              "FROM flexible_structure"
           >> [&](int seed, int sg, int stage, int n_atoms, double a, double b, double c,
                  double al, double be, double ga, std::vector<char> blob) {
            if (n_atoms < 2) return;
            FlexibleStructure fs;
            if (!CSPDatabase::decode_flexible_atoms(n_atoms, blob.data(), blob.size(), fs))
                return;
            fs.cell_params[0] = a;  fs.cell_params[1] = b;  fs.cell_params[2] = c;
            fs.cell_params[3] = al; fs.cell_params[4] = be; fs.cell_params[5] = ga;
            geometry[{seed, sg, stage}] = from_flexible(fs);
        };
    }

    // compute_flex_descriptors takes a batch of one atom count, and Z differs
    // between space groups.
    std::map<std::pair<int, int>, std::vector<std::pair<Row, CrystalGeometry>>> by_shape_stage;
    for (auto& [key, g] : geometry) {
        const auto [seed, sg, stage] = key;
        const int n = static_cast<int>(g.positions.size());
        by_shape_stage[{n, stage}].push_back({{seed, sg, stage}, std::move(g)});
    }

    CSPDatabase out;
    out.open(file);
    int written = 0;
    for (auto& [shape, entries] : by_shape_stage) {
        constexpr std::size_t kChunk = 256;   // bounds the descriptor workspace
        for (std::size_t off = 0; off < entries.size(); off += kChunk) {
            const std::size_t n = std::min(kChunk, entries.size() - off);
            std::vector<CrystalGeometry> geoms;
            geoms.reserve(n);
            for (std::size_t i = 0; i < n; ++i) geoms.push_back(std::move(entries[off + i].second));
            const FlexDescriptors fd = compute_flex_descriptors(geoms);
            for (std::size_t i = 0; i < n && i < fd.amd.size(); ++i) {
                const Row& r = entries[off + i].first;
                out.insert_amd(r.seed, r.sg, r.stage, fd.amd[i]);
                out.insert_powder(r.seed, r.sg, r.stage, fd.powder[i].data(),
                                  static_cast<int>(fd.powder[i].size()));
                ++written;
            }
        }
    }
    out.flush();
    out.close();
    return written;
}

}  // namespace

FinalStageDedup dedup_final_stage(const std::string& db_path) {
    FinalStageDedup out;
    struct Row { int seed, sg; double energy, density; std::vector<double> amd;
                 std::vector<float> powder; std::vector<double> pdd; };
    std::vector<Row> rows;
    std::map<int, std::vector<std::pair<int, CrystalGeometry>>> by_atoms;
    std::string cfg_str;
    {
        sqlite::database db = open_database_readonly(db_path);
        int has_flex = 0;
        db << "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
              "AND name='flexible_structure'" >> has_flex;
        if (!has_flex) return out;
        db << "SELECT s.seed, s.space_group, s.energy, s.density, f.n_atoms, "
              "f.a, f.b, f.c, f.alpha, f.beta, f.gamma, f.atoms "
              "FROM structure s JOIN flexible_structure f ON f.seed = s.seed "
              "AND f.space_group = s.space_group AND f.stage = s.stage "
              "WHERE s.stage = ? AND s.energy IS NOT NULL" << STAGE_FINAL
           >> [&](int seed, int sg, double energy, double density, int n_atoms,
                  double a, double b, double c, double al, double be, double ga,
                  std::vector<char> blob) {
                  FlexibleStructure fs;
                  if (!CSPDatabase::decode_flexible_atoms(n_atoms, blob.data(), blob.size(), fs))
                      return;
                  fs.cell_params[0] = a;  fs.cell_params[1] = b;  fs.cell_params[2] = c;
                  fs.cell_params[3] = al; fs.cell_params[4] = be; fs.cell_params[5] = ga;
                  by_atoms[n_atoms].push_back({static_cast<int>(rows.size()), from_flexible(fs)});
                  rows.push_back({seed, sg, energy, density, {}, {}, {}});
              };
        try { db << "SELECT value FROM meta WHERE key = 'config_json'" >> cfg_str; }
        catch (const std::exception&) {}
    }
    out.rows = static_cast<int>(rows.size());
    if (rows.empty()) return out;

    // The basis gives the PDD fold size and the molecule's symmetry; without it
    // the PDD tier is off and dedup uses AMD and powder alone.
    int n_asym = 0;
    std::vector<DedupConfig::PddBlock> blocks;
    try {
        const BasisConfig basis = load_csp_basis("", db_path);
        n_asym = basis.n_asym_atoms();
        blocks = pdd_symmetry_blocks(basis);
    } catch (const std::exception&) {}

    // compute_flex_descriptors takes a batch of one atom count, and Z differs
    // between space groups.
    for (auto& [n_atoms, entries] : by_atoms) {
        constexpr std::size_t kChunk = 256;   // bounds the descriptor workspace
        for (std::size_t off = 0; off < entries.size(); off += kChunk) {
            const std::size_t n = std::min(kChunk, entries.size() - off);
            std::vector<CrystalGeometry> geoms;
            geoms.reserve(n);
            for (std::size_t i = 0; i < n; ++i) geoms.push_back(entries[off + i].second);
            const FlexDescriptors fd = compute_flex_descriptors(
                geoms, /*want_pdd=*/n_asym > 0, /*cutoff_bohr=*/0.0, AMD_K,
                /*fold_rows=*/n_asym);
            for (std::size_t i = 0; i < n && i < fd.amd.size(); ++i) {
                Row& r = rows[entries[off + i].first];
                r.amd = fd.amd[i];
                r.powder = fd.powder[i];
                if (i < fd.pdd.size()) r.pdd = fd.pdd[i];
            }
        }
    }

    DedupConfig dcfg;
    if (!cfg_str.empty()) {
        try {
            const auto cfg = nlohmann::json::parse(cfg_str);
            if (cfg.contains("dedup")) dcfg = cfg.at("dedup").get<DedupConfig>();
        } catch (const std::exception&) {}
    }
    // The PDD tier only where the run asked for it and every row has a PDD.
    bool all_pdd = n_asym > 0;
    for (const Row& r : rows)
        if (r.pdd.size() != static_cast<std::size_t>(n_asym) * AMD_K) all_pdd = false;
    if (all_pdd && dcfg.pdd_tol > 0) {
        dcfg.pdd_rows = n_asym;
        dcfg.pdd_k = AMD_K;
        dcfg.pdd_blocks = blocks;
    } else {
        dcfg.pdd_tol = 0.0;
    }
    StructureDedup dedup(dcfg);

    std::vector<int> order(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return rows[a].energy < rows[b].energy; });

    CSPDatabase db;
    db.open(db_path);
    for (const int i : order) {
        const Row& r = rows[i];
        if (r.amd.size() != static_cast<std::size_t>(AMD_K)) continue;
        db.insert_amd(r.seed, r.sg, STAGE_FINAL, r.amd);
        db.insert_powder(r.seed, r.sg, STAGE_FINAL, r.powder.data(),
                         static_cast<int>(r.powder.size()));
        StructureResult s{};
        s.seed = r.seed;
        s.space_group = r.sg;
        s.stage = STAGE_FINAL;
        s.energy = r.energy;
        s.density = r.density;
        std::copy(r.amd.begin(), r.amd.end(), s.amd);
        std::copy_n(r.powder.begin(), std::min<std::size_t>(r.powder.size(), POWDER_N_BINS),
                    s.powder);
        if (!r.pdd.empty())
            db.insert_pdd(r.seed, r.sg, STAGE_FINAL, r.pdd.data(), n_asym, AMD_K);
        const StructureDedup::Verdict v = dcfg.pdd_tol > 0
            ? dedup.classify(s, r.pdd.data(), static_cast<int>(r.pdd.size()))
            : dedup.classify(s);
        db.set_verdict(r.seed, r.sg, STAGE_FINAL, v.duplicate_of, v.duplicate_of_sg);
        if (v.is_unique()) ++out.unique;
    }
    db.flush();
    db.close();
    return out;
}

CompactResult cmd_compact(const CompactOptions& o) {
    const std::string& db_path = o.db_path;
    const bool drop_descriptors = o.drop_descriptors;
    const bool dry_run = o.dry_run;
    const bool restore = o.restore;
    const std::string& basis_path_opt = o.basis_path;
    CompactResult result;
    result.dry_run = dry_run;
    std::vector<std::string> db_files;
    if (std::filesystem::is_directory(db_path)) {
        for (auto& entry : std::filesystem::directory_iterator(db_path)) {
            auto p = entry.path();
            if (p.extension() == ".db" && p.filename().string().find("csp") != std::string::npos)
                db_files.push_back(p.string());
        }
        std::sort(db_files.begin(), db_files.end());
    } else {
        db_files.push_back(db_path);
    }
    if (db_files.empty()) throw std::runtime_error("compact: no database files at " + db_path);

    // Dropping is only reversible via --restore, which needs the basis.
    if (drop_descriptors && !dry_run) {
        for (const auto& f : db_files) {
            try {
                (void)load_csp_basis("", f);
            } catch (const std::exception&) {
                throw std::runtime_error(fmt::format(
                    "compact: {} carries no basis, so --restore could not rebuild its\n"
                    "  descriptors and dropping them would be irreversible. Pass the basis to\n"
                    "  --restore later, or keep the descriptors.", f));
            }
        }
    }

    double before_total = 0, after_total = 0;
    for (const auto& f : db_files) {
        const double before = file_mb(f);
        before_total += before;
        logging::info("  {}\n    {:.1f} MB\n", f, before);
        const std::string usage = describe_descriptor_usage(f);
        if (!usage.empty()) logging::info("{}", usage);

        if (dry_run) { after_total += before; continue; }

        if (restore) {
            require_explicit_geometry({f}, STAGE_FINAL, "compact --restore");
            const BasisConfig basis = load_csp_basis(basis_path_opt, f);
            const int n = restore_descriptors_for(f, basis);
            logging::info("    recomputed {} descriptor set{}\n", n, n == 1 ? "" : "s");
        }

        try {
            sqlite::database db = open_database_readwrite(f);
            if (drop_descriptors) {
                // Everything derived from the geometry, including the PDD (a
                // `descriptor` row named 'pdd'). flexible_structure stays: it
                // cannot be recomputed.
                db << "DELETE FROM descriptor;";
            }
            // Outside any transaction, by SQLite's rule.
            db << "VACUUM;";
        } catch (const sqlite::errors::busy& e) {
            throw std::runtime_error(fmt::format("compact: {} on {}: {}", e.what(), f, kDbLockedHint));
        } catch (const sqlite::errors::locked& e) {
            throw std::runtime_error(fmt::format("compact: {} on {}: {}", e.what(), f, kDbLockedHint));
        } catch (const std::exception& e) {
            throw std::runtime_error(fmt::format("compact: {} on {}", e.what(), f));
        }

        const double after = file_mb(f);
        after_total += after;
        // Restoring grows the file; do not report a negative saving.
        if (after > before)
            logging::info("    -> {:.1f} MB (+{:.1f} MB)\n", after, after - before);
        else
            logging::info("    -> {:.1f} MB ({:.0f}% reclaimed)\n", after,
                       before > 0 ? 100.0 * (1.0 - after / before) : 0.0);
    }

    result.ok = true;
    result.files = static_cast<int>(db_files.size());
    result.mb_before = before_total;
    result.mb_after = after_total;
    if (dry_run) {
        logging::info("\n  --dry-run: nothing written. VACUUM alone typically reclaims about a\n"
                   "  third; adding --drop-descriptors takes a finished run to a few percent\n"
                   "  of its size.\n");
        return result;
    }
    if (after_total > before_total)
        logging::info("\n  {} file{}: {:.1f} -> {:.1f} MB (+{:.1f} MB)\n",
                   db_files.size(), db_files.size() == 1 ? "" : "s",
                   before_total, after_total, after_total - before_total);
    else
        logging::info("\n  {} file{}: {:.1f} -> {:.1f} MB ({:.0f}% reclaimed)\n",
                   db_files.size(), db_files.size() == 1 ? "" : "s",
                   before_total, after_total,
                   before_total > 0 ? 100.0 * (1.0 - after_total / before_total) : 0.0);
    if (drop_descriptors)
        logging::info("  Descriptors are gone: prune will refuse until they are put back with\n"
                   "  `klasp compact <db> --restore`. extract and summary still work.\n");
    return result;
}

// ASCII summary of a finished CSP database: per-SG totals, energy and density
// histograms over unique non-catastrophes, and the top-N lowest-energy
// structures.
SummaryResult cmd_summary(const SummaryOptions& o) {
    const std::string& db_path = o.db_path;
    const double catastrophe_threshold = o.catastrophe_threshold;
    const int top_n = o.top_n;
    const int n_bins = o.n_bins;
    const double max_energy_clip = o.max_energy_clip;
    const int stage = o.stage;
    SummaryResult result;
    // A single .db or a directory of csp_rank*.db, as in cmd_extract.
    std::vector<std::string> db_files;
    if (std::filesystem::is_directory(db_path)) {
        for (auto& entry : std::filesystem::directory_iterator(db_path)) {
            auto p = entry.path();
            if (p.extension() == ".db" && p.filename().string().find("csp") != std::string::npos)
                db_files.push_back(p.string());
        }
        std::sort(db_files.begin(), db_files.end());
    } else {
        db_files.push_back(db_path);
    }

    if (db_files.empty()) { logging::info("No database files in {}\n", db_path); return result; }
    const int read_stage = resolve_stage(db_files, stage, "summary");

    struct Row {
        int seed, sg;
        double energy, density;
        bool unique;
    };
    std::vector<Row> rows;
    std::set<std::pair<int,int>> seen;  // (seed, sg) across rank DBs
    for (auto& dbf : db_files) {
        sqlite::database db = open_database_readonly(dbf);
        db << ("SELECT seed, space_group, energy, density, unique_structure"
               " FROM structure WHERE stage = " + std::to_string(read_stage))
           >> [&](int seed, int sg, double energy, double density, int uniq) {
               auto key = std::make_pair(seed, sg);
               if (seen.count(key)) return;
               seen.insert(key);
               rows.push_back({seed, sg, energy, density, uniq != 0});
           };
    }

    if (rows.empty()) { logging::info("No structures found in {}\n", db_path); return result; }

    result.ok = true;
    result.structures = static_cast<long long>(rows.size());
    {
        std::set<int> sgs;
        double lowest = rows[0].energy;
        for (const auto& r : rows) {
            sgs.insert(r.sg);
            if (r.energy < lowest) lowest = r.energy;
        }
        result.space_groups = static_cast<int>(sgs.size());
        result.min_energy = lowest;
    }

    // Split into catastrophes (energy below threshold) and the rest.
    std::vector<Row> catastrophes, clean;
    catastrophes.reserve(rows.size()); clean.reserve(rows.size());
    for (auto& r : rows) {
        if (!std::isfinite(r.energy) || r.energy < catastrophe_threshold)
            catastrophes.push_back(r);
        else
            clean.push_back(r);
    }

    // --- Per-SG table ---
    std::map<int, std::vector<const Row*>> by_sg_all, by_sg_clean;
    for (auto& r : rows)   by_sg_all[r.sg].push_back(&r);
    for (auto& r : clean)  by_sg_clean[r.sg].push_back(&r);

    logging::info("=== klasp CSP summary ({} DB file{}) ===\n",
               db_files.size(), db_files.size() > 1 ? "s" : "");
    logging::info("  {} total structures, {} unique, {} catastrophes (E < {} kJ/mol)\n",
               rows.size(),
               std::count_if(rows.begin(), rows.end(), [](const Row& r){ return r.unique; }),
               catastrophes.size(), catastrophe_threshold);

    logging::info("\n  {:>4s}  {:>8s}  {:>8s}  {:>6s}  {:>12s}  {:>12s}  {:>12s}\n",
               "SG", "total", "unique", "cata", "min E(kJ)", "median E", "max E");
    for (auto& [sg, list_all] : by_sg_all) {
        const auto& list_clean = by_sg_clean[sg];
        int total  = static_cast<int>(list_all.size());
        int unique = static_cast<int>(std::count_if(list_all.begin(), list_all.end(),
                                      [](const Row* r){ return r->unique; }));
        int cata   = total - static_cast<int>(list_clean.size());
        double e_min = 0, e_med = 0, e_max = 0;
        if (!list_clean.empty()) {
            std::vector<double> es;
            es.reserve(list_clean.size());
            for (auto* r : list_clean) es.push_back(r->energy);
            std::sort(es.begin(), es.end());
            e_min = es.front();
            e_max = es.back();
            e_med = es[es.size() / 2];
        }
        logging::info("  {:>4d}  {:>8d}  {:>8d}  {:>6d}  {:>12.3f}  {:>12.3f}  {:>12.3f}\n",
                   sg, total, unique, cata, e_min, e_med, e_max);
    }

    // --- Histograms ---
    auto ascii_hist = [&](const std::string& title, const char* units,
                          std::vector<double>& values) {
        if (values.empty()) { logging::info("\n  {} — no data\n", title); return; }
        const double lo = *std::min_element(values.begin(), values.end());
        const double hi = *std::max_element(values.begin(), values.end());
        if (hi <= lo) {
            logging::info("\n  {} — constant at {:.3f} {}\n", title, lo, units);
            return;
        }
        std::vector<int> bins(n_bins, 0);
        const double w = (hi - lo) / n_bins;
        for (double v : values) {
            int b = static_cast<int>((v - lo) / w);
            if (b >= n_bins) b = n_bins - 1;
            if (b < 0) b = 0;
            ++bins[b];
        }
        int bin_max = *std::max_element(bins.begin(), bins.end());
        const int bar_width = 40;
        logging::info("\n  {}  (n={}, range {:.3f} .. {:.3f} {})\n",
                   title, values.size(), lo, hi, units);
        for (int i = 0; i < n_bins; ++i) {
            const double bin_lo = lo + i * w;
            const int bar_len = bin_max > 0 ? (bins[i] * bar_width) / bin_max : 0;
            std::string bar(bar_len, '#');
            logging::info("  {:>10.3f}  {:>6d}  {}\n", bin_lo, bins[i], bar);
        }
    };

    std::vector<double> energies_all, energies, densities;
    energies_all.reserve(clean.size());
    for (auto& r : clean)
        if (r.unique) energies_all.push_back(r.energy);

    // Without an explicit clip, drop the top 5% of energies from the histogram
    // so outliers do not squash the low-energy region. The top-N table is
    // unaffected.
    double clip = max_energy_clip;
    bool auto_clip = !std::isfinite(max_energy_clip) || max_energy_clip >= 1e29;
    if (auto_clip && !energies_all.empty()) {
        auto sorted = energies_all;
        std::sort(sorted.begin(), sorted.end());
        clip = sorted[static_cast<size_t>(sorted.size() * 0.95)];
    }

    int n_clipped = 0;
    for (auto& r : clean) {
        if (!r.unique) continue;
        if (r.energy > clip) { ++n_clipped; continue; }
        energies.push_back(r.energy);
        if (r.density > 0) densities.push_back(r.density);
    }
    std::string e_title = auto_clip
        ? fmt::format("Energy (kJ/mol, unique, auto-clipped at p95={:.1f}, {} outliers dropped)",
                      clip, n_clipped)
        : fmt::format("Energy (kJ/mol, unique, clipped at {:.1f}, {} dropped)",
                      clip, n_clipped);
    ascii_hist(e_title, "kJ/mol", energies);
    ascii_hist("Density (g/cm^3, unique)", "g/cm^3", densities);

    // --- Top-N by energy ---
    std::vector<const Row*> clean_sorted;
    clean_sorted.reserve(clean.size());
    for (auto& r : clean) if (r.unique) clean_sorted.push_back(&r);
    std::sort(clean_sorted.begin(), clean_sorted.end(),
              [](const Row* a, const Row* b) { return a->energy < b->energy; });
    int n_print = std::min(top_n, static_cast<int>(clean_sorted.size()));
    logging::info("\n  Top {} lowest-energy unique structures:\n", n_print);
    logging::info("    {:>4s}  {:>8s}  {:>12s}  {:>10s}\n", "SG", "seed", "E(kJ/mol)", "rho");
    for (int i = 0; i < n_print; ++i) {
        auto& r = *clean_sorted[i];
        logging::info("    {:>4d}  {:>8d}  {:>12.3f}  {:>10.4f}\n",
                   r.sg, r.seed, r.energy, r.density);
    }
    logging::info("\n");
    return result;
}

// --- packing similarity over a landscape ------------------------------------

void run_compack_explicit(const std::vector<CifStructure>& structures,
                          const std::vector<std::string>& labels_in,
                          const CompackRunOptions& opts);

void group_and_report(const std::vector<CifPacking>& packings,
                      const std::vector<std::string>& labels, const RigidBody& body,
                      const CompackRunOptions& opts, const CompackOptions& copts,
                      const ClusterSpec& spec, double worst_fit);

void run_compack(const std::string& db_path, const CompackRunOptions& opts) {
    const BasisConfig basis = load_csp_basis(opts.basis_path, db_path);
    auto rows = read_landscape(db_path, opts.stage, /*filter_sg=*/0, /*max_energy=*/1e30,
                               /*unique_only=*/true, /*exclude_catastrophes=*/true,
                               /*catastrophe_threshold=*/-1000.0);
    if (rows.empty()) {
        logging::info("compack: no unique structures in {}\n", db_path);
        return;
    }
    const int n = std::min<int>(opts.top > 0 ? opts.top : (int)rows.size(), (int)rows.size());
    rows.resize(n);

    const double B = static_cast<double>(BOHR_TO_ANGSTROM);

    // When the stage stores explicit atoms (e.g. PET), compare those rather than
    // the rigid-body stand-ins.
    {
        CifReadOptions ropts;
        ropts.bond_tolerance = opts.bond_tolerance;
        ropts.merge_tolerance = opts.merge_tolerance;
        ropts.min_occupancy = opts.min_occupancy;
        FlexibleReader flex;
        std::vector<CifStructure> explicit_rows;
        std::vector<std::string> labels;
        int n_flex = 0;
        for (int i = 0; i < n; ++i) {
            FlexibleStructure fs;
            if (!flex.read(db_path, rows[i].seed, rows[i].space_group, rows[i].stage, fs))
                continue;
            ++n_flex;
            explicit_rows.push_back(structure_from_cell(fs.cell_params, fs.frac.data(),
                                                        fs.Z.data(), fs.n_atoms, ropts));
            labels.push_back(fmt::format("sg{}/seed{}", rows[i].space_group, rows[i].seed));
        }
        if (rows[0].stage >= STAGE_FINAL && n_flex < n)
            require_explicit_geometry({db_path}, rows[0].stage, "compack");
        if (n_flex > 0) {
            if (n_flex < n)
                logging::info("  ! {} of {} structures have explicit atoms and {} do not; "
                           "comparing only the {} that do, since a relaxed geometry and a "
                           "rigid reconstruction of one are not the same structure\n",
                           n_flex, n, n - n_flex, n_flex);
            run_compack_explicit(explicit_rows, labels, opts);
            return;
        }
    }

    // Bodies, one per component, in Angstrom (the unit of the tolerances).
    std::vector<std::vector<double>> mol_pos(basis.n_mols);
    std::vector<std::vector<int>> mol_z(basis.n_mols);
    bool heavy_only = opts.heavy_only;
    for (int m = 0; m < basis.n_mols; ++m) {
        const int ns = (m < (int)basis.mol_n_sites.size()) ? basis.mol_n_sites[m] : basis.n_sites;
        for (int j = 0; j < ns; ++j) {
            const int flat = m * basis.n_sites + j;
            for (int d = 0; d < 3; ++d) mol_pos[m].push_back(basis.body_positions[flat][d] * B);
            mol_z[m].push_back(element(basis.elements[flat]).atomic_number);
        }
        if (!heavy_atoms_suffice(mol_pos[m].data(), mol_z[m].data(), ns))
            heavy_only = false;
    }
    if (heavy_only != opts.heavy_only)
        logging::info("  (the heavy atoms lie on a line and cannot fix the molecule's "
                      "orientation, so hydrogens are compared too)\n");
    std::vector<RigidBody> bodies;
    for (int m = 0; m < basis.n_mols; ++m) {
        RigidBody rb = make_rigid_body(mol_pos[m].data(), mol_z[m].data(),
                                       static_cast<int>(mol_z[m].size()), heavy_only);
        detect_self_symmetry(rb, opts.self_symmetry_tolerance);
        bodies.push_back(std::move(rb));
    }
    if (bodies.empty() || bodies[0].m == 0) {
        logging::info("compack: the basis has no atoms to compare\n");
        return;
    }
    logging::info("compack: {} unique structures, {} atoms per molecule ({}), "
               "{} self-symmetry operation{}\n",
               n, bodies[0].m, heavy_only ? "heavy only" : "all atoms",
               bodies[0].n_self_ops(), bodies[0].n_self_ops() == 1 ? "" : "s");
    logging::info("  cluster {} neighbours, tolerance {:.2f} A, radius {:.1f} A, inversion {}\n",
               opts.n_neighbours, opts.tolerance, opts.radius,
               opts.allow_inversion ? "allowed" : "not allowed");

    CompackOptions copts;
    copts.n_neighbours = opts.n_neighbours;
    copts.match_tolerance = opts.tolerance;
    copts.heavy_only = heavy_only;
    copts.allow_inversion = opts.allow_inversion;
    ClusterSpec spec;
    spec.radius = opts.radius;

    std::vector<StructureGeometry> geom(n);
    std::vector<std::vector<double>> tr(n), rv(n);
    for (int i = 0; i < n; ++i) {
        tr[i].assign(rows[i].translation, rows[i].translation + MAX_INDEP * 3);
        rv[i].assign(rows[i].rotation, rows[i].rotation + MAX_INDEP * 3);
        for (int k = 0; k < 6; ++k)
            geom[i].cell_params[k] = (k < 3) ? rows[i].cell_params[k] * B : rows[i].cell_params[k];
        geom[i].space_group = rows[i].space_group;
        geom[i].n_mols = rows[i].n_mols;
        geom[i].translations = tr[i].data();
        geom[i].rotations = rv[i].data();
    }

    // Union-find over "same packing".
    std::vector<int> parent(n);
    for (int i = 0; i < n; ++i) parent[i] = i;
    std::function<int(int)> find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    std::vector<int> nmatch(static_cast<std::size_t>(n) * n, 0);
    for (int i = 0; i < n; ++i) nmatch[static_cast<std::size_t>(i) * n + i] = opts.n_neighbours;

    int compared = 0, same = 0;
    // Z'=1 with one molecule type: the parallel sweep, clusters built around
    // molecule 0. Otherwise the centre is chosen per structure (serial path).
    bool sweepable = (basis.n_mols == 1);
    for (int i = 0; i < n && sweepable; ++i) if (rows[i].n_mols != 1) sweepable = false;

    if (sweepable) {
        SweepClusters sc;
        sc.n = n;
        sc.n_ref = std::min(candidate_pool_size(copts.n_neighbours), compack::MAX_NEIGH);
        sc.n_pool = std::min(candidate_pool_size(copts.n_neighbours), compack::MAX_POOL);
        sc.centrals.resize(n);
        sc.ref.assign(static_cast<std::size_t>(n) * sc.n_ref, PlacedMolecule{});
        sc.pool.assign(static_cast<std::size_t>(n) * sc.n_pool, PlacedMolecule{});
        sc.ref_count.assign(n, 0);
        sc.untied.assign(n, -1);
        ClusterSpec rs = spec, ps = spec;
        rs.n_neighbours = copts.n_neighbours;
        ps.n_neighbours = sc.n_pool;
        // Build clusters in parallel on the host space: build_cluster allocates
        // std::vectors and cannot run on a device.
        int short_cluster = 0;
        {
            ProgressBar bar("placing", n, opts.progress);
            std::atomic<int> short_count{0};
            std::atomic<long long> placed{0};
            std::mutex bar_mu;
            Kokkos::parallel_for(
                "compack_place", Kokkos::RangePolicy<HostExecSpace>(0, n),
                [&](const int i) {
                    const PackingCluster cr = build_cluster(geom[i], bodies, 0, rs);
                    const PackingCluster cp = build_cluster(geom[i], bodies, 0, ps);
                    if (cr.scored < copts.n_neighbours ||
                        (int)cp.neighbours.size() < sc.n_pool)
                        short_count.fetch_add(1, std::memory_order_relaxed);
                    sc.centrals[i] = cr.central;
                    sc.ref_count[i] = std::min<int>(cr.neighbours.size(), sc.n_ref);
                    sc.untied[i] = cr.untied;
                    for (int k = 0; k < sc.ref_count[i]; ++k)
                        sc.ref[static_cast<std::size_t>(i) * sc.n_ref + k] = cr.neighbours[k];
                    for (int k = 0; k < (int)cp.neighbours.size() && k < sc.n_pool; ++k)
                        sc.pool[static_cast<std::size_t>(i) * sc.n_pool + k] = cp.neighbours[k];
                    // ProgressBar is not thread-safe; updates take the lock.
                    const long long d = placed.fetch_add(1, std::memory_order_relaxed) + 1;
                    std::lock_guard<std::mutex> lk(bar_mu);
                    bar.set(d);
                });
            Kokkos::fence();
            short_cluster = short_count.load();
        }
        if (short_cluster > 0)
            logging::info("  ! {} structure(s) found fewer neighbours than asked for within "
                       "{:.1f} A; raise --radius\n", short_cluster, opts.radius);

        std::vector<int> nm;
        std::vector<unsigned char> issame;
        {
            ProgressBar bar("comparing", static_cast<long long>(n) * (n - 1) / 2, opts.progress);
            compack_sweep(sc, bodies[0], copts, nm, issame,
                          opts.progress ? [&bar](long long d, long long) { bar.set(d); }
                                        : std::function<void(long long, long long)>{});
        }
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                ++compared;
                nmatch[static_cast<std::size_t>(i) * n + j] = nm[static_cast<std::size_t>(i) * n + j];
                nmatch[static_cast<std::size_t>(j) * n + i] = nm[static_cast<std::size_t>(i) * n + j];
                if (issame[static_cast<std::size_t>(i) * n + j]) {
                    ++same;
                    const int a = find(i), b = find(j);
                    if (a != b) parent[a] = b;
                }
            }
    } else {
        // Z' > 1: the cluster centre is chosen per structure. Pairs are compared
        // in parallel, then folded into the union-find serially in (i, j) order,
        // which decides each group's root.
        const long long n_pairs = static_cast<long long>(n) * (n - 1) / 2;
        std::vector<unsigned char> pair_same(static_cast<std::size_t>(n_pairs), 0);
        {
            ProgressBar bar("comparing", n_pairs, opts.progress);
            std::atomic<long long> done{0};
            std::mutex bar_mu;
            Kokkos::parallel_for(
                "compack_pairs_serial_path",
                Kokkos::RangePolicy<HostExecSpace, Kokkos::IndexType<long long>>(0, n_pairs),
                [&](const long long p) {
                    // Unflatten the upper triangle, as the device sweep does.
                    long long rem = p;
                    int i = 0;
                    while (rem >= (n - 1 - i)) { rem -= (n - 1 - i); ++i; }
                    const int j = static_cast<int>(i + 1 + rem);
                    const CompackResult r =
                        compare_structures(geom[i], geom[j], bodies, copts, spec);
                    nmatch[static_cast<std::size_t>(i) * n + j] = r.n_matched;
                    nmatch[static_cast<std::size_t>(j) * n + i] = r.n_matched;
                    pair_same[static_cast<std::size_t>(p)] =
                        (r.valid && r.same_packing) ? 1 : 0;
                    const long long d = done.fetch_add(1, std::memory_order_relaxed) + 1;
                    std::lock_guard<std::mutex> lk(bar_mu);
                    bar.set(d);
                });
            Kokkos::fence();
        }
        long long p = 0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j, ++p) {
                ++compared;
                if (pair_same[static_cast<std::size_t>(p)]) {
                    ++same;
                    const int a = find(i), b = find(j);
                    if (a != b) parent[a] = b;
                }
            }
    }

    std::map<int, std::vector<int>> groups;
    for (int i = 0; i < n; ++i) groups[find(i)].push_back(i);
    logging::info("\n{} pairs compared, {} the same packing -> {} distinct packing{}\n",
               compared, same, (int)groups.size(), groups.size() == 1 ? "" : "s");

    int shown = 0;
    for (const auto& kv : groups) {
        const std::vector<int>& members = kv.second;
        if (members.size() < 2) continue;
        if (shown++ == 0)
            logging::info("\nStructures the descriptors kept apart but the packing does not:\n");
        logging::info("  {} structures:", members.size());
        for (int i : members)
            logging::info(" sg{}/seed{}({:.2f})", rows[i].space_group, rows[i].seed, rows[i].energy);
        logging::info("\n");
    }
    if (shown == 0) logging::info("\nEvery structure is a distinct packing.\n");

    if (opts.matrix) {
        logging::info("\nn matched out of {}:\n", opts.n_neighbours);
        for (int i = 0; i < n; ++i) {
            logging::info("  ");
            for (int j = 0; j < n; ++j)
                logging::info("{:3d}", nmatch[static_cast<std::size_t>(i) * n + j]);
            logging::info("   sg{}/seed{}\n", rows[i].space_group, rows[i].seed);
        }
    }
}

// --- packing similarity over a set of CIF files ------------------------------

void run_compack_explicit(const std::vector<CifStructure>& structures,
                          const std::vector<std::string>& labels_in,
                          const CompackRunOptions& opts) {
    if (structures.empty()) {
        logging::info("compack: nothing to compare\n");
        return;
    }
    // The reference molecule, from the first structure that has one; everything
    // else is fitted to it.
    const CifStructure* first = nullptr;
    std::size_t first_index = 0;
    for (std::size_t i = 0; i < structures.size(); ++i)
        if (!structures[i].molecules.empty() && !structures[i].molecules[0].periodic) {
            first = &structures[i];
            first_index = i;
            break;
        }
    if (!first) {
        logging::info("compack: no structure holds a discrete molecule\n");
        return;
    }
    const CifMolecule& ref = first->molecules[0];
    const bool heavy_only =
        opts.heavy_only && heavy_atoms_suffice(ref.positions.data(), ref.numbers.data(), ref.size());
    if (heavy_only != opts.heavy_only)
        logging::info("  (the heavy atoms lie on a line and cannot fix the molecule's "
                      "orientation, so hydrogens are compared too)\n");
    const RigidBody body = body_from_molecule(ref, heavy_only, opts.self_symmetry_tolerance);
    if (body.m == 0) {
        logging::info("compack: the reference molecule has no atoms to compare\n");
        return;
    }

    const std::string ref_label =
        (first_index < labels_in.size() && !labels_in[first_index].empty())
            ? labels_in[first_index]
            : (first->name.empty() ? first->source : first->name);
    logging::info("compack: {} structure(s), reference {}, {} atoms per molecule ({}), "
               "{} self-symmetry operation{}\n",
               structures.size(), ref_label,
               body.m, heavy_only ? "heavy only" : "all atoms",
               body.n_self_ops(), body.n_self_ops() == 1 ? "" : "s");
    logging::info("  cluster {} neighbours, tolerance {:.2f} A, radius {:.1f} A, inversion {}\n",
               opts.n_neighbours, opts.tolerance, opts.radius,
               opts.allow_inversion ? "allowed" : "not allowed");

    CompackOptions copts;
    copts.n_neighbours = opts.n_neighbours;
    copts.match_tolerance = opts.tolerance;
    copts.heavy_only = heavy_only;
    copts.allow_inversion = opts.allow_inversion;
    ClusterSpec spec;
    spec.radius = opts.radius;

    // Place everything. A structure that does not fit the reference is dropped
    // and its reason reported.
    std::vector<CifPacking> packings;
    std::vector<std::string> labels;
    double worst_fit = 0.0;
    {
        ProgressBar bar("placing", static_cast<long long>(structures.size()), opts.progress);
        // Placed in parallel, collected serially in input order so the output
        // does not depend on thread count.
        const int n_in = static_cast<int>(structures.size());
        std::vector<CifPacking> placed(structures.size());
        std::atomic<long long> done{0};
        std::mutex bar_mu;
        Kokkos::parallel_for(
            "compack_place_cifs", Kokkos::RangePolicy<HostExecSpace>(0, n_in),
            [&](const int i) {
                placed[i] = place_cif_structure(structures[i], body, heavy_only,
                                                opts.max_fit_rmsd);
                if (placed[i].valid) find_representatives(placed[i], body, copts, spec);
                const long long d = done.fetch_add(1, std::memory_order_relaxed) + 1;
                std::lock_guard<std::mutex> lk(bar_mu);
                bar.set(d);
            });
        Kokkos::fence();
        for (std::size_t i = 0; i < structures.size(); ++i) {
            const CifStructure& s = structures[i];
            const std::string label = (i < labels_in.size() && !labels_in[i].empty())
                                    ? labels_in[i]
                                    : (s.name.empty() ? s.source : s.name);
            if (!placed[i].valid) {
                logging::info("  ! skipped {}: {}\n", label, placed[i].problem);
                continue;
            }
            worst_fit = std::max(worst_fit, placed[i].worst_fit);
            labels.push_back(label);
            packings.push_back(std::move(placed[i]));
        }
    }
    group_and_report(packings, labels, body, opts, copts, spec, worst_fit);
}

void run_compack_cifs(const std::vector<std::string>& paths, const CompackRunOptions& opts) {
    CifReadOptions ropts;
    ropts.bond_tolerance = opts.bond_tolerance;
    ropts.merge_tolerance = opts.merge_tolerance;
    ropts.min_occupancy = opts.min_occupancy;
    std::vector<CifStructure> structures;
    std::vector<std::string> labels;
    {
        ProgressBar bar("reading", static_cast<long long>(paths.size()), opts.progress);
        long long done = 0;
        for (const std::string& path : paths) {
            std::string err;
            std::vector<CifStructure> read = read_cif(path, ropts, &err);
            bar.set(++done);
            if (read.empty()) {
                logging::info("  ! {}\n", err.empty() ? path + ": no crystal" : err);
                continue;
            }
            for (CifStructure& s : read) {
                labels.push_back(s.name.empty() ? s.source : s.name);
                structures.push_back(std::move(s));
            }
        }
    }
    if (structures.empty()) {
        logging::info("compack: nothing readable in {} file(s)\n", paths.size());
        return;
    }
    run_compack_explicit(structures, labels, opts);
}

// Compare placed packings pairwise, group them, and print the result. Shared by
// the CIF-file and explicit-atom landscape paths.
void group_and_report(const std::vector<CifPacking>& packings,
                      const std::vector<std::string>& labels, const RigidBody& body,
                      const CompackRunOptions& opts, const CompackOptions& copts,
                      const ClusterSpec& spec, double worst_fit) {
    const int n = static_cast<int>(packings.size());
    if (n < 2) {
        logging::info("compack: {} structure(s) placed; need at least two to compare\n", n);
        return;
    }
    // Near zero means the molecule is rigid across the set; a large value means
    // conformers are being overlaid.
    logging::info("  worst molecular fit to the reference: {:.3f} A\n", worst_fit);

    std::vector<int> parent(n);
    for (int i = 0; i < n; ++i) parent[i] = i;
    std::function<int(int)> find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    std::vector<int> nmatch;
    int compared = 0, same = 0;
    {
        std::vector<unsigned char> issame;
        ProgressBar bar("comparing", static_cast<long long>(n) * (n - 1) / 2, opts.progress);
        const int short_cluster = compare_all_packings(
            packings, body, copts, spec, nmatch, issame,
            opts.progress ? [&bar](long long d, long long) { bar.set(d); }
                          : std::function<void(long long, long long)>{});
        if (short_cluster > 0)
            logging::info("  ! {} structure(s) found fewer neighbours than asked for within "
                       "{:.1f} A; raise --radius\n", short_cluster, opts.radius);
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                ++compared;
                if (issame[static_cast<std::size_t>(i) * n + j]) {
                    ++same;
                    const int a = find(i), b = find(j);
                    if (a != b) parent[a] = b;
                }
            }
    }

    std::map<int, std::vector<int>> groups;
    for (int i = 0; i < n; ++i) groups[find(i)].push_back(i);
    logging::info("\n{} pairs compared, {} the same packing -> {} distinct packing{}\n",
               compared, same, (int)groups.size(), groups.size() == 1 ? "" : "s");

    int shown = 0;
    for (const auto& kv : groups) {
        const std::vector<int>& members = kv.second;
        if (members.size() < 2) continue;
        if (shown++ == 0) logging::info("\nSame packing:\n");
        logging::info("  {} structures:", members.size());
        for (int i : members) logging::info(" {}", labels[i]);
        logging::info("\n");
    }
    if (shown == 0) logging::info("\nEvery structure is a distinct packing.\n");

    if (opts.matrix) {
        logging::info("\nn matched out of {}:\n", opts.n_neighbours);
        for (int i = 0; i < n; ++i) {
            logging::info("  ");
            for (int j = 0; j < n; ++j)
                logging::info("{:3d}", nmatch[static_cast<std::size_t>(i) * n + j]);
            logging::info("   {}\n", labels[i]);
        }
    }
}

// --- convergence statistics -------------------------------------------------
namespace {

// One distinct minimum, and how many local searches reached it.
struct Basin { double energy; long long found; };

struct WindowStats {
    double window = 0;
    long long w = 0, N = 0, n1 = 0, n2 = 0;
    double brk = 0, chao1 = 0, cov_brk = 0, cov_gt = 0;
    bool stop = false;
    // False when the searches barely outnumber the basins (no re-finds): BRK is
    // then undefined. Good-Turing is printed either way.
    bool informative = false;
};

// `count` basins, lowest energy first, are inside the window.
static WindowStats window_stats(const std::vector<Basin>& basins, std::size_t count,
                                double window) {
    WindowStats s;
    s.window = window;
    s.w = static_cast<long long>(count);
    for (std::size_t i = 0; i < count; ++i) {
        s.N += basins[i].found;
        if (basins[i].found == 1) ++s.n1;
        else if (basins[i].found == 2) ++s.n2;
    }
    if (s.w == 0 || s.N <= 1) return s;
    const double W = static_cast<double>(s.w), N = static_cast<double>(s.N);
    s.informative = (N - W - 2 > 0);
    s.cov_gt = 1.0 - static_cast<double>(s.n1) / N;
    // W_hat is undefined as w approaches N.
    if (!s.informative) return s;
    s.brk   = W * (N - 1) / (N - W - 2);
    s.chao1 = W + (s.n2 ? static_cast<double>(s.n1) * static_cast<double>(s.n1)
                              / (2.0 * static_cast<double>(s.n2))
                        : static_cast<double>(s.n1) * static_cast<double>(s.n1 - 1) / 2.0);
    s.cov_brk = (N - W - 1) / (N - 1);
    s.stop    = s.brk < W + 0.5;
    return s;
}

struct LineFit { double slope = 0, r2 = 0; };

static LineFit linfit(const std::vector<double>& x, const std::vector<double>& y) {
    LineFit f;
    const std::size_t n = x.size();
    if (n < 3) return f;
    double mx = 0, my = 0;
    for (std::size_t i = 0; i < n; ++i) { mx += x[i]; my += y[i]; }
    mx /= static_cast<double>(n); my /= static_cast<double>(n);
    double sxx = 0, sxy = 0, syy = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = x[i] - mx, dy = y[i] - my;
        sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
    }
    if (sxx <= 0 || syy <= 0) return f;
    f.slope = sxy / sxx;
    const double a = my - f.slope * mx;
    double ss = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double r = y[i] - (a + f.slope * x[i]);
        ss += r * r;
    }
    f.r2 = 1.0 - ss / syy;
    return f;
}

struct TailFit {
    bool ok = false;
    double endpoint = 0, alpha = 0, r2 = 0;
    bool at_boundary = false;   // the scan hit its floor: no interior optimum
};

// Fit N(<E) ~ (E - E*)^alpha to the k lowest energies by scanning E*, over a
// range set by the tail's spread; hitting the floor is reported. `cum`, when
// given, is the count of minima at or below each energy (e.g. Horvitz-Thompson
// weighted); empty means use the rank.
static TailFit fit_weibull_tail(const std::vector<double>& E, std::size_t k,
                                const std::vector<double>& cum = {}) {
    TailFit f;
    if (k < 4 || k > E.size()) return f;
    if (!cum.empty() && cum.size() < k) return f;
    const double spread = E[k - 1] - E[0];
    if (!(spread > 0)) return f;
    const double span = 10.0 * spread;
    std::vector<double> logn(k);
    for (std::size_t i = 0; i < k; ++i)
        logn[i] = std::log(cum.empty() ? static_cast<double>(i + 1) : cum[i]);

    auto score = [&](double Es, LineFit& out) {
        std::vector<double> x(k);
        for (std::size_t i = 0; i < k; ++i) {
            const double d = E[i] - Es;
            if (!(d > 0)) return false;
            x[i] = std::log(d);
        }
        out = linfit(x, logn);
        return true;
    };

    const int steps = 2000;
    double best_es = 0, best_r2 = -1, best_alpha = 0;
    for (int j = 1; j <= steps; ++j) {
        const double Es = E[0] - span * static_cast<double>(j) / steps;
        LineFit lf;
        if (!score(Es, lf)) continue;
        if (lf.r2 > best_r2) { best_r2 = lf.r2; best_es = Es; best_alpha = lf.slope; }
    }
    if (best_r2 < 0) return f;
    // Refine around the winner at a hundredth of the coarse step.
    const double coarse = span / steps;
    for (int j = -100; j <= 100; ++j) {
        const double Es = best_es + coarse * static_cast<double>(j) / 100.0;
        if (Es >= E[0]) continue;
        LineFit lf;
        if (!score(Es, lf)) continue;
        if (lf.r2 > best_r2) { best_r2 = lf.r2; best_es = Es; best_alpha = lf.slope; }
    }
    f.ok = true;
    f.endpoint = best_es;
    f.alpha = best_alpha;
    f.r2 = best_r2;
    f.at_boundary = (E[0] - best_es) > 0.98 * span;
    return f;
}

// The competing model: log N(<E) linear in E, an exponentially growing density
// of minima with no finite endpoint.
static double gumbel_r2(const std::vector<double>& E, std::size_t k) {
    if (k < 4 || k > E.size()) return 0;
    std::vector<double> x(E.begin(), E.begin() + static_cast<std::ptrdiff_t>(k)), logn(k);
    for (std::size_t i = 0; i < k; ++i) logn[i] = std::log(static_cast<double>(i + 1));
    return linfit(x, logn).r2;
}

}  // namespace

ConvergedResult cmd_converged(const ConvergedOptions& o) {
    const std::string& db_path = o.db_path;
    const std::vector<double>& windows = o.windows;
    const int tail_k = o.tail_k;
    const int n_bootstrap = o.n_bootstrap;
    const double energy_lo = o.energy_lo;
    const double energy_hi = o.energy_hi;
    const unsigned bootstrap_seed = o.bootstrap_seed;
    const int stage = o.stage;
    ConvergedResult result;

    std::vector<std::string> db_files;
    if (std::filesystem::is_directory(db_path)) {
        for (auto& entry : std::filesystem::directory_iterator(db_path)) {
            auto p = entry.path();
            if (p.extension() == ".db" && p.filename().string().find("csp") != std::string::npos)
                db_files.push_back(p.string());
        }
        std::sort(db_files.begin(), db_files.end());
    } else {
        db_files.push_back(db_path);
    }
    if (db_files.empty()) { logging::info("No database files in {}\n", db_path); return result; }
    const int read_stage = resolve_stage(db_files, stage, "converged");

    using Key = std::pair<int, int>;                 // (seed, space group)
    std::map<Key, double> representative;            // basin -> its energy
    std::map<Key, Key> parent;                       // duplicate -> what it was merged into
    std::set<Key> seen;

    for (auto& dbf : db_files) {
        sqlite::database db = open_database_readonly(dbf);
        // Pre-v4 databases have no duplicate_of_sg; there a duplicate always
        // names a representative in its own space group.
        const bool has_sg = db_has_column(db, "structure", "duplicate_of_sg");
        const std::string sql =
            "SELECT seed, space_group, unique_structure, energy, "
            "IFNULL(duplicate_of, -1), " +
            std::string(has_sg ? "IFNULL(duplicate_of_sg, space_group)" : "space_group") +
            " FROM structure WHERE stage = " + std::to_string(read_stage);
        db << sql >> [&](int seed, int sg, int uniq, double energy, int dup, int dup_sg) {
            const Key key{seed, sg};
            if (!seen.insert(key).second) return;    // same row from another rank's database
            if (uniq) representative.emplace(key, energy);
            else if (dup >= 0) parent.emplace(key, Key{dup, dup_sg});
        };
    }

    if (representative.empty()) {
        logging::info("No structures at stage {} in {}\n", read_stage, db_path);
        return result;
    }

    // Follow duplicate chains to their root; pre-v4 databases may chain.
    std::map<Key, long long> found;
    for (const auto& [key, energy] : representative) { (void)energy; found[key] = 1; }
    long long chained = 0, orphaned = 0;
    for (const auto& [start, first] : parent) {
        Key k = first;
        std::set<Key> walked{start};
        int hops = 0;
        while (representative.find(k) == representative.end()) {
            if (!walked.insert(k).second) { k = Key{-1, -1}; break; }   // cycle
            auto it = parent.find(k);
            if (it == parent.end()) { k = Key{-1, -1}; break; }         // dangles
            k = it->second;
            ++hops;
        }
        if (k.first < 0) { ++orphaned; continue; }
        if (hops > 0) ++chained;
        ++found[k];
    }

    // Drop non-physical representatives (and their duplicates) before counting.
    std::vector<Basin> basins;
    basins.reserve(representative.size());
    long long dropped_basins = 0, dropped_searches = 0;
    for (const auto& [key, energy] : representative) {
        if (!std::isfinite(energy) || energy < energy_lo || energy > energy_hi) {
            ++dropped_basins;
            dropped_searches += found[key];
            continue;
        }
        basins.push_back({energy, found[key]});
    }
    if (basins.empty()) {
        logging::info("Every basin was outside [{}, {}] kJ/mol -- nothing to report.\n",
                      energy_lo, energy_hi);
        return result;
    }
    std::sort(basins.begin(), basins.end(),
              [](const Basin& a, const Basin& b) { return a.energy < b.energy; });

    long long total_searches = 0;
    for (const auto& b : basins) total_searches += b.found;

    result.ok = true;
    result.basins = static_cast<long long>(basins.size());
    result.searches = total_searches;
    result.min_energy = basins.front().energy;

    logging::info("=== klasp landscape convergence ({} DB file{}, stage {}) ===\n",
                  db_files.size(), db_files.size() > 1 ? "s" : "", read_stage);
    logging::info("  {} distinct minima from {} local searches ({:.2f}% unique)\n",
                  result.basins, result.searches,
                  100.0 * static_cast<double>(result.basins)
                        / static_cast<double>(result.searches));
    logging::info("  lowest energy found: {:.4f} kJ/mol per molecule\n", result.min_energy);
    if (chained)
        logging::info("  {} duplicate pointers were chains, resolved to their root\n", chained);
    if (orphaned)
        logging::info("  {} duplicate pointers dangle or cycle and were not counted\n", orphaned);
    if (dropped_basins)
        logging::info("  {} basins outside [{:g}, {:g}] kJ/mol dropped, taking {} searches\n",
                      dropped_basins, energy_lo, energy_hi, dropped_searches);

    // Warn when the input is unfit for re-find statistics.
    const double unique_rate = static_cast<double>(result.basins)
                             / static_cast<double>(result.searches);
    if (unique_rate > 0.5)
        logging::info("\n  WARNING: {:.0f}% of searches produced a distinct minimum. A landscape\n"
                      "  deduplicated at a sane tolerance re-finds its basins; this one barely\n"
                      "  does, which usually means the deduplication did not run or ran far too\n"
                      "  tight. Every estimate below is computed from re-find counts and none of\n"
                      "  them means anything until that is explained.\n", 100.0 * unique_rate);

    if (basins.size() > 20) {
        const double first_gap = basins[1].energy - basins[0].energy;
        std::vector<double> gaps;
        for (std::size_t i = 1; i < 21; ++i) gaps.push_back(basins[i].energy - basins[i - 1].energy);
        std::sort(gaps.begin(), gaps.end());
        const double median_gap = gaps[gaps.size() / 2];
        if (median_gap > 0 && first_gap > 10 * median_gap)
            logging::info("\n  WARNING: the lowest minimum sits {:.3f} kJ/mol below the next one,\n"
                          "  against a median spacing of {:.3f} among the twenty lowest. It is an\n"
                          "  outlier rather than the bottom of a distribution -- check it is a real\n"
                          "  structure before reading the endpoint estimate, which it anchors.\n",
                          first_gap, median_gap);
    }

    // --- how much of the landscape has been seen -----------------------------
    logging::info("\n  Basins and coverage, by energy window above the minimum.\n"
                  "  BRK and Chao1 estimate how many basins the window holds; coverage is the\n"
                  "  fraction of the volume draining into it that has been sampled. `stop` is\n"
                  "  Boender-Rinnooy Kan's rule -- yes means less than one basin left to find.\n\n");
    logging::info("  {:>8s}  {:>7s}  {:>9s}  {:>6s}  {:>5s}  {:>10s}  {:>10s}  {:>8s}  {:>7s}  {:>5s}\n",
                  "window", "found", "searches", "once", "twice", "BRK", "Chao1",
                  "cov BRK", "cov G-T", "stop");
    std::vector<double> ws(windows);
    std::sort(ws.begin(), ws.end());
    ws.push_back(std::numeric_limits<double>::infinity());
    const double emin = basins.front().energy;
    for (double win : ws) {
        std::size_t count = 0;
        while (count < basins.size() && basins[count].energy <= emin + win) ++count;
        if (count == 0) continue;
        const WindowStats s = window_stats(basins, count, win);
        const std::string label = std::isinf(win) ? "all" : fmt::format("<{:g}", win);
        const std::string brk   = s.informative ? fmt::format("{:.4g}", s.brk) : "-";
        const std::string chao  = s.informative ? fmt::format("{:.4g}", s.chao1) : "-";
        const std::string cbrk  = s.informative ? fmt::format("{:.3f}", s.cov_brk) : "-";
        logging::info("  {:>8s}  {:>7d}  {:>9d}  {:>6d}  {:>5d}  {:>10s}  {:>10s}  "
                      "{:>8s}  {:>7.3f}  {:>5s}\n",
                      label, s.w, s.N, s.n1, s.n2, brk, chao,
                      cbrk, s.cov_gt, s.informative ? (s.stop ? "yes" : "no") : "-");
        if (s.stop && result.converged_window < 0 && !std::isinf(win))
            result.converged_window = win;
    }
    if (result.converged_window >= 0)
        logging::info("\n  The bottom {:g} kJ/mol is converged by the stopping rule.\n",
                      result.converged_window);
    else
        logging::info("\n  No window satisfies the stopping rule: keep searching.\n");

    // --- how basin volume varies with depth ---------------------------------
    // If deep basins are not the large ones, a deep minimum with a small basin
    // is what the search would miss, and no endpoint estimate can see it.
    {
        std::vector<double> x, y;
        x.reserve(basins.size()); y.reserve(basins.size());
        for (const auto& b : basins) {
            x.push_back(b.energy);
            y.push_back(std::log(static_cast<double>(b.found)));
        }
        const LineFit vd = linfit(x, y);
        // Horvitz-Thompson: a basin of relative volume theta is seen at least
        // once with probability 1 - (1 - theta)^N; summing 1/that gives a
        // prior-free estimate of the basin count.
        double ht = 0;
        const double Nall = static_cast<double>(result.searches);
        for (const auto& b : basins) {
            const double theta = static_cast<double>(b.found) / Nall;
            const double pi = 1.0 - std::pow(1.0 - theta, Nall);
            if (pi > 0) ht += 1.0 / pi;
        }
        logging::info("\n  Basin volume against depth.\n");
        logging::info("    Horvitz-Thompson basin count over the whole landscape: {:.0f}\n", ht);
        logging::info("    log(times found) against energy: slope {:+.4f} per kJ/mol, "
                      "R2 {:.3f}\n", vd.slope, vd.r2);
        if (vd.slope < 0 && vd.r2 > 0.1)
            logging::info("    A basin {:.2f} kJ/mol deeper is found e times more often. Deeper\n"
                          "    basins are larger, so what is still unseen is biased towards higher\n"
                          "    energy, and the endpoint below is the conservative reading.\n",
                          1.0 / -vd.slope);
        else if (vd.slope < 0)
            logging::info("    Deeper basins are found more often, but weakly -- the relation\n"
                          "    explains little of the scatter. It leans the right way for the\n"
                          "    endpoint below to be conservative; it does not establish it.\n");
        else
            logging::info("    Depth does not predict how often a basin is found, or predicts it\n"
                          "    backwards. Unseen basins could be at any energy, so a deep minimum\n"
                          "    in a small basin is precisely what this search would have missed --\n"
                          "    and no endpoint estimator can see it, because it is not in the\n"
                          "    sample. Read the endpoint below as a statement about what has been\n"
                          "    sampled, not about what exists.\n");
    }

    // --- how much lower the minimum could be ---------------------------------
    std::vector<double> E;
    E.reserve(basins.size());
    for (const auto& b : basins) E.push_back(b.energy);

    const std::size_t k = std::min<std::size_t>(
        tail_k > 0 ? static_cast<std::size_t>(tail_k) : E.size() / 20, E.size());

    if (E.size() < 4 || k < 4) {
        logging::info("\n  Too few distinct minima ({}) to estimate an endpoint.\n", E.size());
        logging::info("\n");
        return result;
    }
    logging::info("\n  Endpoint estimate for the true minimum, from the {} lowest minima.\n", k);
    {
        logging::info("    best found            {:>10.4f}\n", E[0]);
        // A jackknife above the lowest energy means the gaps are widening;
        // there is nothing to extrapolate to.
        const double jk1 = 2 * E[0] - E[1];
        logging::info("    jackknife, 1st order  {:>10.4f}   (2E1 - E2){}\n", jk1,
                      jk1 > E[0] ? "   above the best found -- see below" : "");
        if (E.size() >= 3) {
            const double jk2 = 3 * E[0] - 3 * E[1] + E[2];
            logging::info("    jackknife, 2nd order  {:>10.4f}   (3E1 - 3E2 + E3){}\n", jk2,
                          jk2 > E[0] ? "   above the best found -- see below" : "");
            if (jk1 > E[0] || jk2 > E[0])
                logging::info("      (a jackknife above the best found means the spacings "
                              "between the lowest\n       minima widen rather than close, and "
                              "the estimator has nothing to extrapolate)\n");
        }

        // Sweep k: the spread across k is an uncertainty in its own right,
        // usually larger than the bootstrap interval at any single k.
        logging::info("\n    {:>7s}  {:>10s}  {:>7s}  {:>9s}  {:>9s}\n",
                      "k", "E* fit", "alpha", "Weibull R2", "Gumbel R2");
        std::vector<std::pair<std::size_t, TailFit>> sweep;
        double gumbel_wins = 0;
        int n_tail_lengths = 0;
        for (std::size_t kk : {k / 4, k / 2, k, 2 * k, 4 * k}) {
            if (kk < 4 || kk > E.size()) continue;
            ++n_tail_lengths;
            const TailFit tf = fit_weibull_tail(E, kk);
            if (!tf.ok) continue;
            const double gr2 = gumbel_r2(E, kk);
            logging::info("    {:>7d}  {:>10.4f}  {:>7.2f}  {:>9.5f}  {:>9.5f}{}\n",
                          kk, tf.endpoint, tf.alpha, tf.r2, gr2,
                          tf.at_boundary ? "   (scan floor -- no interior fit)" : "");
            if (!tf.at_boundary) {
                sweep.emplace_back(kk, tf);
                if (gr2 > tf.r2) ++gumbel_wins;
            }
        }

        if (!sweep.empty()) {
            // Point estimate: median across the sweep; the sweep range is
            // reported beside the bootstrap interval.
            std::vector<double> ends;
            for (const auto& [kk, tf] : sweep) { (void)kk; ends.push_back(tf.endpoint); }
            std::sort(ends.begin(), ends.end());
            result.endpoint_ok = true;
            result.endpoint = ends[ends.size() / 2];
            const double k_lo = ends.front(), k_hi = ends.back();
            // Bootstrap at the k whose fit is the median, so the interval and the
            // point estimate describe the same fit.
            std::size_t k_med = sweep.front().first;
            for (const auto& [kk, tf] : sweep)
                if (tf.endpoint == result.endpoint) { k_med = kk; break; }

            logging::info("\n    E* = {:.4f}   ({:.4f} kJ/mol below the lowest energy found)\n",
                          result.endpoint, E[0] - result.endpoint);
            if (sweep.size() > 1)
                logging::info("    across k:  [{:.4f}, {:.4f}]  ({} of {} tail lengths fitted)\n",
                              k_lo, k_hi, sweep.size(), n_tail_lengths);
            else
                logging::info("    Only 1 of {} tail lengths produced an interior fit, so there "
                              "is no\n    stability to check and this number rests on a single "
                              "one. Treat it as a hint.\n", n_tail_lengths);

            if (n_bootstrap > 0) {
                std::mt19937 rng(bootstrap_seed);
                std::uniform_int_distribution<std::size_t> pick(0, E.size() - 1);
                std::vector<double> boots;
                boots.reserve(static_cast<std::size_t>(n_bootstrap));
                std::vector<double> sample(E.size());
                for (int b = 0; b < n_bootstrap; ++b) {
                    for (auto& v : sample) v = E[pick(rng)];
                    std::sort(sample.begin(), sample.end());
                    const TailFit bf = fit_weibull_tail(sample, k_med);
                    if (bf.ok && !bf.at_boundary) boots.push_back(bf.endpoint);
                }
                if (boots.size() >= 20) {
                    std::sort(boots.begin(), boots.end());
                    result.endpoint_lo = boots[static_cast<std::size_t>(0.025 * boots.size())];
                    result.endpoint_hi = boots[static_cast<std::size_t>(0.975 * boots.size()) - 1];
                    logging::info("    bootstrap 95% CI at k = {}:  [{:.4f}, {:.4f}]  "
                                  "({} of {} fits usable)\n",
                                  k_med, result.endpoint_lo, result.endpoint_hi,
                                  boots.size(), n_bootstrap);
                    if (k_hi - k_lo > result.endpoint_hi - result.endpoint_lo)
                        logging::info("    The spread across k is wider than the bootstrap "
                                      "interval: how much tail to fit,\n    not how many minima "
                                      "there are, is what limits this estimate.\n");
                } else {
                    logging::info("    Too few usable bootstrap fits for an interval.\n");
                }
            }
            if (gumbel_wins * 2 > static_cast<double>(sweep.size()))
                logging::info("\n    The no-endpoint model fits better at {:.0f} of {} tail "
                              "lengths.\n    The density of minima grows exponentially rather "
                              "than vanishing at a floor:\n    the minimum will keep drifting "
                              "down as log N, and E* above means nothing.\n",
                              gumbel_wins, sweep.size());
        } else {
            logging::info("\n    No usable endpoint fit: every tail length ran to the scan "
                          "floor.\n    The lowest minima are too sparse, or too far apart, to "
                          "locate a floor below them.\n");
        }
    }

    logging::info("\n");
    return result;
}

} // namespace klasp
