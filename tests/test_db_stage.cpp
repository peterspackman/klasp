#include "kokkos_listener.hpp"

#include <catch2/catch_test_macros.hpp>

#include "klasp/run_state.hpp"
#include "klasp/database.hpp"
#include "klasp/db_tool.hpp"
#include "klasp/io.hpp"

#include <sqlite_modern_cpp.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace klasp;

// Which stage a reader takes out of a database, and what happens to descriptors
// and duplicate references once more than one stage is in there.

namespace {

struct TempDb {
    std::filesystem::path path;
    explicit TempDb(const char* tag) {
        path = std::filesystem::temp_directory_path() /
               ("klasp_db_stage_" + std::string(tag) + ".db");
        std::filesystem::remove(path);
    }
    ~TempDb() { std::filesystem::remove(path); }
    std::string str() const { return path.string(); }
};

StructureResult row(int seed, int sg, int stage, double energy) {
    StructureResult r{};
    r.seed = seed;
    r.space_group = sg;
    r.stage = stage;
    r.n_mols = 1;
    r.energy = energy;
    r.density = 1.2;
    for (int k = 0; k < 3; ++k) r.cell_params[k] = 10.0;
    for (int k = 3; k < 6; ++k) r.cell_params[k] = 90.0;
    for (int k = 0; k < AMD_K; ++k) r.amd[k] = 1.0 + 0.5 * seed + 0.01 * k;
    for (int k = 0; k < POWDER_N_BINS; ++k)
        r.powder[k] = static_cast<float>((k % (3 + seed)) * 0.001);
    return r;
}

// A descriptor whose values say which stage wrote it, so a clobber is visible.
std::vector<double> amd_tagged(int stage) {
    return std::vector<double>(AMD_K, 100.0 + stage);
}

}  // namespace

TEST_CASE("Stage choice on an ordinary single-stage run", "[db][stage]") {
    const auto c = choose_stage({{2, 500}}, 0);
    REQUIRE(c.stage == 2);
    REQUIRE(c.rows == 500);
    REQUIRE(c.coverage() == 1.0);
    REQUIRE_FALSE(c.silently_partial());
    REQUIRE_FALSE(c.slightly_short());
}

TEST_CASE("Stage choice takes the highest stage when it is complete", "[db][stage]") {
    // The ordinary --pet-relax shape: every structure reached stage 3.
    const auto c = choose_stage({{2, 1000}, {3, 1000}}, 0);
    REQUIRE(c.stage == 3);
    REQUIRE_FALSE(c.silently_partial());
    REQUIRE_FALSE(c.slightly_short());
}

TEST_CASE("A PET stage that dropped a few structures is not an error", "[db][stage]") {
    // SGWorker discards candidates with non-finite energy or a skipped symmetry
    // relax, so an ordinary run can finish a few short.
    const auto c = choose_stage({{2, 1000}, {3, 984}}, 0);
    REQUIRE(c.stage == 3);
    REQUIRE(c.slightly_short());          // worth mentioning
    REQUIRE_FALSE(c.silently_partial());  // not worth refusing
    REQUIRE(c.fullest_stage == 2);
    REQUIRE(c.fullest_rows == 1000);
}

TEST_CASE("A top-K rerank stage is refused, not silently read", "[db][stage]") {
    // `klasp rerank --top 100` over a 10000-structure landscape: stage 3 holds
    // only 100 rows.
    const auto c = choose_stage({{2, 10000}, {3, 100}}, 0);
    REQUIRE(c.stage == 3);
    REQUIRE(c.silently_partial());
    REQUIRE(c.fullest_stage == 2);
    REQUIRE(c.coverage() < 0.02);
}

TEST_CASE("Naming a stage overrides the guard", "[db][stage]") {
    SECTION("the full one") {
        const auto c = choose_stage({{2, 10000}, {3, 100}}, 2);
        REQUIRE(c.stage == 2);
        REQUIRE(c.requested);
        REQUIRE_FALSE(c.silently_partial());
    }
    SECTION("the partial one, deliberately") {
        const auto c = choose_stage({{2, 10000}, {3, 100}}, 3);
        REQUIRE(c.stage == 3);
        REQUIRE(c.requested);
        REQUIRE_FALSE(c.silently_partial());   // asked for, so not silent
    }
}

TEST_CASE("A stage with no rows is reported as unknown", "[db][stage]") {
    const auto c = choose_stage({{2, 10000}}, 7);
    REQUIRE(c.unknown);
    REQUIRE_FALSE(c.silently_partial());
}

TEST_CASE("An empty database yields no stage rather than a crash", "[db][stage]") {
    const auto c = choose_stage({}, 0);
    REQUIRE(c.rows == 0);
    REQUIRE_FALSE(c.unknown);
}

TEST_CASE("stage_histogram counts every stage across files", "[db][stage]") {
    TempDb t("hist");
    {
        CSPDatabase db;
        db.open(t.str());
        for (int i = 0; i < 7; ++i) { auto r = row(i, 14, 2, -10.0 - i); db.insert_result(&r, true); }
        for (int i = 0; i < 3; ++i) { auto r = row(i, 14, 3, -20.0 - i); db.insert_result(&r, true); }
        db.close();
    }
    const auto hist = stage_histogram({t.str()});
    REQUIRE(hist.size() == 2);
    REQUIRE(hist.at(2) == 7);
    REQUIRE(hist.at(3) == 3);
}

TEST_CASE("Descriptors are keyed by stage and do not overwrite each other",
          "[db][stage][descriptor]") {
    // The descriptor primary key includes the stage, so a zero-filled stage-0
    // row cannot overwrite the stage-2 descriptors.
    TempDb t("desc");
    {
        CSPDatabase db;
        db.open(t.str());
        for (int stage : {2, 3}) {
            auto r = row(0, 14, stage, -10.0 * stage);
            db.insert_result(&r, true);
            db.insert_amd(r.seed, r.space_group, stage, amd_tagged(stage));
            db.insert_powder(r.seed, r.space_group, stage, r.powder, POWDER_N_BINS);
        }
        db.close();
    }

    LandscapeFilter lf;
    lf.want_powder = false;
    for (int stage : {2, 3}) {
        lf.stage = stage;
        const Landscape land = load_landscape(t.str(), lf);
        REQUIRE(land.rows.size() == 1);
        REQUIRE(land.has_amd[0]);
        // Each stage reads back the descriptor its own row wrote.
        REQUIRE(land.rows[0].amd[0] == 100.0 + stage);
    }
}

TEST_CASE("Descriptors survive a run that ends on a flush boundary",
          "[db][descriptor]") {
    // Descriptor inserts count toward the commit batch (BATCH_SIZE 100), so a
    // run of exactly 100 keeps its last structure's descriptors on close().
    TempDb t("flush");
    constexpr int kRows = 100;
    {
        CSPDatabase db;
        db.open(t.str());
        for (int i = 0; i < kRows; ++i) {
            auto r = row(i, 14, 2, -10.0 - i);
            db.insert_result(&r, true);
            std::vector<double> amd(r.amd, r.amd + AMD_K);
            db.insert_amd(r.seed, r.space_group, r.stage, amd);
            db.insert_powder(r.seed, r.space_group, r.stage, r.powder, POWDER_N_BINS);
        }
        db.close();
    }
    LandscapeFilter lf;
    lf.stage = 2;
    const Landscape land = load_landscape(t.str(), lf);
    REQUIRE(land.rows.size() == kRows);
    REQUIRE(land.n_missing_amd == 0);
    REQUIRE(land.n_missing_powder == 0);
}

TEST_CASE("Missing descriptors are only counted when they were asked for",
          "[db][descriptor]") {
    // With the AMD stage disabled (--amd-tol 0), no structure counts as
    // missing AMD.
    TempDb t("missing");
    {
        CSPDatabase db;
        db.open(t.str());
        for (int i = 0; i < 4; ++i) {
            auto r = row(i, 14, 2, -10.0 - i);
            db.insert_result(&r, true);
            db.insert_powder(r.seed, r.space_group, r.stage, r.powder, POWDER_N_BINS);
            // No AMD written at all.
        }
        db.close();
    }

    SECTION("asked for and absent") {
        LandscapeFilter lf;
        lf.stage = 2;
        lf.want_amd = true;
        const Landscape land = load_landscape(t.str(), lf);
        REQUIRE(land.n_missing_amd == 4);
    }
    SECTION("not asked for") {
        LandscapeFilter lf;
        lf.stage = 2;
        lf.want_amd = false;
        const Landscape land = load_landscape(t.str(), lf);
        REQUIRE(land.n_missing_amd == 0);
    }
}

TEST_CASE("A displacement re-points the whole basin, leaving no chain",
          "[db][duplicate]") {
    // demote_to_duplicate re-points every row referencing the displaced
    // representative, not just the representative itself.
    TempDb t("chain");
    CSPDatabase db;
    db.open(t.str());

    // seed 0 represents the basin; 1 and 2 are duplicates of it.
    auto rep = row(0, 14, 2, -10.0);
    db.insert_result(&rep, true);
    for (int s : {1, 2}) {
        auto d = row(s, 14, 2, -9.0);
        db.insert_result(&d, false, /*duplicate_of=*/0, /*duplicate_of_sg=*/14);
    }

    // seed 3 arrives lower in energy and takes the basin over.
    auto better = row(3, 14, 2, -12.0);
    db.insert_result(&better, true);
    db.demote_to_duplicate(/*old_seed=*/0, /*space_group=*/14, /*stage=*/2,
                           /*new_seed=*/3, /*new_sg=*/14);
    db.flush();
    db.close();

    // Read duplicate_of directly; unique_structure alone would not show it.
    sqlite::database raw(t.str());

    int n_dup = 0;
    raw << "SELECT COUNT(*) FROM structure WHERE duplicate_of IS NOT NULL" >> n_dup;
    REQUIRE(n_dup == 3);           // seeds 0, 1 and 2

    int chained = 0;
    raw << "SELECT COUNT(*) FROM structure d JOIN structure t "
           "  ON t.seed = d.duplicate_of AND t.space_group = d.duplicate_of_sg "
           " AND t.stage = d.stage "
           "WHERE d.duplicate_of IS NOT NULL AND t.unique_structure = 0"
        >> chained;
    REQUIRE(chained == 0);

    int dangling = 0;
    raw << "SELECT COUNT(*) FROM structure d LEFT JOIN structure t "
           "  ON t.seed = d.duplicate_of AND t.space_group = d.duplicate_of_sg "
           " AND t.stage = d.stage "
           "WHERE d.duplicate_of IS NOT NULL AND t.seed IS NULL"
        >> dangling;
    REQUIRE(dangling == 0);

    int not_pointing_at_3 = 0;
    raw << "SELECT COUNT(*) FROM structure "
           "WHERE duplicate_of IS NOT NULL AND duplicate_of != 3"
        >> not_pointing_at_3;
    REQUIRE(not_pointing_at_3 == 0);

    // And the space group travels with the seed, since seeds are per-SG
    // sequences that all start at 0.
    int null_sg = 0;
    raw << "SELECT COUNT(*) FROM structure "
           "WHERE duplicate_of IS NOT NULL AND duplicate_of_sg IS NULL"
        >> null_sg;
    REQUIRE(null_sg == 0);
}

// The stage-3 dedup gate relaxes whichever basin member arrived first on a rank,
// which may not be the final representative, so a finished run can hold
// representatives with no stage-3 energy. The query must find exactly those.
TEST_CASE("the stage-3 top-up set is exactly the representatives without a row",
          "[db][stage][topup]") {
    TempDb t("topup");
    CSPDatabase db;
    db.open(t.str());

    // Four basins in SG 14. Representatives are seeds 0, 2, 4, 6; each has a
    // duplicate. Stage 3 reached basins 0 and 2 via the representative, basin 4
    // via its duplicate (seed 5), and basin 6 not at all.
    for (int b = 0; b < 4; ++b) {
        auto rep = row(2 * b, 14, STAGE_MULTIPOLE, -20.0 + b);
        db.insert_result(&rep, true);
        auto dup = row(2 * b + 1, 14, STAGE_MULTIPOLE, -19.0 + b);
        db.insert_result(&dup, false, /*duplicate_of=*/2 * b, /*duplicate_of_sg=*/14);
    }
    for (int seed : {0, 2, 5}) {
        auto s3 = row(seed, 14, STAGE_FINAL, -30.0);
        db.insert_result(&s3, true);
    }
    // A second space group: SG 15 seed 0 is not SG 14 seed 0, and has no
    // stage-3 row.
    auto other = row(0, 15, STAGE_MULTIPOLE, -25.0);
    db.insert_result(&other, true);
    db.flush();

    // Representatives 4 (relaxed through its duplicate) and 6 (not relaxed),
    // plus SG 15's seed 0. Seeds 0 and 2 have their own rows; the duplicates
    // are not representatives and are not topped up.
    REQUIRE(db.count_unique_missing_stage(STAGE_MULTIPOLE, STAGE_FINAL) == 3);
    db.close();

    auto rows = read_landscape(t.str(), STAGE_MULTIPOLE, /*filter_sg=*/0,
                               /*max_energy=*/1e30, /*unique_only=*/true,
                               /*exclude_catastrophes=*/false,
                               /*catastrophe_threshold=*/-1000.0,
                               /*exclude_warned=*/true,
                               /*only_missing_stage=*/STAGE_FINAL);
    std::vector<std::pair<int, int>> got;
    for (const auto& r : rows) got.push_back({r.space_group, r.seed});
    std::sort(got.begin(), got.end());
    const std::vector<std::pair<int, int>> want{{14, 4}, {14, 6}, {15, 0}};
    REQUIRE(got == want);

    // Without the filter the whole set of representatives comes back.
    auto all = read_landscape(t.str(), STAGE_MULTIPOLE, 0, 1e30, true, false, -1000.0);
    REQUIRE(all.size() == 5);
}

// Explicit-atom geometry blob round trip through the shared decoder used by
// CSPDatabase::read_flexible and the `extract` connection cache.
TEST_CASE("explicit geometry round-trips through the database", "[db][flexible]") {
    TempDb t("flexible");

    FlexibleStructure in;
    in.n_atoms = 4;
    in.cell_params[0] = 7.1; in.cell_params[1] = 8.2; in.cell_params[2] = 9.3;
    in.cell_params[3] = 90.0; in.cell_params[4] = 97.5; in.cell_params[5] = 90.0;
    in.Z = {6, 1, 7, 8};
    // Outside [0,1): molecules are kept whole, so atoms can sit past a cell face.
    in.frac = {0.10, 0.20, 0.30,
               1.15, -0.05, 0.55,
               0.40, 0.50, 0.60,
               0.70, 0.80, 0.90};

    {
        CSPDatabase db;
        db.open(t.str());
        auto row = ::row(7, 14, STAGE_FINAL, -42.0);
        db.insert_result(&row, true);
        db.insert_flexible(7, 14, STAGE_FINAL, in);
        db.flush();
        db.close();
    }

    CSPDatabase db;
    db.open(t.str());
    FlexibleStructure out;
    REQUIRE(db.read_flexible(7, 14, STAGE_FINAL, out));
    CHECK(out.n_atoms == in.n_atoms);
    CHECK(out.Z == in.Z);
    REQUIRE(out.frac.size() == in.frac.size());
    for (std::size_t i = 0; i < in.frac.size(); ++i)
        CHECK(out.frac[i] == in.frac[i]);
    for (int i = 0; i < 6; ++i) CHECK(out.cell_params[i] == in.cell_params[i]);

    // A structure with no stored geometry is absent, not a zero-atom one.
    FlexibleStructure missing;
    CHECK_FALSE(db.read_flexible(999, 14, STAGE_FINAL, missing));
    db.close();
}

// A truncated geometry blob is refused rather than read past its end.
TEST_CASE("a short explicit-geometry blob is refused, not read past",
          "[db][flexible]") {
    FlexibleStructure fs;
    fs.n_atoms = 3;
    fs.Z = {6, 7, 8};
    fs.frac = {0, 0, 0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25};
    const std::vector<char> blob = CSPDatabase::encode_flexible_atoms(fs);
    REQUIRE(blob.size() == 3 * sizeof(std::int32_t) + 9 * sizeof(double));

    FlexibleStructure out;
    CHECK(CSPDatabase::decode_flexible_atoms(3, blob.data(), blob.size(), out));
    CHECK(out.Z == fs.Z);
    // One byte short, one atom too many, and a nonsense count.
    CHECK_FALSE(CSPDatabase::decode_flexible_atoms(3, blob.data(), blob.size() - 1, out));
    CHECK_FALSE(CSPDatabase::decode_flexible_atoms(4, blob.data(), blob.size(), out));
    CHECK_FALSE(CSPDatabase::decode_flexible_atoms(0, blob.data(), blob.size(), out));
}

// demote_to_duplicate searches on duplicate_of; assert the query plan uses the
// index rather than timing it.
TEST_CASE("the displacement re-point is indexed, not a table scan", "[db][dedup]") {
    TempDb tmp("dup_index");
    {
        CSPDatabase db;
        db.open(tmp.str());
        db.close();
    }
    sqlite::database raw(tmp.str());
    std::string plan;
    raw << "EXPLAIN QUERY PLAN UPDATE structure SET duplicate_of=1, duplicate_of_sg=2 "
           "WHERE duplicate_of=3 AND duplicate_of_sg=14 AND stage=2"
        >> [&](int, int, int, std::string detail) { plan += detail; };

    // SQLite infers duplicate_of=? implies IS NOT NULL, so the partial index
    // applies.
    REQUIRE(plan.find("USING INDEX idx_dup") != std::string::npos);
    REQUIRE(plan.find("SCAN structure") == std::string::npos);
}

// A landscape recorded without potentials.damping_range ran at range 1, so it
// must be read back at range 1, not the current default of 1.5.
TEST_CASE("A recorded basis without a damping range is read at range 1",
          "[db][damping]") {
    const auto basis_with = [](const nlohmann::json& potentials) {
        nlohmann::json b;
        b["molecule_types"] = nlohmann::json::array(
            {{{"sites", nlohmann::json::array(
                {{{"element", "H"}, {"position", {0.0, 0.0, 0.0}}}})}}});
        b["potentials"] = potentials;
        return b.dump();
    };
    const auto recorded_range = [](const TempDb& t, const std::string& basis) {
        {
            CSPDatabase db;
            db.open(t.str());
            db.set_meta("basis_json", basis);
            db.close();
        }
        const BasisConfig cfg = load_csp_basis("", t.str());
        return cfg.basis_json["potentials"]["damping_range"].get<double>();
    };

    TempDb old_run("damping_old");
    CHECK(recorded_range(old_run, basis_with({{"damping", true}})) == 1.0);

    TempDb new_run("damping_new");
    CHECK(recorded_range(new_run, basis_with({{"damping", true}, {"damping_range", 1.5}})) == 1.5);
}

// Runs from before the PET stage recorded no "pet" section in config_json.
TEST_CASE("Run metadata reads a config with sections missing", "[db][extract]") {
    TempDb t("meta_missing_sections");
    {
        CSPDatabase db;
        db.open(t.str());
        db.set_meta("config_json", R"({"dedup": {"energy_bin": 5.0}})");
        db.close();
    }
    CspCifMetadata meta;
    REQUIRE_NOTHROW(meta = csp_metadata_for_run(t.str(), 2));
    CHECK(meta.steps.size() == 1);
    CHECK_FALSE(meta.clustering_method.empty());
}

namespace {

FlexibleStructure some_geometry() {
    FlexibleStructure fs;
    fs.n_atoms = 2;
    fs.cell_params[0] = fs.cell_params[1] = fs.cell_params[2] = 10.0;
    fs.cell_params[3] = fs.cell_params[4] = fs.cell_params[5] = 90.0;
    fs.Z = {6, 8};
    fs.frac = {0.1, 0.1, 0.1, 0.2, 0.1, 0.1};
    return fs;
}

void write_reranked(const TempDb& t, bool with_geometry) {
    CSPDatabase db;
    db.open(t.str());
    auto s2 = row(7, 14, 2, -40.0);
    auto s3 = row(7, 14, STAGE_FINAL, -42.0);
    db.insert_result(&s2, true);
    db.insert_result(&s3, true);
    if (with_geometry) db.insert_flexible(7, 14, STAGE_FINAL, some_geometry());
    db.flush();
    db.close();
}

std::string extract_error(const TempDb& t) {
    ExtractOptions o;
    o.db_path = t.str();
    o.filter.stage = STAGE_FINAL;
    o.output_dir = (std::filesystem::temp_directory_path() / "klasp_db_stage_extract").string();
    try { cmd_extract(o); } catch (const std::exception& e) { return e.what(); }
    return "";
}

}  // namespace

TEST_CASE("stage-3 rows without their relaxed geometry are refused, not rebuilt",
          "[db][flexible]") {
    TempDb t("s3_no_geometry");
    write_reranked(t, /*with_geometry=*/false);
    const std::string err = extract_error(t);
    CHECK(err.find("have no relaxed geometry") != std::string::npos);
    CHECK(err.find("klasp compact --drop-descriptors") != std::string::npos);
}

TEST_CASE("stage-3 rows with their relaxed geometry pass the geometry check",
          "[db][flexible]") {
    TempDb t("s3_geometry");
    write_reranked(t, /*with_geometry=*/true);
    CHECK(extract_error(t).find("relaxed geometry") == std::string::npos);
}

TEST_CASE("stage 3 is labelled the ML step even when a rerank, not the run, made it",
          "[db][extract]") {
    TempDb t("meta_rerank");
    {
        CSPDatabase db;
        db.open(t.str());
        db.set_meta("config_json", R"({"pet": {"relax": false}})");
        db.set_meta("stage3_model", "pbe0");
        db.close();
    }
    const CspCifMetadata s3 = csp_metadata_for_run(t.str(), STAGE_FINAL);
    REQUIRE(s3.steps.size() == 2);
    CHECK(s3.steps.back().method == "Machine learning potential");
    CHECK(s3.steps.back().force_field_name == "pbe0");
    CHECK(s3.steps.back().final_step);

    const CspCifMetadata s2 = csp_metadata_for_run(t.str(), 2);
    REQUIRE(s2.steps.size() == 1);
    CHECK(s2.steps[0].final_step);
}

// A rerank relaxes each candidate on its own, so several can reach one well.
// Two seeds here reach the same relaxed geometry and a third reaches another.
TEST_CASE("stage-3 rows are deduplicated by their relaxed geometry", "[db][flexible]") {
    TempDb t("s3_dedup");
    FlexibleStructure same = some_geometry();
    FlexibleStructure other = some_geometry();
    for (int k = 0; k < 3; ++k) other.cell_params[k] = 4.0;
    {
        CSPDatabase db;
        db.open(t.str());
        const std::pair<int, double> s3[3] = {{1, -10.0}, {2, -9.99}, {3, -9.5}};
        for (const auto& [seed, e] : s3) {
            auto s2 = row(seed, 14, 2, e + 5.0);
            auto r3 = row(seed, 14, STAGE_FINAL, e);
            db.insert_result(&s2, true);
            db.insert_result(&r3, true);
            db.insert_flexible(seed, 14, STAGE_FINAL, seed == 3 ? other : same);
        }
        db.flush();
        db.close();
    }

    const FinalStageDedup d = dedup_final_stage(t.str());
    CHECK(d.rows == 3);
    CHECK(d.unique == 2);

    sqlite::database db(t.str());
    std::map<int, std::pair<int, int>> verdict;   // seed -> (unique, duplicate_of)
    db << "SELECT seed, unique_structure, IFNULL(duplicate_of, -1) FROM structure WHERE stage = ?"
       << STAGE_FINAL
       >> [&](int seed, int uniq, int dup) { verdict[seed] = {uniq, dup}; };
    CHECK(verdict[1] == std::make_pair(1, -1));
    CHECK(verdict[2] == std::make_pair(0, 1));
    CHECK(verdict[3] == std::make_pair(1, -1));
}
