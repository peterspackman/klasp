#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "klasp/run_state.hpp"
#include "klasp/database.hpp"

#include <algorithm>
#include <filesystem>
#include <string>

using namespace klasp;

// A resumed run reads back how much work is left, where the generator sequence
// picks up, and which structures already represent a basin.

namespace {

// Unique scratch path per test so a failed run cannot poison the next one.
struct TempDb {
    std::filesystem::path path;
    explicit TempDb(const char* tag) {
        path = std::filesystem::temp_directory_path() /
               ("klasp_run_state_" + std::string(tag) + ".db");
        std::filesystem::remove(path);
    }
    ~TempDb() { std::filesystem::remove(path); }
    std::string str() const { return path.string(); }
};

StructureResult make_row(int seed, int sg, int stage, double energy, double density) {
    StructureResult r{};
    r.seed = seed;
    r.space_group = sg;
    r.stage = stage;
    r.n_mols = 1;
    r.energy = energy;
    r.density = density;
    for (int k = 0; k < 3; ++k) r.cell_params[k] = 10.0;
    for (int k = 3; k < 6; ++k) r.cell_params[k] = 90.0;
    // Descriptors that vary with the seed.
    for (int k = 0; k < AMD_K; ++k) r.amd[k] = 1.0 + 0.5 * seed + 0.01 * k;
    for (int k = 0; k < POWDER_N_BINS; ++k)
        r.powder[k] = static_cast<float>((k % (3 + seed)) * 0.001);
    return r;
}

void insert(CSPDatabase& db, const StructureResult& r, bool unique) {
    db.insert_result(&r, unique, unique ? -1 : 0);
    std::vector<double> amd(r.amd, r.amd + AMD_K);
    db.insert_amd(r.seed, r.space_group, r.stage, amd);
    db.insert_powder(r.seed, r.space_group, r.stage, r.powder, POWDER_N_BINS);
}

}  // namespace

TEST_CASE("probe_run reports nothing for a directory with no run in it") {
    TempDb t("absent");
    RunState st = probe_run(t.str(), {14, 2}, 2);
    REQUIRE_FALSE(st.exists);
    REQUIRE(st.total_rows == 0);
    // Every requested space group gets an entry.
    REQUIRE(st.per_sg.size() == 2);
    REQUIRE(st.find(14) != nullptr);
    REQUIRE(st.find(14)->max_seed == -1);
}

TEST_CASE("probe_run counts rows, uniques and the seed watermark per space group") {
    TempDb t("counts");
    {
        CSPDatabase db;
        db.open(t.str());
        insert(db, make_row(0, 14, 2, -50.0, 1.2), true);
        insert(db, make_row(7, 14, 2, -49.0, 1.2), false);
        insert(db, make_row(3, 2, 2, -48.0, 1.3), true);
        db.flush();
        db.close();
    }
    RunState st = probe_run(t.str(), {14, 2}, 2);
    REQUIRE(st.exists);
    REQUIRE(st.total_rows == 3);

    const auto* sg14 = st.find(14);
    REQUIRE(sg14 != nullptr);
    CHECK(sg14->rows == 2);
    CHECK(sg14->unique_rows == 1);
    // The watermark is the highest seed the space group ever used, not the
    // highest unique one.
    CHECK(sg14->max_seed == 7);

    const auto* sg2 = st.find(2);
    REQUIRE(sg2 != nullptr);
    CHECK(sg2->rows == 1);
    CHECK(sg2->max_seed == 3);
}

TEST_CASE("the seed watermark covers stages other than the primary one") {
    // Rows at any stage consume a seed, so the watermark counts them all.
    TempDb t("stages");
    {
        CSPDatabase db;
        db.open(t.str());
        insert(db, make_row(4, 14, 2, -50.0, 1.2), true);
        insert(db, make_row(9, 14, 3, -60.0, 1.2), true);   // PET row, no stage 2
        db.flush();
        db.close();
    }
    RunState st = probe_run(t.str(), {14}, 2);
    REQUIRE(st.exists);
    CHECK(st.find(14)->rows == 1);        // structures at the primary stage
    CHECK(st.find(14)->max_seed == 9);    // but seed 9 is spoken for
}

TEST_CASE("load_landscape returns rows in ascending energy with their descriptors") {
    TempDb t("landscape");
    {
        CSPDatabase db;
        db.open(t.str());
        insert(db, make_row(0, 14, 2, -10.0, 1.2), true);
        insert(db, make_row(1, 14, 2, -30.0, 1.3), true);
        insert(db, make_row(2, 14, 2, -20.0, 1.25), false);
        db.flush();
        db.close();
    }
    LandscapeFilter f;
    f.stage = 2;
    Landscape land = load_landscape(t.str(), f);
    REQUIRE(land.rows.size() == 3);
    // Ascending energy, so the first structure seen in a basin is its lowest.
    CHECK(land.rows[0].energy == -30.0);
    CHECK(land.rows[1].energy == -20.0);
    CHECK(land.rows[2].energy == -10.0);
    CHECK(land.has_amd.size() == land.rows.size());
    CHECK(land.n_missing_amd == 0);
    CHECK(land.n_missing_powder == 0);
    CHECK(land.rows[0].amd[0] == 1.5);   // seed 1's descriptor travelled with it
}

TEST_CASE("load_landscape can restrict to the representatives") {
    TempDb t("unique_only");
    {
        CSPDatabase db;
        db.open(t.str());
        insert(db, make_row(0, 14, 2, -10.0, 1.2), true);
        insert(db, make_row(1, 14, 2, -30.0, 1.3), false);
        db.flush();
        db.close();
    }
    LandscapeFilter f;
    f.stage = 2;
    f.unique_only = true;
    Landscape land = load_landscape(t.str(), f);
    REQUIRE(land.rows.size() == 1);
    CHECK(land.rows[0].seed == 0);
}

TEST_CASE("a structure with no stored powder is reported, not silently zero-filled") {
    // Structures without powder rows are kept as unique, not compared against
    // a zero pattern.
    TempDb t("nopowder");
    {
        CSPDatabase db;
        db.open(t.str());
        StructureResult r = make_row(0, 14, 2, -10.0, 1.2);
        db.insert_result(&r, true, -1);
        std::vector<double> amd(r.amd, r.amd + AMD_K);
        db.insert_amd(r.seed, r.space_group, r.stage, amd);   // AMD only
        db.flush();
        db.close();
    }
    LandscapeFilter f;
    f.stage = 2;
    Landscape land = load_landscape(t.str(), f);
    REQUIRE(land.rows.size() == 1);
    CHECK(land.has_amd[0] == 1);
    CHECK(land.has_powder[0] == 0);
    CHECK(land.n_missing_powder == 1);
}

TEST_CASE("seeding the dedup index reproduces the unique count without re-deciding it") {
    // A resumed run starts with the index the previous run finished with, not a
    // reclassification of its representatives.
    DedupConfig cfg;
    cfg.amd_tol = 0.1;
    cfg.use_powder = false;
    StructureDedup dedup(cfg);

    std::vector<StructureResult> reps;
    for (int s = 0; s < 5; ++s) reps.push_back(make_row(s, 14, 2, -50.0 + s, 1.2));
    for (const auto& r : reps) dedup.seed_existing(r);
    CHECK(dedup.n_unique() == 5);

    // A new structure matching one of them is recognised as its duplicate.
    StructureResult again = make_row(2, 14, 2, -48.0 + 1e-9, 1.2);
    again.seed = 99;
    CHECK(dedup.find_duplicate_or_add(again) == 2);
    CHECK(dedup.n_unique() == 5);

    // One that matches nothing still counts as new.
    StructureResult fresh = make_row(50, 14, 2, -10.0, 1.9);
    CHECK(dedup.find_duplicate_or_add(fresh) < 0);
    CHECK(dedup.n_unique() == 6);
}

TEST_CASE("the config fingerprint changes with the settings and the diff names the field") {
    const std::string a = R"({"dedup":{"energy_bin":5.0},"optimizer":{"grad_tol":1e-8}})";
    const std::string b = R"({"dedup":{"energy_bin":3.0},"optimizer":{"grad_tol":1e-8}})";
    CHECK(config_fingerprint(a) == config_fingerprint(a));
    CHECK(config_fingerprint(a) != config_fingerprint(b));

    auto diffs = config_differences(a, b);
    REQUIRE(diffs.size() == 1);
    // A refused resume names the setting that moved.
    CHECK_THAT(diffs[0], Catch::Matchers::ContainsSubstring("dedup.energy_bin"));
    CHECK(config_differences(a, a).empty());
}

TEST_CASE("config_differences survives a database that predates fingerprinting") {
    auto diffs = config_differences("", R"({"a":1})");
    REQUIRE(diffs.size() == 1);
    CHECK_THAT(diffs[0], Catch::Matchers::ContainsSubstring("unreadable"));
}

// --------------------------------------------------------------------------
// The streaming reader must return exactly what load_landscape does, in the
// same order: the order decides which member of a basin is its representative.

namespace {

// Collect a stream into the shape load_landscape returns.
Landscape collect_stream(const std::string& path, const LandscapeFilter& f) {
    Landscape out;
    for_each_landscape_row(path, f, [&](const LandscapeRow& row) {
        out.rows.push_back(row.r);
        out.has_amd.push_back(row.has_amd ? 1 : 0);
        out.has_powder.push_back(row.has_powder ? 1 : 0);
        out.pdd.push_back(row.pdd);
    });
    return out;
}

void require_same_landscape(const Landscape& a, const Landscape& b) {
    REQUIRE(a.rows.size() == b.rows.size());
    for (std::size_t i = 0; i < a.rows.size(); ++i) {
        INFO("row " << i);
        CHECK(a.rows[i].seed == b.rows[i].seed);
        CHECK(a.rows[i].space_group == b.rows[i].space_group);
        CHECK(a.rows[i].stage == b.rows[i].stage);
        CHECK(a.rows[i].energy == b.rows[i].energy);
        CHECK(a.rows[i].density == b.rows[i].density);
        CHECK(a.rows[i].n_mols == b.rows[i].n_mols);
        for (int k = 0; k < 6; ++k)
            CHECK(a.rows[i].cell_params[k] == b.rows[i].cell_params[k]);
        for (int k = 0; k < AMD_K; ++k)
            CHECK(a.rows[i].amd[k] == b.rows[i].amd[k]);
        for (int k = 0; k < POWDER_N_BINS; ++k)
            CHECK(a.rows[i].powder[k] == b.rows[i].powder[k]);
        CHECK(a.has_amd[i] == b.has_amd[i]);
        CHECK(a.has_powder[i] == b.has_powder[i]);
        CHECK(a.pdd[i] == b.pdd[i]);
    }
}

}  // namespace

TEST_CASE("the streaming reader returns what load_landscape returns, in the same order") {
    TempDb t("stream_same");
    {
        CSPDatabase db;
        db.open(t.str());
        // Two space groups, unsorted energies, and a tie (rows 3 and 4) that only
        // the (space_group, seed) tie-break orders.
        insert(db, make_row(0, 14, 2, -10.0, 1.20), true);
        insert(db, make_row(1, 14, 2, -30.0, 1.30), true);
        insert(db, make_row(2, 19, 2, -20.0, 1.25), true);
        insert(db, make_row(3, 19, 2, -25.0, 1.28), true);
        insert(db, make_row(4, 14, 2, -25.0, 1.27), true);
        db.flush();
        db.close();
    }
    LandscapeFilter f;
    f.stage = 2;
    f.want_amd = true;
    f.want_powder = true;

    const Landscape loaded = load_landscape(t.str(), f);
    const Landscape streamed = collect_stream(t.str(), f);

    REQUIRE(loaded.rows.size() == 5);
    require_same_landscape(loaded, streamed);

    // State the order outright too.
    CHECK(streamed.rows[0].energy == -30.0);
    CHECK(streamed.rows[1].seed == 4);    // -25.0, SG 14 sorts before SG 19
    CHECK(streamed.rows[2].seed == 3);    // -25.0, SG 19
    CHECK(streamed.rows[3].energy == -20.0);
    CHECK(streamed.rows[4].energy == -10.0);
}

TEST_CASE("the streaming reader applies the same filters") {
    TempDb t("stream_filter");
    {
        CSPDatabase db;
        db.open(t.str());
        insert(db, make_row(0, 14, 2, -10.0, 1.20), true);
        insert(db, make_row(1, 14, 2, -30.0, 1.30), false);   // duplicate
        insert(db, make_row(2, 19, 2, -20.0, 1.25), true);
        db.flush();
        db.close();
    }

    SECTION("representatives only") {
        LandscapeFilter f;
        f.stage = 2;
        f.unique_only = true;
        require_same_landscape(load_landscape(t.str(), f), collect_stream(t.str(), f));
        CHECK(collect_stream(t.str(), f).rows.size() == 2);
    }
    SECTION("one space group") {
        LandscapeFilter f;
        f.stage = 2;
        f.space_group = 19;
        require_same_landscape(load_landscape(t.str(), f), collect_stream(t.str(), f));
        CHECK(collect_stream(t.str(), f).rows.size() == 1);
    }
    SECTION("an energy ceiling") {
        LandscapeFilter f;
        f.stage = 2;
        f.max_energy = -15.0;
        require_same_landscape(load_landscape(t.str(), f), collect_stream(t.str(), f));
        CHECK(collect_stream(t.str(), f).rows.size() == 2);
    }
}

TEST_CASE("the streaming reader carries the PDD and reports a missing descriptor") {
    TempDb t("stream_pdd");
    const std::vector<double> pdd = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    {
        CSPDatabase db;
        db.open(t.str());
        // Seed 0 gets everything; seed 1 gets no powder and no PDD.
        StructureResult r0 = make_row(0, 14, 2, -30.0, 1.30);
        insert(db, r0, true);
        db.insert_pdd(r0.seed, r0.space_group, r0.stage, pdd.data(), 2, 3);

        StructureResult r1 = make_row(1, 14, 2, -10.0, 1.20);
        db.insert_result(&r1, true, -1);
        std::vector<double> amd(r1.amd, r1.amd + AMD_K);
        db.insert_amd(r1.seed, r1.space_group, r1.stage, amd);
        db.flush();
        db.close();
    }
    LandscapeFilter f;
    f.stage = 2;
    f.want_amd = true;
    f.want_powder = true;
    f.want_pdd = true;

    const Landscape streamed = collect_stream(t.str(), f);
    require_same_landscape(load_landscape(t.str(), f), streamed);

    REQUIRE(streamed.rows.size() == 2);
    CHECK(streamed.rows[0].seed == 0);
    CHECK(streamed.pdd[0] == pdd);
    CHECK(streamed.has_powder[0] == 1);
    // Absent descriptors read as absent, not zero-filled.
    CHECK(streamed.rows[1].seed == 1);
    CHECK(streamed.pdd[1].empty());
    CHECK(streamed.has_powder[1] == 0);
    CHECK(streamed.has_amd[1] == 1);
}

TEST_CASE("the streaming reader reuses its row, so a caller must copy what it keeps") {
    TempDb t("stream_reuse");
    {
        CSPDatabase db;
        db.open(t.str());
        insert(db, make_row(0, 14, 2, -10.0, 1.2), true);
        insert(db, make_row(1, 14, 2, -30.0, 1.3), true);
        db.flush();
        db.close();
    }
    LandscapeFilter f;
    f.stage = 2;
    std::vector<const LandscapeRow*> kept;
    std::vector<int> seeds_when_seen;
    for_each_landscape_row(t.str(), f, [&](const LandscapeRow& row) {
        kept.push_back(&row);
        seeds_when_seen.push_back(row.r.seed);
    });
    REQUIRE(kept.size() == 2);
    CHECK(kept[0] == kept[1]);                  // one row, reused
    CHECK(seeds_when_seen == std::vector<int>{1, 0});   // read in the callback: correct
}

TEST_CASE("reading a database that is not there fails and leaves no file behind") {
    TempDb t("missing");
    LandscapeFilter f;
    CHECK_THROWS_WITH(load_landscape(t.str(), f),
                      Catch::Matchers::ContainsSubstring(t.str()));
    CHECK_FALSE(std::filesystem::exists(t.path));
}

TEST_CASE("a landscape reads while the run that writes it holds a transaction open") {
    TempDb t("live");
    CSPDatabase db;
    db.open(t.str());
    insert(db, make_row(0, 14, 2, -10.0, 1.2), true);
    db.flush();
    insert(db, make_row(1, 14, 2, -30.0, 1.3), true);   // left uncommitted

    LandscapeFilter f;
    f.stage = 2;
    const Landscape land = load_landscape(t.str(), f);
    REQUIRE(land.rows.size() == 1);
    CHECK(land.rows[0].energy == -10.0);
    db.close();
}
