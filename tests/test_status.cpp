#include <limits>
// The status table (COLUMNS in status.cpp) has three renderers. These check
// that they agree, and that the total row blanks columns whose sums mean nothing.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "klasp/status.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <atomic>
#include <unistd.h>
#include <chrono>
#include <thread>

using namespace klasp;
using SGStats = StatusReporter::SGStats;
using Catch::Matchers::ContainsSubstring;

namespace {

std::string status_text(const std::string& dir) {
    std::ifstream f(dir + "/status.txt");
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string line_starting(const std::string& text, const std::string& label) {
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string first;
        if (ls >> first && first == label) return line;
    }
    return {};
}

std::string trim(std::string v) {
    const auto b = v.find_first_not_of(" \t");
    if (b == std::string::npos) return {};
    return v.substr(b, v.find_last_not_of(" \t") - b + 1);
}

// Columns are fixed-width and a cell can be blank, so slice at the header
// tokens' end positions rather than splitting on whitespace.
std::vector<size_t> field_ends(const std::string& header) {
    std::vector<size_t> ends;
    size_t i = 0;
    while (i < header.size()) {
        while (i < header.size() && header[i] == ' ') ++i;
        if (i >= header.size()) break;
        size_t j = header.find(' ', i);
        if (j == std::string::npos) j = header.size();
        ends.push_back(j);
        i = j;
    }
    return ends;
}

std::vector<std::string> cells(const std::string& row, const std::vector<size_t>& ends) {
    std::vector<std::string> out;
    size_t prev = 0;
    for (size_t e : ends) {
        const size_t end = std::min(e, row.size());
        out.push_back(prev <= end ? trim(row.substr(prev, end - prev)) : std::string{});
        prev = end;
    }
    return out;
}

// Unique per process and per instance: ctest -j runs these cases concurrently.
struct TempDir {
    std::filesystem::path p;
    TempDir() {
        static std::atomic<int> counter{0};
        p = std::filesystem::temp_directory_path() /
            ("klasp_status_test_" + std::to_string(::getpid()) + "_" +
             std::to_string(counter.fetch_add(1)));
        std::filesystem::remove_all(p);
        std::filesystem::create_directories(p);
    }
    ~TempDir() { std::filesystem::remove_all(p); }
};

}  // namespace

TEST_CASE("status table reports convergence diagnostics per space group",
          "[status]") {
    TempDir tmp;
    StatusReporter st(tmp.p.string(), 1, {14, 15}, {100, 100});

    // SG14: 4 structures, 3 converged at 10/20/30 iterations.
    st.record_diagnostics(14, 10, 1e-6);
    st.record_diagnostics(14, 20, 1e-6);
    st.record_diagnostics(14, 30, 1e-6);
    st.record_diagnostics(14, -1, 1e-2);          // never reached tolerance
    st.set_total(14, 4); st.set_unique(14, 2);
    // Stage-1 convergence is carried on the same records: 2 of the 4 finished,
    // at 12 and 30 iterations.
    st.record_stage1(14, 12, 1e-7);
    st.record_stage1(14, 30, 1e-7);
    st.record_stage1(14, -1, 1e-3);
    st.record_stage1(14, -1, 1e-3);
    // SG15: 1 structure, converged at 50.
    st.record_diagnostics(15, 50, 1e-7);
    st.set_total(15, 1); st.set_unique(15, 1);
    st.finalize();

    const std::string text = status_text(tmp.p.string());
    REQUIRE_THAT(text, ContainsSubstring("conv"));
    REQUIRE_THAT(text, ContainsSubstring("iters"));

    // Header positions fix the field boundaries for the rows below.
    const std::string hline = line_starting(text, "SG");
    const auto ends = field_ends(hline);
    const auto hdr = cells(hline, ends);
    REQUIRE(hdr.size() > 1);
    auto col = [&](const std::vector<std::string>& r, const char* name) {
        for (size_t i = 0; i < hdr.size(); ++i)
            if (hdr[i] == name) return r.at(i);
        FAIL("no such column: " << name);
        return std::string{};
    };

    const auto r14 = cells(line_starting(text, "14"), ends);
    REQUIRE(r14.size() == hdr.size());
    REQUIRE(col(r14, "conv") == "75.0%");        // 3 of 4
    REQUIRE(col(r14, "iters") == "20");          // mean over the 3 converged
    REQUIRE(col(r14, "s1conv") == "50.0%");      // 2 of 4 finished stage 1
    REQUIRE(col(r14, "s1iters") == "21");        // mean of 12 and 30, converged only

    const auto r15 = cells(line_starting(text, "15"), ends);
    REQUIRE(col(r15, "conv") == "100.0%");
    REQUIRE(col(r15, "iters") == "50");
}

TEST_CASE("status total row aggregates only the columns whose sum means something",
          "[status]") {
    TempDir tmp;
    StatusReporter st(tmp.p.string(), 1, {14, 15}, {10, 10});
    st.record_diagnostics(14, 10, 1e-6);
    st.record_diagnostics(14, -1, 1e-2);
    st.record_diagnostics(15, 30, 1e-6);
    st.set_total(14, 2); st.set_unique(14, 1);
    st.set_total(15, 1); st.set_unique(15, 1);
    st.finalize();

    const std::string text = status_text(tmp.p.string());
    const std::string hline = line_starting(text, "SG");
    const auto ends = field_ends(hline);
    const auto hdr = cells(hline, ends);
    const auto all = cells(line_starting(text, "ALL"), ends);
    REQUIRE(all.size() == hdr.size());
    auto col = [&](const char* name) {
        for (size_t i = 0; i < hdr.size(); ++i)
            if (hdr[i] == name) return all.at(i);
        FAIL("no such column: " << name);
        return std::string{};
    };
    REQUIRE(col("total") == "3");
    REQUIRE(col("unique") == "2");
    REQUIRE(col("target") == "20");
    // 2 converged of 3 with diagnostics; mean iters over the converged pair.
    REQUIRE(col("conv") == "66.7%");
    REQUIRE(col("iters") == "20");
}

TEST_CASE("one blown-up structure does not set the whole grad column",
          "[status]") {
    TempDir tmp;
    StatusReporter st(tmp.p.string(), 1, {5, 14}, {100, 100});

    // SG 5: 99 structures converged around 1e-6, and one catastrophe.
    for (int i = 0; i < 99; ++i) st.record_diagnostics(5, 30, 1.0e-6);
    st.record_diagnostics(5, -1, 4.7e+08);
    st.set_total(5, 100); st.set_unique(5, 10);
    // SG 14, for the total row: everything at 1e-7.
    for (int i = 0; i < 100; ++i) st.record_diagnostics(14, 30, 1.0e-7);
    st.set_total(14, 100); st.set_unique(14, 10);
    st.finalize();

    const std::string text = status_text(tmp.p.string());
    const std::string hline = line_starting(text, "SG");
    const auto ends = field_ends(hline);
    const auto hdr = cells(hline, ends);
    auto col = [&](const std::vector<std::string>& r, const char* name) {
        for (size_t i = 0; i < hdr.size(); ++i)
            if (hdr[i] == name) return r.at(i);
        FAIL("no such column: " << name);
        return std::string{};
    };

    // The median is reported at its bin's centre, so within a fifth of a decade
    // of 1e-6; the mean of these values is about 4.7e+06.
    const double BIN = std::pow(10.0, 1.0 / SGStats::GRAD_PER_DECADE);   // 1.58
    const double sg5 = std::stod(col(cells(line_starting(text, "5"), ends), "grad"));
    INFO("SG 5 grad = " << sg5);
    CHECK(sg5 > 1.0e-6 / BIN);
    CHECK(sg5 < 1.0e-6 * BIN);

    const double sg14 = std::stod(col(cells(line_starting(text, "14"), ends), "grad"));
    CHECK(sg14 > 1.0e-7 / BIN);
    CHECK(sg14 < 1.0e-7 * BIN);

    // 199 of the 200 values in the total row are at or below 1e-6.
    const double all = std::stod(col(cells(line_starting(text, "ALL"), ends), "grad"));
    INFO("ALL grad = " << all);
    CHECK(all < 1.0e-5);
}

TEST_CASE("the ETA counts the ML stage, not just the force field", "[status]") {
    // 100 of a 200 target done, at 10 ms of force field each; `with_stage3`
    // adds 30 ms of ML relaxation per structure on top.
    auto cell = [](bool with_stage3, const char* want) {
        TempDir tmp;
        StatusReporter st(tmp.p.string(), 1, {14}, {200});
        st.show_stage3(true);
        for (int i = 0; i < 100; ++i) st.record(14, 10.0);
        if (with_stage3)
            for (int i = 0; i < 100; ++i) st.record_stage3(14, 20, 1e-4, 30.0);
        st.set_total(14, 100);
        st.set_unique(14, 20);
        st.finalize();
        const std::string text = status_text(tmp.p.string());
        const std::string hline = line_starting(text, "SG");
        const auto ends = field_ends(hline);
        const auto hdr = cells(hline, ends);
        const auto row = cells(line_starting(text, "14"), ends);
        for (size_t i = 0; i < hdr.size(); ++i)
            if (hdr[i] == want) return row.at(i);
        FAIL("no such column: " << want);
        return std::string{};
    };

    // Force field alone: 100 structures left at 10 ms each.
    CHECK(cell(false, "ms/struct") == "10.0");
    CHECK(cell(false, "ETA") == "1s");

    // With the ML stage the same 100 structures cost 40 ms each.
    CHECK(cell(true, "ms/struct") == "40.0");
    CHECK(cell(true, "ETA") == "4s");

    // s3ms is the cost of one relaxation and does not sum into ms/struct.
    CHECK(cell(true, "s3ms") == "30");
}

TEST_CASE("status diagnostics read as unavailable, not as zero", "[status]") {
    TempDir tmp;
    StatusReporter st(tmp.p.string(), 1, {14}, {10});
    st.record_diagnostics(14, -1, -1.0);     // the no-diagnostics sentinel
    st.set_total(14, 1); st.set_unique(14, 1);
    st.finalize();

    const std::string text = status_text(tmp.p.string());
    const std::string hline = line_starting(text, "SG");
    const auto ends = field_ends(hline);
    const auto hdr = cells(hline, ends);
    const auto r = cells(line_starting(text, "14"), ends);
    for (size_t i = 0; i < hdr.size(); ++i)
        if (hdr[i] == "conv" || hdr[i] == "iters" || hdr[i] == "grad"
            || hdr[i] == "s1conv" || hdr[i] == "s1iters")
            REQUIRE(r.at(i) == "---");
    REQUIRE_THAT(text, ContainsSubstring("--no-store-diagnostics"));
}

TEST_CASE("status table reports stage-3 convergence and cost", "[status]") {
    TempDir tmp;
    StatusReporter st(tmp.p.string(), 1, {14}, {100});
    st.show_stage3(true);

    // 4 structures through PET: 3 converged at 20/40/60 iterations, one not.
    st.record_stage3(14, 20, 1e-4, 1000.0);
    st.record_stage3(14, 40, 2e-4, 2000.0);
    st.record_stage3(14, 60, 3e-4, 3000.0);
    st.record_stage3(14, -1, 8e-3, 4000.0);
    st.set_total(14, 4); st.set_unique(14, 2);
    st.finalize();

    const std::string text = status_text(tmp.p.string());
    const std::string hline = line_starting(text, "SG");
    const auto ends = field_ends(hline);
    const auto hdr = cells(hline, ends);
    auto col = [&](const std::vector<std::string>& r, const char* name) {
        for (size_t i = 0; i < hdr.size(); ++i)
            if (hdr[i] == name) return r.at(i);
        FAIL("no such column: " << name);
        return std::string{};
    };
    const auto r14 = cells(line_starting(text, "14"), ends);
    REQUIRE(r14.size() == hdr.size());

    CHECK(col(r14, "s3conv")  == "75.0%");    // 3 of 4
    CHECK(col(r14, "s3iters") == "40");       // mean over the converged only
    CHECK(col(r14, "s3force") == "2.2e-03");  // mean over all four
    CHECK(col(r14, "s3ms")    == "2500");     // mean wall time per structure
}

TEST_CASE("stage-2 and stage-3 basin counts are both reported", "[status]") {
    TempDir tmp;
    StatusReporter st(tmp.p.string(), 1, {14}, {100});
    st.show_stage3(true);
    st.set_total(14, 40);
    st.set_unique(14, 7);            // stage 2 merged 40 structures into 7 basins
    st.set_stage3_unique(14, 12);    // the ML relax separated some of them again
    st.finalize();

    const std::string text = status_text(tmp.p.string());
    const std::string hline = line_starting(text, "SG");
    const auto ends = field_ends(hline);
    const auto hdr = cells(hline, ends);
    auto col = [&](const std::vector<std::string>& r, const char* name) {
        for (size_t i = 0; i < hdr.size(); ++i)
            if (hdr[i] == name) return r.at(i);
        FAIL("no such column: " << name);
        return std::string{};
    };
    const auto r14 = cells(line_starting(text, "14"), ends);
    CHECK(col(r14, "unique") == "7");
    CHECK(col(r14, "s3uniq") == "12");
    const auto all = cells(line_starting(text, "ALL"), ends);
    CHECK(col(all, "unique") == "7");
    CHECK(col(all, "s3uniq") == "12");
}

TEST_CASE("stage-3 columns are absent unless the run has a stage-3 pass",
          "[status]") {
    TempDir tmp;
    StatusReporter st(tmp.p.string(), 1, {14}, {10});
    st.record_diagnostics(14, 10, 1e-6);
    st.set_total(14, 1); st.set_unique(14, 1);
    st.finalize();

    const std::string text = status_text(tmp.p.string());
    REQUIRE_THAT(text, ContainsSubstring("conv"));      // stage-2 columns stay
    CHECK_THAT(text, !ContainsSubstring("s3conv"));
    CHECK_THAT(text, !ContainsSubstring("s3ms"));
    CHECK_THAT(text, !ContainsSubstring("s3uniq"));
}

namespace {

// The ETA cell for one space group. `finalise` picks the summary written after
// the consumer has drained instead of the live table.
std::string pet_eta_cell(bool pet, int total, int target, int n_s3, int dropped,
                         bool finalise, int skipped = 0) {
    TempDir tmp;
    StatusReporter st(tmp.p.string(), 1, {14}, {target});
    st.show_stage3(pet);
    st.set_pet_stage(pet);
    for (int i = 0; i < total; ++i) st.record(14, 10.0);
    for (int i = 0; i < n_s3; ++i) st.record_stage3(14, 30, 1e-4, 100.0);
    if (dropped) st.record_pet_dropped(14, dropped);
    if (skipped) st.record_pet_skipped(14, skipped);
    st.set_total(14, total);
    st.set_unique(14, total);
    if (finalise) {
        st.finalize();
    } else {
        // maybe_update() rate-limits, so poll until the file appears.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            st.maybe_update();
            if (std::filesystem::exists(tmp.p / "status.txt")) break;
        }
    }
    const std::string text = status_text(tmp.p.string());
    const std::string hline = line_starting(text, "SG");
    const auto ends = field_ends(hline);
    const auto hdr = cells(hline, ends);
    const auto row = cells(line_starting(text, "14"), ends);
    for (size_t i = 0; i < hdr.size(); ++i)
        if (hdr[i] == "ETA") return row.at(i);
    FAIL("no ETA column");
    return std::string{};
}

}  // namespace

TEST_CASE("a space group with its ML stage outstanding is not reported done",
          "[status]") {
    // Target met, ML stage configured, nothing back from it yet.
    CHECK(pet_eta_cell(true, 100, 100, 0, 0, false) == "pet");
    // Partway through the ML pass is still not finished.
    CHECK(pet_eta_cell(true, 100, 100, 60, 0, false) == "pet");
    // Once stage 3 has caught up it really is done.
    CHECK(pet_eta_cell(true, 100, 100, 100, 0, false) == "done");
    CHECK(pet_eta_cell(false, 100, 100, 0, 0, false) == "done");
    // An unmet target is unaffected either way.
    CHECK(pet_eta_cell(true, 50, 100, 0, 0, false) != "done");
    CHECK(pet_eta_cell(true, 50, 100, 0, 0, false) != "pet");
}

TEST_CASE("dropped ML candidates still count toward the stage being finished",
          "[status]") {
    // 98 relaxed, 2 rejected by the screen.
    CHECK(pet_eta_cell(true, 100, 100, 98, 2, false) == "done");
    // Genuinely mid-stage: 60 back, 2 dropped, 38 still to come.
    CHECK(pet_eta_cell(true, 100, 100, 60, 2, false) == "pet");
    // The table printed after the run has drained never claims work in flight.
    CHECK(pet_eta_cell(true, 100, 100, 0, 0, true) == "done");
}

TEST_CASE("skipped ML candidates count toward the stage being finished",
          "[status]") {
    // 8 basins relaxed, 92 structures skipped as duplicates of them: finished.
    CHECK(pet_eta_cell(true, 100, 100, 8, 0, false, /*skipped=*/92) == "done");
    // Skips and drops together close the gap.
    CHECK(pet_eta_cell(true, 100, 100, 8, 2, false, /*skipped=*/90) == "done");
    // Still short by 10, so still relaxing.
    CHECK(pet_eta_cell(true, 100, 100, 8, 2, false, /*skipped=*/80) == "pet");
}
TEST_CASE("a resumed pet run can still report done", "[status]") {
    auto eta_after_resume = [](int resumed_total, int resumed_s3,
                               int new_total, int new_s3) {
        TempDir tmp;
        StatusReporter st(tmp.p.string(), 1, {14}, {resumed_total + new_total});
        st.show_stage3(true);
        st.set_pet_stage(true);
        // What ResultConsumer does on resume: seed the counts already on disk.
        st.set_total(14, resumed_total);
        st.seed_stage3(14, resumed_s3);
        // Then the run continues and produces the rest.
        for (int i = 0; i < new_s3; ++i) st.record_stage3(14, 30, 1e-4, 100.0);
        st.set_total(14, resumed_total + new_total);
        // The live table, not finalize(), which reports "done" unconditionally.
        // maybe_update is rate-limited, so poll until the file appears.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            st.maybe_update();
            if (std::filesystem::exists(tmp.p / "status.txt")) break;
        }

        const std::string text = status_text(tmp.p.string());
        const std::string hline = line_starting(text, "SG");
        const auto ends = field_ends(hline);
        const auto hdr = cells(hline, ends);
        const auto row = cells(line_starting(text, "14"), ends);
        for (size_t i = 0; i < hdr.size(); ++i)
            if (hdr[i] == "ETA") return row.at(i);
        FAIL("no ETA column");
        return std::string{};
    };

    // Resume 100 structures that already had their 100 stage-3 rows, then finish
    // the remaining 50. Every structure has a stage-3 row, so this is "done".
    CHECK(eta_after_resume(100, 100, 50, 50) == "done");

    CHECK(eta_after_resume(100, 100, 50, 10) == "pet");
}

TEST_CASE("status table shows a space group whose energy spread has collapsed",
          "[status]") {
    // A space group whose energies have collapsed to one value (vacuum gap past
    // the cutoff) must show a near-zero spread.
    TempDir tmp;
    StatusReporter st(tmp.p.string(), 1, {14, 156}, {100, 100});

    // SG14: a real landscape, energies spread over 5 kJ/mol.
    for (double e : {-29.9, -27.4, -25.1, -24.9}) st.record_energy(14, e);
    st.set_total(14, 4); st.set_unique(14, 4);
    // SG156: decoupled -- four "different" structures, one energy.
    for (double e : {-10.220, -10.221, -10.220, -10.219}) st.record_energy(156, e);
    st.set_total(156, 4); st.set_unique(156, 4);
    st.finalize();

    const std::string text = status_text(tmp.p.string());
    const std::string hline = line_starting(text, "SG");
    const auto ends = field_ends(hline);
    const auto hdr = cells(hline, ends);
    auto col = [&](const std::vector<std::string>& r, const char* name) {
        for (size_t i = 0; i < hdr.size(); ++i)
            if (hdr[i] == name) return r.at(i);
        FAIL("no such column: " << name);
        return std::string{};
    };

    REQUIRE(col(cells(line_starting(text, "14"), ends), "dE") == "5.00");
    REQUIRE(col(cells(line_starting(text, "156"), ends), "dE") == "0.00");
    // The unique count cannot distinguish four basins from one energy four times.
    REQUIRE(col(cells(line_starting(text, "156"), ends), "unique") == "4");
}

TEST_CASE("a space group with no energies yet reports no spread", "[status]") {
    TempDir tmp;
    StatusReporter st(tmp.p.string(), 1, {14}, {100});
    st.set_total(14, 0);
    st.finalize();
    const std::string text = status_text(tmp.p.string());
    const std::string hline = line_starting(text, "SG");
    const auto ends = field_ends(hline);
    const auto hdr = cells(hline, ends);
    const auto r = cells(line_starting(text, "14"), ends);
    for (size_t i = 0; i < hdr.size(); ++i)
        if (hdr[i] == "dE") REQUIRE(r.at(i) == "---");
}

TEST_CASE("a non-finite energy does not poison the spread", "[status]") {
    // The sentinel an unaccepted slot carries is finite; it must not enter the
    // spread.
    TempDir tmp;
    StatusReporter st(tmp.p.string(), 1, {14}, {100});
    st.record_energy(14, -29.9);
    st.record_energy(14, std::numeric_limits<double>::infinity());
    st.record_energy(14, std::numeric_limits<double>::quiet_NaN());
    st.record_energy(14, 1e300);
    st.record_energy(14, -27.4);
    st.set_total(14, 2);
    st.finalize();
    const std::string text = status_text(tmp.p.string());
    const std::string hline = line_starting(text, "SG");
    const auto ends = field_ends(hline);
    const auto hdr = cells(hline, ends);
    const auto r = cells(line_starting(text, "14"), ends);
    for (size_t i = 0; i < hdr.size(); ++i)
        if (hdr[i] == "dE") REQUIRE(r.at(i) == "2.50");
}
