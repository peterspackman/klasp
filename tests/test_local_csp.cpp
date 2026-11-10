#include <catch2/catch_test_macros.hpp>
#include "kokkos_listener.hpp"

#include "klasp/csp_run.hpp"
#include "klasp/config.hpp"
#include "klasp/log.hpp"

#include <sqlite_modern_cpp.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

using namespace klasp;

// A whole CSP with no MPI, checked against the klasp binary on one rank. The
// landscapes must be identical row for row, since seeds index the same
// quasi-random sequence. The test executable does not link MPI, so the reference
// shells out to klasp (built via tests/CMakeLists.txt).

namespace {

using Row = std::tuple<long long, int, int, double>;

std::vector<Row> landscape(const std::string& db_path) {
    std::vector<Row> rows;
    sqlite::database db(db_path);
    db << "select seed, space_group, stage, energy from structure "
          "order by space_group, stage, seed"
       >> [&](long long seed, int sg, int stage, double e) {
              rows.emplace_back(seed, sg, stage, e);
          };
    return rows;
}

std::filesystem::path scratch(const char* leaf) {
    auto p = std::filesystem::temp_directory_path() / "klasp_local_csp" / leaf;
    std::filesystem::remove_all(p);
    return p;
}

}  // namespace

TEST_CASE("a CSP with no MPI produces the landscape the MPI path does",
          "[csp][local]") {
    const auto input = std::filesystem::path(KLASP_SOURCE_DIR)
                     / "example-inputs" / "acetic_acid.json";
    REQUIRE(std::filesystem::exists(input));

    const auto mpi_dir = scratch("mpi");
    const auto local_dir = scratch("local");

    const std::string cmd =
        std::string(KLASP_BINARY_PATH) + " csp " + input.string() +
        " --sg 14 --target 24 -o " + mpi_dir.string() + " > /dev/null 2>&1";
    REQUIRE(std::system(cmd.c_str()) == 0);

    RunConfig cfg;
    cfg.control.basis_paths = {input.string()};
    cfg.control.space_groups = {14};
    cfg.control.targets[14] = 24;
    cfg.output.dir = local_dir.string();

    CspRunResult result;
    {
        logging::Capture quiet(logging::Level::Error);
        result = run_local_csp(cfg);
    }

    const auto a = landscape((mpi_dir / "csp.db").string());
    const auto b = landscape((local_dir / "csp.db").string());

    REQUIRE_FALSE(a.empty());
    CHECK(a.size() == b.size());
    CHECK(a == b);

    CHECK(result.stored > 0);
    CHECK(result.unique > 0);
    CHECK(result.unique <= result.stored);
    CHECK_FALSE(result.stopped_early);

    std::filesystem::remove_all(mpi_dir.parent_path());
}
