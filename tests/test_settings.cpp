#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "klasp/io.hpp"
#include <algorithm>
#include <cmath>

using namespace klasp;
using Catch::Matchers::WithinRel;
using json = nlohmann::json;

static bool mentions(const InputSettings& s, const std::string& needle) {
    return std::any_of(s.warnings.begin(), s.warnings.end(),
                       [&](const std::string& w) {
                           return w.find(needle) != std::string::npos;
                       });
}

TEST_CASE("An absent settings block leaves the defaults alone", "[settings]") {
    const InputSettings s = parse_settings(json(nullptr));
    REQUIRE(s.ewald_accuracy == DEFAULT_EWALD_ACCURACY);
    REQUIRE(s.max_rank_cap == -1);
    REQUIRE(s.warnings.empty());
}

TEST_CASE("An empty settings block leaves the defaults alone", "[settings]") {
    const InputSettings s = parse_settings(json::object());
    REQUIRE(s.ewald_accuracy == DEFAULT_EWALD_ACCURACY);
    REQUIRE(s.warnings.empty());
}

TEST_CASE("ewald_accuracy is read", "[settings]") {
    const InputSettings s = parse_settings(json{{"ewald_accuracy", 1e-8}});
    REQUIRE_THAT(double(s.ewald_accuracy), WithinRel(1e-8, 1e-12));
    REQUIRE(s.warnings.empty());
    REQUIRE(s.ewald_accuracy != DEFAULT_EWALD_ACCURACY);
}

TEST_CASE("A tolerance outside (0, 1) is refused, not clamped", "[settings]") {
    for (const json& bad : {json(-1.0), json(0.0), json(1.0), json(2.5),
                            json("tight"), json(nullptr)}) {
        const InputSettings s = parse_settings(json{{"ewald_accuracy", bad}});
        INFO("value " << bad.dump());
        REQUIRE(s.ewald_accuracy == DEFAULT_EWALD_ACCURACY);
        REQUIRE(mentions(s, "ewald_accuracy"));
    }
}

// The tolerance reaches both sums at the same value via ewald_split_width.
TEST_CASE("The split width tightens as the tolerance does", "[settings]") {
    const RealType loose = ewald_split_width(1e-4);
    const RealType tight = ewald_split_width(1e-8);
    REQUIRE(tight > loose);
    // s = sqrt(-ln(acc)), so exp(-s^2) is the tolerance it was asked for.
    REQUIRE_THAT(std::exp(-double(tight) * double(tight)), WithinRel(1e-8, 1e-9));
    REQUIRE_THAT(std::exp(-double(loose) * double(loose)), WithinRel(1e-4, 1e-9));
}

TEST_CASE("max_interaction_order is reported as a cap", "[settings]") {
    const InputSettings s = parse_settings(json{{"max_interaction_order", 2}});
    REQUIRE(s.max_rank_cap == 2);
    REQUIRE(s.warnings.empty());
}

TEST_CASE("A rank that is not a rank is refused", "[settings]") {
    for (const json& bad : {json(-1), json(2.5), json("quadrupole")}) {
        const InputSettings s = parse_settings(json{{"max_interaction_order", bad}});
        INFO("value " << bad.dump());
        REQUIRE(s.max_rank_cap == -1);
        REQUIRE(mentions(s, "max_interaction_order"));
    }
}

// Keys klasp knows but does not act on are named, with where the value comes from.
TEST_CASE("Ignored keys are named, with the reason", "[settings]") {
    const InputSettings s = parse_settings(json{
        {"use_ewald", true},
        {"neighbor_radius", 20.0},
        {"gradient_tolerance", 1e-5},
        {"max_iterations", 200},
    });
    REQUIRE(s.warnings.size() == 4);
    REQUIRE(mentions(s, "use_ewald"));
    REQUIRE(mentions(s, "neighbor_radius"));
    REQUIRE(mentions(s, "--grad-tol"));
    REQUIRE(mentions(s, "--iterations"));
}

TEST_CASE("An unrecognised key is reported too", "[settings]") {
    const InputSettings s = parse_settings(json{{"ewald_acuracy", 1e-8}});
    REQUIRE(s.warnings.size() == 1);
    REQUIRE(mentions(s, "ewald_acuracy"));
    REQUIRE(s.ewald_accuracy == DEFAULT_EWALD_ACCURACY);
}

// Named without leading dashes: Catch2 would parse "--" as one of its own flags.
TEST_CASE("The ewald-accuracy override replaces what the input asked for", "[settings]") {
    Workspace ws{};
    ws.ewald_accuracy = RealType(1e-8);          // as if the file had said so

    apply_ewald_accuracy_override(-1.0, ws);     // the default: no opinion
    REQUIRE(ws.ewald_accuracy == RealType(1e-8));

    apply_ewald_accuracy_override(1e-6, ws);
    REQUIRE_THAT(double(ws.ewald_accuracy), WithinRel(1e-6, 1e-12));
}

TEST_CASE("A command-line tolerance outside (0, 1) is an error", "[settings]") {
    Workspace ws{};
    ws.ewald_accuracy = DEFAULT_EWALD_ACCURACY;
    for (double bad : {0.0, 1.0, 2.5}) {
        INFO("value " << bad);
        REQUIRE_THROWS(apply_ewald_accuracy_override(bad, ws));
        REQUIRE(ws.ewald_accuracy == DEFAULT_EWALD_ACCURACY);
    }
}

// The two keys handled at their own call sites must not be reported as ignored.
TEST_CASE("Keys applied elsewhere are not called ignored", "[settings]") {
    const InputSettings s = parse_settings(json{
        {"h_bond_extend", 0.1},
        {"spline_min", 2.0},
    });
    REQUIRE(s.warnings.empty());
}

// The neutrality boundary: clears DMA rounding (~1e-6 e), catches anything that
// matters (the omitted term goes as Q^2).
TEST_CASE("a charged cell is distinguished from rounding noise", "[settings]") {
    CHECK(cell_is_neutral({}));
    CHECK(cell_is_neutral({0.0, 0.0, 0.0}));

    // A real DMA file: charges that cancel to about a millionth.
    CHECK(cell_is_neutral({0.412300, -0.206149, -0.206150}));
    CHECK_THAT(net_site_charge({0.412300, -0.206149, -0.206150}),
               Catch::Matchers::WithinAbs(1e-6, 1e-9));

    // An actual ion, either sign.
    CHECK_FALSE(cell_is_neutral({1.0, 0.0, 0.0}));
    CHECK_FALSE(cell_is_neutral({-1.0, 0.0, 0.0}));

    // Either side of the boundary.
    CHECK(cell_is_neutral({NEUTRALITY_TOL_E * 0.5}));
    CHECK_FALSE(cell_is_neutral({NEUTRALITY_TOL_E * 2.0}));

    // NaN must not read as neutral.
    CHECK_FALSE(cell_is_neutral({std::nan("")}));
}
