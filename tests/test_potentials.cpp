#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "klasp/potentials.hpp"
#include "klasp/constants.hpp"
#include <set>
#include <cmath>

using namespace klasp;
using Catch::Matchers::WithinRel;
using json = nlohmann::json;

static json entry(const std::string& ti, const std::string& tj,
                  double A, double rho, double C6,
                  const std::string& ei = "", const std::string& ej = "") {
    json e;
    e["types"] = json::array({ti, tj});
    if (!ei.empty()) e["elements"] = json::array({ei, ej});
    e["A"] = A; e["rho"] = rho; e["C6"] = C6;
    return e;
}

// Two components, each knowing only its own self-pairs: the shape every
// co-crystal load arrives in.
static json two_component_input() {
    json p;
    p["buckingham"] = json::array({
        entry("H_a", "H_a", 131.42, 0.280899, 2.885328, "H", "H"),
        entry("O_a", "O_a", 21460.0, 0.244498, 20.51,   "O", "O"),
        entry("H_a", "O_a", 1200.0,  0.26,     7.0,     "H", "O"),
        entry("N_b", "N_b", 32410.0, 0.253807, 26.44,   "N", "N"),
        entry("C_b", "C_b", 32410.0, 0.277778, 25.29,   "C", "C"),
        entry("N_b", "C_b", 32000.0, 0.265,    25.8,    "N", "C"),
    });
    return p;
}

static std::map<std::string, int> two_component_types() {
    return {{"H_a", 0}, {"O_a", 1}, {"N_b", 2}, {"C_b", 3}};
}

TEST_CASE("Types keep the indices the type map gives them", "[potentials]") {
    const auto t = resolve_potentials(two_component_input(), two_component_types());
    REQUIRE(t.n_types() == 4);
    REQUIRE(t.type_names[0] == "H_a");
    REQUIRE(t.type_names[3] == "C_b");
}

TEST_CASE("A listed pair is read, not combined", "[potentials]") {
    const auto t = resolve_potentials(two_component_input(), two_component_types());
    // H_a/O_a is listed, and its A is nothing like sqrt(A_HH * A_OO).
    REQUIRE_THAT(t.a(0, 1), WithinRel(1200.0 * double(EV_TO_AU), 1e-12));
    REQUIRE_THAT(t.a(0, 1), !WithinRel(std::sqrt(131.42 * 21460.0) * double(EV_TO_AU), 1e-6));
    for (const auto& d : t.derived) REQUIRE(d != "H_a,O_a");
}

// Neither input names a pair spanning the two components, so all four cross
// pairs are derived.
TEST_CASE("Cross-component pairs are derived and named", "[potentials]") {
    const auto t = resolve_potentials(two_component_input(), two_component_types());
    REQUIRE(t.derived.size() == 4);
    for (int i : {0, 1})
        for (int j : {2, 3}) {
            INFO(t.type_names[i] << " / " << t.type_names[j]);
            REQUIRE_FALSE(t.unset(i, j));
        }
}

TEST_CASE("Good-Hope combines the two self-pairs", "[potentials]") {
    const auto t = resolve_potentials(two_component_input(), two_component_types());
    const int H = 0, N = 2;
    REQUIRE_THAT(t.a(H, N), WithinRel(std::sqrt(t.a(H, H) * t.a(N, N)), 1e-12));
    REQUIRE_THAT(t.b(H, N), WithinRel(0.5 * (t.b(H, H) + t.b(N, N)), 1e-12));
    REQUIRE_THAT(t.c(H, N), WithinRel(std::sqrt(t.c(H, H) * t.c(N, N)), 1e-12));
}

TEST_CASE("The table is symmetric", "[potentials]") {
    const auto t = resolve_potentials(two_component_input(), two_component_types());
    for (int i = 0; i < t.n_types(); ++i)
        for (int j = 0; j < t.n_types(); ++j) {
            REQUIRE(t.a(i, j) == t.a(j, i));
            REQUIRE(t.b(i, j) == t.b(j, i));
            REQUIRE(t.c(i, j) == t.c(j, i));
        }
}

// What gets recorded resolves back to what was used, derived pairs included.
TEST_CASE("A resolved table round-trips through the recorded JSON", "[potentials]") {
    const auto types = two_component_types();
    const auto first = resolve_potentials(two_component_input(), types);

    json recorded;
    recorded["buckingham"] = potential_table_to_json(first);
    const auto second = resolve_potentials(recorded, types);

    REQUIRE(tables_agree(first, second));
    // And the record is complete on its own terms: reading it back invents nothing.
    REQUIRE(second.derived.empty());
    REQUIRE(first.derived.size() == 4);
}

TEST_CASE("The recorded JSON lists every pair once", "[potentials]") {
    const auto t = resolve_potentials(two_component_input(), two_component_types());
    const json list = potential_table_to_json(t);
    REQUIRE(list.size() == 4 * 5 / 2);   // n(n+1)/2 for 4 types, all set
    std::set<std::pair<std::string, std::string>> seen;
    for (const auto& e : list) {
        auto ts = e["types"].get<std::vector<std::string>>();
        if (ts[1] < ts[0]) std::swap(ts[0], ts[1]);
        REQUIRE(seen.insert({ts[0], ts[1]}).second);
        REQUIRE(e.contains("elements"));
    }
}

TEST_CASE("A type with no self-pair is left unset rather than guessed", "[potentials]") {
    json p = two_component_input();
    std::map<std::string, int> types = two_component_types();
    types["Xx"] = 4;                       // in the map, in no entry
    const auto t = resolve_potentials(p, types);
    for (int i = 0; i < 4; ++i) {
        INFO("pair with " << t.type_names[i]);
        REQUIRE(t.unset(4, i));
    }
    for (const auto& d : t.derived) REQUIRE(d.find("Xx") == std::string::npos);
}

TEST_CASE("A single component with every pair listed derives nothing", "[potentials]") {
    json p;
    p["buckingham"] = json::array({
        entry("H_a", "H_a", 131.42, 0.280899, 2.885328, "H", "H"),
        entry("O_a", "O_a", 21460.0, 0.244498, 20.51,   "O", "O"),
        entry("H_a", "O_a", 1200.0,  0.26,     7.0,     "H", "O"),
    });
    const auto t = resolve_potentials(p, {{"H_a", 0}, {"O_a", 1}});
    REQUIRE(t.derived.empty());
}

TEST_CASE("Nonsense entries are skipped, not propagated", "[potentials]") {
    json p;
    p["buckingham"] = json::array({
        entry("H_a", "H_a", 131.42, 0.280899, 2.885328, "H", "H"),
        entry("O_a", "O_a", 21460.0, 0.0, 20.51, "O", "O"),   // rho = 0 is 1/0
        entry("Q_z", "Q_z", 1.0, 0.2, 1.0, "Q", "Q"),          // type not in the map
    });
    const auto t = resolve_potentials(p, {{"H_a", 0}, {"O_a", 1}});
    REQUIRE_FALSE(t.unset(0, 0));
    REQUIRE(t.unset(1, 1));      // the zero-rho entry never landed
    REQUIRE(t.unset(0, 1));      // and so nothing could be combined from it
    REQUIRE(t.derived.empty());
    // The unreadable entry is recorded for a warning; the one for an unused type
    // is passed over.
    REQUIRE(t.skipped.size() == 1u);
    CHECK(t.skipped[0].find("O_a,O_a") != std::string::npos);
}

TEST_CASE("A pair nothing defines between two parameterised types is recorded", "[potentials]") {
    json p;
    p["buckingham"] = json::array({
        entry("H_a", "H_a", 131.42, 0.280899, 2.885328),
        entry("H_a", "O_a", 1200.0, 0.26, 7.0),       // O_a has a pair, but no self pair
        entry("N_b", "N_b", 32410.0, 0.253807, 26.44),
    });
    const auto t = resolve_potentials(p, {{"H_a", 0}, {"O_a", 1}, {"N_b", 2}, {"X", 3}});
    CHECK(t.derived == std::vector<std::string>{"H_a,N_b"});
    // X has no pairs at all; it is warned about once as a type, not per partner.
    CHECK(t.undefined == std::vector<std::string>{"O_a,O_a", "O_a,N_b"});
}

TEST_CASE("An absent potentials block gives an empty table, not a crash", "[potentials]") {
    REQUIRE(resolve_potentials(json(nullptr), {{"H_a", 0}}).unset(0, 0));
    REQUIRE(resolve_potentials(json::object(), {{"H_a", 0}}).unset(0, 0));
    json p; p["buckingham"] = "not a list";
    REQUIRE(resolve_potentials(p, {{"H_a", 0}}).unset(0, 0));
}

// Pins the unit conversions to the units the input format documents.
TEST_CASE("Units convert to the atomic units the kernels want", "[potentials]") {
    json p;
    p["buckingham"] = json::array({entry("H_a", "H_a", 100.0, 0.25, 4.0, "H", "H")});
    const auto t = resolve_potentials(p, {{"H_a", 0}});
    REQUIRE_THAT(t.a(0, 0), WithinRel(100.0 * double(EV_TO_AU), 1e-12));
    REQUIRE_THAT(t.b(0, 0), WithinRel((1.0 / 0.25) * double(BOHR_TO_ANGSTROM), 1e-12));
    REQUIRE_THAT(t.c(0, 0),
                 WithinRel(4.0 * double(EV_TO_AU) * std::pow(double(ANGSTROM_TO_BOHR), 6), 1e-12));
}

TEST_CASE("Damping is on at range 1.5 unless the input says otherwise",
          "[potentials][damping]") {
    const TangToenniesDamping absent = parse_damping(json::object());
    CHECK(absent.enabled);
    CHECK(absent.range == RealType(1.5));

    const TangToenniesDamping named = parse_damping(json{{"damping", false},
                                                         {"damping_range", 1.0}});
    CHECK_FALSE(named.enabled);
    CHECK(named.range == RealType(1.0));

    CHECK_THROWS(parse_damping(json{{"damping_range", 0.0}}));
    CHECK_THROWS(parse_damping(json{{"damping_range", -1.5}}));
    CHECK_THROWS(parse_damping(json{{"damping_range", "wide"}}));
}

// The record names the damping even where the input did not.
TEST_CASE("The recorded potentials state the damping that ran", "[potentials][damping]") {
    json p = two_component_input();
    REQUIRE_FALSE(p.contains("damping"));
    record_damping(p, parse_damping(p));
    CHECK(p["damping"] == true);
    CHECK(p["damping_range"] == 1.5);

    const TangToenniesDamping back = parse_damping(p);
    CHECK(back.enabled);
    CHECK(back.range == RealType(1.5));
}
