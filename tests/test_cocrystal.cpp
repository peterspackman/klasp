#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "klasp/io.hpp"
#include "klasp/potentials.hpp"
#include "klasp/batch.hpp"
#include "klasp/powder.hpp"
#include "klasp/constants.hpp"
#include "klasp/db_tool.hpp"
#include "klasp/sgdata.hpp"
#ifdef KLASP_HAS_PET
#include "klasp/pet_relax.hpp"
#endif

using namespace klasp;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;
using json = nlohmann::json;

static json site(const std::string& el, const std::string& type,
                 double x, double y, double z, double q) {
    json s;
    s["element"] = el;
    s["label"] = el;
    s["type"] = type;
    s["position"] = json::array({x, y, z});
    s["multipoles"] = json{{"charge", q}};
    return s;
}

static json buck(const std::string& ti, const std::string& tj,
                 double A, double rho, double C6) {
    json e;
    e["types"] = json::array({ti, tj});
    e["elements"] = json::array({ti.substr(0, 1), tj.substr(0, 1)});
    e["A"] = A; e["rho"] = rho; e["C6"] = C6;
    return e;
}

// Two components of different size; `big_first` sets which type comes first.
static json two_component_doc(bool big_first) {
    json big = json::object();
    big["name"] = "big";
    big["sites"] = json::array({
        site("O", "O_x", 0.0, 0.0, 0.0, -0.4),
        site("C", "C_x", 1.2, 0.0, 0.0,  0.3),
        site("H", "H_x", 1.8, 0.9, 0.0,  0.05),
        site("H", "H_x", 1.8,-0.9, 0.0,  0.05),
    });
    json small = json::object();
    small["name"] = "small";
    small["sites"] = json::array({
        site("N", "N_y", 0.0, 0.0, 0.0, -0.2),
        site("H", "H_y", 1.0, 0.0, 0.0,  0.2),
    });

    json d;
    d["cell"] = {{"a", 12.0}, {"b", 12.0}, {"c", 12.0},
                 {"alpha", 90.0}, {"beta", 90.0}, {"gamma", 90.0}};
    d["molecule_types"] = big_first ? json::array({big, small})
                                    : json::array({small, big});
    d["molecules"] = json::array({
        json{{"type", "big"},   {"translation", json::array({0.2, 0.2, 0.2})}},
        json{{"type", "small"}, {"translation", json::array({0.7, 0.7, 0.7})}},
    });
    d["potentials"] = json{
        {"cutoff", 15.0},
        {"buckingham", json::array({
            buck("O_x", "O_x", 21460.0, 0.244498, 20.51),
            buck("C_x", "C_x", 32410.0, 0.277778, 25.29),
            buck("H_x", "H_x",   131.42, 0.280899, 2.885328),
            buck("N_y", "N_y", 32410.0, 0.253807, 26.44),
            buck("H_y", "H_y",   131.42, 0.280899, 2.885328),
        })},
    };
    return d;
}

static std::pair<MoleculeData, Workspace> load(const json& doc, CrystalConfig* cfg_out) {
    auto ws = allocate_workspace(512, 27 * 27 * 27, 50000, 32, 4);
    MoleculeData mol{};
    const CrystalConfig cfg = load_structure_json(doc, mol, ws);
    if (cfg_out) *cfg_out = cfg;
    return {mol, ws};
}

TEST_CASE("Unequal components are laid out at the larger stride", "[cocrystal]") {
    for (bool big_first : {true, false}) {
        INFO((big_first ? "larger type first" : "smaller type first"));
        CrystalConfig cfg{};
        auto loaded = load(two_component_doc(big_first), &cfg);
        MoleculeData& mol = loaded.first;
        REQUIRE(mol.n_indep == 2);
        REQUIRE(mol.n_sites_per_mol == 4);   // the LARGER component, not the first
    }
}

TEST_CASE("No atom of either component is dropped", "[cocrystal]") {
    for (bool big_first : {true, false}) {
        INFO((big_first ? "larger type first" : "smaller type first"));
        auto loaded = load(two_component_doc(big_first), nullptr);
        MoleculeData& mol = loaded.first;

        auto h_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mol.body_charges);
        double total_abs = 0;
        int n_charged = 0;
        for (int i = 0; i < mol.n_indep * mol.n_sites_per_mol; ++i) {
            total_abs += std::abs(h_q(i));
            if (h_q(i) != 0.0) ++n_charged;
        }
        REQUIRE(n_charged == 6);                                  // 4 + 2 real sites
        REQUIRE_THAT(total_abs, WithinRel(0.4 + 0.3 + 0.05 + 0.05 + 0.2 + 0.2, 1e-12));
    }
}

// Padded slots are ordinary atoms to the pair kernel; only the inert type
// silences their Buckingham.
TEST_CASE("Padded slots carry the inert type", "[cocrystal]") {
    auto loaded = load(two_component_doc(true), nullptr);
    MoleculeData& mol = loaded.first; Workspace& ws = loaded.second;
    auto h_t = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mol.body_types);

    const int stride = mol.n_sites_per_mol;
    const int ghost = ws.ghost_type;
    // Molecule 1 is the 2-site component padded to 4.
    REQUIRE(h_t(1 * stride + 2) == ghost);
    REQUIRE(h_t(1 * stride + 3) == ghost);
    REQUIRE(h_t(1 * stride + 0) != ghost);
    REQUIRE(h_t(0 * stride + 3) != ghost);
}

TEST_CASE("The inert type has no Buckingham parameters", "[cocrystal]") {
    auto loaded = load(two_component_doc(true), nullptr);
    Workspace& ws = loaded.second;
    const int ghost = ws.ghost_type;
    auto h_A = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ws.buck_A);
    auto h_B = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ws.buck_B);
    auto h_C = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ws.buck_C);
    for (int i = 0; i < ws.n_types; ++i) {
        INFO("pair (" << i << ", ghost)");
        REQUIRE(h_A(i, ghost) == 0.0);
        REQUIRE(h_B(i, ghost) == 0.0);
        REQUIRE(h_C(i, ghost) == 0.0);
        REQUIRE(h_A(ghost, i) == 0.0);
    }
    REQUIRE(h_A(0, 0) > 0.0);
}

TEST_CASE("Padding a component does not change the energy", "[cocrystal]") {
    json alone = two_component_doc(true);
    alone["molecule_types"] = json::array({alone["molecule_types"][1]});  // the 2-site one
    alone["molecules"] = json::array({
        json{{"type", "small"}, {"translation", json::array({0.7, 0.7, 0.7})}}});

    // Same molecule in a document whose other type is larger, so it is padded from
    // 2 sites to 4; the other type's molecule is dropped so both cells match.
    json padded = alone;
    json big = two_component_doc(true)["molecule_types"][0];
    padded["molecule_types"] = json::array({big, alone["molecule_types"][0]});

    CrystalConfig cfg_a{}, cfg_p{};
    auto la = load(alone, &cfg_a);
    auto lp = load(padded, &cfg_p);
    MoleculeData& mol_a = la.first; Workspace& ws_a = la.second;
    MoleculeData& mol_p = lp.first; Workspace& ws_p = lp.second;

    REQUIRE(mol_a.n_sites_per_mol == 2);
    REQUIRE(mol_p.n_sites_per_mol == 4);   // padded, so the comparison is meaningful

    const RealType e_alone  = batch_evaluate_energy(mol_a, ws_a, cfg_a.max_rank,
                                                    cfg_a.taper_on, cfg_a.taper_off,
                                                    cfg_a.buck_damping);
    const RealType e_padded = batch_evaluate_energy(mol_p, ws_p, cfg_p.max_rank,
                                                    cfg_p.taper_on, cfg_p.taper_off,
                                                    cfg_p.buck_damping);
    // Not bit-exact: kappa depends on the atom count, so the sums agree only to
    // their convergence. A live ghost would be orders of magnitude larger.
    INFO("alone " << e_alone << " padded " << e_padded);
    REQUIRE_THAT(double(e_padded), WithinAbs(double(e_alone), 1e-8));
}

// gemmi's El::X is ordinal 0, which IT92 tables map to oxygen, so the ghost
// element must be excluded explicitly.
TEST_CASE("A ghost element scatters nothing", "[cocrystal][powder]") {
    const auto ff = build_powder_form_factors({"C", "X", "O"});
    REQUIRE(ff.n_atoms == 3);
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ff.cm);

    double ghost = 0, carbon = 0, oxygen = 0;
    for (int j = 0; j < 9; ++j) {
        carbon += std::abs(double(h(0, j)));
        ghost  += std::abs(double(h(1, j)));
        oxygen += std::abs(double(h(2, j)));
    }
    REQUIRE(ghost == 0.0);
    REQUIRE(carbon > 0.0);
    REQUIRE(oxygen > 0.0);

    for (int j = 0; j < 9; ++j) {
        INFO("coefficient " << j);
        REQUIRE(double(h(1, j)) != double(h(2, j)));
    }
}

#ifdef KLASP_HAS_PET
// Components of different size at the larger stride, as a co-crystal reaches
// expand_rigid_structure.
static BasisConfig padded_two_component_basis(bool big_first) {
    const int big = 4, small = 2;
    BasisConfig b;
    b.n_mols = 2;
    b.mol_n_sites = big_first ? std::vector<int>{big, small} : std::vector<int>{small, big};
    b.n_sites = std::max(big, small);          // padded stride
    b.body_positions.assign(static_cast<std::size_t>(b.n_mols) * b.n_sites, {0.0, 0.0, 0.0});
    b.elements.assign(static_cast<std::size_t>(b.n_mols) * b.n_sites, "C");
    for (int m = 0; m < b.n_mols; ++m)
        for (int j = 0; j < b.mol_n_sites[m]; ++j)
            b.body_positions[m * b.n_sites + j] = {1.0 * j, 0.0, 0.0};
    return b;
}

// PET symmetry and expand_rigid_structure must share the op*Z' + m layout with
// real per-component site counts; n_full() equals the expanded atom count only
// when they agree.
TEST_CASE("Expanded co-crystal symmetry describes the atoms produced", "[cocrystal][pet]") {
    for (bool big_first : {true, false}) {
        INFO((big_first ? "larger type first" : "smaller type first"));
        const BasisConfig basis = padded_two_component_basis(big_first);
        const std::vector<int> sites = basis.mol_n_sites;
        REQUIRE(sites[0] != sites[1]);   // a swap cannot hide behind equal sizes

        for (int sg : {2, 14, 19}) {
            INFO("space group " << sg);
            const int nop = get_space_group(sg).n_symops;
            const double cell[6] = {12.0, 12.0, 12.0, 90.0, 90.0, 90.0};
            const double tr[6] = {0.2, 0.2, 0.2, 0.7, 0.7, 0.7};
            const double rv[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            CrystalGeometry ex = expand_rigid_structure(basis, sg, cell, tr, rv, basis.n_mols);

            PetSymmetry sy = expanded_row_symmetry(sg, basis.n_mols, sites);
            REQUIRE(sy.valid());
            REQUIRE(sy.n_indep == 2);
            REQUIRE(sy.n_op == nop * 2);
            // One formula unit in the asymmetric unit, whatever Z' is.
            REQUIRE(sy.n_asym() == sites[0] + sites[1]);
            REQUIRE(sy.n_full() == static_cast<int>(ex.positions.size()));
            REQUIRE(sy.n_full() == nop * (sites[0] + sites[1]));
            for (int op = 0; op < sy.n_op; ++op)
                REQUIRE(sy.indep_of_op[op] == op % 2);
        }
    }
}

// On !valid() the caller falls back to free relaxation.
TEST_CASE("Symmetry with a missing component is rejected", "[cocrystal][pet]") {
    PetSymmetry sy = expanded_row_symmetry(14, 2, {4});   // Z'=2 but one count
    REQUIRE_FALSE(sy.valid());
}
#endif
