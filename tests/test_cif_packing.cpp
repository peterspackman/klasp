// Fitting a CIF's molecules onto a reference body, and comparing the resulting
// packings, independently of atom order, origin and space-group description.

#include "kokkos_listener.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "klasp/cif_packing.hpp"
#include "klasp/cif_reader.hpp"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using namespace klasp;
using Catch::Matchers::WithinAbs;

namespace {

// Benzene, Pbca, Z = 4, the molecule on an inversion centre.
const char* kBenzene = R"(data_benzene
_cell_length_a 7.460
_cell_length_b 9.660
_cell_length_c 7.030
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'P b c a'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C -0.0569 0.1387 -0.0054
C2 C -0.1387 0.0460 0.0942
C3 C -0.0816 -0.0925 0.0996
H1 H -0.0999 0.2433 -0.0096
H2 H -0.2434 0.0810 0.1671
H3 H -0.1435 -0.1621 0.1770
)";

// The same crystal, atoms listed in a different order.
const char* kBenzeneShuffled = R"(data_benzene_shuffled
_cell_length_a 7.460
_cell_length_b 9.660
_cell_length_c 7.030
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'P b c a'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
H3 H -0.1435 -0.1621 0.1770
C3 C -0.0816 -0.0925 0.0996
H1 H -0.0999 0.2433 -0.0096
C1 C -0.0569 0.1387 -0.0054
H2 H -0.2434 0.0810 0.1671
C2 C -0.1387 0.0460 0.0942
)";

// Two minima of one acetic acid packing from a klasp search, P2_1/c in
// different settings. Their clusters are related by an inversion, so the
// overlay has to be improper.
const char* kAceticA = R"(data_final-fix-QR-14-2920-2
_cell_length_a 7.460106
_cell_length_b 4.504417
_cell_length_c 13.686060
_cell_angle_alpha 90.0000
_cell_angle_beta 136.8924
_cell_angle_gamma 90.0000
_space_group_IT_number 14
_space_group_name_H-M_alt 'P 1 21/c 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1_1 C 0.80978062 0.66482000 0.31474788
C2_1 C 0.65324641 0.78108589 0.16980084
H3_1 H 0.87394389 0.72080626 0.47649689
H4_1 H 0.71168037 0.66529309 0.12720923
H5_1 H 0.44559561 0.73953342 0.09661652
H6_1 H 0.68346872 1.01657672 0.17051732
O7_1 O 0.76659441 0.80368292 0.38071266
O8_1 O 0.96218102 0.45703460 0.36917739

loop_
_space_group_symop_id
_space_group_symop_operation_xyz
1 'x, y, z'
2 '-x, -y, -z'
3 '-x, y+1/2, -z+1/2'
4 'x, -y+1/2, z+1/2'
)";

const char* kAceticB = R"(data_final-fix-QR-14-9147-2
_cell_length_a 7.579476
_cell_length_b 4.367129
_cell_length_c 10.793504
_cell_angle_alpha 90.0000
_cell_angle_beta 61.8885
_cell_angle_gamma 90.0000
_space_group_IT_number 14
_space_group_name_H-M_alt 'P 1 21/c 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1_1 C 0.68064221 0.33765254 0.81695569
C2_1 C 0.81526077 0.22284969 0.67395207
H3_1 H 0.42083433 0.28040045 0.97266380
H4_1 H 0.94977434 0.36103892 0.62928989
H5_1 H 0.86213407 -0.01080126 0.67528145
H6_1 H 0.74773668 0.23667848 0.60526025
O7_1 O 0.50609376 0.19804728 0.87815573
O8_1 O 0.72520662 0.53864526 0.87561081

loop_
_space_group_symop_id
_space_group_symop_operation_xyz
1 'x, y, z'
2 '-x, -y, -z'
3 '-x, y+1/2, -z+1/2'
4 'x, -y+1/2, z+1/2'
)";

// One minimum of blind-test target VI in two P-1 settings, from a klasp search:
// b and c swapped, alpha 119.1 against 109.0 degrees.
const char* kAceticMirrorA = R"(data_final-fix-QR-14-16549-2
_cell_length_a 4.006306
_cell_length_b 7.184164
_cell_length_c 11.252175
_cell_angle_alpha 90.0000
_cell_angle_beta 88.9209
_cell_angle_gamma 90.0000
_space_group_IT_number 14
_space_group_name_H-M_alt 'P 1 21/c 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1_1 C 0.78075055 0.30251481 0.91557198
C2_1 C 0.63325895 0.14830913 0.84690115
H3_1 H 1.00843623 0.35499954 1.05931898
H4_1 H 0.51504026 0.20627806 0.76944562
H5_1 H 0.82118982 0.05177360 0.81424985
H6_1 H 0.44662935 0.07218139 0.89802570
O7_1 O 0.91351379 0.24906667 1.01618756
O8_1 O 0.78674149 0.46431949 0.88287058

loop_
_space_group_symop_id
_space_group_symop_operation_xyz
1 'x, y, z'
2 '-x, -y, -z'
3 '-x, y+1/2, -z+1/2'
4 'x, -y+1/2, z+1/2'
)";

const char* kAceticMirrorB = R"(data_final-fix-QR-14-3734-2
_cell_length_a 4.006284
_cell_length_b 7.572150
_cell_length_c 11.029158
_cell_angle_alpha 90.0000
_cell_angle_beta 93.1154
_cell_angle_gamma 90.0000
_space_group_IT_number 14
_space_group_name_H-M_alt 'P 1 21/c 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1_1 C 0.84352968 0.20557654 0.91070407
C2_1 C 0.72613698 0.36387300 0.84216503
H3_1 H 1.12717667 0.13316714 1.04637032
H4_1 H 0.56930742 0.32130726 0.76455475
H5_1 H 0.57318366 0.44847220 0.89559829
H6_1 H 0.93091722 0.44057749 0.80943835
O7_1 O 1.05455103 0.24126652 1.00320270
O8_1 O 0.75247892 0.05533027 0.88671857

loop_
_space_group_symop_id
_space_group_symop_operation_xyz
1 'x, y, z'
2 '-x, -y, -z'
3 '-x, y+1/2, -z+1/2'
4 'x, -y+1/2, z+1/2'
)";

const char* kAceticRegrowA = R"(data_final-fix-QR-14-14424-2
_cell_length_a 9.543623
_cell_length_b 6.235194
_cell_length_c 5.298777
_cell_angle_alpha 90.0000
_cell_angle_beta 89.6555
_cell_angle_gamma 90.0000
_space_group_IT_number 14
_space_group_name_H-M_alt 'P 1 21/c 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1_1 C 0.18906123 0.99608441 0.12907076
C2_1 C 0.33162963 0.99789311 0.23746965
H3_1 H 0.09140257 1.00483232 -0.18421922
H4_1 H 0.32235719 1.00093609 0.44124195
H5_1 H 0.38979844 0.85422143 0.18994387
H6_1 H 0.39154928 1.13681200 0.17846640
O7_1 O 0.18764666 1.00614887 -0.11918732
O8_1 O 0.08165029 0.98291234 0.25389699

loop_
_space_group_symop_id
_space_group_symop_operation_xyz
1 'x, y, z'
2 '-x, -y, -z'
3 '-x, y+1/2, -z+1/2'
4 'x, -y+1/2, z+1/2'
)";

const char* kAceticRegrowB = R"(data_final-fix-QR-5-9536-2
_cell_length_a 5.290055
_cell_length_b 6.243002
_cell_length_c 9.559211
_cell_angle_alpha 90.0000
_cell_angle_beta 90.1450
_cell_angle_gamma 90.0000
_space_group_IT_number 5
_space_group_name_H-M_alt 'C 1 2 1'

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1_1 C 0.86926392 0.06064342 0.81084814
C2_1 C 0.75921882 0.06085227 0.66833710
H3_1 H 1.18434799 0.05804253 0.90872421
H4_1 H 0.55548781 0.05106465 0.67718438
H5_1 H 0.79986312 0.20790371 0.61270571
H6_1 H 0.82365697 -0.07362437 0.60637384
O7_1 O 1.11821682 0.05834398 0.81253158
O8_1 O 0.74496344 0.06578431 0.91804596

loop_
_space_group_symop_id
_space_group_symop_operation_xyz
1 'x, y, z'
2 '-x, y, -z'
3 '-x+1/2, y+1/2, -z'
4 'x+1/2, y+1/2, z'
)";

const char* kTargetVIa = R"(data_sg2_seed4631_-162.90
_cell_length_a 9.447851
_cell_length_b 6.962606
_cell_length_c 17.094636
_cell_angle_alpha 119.1124
_cell_angle_beta 40.5976
_cell_angle_gamma 123.6981
_space_group_IT_number 2
_space_group_name_H-M_alt 'P -1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
H1_1 H -0.21911898 0.51721176 0.38983908
C2_1 C 0.04071243 0.82044167 0.34872208
C3_1 C -0.29573967 0.74228202 0.38986010
C4_1 C -0.22252629 0.96554757 0.37555260
H5_1 H -0.32271335 1.02002263 0.38521875
C6_1 C -0.01531079 1.11821770 0.34871980
H7_1 H 0.04353098 1.29614202 0.33830747
C8_1 C 0.11515726 1.05311847 0.33515050
H9_1 H 0.27056614 1.17432181 0.31764735
C10_1 C 0.57641312 0.80057654 0.14395319
C11_1 C 0.70616426 0.98967403 0.08326150
H12_1 H 0.68755926 1.15842652 0.13443016
C13_1 C 0.86182814 0.96070049 -0.04236460
H14_1 H 0.96470980 1.10817622 -0.09036758
C15_1 C 0.88677176 0.74427320 -0.10608155
H16_1 H 1.00896243 0.72224422 -0.20416393
C17_1 C 0.75707897 0.55585618 -0.04456129
H18_1 H 0.77857763 0.38730327 -0.09453493
C19_1 C 0.60167970 0.58348752 0.08066740
H20_1 H 0.50148692 0.43951573 0.13016858
N21_1 N 0.13037819 0.71231860 0.33989439
N22_1 N -0.16553296 0.67912642 0.37652321
N23_1 N -0.49865061 0.57602371 0.42167760
H24_1 H -0.58018230 0.61341639 0.41681100
H25_1 H -0.50711354 0.40836357 0.39514913
O26_1 O 0.37995898 0.68848653 0.33904722
O27_1 O 0.42222017 1.07810081 0.34533779
S28_1 S 0.37412556 0.83271948 0.30466315

loop_
_symmetry_equiv_pos_site_id
_symmetry_equiv_pos_as_xyz
1 'x, y, z'
2 '-x, -y, -z'
)";

const char* kTargetVIb = R"(data_sg2_seed2710_-162.89
_cell_length_a 17.094849
_cell_length_b 11.911366
_cell_length_c 6.962621
_cell_angle_alpha 109.0473
_cell_angle_beta 119.1130
_cell_angle_gamma 123.2650
_space_group_IT_number 2
_space_group_name_H-M_alt 'P -1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
H1_1 H -0.32929148 0.28086822 -0.20192808
C2_1 C -0.11057753 0.54069587 0.36112800
C3_1 C -0.40588861 0.20425251 -0.05347302
C4_1 C -0.34698189 0.27746639 0.24300604
H5_1 H -0.43750132 0.17728287 0.19729819
C6_1 C -0.16660035 0.48467764 0.60288626
H7_1 H -0.11817030 0.54351973 0.83965239
C8_1 C -0.04970345 0.61514119 0.66824978
H9_1 H 0.08820135 0.77054682 0.94485790
C10_1 C 0.22035450 1.07638890 0.87695095
C11_1 C 0.28941565 1.20614077 1.19579911
H12_1 H 0.32197846 1.18753530 1.34594695
C13_1 C 0.31945616 1.36180587 1.32248841
H14_1 H 0.37433616 1.46468808 1.57284539
C15_1 C 0.28068382 1.38674995 1.13100416
H16_1 H 0.30479423 1.50894155 1.23116502
C17_1 C 0.21250956 1.25705639 0.81289479
H18_1 H 0.18403533 1.27855535 0.66584003
C19_1 C 0.18233625 1.10165590 0.68512790
H20_1 H 0.13164325 1.00146243 0.44096360
N21_1 N -0.02974113 0.63035786 0.34266633
N22_1 N -0.28902062 0.33445475 0.01357282
N23_1 N -0.57698125 0.00134525 -0.42263741
H24_1 H -0.66337784 -0.08018293 -0.46677256
H25_1 H -0.61197242 -0.00711762 -0.59876081
O26_1 O 0.21898851 0.87992969 0.56840492
O27_1 O 0.26754171 0.92219300 1.00028267
S28_1 S 0.17877344 0.87409985 0.70680775

loop_
_symmetry_equiv_pos_site_id
_symmetry_equiv_pos_as_xyz
1 'x, y, z'
2 '-x, -y, -z'
)";

CifStructure read_one(const char* text, const char* name) {
    const auto s = read_cif_string(text, name);
    REQUIRE(s.size() == 1);
    return s[0];
}

}  // namespace

TEST_CASE("a fit recovers the placement of a shuffled molecule", "[cif][packing]") {
    const CifStructure s = read_one(kBenzene, "benzene");
    REQUIRE(s.molecules.size() == 4);

    const RigidBody body = body_from_molecule(s.molecules[0], /*heavy_only=*/true);
    REQUIRE(body.m == 6);

    // Every molecule of the cell is the same benzene, so every one must fit.
    for (const CifMolecule& m : s.molecules) {
        const BodyFit fit = fit_to_body(body, m.positions.data(), m.numbers.data(),
                                        m.size(), /*heavy_only=*/true);
        REQUIRE(fit.ok);
        CHECK(fit.rmsd < 1e-6);
        REQUIRE(fit.permutation.size() == 6);

        // Applying the fit to the body's atoms must land on the molecule's, in
        // permutation order.
        for (int k = 0; k < body.m; ++k) {
            double moved[3];
            for (int d = 0; d < 3; ++d) {
                moved[d] = fit.placed.centroid[d];
                for (int j = 0; j < 3; ++j)
                    moved[d] += fit.placed.orient[d * 3 + j] * body.positions[k * 3 + j];
            }
            const int t = fit.permutation[k];
            for (int d = 0; d < 3; ++d)
                CHECK_THAT(moved[d], WithinAbs(m.positions[t * 3 + d], 1e-6));
        }
    }
}

TEST_CASE("a molecule with the wrong atoms does not fit", "[cif][packing]") {
    const CifStructure s = read_one(kBenzene, "benzene");
    const RigidBody body = body_from_molecule(s.molecules[0], /*heavy_only=*/true);

    // Benzene's six carbons against five of them.
    const CifMolecule& m = s.molecules[0];
    std::vector<double> pos(m.positions.begin(), m.positions.begin() + 5 * 3);
    std::vector<int> num(m.numbers.begin(), m.numbers.begin() + 5);
    const BodyFit fit = fit_to_body(body, pos.data(), num.data(), 5, /*heavy_only=*/false);
    CHECK_FALSE(fit.ok);

    std::vector<int> nitrogen(m.numbers.size(), 7);
    const BodyFit wrong = fit_to_body(body, m.positions.data(), nitrogen.data(),
                                      m.size(), /*heavy_only=*/true);
    CHECK_FALSE(wrong.ok);
}

TEST_CASE("a P1 description collapses to one environment", "[cif][packing]") {
    // Written as P1, the file carries no symmetry; geometry alone must identify
    // the equivalent environments.
    const CifStructure s = read_one(kBenzene, "benzene");
    const RigidBody body = body_from_molecule(s.molecules[0], /*heavy_only=*/true);
    CifPacking p = place_cif_structure(s, body, /*heavy_only=*/true);
    REQUIRE(p.valid);
    REQUIRE(p.molecules.size() == 4);
    // Before the geometric pass every molecule is its own environment.
    CHECK(p.representatives.size() == 4);

    CompackOptions opts;
    opts.n_neighbours = 12;
    // Benzene sits on an inversion centre, so half the molecules need an improper
    // overlay.
    opts.allow_inversion = true;
    find_representatives(p, body, opts);
    CHECK(p.representatives.size() == 1);
}

TEST_CASE("atom order does not change the packing", "[cif][packing]") {
    const CifStructure a = read_one(kBenzene, "benzene");
    const CifStructure b = read_one(kBenzeneShuffled, "shuffled");

    const RigidBody body = body_from_molecule(a.molecules[0], /*heavy_only=*/true);
    CompackOptions opts;
    opts.n_neighbours = 12;
    opts.allow_inversion = true;

    CifPacking pa = place_cif_structure(a, body, /*heavy_only=*/true);
    CifPacking pb = place_cif_structure(b, body, /*heavy_only=*/true);
    REQUIRE(pa.valid);
    REQUIRE(pb.valid);
    CHECK(pb.worst_fit < 1e-6);

    find_representatives(pa, body, opts);
    find_representatives(pb, body, opts);

    const CompackResult r = compare_packings(pa, pb, body, opts);
    REQUIRE(r.valid);
    CHECK(r.n_matched == 12);
    CHECK(r.same_packing);
}

TEST_CASE("a structure is the same packing as itself", "[cif][packing]") {
    const CifStructure s = read_one(kBenzene, "benzene");
    const RigidBody body = body_from_molecule(s.molecules[0], /*heavy_only=*/true);
    CompackOptions opts;
    opts.n_neighbours = 12;
    opts.allow_inversion = true;

    CifPacking p = place_cif_structure(s, body, /*heavy_only=*/true);
    REQUIRE(p.valid);
    find_representatives(p, body, opts);

    const CompackResult r = compare_packings(p, p, body, opts);
    REQUIRE(r.valid);
    CHECK(r.n_matched == 12);
    CHECK(r.same_packing);
    // Not exactly zero: SVD orientation fits are orthogonal only to rounding, and
    // the closed-form RMSD subtracts nearly equal traces. Varies by platform.
    CHECK(r.rmsd < 1e-4);
}

TEST_CASE("a framework is refused rather than compared", "[cif][packing]") {
    const char* framework = R"(data_framework
_cell_length_a 1.5
_cell_length_b 1.5
_cell_length_c 1.5
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.0 0.0 0.0
)";
    const CifStructure s = read_one(framework, "framework");
    const CifStructure b = read_one(kBenzene, "benzene");
    const RigidBody body = body_from_molecule(b.molecules[0], /*heavy_only=*/true);

    const CifPacking p = place_cif_structure(s, body, /*heavy_only=*/true);
    CHECK_FALSE(p.valid);
    CHECK(p.problem.find("periodic") != std::string::npos);
}


TEST_CASE("all pairs agree with one pair at a time, multi-environment structures included",
          "[cif][packing][sweep]") {
    std::string wider = kBenzene;
    wider.replace(wider.find("_cell_length_b 9.660"), 20, "_cell_length_b 11.20");
    const CifStructure a = read_one(kBenzene, "benzene");
    const RigidBody body = body_from_molecule(a.molecules[0], /*heavy_only=*/true);
    CompackOptions opts;
    opts.n_neighbours = 12;
    opts.allow_inversion = true;

    std::vector<CifPacking> packings;
    for (const char* text : {kBenzene, kBenzeneShuffled, wider.c_str(), kBenzene}) {
        packings.push_back(place_cif_structure(read_one(text, "b"), body, /*heavy_only=*/true));
        REQUIRE(packings.back().valid);
        find_representatives(packings.back(), body, opts);
    }
    // Every molecule its own environment, so these pairs take the one-pair path.
    packings[1].representatives = {0, 1, 2, 3};
    packings[3].representatives = {0, 1, 2, 3};

    std::vector<int> nm;
    std::vector<unsigned char> same;
    compare_all_packings(packings, body, opts, {}, nm, same);
    const int n = static_cast<int>(packings.size());
    REQUIRE((int)nm.size() == n * n);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const CompackResult r = compare_packings(packings[i], packings[j], body, opts);
            INFO("pair " << i << "," << j);
            CHECK((same[(size_t)i * n + j] != 0) == (r.valid && r.same_packing));
            CHECK(same[(size_t)j * n + i] == same[(size_t)i * n + j]);
            CHECK(nm[(size_t)j * n + i] == nm[(size_t)i * n + j]);
            if (packings[i].representatives.size() > 1)
                CHECK(nm[(size_t)i * n + j] == r.n_matched);
            else
                CHECK(std::abs(nm[(size_t)i * n + j] - r.n_matched) <= 1);
        }
    CHECK(same[0 * n + 1]);
    CHECK_FALSE(same[0 * n + 2]);
}

TEST_CASE("one structure in two cell settings is the same packing", "[cif][packing]") {
    const CifStructure a_cif = read_one(kTargetVIa, "vi_a");
    const CifStructure b_cif = read_one(kTargetVIb, "vi_b");
    const RigidBody body = body_from_molecule(a_cif.molecules[0], /*heavy_only=*/true);
    CompackOptions opts;
    opts.allow_inversion = true;

    CifPacking a = place_cif_structure(a_cif, body, /*heavy_only=*/true);
    CifPacking b = place_cif_structure(b_cif, body, /*heavy_only=*/true);
    REQUIRE(a.valid);
    REQUIRE(b.valid);
    find_representatives(a, body, opts);
    find_representatives(b, body, opts);
    CHECK(compare_packings(a, b, body, opts).n_matched == opts.n_neighbours);
    CHECK(compare_packings(b, a, body, opts).n_matched == opts.n_neighbours);
}

TEST_CASE("clusters related by an inversion are the same packing when inversion is allowed",
          "[cif][packing]") {
    const CifStructure a_cif = read_one(kAceticA, "acetic_a");
    const CifStructure b_cif = read_one(kAceticB, "acetic_b");
    const RigidBody body = body_from_molecule(a_cif.molecules[0], /*heavy_only=*/true);
    CompackOptions opts;
    opts.allow_inversion = true;

    CifPacking a = place_cif_structure(a_cif, body, /*heavy_only=*/true);
    CifPacking b = place_cif_structure(b_cif, body, /*heavy_only=*/true);
    REQUIRE(a.valid);
    REQUIRE(b.valid);
    find_representatives(a, body, opts);
    find_representatives(b, body, opts);
    CHECK(compare_packings(a, b, body, opts).n_matched == opts.n_neighbours);
    CHECK(compare_packings(b, a, body, opts).n_matched == opts.n_neighbours);
}

TEST_CASE("a planar molecule's mirror-image overlay is kept through the refit", "[cif][packing]") {
    // Acetic acid's heavy atoms are coplanar, so a refit allowing improper
    // transforms can flip back to proper. These two match only through the mirror.
    const CifStructure a_cif = read_one(kAceticMirrorA, "acetic_mirror_a");
    const CifStructure b_cif = read_one(kAceticMirrorB, "acetic_mirror_b");
    const RigidBody body = body_from_molecule(a_cif.molecules[0], /*heavy_only=*/true);
    CompackOptions opts;
    opts.allow_inversion = true;

    CifPacking a = place_cif_structure(a_cif, body, /*heavy_only=*/true);
    CifPacking b = place_cif_structure(b_cif, body, /*heavy_only=*/true);
    REQUIRE(a.valid);
    REQUIRE(b.valid);
    find_representatives(a, body, opts);
    find_representatives(b, body, opts);
    const CompackResult ab = compare_packings(a, b, body, opts);
    CHECK(ab.n_matched == opts.n_neighbours);
    CHECK(ab.inverted);
    CHECK(compare_packings(b, a, body, opts).n_matched == opts.n_neighbours);
}

TEST_CASE("a tight tolerance still finds an overlay its seeds start just outside",
          "[cif][packing]") {
    // Every neighbour fits within 0.09 A, but the seed that lays the centrals on
    // each other leaves them all beyond 0.1 A.
    const CifStructure a_cif = read_one(kAceticRegrowA, "acetic_regrow_a");
    const CifStructure b_cif = read_one(kAceticRegrowB, "acetic_regrow_b");
    const RigidBody body = body_from_molecule(a_cif.molecules[0], /*heavy_only=*/true);
    CompackOptions opts;
    opts.allow_inversion = true;
    opts.match_tolerance = 0.1;

    CifPacking a = place_cif_structure(a_cif, body, /*heavy_only=*/true);
    CifPacking b = place_cif_structure(b_cif, body, /*heavy_only=*/true);
    REQUIRE(a.valid);
    REQUIRE(b.valid);
    find_representatives(a, body, opts);
    find_representatives(b, body, opts);
    CHECK(compare_packings(a, b, body, opts).n_matched == opts.n_neighbours);
    CHECK(compare_packings(b, a, body, opts).n_matched == opts.n_neighbours);
}
