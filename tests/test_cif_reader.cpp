// Reading a crystal out of a CIF: expansion, bonding, unwrapping.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "klasp/cif_reader.hpp"

#include <cmath>
#include <string>

using namespace klasp;
using Catch::Matchers::WithinAbs;

namespace {

double norm3(const double* p) { return std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]); }

// Longest atom-to-centroid distance. A molecule that failed to unwrap is spread
// over the cell, so this is huge; a whole one keeps it near the molecular radius.
double max_extent(const CifMolecule& m) {
    double worst = 0.0;
    for (int i = 0; i < m.size(); ++i) {
        double d[3];
        for (int k = 0; k < 3; ++k) d[k] = m.positions[i * 3 + k] - m.centroid[k];
        worst = std::max(worst, norm3(d));
    }
    return worst;
}

// P1, 10 A cube, one N2 lying across the x = 0 face: 0.2 A inside on one side
// and 1.0 A outside on the other, so the two are 1.2 A apart THROUGH the face
// and 8.8 A apart within the box.
const char* kSplitDiatomic = R"(data_split
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
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
N1 N 0.02 0.5 0.5
N2 N 0.90 0.5 0.5
)";

// Urea as it is usually distributed: the full operator set AND a whole molecule
// under _atom_site, so half the listed sites are symmetry images of the other
// half. Z = 2, 8 atoms each.
const char* kUrea = R"(data_urea
_symmetry_space_group_name_H-M   'P -4 21 m'
_symmetry_Int_Tables_number      113
loop_
_symmetry_equiv_pos_as_xyz
x,y,z
-y,x,-z
-x,-y,z
y,-x,-z
1/2-x,1/2+y,-z
1/2+y,1/2+x,z
1/2+x,1/2-y,-z
1/2-y,1/2-x,z
_cell_length_a                   5.582
_cell_length_b                   5.582
_cell_length_c                   4.686
_cell_angle_alpha                90
_cell_angle_beta                 90
_cell_angle_gamma                90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.00000 0.50000 0.32720
H1 H 0.26900 0.76900 0.27900
H2 H 0.14200 0.64200 -0.02800
N1 N 0.14550 0.64550 0.18000
O1 O 0.00000 0.50000 0.59660
N1B N -0.14550 0.35450 0.18000
H1B H -0.26900 0.23100 0.27900
H2B H -0.14200 0.35800 -0.02800
)";

// One carbon per cell in a 1.5 A cube: closer than C + C + tolerance (1.76 A)
// along each axis, so every atom bonds to its own image and the component is
// infinite in all three directions. A framework, not a molecule.
const char* kFramework = R"(data_framework
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

// P-1 with a water whose oxygen sits exactly on the inversion centre at the
// origin. The identity and the inversion both put it there, and it must appear
// once.
const char* kSpecialPosition = R"(data_special
_cell_length_a 8.0
_cell_length_b 8.0
_cell_length_c 8.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'P -1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
O1 O 0.0 0.0 0.0
)";

// Benzene, Pbca, Z = 4. A real molecular crystal with a general-position
// molecule, so nothing merges and the count is exactly 4 x 12.
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

}  // namespace

TEST_CASE("a molecule split across a cell face is read whole", "[cif]") {
    std::string err;
    const auto structures = read_cif_string(kSplitDiatomic, "split", {}, &err);
    REQUIRE(err.empty());
    REQUIRE(structures.size() == 1);
    const CifStructure& s = structures[0];

    REQUIRE(s.n_atoms == 2);
    REQUIRE(s.molecules.size() == 1);
    const CifMolecule& m = s.molecules[0];
    REQUIRE(m.size() == 2);
    REQUIRE_FALSE(m.periodic);

    // The two nitrogens are 1.2 A apart, not 8.8.
    double d[3];
    for (int k = 0; k < 3; ++k) d[k] = m.positions[k] - m.positions[3 + k];
    CHECK_THAT(norm3(d), WithinAbs(1.2, 1e-6));
    CHECK(max_extent(m) < 1.0);
}

TEST_CASE("cell, symmetry and volume come through", "[cif]") {
    const auto structures = read_cif_string(kUrea, "urea");
    REQUIRE(structures.size() == 1);
    const CifStructure& s = structures[0];

    CHECK(s.name == "urea");
    CHECK_THAT(s.cell.params[0], WithinAbs(5.582, 1e-9));
    CHECK_THAT(s.cell.params[2], WithinAbs(4.686, 1e-9));
    CHECK_THAT(s.cell.params[3], WithinAbs(90.0, 1e-9));   // degrees, not radians
    CHECK(s.space_group == 113);
    CHECK(s.n_symops == 8);
    CHECK_THAT(s.cell.volume, WithinAbs(5.582 * 5.582 * 4.686, 1e-6));
}

TEST_CASE("an over-specified asymmetric unit does not double the cell", "[cif]") {
    const auto structures = read_cif_string(kUrea, "urea");
    REQUIRE(structures.size() == 1);
    const CifStructure& s = structures[0];

    // Z = 2 urea: 16 atoms, two whole molecules of 8 (28 without the global merge).
    CHECK(s.n_atoms == 16);
    REQUIRE(s.molecules.size() == 2);
    for (const CifMolecule& m : s.molecules) {
        CHECK(m.size() == 8);
        CHECK_FALSE(m.periodic);
        CHECK(max_extent(m) < 2.5);        // urea is small; a torn one would be ~3 A+
    }
    // The listed molecule really was over-specified, and the reader noticed.
    CHECK(s.n_coincident > 0);
}

TEST_CASE("an atom on a special position appears once", "[cif]") {
    const auto structures = read_cif_string(kSpecialPosition, "special");
    REQUIRE(structures.size() == 1);
    const CifStructure& s = structures[0];

    CHECK(s.n_symops == 2);      // identity and the inversion
    CHECK(s.n_atoms == 1);       // but they put the oxygen in the same place
    REQUIRE(s.molecules.size() == 1);
    CHECK(s.molecules[0].size() == 1);
}

TEST_CASE("a framework is flagged rather than passed off as a molecule", "[cif]") {
    const auto structures = read_cif_string(kFramework, "framework");
    REQUIRE(structures.size() == 1);
    const CifStructure& s = structures[0];

    REQUIRE(s.molecules.size() == 1);
    CHECK(s.molecules[0].periodic);
}

TEST_CASE("a general-position molecular crystal expands to Z whole molecules", "[cif]") {
    const auto structures = read_cif_string(kBenzene, "benzene");
    REQUIRE(structures.size() == 1);
    const CifStructure& s = structures[0];

    CHECK(s.space_group == 61);          // Pbca
    CHECK(s.n_symops == 8);
    // Z' = 1/2: the molecule sits on an inversion centre, so the file lists half
    // of it and the cell holds 4 whole molecules of 12 atoms.
    CHECK(s.n_atoms == 48);
    REQUIRE(s.molecules.size() == 4);
    for (const CifMolecule& m : s.molecules) {
        CHECK(m.size() == 12);
        CHECK_FALSE(m.periodic);
        CHECK(max_extent(m) < 3.0);
    }
}

TEST_CASE("partial occupancy sites are dropped and counted", "[cif]") {
    std::string text = R"(data_disordered
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
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
_atom_site_occupancy
C1 C 0.10 0.10 0.10 1.00
C2 C 0.20 0.10 0.10 0.30
)";
    const auto structures = read_cif_string(text, "disordered");
    REQUIRE(structures.size() == 1);
    CHECK(structures[0].n_sites == 1);
    CHECK(structures[0].n_partial_occupancy == 1);
    CHECK(structures[0].n_atoms == 1);
}

TEST_CASE("a block without a cell is skipped, not an error", "[cif]") {
    std::string text = R"(data_notacrystal
_journal_name_full 'Some Journal'
)";
    std::string err;
    const auto structures = read_cif_string(text, "notacrystal", {}, &err);
    CHECK(structures.empty());
    CHECK_FALSE(err.empty());
}

TEST_CASE("every data block of a multi-block file is read", "[cif]") {
    const std::string text = std::string(kSplitDiatomic) + "\n" + kSpecialPosition;
    const auto structures = read_cif_string(text, "multi");
    REQUIRE(structures.size() == 2);
    CHECK(structures[0].name == "split");
    CHECK(structures[1].name == "special");
}

TEST_CASE("molecules come out whole in a badly skewed cell", "[cif]") {
    // Ethyl carbamate in P-1 as a CSP run left it: beta of 20.7 degrees and a
    // 32 A axis, so bonded neighbours sit two cells apart along a.
    const char* text = R"(data_skewed
_cell_length_a 32.246253
_cell_length_b 6.748687
_cell_length_c 4.292466
_cell_angle_alpha 98.3960
_cell_angle_beta 20.6742
_cell_angle_gamma 112.0710
_space_group_name_H-M_alt 'P -1'
loop_
_space_group_symop_operation_xyz
'x, y, z'
'-x, -y, -z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.35173494 1.12316233 -0.15288310
C2 C 0.31167183 0.86781184 -0.02921089
C3 C 0.10789608 0.39443466 1.42147026
H4 H 0.45758375 1.31494640 -0.99589202
H5 H 0.27856524 1.15328184 0.27346611
H6 H 0.41800951 1.18535091 -0.36894614
H7 H 0.25380663 0.82088048 0.12879451
H8 H 0.40391655 0.86164638 -0.62395778
H9 H -0.07366587 0.23940776 2.96680105
H10 H -0.07276478 0.00675151 2.76732509
N11 N -0.02685141 0.19730022 2.48493292
O12 O 0.16571594 0.64364064 1.11866020
O13 O 0.17092933 0.34671459 0.79482220
)";
    const auto structures = read_cif_string(text, "skewed");
    REQUIRE(structures.size() == 1);
    const CifStructure& s = structures[0];
    REQUIRE(s.molecules.size() == 2);
    for (const auto& m : s.molecules) {
        CHECK(m.numbers.size() == 13);
        CHECK_FALSE(m.periodic);
    }
}
