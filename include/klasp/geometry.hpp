#pragma once
// The host-side explicit-atom crystal geometry. Everything here is in Bohr.
// Conversions: to/from the database row (`to_flexible`/`from_flexible` in
// database.hpp, Angstrom cell parameters and fractional coordinates) and
// `to_pet_system` (pet_bridge.cpp, PET's Angstrom row-major convention).

#include <array>
#include <string>
#include <vector>

namespace klasp {

// Every atom in the cell, with no rigid-molecule assumption. Rigid candidates
// reach this form via `expand_rigid_structure`.
struct CrystalGeometry {
    // Cartesian positions in Bohr, [N].
    std::vector<std::array<double, 3>> positions;

    // Lattice vectors as columns, in Bohr: cell[i][j] is component i of vector j.
    // PET uses the transpose; to_pet_system is the only place that transposes.
    double cell[3][3] = {{0}};

    // [N] atomic numbers, parallel to `positions`. Always filled.
    std::vector<int> atomic_numbers;

    // [N] element labels, parallel to `positions`, or empty. Filled only for
    // writers (CIF, chemiscope, xyz); atomic numbers are authoritative.
    std::vector<std::string> elements;

    // false = isolated molecule: no periodic boundary, `cell` ignored.
    bool periodic = true;

    int n_atoms() const { return static_cast<int>(atomic_numbers.size()); }
};

}  // namespace klasp
