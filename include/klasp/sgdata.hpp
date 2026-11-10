#pragma once

#include <cmath>

namespace klasp {

// The seven lattice types. Not gemmi's CrystalSystem: trigonal groups fold into
// HEXAGONAL, and RHOMBOHEDRAL names the rhombohedral setting of an R-centred
// group, whose strain basis depends on the cell (see build_strain_basis).
// The numeric values cross the device boundary as plain ints.
enum LatticeType {
    TRICLINIC = 0, MONOCLINIC = 1, ORTHORHOMBIC = 2,
    TETRAGONAL = 3, RHOMBOHEDRAL = 4, HEXAGONAL = 5, CUBIC = 6
};

static constexpr int MAX_SYMOPS = 192;

struct SymopData {
    double rot[3][3];
    double trans[3];
};

struct SpaceGroupInfo {
    int number;
    int n_symops;
    LatticeType lattice;
    SymopData symops[MAX_SYMOPS];
};

// Get space group info, preferring the rhombohedral setting for R-lattice
// groups. Results are cached. src/sgdata.cpp is the only place gemmi is
// consulted; keep gemmi out of this header, which device code includes.
const SpaceGroupInfo& get_space_group(int sg_num);

// Access symop rotation and translation by index.
inline void get_symop(const SpaceGroupInfo& sg, int idx,
                      double rot[3][3], double trans[3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            rot[i][j] = sg.symops[idx].rot[i][j];
    for (int i = 0; i < 3; ++i)
        trans[i] = sg.symops[idx].trans[i];
}

// Same as get_symop; the first argument is ignored.
inline void decode_symop(int /*unused*/, double rot[3][3], double trans[3],
                         const SpaceGroupInfo* sg = nullptr, int idx = 0) {
    if (sg) {
        get_symop(*sg, idx, rot, trans);
    }
}

// Find the index of the identity operation in a space group.
inline int find_identity_symop(const SpaceGroupInfo& sg) {
    for (int i = 0; i < sg.n_symops; ++i) {
        bool is_identity = true;
        for (int r = 0; r < 3 && is_identity; ++r)
            for (int c = 0; c < 3 && is_identity; ++c)
                if (std::abs(sg.symops[i].rot[r][c] - (r == c ? 1.0 : 0.0)) > 1e-6)
                    is_identity = false;
        for (int d = 0; d < 3 && is_identity; ++d)
            if (std::abs(sg.symops[i].trans[d]) > 1e-6)
                is_identity = false;
        if (is_identity) return i;
    }
    return 0;  // fallback
}

inline void lattice_free_params(LatticeType lt, int& n_lengths, int& n_angles) {
    // Triclinic default so values outside the switch leave outputs initialised.
    n_lengths = 3;
    n_angles = 3;
    switch (lt) {
        case TRICLINIC:    n_lengths = 3; n_angles = 3; break;
        case MONOCLINIC:   n_lengths = 3; n_angles = 1; break;
        case ORTHORHOMBIC: n_lengths = 3; n_angles = 0; break;
        case TETRAGONAL:   n_lengths = 2; n_angles = 0; break;
        case RHOMBOHEDRAL: n_lengths = 1; n_angles = 1; break;
        case HEXAGONAL:    n_lengths = 2; n_angles = 0; break;
        case CUBIC:        n_lengths = 1; n_angles = 0; break;
    }
}

inline const char* lattice_name(LatticeType lt) {
    switch (lt) {
        case TRICLINIC: return "triclinic";
        case MONOCLINIC: return "monoclinic";
        case ORTHORHOMBIC: return "orthorhombic";
        case TETRAGONAL: return "tetragonal";
        case RHOMBOHEDRAL: return "rhombohedral";
        case HEXAGONAL: return "hexagonal";
        case CUBIC: return "cubic";
    }
    return "unknown";
}

}  // namespace klasp
