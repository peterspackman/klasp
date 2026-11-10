#include "klasp/sgdata.hpp"

#include <gemmi/symmetry.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

// The only translation unit that includes gemmi's symmetry header.

namespace klasp {

namespace {

// Lattice type from gemmi's crystal system. Trigonal folds into HEXAGONAL;
// RHOMBOHEDRAL is reserved for the R-centred rhombohedral setting.
LatticeType lattice_from_gemmi(const gemmi::SpaceGroup& sg) {
    auto cs = sg.crystal_system();
    if (sg.ext == 'R')
        return RHOMBOHEDRAL;
    switch (cs) {
        case gemmi::CrystalSystem::Triclinic:    return TRICLINIC;
        case gemmi::CrystalSystem::Monoclinic:   return MONOCLINIC;
        case gemmi::CrystalSystem::Orthorhombic: return ORTHORHOMBIC;
        case gemmi::CrystalSystem::Tetragonal:   return TETRAGONAL;
        case gemmi::CrystalSystem::Trigonal:     return HEXAGONAL;
        case gemmi::CrystalSystem::Hexagonal:    return HEXAGONAL;
        case gemmi::CrystalSystem::Cubic:        return CUBIC;
    }
    return TRICLINIC;
}

}  // namespace

const SpaceGroupInfo& get_space_group(int sg_num) {
    static SpaceGroupInfo cache[230] = {};
    static bool initialized[230] = {};

    if (sg_num < 1 || sg_num > 230)
        throw std::invalid_argument("Invalid space group number: " + std::to_string(sg_num));

    auto& info = cache[sg_num - 1];
    if (initialized[sg_num - 1])
        return info;

    // Try rhombohedral setting first for R-lattice groups (fewer symops)
    const gemmi::SpaceGroup* sg = nullptr;
    // SGs 146, 148, 155, 160, 161, 166, 167 are R-lattice
    sg = gemmi::find_spacegroup_by_number(sg_num);
    if (sg && sg->ext == 'H') {
        // This is the hexagonal setting of an R-lattice group — find the R setting
        std::string hm = sg->hm;
        const gemmi::SpaceGroup* sg_r = gemmi::find_spacegroup_by_name(
            hm, 0, 0, "R");
        if (sg_r) sg = sg_r;
    }

    if (!sg)
        throw std::invalid_argument("Space group not found: " + std::to_string(sg_num));

    info.number = sg_num;
    info.lattice = lattice_from_gemmi(*sg);

    auto ops = sg->operations();
    auto all_ops = ops.all_ops_sorted();
    info.n_symops = static_cast<int>(all_ops.size());

    for (int i = 0; i < info.n_symops && i < MAX_SYMOPS; ++i) {
        auto& op = all_ops[i];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                info.symops[i].rot[r][c] = static_cast<double>(op.rot[r][c]) / gemmi::Op::DEN;
        for (int d = 0; d < 3; ++d) {
            double t = static_cast<double>(op.tran[d]) / gemmi::Op::DEN;
            info.symops[i].trans[d] = t - std::floor(t);  // wrap to [0, 1)
        }
    }

    initialized[sg_num - 1] = true;
    return info;
}

}  // namespace klasp
