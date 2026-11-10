#include "klasp/symmetry_ops.hpp"

#include "klasp/sgdata.hpp"

namespace klasp {

std::vector<int> identity_first_order(const SpaceGroupInfo& sg) {
    const int unity_idx = find_identity_symop(sg);
    std::vector<int> order;
    order.reserve(static_cast<std::size_t>(sg.n_symops));
    order.push_back(unity_idx);
    for (int i = 0; i < sg.n_symops; ++i)
        if (i != unity_idx) order.push_back(i);
    return order;
}

void fill_symmetry_ops(SymmetryOps& ops, const SpaceGroupInfo& sg,
                       int n_crystal_ops, int n_indep,
                       const std::vector<int>& site_counts, int default_sites) {
    auto h_sr = Kokkos::create_mirror_view(ops.sym_rotations);
    auto h_st = Kokkos::create_mirror_view(ops.sym_translations);
    auto h_si = Kokkos::create_mirror_view(ops.sym_indep_index);
    auto h_ns = Kokkos::create_mirror_view(ops.mol_site_count);

    const std::vector<int> order = identity_first_order(sg);

    // Interleave: each crystal op applied to each molecule.
    int flat = 0;
    for (int oi = 0; oi < n_crystal_ops; ++oi) {
        double rot[3][3], trans[3];
        get_symop(sg, order[static_cast<std::size_t>(oi)], rot, trans);
        for (int m = 0; m < n_indep; ++m) {
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    h_sr(flat, r * 3 + c) = rot[r][c];
            for (int d = 0; d < 3; ++d)
                h_st(flat, d) = trans[d];
            h_si(flat) = m;
            h_ns(flat) = (m < static_cast<int>(site_counts.size()))
                       ? site_counts[static_cast<std::size_t>(m)] : default_sites;
            ++flat;
        }
    }

    Kokkos::deep_copy(ops.sym_rotations,    h_sr);
    Kokkos::deep_copy(ops.sym_translations, h_st);
    Kokkos::deep_copy(ops.sym_indep_index,  h_si);
    Kokkos::deep_copy(ops.mol_site_count,   h_ns);

    ops.n_indep  = n_indep;
    ops.n_symops = flat;
}

SymmetryOps crystal_symmetry_ops(const SpaceGroupInfo& sg) {
    SymmetryOps ops;
    allocate_symmetry_ops(ops, sg.n_symops);
    // One copy per crystal operator; sym_indep_index and mol_site_count are zero.
    fill_symmetry_ops(ops, sg, sg.n_symops, /*n_indep=*/1, {}, /*default_sites=*/0);
    return ops;
}

}  // namespace klasp
