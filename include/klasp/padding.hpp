#pragma once

#include "klasp/types.hpp"

#include <vector>

namespace klasp {

// Co-crystal atom arrays use one stride (the largest component's site count) with
// smaller components padded; atom address is `mol * stride + site`. Padded slots
// must never be treated as atoms, so every loop tests them through these helpers.

// Is `site` of molecule `mol` padding? `nsites(mol)` is the real site count
// (MoleculeData::mol_site_count or its workspace mirror).
template <typename NsView>
KOKKOS_INLINE_FUNCTION
bool site_is_padding(int mol, int site, const NsView& nsites) {
    return site >= nsites(mol);
}

// Same for a flat atom index; `stride` is n_sites_per_mol.
template <typename NsView>
KOKKOS_INLINE_FUNCTION
bool atom_is_padding(int atom, int stride, const NsView& nsites) {
    return (atom % stride) >= nsites(atom / stride);
}

// Flat indices of the real atoms, in order. `site_count` is per molecule and
// `stride` is n_sites_per_mol; the result has size sum(site_count).
inline std::vector<int> real_atom_indices(int stride,
                                          const std::vector<int>& site_count) {
    std::vector<int> out;
    out.reserve(static_cast<size_t>(stride) * site_count.size());
    for (size_t mol = 0; mol < site_count.size(); ++mol) {
        const int ns = site_count[mol] > 0 ? site_count[mol] : stride;
        for (int s = 0; s < ns && s < stride; ++s)
            out.push_back(static_cast<int>(mol) * stride + s);
    }
    return out;
}

// Host copy of a device site-count view.
template <typename NsView>
inline std::vector<int> host_site_counts(const NsView& nsites, int n_molecules) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nsites);
    std::vector<int> out(static_cast<size_t>(n_molecules));
    for (int m = 0; m < n_molecules; ++m) out[m] = h(m);
    return out;
}

}  // namespace klasp
