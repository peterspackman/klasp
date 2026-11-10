#include "klasp/ewald.hpp"
#include "klasp/pair_skip.hpp"
#include "klasp/cell.hpp"
#include <cmath>
#include <vector>

namespace klasp {

void compute_ewald_params(const RealType cell[3][3], int n_particles,
                          RealType accuracy, Workspace& ws) {
    const RealType vol = cell_volume(cell);
    const RealType s = std::sqrt(-std::log(accuracy));

    // Optimal eta from a weighted-cost model.
    const RealType eta = std::cbrt(
        (static_cast<RealType>(n_particles) * M_PI * M_PI * M_PI) / (vol * vol));
    const RealType kappa = std::sqrt(eta);
    const RealType rcut = s / kappa;
    const RealType gmax = 2.0 * kappa * s;

    ws.kappa = kappa;
    ws.rcut  = rcut;

    // Columns of `cell` are lattice vectors, so the rows of inv(cell) are the
    // reciprocal vectors over 2*pi. Zero on a singular cell.
    RealType inv_cell[3][3] = {};
    mat3_inv(cell, inv_cell);
    RealType recip[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            recip[i][j] = 2.0 * M_PI * inv_cell[i][j];

    // Index box covering the gmax sphere, from lattice vector lengths (see
    // reciprocal_index_bounds).
    int bounds[3];
    reciprocal_index_bounds(cell, gmax, bounds);
    const int kmax_h = bounds[0], kmax_k = bounds[1], kmax_l = bounds[2];

    const RealType gmax_sq = gmax * gmax;
    const RealType four_kappa_sq = 4.0 * kappa * kappa;
    const RealType two_pi_over_vol = 2.0 * M_PI / vol;

    std::vector<RealType> kvecs_host;
    std::vector<RealType> pref_host;

    for (int h = -kmax_h; h <= kmax_h; ++h) {
        for (int k = -kmax_k; k <= kmax_k; ++k) {
            for (int l = -kmax_l; l <= kmax_l; ++l) {
                if (h == 0 && k == 0 && l == 0) continue;
                RealType kv[3];
                for (int d = 0; d < 3; ++d)
                    kv[d] = h * recip[0][d] + k * recip[1][d] + l * recip[2][d];
                RealType k2 = kv[0]*kv[0] + kv[1]*kv[1] + kv[2]*kv[2];
                if (k2 < gmax_sq) {
                    kvecs_host.push_back(kv[0]);
                    kvecs_host.push_back(kv[1]);
                    kvecs_host.push_back(kv[2]);
                    pref_host.push_back(two_pi_over_vol * std::exp(-k2 / four_kappa_sq) / k2);
                }
            }
        }
    }

    int n_kvecs = static_cast<int>(pref_host.size());
    ws.n_kvecs = n_kvecs;

    auto h_kvecs = Kokkos::create_mirror_view(ws.kvecs);
    auto h_pref  = Kokkos::create_mirror_view(ws.kprefactors);

    for (int i = 0; i < n_kvecs; ++i) {
        h_kvecs(i, 0) = kvecs_host[3 * i + 0];
        h_kvecs(i, 1) = kvecs_host[3 * i + 1];
        h_kvecs(i, 2) = kvecs_host[3 * i + 2];
        h_pref(i)     = pref_host[i];
    }

    Kokkos::deep_copy(ws.kvecs, h_kvecs);
    Kokkos::deep_copy(ws.kprefactors, h_pref);
}

}  // namespace klasp
