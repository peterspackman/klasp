#pragma once
// The soft-sphere repulsion the rigid press minimises.

#include "klasp/types.hpp"
#include "klasp/pair_iteration.hpp"
#include "klasp/constants.hpp"

namespace klasp {

// Pure-repulsive (sigma/r)^12 with a smooth quintic taper from r = sigma to
// r = alpha*sigma.
//   sigma_ij = vdw_i + vdw_j (Bohr), the contact distance
//   eps      = single energy scale
//   alpha    = cutoff multiplier; energy and gradient vanish at r = alpha*sigma
// The quintic has continuous value, first and second derivatives at both ends.
// No attractive term, so nothing to damp as r -> 0.
inline constexpr RealType RP_EPS   = 1.0e-3;  // Hartree (~2.6 kJ/mol)
inline constexpr RealType RP_ALPHA = 1.5;     // cutoff = alpha * sigma_ij

// Per-pair upper bound for the image-shift early exit: alpha * 2 * max_vdw.
inline constexpr RealType RP_MAX_CUTOFF_ANG = 6.0;
inline constexpr RealType RP_MAX_CUTOFF = RP_MAX_CUTOFF_ANG * ANGSTROM_TO_BOHR;

// A batch_asymunit_pair_iteration fold: one term, no electrostatics, no tables.
template <typename Space>
struct WCA12PairFn {
    SView1D<RealType, Space> vdw;  // (n_indep * n_sites) -- Angstrom
    RealType eps;
    RealType alpha;
    RealType max_cutoff_sq;        // coarse upper bound for early exit

    struct Batch {
        RealType rcut_sq;
        // The press tapers per pair in eval(), so the global taper is the identity.
        KOKKOS_INLINE_FUNCTION
        void taper(RealType, RealType& ft, RealType& dft) const {
            ft = RealType(1);
            dft = RealType(0);
        }
    };

    KOKKOS_INLINE_FUNCTION
    Batch begin_batch(int /*b*/) const { return Batch{max_cutoff_sq}; }

    // The pair physics alone. body_i / body_j are asymmetric-unit site indices
    // (AtomState*::body), the key of the vdW table.
    KOKKOS_INLINE_FUNCTION
    bool eval(int body_i, int body_j, RealType r, RealType r2,
              RealType& e, RealType& fs) const {
        const RealType vdw_i = vdw(body_i) * ANGSTROM_TO_BOHR;
        const RealType vdw_j = vdw(body_j) * ANGSTROM_TO_BOHR;
        const RealType sigma = vdw_i + vdw_j;
        const RealType r_cut = alpha * sigma;
        if (r2 >= r_cut * r_cut) return false;
        if (r2 >= max_cutoff_sq) return false;

        const RealType inv_r = RealType(1) / r;
        const RealType so_r = sigma * inv_r;
        const RealType so_r2 = so_r * so_r;
        const RealType so_r4 = so_r2 * so_r2;
        const RealType so_r12 = so_r4 * so_r4 * so_r4;

        RealType taper_v, dtaper_dr;
        if (r <= sigma) {
            taper_v = RealType(1);
            dtaper_dr = RealType(0);
        } else {
            const RealType inv_span = RealType(1) / (sigma * (alpha - RealType(1)));
            const RealType t = (r - sigma) * inv_span;
            const RealType t2 = t * t;
            const RealType t3 = t2 * t;
            const RealType t4 = t2 * t2;
            const RealType t5 = t4 * t;
            taper_v = RealType(1) - (RealType(10) * t3 - RealType(15) * t4 + RealType(6) * t5);
            const RealType one_m_t = RealType(1) - t;
            dtaper_dr = -RealType(30) * t2 * one_m_t * one_m_t * inv_span;
        }

        e = eps * so_r12 * taper_v;
        const RealType dE_dr = eps * so_r12 *
            (-RealType(12) * taper_v * inv_r + dtaper_dr);
        // fs = +V'(r)/r: the scaffolding applies f_i = fs * (pos(j) - pos(i)).
        fs = dE_dr * inv_r;
        return true;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const Batch& /*bc*/, const PairGeom& g,
                    const AtomStateI& ai, const AtomStateJ& aj,
                    PairAcc& out) const {
        RealType e, fs;
        if (!eval(ai.body, aj.body, g.r, g.r2, e, fs)) return;
        out.F_i.add(fs * g.rx, fs * g.ry, fs * g.rz);
        if (g.is_self_shift)
            out.F_j.add(-fs * g.rx, -fs * g.ry, -fs * g.rz);
        out.shared.e += e * g.e_weight;
        if (g.compute_strain)
            accumulate_strain_virial(out.shared, g, g.e_weight * fs);
    }
};

}  // namespace klasp
