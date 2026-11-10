#pragma once
// Composable pair terms for the batch energy/gradient kernel. Each term has a
// Params struct and an inlined evaluate(PairGeom, AtomStateI, AtomStateJ, Params,
// PairAcc); the pair loop folds them per atom pair.

#include "klasp/types.hpp"
#include "klasp/energy_kernels.hpp"
#include "klasp/multipole.hpp"

#include <Kokkos_Core.hpp>

namespace klasp {

// Energy + 6-component strain reducer of the asymunit pair kernel.
struct EnergyStrainAcc {
    RealType e;
    RealType s[6];
    // Translation and rotation gradient of the single independent molecule;
    // only Z' = 1 takes this contracted path.
    RealType g[6];

    KOKKOS_INLINE_FUNCTION EnergyStrainAcc() : e(0) {
        for (int i = 0; i < 6; ++i) { s[i] = 0; g[i] = 0; }
    }
    KOKKOS_INLINE_FUNCTION void operator+=(const EnergyStrainAcc& o) {
        e += o.e;
        for (int i = 0; i < 6; ++i) { s[i] += o.s[i]; g[i] += o.g[i]; }
    }
    KOKKOS_INLINE_FUNCTION void operator+=(const volatile EnergyStrainAcc& o) volatile {
        e += o.e;
        for (int i = 0; i < 6; ++i) { s[i] += o.s[i]; g[i] += o.g[i]; }
    }
};

// Translation and rotation gradient for one independent molecule. Reduced,
// not accumulated atomically, so the result is independent of thread order.
struct DofGrad6 {
    RealType g[6];

    KOKKOS_INLINE_FUNCTION DofGrad6() {
        for (int i = 0; i < 6; ++i) g[i] = 0;
    }
    KOKKOS_INLINE_FUNCTION void operator+=(const DofGrad6& o) {
        for (int i = 0; i < 6; ++i) g[i] += o.g[i];
    }
    KOKKOS_INLINE_FUNCTION void operator+=(const volatile DofGrad6& o) volatile {
        for (int i = 0; i < 6; ++i) g[i] += o.g[i];
    }
};

// One asymmetric-unit atom's contribution to the trust-region seed: twelve
// force-constant components and the lower triangle (21) of the strain block.
// Reduced rather than atomically accumulated so the result is deterministic.
struct FcAcc {
    RealType k[12];
    RealType h[21];

    KOKKOS_INLINE_FUNCTION FcAcc() {
        for (int i = 0; i < 12; ++i) k[i] = 0;
        for (int i = 0; i < 21; ++i) h[i] = 0;
    }
    KOKKOS_INLINE_FUNCTION void operator+=(const FcAcc& o) {
        for (int i = 0; i < 12; ++i) k[i] += o.k[i];
        for (int i = 0; i < 21; ++i) h[i] += o.h[i];
    }
    KOKKOS_INLINE_FUNCTION void operator+=(const volatile FcAcc& o) volatile {
        for (int i = 0; i < 12; ++i) k[i] += o.k[i];
        for (int i = 0; i < 21; ++i) h[i] += o.h[i];
    }
};

// Reciprocal-sum energy and six rigid-body gradient components of one molecule.
struct RecipAcc {
    RealType e;
    RealType g[6];

    KOKKOS_INLINE_FUNCTION RecipAcc() : e(0) {
        for (int i = 0; i < 6; ++i) g[i] = 0;
    }
    KOKKOS_INLINE_FUNCTION void operator+=(const RecipAcc& o) {
        e += o.e;
        for (int i = 0; i < 6; ++i) g[i] += o.g[i];
    }
    KOKKOS_INLINE_FUNCTION void operator+=(const volatile RecipAcc& o) volatile {
        e += o.e;
        for (int i = 0; i < 6; ++i) g[i] += o.g[i];
    }
};

// Per-pair geometry, computed once per (i,j,shift) and read-only for the terms.
struct PairGeom {
    RealType rx, ry, rz;      // cartesian separation (j - i + shift)
    RealType r, r2;           // scalar distance and its square
    RealType inv_r;           // 1/r, shared by every term
    RealType ft, dft;         // cutoff-spline value + derivative
    RealType cdx, cdy, cdz;   // COM separation for the strain virial
    RealType e_weight;        // energy counting weight (1.0 or 2.0)
    bool is_self_shift;       // same indep mol at nonzero shift (needs force on j)
    bool compute_strain;      // n_cell_dof > 0 at call site
    // This site pair's moments (PairMoments: s then the three v_m) and, for a
    // self-shift pair, the partner's torque moments; null when unused.
    const RealType* mom = nullptr;
    const RealType* wmom = nullptr;
};

// Per-atom state as seen by a term. Optional pointers are null when the model
// lacks them (mp when MaxRank == 0, dip when max_multipole_rank < 1).
// `body` is the asymmetric-unit index (indep_mol * n_sites + site), the key for
// per-site tables; for atom j it differs from the expanded-cell index.
struct AtomStateI {
    RealType pos[3];
    RealType q;
    int type;
    const RealType* dip;   // dipole (3 components) or nullptr
    const RealType* mp;    // multipole cartesian coeffs (nhermsum(MaxRank)) or nullptr
    int body;
};
struct AtomStateJ {
    RealType pos[3];
    RealType q;
    int type;
    const RealType* dip;
    const RealType* mp;
    int body;
};

// Force on one asymmetric-unit atom, contracted to rigid-body gradients as it is
// added. `jt` is the molecule's translation Jacobian, `jr` this atom's rotation
// Jacobian, both 3x3 component-major; `g` is the six-slot reducer. Null `jt`: the
// translation contribution cancels. Null `g`: discard.
struct ForceAcc {
    RealType* base = nullptr;   // retained for callers that want the raw force
    const RealType* jt = nullptr;
    const RealType* jr = nullptr;
    RealType* g = nullptr;

    KOKKOS_INLINE_FUNCTION void add(RealType fx, RealType fy, RealType fz) const {
        if (g) {
            // Gradient is minus the force contracted with the Jacobian.
            if (jt)
                for (int m = 0; m < 3; ++m)
                    g[m] -= fx*jt[m*3+0] + fy*jt[m*3+1] + fz*jt[m*3+2];
            if (jr)
                for (int m = 0; m < 3; ++m)
                    g[3+m] -= fx*jr[m*3+0] + fy*jr[m*3+1] + fz*jr[m*3+2];
        }
    }
};

// Multipole adjoint dE/dM, contracted on the way past against this atom's
// 3 x 35 multipole Jacobian.
struct DmpAcc {
    RealType* base = nullptr;        // &b_dmp(b, atom, 0), or null
    const RealType* jm = nullptr;    // this atom's 3 x 35 multipole Jacobian
    RealType* g = nullptr;

    KOKKOS_INLINE_FUNCTION void add(int c, RealType v) const {
        if (g && jm)
            for (int m = 0; m < 3; ++m) g[3+m] += v * jm[m*35 + c];
    }
    KOKKOS_INLINE_FUNCTION explicit operator bool() const { return base || jm; }
};

// Shared outputs handed to every term in a pair. `shared` references the
// thread-local reduce variable.
struct PairAcc {
    EnergyStrainAcc& shared;
    ForceAcc F_i;   // Buckingham-style forces on atom i
    ForceAcc F_j;   // only valid if PairGeom::is_self_shift
    ForceAcc EF_i;  // electrostatic forces on atom i (Ewald/multipole)
    ForceAcc EF_j;
    DmpAcc dmp_i;     // dE/dmp for atom i -- scattered, contracted, or both
    DmpAcc dmp_j;
};

// Accumulate the 6-component strain virial for a pair with fw = e_weight * fs.
KOKKOS_INLINE_FUNCTION
void accumulate_strain_virial(EnergyStrainAcc& acc, const PairGeom& g, RealType fw) {
    acc.s[0] += fw * g.rx * g.cdx;
    acc.s[1] += fw * g.ry * g.cdy;
    acc.s[2] += fw * g.rz * g.cdz;
    acc.s[3] += RealType(0.5) * fw * (g.ry * g.cdz + g.rz * g.cdy);
    acc.s[4] += RealType(0.5) * fw * (g.rx * g.cdz + g.rz * g.cdx);
    acc.s[5] += RealType(0.5) * fw * (g.rx * g.cdy + g.ry * g.cdx);
}

// PairPotentialTerm: the short-range pair potential given by a PairModel
// (pair_model.hpp), with taper, forces and strain virial. Caller gates r < rcut.
template <typename Model>
struct PairPotentialTerm {
    struct Params {
        Model model;
    };

    KOKKOS_INLINE_FUNCTION
    void evaluate(const PairGeom& g,
                  const AtomStateI& ai, const AtomStateJ& aj,
                  const Params& p, PairAcc& out) const {
        RealType e_raw, de_dr;
        p.model.eval(g.r, g.r2, g.inv_r, ai.type, aj.type, e_raw, de_dr);
        RealType e_pair, de_t;
        apply_taper(e_raw, de_dr, g.ft, g.dft, e_pair, de_t);
        // The radial force scalar: V'(r)/r, so that f = fs * (r_j - r_i).
        const RealType fs = de_t * g.inv_r;
        out.F_i.add(fs * g.rx, fs * g.ry, fs * g.rz);
        if (g.is_self_shift)
            out.F_j.add(-fs * g.rx, -fs * g.ry, -fs * g.rz);
        out.shared.e += e_pair * g.e_weight;
        if (g.compute_strain)
            accumulate_strain_virial(out.shared, g, g.e_weight * fs);
    }
};

// EwaldRealTerm: real-space Ewald charge-charge plus, when ai.dip != nullptr,
// charge-dipole and dipole-dipole. Kept together to share erfc(κr) and exp(-κ²r²).
struct EwaldRealTerm {
    struct Params {
        RealType kappa;
        RealType two_kappa_sqrtpi;  // ewald_gauss_prefactor(kappa), per structure
        RealType rcut_sq;   // real-space cutoff squared (Ewald rcut)
        RealType taper_on;  // where this term starts ramping to zero (Bohr)
        RealType taper_off; // == Ewald rcut
        WolfDSF wolf;       // wolf.enabled == false ⇒ plain Ewald
    };

    // Tapered over its own window (taper_on..taper_off == Ewald rcut), not the
    // pair taper g.ft, which is typically 1 throughout the Ewald range. A hard
    // cutoff makes the energy discontinuous enough to stall the trust region.
    KOKKOS_INLINE_FUNCTION
    void evaluate(const PairGeom& g,
                  const AtomStateI& ai, const AtomStateJ& aj,
                  const Params& p, PairAcc& out) const {
        if (g.r2 >= p.rcut_sq) return;
        const RealType ft  = cutoff_spline(g.r, p.taper_on, p.taper_off);
        if (ft <= RealType(0)) return;
        const RealType dft = cutoff_spline_deriv(g.r, p.taper_on, p.taper_off);

        const RealType erfc_val = Kokkos::erfc(p.kappa * g.r);
        const RealType exp_kr2  = Kokkos::exp(-p.kappa * p.kappa * g.r2);

        RealType e_qq, fs_qq;
        pair_ewald_qq(g.r, g.inv_r, ai.q, aj.q, p.kappa, p.two_kappa_sqrtpi,
                      erfc_val, exp_kr2, e_qq, fs_qq, p.wolf);
        // e' = e*ft and fs' = fs*ft + e*dft/r, matching pair_buckingham.
        fs_qq = fs_qq * ft + e_qq * dft * g.inv_r;
        e_qq *= ft;
        out.shared.e += e_qq * g.e_weight;
        out.EF_i.add(fs_qq * g.rx, fs_qq * g.ry, fs_qq * g.rz);
        if (g.is_self_shift)
            out.EF_j.add(-fs_qq * g.rx, -fs_qq * g.ry, -fs_qq * g.rz);
        if (g.compute_strain)
            accumulate_strain_virial(out.shared, g, g.e_weight * fs_qq);

        // ai.dip / aj.dip are nullptr when max_multipole_rank < 1.
        if (ai.dip == nullptr) return;

        RealType e_dip;
        RealType ef_iv[3], ef_jv[3], dmp_iv[3], dmp_jv[3];
        RealType miR, mjR, mimj;  // not shared out; multipole-higher recomputes
        pair_ewald_dipole(g.inv_r, ai.q, aj.q, p.kappa, p.two_kappa_sqrtpi,
                          erfc_val, exp_kr2,
                          ai.dip, aj.dip,
                          g.rx, g.ry, g.rz,
                          e_dip, ef_iv, ef_jv, dmp_iv, dmp_jv,
                          miR, mjR, mimj, p.wolf);
        // Same rule, per component: F' = F*ft + e*dft*rhat, and the dE/dmultipole
        // (torque) contributions carry the plain factor ft.
        {
            const RealType radial = e_dip * dft * g.inv_r;
            ef_iv[0] = ef_iv[0]*ft + radial*g.rx;
            ef_iv[1] = ef_iv[1]*ft + radial*g.ry;
            ef_iv[2] = ef_iv[2]*ft + radial*g.rz;
            ef_jv[0] = ef_jv[0]*ft - radial*g.rx;
            ef_jv[1] = ef_jv[1]*ft - radial*g.ry;
            ef_jv[2] = ef_jv[2]*ft - radial*g.rz;
            for (int d = 0; d < 3; ++d) { dmp_iv[d] *= ft; dmp_jv[d] *= ft; }
            e_dip *= ft;
        }
        out.shared.e += e_dip * g.e_weight;
        // Through the accumulator so the DOF-gradient contraction happens.
        out.EF_i.add(ef_iv[0], ef_iv[1], ef_iv[2]);
        if (g.is_self_shift)
            out.EF_j.add(ef_jv[0], ef_jv[1], ef_jv[2]);
        // dE/dmp slots 1..3 are the dipole components (slot 0 is charge).
        for (int d = 0; d < 3; ++d) {
            out.dmp_i.add(1 + d, dmp_iv[d]);
            if (g.is_self_shift)
                out.dmp_j.add(1 + d, dmp_jv[d]);
        }
        if (g.compute_strain) {
            // Virial from the per-component dipole forces.
            const RealType w = g.e_weight;
            out.shared.s[0] += w * ef_iv[0] * g.cdx;
            out.shared.s[1] += w * ef_iv[1] * g.cdy;
            out.shared.s[2] += w * ef_iv[2] * g.cdz;
            out.shared.s[3] += w * RealType(0.5) * (ef_iv[1] * g.cdz + ef_iv[2] * g.cdy);
            out.shared.s[4] += w * RealType(0.5) * (ef_iv[0] * g.cdz + ef_iv[2] * g.cdx);
            out.shared.s[5] += w * RealType(0.5) * (ef_iv[0] * g.cdy + ef_iv[1] * g.cdx);
        }
    }
};

// MultipoleHigherTerm: rank >= 2 multipole interactions. Gated on g.ft > 0 and on
// MaxRank > 0 at the call site.
template <int MaxRank, typename Real = RealType>
struct MultipoleHigherTerm {
    // pair_multipole_higher always reads/writes 35-element arrays regardless of rank.
    static constexpr int NMP_FULL = nhermsum(4);
    static constexpr int NMP = nhermsum(MaxRank);

    struct Params {
        RealType t_on = 0, t_off = 0;   // own taper window, Bohr
        bool own_taper = false;         // false: the pair taper (g.ft) alone
    };

    // Pair-moment path: E = sum_k s_k T_k, with torques from moments that already
    // carry the rotation Jacobians. Same taper as the direct path.
    KOKKOS_INLINE_FUNCTION
    void evaluate_moments(const PairGeom& g, const Params& p, PairAcc& out) const {
        if constexpr (MaxRank < 2) {
            (void)g; (void)p; (void)out;
        } else {
            using PM = PairMoments<MaxRank>;
            RealType ft = g.ft, dft = g.dft;
            if (p.own_taper) {
                const RealType ft_m = cutoff_spline(g.r, p.t_on, p.t_off);
                if (ft_m <= RealType(0)) return;
                const RealType dft_m = cutoff_spline_deriv(g.r, p.t_on, p.t_off);
                dft = g.dft * ft_m + g.ft * dft_m;
                ft = g.ft * ft_m;
            }
            RealType T[nhermsum(MaxRank + 1)];
            TTensor<MaxRank + 1, RealType>::run(g.rx, g.ry, g.rz, g.inv_r * g.inv_r, g.inv_r, T);
            RealType f[3], tq[3];
            const RealType e = PM::contract(g.mom, g.mom + PM::NT, T, f, tq);
            const RealType radial = e * dft * g.inv_r;
            const RealType force_mp[3] = {f[0] * ft + radial * g.rx,
                                          f[1] * ft + radial * g.ry,
                                          f[2] * ft + radial * g.rz};
            out.shared.e += e * ft * g.e_weight;
            out.EF_i.add(force_mp[0], force_mp[1], force_mp[2]);
            if (g.is_self_shift)
                out.EF_j.add(-force_mp[0], -force_mp[1], -force_mp[2]);
            if (g.compute_strain) {
                const RealType w = g.e_weight;
                out.shared.s[0] += w * force_mp[0] * g.cdx;
                out.shared.s[1] += w * force_mp[1] * g.cdy;
                out.shared.s[2] += w * force_mp[2] * g.cdz;
                out.shared.s[3] += w * RealType(0.5) * (force_mp[1] * g.cdz + force_mp[2] * g.cdy);
                out.shared.s[4] += w * RealType(0.5) * (force_mp[0] * g.cdz + force_mp[2] * g.cdx);
                out.shared.s[5] += w * RealType(0.5) * (force_mp[0] * g.cdy + force_mp[1] * g.cdx);
            }
            // Torques land where DmpAcc::add would have put the contracted
            // adjoint: the rotation slots of the pair reducer.
            if (out.dmp_i.g != nullptr && out.dmp_i.jm != nullptr)
                for (int m = 0; m < 3; ++m) out.dmp_i.g[3 + m] += tq[m] * ft;
            if (g.is_self_shift && g.wmom != nullptr
                && out.dmp_j.g != nullptr && out.dmp_j.jm != nullptr) {
                RealType tqb[3];
                PM::contract_b(g.wmom, T, tqb);
                for (int m = 0; m < 3; ++m) out.dmp_j.g[3 + m] += tqb[m] * ft;
            }
        }
    }


    KOKKOS_INLINE_FUNCTION
    void evaluate(const PairGeom& g,
                  const AtomStateI& ai, const AtomStateJ& aj,
                  const Params& p, PairAcc& out) const {
        if (g.ft <= 0) return;
        if (ai.mp == nullptr || aj.mp == nullptr) return;
        if (g.mom != nullptr) {
            evaluate_moments(g, p, out);
            return;
        }
        // The taper this term sees: the pair's, times its own.
        //   (ft_g ft_m)' = ft_g' ft_m + ft_g ft_m'
        RealType ft = g.ft, dft = g.dft;
        if (p.own_taper) {
            const RealType ft_m = cutoff_spline(g.r, p.t_on, p.t_off);
            if (ft_m <= RealType(0)) return;
            const RealType dft_m = cutoff_spline_deriv(g.r, p.t_on, p.t_off);
            dft = g.dft * ft_m + g.ft * dft_m;
            ft = g.ft * ft_m;
        }

        // pair_multipole_higher reads all 35 slots, so a lower rank is staged into
        // a zero-padded buffer; at rank 4 the pointers pass straight through.
        const RealType* cart_a;
        const RealType* cart_b;
        RealType pad_a[NMP_FULL], pad_b[NMP_FULL];
        if constexpr (NMP == NMP_FULL) {
            cart_a = ai.mp;
            cart_b = aj.mp;
            (void)pad_a; (void)pad_b;
        } else {
            for (int c = 0; c < NMP; ++c)      { pad_a[c] = ai.mp[c]; pad_b[c] = aj.mp[c]; }
            for (int c = NMP; c < NMP_FULL; ++c) { pad_a[c] = 0;        pad_b[c] = 0; }
            cart_a = pad_a;
            cart_b = pad_b;
        }

        RealType e_higher;
        RealType force_mp[3];
        RealType dE_da[NMP_FULL], dE_db[NMP_FULL];
        // dE_db is needed only when the partner's adjoint has a sink (dmp_j).
        if (out.dmp_j)
            pair_multipole_higher<MaxRank, true, Real>(
                cart_a, cart_b, g.rx, g.ry, g.rz, g.inv_r, ft, dft,
                e_higher, force_mp, dE_da, dE_db);
        else
            pair_multipole_higher<MaxRank, false, Real>(
                cart_a, cart_b, g.rx, g.ry, g.rz, g.inv_r, ft, dft,
                e_higher, force_mp, dE_da, dE_db);

        out.shared.e += e_higher * g.e_weight;
        out.EF_i.add(force_mp[0], force_mp[1], force_mp[2]);
        if (g.is_self_shift)
            out.EF_j.add(-force_mp[0], -force_mp[1], -force_mp[2]);
        if (g.compute_strain) {
            const RealType w = g.e_weight;
            out.shared.s[0] += w * force_mp[0] * g.cdx;
            out.shared.s[1] += w * force_mp[1] * g.cdy;
            out.shared.s[2] += w * force_mp[2] * g.cdz;
            out.shared.s[3] += w * RealType(0.5) * (force_mp[1] * g.cdz + force_mp[2] * g.cdy);
            out.shared.s[4] += w * RealType(0.5) * (force_mp[0] * g.cdz + force_mp[2] * g.cdx);
            out.shared.s[5] += w * RealType(0.5) * (force_mp[0] * g.cdy + force_mp[1] * g.cdx);
        }
        for (int c = 0; c < NMP; ++c) {
            out.dmp_i.add(c, dE_da[c]);
            if (g.is_self_shift)
                out.dmp_j.add(c, dE_db[c]);
        }
    }
};

}  // namespace klasp

namespace Kokkos {
template <>
struct reduction_identity<klasp::RecipAcc> {
    KOKKOS_FORCEINLINE_FUNCTION static klasp::RecipAcc sum() { return klasp::RecipAcc(); }
};

template <>
struct reduction_identity<klasp::FcAcc> {
    KOKKOS_FORCEINLINE_FUNCTION static klasp::FcAcc sum() { return klasp::FcAcc(); }
};

template <>
struct reduction_identity<klasp::DofGrad6> {
    KOKKOS_FORCEINLINE_FUNCTION static klasp::DofGrad6 sum() {
        return klasp::DofGrad6();
    }
};

template <>
struct reduction_identity<klasp::EnergyStrainAcc> {
    KOKKOS_FORCEINLINE_FUNCTION static klasp::EnergyStrainAcc sum() {
        return klasp::EnergyStrainAcc();
    }
};
}  // namespace Kokkos
