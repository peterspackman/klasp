#pragma once

#include "klasp/types.hpp"
#include "klasp/taper.hpp"
#include "klasp/multipole.hpp"
#include <Kokkos_Core.hpp>

namespace klasp {

// Wolf DSF (Damped Shifted Force) constants, precomputed at the cutoff Rc for
// a given kappa. Trivially copyable for capture in Kokkos lambdas.

struct WolfDSF {
    RealType Rc = 0;        // cutoff radius
    RealType phi_rc = 0;    // erfc(κRc)/Rc
    RealType dphi_rc = 0;   // -φ'(Rc) = erfc(κRc)/Rc² + 2κ/√π·exp(-κ²Rc²)/Rc
    RealType b_rc = 0;      // B(Rc) for dipole shift
    RealType c_rc = 0;      // C(Rc) for dipole-dipole shift
    RealType self_shift = 0; // erfc(κRc)/(2Rc)
    bool enabled = false;
};

inline WolfDSF compute_wolf_constants(RealType kappa, RealType Rc) {
    WolfDSF w;
    w.enabled = true;
    w.Rc = Rc;
    RealType kRc = kappa * Rc;
    RealType erfc_kRc = std::erfc(kRc);
    RealType exp_kRc2 = std::exp(-kRc * kRc);
    RealType inv_Rc = 1.0 / Rc;
    RealType inv_Rc2 = inv_Rc * inv_Rc;
    RealType inv_Rc3 = inv_Rc2 * inv_Rc;
    RealType inv_Rc4 = inv_Rc2 * inv_Rc2;
    RealType inv_Rc5 = inv_Rc4 * inv_Rc;
    RealType two_kappa_sqrtpi = 2.0 * kappa / std::sqrt(M_PI);

    w.phi_rc = erfc_kRc * inv_Rc;
    w.dphi_rc = erfc_kRc * inv_Rc2 + two_kappa_sqrtpi * exp_kRc2 * inv_Rc;
    w.self_shift = erfc_kRc * 0.5 * inv_Rc;
    w.b_rc = -two_kappa_sqrtpi * exp_kRc2 * inv_Rc2 - erfc_kRc * inv_Rc3;
    w.c_rc = 4.0*kappa*kappa*kappa/std::sqrt(M_PI)*exp_kRc2*inv_Rc2
           + 6.0*kappa/std::sqrt(M_PI)*exp_kRc2*inv_Rc4
           + 3.0*erfc_kRc*inv_Rc5;
    return w;
}

// Electrostatic configuration (Ewald/Wolf parameters), captured by value.

struct ElecParams {
    RealType kappa = 0;
    bool has_dipoles = false;
    WolfDSF wolf;  // wolf.enabled selects Wolf DSF vs standard Ewald
};


KOKKOS_INLINE_FUNCTION
void fast_sincos(RealType x, RealType& s, RealType& c) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    #ifdef KLASP_SINGLE_PRECISION
        ::sincosf(x, &s, &c);
    #else
        ::sincos(x, &s, &c);
    #endif
#else
    s = Kokkos::sin(x);
    c = Kokkos::cos(x);
#endif
}

// Scalar pair functions: pure math, no views, no atomic_add.

// Inner turning point of the undamped Buckingham form, where
// A B e^{-Br} = 6C/r^7. Inside it the energy dives to -infinity (finite in
// floating point); pair_buckingham uses this to cap that branch.
inline RealType buckingham_r_min(RealType A, RealType B, RealType C) {
    if (C <= 0 || A <= 0 || B <= 0) return 0;
    RealType r = std::pow(RealType(6.0) * C / (A * B), RealType(1.0/8.0));
    for (int iter = 0; iter < 20; ++iter) {
        RealType exp_br = std::exp(-B * r);
        RealType r7 = r*r*r*r*r*r*r;
        RealType f = A*B*exp_br - RealType(6.0)*C/r7;
        RealType df = -A*B*B*exp_br + RealType(42.0)*C/(r7*r);
        RealType dr = f / df;
        r -= dr;
        if (r < 0.1) r = 0.1;
        if (std::abs(dr) < 1e-12 * r) break;
    }
    return r;
}

// Tang-Toennies damping function f_n(x) = 1 - exp(-x) * sum_{k=0}^{n} x^k/k!
KOKKOS_INLINE_FUNCTION
void tang_toennies_6(RealType x, RealType& f6, RealType& df6_dx,
                     RealType* d2f6_dx2 = nullptr) {
    RealType exp_mx = Kokkos::exp(-x);
    // f6 = 1 - exp(-x) * S6 where S6 = sum_{k=0}^{6} x^k/k!
    RealType x2 = x*x, x3 = x2*x, x4 = x3*x, x5 = x4*x, x6 = x5*x;
    RealType S6 = 1.0 + x + x2/2.0 + x3/6.0 + x4/24.0 + x5/120.0 + x6/720.0;
    f6 = 1.0 - exp_mx * S6;
    // df6/dx = exp(-x) * x^6/720 (the last term of the series)
    df6_dx = exp_mx * x6 / 720.0;
    // d2f6/dx2 = exp(-x) x^5 (6 - x) / 720
    if (d2f6_dx2) *d2f6_dx2 = exp_mx * x5 * (RealType(6.0) - x) / RealType(720.0);
}

// Product rule for the cutoff taper ft applied to an untapered potential V:
//     (V ft)'  = V' ft + V ft'
//     (V ft)'' = V'' ft + 2 V' ft' + V ft''
// The ft'' term matters for the strain Hessian, since most of the pair sum sits
// in the taper shell. Pass d2ft = 0 (or use the two-derivative overload) for
// gradients only.
KOKKOS_INLINE_FUNCTION
void apply_taper(RealType e, RealType de_dr,
                 RealType ft, RealType dft,
                 RealType& e_t, RealType& de_t) {
    e_t  = e * ft;
    de_t = de_dr * ft + e * dft;
}

KOKKOS_INLINE_FUNCTION
void apply_taper(RealType e, RealType de_dr, RealType d2e_dr2,
                 RealType ft, RealType dft, RealType d2ft,
                 RealType& e_t, RealType& de_t, RealType& d2e_t) {
    apply_taper(e, de_dr, ft, dft, e_t, de_t);
    d2e_t = d2e_dr2 * ft + RealType(2.0) * de_dr * dft + e * d2ft;
}

// Tang-Toennies damping of C/r^6 as f6(range * B r), B being the pair's
// repulsive exponent. range 1.5 (default) removes the short-range dive while
// leaving W99 wells within ~0.5% of undamped; range 1 shallows the wells and
// large ranges (~2.5) let the dive back. tests/test_buckingham.cpp pins both.
struct TangToenniesDamping {
    bool enabled = true;
    RealType range = RealType(1.5);

    static TangToenniesDamping off() {
        TangToenniesDamping d;
        d.enabled = false;
        return d;
    }
};

// Buckingham energy and its first two radial derivatives, untapered,
// optionally Tang-Toennies damped. inv_r = 1/r is supplied by the caller.
// d2e_dr2 may be null for gradient-only callers; the trust-region seed needs it.
KOKKOS_INLINE_FUNCTION
void buckingham_untapered(RealType r, RealType inv_r,
                          RealType A, RealType B, RealType C,
                          const TangToenniesDamping& damping, RealType r_min,
                          RealType& e, RealType& de_dr,
                          RealType* d2e_dr2 = nullptr) {
    // Undamped only: inside r_min continue with the harmonic matching V, V' = 0
    // and the repulsive curvature at r_min, so the branch stays finite and
    // repulsive. Damping moves the turning point, so it is excluded.
    if (!damping.enabled && r_min > RealType(0) && r < r_min) {
        const RealType exp_min = Kokkos::exp(-B * r_min);
        const RealType rm6 = r_min*r_min*r_min*r_min*r_min*r_min;
        const RealType v_min = A*exp_min - C/rm6;         // V(r_min); V'(r_min) = 0
        const RealType kappa = A*B*B*exp_min;             // repulsive curvature there
        const RealType dr = r_min - r;
        e = v_min + RealType(0.5)*kappa*dr*dr;
        de_dr = -kappa * dr;
        if (d2e_dr2) *d2e_dr2 = kappa;
        return;
    }

    RealType exp_br = Kokkos::exp(-B*r);
    const RealType inv_r2 = inv_r*inv_r;
    const RealType C_r6 = C * (inv_r2*inv_r2*inv_r2);   // C / r^6

    // x = Bd r, so d/dr = Bd d/dx.
    const RealType Bd = damping.range * B;
    RealType f6 = RealType(1), df6 = RealType(0), d2f6 = RealType(0);
    if (damping.enabled) {
        tang_toennies_6(Bd * r, f6, df6, &d2f6);
    }

    e = A*exp_br - C_r6 * f6;
    de_dr = -B*A*exp_br + RealType(6.0)*C_r6*inv_r * f6 - C_r6 * df6 * Bd;

    if (d2e_dr2) {
        // V = A e^-Br - C f6(Bd r)/r^6, so with x = Bd r:
        //   V'' = A B^2 e^-Br - 42 C f6/r^8 + 12 C Bd f6'/r^7 - C Bd^2 f6''/r^6
        // Damping's second-order effect on the taper product rule is omitted; this
        // only seeds BFGS.
        *d2e_dr2 = A*B*B*exp_br
                 - RealType(42.0)*C_r6*f6*inv_r2
                 + RealType(12.0)*C_r6*Bd*df6*inv_r
                 - C_r6*Bd*Bd*d2f6;
    }
}

// The Ewald real-space Gaussian prefactor 2 kappa / sqrt(pi), fixed per structure.
KOKKOS_INLINE_FUNCTION
RealType ewald_gauss_prefactor(RealType kappa) {
    return RealType(2.0)*kappa/Kokkos::sqrt(RealType(M_PI));
}

// Ewald real charge-charge: qi*qj*erfc(κr)/r. Returns energy + radial force scalar.
// With wolf.enabled, applies DSF shift so both potential and force vanish at Rc.
// inv_r is per pair, two_kappa_sqrtpi per structure, both from the caller.
KOKKOS_INLINE_FUNCTION
void pair_ewald_qq(RealType r, RealType inv_r,
                   RealType qi, RealType qj, RealType kappa,
                   RealType two_kappa_sqrtpi,
                   RealType erfc_val, RealType exp_kr2,
                   RealType& e_pair, RealType& fs,
                   const WolfDSF& wolf = WolfDSF{},
                   RealType* d2v_dr2 = nullptr) {
    const RealType inv_r2 = inv_r*inv_r;
    RealType phi = erfc_val * inv_r;
    RealType dphi = -erfc_val*inv_r2 - two_kappa_sqrtpi*exp_kr2*inv_r;
    if (wolf.enabled) {
        phi += -wolf.phi_rc + wolf.dphi_rc * (r - wolf.Rc);
        dphi += wolf.dphi_rc;
    }
    e_pair = qi * qj * phi;
    fs = qi * qj * dphi * inv_r;

    // Radial second derivative V''(r) for the analytic Hessian guess. The Wolf
    // shift is linear in r, so it doesn't contribute to the curvature.
    //   phi''(r) = 2 erfc/r^3 + (2k/sqrt(pi)) e^{-k^2 r^2} (2/r^2 + 2 k^2)
    if (d2v_dr2) {
        RealType phi2 = RealType(2.0)*erfc_val*inv_r2*inv_r
                      + two_kappa_sqrtpi*exp_kr2*(RealType(2.0)*inv_r2 + RealType(2.0)*kappa*kappa);
        *d2v_dr2 = qi * qj * phi2;
    }
}

// Charge-dipole and dipole-dipole terms for one pair, given the T-tensor
// coefficients of the caller's radial family (erfc for Ewald real space, erf
// for the intramolecular correction).
//
// C1/C2 multiply directly; C2d/C3d stand in for dC1/dR and dC2/dR via
// dC_n/dR[d] = -C_{n+1} R[d]. They differ only when the caller has shifted the
// coefficients (Wolf), where the shift is a constant and so drops out of the
// derivative. Force is returned for i; j takes the negative.
KOKKOS_INLINE_FUNCTION
void dipole_pair_terms(RealType C1, RealType C2, RealType C2d, RealType C3d,
                       RealType qi, RealType qj,
                       const RealType dip_i[3], const RealType dip_j[3],
                       const RealType Rv[3],
                       RealType& e_pair, RealType f_i[3],
                       RealType dmp_i[3], RealType dmp_j[3],
                       RealType& miR_out, RealType& mjR_out, RealType& mimj_out) {
    const RealType miR = dip_i[0]*Rv[0] + dip_i[1]*Rv[1] + dip_i[2]*Rv[2];
    const RealType mjR = dip_j[0]*Rv[0] + dip_j[1]*Rv[1] + dip_j[2]*Rv[2];
    const RealType mimj = dip_i[0]*dip_j[0] + dip_i[1]*dip_j[1] + dip_i[2]*dip_j[2];
    miR_out = miR; mjR_out = mjR; mimj_out = mimj;

    const RealType P = qi*mjR - qj*miR;
    e_pair = -C1*P + C1*mimj - C2*miR*mjR;

    for (int d = 0; d < 3; ++d) {
        const RealType f_qd = C2d*Rv[d]*P - C1*(qi*dip_j[d] - qj*dip_i[d]);
        const RealType f_dd = -C2d*Rv[d]*mimj + C3d*Rv[d]*miR*mjR
                            - C2*(dip_i[d]*mjR + dip_j[d]*miR);
        f_i[d] = f_qd + f_dd;
        dmp_i[d] =  C1*qj*Rv[d] + C1*dip_j[d] - C2*mjR*Rv[d];
        dmp_j[d] = -C1*qi*Rv[d] + C1*dip_i[d] - C2*miR*Rv[d];
    }
}

// Ewald real dipole terms (charge-dipole + dipole-dipole).
// Returns erfc-damped qd+dd energy, forces on i/j, and dE/dmu for i/j.
// Also outputs dot products (miR, mjR, mimj) for reuse by pair_multipole_higher.
//
// Uses Ewald T-tensor coefficients C1,C2,C3 via recursion:
//   C_{n+1} = (2n+1)*C_n/r^2 + (2kappa^2)^n * (2kappa/sqrt(pi)) * exp / r^2
// with the identity dC_n/dR[d] = -C_{n+1} * R[d].
// Bare-Coulomb limits: C1->1/r^3, C2->3/r^5, C3->15/r^7.
//
// inv_r and two_kappa_sqrtpi as in pair_ewald_qq.
KOKKOS_INLINE_FUNCTION
void pair_ewald_dipole(RealType inv_r,
                       RealType qi, RealType qj, RealType kappa,
                       RealType two_kappa_sqrtpi,
                       RealType erfc_val, RealType exp_kr2,
                       const RealType dip_i[3], const RealType dip_j[3],
                       RealType rx, RealType ry, RealType rz,
                       RealType& e_pair,
                       RealType ef_i[3], RealType ef_j[3],
                       RealType dmp_i[3], RealType dmp_j[3],
                       RealType& miR_out, RealType& mjR_out, RealType& mimj_out,
                       const WolfDSF& wolf = WolfDSF{}) {
    RealType kappa2 = kappa*kappa;
    RealType inv_r2 = inv_r*inv_r;

    // T-tensor coefficients via recursion: C0 = erfc/r (= phi)
    RealType C0 = erfc_val * inv_r;
    RealType C1 = C0*inv_r2 + two_kappa_sqrtpi*exp_kr2*inv_r2;
    RealType C2 = RealType(3)*C1*inv_r2 + (RealType(2)*kappa2)*two_kappa_sqrtpi*exp_kr2*inv_r2;
    RealType C3 = RealType(5)*C2*inv_r2 + (RealType(4)*kappa2*kappa2)*two_kappa_sqrtpi*exp_kr2*inv_r2;

    // Wolf shifts the coefficients so the dipole energy vanishes at Rc:
    // b_rc = -C1(Rc), c_rc = +C2(Rc). The shift is constant, so the derivative
    // identity dC_n/dR = -C_{n+1} R uses the unshifted coefficients; C2u keeps the
    // unshifted C2 for that. Shifted potential, not shifted force: the force has a
    // kink at Rc.
    const RealType C2u = C2;
    if (wolf.enabled) { C1 += wolf.b_rc; C2 -= wolf.c_rc; }

    const RealType Rv[3] = {rx, ry, rz};
    dipole_pair_terms(C1, C2, C2u, C3, qi, qj, dip_i, dip_j, Rv,
                      e_pair, ef_i, dmp_i, dmp_j, miR_out, mjR_out, mimj_out);
    for (int d = 0; d < 3; ++d) ef_j[d] = -ef_i[d];
}

// The tapered rank >= 2 multipole interaction of one pair: the contraction
// over 2 <= la + lb <= MaxRank (MultipoleHigherContraction), i.e. what the
// Ewald sums do not cover.
//
// Real is the scalar the T tensor and contraction run in; inputs and outputs
// are RealType (see EnergyModelParams::mp_single). NeedB = false leaves dE_db
// untouched.
template <int MaxRank, bool NeedB = true, typename Real = RealType>
KOKKOS_INLINE_FUNCTION
void pair_multipole_higher(const RealType cart_a[35], const RealType cart_b[35],
                           RealType rx, RealType ry, RealType rz, RealType inv_r,
                           RealType ft, RealType dft,
                           RealType& e_higher,
                           RealType force_mp[3],
                           RealType dE_da[35], RealType dE_db[35]) {
    constexpr int NMP = nhermsum(MaxRank);
    Real T[nhermsum(MaxRank + 1)];
    TTensor<MaxRank + 1, Real>::run(Real(rx), Real(ry), Real(rz),
                                    Real(inv_r * inv_r), Real(inv_r), T);
    Real f[3], dEa[nhermsum(4)], dEb[nhermsum(4)];
    const Real e = MultipoleHigherContraction<MaxRank, NeedB, Real>::run(
        cart_a, cart_b, T, f, dEa, dEb);

    // Taper: (e ft)' = e' ft + e ft', with the radial part of ft' along R/r.
    const RealType e_un = RealType(e);
    e_higher = e_un * ft;
    const RealType radial = e_un * dft * inv_r;
    force_mp[0] = RealType(f[0]) * ft + radial * rx;
    force_mp[1] = RealType(f[1]) * ft + radial * ry;
    force_mp[2] = RealType(f[2]) * ft + radial * rz;
    for (int c = 0; c < NMP; ++c) {
        dE_da[c] = RealType(dEa[c]) * ft;
        if constexpr (NeedB) dE_db[c] = RealType(dEb[c]) * ft;
    }
}

}  // namespace klasp
