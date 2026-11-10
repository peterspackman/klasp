#pragma once

// What a run's energy model is made of, for the code around the kernels
// (geometry sizing, which post-pair terms run). A trivially copyable POD
// captured by value in kernels.
//
// Composition (which terms the model has) sizes the geometry; the selector only
// narrows which terms are evaluated. real_space_cutoff() therefore ignores the
// selector, so per-term energies are summed over the same images as the total.

#include "klasp/ewald.hpp"
#include "klasp/types.hpp"

#include <Kokkos_Core.hpp>

#include <cstdint>

namespace klasp {

// Debugging selector for isolating components; narrows a ForceField. Its
// semantics are pinned by the Ewald, gradient, multipole and Hessian tests.
enum EwaldComponent : uint32_t {
    EWALD_PAIR_ERFC   = 1u << 0,   // pair erfc-damped qq + qd + dd
    EWALD_RECIP       = 1u << 1,   // reciprocal-space sum
    EWALD_SELF        = 1u << 2,   // self-energy correction
    EWALD_INTRA       = 1u << 3,   // intra-molecular erf correction
    EWALD_BUCKINGHAM  = 1u << 4,   // the short-range pair potential
    EWALD_MULTIPOLE_H = 1u << 5,   // pair_multipole_higher (tapered)
    EWALD_ALL         = 0xFFFFFFFFu
};

// How a model sums the electrostatics; decides whether there is a reciprocal
// box, reciprocal sum, self and intramolecular correction.
enum class Electrostatics : uint8_t {
    // No charges (rigid press: WCA-12 soft sphere plus pV).
    None = 0,
    // erfc-damped real space, reciprocal sum, self and intramolecular corrections.
    Ewald = 1,
    // Damped shifted force: erfc-damped real space to a cutoff, a self shift, no
    // reciprocal sum.
    Wolf = 2,
};

struct ForceField {
    // ---------------- composition: what this model is made of --------------

    // The short-range pair potential and its cutoff (Bohr).
    bool     pair_potential = true;
    RealType pair_cutoff = 0;

    Electrostatics electrostatics = Electrostatics::Ewald;

    // ewald_split_width(accuracy): real-space cutoff is split/kappa, gmax is
    // 2*kappa*split. Unused when electrostatics is None.
    RealType ewald_split = 0;

    // erfc(kappa*Rc)/(2*Rc), for the Wolf self term. Zero otherwise.
    RealType wolf_self_shift = 0;

    // ---------------- selector: which of those to evaluate -----------------
    uint32_t selector = EWALD_ALL;

    // ---------------- the questions the call sites ask ---------------------

    // Real-space enumeration radius for one structure's kappa; sizes the shift
    // box and the pair-fold rejection radius. Selector-independent.
    KOKKOS_INLINE_FUNCTION
    RealType real_space_cutoff(RealType kappa) const {
        RealType c = pair_cutoff;
        if (electrostatics != Electrostatics::None)
            c = Kokkos::fmax(ewald_split / kappa, c);
        return c;
    }

    KOKKOS_INLINE_FUNCTION
    bool has_reciprocal() const {
        return electrostatics == Electrostatics::Ewald && (selector & EWALD_RECIP);
    }

    // Ewald self term -kappa/sqrt(pi) sum q^2; Wolf adds its shift, which applies
    // regardless of EWALD_SELF.
    KOKKOS_INLINE_FUNCTION
    bool has_self() const {
        return electrostatics == Electrostatics::Wolf ||
               (electrostatics == Electrostatics::Ewald && (selector & EWALD_SELF));
    }

    // Intramolecular erf correction; exists only with a reciprocal sum.
    KOKKOS_INLINE_FUNCTION
    bool has_intra() const {
        return electrostatics == Electrostatics::Ewald && (selector & EWALD_INTRA);
    }

    KOKKOS_INLINE_FUNCTION
    bool is_wolf() const { return electrostatics == Electrostatics::Wolf; }

    KOKKOS_INLINE_FUNCTION
    bool has_pair_potential() const {
        return pair_potential && (selector & EWALD_BUCKINGHAM);
    }
    KOKKOS_INLINE_FUNCTION
    bool has_real_space_electrostatics() const {
        return electrostatics != Electrostatics::None && (selector & EWALD_PAIR_ERFC);
    }
    KOKKOS_INLINE_FUNCTION
    bool has_multipoles() const {
        return electrostatics != Electrostatics::None && (selector & EWALD_MULTIPOLE_H);
    }
};

// A short-range pair potential alone, cut at `cutoff` Bohr (the rigid press).
inline ForceField pair_only_force_field(RealType cutoff) {
    ForceField ff;
    ff.pair_potential = true;
    ff.pair_cutoff = cutoff;
    ff.electrostatics = Electrostatics::None;
    ff.ewald_split = 0;
    return ff;
}

}  // namespace klasp
