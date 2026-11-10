#pragma once
// Backend-agnostic device-resident batched relaxer. Geometry, DOF gradient and
// optimizer state stay on the device across rounds, and converged structures are
// harvested so the active set shrinks. The backend supplies staging views, the
// reference cell in its own units, unit conversions and one evaluation callback.
#include <functional>
#include <vector>

#include "klasp/device_optim.hpp"  // DeviceGeom, DOF policies, DevStep, kernels
#include "klasp/pet_bridge.hpp"    // CrystalGeometry, PetSymBatch
#include "klasp/pet_relax.hpp"     // PetRelaxOptions, PetRelaxResult, PetSymmetry

namespace klasp {

// Where an evaluation deposits its answer: a DOF gradient, not per-atom forces
// (force_fn plus the DOF policy's fold covers force-producing backends).
//
// `energy`/`maxg`/`maxs` describe the geometry staged this round, which may be
// an unaccepted trial; DevStep decides which of them reach the optimizer state.
struct DofEval {
    devopt::RView1D grad;    // [D] packed DOF gradient; structure b starts at dofoff(b)
    devopt::RView1D energy;  // [B] Hartree
    devopt::RView1D maxg;    // [B] max per-atom |force|, Ha/Bohr (the force_tol metric)
    devopt::RView1D maxs;    // [B] max |stress|, Ha/Bohr^3 (the stress_tol metric)
    devopt::IView1D act;     // [Kc] compact slot -> structure index b
    devopt::IView1D caoff;   // [Kc] compact slot -> its first atom in the compact set
    devopt::IView1D dofoff;  // [B] structure -> base of its DOF block in grad
};

// Per-atom forces/virial/energy in the backend's units, for the active compact set.
struct BackendForces {
    devopt::RView2D forces;  // [Mc,3]
    devopt::RView2D virial;  // [Kc,6] full-tensor Voigt
    devopt::RView1D energy;  // [Kc]
};

// Everything the device relaxer needs from an energy backend.
struct DeviceBackend {
    devopt::DeviceGeom geom;     // geometry staging (the optimizer writes pos/scell each round)
    devopt::RView2D h0;          // [B,9] reference cell rows, CellKernel frame, BACKEND length units
    devopt::IView1D spec_full;   // [Ntot] per-atom species (PET) or unused (pairwise backends)
    double gunit = 1.0;          // Bohr -> backend length unit (geom.pos = Bohr*gunit)
    double volunit = 1.0;        // cell determinant -> the volume unit the optimizer uses (Bohr^3)
    double f2hb = 1.0, e2h = 1.0;// backend force/energy -> Ha/Bohr, Hartree (1 if native a.u.)

    // Fill `out` for the active compact set from the geometry the optimizer has
    // already staged in `geom` this round. Set this OR force_fn, not both.
    std::function<void(int Kc, int Mc, const DofEval& out)> eval_fn;

    // For a backend whose output is per-atom forces and a virial: device_relax
    // folds them onto the DOFs with the DOF policy and f2hb/e2h. Ignored when
    // eval_fn is set.
    std::function<BackendForces(int Kc, int Mc)> force_fn;
};

// Relax `inits` to convergence on device under `backend`. symb != nullptr
// selects the symmetry-constrained parameterization. Returns one result per
// input, geometry in Bohr.
std::vector<PetRelaxResult> device_relax(const std::vector<CrystalGeometry>& inits,
                                         const PetRelaxOptions& opts,
                                         const PetSymBatch* symb,
                                         DeviceBackend& backend);

}  // namespace klasp
