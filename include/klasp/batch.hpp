#pragma once

#include <atomic>
#include <stdexcept>

#include "klasp/types.hpp"
#include "klasp/ewald.hpp"   // DEFAULT_EWALD_ACCURACY, ewald_split_width
#include "klasp/force_field.hpp"   // ForceField, EwaldComponent
#include "klasp/molecule.hpp"
#include "klasp/pair_model.hpp"   // PairModelKind, RadialSpline
#include "klasp/strain_basis.hpp"
#include "klasp/symmetry_ops.hpp"
#include <string>

namespace klasp {

template <typename Space = ExecSpace>
struct BatchLBFGSStateT {
    SView3D<RealType, Space> s_hist;         // (batch, memory, n_dof)
    SView3D<RealType, Space> y_hist;         // (batch, memory, n_dof)
    SView2D<RealType, Space> rho;            // (batch, memory)
    SView2D<RealType, Space> alpha_scratch;  // (batch, memory)
    SView2D<RealType, Space> direction;      // (batch, n_dof)
    SView2D<RealType, Space> x;              // (batch, n_dof)
    SView2D<RealType, Space> g;              // (batch, n_dof)
    SView2D<RealType, Space> x_new;          // (batch, n_dof)
    SView2D<RealType, Space> g_new;          // (batch, n_dof)
    SView1D<int, Space>      num_stored;     // (batch)
    SView1D<int, Space>      cursor;         // (batch)

    int n_dof;
    int memory;
};

using BatchLBFGSState = BatchLBFGSStateT<ExecSpace>;
using HostBatchLBFGSState = BatchLBFGSStateT<HostExecSpace>;

template <typename Space = ExecSpace>
BatchLBFGSStateT<Space> allocate_batch_lbfgs_state(int n_batch, int n_dof, int memory = 5);

// Slot lifecycle values carried by BatchGeometryT::slot_mask; only SLOT_LIVE
// slots are evaluated.
enum SlotState : int { SLOT_EMPTY = 0, SLOT_LIVE = 1, SLOT_DONE = 2 };

// Per-kernel timings, accumulated over iterations.
struct KernelTimings {
    double expand_ms = 0;   // batch_expand_asymmetric_unit
    double pairs_ms = 0;    // zero + pair kernel + force expansion
    double recip_ms = 0;    // Ewald reciprocal + recip force expansion
    double post_ms = 0;     // self + intra + combine + chain rule + L-BFGS
};

// Energy-model parameters. Trivially copyable so kernels can capture it by value.
struct EnergyModelParams {
    // Where the short-range terms are switched off; 0 means a 2 Angstrom ramp below the cutoff.
    RealType taper_on = 0.0;
    RealType taper_off = 0.0;

    // Tang-Toennies damping on the dispersion term and its range.
    TangToenniesDamping buck_damping;

    // Cutoff for the rank>=2 multipole term (Bohr); 0 uses the full cutoff.
    RealType mp_cutoff = 0;

    // Evaluate the rank>=2 multipole term in single precision, accumulated into
    // double. Ignored for rank < 2.
    bool mp_single = false;

    // Evaluate the rank>=2 multipole term through per-site-pair moments
    // (PairMoments, multipole_contraction.hpp) reused across lattice images.
    bool mp_pair_moments = true;

    // Wolf damped-shifted-force electrostatics instead of Ewald.
    // wolf_self_shift = erfc(kappa*Rc)/(2*Rc).
    bool     wolf = false;
    RealType wolf_cutoff = 0;
    RealType wolf_self_shift = 0;

    // Debugging selector: narrows which terms are evaluated. Never changes sizing.
    uint32_t ewald_components = EWALD_ALL;

    // Auto uses the tabulated model when the input built tables, else the closed
    // form. Resolved on the host; see use_tabulated_pair_model.
    PairModelKind pair_model = PairModelKind::Auto;
};

// Host-only run controls; not device-copyable.
struct OptimizerHooks {
    // Polled on the host between refill checkpoints; when true the driver stops
    // after the next harvest.
    const std::atomic<bool>* abort = nullptr;
};

struct BatchOptimizeSettings {
    // model is device-copyable; hooks is not.
    EnergyModelParams model;
    OptimizerHooks hooks;

    // Contract the rigid-body gradient inside the pair kernel instead of
    // scattering per-atom forces (see tests/test_contraction_collapse.cpp).
    bool contract_pairs = true;

    // With model.mp_single: the scaled-gradient level below which a slot switches to
    // the double-precision multipole term. Must stay well above single precision's
    // noise floor (~3e-7); convergence is only certified in double.
    RealType mp_single_switch_tol = RealType(1e-5);

    int n_iterations = 20;
    RealType max_step = 0.1;         // max DOF displacement per L-BFGS step (rigid press)
    RealType rotation_scale = 0.2;   // scale factor for rotation DOFs
    RealType cell_scale = 0.1;       // scale factor for cell strain DOFs
    RealType max_strain_step = 0.05; // max strain change per component per iteration
    RealType pressure_gpa = 0.1;    // external pressure for rigid press (GPa)
    int lbfgs_memory = 5;            // L-BFGS history length (rigid press)

    bool energy_only = false;        // skip the optimizer step (for single-point energy)
    // Stop when the L-infinity norm of the scaled gradient falls below this for the
    // whole batch; 0 disables. Units Hartree per Bohr or rad, by DOF.
    RealType grad_tol = 0.0;

    // Cell volume (length units cubed) beyond which a structure is abandoned. Set
    // from SamplingPrior::max_volume. 0 disables.
    RealType max_cell_volume = 0.0;

    // Restrict strain DOFs to the diagonal Voigt components 0,1,2 regardless of lattice.
    bool fix_angles = false;

    // Record per-structure convergence diagnostics into BatchOptimizeResult.
    bool store_diagnostics = false;

    // Record every trust-region iteration into BatchOptimizeResult::trajectory.
    bool record_trajectory = false;

    // Trust region: warm-start the molecular Hessian block from the previous
    // optimize call's stored block. Strain DOFs are still probed.
    bool reuse_hessian = false;

    // Streaming stage 2 (batch_optimize_stream_tr): harvest converged slots and
    // refill them from a pending pool. Trust region only.
    bool stream = false;
    // Iterations between streaming harvest/refill checkpoints.
    int refill_interval = 2;
    // Per-slot iteration cap; 0 uses n_iterations.
    int max_iters_per_slot = 0;
    // Pending pool size as a multiple of batch_size.
    int stream_pool_mult = 8;
};

// Per-structure convergence diagnostics, all (batch); empty unless
// store_diagnostics was set (check valid()). grad_* are final max/RMS gradient
// (a.u.); iters_to_converge is the first iteration below grad_tol, or -1.
struct BatchDiagnostics {
    Kokkos::View<RealType*, Kokkos::HostSpace> grad_max;
    Kokkos::View<RealType*, Kokkos::HostSpace> grad_rms;
    Kokkos::View<int*, Kokkos::HostSpace> iters_to_converge;
    // Trust-region state at exit. A radius far below its 0.2 start means the step
    // collapsed rather than reaching a minimum. restarts counts radius-floor rebuilds.
    Kokkos::View<RealType*, Kokkos::HostSpace> final_delta;
    Kokkos::View<int*, Kokkos::HostSpace> restarts;
    bool valid() const { return grad_max.size() > 0; }
    bool has_tr() const { return final_delta.size() > 0; }
};

// Per-iteration trust-region record: frame 0 is the start, frame f the state after
// iteration f. A step was accepted exactly when energy == trial_energy.
struct TrustRegionTrajectory {
    using Host2 = Kokkos::View<RealType**, Kokkos::LayoutRight, Kokkos::HostSpace>;
    using Host3 = Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace>;
    using Host4 = Kokkos::View<RealType****, Kokkos::LayoutRight, Kokkos::HostSpace>;
    int n_frames = 0;       // iterations run + 1; the views may be longer
    Host2 energy;           // (frame, batch) Hartree
    Host2 trial_energy;     // (frame, batch) Hartree
    Host2 grad_max;         // (frame, batch) scaled, what grad_tol is measured on
    Host2 trust_radius;     // (frame, batch) Bohr
    Host4 translations;     // (frame, batch, n_indep, 3) fractional
    Host4 rotations;        // (frame, batch, n_indep, 3) angle-axis
    Host3 strain;           // (frame, batch, 6) Voigt; empty when the cell was fixed
    bool valid() const { return n_frames > 0; }
};

struct BatchOptimizeResult {
    Kokkos::View<RealType*, Kokkos::HostSpace> final_energies;       // (batch)
    Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace> final_translations;   // (batch, n_indep, 3)
    Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace> final_rotations;      // (batch, n_indep, 3)
    // Iterations actually run; less than settings.n_iterations on early exit.
    int n_iterations;
    int n_batch;
    double wall_time_s;
    KernelTimings timings;
    BatchDiagnostics diagnostics;   // empty unless settings.store_diagnostics
    TrustRegionTrajectory trajectory;   // empty unless settings.record_trajectory
};

// Per-structure cell and Ewald state; the unit that setup, strain update and
// StreamPool entries copy as a whole.
template <typename Space = ExecSpace>
struct CellStateT {
    SView2D<RealType, Space> cell_params;    // (n, 6) [a,b,c,alpha,beta,gamma]
    SView3D<RealType, Space> cell;           // (n, 3, 3) cell matrix, vectors in columns
    SView3D<RealType, Space> cell_inv;       // (n, 3, 3) inverse cell matrix
    SView1D<RealType, Space> kappa;          // (n) Ewald splitting parameter
    SView1D<RealType, Space> volume;         // (n) cell volume
    SView2D<int, Space>      max_hkl;        // (n, 3) max reciprocal lattice index per axis
    SView2D<int, Space>      max_shift;      // (n, 3) max integer image index per axis
};

// Allocate one cell-state block of n entries; `tag` prefixes the Kokkos labels.
template <typename Space = ExecSpace>
CellStateT<Space> allocate_cell_state(int n, const std::string& tag) {
    CellStateT<Space> cs;
    cs.cell_params = SView2D<RealType, Space>(tag + "_cp", n, 6);
    cs.cell        = SView3D<RealType, Space>(tag + "_cell", n, 3, 3);
    cs.cell_inv    = SView3D<RealType, Space>(tag + "_cellinv", n, 3, 3);
    cs.kappa       = SView1D<RealType, Space>(tag + "_kappa", n);
    cs.volume      = SView1D<RealType, Space>(tag + "_vol", n);
    cs.max_hkl     = SView2D<int, Space>(tag + "_hkl", n, 3);
    cs.max_shift   = SView2D<int, Space>(tag + "_shift", n, 3);
    return cs;
}

template <typename Space>
void copy_cell_state(const CellStateT<Space>& dst, int dst_off,
                     const CellStateT<Space>& src, int src_off, int count) {
    using Range = std::pair<int, int>;
    const Range d(dst_off, dst_off + count), s(src_off, src_off + count);
    Kokkos::deep_copy(Kokkos::subview(dst.cell_params, d, Kokkos::ALL),
                      Kokkos::subview(src.cell_params, s, Kokkos::ALL));
    Kokkos::deep_copy(Kokkos::subview(dst.cell, d, Kokkos::ALL, Kokkos::ALL),
                      Kokkos::subview(src.cell, s, Kokkos::ALL, Kokkos::ALL));
    Kokkos::deep_copy(Kokkos::subview(dst.cell_inv, d, Kokkos::ALL, Kokkos::ALL),
                      Kokkos::subview(src.cell_inv, s, Kokkos::ALL, Kokkos::ALL));
    Kokkos::deep_copy(Kokkos::subview(dst.kappa, d), Kokkos::subview(src.kappa, s));
    Kokkos::deep_copy(Kokkos::subview(dst.volume, d), Kokkos::subview(src.volume, s));
    Kokkos::deep_copy(Kokkos::subview(dst.max_hkl, d, Kokkos::ALL),
                      Kokkos::subview(src.max_hkl, s, Kokkos::ALL));
    Kokkos::deep_copy(Kokkos::subview(dst.max_shift, d, Kokkos::ALL),
                      Kokkos::subview(src.max_shift, s, Kokkos::ALL));
}

// Run-constant tables: potential, body-frame molecule template, space group.
template <typename Space = ExecSpace>
struct ModelTablesT : SymmetryOpsT<Space> {
    SymmetryOpsT<Space>& symmetry() { return *this; }
    const SymmetryOpsT<Space>& symmetry() const { return *this; }

    // The model these tables describe. Defaults to Buckingham plus Ewald from
    // buck_cutoff and ewald_accuracy.
    ForceField ff;

    SView2D<RealType, Space> buck_A;         // (n_types, n_types)
    SView2D<RealType, Space> buck_B;
    SView2D<RealType, Space> buck_C;
    SView2D<RealType, Space> buck_Rmin;      // Buckingham minimum distance (catastrophe cap)
    RealType buck_cutoff = 0;
    // Tabulated pair model: one C2 radial spline per ordered type pair, indexed
    // type_i * n_types + type_j. Zero extent means no tables. See pair_model.hpp.
    SView1D<RealType, Space>       pair_table_coeffs;
    SView1D<RadialSpline<>, Space> pair_tables;
    // Relative truncation target for both Ewald sums; read via ewald_split_width.
    RealType ewald_accuracy = DEFAULT_EWALD_ACCURACY;
    // Type index of padded sites, or -1 if none.
    int ghost_type = -1;
    int n_types = 0;

    SView2D<RealType, Space> body_positions;   // (n_indep * n_sites, 3)
    // Bounding radius (Bohr) of each independent molecule about its rigid-body origin,
    // over real sites. Used to cull whole (molecule, shift) pairs.
    SView1D<RealType, Space> mol_radius;       // (n_indep)
    SView1D<RealType, Space> body_charges;     // (n_indep * n_sites)
    SView1D<int, Space>      body_types;       // (n_indep * n_sites)
    SView2D<RealType, Space> body_multipoles;  // (n_indep * n_sites, 35)
    // Radii in Angstrom; filled by the caller, not allocate_batch_workspace.
    SView1D<RealType, Space> cov_radii;        // (n_indep * n_sites)
    SView1D<RealType, Space> vdw_radii;        // (n_indep * n_sites)
    int max_multipole_rank = 0;

    // Symmetry tables are inherited from SymmetryOpsT.
    int n_sites_per_mol = 0;
};

// The ForceField described by this run's tables and model parameters.
template <typename Space>
inline ForceField make_force_field(const ModelTablesT<Space>& tables,
                                   const EnergyModelParams& model) {
    ForceField ff;
    ff.pair_potential   = true;
    ff.pair_cutoff      = tables.buck_cutoff;
    ff.electrostatics   = model.wolf ? Electrostatics::Wolf : Electrostatics::Ewald;
    ff.ewald_split      = ewald_split_width(tables.ewald_accuracy);
    ff.wolf_self_shift  = model.wolf_self_shift;
    ff.selector         = model.ewald_components;
    return ff;
}

// Resolve the pair model on the host. Explicit Tabulated without tables is an
// error, since every pair would evaluate to zero.
template <typename Space>
inline bool use_tabulated_pair_model(const EnergyModelParams& model,
                                     const ModelTablesT<Space>& tables) {
    const bool have = tables.pair_tables.extent(0) > 0;
    switch (model.pair_model) {
        case PairModelKind::Buckingham: return false;
        case PairModelKind::Tabulated:
            if (!have)
                throw std::runtime_error(
                    "the tabulated pair model was asked for, but this input built "
                    "no tables -- add a `potentials.tabulated` block");
            return true;
        case PairModelKind::Auto:
        default:
            return have;
    }
}

// The structures: cells, expanded atoms and energies.
template <typename Space = ExecSpace>
struct BatchGeometryT : CellStateT<Space> {
    CellStateT<Space>& cells() { return *this; }
    const CellStateT<Space>& cells() const { return *this; }

    int n_batch = 0;
    int n_atoms = 0;    // = n_symops * n_sites_per_mol, the padded stride

    // Slots to evaluate; zero extent means all of [0, n_batch). Empty slots must be
    // masked out: their placeholder cell enumerates a huge number of images.
    SView1D<int, Space> slot_mask;
    // Per-slot multipole precision when mp_single is on: 1 single, 0 double. Zero
    // extent means every slot uses mp_single as given.
    SView1D<int, Space> mp_precision;

    SView3D<RealType, Space> positions;      // (batch, n_atoms, 3)
    SView2D<RealType, Space> charges;        // (batch, n_atoms)
    SView2D<int, Space>      types;          // (batch, n_atoms)
    SView3D<RealType, Space> multipoles;     // (batch, n_atoms, 35)
    SView3D<RealType, Space> dipoles;        // (batch, n_atoms, 3)
    SView2D<int, Space>      mol_id;         // (batch, n_atoms)

    SView3D<RealType, Space> com_positions;  // (batch, n_symops, 3) Cartesian COM per molecule
    // COMs in the reference cell (same fractional coordinates through cell_ref): the
    // lever arm for the strain virial dE/dF = sum_mol f_mol (x) com_ref. Using
    // current COMs loses the antisymmetric part of the virial.
    SView3D<RealType, Space> com_positions_ref;  // (batch, n_symops, 3)

    SView1D<RealType, Space> energies;       // (batch)
};

// The optimiser's DOFs, their reference frames and gradients.
template <typename Space = ExecSpace>
struct RigidDofStateT {
    SView3D<RealType, Space> translations;   // (batch, n_indep, 3)
    SView3D<RealType, Space> rot_vectors;    // (batch, n_indep, 3)

    // Cell strain; allocated by the caller.
    SView3D<RealType, Space> cell_ref;       // (batch, 3, 3) reference cell before strain
    SView2D<RealType, Space> strain;         // (batch, 6) Voigt strain components
    int n_cell_dof = 0;                      // active strain DOFs (0 = no cell opt)
    // (batch, 6, 6) row-major: rows 0..n_cell_dof-1 are the allowed Voigt directions
    // (klasp/strain_basis.hpp). Per structure because the rhombohedral subspace
    // depends on the 3-fold axis direction.
    SView3D<RealType, Space> cell_dof_basis;
    // Lattice the basis was built for; -1 = unset.
    int cell_dof_lattice = -1;

    SView3D<RealType, Space> d_gt;           // (batch, n_indep, 3)
    SView3D<RealType, Space> d_gr;           // (batch, n_indep, 3)
    SView2D<RealType, Space> d_g_strain;     // (batch, 6) strain gradient
    // Gradients from contraction inside the pair kernel.
    SView3D<RealType, Space> d_gt_c;         // (batch, n_indep, 3)
    SView3D<RealType, Space> d_gr_c;         // (batch, n_indep, 3)

    // Rigid-body Jacobians, indexed by asymmetric-unit atom (n_indep*n_sites).
    SView3D<RealType, Space> jac_t;   // (batch, n_indep, 9)  S_k . cell(:,d), d-major
    SView3D<RealType, Space> jac_r;   // (batch, n_asym, 9)   S_k . (dR^m . body), m-major
    SView3D<RealType, Space> jac_m;   // (batch, n_asym, 3*35) d(rotated multipoles)/dtheta_m
};

// Trust-region curvature, allocated on first use.
template <typename Space = ExecSpace>
struct TrustRegionScratchT {
    // Molecular Hessian block (n_indep*6 square, scaled coords) kept for warm-starting stage 2.
    SView3D<RealType, Space> tr_hessian_mol;

    // Per-atom force-constant tensor (batch, n_atoms, 6 Voigt) for the Gauss-Newton Hessian.
    SView3D<RealType, Space> fc;

    // Analytic strain second derivative (batch, 6, 6), Voigt, symmetric:
    //     d2E/ds_v ds_w = sum_pairs [ a (R.d_v)(R.d_w) + fs (d_v.d_w) ],
    //     d_v = E_v u,  fs = phi'/r,  a = (phi'' - phi'/r)/r^2
    // over real-space Buckingham and erfc charge-charge pairs only.
    SView3D<RealType, Space> strain_hess;

    // Slots whose Hessian seeding is recomputed; zero extent means all.
    SView1D<int, Space> hess_mask;
};

// Owning aggregate of the four workspace parts. Members are named fields, not
// by-value accessors, so that assigning a View after allocation sticks.
template <typename Space = ExecSpace>
struct BatchWorkspaceT {
    ModelTablesT<Space>        tables;
    BatchGeometryT<Space>      geom;
    RigidDofStateT<Space>      dofs;
    TrustRegionScratchT<Space> tr;
};

using BatchWorkspace = BatchWorkspaceT<ExecSpace>;
using HostBatchWorkspace = BatchWorkspaceT<HostExecSpace>;
using CellState = CellStateT<ExecSpace>;
using ModelTables = ModelTablesT<ExecSpace>;
using BatchGeometry = BatchGeometryT<ExecSpace>;
using RigidDofState = RigidDofStateT<ExecSpace>;
using TrustRegionScratch = TrustRegionScratchT<ExecSpace>;
using HostModelTables = ModelTablesT<HostExecSpace>;
using HostBatchGeometry = BatchGeometryT<HostExecSpace>;

// Allocate a BatchWorkspace from ws/mol. The caller must fill cov_radii,
// vdw_radii, cell_ref, strain and d_g_strain where needed; trust-region scratch
// is allocated on first use.
template <typename Space = ExecSpace>
BatchWorkspaceT<Space> allocate_batch_workspace(int n_batch,
                                        const Workspace& ws,
                                        const MoleculeData& mol);

// Everything that is not a pair sum (reciprocal, self, intra, external pressure),
// accumulated into energy and DOF gradients as `ff` says the model has them.
// Takes no optimizer step.
template <int MaxRank, typename Space = ExecSpace>
void batch_post_pairs(const ForceField& ff,
                      const ModelTablesT<Space>& tables,
                      BatchGeometryT<Space>& geom,
                      RigidDofStateT<Space>& dofs,
                      const BatchOptimizeSettings& settings,
                      KernelTimings* timings = nullptr);

#define KLASP_POST_PAIRS_DECL(rank)                                   \
    extern template void batch_post_pairs<rank, ExecSpace>(           \
        const ForceField&, const ModelTables&, BatchGeometry&,        \
        RigidDofState&, const BatchOptimizeSettings&, KernelTimings*);
KLASP_POST_PAIRS_DECL(0)
KLASP_POST_PAIRS_DECL(1)
KLASP_POST_PAIRS_DECL(2)
KLASP_POST_PAIRS_DECL(3)
KLASP_POST_PAIRS_DECL(4)
#undef KLASP_POST_PAIRS_DECL


// Fill dofs.cell_dof_basis for each structure from the lattice type, set and
// return dofs.n_cell_dof. Call once per stage, after cell setup.
template <typename Space = ExecSpace>
int batch_set_strain_dofs(const BatchGeometryT<Space>& geom,
                          RigidDofStateT<Space>& dofs, int lattice_type,
                          bool fix_angles = false);

extern template int batch_set_strain_dofs<ExecSpace>(const BatchGeometry&,
                                                     RigidDofState&, int, bool);

// Update cells, volume, max_shift and the hkl box from the Voigt strain, on
// device. No-op when n_cell_dof == 0.
template <typename Space = ExecSpace>
void batch_update_cell_from_strain(const ForceField& ff,
                                   BatchGeometryT<Space>& geom,
                                   RigidDofStateT<Space>& dofs);

extern template void batch_update_cell_from_strain<ExecSpace>(
    const ForceField&, BatchGeometry&, RigidDofState&);

// kappa is held fixed while the cell moves, since the strain gradient is taken
// at constant kappa. It is reset between optimiser steps when the volume has
// moved by this factor either way, which also bounds the hkl box growth.
inline constexpr double EWALD_REBRACKET_VOLUME_RATIO = 8.0;

// Reset kappa for live slots whose volume has drifted past
// EWALD_REBRACKET_VOLUME_RATIO and recompute max_hkl and max_shift. Sets
// `rebracketed` to 1 for those slots (they need re-evaluating), 0 elsewhere.
// An empty `live` means all slots.
template <typename Space = ExecSpace>
void batch_rebracket_ewald(const ForceField& ff, BatchGeometryT<Space>& geom,
                           const SView1D<int, Space>& live,
                           const SView1D<int, Space>& rebracketed);

extern template void batch_rebracket_ewald<ExecSpace>(
    const ForceField&, BatchGeometry&, const SView1D<int, ExecSpace>&,
    const SView1D<int, ExecSpace>&);

// Reciprocal strain gradient by Dual6 forward-mode AD.
template <typename Space = ExecSpace>
void batch_recip_strain_ad(const ModelTablesT<Space>& tables,
                           const BatchGeometryT<Space>& geom,
                           RigidDofStateT<Space>& dofs, bool has_dipoles);

// Ewald Q-Q diagnostic: prints energy and virial decomposition.
template <typename Space = ExecSpace>
void batch_ewald_nl_diagnostic(const ModelTablesT<Space>& tables,
                               const BatchGeometryT<Space>& geom,
                               const RigidDofStateT<Space>& dofs);

// L-BFGS step: pack gradients, compute direction, update DOFs. Used by the rigid press.
template <typename Space = ExecSpace>
void batch_lbfgs_step(const ModelTablesT<Space>& tables,
                       const BatchGeometryT<Space>& geom,
                       RigidDofStateT<Space>& dofs,
                       BatchLBFGSStateT<Space>& lbfgs,
                       const BatchOptimizeSettings& settings,
                       int iteration);

// Compute per-structure cell, cell_inv, kappa, volume, max_hkl, max_shift from
// cell_params. Ewald truncation comes from tables.ewald_accuracy.
template <typename Space = ExecSpace>
void batch_setup_cells(const ModelTablesT<Space>& tables,
                       BatchGeometryT<Space>& geom);

template <int MaxRank, typename Space = ExecSpace>
void batch_expand_asymmetric_unit(const ModelTablesT<Space>& tables,
                                  BatchGeometryT<Space>& geom,
                                  const RigidDofStateT<Space>& dofs);

#define KLASP_EXPAND_DECL(rank)                                        \
    extern template void batch_expand_asymmetric_unit<rank, ExecSpace>( \
        const ModelTables&, BatchGeometry&, const RigidDofState&);
KLASP_EXPAND_DECL(0)
KLASP_EXPAND_DECL(1)
KLASP_EXPAND_DECL(2)
KLASP_EXPAND_DECL(3)
KLASP_EXPAND_DECL(4)
#undef KLASP_EXPAND_DECL

// Single-point energy (Hartree) via the asymunit kernel with n_batch = 1;
// `components` restricts to the selected EwaldComponent terms.
RealType batch_evaluate_energy(const MoleculeData& mol, const Workspace& ws,
                                int max_rank,
                                RealType taper_on = 0, RealType taper_off = 0,
                                TangToenniesDamping buck_damping = TangToenniesDamping::off(),
                                uint32_t components = EWALD_ALL,
                                RealType mp_cutoff = 0,    // Bohr; 0 = the pair cutoff
                                bool mp_single = false);

}  // namespace klasp
