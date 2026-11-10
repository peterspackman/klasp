#pragma once
// Host flexible and symmetry-constrained crystal relaxer (batched L-BFGS), driven
// by src/pet_relax.cpp. The backend is a template parameter `Eval` exposing
//   PetResult evaluate(const std::vector<PetStructure>&, bool forces) const
// The L-BFGS step and step cap are devopt's, shared with the device relaxer.
// Parameterizations:
//   * CellState  - free atoms: r = (I+E) p, h = (I+E) h0.
//   * SymState   - asymmetric-unit Cartesian coords (Bohr) + symmetrized strain;
//     the full cell is rebuilt from the space-group operators each step.
//   * RigidState - translation + rotation vector per molecule, cell fixed.
// Atomic units throughout (Bohr, Hartree, Hartree/Bohr).
#include "klasp/pet_relax.hpp"
#include "klasp/pet_bridge.hpp"  // PetRelaxOptions/Result, PetSymmetry, PetStructure, PetResult
#include "klasp/mat3.hpp"
#include "klasp/crystal_dof.hpp"     // crystdof::strain_correct_virial
#include "klasp/device_optim.hpp"    // the shared L-BFGS step, the step cap and the DOF folds

#include <array>
#include <cmath>
#include <vector>

namespace klasp {
namespace flexopt {

using Vec3 = std::array<double, 3>;

// 3x3 matrix, row-major m[r][c]. Cells store lattice vectors as columns, and a
// linear map acts as r' = M r, h' = M h.
struct Mat3 { double m[3][3]; };

inline Mat3 identity3() { return {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}; }

inline Vec3 matvec(const Mat3& A, const Vec3& v) {
    Vec3 out{};
    mat3_vec(&A.m[0][0], v.data(), out.data());
    return out;
}

inline Mat3 matmul(const Mat3& A, const Mat3& B) {
    Mat3 C{};
    // Use the [9] overload: it matches Mat3's row-major memory and differs from the
    // [3][3] one in the last bit.
    mat3_mul(&A.m[0][0], &B.m[0][0], &C.m[0][0]);
    return C;
}

inline double det3(const Mat3& A) { return mat3_det(A.m); }

// Symmetric strain matrix from Voigt [xx,yy,zz,xy,xz,yz].
inline Mat3 voigt_to_mat(const double e[6]) {
    return {{{e[0], e[3], e[4]}, {e[3], e[1], e[5]}, {e[4], e[5], e[2]}}};
}

inline Mat3 cell_to_mat(const double cell[3][3]) {
    Mat3 M{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) M.m[i][j] = cell[i][j];
    return M;
}
inline void mat_to_cell(const Mat3& M, double cell[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) cell[i][j] = M.m[i][j];
}

// Per-structure L-BFGS state with a batched Armijo line search: propose moves x to
// a capped trial; accept commits it (and proposes the next) or halves the step for
// another batched evaluation. The step is devopt::lbfgs_propose / lbfgs_accept;
// derived classes supply build() and grad_at().
struct LbfgsBase {
    std::vector<double> x, d, x_base, grad, grad_prev, last_step, q;
    std::vector<double> S, Y, rho;   // [kHist*ndof] / [kHist*ndof] / [kHist]
    int hist_head = 0, hist_count = 0, stall_count = 0;
    double grad_prev_max = 0;
    int have_prev = 0, active = 1, converged = 0;
    int iters = 0, max_iter = 300;
    double e_base = 0, gd = 0, energy = 0, max_g = 0, max_s = 0;
    double force_tol = 2e-4, stress_tol = 1e-6;
    // Trust-region cap inputs to devopt::step_cap.
    double max_step = 0.30, max_strain_step = 0.02;

    virtual ~LbfgsBase() {}
    virtual int ndof() const = 0;
    virtual int natoms() const = 0;
    bool periodic = true;

    virtual PetStructure build() = 0;                                  // geometry at current x
    virtual void grad_at(const double* F, const double* W,
                         std::vector<double>& g, double& mg, double& ms) = 0;
    // DOF split, and the coordinate step metric (max |component| for free atoms, max
    // |dq| for an asymmetric unit or rigid body).
    virtual int n_coord_dof() const = 0;
    virtual int n_dof_atoms() const = 0;
    virtual double cell_scale() const = 0;
    virtual double coord_metric(const double* dir) const = 0;

    // Size scratch for ndof(); x is kept if already the right length.
    void ensure_storage() {
        const int n = ndof();
        if (static_cast<int>(x.size()) != n) x.assign(n, 0.0);
        d.assign(n, 0.0); x_base.assign(n, 0.0); grad.assign(n, 0.0);
        grad_prev.assign(n, 0.0); last_step.assign(n, 0.0); q.assign(n, 0.0);
        S.assign(static_cast<std::size_t>(devopt::kHist) * n, 0.0);
        Y.assign(static_cast<std::size_t>(devopt::kHist) * n, 0.0);
        rho.assign(devopt::kHist, 0.0);
    }

    devopt::LbfgsState slot() {
        devopt::LbfgsState s;
        s.x = x.data(); s.grad = grad.data(); s.gprev = grad_prev.data(); s.d = d.data();
        s.xbase = x_base.data(); s.last = last_step.data(); s.q = q.data();
        s.S = S.data(); s.Y = Y.data(); s.rho = rho.data();
        // No model-Hessian preconditioner on the host path.
        s.ndof = ndof(); s.nfree = n_coord_dof(); s.nstrain = ndof() - n_coord_dof();
        s.ndofatoms = n_dof_atoms(); s.cellfac = cell_scale();
        s.energy = &energy; s.ebase = &e_base; s.gd = &gd;
        s.maxg = &max_g; s.maxs = &max_s; s.gpm = &grad_prev_max;
        s.active = &active; s.conv = &converged; s.iters = &iters;
        s.hh = &hist_head; s.hc = &hist_count; s.haveprev = &have_prev;
        s.stallc = &stall_count;
        return s;
    }

    devopt::LbfgsTuning tuning() const {
        return devopt::LbfgsTuning{force_tol, stress_tol, max_step, max_strain_step, max_iter};
    }

    // Coordinate half of the step cap, dispatched to the derived state.
    struct HostCap {
        const LbfgsBase* self;
        double operator()(const double* dir, int, int) const { return self->coord_metric(dir); }
    };

    // One line-search round on the freshly evaluated geometry; `initial` proposes from
    // the start point, otherwise the trial is judged. energy / max_g / max_s always
    // describe the last accepted point.
    void round(double e, const double* F, const double* W, bool initial) {
        grad.assign(ndof(), 0.0);
        double rg = 0, rs = 0;
        grad_at(F, W, grad, rg, rs);
        const HostCap cap{this};
        if (initial) devopt::lbfgs_propose(slot(), tuning(), cap, e, rg, rs);
        else         devopt::lbfgs_accept(slot(), tuning(), cap, e, rg, rs);
    }
};

// Run a homogeneous batch to convergence: one batched evaluation per round, each
// state accepting or backtracking independently.
template <class State, class Eval>
void run_lbfgs_batch(const Eval& eval, std::vector<State>& st,
                     std::vector<PetRelaxResult>& res, const PetRelaxOptions& opts) {
    const int B = (int)st.size();
    for (int b = 0; b < B; ++b) st[b].ensure_storage();
    auto eval_round = [&](bool initial) {
        std::vector<int> act;
        std::vector<PetStructure> batch;
        for (int b = 0; b < B; ++b)
            if (st[b].active) { act.push_back(b); batch.push_back(st[b].build()); }
        if (act.empty()) return false;
        PetResult r = eval.evaluate(batch, /*forces=*/true);
        int off = 0;
        for (std::size_t k = 0; k < act.size(); ++k) {
            const int b = act[k], N = st[b].natoms();
            const double* F = r.forces.data() + (std::size_t)off * 3;
            const double* W = (!r.virial.empty()) ? r.virial.data() + k * 6 : nullptr;
            st[b].round(r.energy[k], F, W, initial);
            if (initial) res[b].energy_trace.push_back(r.energy[k]);
            off += N;
        }
        return true;
    };
    eval_round(/*initial=*/true);
    const int max_rounds = opts.max_iter_full * 10 + 50;  // generous (includes backtracks)
    for (int round = 0; round < max_rounds; ++round)
        if (!eval_round(false)) break;
    // States still active hit the round budget with x at an unaccepted trial; restore
    // the accepted point before harvesting.
    for (auto& s : st)
        if (s.active) { s.x = s.x_base; s.energy = s.e_base; s.active = 0; }
    for (int b = 0; b < B; ++b) {
        res[b].structure = st[b].build();
        res[b].energy = st[b].energy;
        res[b].max_force = st[b].max_g;
        res[b].max_stress = st[b].max_s;
        res[b].full_iterations = st[b].iters;
        res[b].converged = st[b].converged;
    }
}

// Free atomic positions + strain: r = (I+E)p, h = (I+E)h0.
struct CellState : LbfgsBase {
    int N = 0;
    bool cell = true;
    double cell_factor = 1.0;
    Mat3 h0{}, def{};
    std::vector<int> Z;

    int ndof() const override { return 3 * N + (cell ? 6 : 0); }
    int natoms() const override { return N; }
    int n_coord_dof() const override { return 3 * N; }
    int n_dof_atoms() const override { return N; }
    double cell_scale() const override { return cell_factor; }
    double coord_metric(const double* dir) const override { return devopt::free_cap(dir, 3 * N); }

    PetStructure build() override {
        double eps[6] = {0, 0, 0, 0, 0, 0};
        if (cell)
            for (int k = 0; k < 6; ++k) eps[k] = x[3 * N + k] / cell_factor;
        Mat3 E = voigt_to_mat(eps);
        def = identity3();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) def.m[i][j] += E.m[i][j];
        PetStructure s;
        s.atomic_numbers = Z;
        s.periodic = periodic;
        s.positions.resize(N);
        for (int i = 0; i < N; ++i) {
            Vec3 pi{x[3*i], x[3*i+1], x[3*i+2]};
            s.positions[i] = matvec(def, pi);
        }
        mat_to_cell(matmul(def, h0), s.cell);
        return s;
    }

    // devopt::FreeDof::fold with f2hb = e2h = 1 (backend units are already atomic).
    void grad_at(const double* F, const double* W, std::vector<double>& g, double& mg, double& ms) override {
        mg = devopt::free_fold_atoms(&def.m[0][0], F, 1.0, N, g.data());
        ms = 0;
        if (cell && W) {
            double Wc[6];
            crystdof::strain_correct_virial(W, &def.m[0][0], Wc);
            double g6[6];
            ms = crystdof::strain_grad(Wc, cell_factor, std::fabs(det3(matmul(def, h0))), g6);
            for (int k = 0; k < 6; ++k) g[3*N+k] = g6[k];
        }
    }

};

// Symmetry-constrained state. DOFs are the asymmetric-unit Cartesian coords in the
// reference cell (q, Bohr) + symmetrized strain. build() regenerates the cell as
// atom(op) = (I+E)(M0_op q_a + tc_op), M0_op = h0 R_op h0^-1, tc_op = h0 t_op.
struct SymState : LbfgsBase {
    int n_indep = 0, n_op = 0;
    // Ragged layout; see devopt::SymLayout.
    std::vector<int> sites_of_indep, asym_off, full_off, op_of_atom;
    bool cell = true;
    double cell_factor = 1.0;
    Mat3 h0{}, def{}, h{};
    std::vector<Mat3> R, M0;        // R: fractional rotations (strain); M0: cartesian (ref cell)
    std::vector<Vec3> tc;          // cartesian translation in the reference cell
    std::vector<int> indep_of_op, Z;

    int n_asym() const { return asym_off.empty() ? 0 : asym_off.back(); }
    int nfree() const { return 3 * n_asym(); }
    int ndof() const override { return nfree() + (cell ? 6 : 0); }
    int natoms() const override { return full_off.empty() ? 0 : full_off.back(); }
    int n_coord_dof() const override { return nfree(); }
    int n_dof_atoms() const override { return n_asym(); }
    double cell_scale() const override { return cell_factor; }
    double coord_metric(const double* dir) const override { return devopt::sym_cap(dir, n_asym()); }

    PetStructure build() override {
        double eps[6] = {0, 0, 0, 0, 0, 0};
        if (cell)
            for (int k = 0; k < 6; ++k) eps[k] = x[nfree() + k] / cell_factor;
        Mat3 E = voigt_to_mat(eps);
        def = identity3();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) def.m[i][j] += E.m[i][j];
        h = matmul(def, h0);
        PetStructure s;
        s.atomic_numbers = Z;
        s.periodic = periodic;
        s.positions.resize(static_cast<std::size_t>(natoms()));
        mat_to_cell(h, s.cell);
        for (int op = 0; op < n_op; ++op) {
            const int k = indep_of_op[op];
            const int ns = sites_of_indep[k];
            for (int site = 0; site < ns; ++site) {
                const int a = asym_off[k] + site;
                Vec3 q{x[3*a], x[3*a+1], x[3*a+2]};
                Vec3 base = matvec(M0[op], q);              // M0_op q
                for (int c = 0; c < 3; ++c) base[c] += tc[op][c];
                s.positions[full_off[op] + site] = matvec(def, base);  // (I+E)(M0 q + tc)
            }
        }
        return s;
    }

    // Shared with the device via devopt::sym_fold. std::vector<Mat3>/<Vec3> are
    // contiguous 9/3 doubles, the flat layout sym_fold reads.
    void grad_at(const double* F, const double* W, std::vector<double>& g, double& mg, double& ms) override {
        const devopt::SymLayout L{n_indep, sites_of_indep.data(), asym_off.data(),
                                  full_off.data(), op_of_atom.data()};
        if (cell && W) {
            devopt::sym_fold(&def.m[0][0], F, 1.0, W, 1.0, n_op, L, &M0[0].m[0][0],
                             &tc[0][0], &R[0].m[0][0], indep_of_op.data(), &h0.m[0][0],
                             nfree(), cell_factor, g.data(), mg, ms);
            return;
        }
        // Fixed cell or no virial: give sym_fold a strain block of its own to write and discard.
        const double wzero[6] = {0, 0, 0, 0, 0, 0};
        std::vector<double> gfull(static_cast<std::size_t>(nfree()) + 6, 0.0);
        devopt::sym_fold(&def.m[0][0], F, 1.0, wzero, 1.0, n_op, L, &M0[0].m[0][0],
                         &tc[0][0], &R[0].m[0][0], indep_of_op.data(), &h0.m[0][0],
                         nfree(), cell_factor, gfull.data(), mg, ms);
        for (int k = 0; k < nfree(); ++k) g[k] = gfull[k];
        ms = 0;
    }

};

// Rigid-body state: translation + rotation vector per molecule, cell fixed, on the
// exact Rodrigues gradient (devopt::rigid_fold).
struct RigidState : LbfgsBase {
    std::vector<std::vector<int>> groups;  // atom indices of each rigid body
    std::vector<double> p0;                // [3*N] reference coordinates, Bohr
    std::vector<Vec3> centroid;            // [G] reference centroid of each group
    std::vector<int> Z;
    Mat3 h0{};
    int N = 0;

    int ngroups() const { return static_cast<int>(groups.size()); }
    int ndof() const override { return 6 * ngroups(); }
    int natoms() const override { return N; }
    int n_coord_dof() const override { return ndof(); }   // no strain block
    int n_dof_atoms() const override { return ngroups(); }
    double cell_scale() const override { return 1.0; }
    // One metric for both halves: max_step bounds translation (Bohr) and rotation (rad).
    double coord_metric(const double* dir) const override {
        return devopt::free_cap(dir, ndof());
    }

    // Per-group force scratch in rigid_fold's layout.
    mutable std::vector<double> gp0, gF, gr;

    PetStructure build() override {
        PetStructure s;
        s.atomic_numbers = Z;
        s.periodic = periodic;
        mat_to_cell(h0, s.cell);
        s.positions.resize(static_cast<std::size_t>(N));
        for (int g = 0; g < ngroups(); ++g) {
            const int n = static_cast<int>(groups[g].size());
            gp0.assign(3 * n, 0.0); gr.assign(3 * n, 0.0);
            for (int i = 0; i < n; ++i)
                for (int c = 0; c < 3; ++c) gp0[3*i+c] = p0[3 * groups[g][i] + c];
            devopt::rigid_expand(&x[6*g], gp0.data(), centroid[g].data(), n, gr.data());
            for (int i = 0; i < n; ++i)
                for (int c = 0; c < 3; ++c) s.positions[groups[g][i]][c] = gr[3*i+c];
        }
        return s;
    }

    // mg is the largest atomic force component, so force_tol keeps its per-atom meaning.
    void grad_at(const double* F, const double* W, std::vector<double>& g,
                 double& mg, double& ms) override {
        (void)W;
        mg = 0;
        for (int k = 0; k < 3 * N; ++k) mg = std::max(mg, std::fabs(F[k]));
        ms = 0;
        for (int gi = 0; gi < ngroups(); ++gi) {
            const int n = static_cast<int>(groups[gi].size());
            gp0.assign(3 * n, 0.0); gF.assign(3 * n, 0.0);
            for (int i = 0; i < n; ++i)
                for (int c = 0; c < 3; ++c) {
                    gp0[3*i+c] = p0[3 * groups[gi][i] + c];
                    gF[3*i+c] = F[3 * groups[gi][i] + c];
                }
            devopt::rigid_fold(&x[6*gi], gp0.data(), centroid[gi].data(), gF.data(), n,
                               &g[6*gi]);
        }
    }
};

// Build a RigidState from a geometry and its rigid grouping; ungrouped atoms stay fixed.
inline RigidState make_rigid_state(const PetStructure& init,
                                   const std::vector<std::vector<int>>& groups,
                                   const PetRelaxOptions& opts) {
    RigidState c;
    c.N = static_cast<int>(init.atomic_numbers.size());
    c.Z = init.atomic_numbers;
    c.periodic = init.periodic;
    c.h0 = cell_to_mat(init.cell);
    c.p0.assign(3 * c.N, 0.0);
    for (int i = 0; i < c.N; ++i)
        for (int k = 0; k < 3; ++k) c.p0[3*i+k] = init.positions[i][k];
    for (const auto& gsel : groups)
        if (!gsel.empty()) c.groups.push_back(gsel);
    c.centroid.resize(c.groups.size());
    for (std::size_t g = 0; g < c.groups.size(); ++g) {
        Vec3 com{0, 0, 0};
        for (int idx : c.groups[g])
            for (int k = 0; k < 3; ++k) com[k] += init.positions[idx][k];
        for (int k = 0; k < 3; ++k) com[k] /= static_cast<double>(c.groups[g].size());
        c.centroid[g] = com;
    }
    c.x.assign(c.ndof(), 0.0);
    c.max_step = opts.max_step; c.max_strain_step = opts.max_strain_step;
    c.force_tol = opts.force_tol; c.stress_tol = opts.stress_tol;
    c.max_iter = opts.max_iter_rigid;
    return c;
}

// True if a fractional operator (R, t) is the identity (R = I, t = 0 mod 1).
inline bool op_is_identity(const Mat3& R, const Vec3& t) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (std::fabs(R.m[i][j] - (i == j ? 1.0 : 0.0)) > 1e-6) return false;
    for (int c = 0; c < 3; ++c) {
        double f = t[c] - std::floor(t[c] + 0.5);  // distance to nearest integer
        if (std::fabs(f) > 1e-6) return false;
    }
    return true;
}

// Initialize a SymState from a full-cell structure and its symmetry, taking each
// independent molecule from its identity-operator image. Returns an invalid state
// (n_op == 0) if one has no identity operator; the caller then relaxes unconstrained.
inline SymState make_sym_state(const PetStructure& init, const PetSymmetry& sym,
                               const PetRelaxOptions& opts) {
    SymState c;
    c.n_indep = sym.n_indep;
    c.sites_of_indep = sym.sites_of_indep;
    c.n_op = sym.n_op;
    c.indep_of_op = sym.indep_of_op;
    // Prefix sums and the flat atom -> operator lookup.
    c.asym_off.assign(c.n_indep + 1, 0);
    for (int k = 0; k < c.n_indep; ++k) c.asym_off[k+1] = c.asym_off[k] + c.sites_of_indep[k];
    c.full_off.assign(c.n_op + 1, 0);
    for (int op = 0; op < c.n_op; ++op)
        c.full_off[op+1] = c.full_off[op] + c.sites_of_indep[c.indep_of_op[op]];
    c.op_of_atom.assign(c.full_off.back(), 0);
    for (int op = 0; op < c.n_op; ++op)
        for (int i = c.full_off[op]; i < c.full_off[op+1]; ++i) c.op_of_atom[i] = op;
    c.cell = opts.relax_cell;
    c.h0 = cell_to_mat(init.cell);
    c.periodic = init.periodic;
    c.cell_factor = static_cast<double>(std::max(1, c.natoms()));
    c.max_step = opts.max_step; c.max_strain_step = opts.max_strain_step;
    c.force_tol = opts.force_tol; c.stress_tol = opts.stress_tol;
    c.max_iter = opts.max_iter_full;
    Mat3 h0inv = identity3();
    mat3_inv(c.h0.m, h0inv.m);  // identity if the reference cell is singular
    c.R.resize(c.n_op); c.M0.resize(c.n_op); c.tc.resize(c.n_op);
    std::vector<Vec3> tfrac(c.n_op);
    for (int op = 0; op < c.n_op; ++op) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) c.R[op].m[i][j] = sym.rot[op * 9 + i * 3 + j];
        tfrac[op] = {sym.trans[op * 3], sym.trans[op * 3 + 1], sym.trans[op * 3 + 2]};
        c.M0[op] = matmul(c.h0, matmul(c.R[op], h0inv));
        c.tc[op] = matvec(c.h0, tfrac[op]);
    }
    std::vector<int> ref_op(c.n_indep, -1);
    for (int op = 0; op < c.n_op; ++op) {
        const int k = c.indep_of_op[op];
        if (k >= 0 && k < c.n_indep && ref_op[k] < 0 && op_is_identity(c.R[op], tfrac[op]))
            ref_op[k] = op;
    }
    // Invalid state: n_op == 0 with every size cleared, so build() yields an empty structure.
    auto invalidate = [&c]() -> SymState& {
        c.n_op = 0;
        c.asym_off.assign(1, 0);
        c.full_off.assign(1, 0);
        c.op_of_atom.clear();
        c.x.assign(c.ndof(), 0.0);   // nfree() == 0 now, so this is the strain block only
        return c;
    };

    for (int k = 0; k < c.n_indep; ++k)
        if (ref_op[k] < 0) return invalidate();  // no reference placement for molecule k

    // Input must already be in compacted ragged (op, site) order.
    if (static_cast<int>(init.atomic_numbers.size()) != c.natoms() ||
        static_cast<int>(init.positions.size()) != c.natoms())
        return invalidate();

    c.x.assign(c.ndof(), 0.0);
    for (int k = 0; k < c.n_indep; ++k)
        for (int site = 0; site < c.sites_of_indep[k]; ++site) {
            const int atom = c.full_off[ref_op[k]] + site, a = c.asym_off[k] + site;
            for (int comp = 0; comp < 3; ++comp) c.x[3 * a + comp] = init.positions[atom][comp];
        }
    c.Z = init.atomic_numbers;  // [natoms()] in ragged (op, site) order
    return c;
}


// Flatten host SymStates into the device relaxer's staging struct.
inline PetSymBatch flatten_sym_states(const std::vector<SymState>& st) {
    const int B = static_cast<int>(st.size());
    PetSymBatch sb;
    sb.valid.assign(B, 1);
    sb.n_op.resize(B); sb.n_indep.resize(B);
    sb.sites_of_indep.resize(B); sb.asym_off.resize(B);
    sb.full_off.resize(B); sb.op_of_atom.resize(B);
    sb.x_init.resize(B); sb.M0.resize(B); sb.tc.resize(B); sb.R.resize(B);
    sb.indep_of_op.resize(B);
    for (int b = 0; b < B; ++b) {
        const SymState& c = st[b];
        sb.n_op[b] = c.n_op; sb.n_indep[b] = c.n_indep;
        sb.sites_of_indep[b] = c.sites_of_indep;
        sb.asym_off[b] = c.asym_off;
        sb.full_off[b] = c.full_off;
        sb.op_of_atom[b] = c.op_of_atom;
        sb.x_init[b] = c.x;
        sb.indep_of_op[b] = c.indep_of_op;
        sb.M0[b].resize(static_cast<std::size_t>(c.n_op) * 9);
        sb.R[b].resize(static_cast<std::size_t>(c.n_op) * 9);
        sb.tc[b].resize(static_cast<std::size_t>(c.n_op) * 3);
        for (int op = 0; op < c.n_op; ++op) {
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    sb.M0[b][op * 9 + i * 3 + j] = c.M0[op].m[i][j];
                    sb.R[b][op * 9 + i * 3 + j] = c.R[op].m[i][j];
                }
            for (int e = 0; e < 3; ++e) sb.tc[b][op * 3 + e] = c.tc[op][e];
        }
    }
    return sb;
}

}  // namespace flexopt
}  // namespace klasp
