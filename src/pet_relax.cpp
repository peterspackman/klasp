#include "klasp/pet_relax.hpp"

#include "klasp/constants.hpp"
#include "klasp/io.hpp"
#include "klasp/log.hpp"
#include "klasp/element.hpp"
#include "klasp/flex_relax.hpp"
#include "klasp/crystal_dof.hpp"
#include "klasp/sgdata.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace klasp {
using namespace flexopt;

// ---------------------------------------------------------------------------
// Stage A: rigid-molecule pre-relaxation (translation + rotation per group,
// fixed cell), on the shared batched L-BFGS through flexopt::RigidState.
// ---------------------------------------------------------------------------
static void relax_rigid(const PetEvaluator& eval, const PetRelaxOptions& opts,
                        CrystalGeometry& s, PetRelaxResult& res) {
    std::vector<RigidState> st{make_rigid_state(s, opts.rigid_groups, opts)};
    if (st.front().ndof() == 0) return;   // no non-empty group: nothing to move
    std::vector<PetRelaxResult> rr(1);
    run_lbfgs_batch(eval, st, rr, opts);
    s = rr.front().structure;
    res.energy_trace = std::move(rr.front().energy_trace);
    res.rigid_iterations = rr.front().full_iterations;
    res.energy = rr.front().energy;
    res.max_force = rr.front().max_force;
}


std::vector<PetRelaxResult> pet_relax_batch(const PetEvaluator& eval,
                                            const std::vector<CrystalGeometry>& inits,
                                            const PetRelaxOptions& opts) {
    const int B = static_cast<int>(inits.size());
    std::vector<PetRelaxResult> res(B);
    for (int b = 0; b < B; ++b) res[b].structure = inits[b];

    if (opts.device && B >= 1 && opts.relax_cell)
        return eval.relax_batch_device(inits, opts);

    std::vector<CellState> st(B);
    for (int b = 0; b < B; ++b) {
        CellState& c = st[b];
        c.N = static_cast<int>(inits[b].atomic_numbers.size());
        c.cell = opts.relax_cell;
        c.cell_factor = static_cast<double>(c.N > 0 ? c.N : 1);
        c.h0 = cell_to_mat(inits[b].cell);
        c.Z = inits[b].atomic_numbers;
        c.periodic = inits[b].periodic;
        c.x.assign(c.ndof(), 0.0);
        for (int i = 0; i < c.N; ++i)
            for (int d = 0; d < 3; ++d) c.x[3 * i + d] = inits[b].positions[i][d];
        c.max_step = opts.max_step; c.max_strain_step = opts.max_strain_step;
        c.force_tol = opts.force_tol; c.stress_tol = opts.stress_tol;
        c.max_iter = opts.max_iter_full;
    }
    run_lbfgs_batch(eval, st, res, opts);
    return res;
}


std::vector<PetRelaxResult> pet_relax_sym_batch(const PetEvaluator& eval,
                                                const std::vector<CrystalGeometry>& inits,
                                                const std::vector<PetSymmetry>& syms,
                                                const PetRelaxOptions& opts) {
    const int B = static_cast<int>(inits.size());
    std::vector<SymState> st;
    st.reserve(B);
    std::vector<PetRelaxResult> res(B);
    for (int b = 0; b < B; ++b) {
        st.push_back(make_sym_state(inits[b], syms[b], opts));
        res[b].structure = inits[b];
        if (st[b].n_op == 0) {  // no identity op for some independent molecule -> skip
            st[b].active = false;
            st[b].energy = eval.evaluate(inits[b], true).energy[0];
        }
    }

    // Device-resident symmetry-constrained relax; needs every structure valid
    // (identity operators present), otherwise the host loop runs.
    if (opts.device && opts.relax_cell && B >= 1) {
        bool all_valid = true;
        for (int b = 0; b < B; ++b) if (st[b].n_op == 0) { all_valid = false; break; }
        if (all_valid) {
            const PetSymBatch sb = flatten_sym_states(st);
            return eval.relax_batch_device(inits, opts, &sb);
        }
    }

    run_lbfgs_batch(eval, st, res, opts);
    // For skipped (n_op==0) states, build() is empty and convergence is bogus;
    // restore the init geometry and mark not-converged.
    for (int b = 0; b < B; ++b)
        if (st[b].n_op == 0) { res[b].structure = inits[b]; res[b].full_iterations = 0; res[b].converged = false; }
    return res;
}

std::vector<PetRelaxResult> pet_relax_chunked(const PetEvaluator& eval,
                                              const std::vector<CrystalGeometry>& inits,
                                              const std::vector<PetSymmetry>& syms,
                                              const PetRelaxOptions& opts) {
    const int M = static_cast<int>(inits.size());
    if (!syms.empty() && static_cast<int>(syms.size()) != M)
        throw std::invalid_argument("pet_relax_chunked: one symmetry per structure, or none");
    std::vector<PetRelaxResult> out;
    out.reserve(M);
    int i = 0;
    while (i < M) {
        const int cap_atoms = eval.recommended_batch_atoms();
        int j = i, atoms = 0;
        while (j < M) {
            const int n = static_cast<int>(inits[j].atomic_numbers.size());
            if (j > i && atoms + n > cap_atoms) break;
            atoms += n;
            ++j;
        }
        logging::detail("  PET relax chunk: structures {}-{} of {}, {} atoms (budget {})\n",
                        i + 1, j, M, atoms, cap_atoms);
        const std::vector<CrystalGeometry> cs(inits.begin() + i, inits.begin() + j);
        std::vector<PetRelaxResult> pr;
        if (syms.empty()) {
            pr = pet_relax_batch(eval, cs, opts);
        } else {
            const std::vector<PetSymmetry> cy(syms.begin() + i, syms.begin() + j);
            pr = pet_relax_sym_batch(eval, cs, cy, opts);
        }
        out.insert(out.end(), std::make_move_iterator(pr.begin()),
                   std::make_move_iterator(pr.end()));
        i = j;
    }
    return out;
}

PetRelaxResult pet_relax_symmetric(const PetEvaluator& eval, const CrystalGeometry& init,
                                   const PetSymmetry& sym, const PetRelaxOptions& opts) {
    return pet_relax_sym_batch(eval, {init}, {sym}, opts).front();
}

PetRelaxResult pet_relax(const PetEvaluator& eval, const CrystalGeometry& init,
                         const PetRelaxOptions& opts) {
    PetRelaxResult res;
    CrystalGeometry s = init;
    if (!opts.rigid_groups.empty() && opts.max_iter_rigid > 0)
        relax_rigid(eval, opts, s, res);
    // Stage B is a batch of one.
    std::vector<PetRelaxResult> br = pet_relax_batch(eval, {s}, opts);
    const std::vector<double> trace = std::move(res.energy_trace);
    const int rigid_it = res.rigid_iterations;
    res = std::move(br.front());
    res.rigid_iterations = rigid_it;
    // Stage A's trace comes first; the batch reports its own from the first
    // Stage B evaluation, so front() is the energy before either stage ran.
    res.energy_trace.insert(res.energy_trace.begin(), trace.begin(), trace.end());
    return res;
}


// See pet_relax.hpp. The molecule sits alone in a cube 20 Bohr larger than its
// extent; relax_cell is off because the box is not a crystal.
double pet_isolated_reference(const PetEvaluator& eval, const BasisConfig& basis) {
    const int ntypes = std::max(1, basis.n_mols);
    double sum = 0.0;
    for (int t = 0; t < ntypes; ++t) {
        const auto& pos = t < (int)basis.mol_body_positions.size()
                        ? basis.mol_body_positions[t] : basis.body_positions;
        const auto& el = t < (int)basis.mol_elements.size()
                       ? basis.mol_elements[t] : basis.elements;
        const int n = t < (int)basis.mol_n_sites.size() ? basis.mol_n_sites[t] : basis.n_sites;
        CrystalGeometry s;
        s.periodic = false;
        s.positions.resize(n);
        s.atomic_numbers.resize(n);
        double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
        for (int i = 0; i < n; ++i) {
            for (int c = 0; c < 3; ++c) {
                s.positions[i][c] = pos[i][c];
                lo[c] = std::min(lo[c], pos[i][c]);
                hi[c] = std::max(hi[c], pos[i][c]);
            }
            s.atomic_numbers[i] = element(el[i]).atomic_number;
        }
        double ext = 0;
        for (int c = 0; c < 3; ++c) ext = std::max(ext, hi[c] - lo[c]);
        const double L = ext + 20.0;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) s.cell[r][c] = (r == c) ? L : 0.0;
        PetRelaxOptions opts;
        opts.relax_cell = false;
        opts.verbose = false;
        opts.max_iter_full = 300;
        sum += pet_relax(eval, s, opts).energy * AU_TO_KJ_PER_MOL;
    }
    return sum;
}

PetSymmetry expanded_row_symmetry(int space_group, int n_mols,
                                  const std::vector<int>& sites_of_indep) {
    PetSymmetry sy;
    const int nm = std::max(1, n_mols);
    const auto& sgi = get_space_group(space_group);
    const int nop = sgi.n_symops;
    sy.n_indep = nm;
    sy.sites_of_indep = sites_of_indep;
    sy.sites_of_indep.resize(nm, 0);   // valid() rejects a zero, so a short list fails loudly
    sy.n_op = nop * nm;
    sy.rot.resize(static_cast<std::size_t>(sy.n_op) * 9);
    sy.trans.resize(static_cast<std::size_t>(sy.n_op) * 3);
    sy.indep_of_op.resize(sy.n_op);
    for (int op = 0; op < nop; ++op) {
        double S[3][3], t[3];
        get_symop(sgi, op, S, t);
        for (int m = 0; m < nm; ++m) {
            const int idx = op * nm + m;   // matches expand_rigid_structure's atom order
            sy.indep_of_op[idx] = m;
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) sy.rot[idx * 9 + p * 3 + q] = S[p][q];
            for (int k = 0; k < 3; ++k) sy.trans[idx * 3 + k] = t[k];
        }
    }
    return sy;
}

}  // namespace klasp
