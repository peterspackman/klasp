#include <cstdlib>
#include "klasp/log.hpp"
#include "klasp/sg_worker.hpp"
#include "klasp/padding.hpp"
#include "klasp/sgdata.hpp"
#include "klasp/descriptors.hpp"
#include "klasp/constants.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/batch_stream.hpp"
#include "klasp/scheduling.hpp"
#include "klasp/fft_generator.hpp"
#include "klasp/generator.hpp"
#include "klasp/cell.hpp"
#include "klasp/rotation.hpp"
#include "klasp/warnings.hpp"
#include "klasp/vacuum.hpp"
#include "klasp/db_tool.hpp"
#include "klasp/flex_descriptors.hpp"
#ifdef KLASP_HAS_PET
#include "klasp/pet_relax.hpp"
#include "klasp/element.hpp"
#endif

#include <fmt/format.h>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace klasp {

static double compute_density(const double cp[6], double mol_mass, int Z);

// Rotate the stored orientations from the optimizer's cell frame into the
// canonical one (a along x, b in xy) that a cell rebuilt from cell_params uses;
// see align_orientations_to_canonical_cell. Fractional translations are invariant.
static void canonicalize_cell_frame(
    const Kokkos::View<RealType**, Kokkos::LayoutRight, Kokkos::HostSpace>& h_cell_params,
    Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace>& h_rt,
    Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace>& h_rr,
    const Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace>& h_cell_matrix,
    int count, int n_indep)
{
    (void)h_rt;  // translations are fractional → invariant under R_align
    std::vector<RealType> rv(3 * static_cast<size_t>(n_indep));
    for (int b = 0; b < count; ++b) {
        RealType M_orig[3][3];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                M_orig[r][c] = h_cell_matrix(b, r, c);
        RealType cp[6];
        for (int k = 0; k < 6; ++k) cp[k] = h_cell_params(b, k);
        for (int k = 0; k < n_indep; ++k)
            for (int d = 0; d < 3; ++d) rv[3 * k + d] = h_rr(b, k, d);
        // A singular cell has no alignment; its orientations are left as they are.
        if (!align_orientations_to_canonical_cell(M_orig, cp, rv.data(), n_indep)) continue;
        for (int k = 0; k < n_indep; ++k)
            for (int d = 0; d < 3; ++d) h_rr(b, k, d) = rv[3 * k + d];
    }
}

SGWorker::BatchDescriptors SGWorker::compute_batch_descriptors() {
    using Clock = std::chrono::steady_clock;
    using ms = std::chrono::duration<double, std::milli>;
    BatchDescriptors out;
    auto t_amd0 = Clock::now();
    out.amd = compute_amd(bws_.tables, bws_.geom, AMD_K, /*want_pdd=*/store_pdd_);
    auto t_amd1 = Clock::now();
    out.powder = compute_powder_pattern(bws_.geom, powder_ff_);
    auto t_pow1 = Clock::now();
    out.amd_ms = ms(t_amd1 - t_amd0).count();
    out.powder_ms = ms(t_pow1 - t_amd1).count();
    return out;
}

std::vector<StructureResult> SGWorker::pack_results(
    int stage, int count,
    const std::vector<int>& batch_seeds,
    int n_mols,
    const Kokkos::View<RealType*, Kokkos::HostSpace>& h_E,
    const Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace>& h_rt,
    const Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace>& h_rr,
    const BatchDescriptors& desc,
    std::vector<std::vector<double>>& pdds_out,
    const BatchDiagnostics& diag,
    const BatchDiagnostics& s1_diag)
{
    const bool have_pdd = store_pdd_ && !desc.amd.pdd.empty();
    std::vector<StructureResult> results;
    results.reserve(count);
    // Batch slot of each packed row. Non-finite rows are dropped, so the packed
    // index is not the slot, and stage 3 reads geometry by slot.
    last_pack_slots_.clear();
    last_pack_slots_.reserve(count);
    for (int b = 0; b < count; ++b) {
        if (!std::isfinite(h_E(b))) continue;

        double cp[6];
        for (int j = 0; j < 6; ++j) cp[j] = h_cell_params_(b, j);

        // Vacuum gap, measured here so a --max-vacuum rejection is never packed and
        // its seed is reopened.
        double gap = -1.0;
        if (basis_) {
            const CrystalGeometry ex = expand_rigid_structure(
                *basis_, space_group_, cp, &h_rt(b, 0, 0), &h_rr(b, 0, 0), n_mols);
            const int na = static_cast<int>(ex.positions.size());
            if (na > 1) {
                // Zeros for a degenerate cell: the gap then reads as zero and the
                // structure is kept.
                double inv[3][3] = {};
                mat3_inv(ex.cell, inv);
                std::vector<double> frac(static_cast<std::size_t>(3 * na));
                for (int i = 0; i < na; ++i)
                    for (int k = 0; k < 3; ++k)
                        frac[3*i + k] = inv[k][0]*ex.positions[i][0]
                                      + inv[k][1]*ex.positions[i][1]
                                      + inv[k][2]*ex.positions[i][2];
                // cell_params are Bohr; the gap is in Angstrom.
                double cp_ang[6];
                for (int j = 0; j < 3; ++j) cp_ang[j] = cp[j] * BOHR_TO_ANGSTROM;
                for (int j = 3; j < 6; ++j) cp_ang[j] = cp[j];
                gap = max_vacuum_gap(cp_ang, frac.data(), na).gap;
            }
        }
        if (max_vacuum_ > 0.0 && stage == primary_stage_ && gap > max_vacuum_) {
            ++n_vacuum_dropped_;
            continue;
        }

        last_pack_slots_.push_back(b);
        StructureResult r{};
        r.seed = batch_seeds[b];
        r.space_group = space_group_;
        r.stage = stage;
        r.energy = h_E(b) * AU_TO_KJ_PER_MOL / Z_;
        r.n_mols = n_mols;
        for (int j = 0; j < 6; ++j) r.cell_params[j] = cp[j];
        for (int m = 0; m < n_mols; ++m)
            for (int d = 0; d < 3; ++d) {
                r.translation[m*3+d] = h_rt(b, m, d);
                r.rotation[m*3+d]    = h_rr(b, m, d);
            }
        r.density = compute_density(r.cell_params, mol_mass_, Z_);
        r.vacuum_gap = gap;
        r.warnings = classify_warnings(r.cell_params, r.density, gap);
        std::copy_n(desc.amd.vectors[b].data(), AMD_K, r.amd);
        for (int j = 0; j < POWDER_N_BINS; ++j)
            r.powder[j] = static_cast<float>(desc.powder.patterns[b][j]);
        if (diag.valid()) {
            r.grad_max = static_cast<float>(diag.grad_max(b));
            r.grad_rms = static_cast<float>(diag.grad_rms(b));
            r.iters_to_converge = diag.iters_to_converge(b);
            if (diag.has_tr()) {
                r.final_delta = static_cast<float>(diag.final_delta(b));
                r.n_restarts = diag.restarts(b);
            }
        }
        // Stage-1 convergence rides along on the stage-2 row.
        if (s1_diag.valid()) {
            r.s1_grad_max = static_cast<float>(s1_diag.grad_max(b));
            r.s1_iters_to_converge = s1_diag.iters_to_converge(b);
        }
        results.push_back(r);
        // PDD aligned 1:1 with results (same b-filtering). Stage 0 / disabled
        // => empty (desc.amd.pdd is empty there).
        pdds_out.push_back(have_pdd ? desc.amd.pdd[b] : std::vector<double>{});
        ++local_valid_;
    }
    return results;
}

// Cell volume in Ang^3 from cell params (Bohr, degrees)
static double cell_volume_ang3(const double cp[6]) {
    const double ang[6] = {cp[0] * BOHR_TO_ANGSTROM, cp[1] * BOHR_TO_ANGSTROM,
                           cp[2] * BOHR_TO_ANGSTROM, cp[3], cp[4], cp[5]};
    return cell_volume_from_params(ang);
}

static double compute_density(const double cp[6], double mol_mass, int Z) {
    const double vol = cell_volume_ang3(cp);
    return vol > 0 ? Z * mol_mass * AMU_TO_GRAM / (vol * 1e-24) : 0.0;
}

void SGWorker::init(int space_group, int target, int batch_size,
                    const BasisConfig& basis, Workspace& ws,
                    const BatchOptimizeSettings& bsettings,
                    const std::vector<double>& body_pos,
                    const std::vector<double>& cov_radii,
                    const std::vector<double>& vdw_radii,
                    double mol_mass, double tvp,
                    int effective_max_rank, int rank, int nranks) {
    space_group_ = space_group;
    Z_ = get_space_group(space_group).n_symops;  // crystallographic Z (formula units)
    target_ = target;
    remaining_ = target;
    batch_size_ = batch_size;
    mol_mass_ = mol_mass;
    tvp_ = tvp;
    effective_max_rank_ = effective_max_rank;
    rank_ = rank;
    nranks_ = nranks;

    basis_ = &basis;
    bsettings_ = &bsettings;

    csp_mol_ = build_csp_molecule(basis, space_group);

    // n_atoms = total symops * sites_per_mol (includes all molecule types)
    int n_atoms = csp_mol_.n_symops * basis.n_sites;
    // Holds a whole generated block (~1.2x a batch of valids plus variance).
    gen_cap_ = std::max(batch_size + batch_size / 2, batch_size + 64);
    bws_ = allocate_batch_workspace(gen_cap_, ws, csp_mol_);

    // Per-atom element symbols, for the powder form factors and PET. Atoms are
    // laid out molecule-major; mol_m's component is sym_indep_index(m) (0 for
    // Z'=1). Padded ghost atoms (site >= mol_n_sites[comp]) get "X".
    atom_elements_.assign(n_atoms, "X");
    {
        auto h_sidx = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), csp_mol_.sym_indep_index);
        const int n_sites = basis.n_sites;
        for (int mol = 0; mol < csp_mol_.n_symops; ++mol) {
            int comp = basis.n_mols > 1 ? h_sidx(mol) : 0;
            int ns_comp = (comp < (int)basis.mol_n_sites.size())
                        ? basis.mol_n_sites[comp] : (int)basis.elements.size();
            for (int si = 0; si < n_sites; ++si) {
                int a = mol * n_sites + si;
                if (si >= ns_comp) { atom_elements_[a] = "X"; continue; }
                if (basis.n_mols > 1 && comp < (int)basis.mol_elements.size()
                    && si < (int)basis.mol_elements[comp].size())
                    atom_elements_[a] = basis.mol_elements[comp][si];
                else if (si < (int)basis.elements.size())
                    atom_elements_[a] = basis.elements[si];
            }
        }
    }

    // Indices of the real atoms, for consumers that gather rather than skip
    // (PET, stored flexible geometry).
    {
        std::vector<int> per_mol(csp_mol_.n_symops, basis.n_sites);
        auto h_sidx2 = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), csp_mol_.sym_indep_index);
        for (int mol = 0; mol < csp_mol_.n_symops; ++mol) {
            const int comp = basis.n_mols > 1 ? h_sidx2(mol) : 0;
            per_mol[mol] = (comp < (int)basis.mol_n_sites.size())
                         ? basis.mol_n_sites[comp] : basis.n_sites;
        }
        real_atoms_ = real_atom_indices(basis.n_sites, per_mol);
        basis_mol_n_sites_.assign(csp_mol_.n_indep, basis.n_sites);
        for (int k = 0; k < csp_mol_.n_indep; ++k)
            basis_mol_n_sites_[k] = (k < (int)basis.mol_n_sites.size())
                                  ? basis.mol_n_sites[k] : basis.n_sites;
    }

    // Cromer-Mann form factors for the powder-pattern descriptor. Built once
    // from the per-atom element labels (padded ghost atoms → zero scatterers).
    powder_ff_ = build_powder_form_factors(atom_elements_);

    // Populate cov + vdw radii for rigid press, sized n_indep × n_sites_per_mol
    // so per-component lookups work for co-crystals. Layout matches
    // basis.body_positions: padded ghost atoms get radius 0.
    {
        const int total_body_rp = csp_mol_.n_indep * basis.n_sites;
        auto h_cr = Kokkos::View<RealType*, Kokkos::HostSpace>("h_cov_rp", total_body_rp);
        auto h_vr = Kokkos::View<RealType*, Kokkos::HostSpace>("h_vdw_rp", total_body_rp);
        for (int i = 0; i < total_body_rp; i++) {
            h_cr(i) = cov_radii[i];
            h_vr(i) = vdw_radii[i];
        }
        bws_.tables.cov_radii = SView1D<RealType, ExecSpace>("b_cov_radii", total_body_rp);
        bws_.tables.vdw_radii = SView1D<RealType, ExecSpace>("b_vdw_radii", total_body_rp);
        Kokkos::deep_copy(bws_.tables.cov_radii, h_cr);
        Kokkos::deep_copy(bws_.tables.vdw_radii, h_vr);
    }

    // Pre-allocate cell strain views for rigid press stage 0 (sized to gen_cap_
    // so a whole generated block participates).
    bws_.dofs.cell_ref = SView3D<RealType, ExecSpace>("cell_ref", gen_cap_, 3, 3);
    bws_.dofs.strain = SView2D<RealType, ExecSpace>("strain", gen_cap_, 6);
    bws_.dofs.d_g_strain = SView2D<RealType, ExecSpace>("d_g_strain", gen_cap_, 6);

    h_cell_params_ = Kokkos::View<RealType**, Kokkos::HostSpace>("h_cp", gen_cap_, 6);

    // Device Views for GPU generation. For co-crystals body/cov/vdw data are
    // padded as n_mols × n_sites (host layout already pads ghost atoms to 0).
    // Generation writes straight into bws_ (sized gen_cap_); no separate buffers.
    int n_mols = basis.n_mols;
    int total_body = n_mols * basis.n_sites;
    d_body_pos_ang_ = View2D<RealType>("d_bp_ang", total_body, 3);
    d_cov_radii_ = View1D<RealType>("d_cov", total_body);
    d_vdw_radii_ = View1D<RealType>("d_vdw", total_body);
    d_batch_seeds_ = View1D<int>("d_seeds", gen_cap_);
    d_mol_n_sites_ = View1D<int>("d_mns", n_mols);
    d_mol_site_offsets_ = View1D<int>("d_mso", n_mols);
    d_mol_R_ = View1D<RealType>("d_mol_R", n_mols);
    {
        auto h_bp = Kokkos::create_mirror_view(d_body_pos_ang_);
        auto h_cr = Kokkos::create_mirror_view(d_cov_radii_);
        auto h_vr = Kokkos::create_mirror_view(d_vdw_radii_);
        auto h_mns = Kokkos::create_mirror_view(d_mol_n_sites_);
        auto h_mso = Kokkos::create_mirror_view(d_mol_site_offsets_);
        auto h_mR  = Kokkos::create_mirror_view(d_mol_R_);
        for (int i = 0; i < total_body; ++i) {
            for (int d = 0; d < 3; ++d)
                h_bp(i, d) = body_pos[i*3+d] * BOHR_TO_ANGSTROM;  // Bohr → Angstrom
            h_cr(i) = cov_radii[i];
            h_vr(i) = vdw_radii[i];
        }
        for (int m = 0; m < n_mols; ++m) {
            h_mns(m) = (n_mols > 1) ? basis.mol_n_sites[m] : basis.n_sites;
            h_mso(m) = m * basis.n_sites;
            // Per-mol bounding sphere (Angstrom): centroid → max(|atom-c| + vdw).
            // Used as a coarse mol-pair filter in the multi-comp generator.
            int ns_m = h_mns(m);
            int off  = h_mso(m);
            double cx = 0, cy = 0, cz = 0;
            for (int i = 0; i < ns_m; ++i) {
                cx += h_bp(off + i, 0);
                cy += h_bp(off + i, 1);
                cz += h_bp(off + i, 2);
            }
            cx /= ns_m; cy /= ns_m; cz /= ns_m;
            double R_max = 0;
            for (int i = 0; i < ns_m; ++i) {
                double dx = h_bp(off + i, 0) - cx;
                double dy = h_bp(off + i, 1) - cy;
                double dz = h_bp(off + i, 2) - cz;
                double r = std::sqrt(dx*dx + dy*dy + dz*dz) + h_vr(off + i);
                R_max = std::max(R_max, r);
            }
            h_mR(m) = static_cast<RealType>(R_max);
        }
        Kokkos::deep_copy(ExecSpace(), d_body_pos_ang_, h_bp);
        Kokkos::deep_copy(ExecSpace(), d_cov_radii_, h_cr);
        Kokkos::deep_copy(ExecSpace(), d_vdw_radii_, h_vr);
        Kokkos::deep_copy(ExecSpace(), d_mol_R_, h_mR);
        Kokkos::deep_copy(ExecSpace(), d_mol_n_sites_, h_mns);
        Kokkos::deep_copy(ExecSpace(), d_mol_site_offsets_, h_mso);
    }

    workspace_allocated_ = true;
}

// Generate one block of valid (non-colliding) structures into bws_ slots
// [0, count) and read back seeds + cell params. Seeds are over-generated only
// enough to yield ~`batch` valids, and the whole yield is returned: count may
// exceed `batch`, bounded by gen_cap_. Sets bws_.geom.n_batch = count.
int SGWorker::generate_one(int batch, std::vector<int>& batch_seeds) {
    int count = 0;
    const int n_mols = csp_mol_.n_indep;

    // The destination, shared by every generator.
    DofBatch out{bws_.geom.cell_params, bws_.dofs.translations,
                 bws_.dofs.rot_vectors, d_batch_seeds_, gen_cap_, n_mols};

    if (collision_method_ == CollisionMethod::FFT_PACK && n_mols == 1) {
        FftGenerator gen;
        gen.body_positions_ang = d_body_pos_ang_;
        gen.cov_radii = d_cov_radii_;
        gen.vdw_radii = d_vdw_radii_;   // the volume budget is measured on these
        gen.n_sites = basis_->n_sites;
        gen.config.tvp = tvp_;
        gen.config.collision_buffer = 0.5;
        gen.config.k_per_trial = fft_k_per_trial_;
        gen.config.grid_spacing = fft_grid_spacing_;
        gen.config.collision_tol = fft_collision_tol_;
        gen.config.max_aspect = fft_max_aspect_;

        const int n_trials = std::max(batch / gen.config.k_per_trial + 1, 4);
        const GeneratorRequest req{space_group_,
                                   static_cast<int>(claim_seeds(n_trials)), n_trials};
        count = gen.generate(req, out);
        last_rejected_ = n_trials * gen.config.k_per_trial - count;
    } else {
        QuasiRandomGenerator gen;
        gen.body_positions_ang = d_body_pos_ang_;
        gen.cov_radii = d_cov_radii_;
        gen.vdw_radii = d_vdw_radii_;
        gen.n_sites = basis_->n_sites;
        gen.tvp = tvp_;
        gen.collision_buffer = 0.5;
        gen.collision_method = collision_method_;
        gen.mol_n_sites = d_mol_n_sites_;
        gen.mol_site_offsets = d_mol_site_offsets_;
        gen.mol_bounding_radius = d_mol_R_;
        gen.nudge_budget = gen_nudge_;
        gen.contract = gen_contract_;

        // Try enough seeds for ~GEN_MARGIN x one batch of valids, from the
        // running acceptance estimate, clamped to gen_cap_ so the yield always fits.
        constexpr double GEN_MARGIN = 1.2;
        const int n_try = std::min(gen_cap_, std::max(batch + 1, static_cast<int>(
            std::ceil(batch * GEN_MARGIN / std::max(accept_est_, 0.01)))));
        const GeneratorRequest req{space_group_,
                                   static_cast<int>(claim_seeds(n_try)), n_try};

        int raw_valid = 0;
        count = gen.generate(req, out, &raw_valid, &gen_prior_);
        gen_prior_valid_ = true;
        last_rejected_ = n_try - raw_valid;

        const double acceptance = n_try > 0 ? double(raw_valid) / n_try : accept_est_;
        accept_est_ = 0.7 * accept_est_ + 0.3 * std::max(acceptance, 0.01);
    }
    if (count <= 0) return 0;
    bws_.geom.n_batch = count;

    // Read back seeds + cell params for the whole block.
    {
        auto h_seeds = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_batch_seeds_);
        batch_seeds.resize(count);
        for (int i = 0; i < count; ++i) batch_seeds[i] = h_seeds(i);
        Kokkos::deep_copy(
            Kokkos::subview(h_cell_params_, std::make_pair(0, count), Kokkos::ALL),
            Kokkos::subview(bws_.geom.cell_params, std::make_pair(0, count), Kokkos::ALL));
    }
    return count;
}

// ---------------------------------------------------------------------------
// The relaxation pipeline, as data. See StageSpec in sg_worker.hpp.
// ---------------------------------------------------------------------------

// Dispatch the runtime multipole rank to the template parameter.
static BatchOptimizeResult optimize_at_rank(int max_rank, BatchWorkspace& bws,
                                            const BatchOptimizeSettings& rs) {
    switch (max_rank) {
        case 0:  return batch_optimize_asymunit<0>(bws, rs);
        case 1:  return batch_optimize_asymunit<1>(bws, rs);
        case 2:  return batch_optimize_asymunit<2>(bws, rs);
        case 3:  return batch_optimize_asymunit<3>(bws, rs);
        default: return batch_optimize_asymunit<4>(bws, rs);
    }
}

static StreamResult stream_at_rank(int max_rank, BatchWorkspace& bws, StreamPool& pool,
                                   const BatchOptimizeSettings& rs, StreamPool* out) {
    switch (max_rank) {
        case 0:  return batch_optimize_stream_tr<0>(bws, pool, rs, out);
        case 1:  return batch_optimize_stream_tr<1>(bws, pool, rs, out);
        case 2:  return batch_optimize_stream_tr<2>(bws, pool, rs, out);
        case 3:  return batch_optimize_stream_tr<3>(bws, pool, rs, out);
        default: return batch_optimize_stream_tr<4>(bws, pool, rs, out);
    }
}

// The same dispatch for the Metropolis stepper; no destination pool (see
// stream_mc.hpp).
static StreamResult stream_mc_at_rank(int max_rank, BatchWorkspace& bws, StreamPool& pool,
                                      const BatchOptimizeSettings& rs, const McSettings& mc) {
    switch (max_rank) {
        case 0:  return batch_optimize_stream_mc<0>(bws, pool, rs, mc);
        case 1:  return batch_optimize_stream_mc<1>(bws, pool, rs, mc);
        case 2:  return batch_optimize_stream_mc<2>(bws, pool, rs, mc);
        case 3:  return batch_optimize_stream_mc<3>(bws, pool, rs, mc);
        default: return batch_optimize_stream_mc<4>(bws, pool, rs, mc);
    }
}

std::vector<SGWorker::StageSpec> SGWorker::stage_plan() const {
    // The run's --iterations is the multipole stage's count; stage 1 falls back
    // to it when --iterations-stage1 was not given.
    const int iters = bsettings_->n_iterations;
    std::vector<StageSpec> plan;

    StageSpec charges;
    charges.max_rank = 0;
    charges.stage_id = STAGE_CHARGES;
    charges.n_iterations = (stage1_iterations_ > 0) ? stage1_iterations_ : iters;
    // External pressure applies to the charge-only pass only; stage 2 runs at
    // P=0 so stored energies are lattice energies.
    charges.pressure_gpa = static_cast<RealType>(stage1_pressure_gpa_);
    charges.fix_angles = fix_angles_stage1_;
    charges.warm_hessian = false;   // builds the molecular Hessian fresh
    charges.max_iters_per_slot = 0; // the stage-1 cap is n_iterations
    charges.reanchor_after = true;
    plan.push_back(charges);

    if (effective_max_rank_ > 0) {
        StageSpec mp;
        mp.max_rank = effective_max_rank_;
        mp.stage_id = STAGE_MULTIPOLE;
        mp.n_iterations = iters;
        mp.pressure_gpa = RealType(0);
        mp.fix_angles = false;      // the full lattice-allowed shape
        mp.warm_hessian = true;     // warm-starts from stage 1 (same batch)
        mp.max_iters_per_slot = max_iters_per_slot_;
        mp.reanchor_after = false;
        plan.push_back(mp);
    }
    return plan;
}

void SGWorker::apply_stage(const StageSpec& s, BatchOptimizeSettings& rs) const {
    rs.n_iterations = s.n_iterations;
    rs.pressure_gpa = s.pressure_gpa;
    rs.reuse_hessian = s.warm_hessian;
    rs.max_iters_per_slot = s.max_iters_per_slot;
    // The generator's volume budget is Angstrom^3; geom.volume is Bohr^3.
    constexpr double A3_TO_BOHR3 =
        ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR;
    rs.max_cell_volume = gen_prior_valid_
        ? RealType(VOLUME_ABORT_FACTOR * gen_prior_.max_volume * A3_TO_BOHR3)
        : RealType(0);
}

void SGWorker::run_rigid_press(int count) {
    BatchOptimizeSettings rp;
    rp.n_iterations = rp_iterations_;
    rp.max_step = 0.1;
    rp.lbfgs_memory = 10;
    rp.pressure_gpa = rp_pressure_gpa_;
    rp.cell_scale = 0.5;
    rp.max_strain_step = rp_max_strain_step_;
    auto& sg = get_space_group(space_group_);
    batch_optimize_rigid_press(bws_, rp, static_cast<int>(sg.lattice));
    // Recompute cell_params from the deformed cell on device.
    cell_matrix_to_params_device(bws_.geom, count);
}

void SGWorker::set_stage_strain_dofs(bool fix_angles) {
    if (fixed_cell_) { bws_.dofs.n_cell_dof = 0; return; }
    auto& sg = get_space_group(space_group_);
    batch_set_strain_dofs(bws_.geom, bws_.dofs, static_cast<int>(sg.lattice), fix_angles);
    Kokkos::deep_copy(bws_.dofs.cell_ref, bws_.geom.cell);
    Kokkos::deep_copy(bws_.dofs.strain, RealType(0));
    Kokkos::deep_copy(bws_.dofs.d_g_strain, RealType(0));
}

void SGWorker::reanchor_strain(int count, bool next_fix_angles) {
    if (bws_.dofs.n_cell_dof <= 0) return;
    cell_matrix_to_params_device(bws_.geom, count);
    // Anchor to the optimiser's cell before batch_setup_cells rebuilds
    // geom.cell in the canonical frame; the two differ by a rigid rotation.
    Kokkos::deep_copy(bws_.dofs.cell_ref, bws_.geom.cell);
    Kokkos::deep_copy(bws_.dofs.strain, RealType(0));
    batch_setup_cells(bws_.tables, bws_.geom);
    if (fixed_cell_) { bws_.dofs.n_cell_dof = 0; return; }
    auto& sg = get_space_group(space_group_);
    batch_set_strain_dofs(bws_.geom, bws_.dofs, static_cast<int>(sg.lattice), next_fix_angles);
    Kokkos::deep_copy(bws_.dofs.d_g_strain, RealType(0));
}

SGWorker::HostDofs SGWorker::evaluate_generated(int count) {
    BatchOptimizeSettings s0 = *bsettings_;
    s0.energy_only = true;
    s0.n_iterations = 1;
    s0.pressure_gpa = RealType(0);
    batch_expand_asymmetric_unit<0, ExecSpace>(bws_.tables, bws_.geom, bws_.dofs);
    batch_energy_gradient_asymunit<0>(bws_.tables, bws_.geom, bws_.dofs, s0, 0);
    Kokkos::deep_copy(
        Kokkos::subview(h_cell_params_, std::make_pair(0, count), Kokkos::ALL),
        Kokkos::subview(bws_.geom.cell_params, std::make_pair(0, count), Kokkos::ALL));
    HostDofs h;
    h.energies = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bws_.geom.energies);
    h.translations = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bws_.dofs.translations);
    h.rotations = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bws_.dofs.rot_vectors);
    return h;
}

SGWorker::BatchDescriptors SGWorker::null_descriptors() const {
    BatchDescriptors d{};
    d.amd.vectors.assign(bws_.geom.n_batch, std::vector<double>(AMD_K, 0.0));
    d.powder.patterns.assign(bws_.geom.n_batch, std::vector<double>(POWDER_N_BINS, 0.0));
    return d;
}

std::vector<StructureResult> SGWorker::run_batch(int batch_override) {
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    int batch = (batch_override > 0 && batch_override <= batch_size_)
                ? batch_override : batch_size_;

    // Stream when stage 2 is active and store_all_stages_ is off (per-stage rows
    // are not wired through the streaming path).
    const bool route_stream = stream_ && effective_max_rank_ > 0 && !store_all_stages_
                           && !generate_only_;
    // Metropolis is a streaming stage-2 stepper only; refuse any route that
    // would fall back to lockstep.
    if (sampler_ == SamplerKind::METROPOLIS && !route_stream && !generate_only_)
        throw std::runtime_error(
            "--sampler mc is a stage-2 stepper for the streaming engine; this run "
            "is routed to the lockstep trust region (output.store_all_stages "
            "or no multipole stage), and there is no lockstep "
            "Metropolis to fall back to");
    if (route_stream)
        return run_batch_stream(batch);

    int count = 0;
    int n_mols = csp_mol_.n_indep;
    std::vector<int> batch_seeds;  // seed per valid structure
    auto t_gen0 = Clock::now();

    count = generate_one(batch, batch_seeds);
    last_batch_structures_ = count;   // rows != structures once stage 0/3 are on
    if (count == 0) return {};
    auto t_gen1 = Clock::now();
    auto t_setup0 = Clock::now();
    batch_setup_cells(bws_.tables, bws_.geom);
    auto t_setup1 = Clock::now();

    auto t_post_setup = Clock::now();
    // Stage 0: rigid press (if enabled): resolve collisions, minimize volume.
    if (collision_method_ == CollisionMethod::RIGID_PRESS) run_rigid_press(count);

    const auto plan = stage_plan();

    // Set up cell strain DOFs for the first stage.
    set_stage_strain_dofs(plan.front().fix_angles);

    // Per-batch settings copy: each stage rewrites it; bsettings_ is shared.
    BatchOptimizeSettings run_settings = *bsettings_;
    run_settings.hooks.abort = abort_flag_;

    // Generator-only: score the raw structures and stop. Stage 0 is the primary
    // stage and is deduplicated and written as usual.
    if (generate_only_) {
        auto h0 = evaluate_generated(count);
        // Real descriptors: these are the rows the run is deduplicated on.
        auto desc = compute_batch_descriptors();
        std::vector<std::vector<double>> pdds;
        auto results = pack_results(STAGE_GENERATED, count, batch_seeds, n_mols,
                                     h0.energies, h0.translations, h0.rotations,
                                     desc, pdds);
        last_batch_pdds_ = std::move(pdds);
        auto t_end = Clock::now();
        const double total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
        const double per_ms = results.empty() ? 0 : total_ms / results.size();
        for (auto& r : results) r.time_ms = per_ms;
        return results;
    }

    // Optionally persist the raw generated state as stage 0: one charge-only
    // evaluation, no optimizer step.
    if (store_all_stages_) {
        auto h0 = evaluate_generated(count);
        // Zero-filled descriptors; see null_descriptors.
        BatchDescriptors desc0 = null_descriptors();
        std::vector<std::vector<double>> stage0_pdds;
        auto stage0 = pack_results(STAGE_GENERATED, count, batch_seeds, n_mols,
                                    h0.energies, h0.translations, h0.rotations,
                                    desc0, stage0_pdds);
        extra_stage_results_pending_.insert(extra_stage_results_pending_.end(),
                                        stage0.begin(), stage0.end());
        stage0_pdds_pending_.insert(stage0_pdds_pending_.end(),
                                    std::make_move_iterator(stage0_pdds.begin()),
                                    std::make_move_iterator(stage0_pdds.end()));
    }

    // Run the plan: per stage, take its settings, relax, then re-anchor the
    // strain reference for the next.
    run_settings.store_diagnostics = store_diagnostics_;  // persists across stages

    Clock::time_point t_s1_0 = t_post_setup, t_s1_1 = t_post_setup;
    Clock::time_point t_post_s1_sync = t_post_setup, t_post_s2_setup = t_post_setup;
    Clock::time_point t_s2_0 = t_post_setup, t_s2_1 = t_post_setup;
    std::vector<BatchOptimizeResult> stage_res(plan.size());

    for (size_t i = 0; i < plan.size(); ++i) {
        const StageSpec& st = plan[i];
        const bool last = (i + 1 == plan.size());
        apply_stage(st, run_settings);

        auto ts0 = Clock::now();
        stage_res[i] = optimize_at_rank(st.max_rank, bws_, run_settings);
        auto ts1 = Clock::now();

        // Sync cell_params from the deformed cell on device and widen the strain
        // mask for the next stage.
        if (st.reanchor_after)
            reanchor_strain(count, last ? false : plan[i + 1].fix_angles);
        auto ts2 = Clock::now();

        if (i == 0) { t_s1_0 = ts0; t_s1_1 = ts1; t_post_s1_sync = ts2; }
        else        { t_s2_0 = ts0; t_s2_1 = ts1; }

        if (last) break;

        // Intermediate rows, when asked for, with zero descriptors (never
        // deduplicated).
        if (store_all_stages_) {
            Kokkos::deep_copy(
                Kokkos::subview(h_cell_params_, std::make_pair(0, count), Kokkos::ALL),
                Kokkos::subview(bws_.geom.cell_params, std::make_pair(0, count), Kokkos::ALL));
            BatchDescriptors desc_i = null_descriptors();
            std::vector<std::vector<double>> pdds_i;
            auto rows = pack_results(st.stage_id, count, batch_seeds, n_mols,
                                     stage_res[i].final_energies,
                                     stage_res[i].final_translations,
                                     stage_res[i].final_rotations,
                                     desc_i, pdds_i, stage_res[i].diagnostics);
            extra_stage_results_pending_.insert(extra_stage_results_pending_.end(),
                                                std::make_move_iterator(rows.begin()),
                                                std::make_move_iterator(rows.end()));
        }

        // The next stage starts from the current device state; only the
        // neighbour lists need refreshing.
        batch_setup_cells(bws_.tables, bws_.geom);
        t_post_s2_setup = Clock::now();
    }

    auto& r1 = stage_res.front();

    // No multipole stage: the charge-only rows are the landscape, packed
    // without frame canonicalisation or the ML stage.
    if (plan.size() == 1) {
        auto desc = compute_batch_descriptors();
        // Refresh host cell_params mirror once for result packing.
        Kokkos::deep_copy(
            Kokkos::subview(h_cell_params_, std::make_pair(0, count), Kokkos::ALL),
            Kokkos::subview(bws_.geom.cell_params, std::make_pair(0, count), Kokkos::ALL));
        std::vector<std::vector<double>> pdds;
        auto results = pack_results(STAGE_CHARGES, count, batch_seeds, n_mols,
                                     r1.final_energies,
                                     r1.final_translations,
                                     r1.final_rotations, desc, pdds,
                                     r1.diagnostics);
        last_batch_pdds_ = std::move(pdds);
        auto t_end = Clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
        double per_ms = results.empty() ? 0 : total_ms / results.size();
        for (auto& r : results) r.time_ms = per_ms;
        if (perf_) {
            using ms = std::chrono::duration<double, std::milli>;
            perf_->record({space_group_, next_seed_, count, count, count,
                          static_cast<int>(results.size()),
                          ms(t_gen1 - t_gen0).count(), ms(t_setup1 - t_setup0).count(),
                          ms(t_s1_1 - t_s1_0).count(), 0,
                          desc.amd_ms, desc.powder_ms,
                          0, total_ms,
                          0, 0, 0, 0});
        }

        return results;
    }

    auto& r2 = stage_res.back();

    // Sync cell params on device after stage-2 cell optimization.
    if (bws_.dofs.n_cell_dof > 0) {
        cell_matrix_to_params_device(bws_.geom, count);
    }
    auto t_post_s2_cellkernel = Clock::now();
    // Single device->host copy of the final cell_params for result packing.
    Kokkos::deep_copy(
        Kokkos::subview(h_cell_params_, std::make_pair(0, count), Kokkos::ALL),
        Kokkos::subview(bws_.geom.cell_params, std::make_pair(0, count), Kokkos::ALL));
    auto t_post_s2_hostcopy = Clock::now();

    auto& h_E2 = r2.final_energies;
    auto& h_rt2 = r2.final_translations;
    auto& h_rr2 = r2.final_rotations;

    // Rotate stored rot_vectors into the canonical cell frame; see
    // canonicalize_cell_frame.
    if (bws_.dofs.n_cell_dof > 0) {
        auto h_cell = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), bws_.geom.cell);
        canonicalize_cell_frame(h_cell_params_, h_rt2, h_rr2, h_cell, count, csp_mol_.n_indep);
    }

    auto desc = compute_batch_descriptors();
    std::vector<std::vector<double>> pdds;
    auto results = pack_results(STAGE_MULTIPOLE, count, batch_seeds, n_mols,
                                 h_E2, h_rt2, h_rr2, desc, pdds, r2.diagnostics, r1.diagnostics);
    last_batch_pdds_ = std::move(pdds);

    // Optional PET-MAD final stage: re-relax each candidate under the ML
    // potential and append stage-3 results (PET energy + relaxed cell).
    double stage3_ms = 0;
    int pet_n_struct = 0;
    pet_last_evals_ = 0;
    if (pet_relax_ && pet_eval_) {
        auto t_pet0 = Clock::now();
        pet_stage_inputs(results, last_batch_pdds_);
        pet_flush_results(results);
        stage3_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_pet0).count();
        pet_n_struct = count;
    }

    auto t_end = Clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
    // Force-field time over force-field rows only; stage-3 rows carry their own.
    int n_primary = 0;
    for (const auto& r : results) if (r.stage != STAGE_FINAL) ++n_primary;
    const double per_ms = n_primary > 0 ? (total_ms - stage3_ms) / n_primary : 0.0;
    for (auto& r : results) if (r.stage != STAGE_FINAL) r.time_ms = per_ms;

    // Dump the phase breakdown when the total exceeds 1.5x the sum of the
    // measured stages.
    using msd_diag = std::chrono::duration<double, std::milli>;
    double sum_stages = msd_diag(t_gen1 - t_gen0).count()
                      + msd_diag(t_setup1 - t_setup0).count()
                      + msd_diag(t_s1_1 - t_s1_0).count()
                      + msd_diag(t_s2_1 - t_s2_0).count()
                      + desc.amd_ms + desc.powder_ms;
    if (total_ms > 5000.0 && total_ms > 1.5 * sum_stages) {
        using msd = std::chrono::duration<double, std::milli>;
        logging::warn("[rank {} SG {:>3d} batch={:>4d} seed={:>8d}] SLOW total={:.1f}s | "
            "gen={:.1f} setup={:.1f} pre_s1={:.1f} s1={:.1f} s1_sync={:.1f} "
            "s2_setup={:.1f} s2={:.1f} s2_cellk={:.1f} s2_hostcpy={:.1f} "
            "amd={:.1f} pow={:.1f}\n",
            rank_, space_group_, count, next_seed_, total_ms / 1000.0,
            msd(t_gen1 - t_gen0).count(),
            msd(t_setup1 - t_setup0).count(),
            msd(t_s1_0 - t_post_setup).count(),
            msd(t_s1_1 - t_s1_0).count(),
            msd(t_post_s1_sync - t_s1_1).count(),
            msd(t_post_s2_setup - t_post_s1_sync).count(),
            msd(t_s2_1 - t_s2_0).count(),
            msd(t_post_s2_cellkernel - t_s2_1).count(),
            msd(t_post_s2_hostcopy - t_post_s2_cellkernel).count(),
            desc.amd_ms, desc.powder_ms);
    }

    if (perf_) {
        using ms = std::chrono::duration<double, std::milli>;
        auto& kt = r2.timings;
        PerfRecorder::BatchRecord rec{space_group_, next_seed_, count, count, count,
                      static_cast<int>(results.size()),
                      ms(t_gen1 - t_gen0).count(), ms(t_setup1 - t_setup0).count(),
                      ms(t_s1_1 - t_s1_0).count(), ms(t_s2_1 - t_s2_0).count(),
                      desc.amd_ms, desc.powder_ms,
                      0, total_ms,
                      kt.expand_ms, kt.pairs_ms, kt.recip_ms, kt.post_ms};
        rec.stage3_ms = stage3_ms;
        rec.pet_n_struct = pet_n_struct;
        rec.pet_atoms = pet_atoms_;  // real atoms; bws_.geom.n_atoms is the padded stride
        rec.pet_evals = pet_last_evals_;
        perf_->record(rec);
    }

    // Merge any pre-optimisation (stage-0) results captured earlier in this
    // batch. They share seeds with the stage-2 results but have their own
    // (seed, sg, stage) PK so they don't collide.
    if (!extra_stage_results_pending_.empty()) {
        results.insert(results.end(),
                       std::make_move_iterator(extra_stage_results_pending_.begin()),
                       std::make_move_iterator(extra_stage_results_pending_.end()));
        last_batch_pdds_.insert(last_batch_pdds_.end(),
                       std::make_move_iterator(stage0_pdds_pending_.begin()),
                       std::make_move_iterator(stage0_pdds_pending_.end()));
        extra_stage_results_pending_.clear();
        stage0_pdds_pending_.clear();
    }

    return results;
}

// ---------------------------------------------------------------------------
// Streaming (pipelined) stage-2 path. Builds a pool of stage-1-relaxed
// structures (P = stream_pool_mult_ * batch_size_), then runs
// batch_optimize_stream_tr, which harvests converged slots and refills them
// from the pool. Descriptors and packing run in finalize_stream.
// ---------------------------------------------------------------------------
std::vector<StructureResult> SGWorker::run_batch_stream(int batch) {
    using Clock = std::chrono::steady_clock;
    using msd = std::chrono::duration<double, std::milli>;
    auto t0 = Clock::now();
    unit_t0_ = t0;
    const int n_mols = csp_mol_.n_indep;

    // Lazily (re)allocate the pools at full capacity. raw_pool_ holds generated
    // structures awaiting a streamed stage 1; pool_ holds stage-1-relaxed ones
    // awaiting stage 2. Both have gen_cap_ of headroom above P, since a fill can
    // overshoot P_use by a generated block.
    const int P = std::max(batch_size_, stream_pool_mult_ * batch_size_);
    const int pool_cap = P + gen_cap_;
    if (!pool_allocated_ || pool_.P != pool_cap || pool_.n_indep != n_mols) {
        pool_ = allocate_stream_pool(pool_cap, n_mols);
        pool_allocated_ = true;
    }
    if (stream_stage1_ && (raw_pool_.P != pool_cap || raw_pool_.n_indep != n_mols))
        raw_pool_ = allocate_stream_pool(pool_cap, n_mols);
    pool_.reset();
    raw_pool_.reset();

    // Build only what this worker still needs (remaining_ <= 0: the full pool).
    // The final generate is tapered so pool.filled lands at P_use.
    int P_use = sched::stream_pool_target(remaining_, P);

    BatchOptimizeSettings run_settings = *bsettings_;
    run_settings.hooks.abort = abort_flag_;
    run_settings.store_diagnostics = store_diagnostics_;

    // The same stage list as the lockstep driver. Streaming implies a multipole
    // stage, so it has two entries.
    const auto plan = stage_plan();
    const StageSpec& s1_spec = plan.front();
    const StageSpec& s2_spec = plan.back();

    double gen_ms = 0, s1_ms = 0;
    int empty_streak = 0;

    // --- build the pool: generate chunks, relaxing each through stage 1 ---
    // Streamed stage 1 parks raw structures in raw_pool_ for the slot
    // scheduler; lockstep relaxes each chunk in place and appends it.
    StreamPool& build_pool = stream_stage1_ ? raw_pool_ : pool_;
    while (build_pool.filled < P_use && empty_streak < 3) {
        std::vector<int> batch_seeds;
        auto tg0 = Clock::now();
        // Taper the final chunk so filled lands at P_use exactly.
        int want = std::min(batch, P_use - build_pool.filled);
        int count = generate_one(want, batch_seeds);
        gen_ms += msd(Clock::now() - tg0).count();
        if (count == 0) { ++empty_streak; continue; }
        empty_streak = 0;

        batch_setup_cells(bws_.tables, bws_.geom);

        // Stage 0: rigid press collision resolution (if enabled).
        if (collision_method_ == CollisionMethod::RIGID_PRESS) run_rigid_press(count);

        if (stream_stage1_) {
            // Park the raw structures; stage 1 runs over the whole pool below. No
            // Hessian is carried: the stream probes its own.
            stream_pool_append(raw_pool_, bws_, count, batch_seeds, {},
                               /*with_hessian=*/false);
            continue;
        }

        set_stage_strain_dofs(s1_spec.fix_angles);

        // Stage 1: charges only (TR); builds the molecular Hessian carried into
        // the pool for the stage-2 warm start.
        auto ts0 = Clock::now();
        apply_stage(s1_spec, run_settings);
        auto r1s = optimize_at_rank(s1_spec.max_rank, bws_, run_settings);
        s1_ms += msd(Clock::now() - ts0).count();

        // Widen strain mask for stage 2 + reset strain reference to the relaxed cell.
        reanchor_strain(count, s2_spec.fix_angles);

        stream_pool_append(pool_, bws_, count, batch_seeds, r1s.diagnostics);
    }

    last_batch_structures_ = 0;

    // --- stream stage 1 over the raw pool, harvesting into the stage-2 pool ---
    if (stream_stage1_ && raw_pool_.filled > 0) {
        // Use all gen_cap_ slots; empty slots are masked out of every kernel.
        bws_.geom.n_batch = wide_slots_ ? gen_cap_ : batch_size_;
        set_stage_strain_dofs(s1_spec.fix_angles);
        apply_stage(s1_spec, run_settings);  // warm_hessian false: nothing upstream
        run_settings.refill_interval = refill_interval_;
        auto ts0 = Clock::now();
        stream_at_rank(s1_spec.max_rank, bws_, raw_pool_, run_settings, &pool_);
        s1_ms += msd(Clock::now() - ts0).count();

        // Widen the mask for stage 2; the refill re-anchors cell_ref and zeroes
        // strain per slot.
        set_stage_strain_dofs(s2_spec.fix_angles);
    }

    if (pool_.filled == 0) return {};

    // --- stream stage 2 over the pool ---
    // Use all gen_cap_ slots; empty slots are masked out of every kernel.
        bws_.geom.n_batch = wide_slots_ ? gen_cap_ : batch_size_;
    apply_stage(s2_spec, run_settings);
    run_settings.refill_interval = refill_interval_;

    auto ts2_0 = Clock::now();
    StreamResult sr;
    if (sampler_ == SamplerKind::METROPOLIS) {
        // A chain must propose inside the generator's prior, which only the
        // quasi-random generator reports.
        if (!gen_prior_valid_)
            throw std::runtime_error(
                "--sampler mc needs the sampling prior the quasi-random generator "
                "reports; this run generated structures another way");
        McSettings mc;
        mc.kT = mc_kT_;
        mc.n_moves = mc_moves_;
        mc.step_trans = mc_step_trans_;
        mc.step_rot = mc_step_rot_;
        mc.step_strain = mc_step_strain_;
        mc.min_contact = mc_min_contact_;
        mc.prior = gen_prior_;
        sr = stream_mc_at_rank(s2_spec.max_rank, bws_, pool_, run_settings, mc);
    } else {
        sr = stream_at_rank(s2_spec.max_rank, bws_, pool_, run_settings, nullptr);
    }
    double s2_ms = msd(Clock::now() - ts2_0).count();

    // --- post-pass: descriptors + result packing in batch_size chunks ---
    // finalize_stream sets last_batch_structures_.
    auto results = finalize_stream(sr, n_mols);

    double total_ms = msd(Clock::now() - t0).count();
    double per_ms = results.empty() ? 0 : total_ms / results.size();
    for (auto& r : results) r.time_ms = per_ms;
    if (perf_) {
        PerfRecorder::BatchRecord rec{space_group_, next_seed_, pool_.filled, pool_.filled,
                      pool_.filled, static_cast<int>(results.size()),
                      gen_ms, 0, s1_ms, s2_ms, 0, 0, 0, total_ms, 0, 0, 0, 0};
        rec.stage3_ms = pet_stage_ms_;
        rec.pet_n_struct = pet_n_struct_;
        rec.pet_atoms = pet_atoms_;
        rec.pet_evals = pet_last_evals_;
        perf_->record(rec);
    }
    return results;
}

// Pack a StreamResult into StructureResults. The harvested geometry is reloaded
// into bws_ in batch_size chunks so the existing descriptor + packing kernels
// (which operate over bws_ slots [0, n_batch)) can be reused unchanged.
std::vector<StructureResult> SGWorker::finalize_stream(StreamResult& sr, int n_mols) {
    using P2 = std::pair<int, int>;
    std::vector<StructureResult> all;
    int n_structures = 0;  // structures packed, whether or not they were emitted
    last_batch_pdds_.clear();
    const int N = sr.n;
    const int ni = csp_mol_.n_indep;
    const bool diag_valid = sr.diagnostics.valid();

    for (int base = 0; base < N; base += batch_size_) {
        const int c = std::min(batch_size_, N - base);

        // Build host chunk views (0-indexed) from the harvested rows.
        Kokkos::View<RealType*, Kokkos::HostSpace> h_E("st_E", c);
        Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace> h_t("st_t", c, ni, 3);
        Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace> h_r("st_r", c, ni, 3);
        Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace> h_cell("st_cell", c, 3, 3);
        std::vector<int> seeds(c);
        BatchDiagnostics diag, s1diag;
        const bool s1_valid = sr.s1_diagnostics.valid();
        if (s1_valid) {
            s1diag.grad_max = Kokkos::View<RealType*, Kokkos::HostSpace>("st_s1g", c);
            s1diag.grad_rms = s1diag.grad_max;
            s1diag.iters_to_converge = Kokkos::View<int*, Kokkos::HostSpace>("st_s1i", c);
        }
        if (diag_valid) {
            diag.grad_max = Kokkos::View<RealType*, Kokkos::HostSpace>("st_gm", c);
            diag.grad_rms = Kokkos::View<RealType*, Kokkos::HostSpace>("st_gr", c);
            diag.iters_to_converge = Kokkos::View<int*, Kokkos::HostSpace>("st_it", c);
            if (sr.diagnostics.has_tr()) {
                diag.final_delta = Kokkos::View<RealType*, Kokkos::HostSpace>("st_dl", c);
                diag.restarts = Kokkos::View<int*, Kokkos::HostSpace>("st_rs", c);
            }
        }
        for (int j = 0; j < c; ++j) {
            const int o = base + j;
            h_E(j) = sr.energies(o);
            seeds[j] = sr.seeds[o];
            for (int k = 0; k < ni; ++k)
                for (int d = 0; d < 3; ++d) {
                    h_t(j,k,d) = sr.translations(o,k,d);
                    h_r(j,k,d) = sr.rotations(o,k,d);
                }
            for (int rr = 0; rr < 3; ++rr)
                for (int cc = 0; cc < 3; ++cc) h_cell(j,rr,cc) = sr.cell(o,rr,cc);
            if (diag_valid) {
                diag.grad_max(j) = sr.diagnostics.grad_max(o);
                diag.grad_rms(j) = sr.diagnostics.grad_rms(o);
                diag.iters_to_converge(j) = sr.diagnostics.iters_to_converge(o);
                if (diag.has_tr()) {
                    diag.final_delta(j) = sr.diagnostics.final_delta(o);
                    diag.restarts(j) = sr.diagnostics.restarts(o);
                }
            }
            if (s1_valid) {
                s1diag.grad_max(j) = sr.s1_diagnostics.grad_max(o);
                s1diag.iters_to_converge(j) = sr.s1_diagnostics.iters_to_converge(o);
            }
        }

        // Rebuild a self-consistent geometry for the descriptors: cell_params from
        // the harvested cell, rotations canonicalised, then batch_setup_cells to
        // refresh cell_inv, volume, kappa, max_hkl and max_shift, which the
        // descriptor kernels read. Descriptors are frame-invariant.
        bws_.geom.n_batch = c;
        Kokkos::deep_copy(Kokkos::subview(bws_.geom.cell, P2(0,c), Kokkos::ALL, Kokkos::ALL), h_cell);
        cell_matrix_to_params_device(bws_.geom, c);
        Kokkos::deep_copy(
            Kokkos::subview(h_cell_params_, P2(0,c), Kokkos::ALL),
            Kokkos::subview(bws_.geom.cell_params, P2(0,c), Kokkos::ALL));
        if (bws_.dofs.n_cell_dof > 0)
            canonicalize_cell_frame(h_cell_params_, h_t, h_r, h_cell, c, ni);

        Kokkos::deep_copy(Kokkos::subview(bws_.dofs.translations, P2(0,c), Kokkos::ALL, Kokkos::ALL), h_t);
        Kokkos::deep_copy(Kokkos::subview(bws_.dofs.rot_vectors,  P2(0,c), Kokkos::ALL, Kokkos::ALL), h_r);
        batch_setup_cells(bws_.tables, bws_.geom);   // canonical cell + cell_inv/volume/kappa/max_hkl/max_shift
        switch (effective_max_rank_) {
            case 1: batch_expand_asymmetric_unit<1, ExecSpace>(bws_.tables, bws_.geom, bws_.dofs); break;
            case 2: batch_expand_asymmetric_unit<2, ExecSpace>(bws_.tables, bws_.geom, bws_.dofs); break;
            case 3: batch_expand_asymmetric_unit<3, ExecSpace>(bws_.tables, bws_.geom, bws_.dofs); break;
            default: batch_expand_asymmetric_unit<4, ExecSpace>(bws_.tables, bws_.geom, bws_.dofs); break;
        }

        auto desc = compute_batch_descriptors();
        std::vector<std::vector<double>> pdds;
        auto chunk = pack_results(STAGE_MULTIPOLE, c, seeds, n_mols, h_E, h_t, h_r, desc, pdds,
                                  diag, s1diag);
        // Extract the PET inputs now (the next chunk overwrites the workspace) and
        // relax them once after the loop; see pet_stage_inputs.
        if (pet_relax_ && pet_eval_) pet_stage_inputs(chunk, pdds);
        n_structures += static_cast<int>(chunk.size());
        // Per-structure wall time as the running average over the unit so far:
        // emitted rows leave before the unit's total is known.
        {
            const double unit_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - unit_t0_).count();
            const double per = n_structures > 0 ? unit_ms / n_structures : 0.0;
            for (auto& r : chunk) r.time_ms = per;
        }
        if (emit_) {
            // Emit now; the PET staging above took its own copy.
            emit_(chunk, pdds, {});
        } else {
            all.insert(all.end(), std::make_move_iterator(chunk.begin()),
                       std::make_move_iterator(chunk.end()));
            last_batch_pdds_.insert(last_batch_pdds_.end(),
                       std::make_move_iterator(pdds.begin()),
                       std::make_move_iterator(pdds.end()));
        }
    }
    bws_.geom.n_batch = batch_size_;  // restore (chunks left it at the last chunk size)
    // Structures produced, not rows: stage-3 rows are extra rows for the same
    // structures.
    last_batch_structures_ = n_structures;
    if (pet_relax_ && pet_eval_) pet_flush_results(all);
    return all;
}

#ifdef KLASP_HAS_PET
namespace {
// Stand-in for a missing PDD.
const std::vector<double> kNoPdd;
}  // namespace

void SGWorker::pet_stage_inputs(const std::vector<StructureResult>& rows,
                                const std::vector<std::vector<double>>& pdds) {
    const int count = static_cast<int>(rows.size());
    if (count <= 0) return;
    // Real atoms only: padded slots sit at the body origin and must not reach
    // PET (see klasp/padding.hpp).
    const int N = static_cast<int>(real_atoms_.size());

    // Atomic numbers per atom (full-cell order, shared across the batch).
    std::vector<int> Z(N);
    for (int i = 0; i < N; ++i)
        Z[i] = element(atom_elements_[real_atoms_[i]]).atomic_number;

    // Final geometry is read by batch slot (last_pack_slots_), not packed index.
    const int n_slots = bws_.geom.n_batch;
    auto h_pos = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),
        Kokkos::subview(bws_.geom.positions, std::make_pair(0, n_slots), Kokkos::ALL, Kokkos::ALL));
    auto h_cell = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),
        Kokkos::subview(bws_.geom.cell, std::make_pair(0, n_slots), Kokkos::ALL, Kokkos::ALL));

    // Shared symmetry (one space group for the whole batch).
    PetSymmetry sym;
    {
        auto h_sr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bws_.tables.sym_rotations);
        auto h_st = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bws_.tables.sym_translations);
        auto h_si = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bws_.tables.sym_indep_index);
        sym.n_indep = bws_.tables.n_indep;
        // Real atoms of each component; n_sites_per_mol is the padded stride.
        sym.sites_of_indep.resize(bws_.tables.n_indep);
        for (int k = 0; k < bws_.tables.n_indep; ++k)
            sym.sites_of_indep[k] = (k < (int)basis_mol_n_sites_.size())
                                  ? basis_mol_n_sites_[k] : bws_.tables.n_sites_per_mol;
        sym.n_op = bws_.tables.n_symops;
        sym.indep_of_op.resize(sym.n_op);
        sym.rot.resize(static_cast<std::size_t>(sym.n_op) * 9);
        sym.trans.resize(static_cast<std::size_t>(sym.n_op) * 3);
        for (int op = 0; op < sym.n_op; ++op) {
            sym.indep_of_op[op] = h_si(op);
            for (int e = 0; e < 9; ++e) sym.rot[op * 9 + e] = h_sr(op, e);
            for (int e = 0; e < 3; ++e) sym.trans[op * 3 + e] = h_st(op, e);
        }
    }
    const bool use_sym = pet_symmetry_ && sym.valid();
    if (pet_symmetry_ && !use_sym && !pet_sym_warned_) {
        // Relaxing freely may break the group the space_group column records.
        logging::warn("[stage3 pet] SG{}: symmetry operators unusable; relaxing freely "
                      "(the stored space group will not be preserved)\n",
                      space_group_);
        pet_sym_warned_ = true;
    }
    pet_pending_sym_ = sym;
    pet_pending_use_sym_ = use_sym;

    for (int r = 0; r < count; ++r) {
        // One relaxation per basin, checked before the geometry is assembled.
        // seen_basin() rather than classify(): a deeper member of a covered basin
        // is not re-relaxed; `klasp rerank --top-up` settles representatives later.
        if (pet_dedup_enabled_) {
            const std::vector<double>& pdd =
                r < static_cast<int>(pdds.size()) ? pdds[r] : kNoPdd;
            if (pet_dedup_.seen_basin(rows[r], pdd.empty() ? nullptr : pdd.data(),
                                      static_cast<int>(pdd.size()))) {
                ++pet_skipped_this_unit_;
                continue;
            }
        }
        const int slot = (r < static_cast<int>(last_pack_slots_.size())) ? last_pack_slots_[r] : r;
        CrystalGeometry s;
        s.atomic_numbers = Z;
        s.positions.resize(N);
        for (int i = 0; i < N; ++i)
            for (int d = 0; d < 3; ++d) s.positions[i][d] = h_pos(slot, real_atoms_[i], d);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) s.cell[i][j] = h_cell(slot, i, j);
        // Screen out sub-Angstrom contacts, which an ML model cannot relax out.
        if (pet_min_contact_ > 0.0) {
            const double mc = min_contact_bohr(s);
            if (mc < pet_min_contact_ * ANGSTROM_TO_BOHR) {
                ++pet_clashed_;
                if (mc < pet_worst_clash_) { pet_worst_clash_ = mc; pet_worst_seed_ = rows[r].seed; }
                continue;
            }
        }
        pet_pending_structs_.push_back(std::move(s));
        pet_pending_rows_.push_back(rows[r]);
        // This row's stage-2 PDD, captured now for its stage-3 row at flush time.
        pet_pending_pdds_.push_back(r < static_cast<int>(pdds.size()) ? pdds[r]
                                                                     : std::vector<double>{});
    }
}

void SGWorker::pet_flush_results(std::vector<StructureResult>& results) {
    const int count = static_cast<int>(pet_pending_structs_.size());
    // Both counts must reach rank 0 on every exit path, or the space group
    // never reads as finished.
    pet_skipped_ += pet_skipped_this_unit_;
    if (count <= 0) {
        // Explain an empty stage 3.
        if (pet_clashed_ > 0) {
            logging::warn("[stage3 pet] SG{}: all {} candidates rejected by the clash screen "
                          "(< {:.2f} A; worst seed{} at {:.3f} A). No stage-3 rows.\n",
                          space_group_, pet_clashed_, pet_min_contact_, pet_worst_seed_,
                          pet_worst_clash_ * BOHR_TO_ANGSTROM);
            pet_dropped_ += pet_clashed_;
            pet_clashed_ = 0; pet_worst_clash_ = 1e300; pet_worst_seed_ = -1;
        } else if (pet_skipped_this_unit_ > 0) {
            logging::info("[stage3 pet] SG{}: all {} candidates were already-seen basins. "
                          "No stage-3 rows.\n",
                          space_group_, pet_skipped_this_unit_);
        }
        pet_skipped_this_unit_ = 0;
        return;
    }
    std::vector<CrystalGeometry> structs = std::move(pet_pending_structs_);
    std::vector<StructureResult> rows = std::move(pet_pending_rows_);
    std::vector<std::vector<double>> row_pdds = std::move(pet_pending_pdds_);
    pet_pending_structs_.clear();
    pet_pending_rows_.clear();
    pet_pending_pdds_.clear();
    const PetSymmetry& sym = pet_pending_sym_;
    const bool use_sym = pet_pending_use_sym_;
    const int N = static_cast<int>(structs[0].atomic_numbers.size());

    PetRelaxOptions opts;
    opts.verbose = false;
    // KLASP_PET_NODEVICE forces the host path.
    opts.device = pet_device_ && (std::getenv("KLASP_PET_NODEVICE") == nullptr);
    // For the symmetric path the stretch model is built over the full expanded
    // cell and folded to the asym DOFs via the symop Jacobian (J = M0_op).
    opts.precondition = pet_precondition_;
    if (pet_max_iter_ > 0) opts.max_iter_full = pet_max_iter_;
    if (pet_force_tol_ > 0) opts.force_tol = pet_force_tol_;
    if (pet_stress_tol_ > 0) opts.stress_tol = pet_stress_tol_;
    // Stage 3 is relaxed in chunks sized to fit GPU memory; structures are
    // independent, so this is exact.
    const auto t_pet0 = std::chrono::steady_clock::now();
    std::vector<PetRelaxResult> pet(count);
    pet_last_evals_ = 0;

    // --pet-pass: carry unconverged structures forward between short passes and
    // top up from unstarted ones, keeping the batch full. Off by default: it
    // does not save time, and each pass boundary restarts L-BFGS, so very short
    // passes lose.
    const int kPass = pet_pass_rounds_;
    const int total_cap = opts.max_iter_full;

    std::vector<CrystalGeometry> geom = structs;   // current geometry per structure
    std::vector<int> spent(count, 0);              // iterations already used
    std::vector<int> live;                         // in the batch right now
    int next = 0, done = 0;
    while (done < count) {
        // Asked per pass: the budget is learned from the previous batch.
        const int cap_atoms = pet_eval_->recommended_batch_atoms();
        const int width = std::max(1, cap_atoms / std::max(1, N));
        while (static_cast<int>(live.size()) < width && next < count) live.push_back(next++);
        if (live.empty()) break;

        std::vector<CrystalGeometry> cs;
        std::vector<PetSymmetry> csym;
        cs.reserve(live.size());
        for (int idx : live) cs.push_back(geom[idx]);
        if (use_sym) csym.assign(live.size(), sym);

        PetRelaxOptions pass_opts = opts;
        if (kPass > 0) pass_opts.max_iter_full = kPass;
        auto pr = use_sym ? pet_relax_sym_batch(*pet_eval_, cs, csym, pass_opts)
                          : pet_relax_batch(*pet_eval_, cs, pass_opts);

        std::vector<int> still;
        for (std::size_t k = 0; k < live.size(); ++k) {
            const int idx = live[k];
            spent[idx] += pr[k].full_iterations;
            pet_last_evals_ = std::max(pet_last_evals_, spent[idx]);
            pet[idx] = pr[k];
            pet[idx].full_iterations = spent[idx];   // report the total, not the pass
            // Carry on only if the pass cap stopped it short of converging and it
            // has budget left; anything else is finished.
            if (kPass > 0 && !pr[k].converged && spent[idx] < total_cap) {
                geom[idx] = pr[k].structure;
                still.push_back(idx);
            } else {
                ++done;
            }
        }
        live.swap(still);
        // Progress from inside the work unit; the orchestrator's watchdog is
        // blocked for the duration of run_batch.
        if (done < count || count > width)
            logging::info("[stage3 pet] SG{}: {}/{} relaxed ({:.0f}%)\n",
                          space_group_, done, count,
                          100.0 * done / std::max(1, count));
    }

    // Emit one stage-3 result per candidate: PET LATTICE energy + relaxed cell,
    // carrying the stage-2 descriptors. Lattice energy = E_crystal (Hartree, per
    // cell) -> kJ/mol per formula unit, minus the isolated-molecule reference.
    const int z = std::max(1, Z_);
    // Explicit PET geometry for the flexible_structure table; the row's
    // rigid-body DOFs are stage 2's and cannot represent a flexible relaxation.
    last_batch_flex_.clear();
    std::vector<StructureResult> extra;
    std::vector<std::vector<double>> extra_pdd;
    extra.reserve(count);
    extra_pdd.reserve(count);
    // Report the stage-3 convergence; grad_max >= 0 sets the DB's have_diag.
    int n_conv = 0;
    double gmax_lo = 1e300, gmax_hi = 0.0;
    for (int b = 0; b < count; ++b) {
        n_conv += pet[b].converged ? 1 : 0;
        gmax_lo = std::min(gmax_lo, pet[b].max_force);
        gmax_hi = std::max(gmax_hi, pet[b].max_force);
    }
    // Descriptors from the relaxed geometry, via flex_descriptors (the rigid
    // workspace cannot hold it).
    std::vector<CrystalGeometry> flex_geoms;
    std::vector<int> flex_of_slot(count, -1);
    flex_geoms.reserve(count);
    for (int b = 0; b < count; ++b) {
        const double e = pet[b].energy * AU_TO_KJ_PER_MOL / z - pet_reference_;
        // See the row loop below for why `converged` is part of this test.
        if (!std::isfinite(e) || (pet[b].full_iterations == 0 && !pet[b].converged))
            continue;
        flex_of_slot[b] = static_cast<int>(flex_geoms.size());
        flex_geoms.push_back(pet[b].structure);
    }
    FlexDescriptors flex_desc;
    if (!flex_geoms.empty()) {
        // The PDD folded to one row per asymmetric-unit atom, so it has the
        // shape stage 2 stores and the dedup compares.
        flex_desc = compute_flex_descriptors(flex_geoms, /*want_pdd=*/store_pdd_,
                                             /*cutoff_bohr=*/0.0, AMD_K,
                                             /*fold_rows=*/pdd_rows_);
    }

    int n_dropped = 0;
    for (int b = 0; b < count; ++b) {
        const double e_lat = pet[b].energy * AU_TO_KJ_PER_MOL / z - pet_reference_;
        // Drop non-finite energies and structures never relaxed. full_iterations
        // == 0 also means already converged on entry, so `converged` separates them.
        if (!std::isfinite(e_lat) || (pet[b].full_iterations == 0 && !pet[b].converged)) {
            ++n_dropped;
            continue;
        }
        StructureResult r3 = rows[b];  // stage-2 metadata (seed, sg, n_mols)
        r3.stage = STAGE_FINAL;
        r3.energy = e_lat;
        // Replace the inherited descriptors with the relaxed structure's own.
        const int fi = flex_of_slot[b];
        if (fi >= 0 && fi < static_cast<int>(flex_desc.amd.size())) {
            std::copy_n(flex_desc.amd[fi].data(), AMD_K, r3.amd);
            std::copy_n(flex_desc.powder[fi].data(), POWDER_N_BINS, r3.powder);
        }
        matrix_to_cell_params(pet[b].structure.cell, r3.cell_params);
        r3.grad_max = static_cast<float>(pet[b].max_force);   // Ha/Bohr, final (>=0 => have_diag)
        r3.grad_rms = static_cast<float>(pet[b].max_stress);  // Ha/Bohr^3, final cell gradient
        r3.iters_to_converge = pet[b].converged ? pet[b].full_iterations : -1;
        // Density follows the relaxed cell (mass is conserved, so rho ~ 1/V).
        {
            const double v_old = cell_volume_from_params(rows[b].cell_params);
            const double v_new = cell_volume_from_params(r3.cell_params);
            if (v_new > 0 && v_old > 0) r3.density = rows[b].density * (v_old / v_new);
        }
        // Re-classify against the relaxed cell and density. The vacuum bit is set
        // by rank 0 from the explicit geometry (result_consumer.cpp).
        r3.warnings = classify_warnings(r3.cell_params, r3.density);
        extra.push_back(r3);
        // The relaxed structure's own PDD; stage 2's only when it has none.
        if (fi >= 0 && fi < static_cast<int>(flex_desc.pdd.size()))
            extra_pdd.push_back(flex_desc.pdd[fi]);
        else
            extra_pdd.push_back(b < (int)row_pdds.size() ? row_pdds[b] : std::vector<double>{});
        // Geometry for rank 0's flexible_structure row; to_flexible handles the
        // units. The atom list is the compacted one.
        last_batch_flex_[r3.seed] = to_flexible(pet[b].structure);
    }
    std::string extra_note;
    if (n_dropped) extra_note += fmt::format(", {} dropped (non-finite or not relaxed)", n_dropped);
    // What the gate saved, reported next to what it cost: `count` is what was
    // relaxed, and this is what would have been relaxed without it.
    if (pet_skipped_this_unit_)
        extra_note += fmt::format(", {} skipped (already-seen basin)", pet_skipped_this_unit_);
    if (pet_clashed_)
        extra_note += fmt::format(", {} clashed (< {:.2f} A, worst seed{} at {:.3f} A)",
                                  pet_clashed_, pet_min_contact_, pet_worst_seed_,
                                  pet_worst_clash_ * BOHR_TO_ANGSTROM);
    logging::info("[stage3 pet] SG{}: {}/{} converged, max|F| {:.2e}..{:.2e} Ha/Bohr, "
                  "rounds<={}{}\n",
                  space_group_, n_conv, count, gmax_lo, gmax_hi, pet_last_evals_,
                  extra_note);
    // Both kinds of loss, so rank 0 can tell this space group's ML stage is
    // finished.
    pet_dropped_ += pet_clashed_ + n_dropped;
    pet_clashed_ = 0; pet_worst_clash_ = 1e300; pet_worst_seed_ = -1;
    pet_skipped_this_unit_ = 0;
    const double pet_batch_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pet0).count();
    // Share this batch's PET time over the stage-3 rows.
    if (!extra.empty()) {
        const double per_pet_ms = pet_batch_ms / static_cast<double>(extra.size());
        for (auto& r : extra) r.time_ms = per_pet_ms;
    }
    pet_stage_ms_ += pet_batch_ms;
    pet_n_struct_ += count;
    // Real atoms, not the padded stride.
    pet_atoms_ = static_cast<int>(real_atoms_.size());
    if (emit_) {
        emit_(extra, extra_pdd, last_batch_flex_);
    } else {
        results.insert(results.end(), extra.begin(), extra.end());
        last_batch_pdds_.insert(last_batch_pdds_.end(), extra_pdd.begin(), extra_pdd.end());
    }
}
#else
void SGWorker::pet_stage_inputs(const std::vector<StructureResult>&,
                                const std::vector<std::vector<double>>&) {}
void SGWorker::pet_flush_results(std::vector<StructureResult>&) {}
#endif

}  // namespace klasp
