#include "klasp/log.hpp"
#include "klasp/cmd_rerank.hpp"

#include "klasp/cell.hpp"
#include "klasp/constants.hpp"
#include "klasp/database.hpp"
#include "klasp/db_tool.hpp"
#include "klasp/dedup.hpp"
#include "klasp/element.hpp"
#include "klasp/io.hpp"
#include "klasp/mat3.hpp"
#include "klasp/pet_bridge.hpp"
#include "klasp/pet_relax.hpp"
#include "klasp/sgdata.hpp"
#include "klasp/warnings.hpp"
#include "klasp/types.hpp"

#include <filesystem>
#include <fmt/base.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace klasp {

static double cell_volume_bohr(const double cell[3][3]) {
    return std::fabs(mat3_det(cell));
}

// Post-CSP PET re-ranking: take the global top-K by force-field energy, relax under
// PET (symmetry-constrained when possible), rank by PET lattice energy, and write
// stage-3 `structure` rows plus the relaxed geometry in `flexible_structure`.

static void report_final_stage(const FinalStageDedup& d) {
    logging::info("rerank: stage 3 deduplicated by its relaxed geometry: {} unique of {}.\n",
                  d.unique, d.rows);
}

RerankResult cmd_rerank(const RerankOptions& o) {
    RerankResult result;
    const std::string& db_path = o.db_path;
    const std::string& basis_path = o.basis_path;
    const int target = o.top;
    const bool symmetry = o.symmetry;
    const double force_tol = o.force_tol;
    const double stress_tol = o.stress_tol;
    const int max_iter = o.max_iter;
    const int from_stage = o.from_stage;
    const double coarse_factor = o.coarse_factor;
    const double min_contact_ang = o.min_contact;
    const bool top_up = o.top_up;

    // An explicit model pair wins over the named registry, as in `energy` and `csp`.
    std::string mj = o.model_json, ms = o.model_weights;
    if (mj.empty() && ms.empty()) resolve_pet_model(o.model, mj, ms);
    PetEvaluator eval(mj, ms);
    BasisConfig basis = load_csp_basis(basis_path, db_path);

    auto rows = read_landscape(db_path, from_stage, /*filter_sg=*/0, /*max_energy=*/1e30,
                               /*unique_only=*/true, /*exclude_catastrophes=*/true,
                               /*catastrophe_threshold=*/-1000.0,
                               /*exclude_warned=*/true,
                               /*only_missing_stage=*/top_up ? STAGE_FINAL : 0);
    if (rows.empty()) {
        if (top_up) {
            logging::info("rerank --top-up: every unique structure in {} already has a "
                       "stage-3 energy. Nothing to relax.\n", db_path);
            // Nothing to relax (e.g. --top-up on a finished landscape) is not a failure; the
            // stage is still deduplicated.
            report_final_stage(dedup_final_stage(db_path));
            result.ok = true;
            return result;
        }
        logging::info("rerank: no structures found in {} (stage {}).\n", db_path,
                   from_stage > 0 ? std::to_string(from_stage) : "max");
        return result;
    }
    const int K = (target > 0) ? std::min(target, (int)rows.size()) : (int)rows.size();
    rows.resize(K);
    if (top_up)
        logging::info("rerank --top-up: {} unique structure{} with no stage-3 energy, "
                   "model '{}'.\n",
                   K, K == 1 ? "" : "s", o.model.empty() ? "pet-mad-xs" : o.model);
    else
        logging::info("rerank: {} candidates (top-{} of the landscape), model '{}'.\n",
                   K, K, o.model.empty() ? "pet-mad-xs" : o.model);

    const double reference = pet_isolated_reference(eval, basis);
    logging::info("Isolated-molecule reference = {:.4f} kJ/mol/fu.\n", reference);

    // Reconstruct explicit geometry (Bohr) and symmetry for every candidate.
    std::vector<CrystalGeometry> structs(K);
    std::vector<PetSymmetry> syms(K);
    std::vector<int> zfu(K), sgv(K);
    const bool want_sym = symmetry;
    bool sym_ok = want_sym;
    for (int i = 0; i < K; ++i) {
        const auto& r = rows[i];
        structs[i] = expand_rigid_structure(basis, r.space_group, r.cell_params,
                                            r.translation, r.rotation, r.n_mols);
        const int N = structs[i].n_atoms();
        const int nop = get_space_group(r.space_group).n_symops;
        sgv[i] = r.space_group;
        // Formula units per cell is n_symops, not n_symops * Z': the isolated reference
        // (pet_isolated_reference) is one molecule of each component, i.e. one formula
        // unit. Matches SGWorker.
        zfu[i] = nop;
        // Symmetry for the constrained relaxation, any Z'. Operators are crystal ops
        // crossed with independent molecules, flattened as op*Z' + m to match
        // expand_rigid_structure. sites_of_indep is the real per-component atom count.
        if (want_sym) {
            const int nm = std::max(1, r.n_mols);
            // For Z'=1 derive the site count from the atom count.
            std::vector<int> sites = (nm == 1)
                ? std::vector<int>{N / nop}
                : std::vector<int>(basis.mol_n_sites.begin(), basis.mol_n_sites.end());
            syms[i] = expanded_row_symmetry(r.space_group, nm, sites);
            // n_full() == N checks the layout agrees with the expansion.
            if (!syms[i].valid() || syms[i].n_full() != N) sym_ok = false;
        }
    }

    // Reject sub-Angstrom clashes before PET: a weak-force model cannot relax them out,
    // and they rank artificially deep in FF energy.
    if (min_contact_ang > 0.0) {
        const double min_contact_b = min_contact_ang * ANGSTROM_TO_BOHR;
        int kept = 0, dropped = 0;
        double worst_drop = 1e300; int worst_seed = -1, worst_sg = 0;
        for (int i = 0; i < K; ++i) {
            const double mc = min_contact_bohr(structs[i]);
            if (mc < min_contact_b) {
                ++dropped;
                if (mc < worst_drop) { worst_drop = mc; worst_seed = rows[i].seed; worst_sg = sgv[i]; }
                continue;
            }
            if (kept != i) {  // compact the parallel arrays in place
                structs[kept] = std::move(structs[i]);
                syms[kept] = std::move(syms[i]);
                zfu[kept] = zfu[i]; sgv[kept] = sgv[i]; rows[kept] = rows[i];
            }
            ++kept;
        }
        if (dropped > 0) {
            structs.resize(kept); syms.resize(kept); zfu.resize(kept);
            sgv.resize(kept); rows.resize(kept);
            logging::info("Clash filter (< {:.2f} A): dropped {} FF catastrophe(s), kept {}. "
                       "Worst: sg{} seed{} contact {:.3f} A.\n",
                       min_contact_ang, dropped, kept, worst_sg, worst_seed,
                       worst_drop * BOHR_TO_ANGSTROM);
        }
        if (kept == 0) {
            logging::info("rerank: all {} candidates were clashed; nothing to relax.\n", K);
            result.ok = true;
            result.rejected = dropped;
            return result;
        }
        result.rejected = dropped;
    }
    const int Kk = (int)structs.size();  // candidates surviving the clash filter

    if (want_sym && !sym_ok)
        logging::info("  (a candidate's symmetry could not be described; relaxing freely, "
                   "not symmetry-constrained)\n");
    const bool use_sym = want_sym && sym_ok;

    PetRelaxOptions opts;
    opts.device = (o.device && std::getenv("KLASP_PET_NODEVICE") == nullptr);
    opts.force_tol = force_tol;
    opts.stress_tol = stress_tol;
    opts.max_iter_full = max_iter;

    // Relax in chunks that fit the device.
    auto relax_all = [&](const PetRelaxOptions& o) {
        return pet_relax_chunked(eval, structs, use_sym ? syms : std::vector<PetSymmetry>{}, o);
    };

    // Optional coarse->fine: a loose-tolerance pass over all K, then re-rank and relax
    // only the top fraction to the tight tolerance.
    if (coarse_factor > 1.0) {
        PetRelaxOptions co = opts;
        co.force_tol = force_tol * coarse_factor;
        co.stress_tol = stress_tol * coarse_factor;
        co.max_iter_full = std::max(30, max_iter / 2);
        logging::info("Coarse pass (tol x{:.0f}) over {} candidates...\n", coarse_factor, Kk);
        auto coarse = relax_all(co);
        // Take the coarse-relaxed structures, re-rank, keep the top half.
        std::vector<int> order(Kk);
        for (int i = 0; i < Kk; ++i) order[i] = i;
        auto latt = [&](int i) { return coarse[i].energy * AU_TO_KJ_PER_MOL / zfu[i] - reference; };
        std::sort(order.begin(), order.end(), [&](int a, int b) { return latt(a) < latt(b); });
        const int Kf = std::max(1, Kk / 2);
        std::vector<CrystalGeometry> ns(Kf);
        std::vector<PetSymmetry> nsy(Kf);
        std::vector<int> nz(Kf), nsg(Kf);
        std::vector<StructureResult> nrows(Kf);
        for (int i = 0; i < Kf; ++i) {
            const int o = order[i];
            ns[i] = coarse[o].structure;   // continue from the coarse geometry
            nsy[i] = syms[o]; nz[i] = zfu[o]; nsg[i] = sgv[o]; nrows[i] = rows[o];
        }
        structs = std::move(ns); syms = std::move(nsy); zfu = std::move(nz);
        sgv = std::move(nsg); rows = std::move(nrows);
        logging::info("Fine pass over the top {} coarse candidates...\n", Kf);
    }

    const int Kf = (int)structs.size();
    auto pet = relax_all(opts);

    std::vector<int> order(Kf);
    for (int i = 0; i < Kf; ++i) order[i] = i;
    auto latt = [&](int i) { return pet[i].energy * AU_TO_KJ_PER_MOL / zfu[i] - reference; };
    std::sort(order.begin(), order.end(), [&](int a, int b) { return latt(a) < latt(b); });

    CSPDatabase db;
    db.open(db_path);
    db.set_meta("stage3_model", !o.model_json.empty()
        ? std::filesystem::path(o.model_json).stem().string()
        : (o.model.empty() ? std::string("xs") : o.model));
    logging::info("\n  rank   sg   seed   E_lattice(kJ/mol/fu)   max|F|(Ha/Bohr)  conv\n");
    int n_conv = 0;
    for (int rank = 0; rank < Kf; ++rank) {
        const int i = order[rank];
        const auto& pr = pet[i];
        const double E = latt(i);
        n_conv += pr.converged ? 1 : 0;
        if (rank < 25)
            logging::info("  {:4d} {:4d} {:6d}   {:18.4f}   {:14.3e}   {}\n",
                       rank, sgv[i], rows[i].seed, E, pr.max_force, pr.converged ? "yes" : "no");

        // Relaxed cell params and density (mass conserved: rho ~ 1/V).
        const double Vnew = cell_volume_bohr(pr.structure.cell);
        double cp_ref[3][3];
        {  // parent cell volume from stored params
            RealType cp_rt[6];
            for (int k = 0; k < 6; ++k) cp_rt[k] = static_cast<RealType>(rows[i].cell_params[k]);
            RealType cm[3][3]; cell_params_to_matrix(cp_rt, cm);
            for (int p = 0; p < 3; ++p) for (int q = 0; q < 3; ++q) cp_ref[p][q] = cm[p][q];
        }
        const double Vold = cell_volume_bohr(cp_ref);
        const double density = rows[i].density * (Vnew > 0 ? Vold / Vnew : 1.0);

        double newcp[6];
        matrix_to_cell_params(pr.structure.cell, newcp);
        // Stage-3 ranking row: PET lattice energy and relaxed cell; rigid DOFs are the
        // parent's placeholders (the real geometry is in flexible_structure).
        StructureResult r3 = rows[i];
        r3.stage = STAGE_FINAL;
        r3.energy = E;
        r3.density = density;
        for (int k = 0; k < 6; ++k) r3.cell_params[k] = newcp[k];
        r3.grad_max = static_cast<float>(pr.max_force);
        r3.grad_rms = static_cast<float>(pr.max_stress);
        r3.iters_to_converge = pr.converged ? pr.full_iterations : -1;
        // Re-classify warnings against the relaxed cell.
        r3.warnings = classify_warnings(r3.cell_params, r3.density);
        db.insert_result(&r3, /*unique=*/true);

        db.insert_flexible(rows[i].seed, sgv[i], STAGE_FINAL, to_flexible(pr.structure));
    }
    db.flush();
    db.close();
    logging::info("\nrerank: {}/{} converged. Wrote stage-3 rows + flexible geometry to {}.\n",
               n_conv, Kf, db_path);
    report_final_stage(dedup_final_stage(db_path));
    result.ok = true;
    result.candidates = Kf;
    result.relaxed = n_conv;
    return result;
}

}  // namespace klasp
