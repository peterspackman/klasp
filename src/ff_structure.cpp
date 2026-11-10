#include "klasp/ff_structure.hpp"

#include "klasp/batch.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/cell.hpp"
#include "klasp/constants.hpp"
#include "klasp/element.hpp"
#include "klasp/io.hpp"
#include "klasp/mat3.hpp"
#include "klasp/molecule.hpp"
#include "klasp/padding.hpp"
#include "klasp/sgdata.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>

namespace klasp {

using json = nlohmann::json;

namespace {

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point from, Clock::time_point to = Clock::now()) {
    return std::chrono::duration<double>(to - from).count();
}

// A loaded document plus what results need to describe its atoms.
struct Loaded {
    Workspace ws;
    MoleculeData mol{};
    CrystalConfig config;
    int max_rank = 0;
    RealType mp_cutoff = 0;           // rank>=2 multipole cutoff, Bohr; 0 = the pair cutoff
    bool mp_single = false;
    int n_molecules = 1;
    std::vector<int> real_atoms;      // flat indices into one structure's atom arrays
    std::vector<int> atomic_numbers;  // one per real atom
};

// A site's element: its `element`, or the leading letters of its `type`
// ("C_W3" is carbon, "Cl01" chlorine).
std::string site_element(const json& site) {
    const std::string name = site.value("element", site.value("type", std::string("X")));
    if (name.size() > 1 && std::islower(static_cast<unsigned char>(name[1])))
        return name.substr(0, 2);
    return name.substr(0, 1);
}

// `doc` must already be normalised: the atom list is read off its molecules.
Loaded load(const json& doc, const FFModelOptions& o) {
    // Size for every atom the document places: one copy per symmetry entry or
    // molecule, at the largest molecule type's stride.
    size_t stride_sites = 0;
    for (const auto& t : doc.at("basis").at("molecule_types"))
        stride_sites = std::max(stride_sites, t.at("sites").size());
    const bool listed = doc.contains("symmetry") && !doc.at("symmetry").empty();
    const size_t copies = listed ? doc.at("symmetry").size()
                                 : doc.at("crystal").at("molecules").size();
    Loaded l;
    l.ws = allocate_workspace(std::max(512, static_cast<int>(copies * stride_sites)),
                              27 * 27 * 27, 50000, 32, 4);
    l.config = load_structure_json(doc, l.mol, l.ws);
    if (o.ewald_accuracy >= 0) apply_ewald_accuracy_override(o.ewald_accuracy, l.ws);
    if (o.no_taper) { l.config.taper_on = 0.0; l.config.taper_off = 0.0; }
    l.max_rank = o.max_order >= 0 ? o.max_order : l.config.max_rank;
    if (l.max_rank > 4)
        throw std::invalid_argument(fmt::format("max_order is at most 4; got {}", l.max_rank));
    l.mp_cutoff = o.mp_cutoff_ang > 0
        ? static_cast<RealType>(o.mp_cutoff_ang * ANGSTROM_TO_BOHR) : RealType(0);
    l.mp_single = o.mp_single;
    // Molecules per cell, the divisor of a per-molecule lattice energy.
    l.n_molecules = std::max(1, l.mol.n_symops);

    const int stride = l.mol.n_sites_per_mol;
    l.real_atoms = real_atom_indices(stride,
                                     host_site_counts(l.mol.mol_site_count, l.mol.n_symops));

    std::map<std::string, const json*> type_by_name;
    for (const auto& t : doc.at("basis").at("molecule_types"))
        type_by_name[t.at("name").get<std::string>()] = &t;
    const json& molecules = doc.at("crystal").at("molecules");
    const auto indep = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                           l.mol.sym_indep_index);
    l.atomic_numbers.reserve(l.real_atoms.size());
    for (int a : l.real_atoms) {
        const std::string type = molecules.at(indep(a / stride)).at("type").get<std::string>();
        const json& site = type_by_name.at(type)->at("sites").at(a % stride);
        l.atomic_numbers.push_back(element(site_element(site)).atomic_number);
    }
    return l;
}

double energy_of(const Loaded& l, uint32_t terms = EWALD_ALL) {
    const RealType e = batch_evaluate_energy(l.mol, l.ws, l.max_rank, l.config.taper_on,
                                             l.config.taper_off, l.config.buck_damping, terms,
                                             l.mp_cutoff, l.mp_single);
    return double(e) * AU_TO_KJ_PER_MOL / l.n_molecules;
}

FFEnergyComponents components_of(const Loaded& l) {
    FFEnergyComponents c;
    c.total = energy_of(l);
    c.buckingham = energy_of(l, EWALD_BUCKINGHAM);
    c.electrostatic = c.total - c.buckingham;
    return c;
}

FFStructure describe(const Loaded& l) {
    FFStructure s;
    s.space_group = l.config.space_group;
    s.n_molecules = l.n_molecules;
    s.n_atoms = static_cast<int>(l.real_atoms.size());
    s.atomic_numbers = l.atomic_numbers;
    return s;
}

// `at(r, c)` is the cell matrix in Bohr.
template <typename CellAt>
std::array<double, 9> cell_in_angstrom(const CellAt& at) {
    std::array<double, 9> out{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) out[r * 3 + c] = double(at(r, c)) * BOHR_TO_ANGSTROM;
    return out;
}

// `at(atom, d)` is a flat atom array in Bohr; only the real atoms are kept.
template <typename PosAt>
std::vector<double> positions_in_angstrom(const std::vector<int>& atoms, const PosAt& at) {
    std::vector<double> out;
    out.reserve(3 * atoms.size());
    for (int a : atoms)
        for (int d = 0; d < 3; ++d) out.push_back(double(at(a, d)) * BOHR_TO_ANGSTROM);
    return out;
}

std::array<double, 6> params_of(const std::array<double, 9>& cell) {
    double m[3][3], p[6];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) m[r][c] = cell[r * 3 + c];
    matrix_to_cell_params(m, p);
    return {p[0], p[1], p[2], p[3], p[4], p[5]};
}

json document_of(const Loaded& l, const json& doc, double energy) {
    BasisConfig basis;
    basis.basis_json = l.config.basis_json;
    json out = build_occ_json(basis, l.mol, energy, l.config.molecule_type_index);
    for (const char* key : {"title", "seed"})
        if (doc.contains(key)) out[key] = doc[key];
    return out;
}

// Rigid-body state and cell into every slot: slot 0 exactly as given, the rest
// jittered; then set up the cells.
void set_starts(BatchWorkspace& bws, const Loaded& l, int n_batch) {
    const int ni = l.mol.n_indep;
    auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    const auto h_mt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), l.mol.translations);
    const auto h_mr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), l.mol.rot_vectors);
    std::mt19937 rng(42);
    std::normal_distribution<double> jitter(0.0, 0.02);
    for (int b = 0; b < n_batch; ++b)
        for (int k = 0; k < ni; ++k)
            for (int d = 0; d < 3; ++d) {
                h_bt(b, k, d) = h_mt(k, d) + (b ? jitter(rng) : 0.0);
                h_br(b, k, d) = h_mr(k, d) + (b ? jitter(rng) * 0.5 : 0.0);
            }
    Kokkos::deep_copy(bws.dofs.translations, h_bt);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);

    auto h_cp = Kokkos::create_mirror_view(bws.geom.cell_params);
    for (int b = 0; b < n_batch; ++b)
        for (int i = 0; i < 6; ++i) h_cp(b, i) = l.mol.cell_params[i];
    Kokkos::deep_copy(bws.geom.cell_params, h_cp);
    batch_setup_cells(bws.tables, bws.geom);
}

}  // namespace

std::map<std::string, double> timings_by_name(const FFTimings& t) {
    return {{"total", t.total},           {"load", t.load},
            {"initial_energy", t.initial_energy}, {"setup", t.setup},
            {"optimize", t.optimize},     {"per_iteration", t.per_iteration},
            {"pairs", t.pairs},           {"reciprocal", t.reciprocal},
            {"post", t.post},             {"expand", t.expand},
            {"step", t.step},             {"final_structure", t.final_structure},
            {"trajectory", t.trajectory}};
}

json normalize_structure_json(const json& doc) {
    if (!doc.is_object())
        throw std::invalid_argument("a structure document is a JSON object");
    if (doc.contains("crystal")) {
        if (!doc.contains("basis"))
            throw std::invalid_argument(
                "a structure document with a \"crystal\" needs its \"basis\" beside it");
        return doc;
    }
    if (doc.contains("basis"))
        throw std::invalid_argument(
            "this is a CSP basis with no crystal; a structure also needs \"crystal\": "
            "{\"cell\": ..., \"molecules\": ...}");
    if (!doc.contains("cell") || !doc.contains("molecules") || !doc.contains("molecule_types"))
        throw std::invalid_argument(
            "a structure document needs a basis and a crystal, or cell, molecules and "
            "molecule_types at its top level");
    json out;
    out["basis"]["molecule_types"] = doc["molecule_types"];
    for (const char* key : {"potentials", "settings"})
        if (doc.contains(key)) out["basis"][key] = doc[key];
    out["crystal"]["cell"] = doc["cell"];
    out["crystal"]["molecules"] = doc["molecules"];
    for (const char* key : {"symmetry", "space_group", "reference", "title", "seed"})
        if (doc.contains(key)) out[key] = doc[key];
    return out;
}

FFStructure ff_evaluate(const json& doc_in, const FFModelOptions& opts) {
    const auto start = Clock::now();
    const json doc = normalize_structure_json(doc_in);
    const Loaded l = load(doc, opts);
    const auto loaded = Clock::now();
    FFStructure s = describe(l);
    s.components = components_of(l);
    s.initial_energy = s.energy = s.components.total;
    const auto evaluated = Clock::now();

    // Positions from the rigid-body state via the batch kernels, not the loader's
    // placement (which ignores rotation matrices and symmetry copies).
    auto bws = allocate_batch_workspace(1, l.ws, l.mol);
    set_starts(bws, l, 1);
    batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
    Kokkos::fence();
    auto h_cell = Kokkos::create_mirror(Kokkos::HostSpace(), bws.geom.cell);
    auto h_pos = Kokkos::create_mirror(Kokkos::HostSpace(), bws.geom.positions);
    Kokkos::deep_copy(h_cell, bws.geom.cell);
    Kokkos::deep_copy(h_pos, bws.geom.positions);
    s.cell = cell_in_angstrom([&](int r, int c) { return h_cell(0, r, c); });
    s.cell_params = params_of(s.cell);
    s.positions = positions_in_angstrom(l.real_atoms, [&](int a, int d) { return h_pos(0, a, d); });
    s.structure = document_of(l, doc, s.energy);
    s.timings.load = seconds(start, loaded);
    s.timings.initial_energy = seconds(loaded, evaluated);
    s.timings.final_structure = seconds(evaluated);
    s.timings.total = seconds(start);
    return s;
}

std::vector<FFStructure> ff_relax(const json& doc_in, const FFRelaxOptions& o) {
    if (o.iterations < 1)
        throw std::invalid_argument(
            fmt::format("a relaxation needs at least one iteration; got {}", o.iterations));
    if (o.grad_tol < 0)
        throw std::invalid_argument("grad_tol cannot be negative (0 runs every iteration)");
    if (o.perturb < 0)
        throw std::invalid_argument("perturb cannot be negative");

    const auto start = Clock::now();
    const json doc = normalize_structure_json(doc_in);
    Loaded l = load(doc, o.model);
    const auto loaded = Clock::now();
    const double e0 = energy_of(l);
    const auto initial = Clock::now();

    const int n_batch = 1 + o.perturb;
    const int ni = l.mol.n_indep;
    auto bws = allocate_batch_workspace(n_batch, l.ws, l.mol);

    set_starts(bws, l, n_batch);

    if (!o.fixed_cell) {
        int lattice = TRICLINIC;
        if (l.config.space_group > 0)
            lattice = static_cast<int>(get_space_group(l.config.space_group).lattice);
        bws.dofs.d_g_strain = SView2D<RealType, ExecSpace>("d_g_strain", n_batch, 6);
        bws.dofs.strain     = SView2D<RealType, ExecSpace>("strain", n_batch, 6);
        bws.dofs.cell_ref   = SView3D<RealType, ExecSpace>("cell_ref", n_batch, 3, 3);
        batch_set_strain_dofs(bws.geom, bws.dofs, lattice);
        Kokkos::deep_copy(bws.dofs.cell_ref, bws.geom.cell);
        Kokkos::deep_copy(bws.dofs.strain, RealType(0));
        Kokkos::deep_copy(bws.dofs.d_g_strain, RealType(0));
    }

    BatchOptimizeSettings bs;
    bs.n_iterations = o.iterations;
    bs.grad_tol = o.grad_tol;
    bs.model.taper_on = l.config.taper_on;
    bs.model.taper_off = l.config.taper_off;
    bs.model.buck_damping = l.config.buck_damping;
    bs.model.mp_cutoff = l.mp_cutoff;
    bs.model.mp_single = l.mp_single;
    bs.pressure_gpa = o.pressure_gpa;
    bs.store_diagnostics = true;
    bs.record_trajectory = o.trajectory;

    const auto optimize_start = Clock::now();
    BatchOptimizeResult res;
    switch (l.max_rank) {
        case 0:  res = batch_optimize_asymunit<0>(bws, bs); break;
        case 1:  res = batch_optimize_asymunit<1>(bws, bs); break;
        case 2:  res = batch_optimize_asymunit<2>(bws, bs); break;
        case 3:  res = batch_optimize_asymunit<3>(bws, bs); break;
        default: res = batch_optimize_asymunit<4>(bws, bs); break;
    }
    Kokkos::fence();
    const auto optimized = Clock::now();

    // Snapshots, not mirror views: the trajectory replay rewrites the workspace, and
    // on a host build a mirror view aliases it.
    auto h_cell = Kokkos::create_mirror(Kokkos::HostSpace(), bws.geom.cell);
    auto h_pos = Kokkos::create_mirror(Kokkos::HostSpace(), bws.geom.positions);
    Kokkos::deep_copy(h_cell, bws.geom.cell);
    Kokkos::deep_copy(h_pos, bws.geom.positions);

    // Relaxed geometry as a document. Documents are read back in canonical cell
    // orientation, so the Cartesian molecular orientations are rotated to match.
    const auto relaxed_document = [&](int b, double energy) {
        RealType m[3][3], cp[6];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) m[r][c] = h_cell(b, r, c);
        matrix_to_cell_params(m, cp);
        std::vector<RealType> rv(3 * static_cast<size_t>(ni));
        for (int k = 0; k < ni; ++k)
            for (int d = 0; d < 3; ++d) rv[3 * k + d] = res.final_rotations(b, k, d);
        align_orientations_to_canonical_cell(m, cp, rv.data(), ni);

        for (int i = 0; i < 6; ++i) l.mol.cell_params[i] = cp[i];
        auto h_t = Kokkos::create_mirror_view(l.mol.translations);
        auto h_r = Kokkos::create_mirror_view(l.mol.rot_vectors);
        for (int k = 0; k < ni; ++k)
            for (int d = 0; d < 3; ++d) {
                h_t(k, d) = res.final_translations(b, k, d);
                h_r(k, d) = rv[3 * k + d];
            }
        Kokkos::deep_copy(l.mol.translations, h_t);
        Kokkos::deep_copy(l.mol.rot_vectors, h_r);
        return document_of(l, doc, energy);
    };

    std::vector<FFStructure> out;
    out.reserve(n_batch);
    for (int b = 0; b < n_batch; ++b) {
        FFStructure s = describe(l);
        s.label = b ? fmt::format("#{}", b) : std::string();
        s.relaxed = true;
        s.initial_energy = e0;
        s.energy = double(res.final_energies(b)) * AU_TO_KJ_PER_MOL / l.n_molecules;
        s.iterations = res.n_iterations;
        if (res.diagnostics.valid()) {
            s.grad_max = double(res.diagnostics.grad_max(b));
            const int converged_at = res.diagnostics.iters_to_converge(b);
            s.converged = converged_at >= 0;
            if (converged_at >= 0) s.iterations = converged_at;
            if (res.diagnostics.has_tr()) {
                s.trust_radius = double(res.diagnostics.final_delta(b)) * BOHR_TO_ANGSTROM;
                s.restarts = res.diagnostics.restarts(b);
            }
        }
        s.cell = cell_in_angstrom([&](int r, int c) { return h_cell(b, r, c); });
        s.cell_params = params_of(s.cell);
        s.positions = positions_in_angstrom(l.real_atoms, [&](int a, int d) { return h_pos(b, a, d); });
        const auto slot_start = Clock::now();
        if (std::isfinite(s.energy)) {
            s.structure = relaxed_document(b, s.energy);
            s.components = ff_evaluate(s.structure, o.model).components;
        }
        s.timings.final_structure = seconds(slot_start);
        out.push_back(std::move(s));
    }

    // Rebuild each trajectory frame's positions by expanding its stored state with
    // the optimiser's kernels. A rejected step repeats the accepted geometry.
    const auto replay_start = Clock::now();
    if (res.trajectory.valid()) {
        const TrustRegionTrajectory& tj = res.trajectory;
        const ForceField ff = make_force_field(bws.tables, bs.model);
        const bool with_strain = bws.dofs.n_cell_dof > 0 && tj.strain.extent(0) > 0;
        auto h_t = Kokkos::create_mirror_view(bws.dofs.translations);
        auto h_r = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
        auto h_s = Kokkos::create_mirror_view(bws.dofs.strain);
        auto f_cell = Kokkos::create_mirror(Kokkos::HostSpace(), bws.geom.cell);
        auto f_pos = Kokkos::create_mirror(Kokkos::HostSpace(), bws.geom.positions);
        for (auto& s : out) s.trajectory.reserve(tj.n_frames);

        for (int f = 0; f < tj.n_frames; ++f) {
            for (int b = 0; b < n_batch; ++b) {
                for (int k = 0; k < ni; ++k)
                    for (int d = 0; d < 3; ++d) {
                        h_t(b, k, d) = tj.translations(f, b, k, d);
                        h_r(b, k, d) = tj.rotations(f, b, k, d);
                    }
                if (with_strain)
                    for (int v = 0; v < 6; ++v) h_s(b, v) = tj.strain(f, b, v);
            }
            Kokkos::deep_copy(bws.dofs.translations, h_t);
            Kokkos::deep_copy(bws.dofs.rot_vectors, h_r);
            if (with_strain) {
                Kokkos::deep_copy(bws.dofs.strain, h_s);
                batch_update_cell_from_strain(ff, bws.geom, bws.dofs);
            }
            batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
            Kokkos::fence();
            Kokkos::deep_copy(f_cell, bws.geom.cell);
            Kokkos::deep_copy(f_pos, bws.geom.positions);

            for (int b = 0; b < n_batch; ++b) {
                FFFrame fr;
                fr.iteration = f;
                fr.energy = double(tj.energy(f, b)) * AU_TO_KJ_PER_MOL / l.n_molecules;
                fr.trial_energy = double(tj.trial_energy(f, b)) * AU_TO_KJ_PER_MOL / l.n_molecules;
                fr.accepted = f == 0 || tj.energy(f, b) == tj.trial_energy(f, b);
                fr.grad_max = double(tj.grad_max(f, b));
                fr.trust_radius = double(tj.trust_radius(f, b)) * BOHR_TO_ANGSTROM;
                fr.cell = cell_in_angstrom([&](int r, int c) { return f_cell(b, r, c); });
                fr.positions = positions_in_angstrom(
                    l.real_atoms, [&](int a, int d) { return f_pos(b, a, d); });
                out[b].trajectory.push_back(std::move(fr));
            }
        }
    }
    const double replay = seconds(replay_start);

    const double total = seconds(start);
    for (auto& s : out) {
        FFTimings& t = s.timings;
        t.total = total;
        t.load = seconds(start, loaded);
        t.initial_energy = seconds(loaded, initial);
        t.setup = seconds(initial, optimize_start);
        t.optimize = seconds(optimize_start, optimized);
        t.per_iteration = res.n_iterations > 0 ? t.optimize / res.n_iterations : 0.0;
        t.pairs = res.timings.pairs_ms / 1000.0;
        t.reciprocal = res.timings.recip_ms / 1000.0;
        t.post = res.timings.post_ms / 1000.0;
        t.expand = res.timings.expand_ms / 1000.0;
        t.step = std::max(0.0, t.optimize - t.pairs - t.reciprocal - t.post - t.expand);
        t.trajectory = replay;
    }
    return out;
}

namespace {

void write_xyz_frame(std::ostream& out, const std::vector<int>& atomic_numbers,
                     const std::array<double, 9>& cell, const std::vector<double>& positions,
                     const std::string& info) {
    // ASE's Lattice lists the lattice vectors in turn (the matrix columns).
    out << atomic_numbers.size() << "\n";
    out << fmt::format("Lattice=\"{:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\" "
                       "Properties=species:S:1:pos:R:3 pbc=\"T T T\" {}\n",
                       cell[0], cell[3], cell[6], cell[1], cell[4], cell[7],
                       cell[2], cell[5], cell[8], info);
    for (size_t a = 0; a < atomic_numbers.size(); ++a)
        out << fmt::format("{:<2s} {:15.8f} {:15.8f} {:15.8f}\n",
                           element_symbol(atomic_numbers[a]), positions[3 * a],
                           positions[3 * a + 1], positions[3 * a + 2]);
}

}  // namespace

void write_trajectory_xyz(std::ostream& out, const FFStructure& s) {
    for (const auto& f : s.trajectory)
        write_xyz_frame(out, s.atomic_numbers, f.cell, f.positions,
                        fmt::format("iteration={} energy_kj_per_mol={:.10f} "
                                    "trial_energy_kj_per_mol={:.10f} accepted={} "
                                    "grad_max={:.6e} trust_radius={:.6e}",
                                    f.iteration, f.energy, f.trial_energy,
                                    f.accepted ? "T" : "F", f.grad_max, f.trust_radius));
}

void write_structure_cif(std::ostream& out, const FFStructure& s, const std::string& name) {
    double m[3][3], inv[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) m[r][c] = s.cell[r * 3 + c];
    if (!mat3_inv(m, inv))
        throw std::runtime_error("cannot write a CIF for a cell with no volume");
    std::vector<double> frac(s.positions.size());
    for (int a = 0; a < s.n_atoms; ++a)
        for (int r = 0; r < 3; ++r)
            frac[3 * a + r] = inv[r][0] * s.positions[3 * a] + inv[r][1] * s.positions[3 * a + 1]
                            + inv[r][2] * s.positions[3 * a + 2];
    CspStructureId id;
    id.name = name;
    write_cif_explicit(out, s.cell_params.data(), s.n_atoms, s.atomic_numbers.data(),
                       frac.data(), s.space_group, s.energy, id);
}

}  // namespace klasp
