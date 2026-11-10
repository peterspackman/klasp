#include "klasp/pet_bridge.hpp"

#include "klasp/constants.hpp"
#include "klasp/device_relax.hpp"
#include "klasp/device_optim.hpp"
#include "klasp/mat3.hpp"
#include "klasp/pet_relax.hpp"

#include "pet/calculator.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace klasp {

namespace {
// klasp internal units are Bohr / Hartree; PET speaks Angstrom / eV.
constexpr double kHartreePerEv = EV_TO_AU;            // 1 eV  -> Hartree
constexpr double kAngstromPerBohr = BOHR_TO_ANGSTROM; // 1 Bohr -> Angstrom
// Force: eV/Angstrom -> Hartree/Bohr.
constexpr double kForceEvAngToHaBohr = kHartreePerEv * kAngstromPerBohr;

// Build a PET System (Angstrom, eV, rows = lattice vectors) from a klasp
// CrystalGeometry (Bohr, columns = lattice vectors), periodic in 3D.
pet::System to_pet_system(const CrystalGeometry& s) {
    pet::System sys;
    sys.n_atoms = static_cast<int>(s.atomic_numbers.size());
    sys.atomic_numbers = s.atomic_numbers;
    sys.positions.reserve(sys.n_atoms * 3);
    for (const auto& p : s.positions) {
        sys.positions.push_back(p[0] * kAngstromPerBohr);
        sys.positions.push_back(p[1] * kAngstromPerBohr);
        sys.positions.push_back(p[2] * kAngstromPerBohr);
    }
    // Transpose: PET row i (lattice vector i) = klasp column i.
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            sys.cell[i * 3 + j] = s.cell[j][i] * kAngstromPerBohr;
    sys.pbc = {s.periodic, s.periodic, s.periodic};

    // Wrap positions into the unit cell: pet-kokkos's host neighbour search
    // assumes atoms sit in the cell. Kept in double (not RealType) so a
    // single-precision build does not add noise to the PET geometry.
    double M[3][3], Minv[3][3] = {};
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) M[i][j] = sys.cell[i * 3 + j];
    const double det = mat3_det(M);
    if (s.periodic && std::fabs(det) > 1e-12) {  // skip degenerate/zero cell (isolated molecule)
        (void)mat3_inv(M, Minv);
        for (int a = 0; a < sys.n_atoms; ++a) {
            const double r0 = sys.positions[3*a], r1 = sys.positions[3*a+1], r2 = sys.positions[3*a+2];
            double f[3];
            for (int k = 0; k < 3; ++k) f[k] = r0*Minv[0][k] + r1*Minv[1][k] + r2*Minv[2][k];
            for (int k = 0; k < 3; ++k) f[k] -= std::floor(f[k]);  // wrap to [0,1)
            for (int j = 0; j < 3; ++j) sys.positions[3*a+j] = f[0]*M[0][j] + f[1]*M[1][j] + f[2]*M[2][j];
        }
    }
    return sys;
}

// klasp's run-time knobs, onto pet-kokkos's options.
pet::Options pet_options() {
    pet::Options o;
    // KLASP_PET_HOST_NEIGH=1: neighbour lists on the host, to check the device
    // build against a reference.
    if (const char* e = std::getenv("KLASP_PET_HOST_NEIGH"))
        o.device_neighbors = (e[0] == '0' || e[0] == '\0');
    // KLASP_PET_MEM_MB: size batches as if the device had this much memory.
    if (const char* e = std::getenv("KLASP_PET_MEM_MB"))
        if (const long mb = std::atol(e); mb > 0) o.memory_budget_bytes = std::size_t(mb) << 20;
    // KLASP_PET_MAXATOMS: a hard cap on atoms per batch, ahead of the estimate.
    if (const char* e = std::getenv("KLASP_PET_MAXATOMS")) o.max_batch_atoms = std::max(0, std::atoi(e));
    // KLASP_PET_TF32=1: TF32 tensor-core GEMMs (NVIDIA). Faster, changes results
    // by ~0.1 meV/atom.
    if (const char* e = std::getenv("KLASP_PET_TF32"); e && e[0] == '1') o.gemm = pet::GemmMode::TF32;
    return o;
}
}  // namespace

struct PetEvaluator::Impl {
    pet::Calculator calc;
    Impl(const std::string& json_path, const std::string& st_path)
        : calc(json_path, st_path, pet_options()) {}
};

// Directories searched for a named model, highest priority first. Weights are
// not shipped in the source tree or install image.
static std::vector<std::string> pet_model_search_dirs() {
    std::vector<std::string> dirs;
    auto add = [&dirs](std::string d) {
        if (!d.empty()) dirs.push_back(std::move(d));
    };

    // 1. KLASP_PET_MODEL_DIR: an explicit ':'-separated list.
    if (const char* e = std::getenv("KLASP_PET_MODEL_DIR")) {
        const std::string s(e);
        for (size_t p = 0; p <= s.size();) {
            const size_t q = s.find(':', p);
            add(s.substr(p, (q == std::string::npos ? s.size() : q) - p));
            if (q == std::string::npos) break;
            p = q + 1;
        }
    }
    // 2. The working directory.
    add("models");
    add(".");
    // 3. Per-user data directory.
    if (const char* x = std::getenv("XDG_DATA_HOME"); x && x[0])
        add(std::string(x) + "/klasp/models");
    else if (const char* h = std::getenv("HOME"); h && h[0])
        add(std::string(h) + "/.local/share/klasp/models");
    // 4. The install prefix's data directory (a `make install`ed klasp).
#ifdef KLASP_INSTALL_MODEL_DIR
    add(KLASP_INSTALL_MODEL_DIR);
#endif
    // 5. pet-kokkos's own: $PET_MODEL_DIR, its per-user directory, its source tree.
    for (auto& d : pet::model_search_dirs()) add(std::move(d));
    return dirs;
}

// A directory provides a model only if both the .json and the weights are there.
static bool model_dir_has(const std::string& dir, const std::string& name,
                          std::string& json_out, std::string& st_out) {
    const std::string j = dir + "/" + name + ".json";
    const std::string s = dir + "/" + name + ".safetensors";
    std::error_code ec;
    if (!std::filesystem::exists(j, ec) || !std::filesystem::exists(s, ec)) return false;
    json_out = j;
    st_out = s;
    return true;
}

static void find_named_model(const std::string& name, std::string& json_out,
                             std::string& st_out) {
    const auto dirs = pet_model_search_dirs();
    for (const auto& d : dirs)
        if (model_dir_has(d, name, json_out, st_out)) return;

    // Report every location tried.
    std::string msg = "PET model '" + name + "' not found. Model weights ship separately "
                      "from the source tree — set KLASP_PET_MODEL_DIR to a directory "
                      "containing " + name + ".json and " + name + ".safetensors.\nSearched:";
    for (const auto& d : dirs) msg += "\n  " + d;
    throw std::runtime_error(msg);
}

std::string PetEvaluator::default_model_json() {
    std::string j, s;
    find_named_model("pet-mad-xs", j, s);
    return j;
}
std::string PetEvaluator::default_model_safetensors() {
    std::string j, s;
    find_named_model("pet-mad-xs", j, s);
    return s;
}

void resolve_pet_model(const std::string& spec, std::string& json_out,
                       std::string& st_out) {
    // Named models (short aliases + canonical names), located via the search path.
    // "pbe0" and "pbe0-pet" are aliases for pet-mols-s.
    static const std::pair<const char*, const char*> kAliases[] = {
        {"xs", "pet-mad-xs"},   {"pet-mad-xs", "pet-mad-xs"},
        {"mols", "pet-mols-s"}, {"pet-mols-s", "pet-mols-s"}, {"pbe0", "pet-mols-s"},
        {"pbe0-pet", "pet-mols-s"},
    };
    if (spec.empty()) {
        find_named_model("pet-mad-xs", json_out, st_out);
        return;
    }
    for (const auto& [alias, name] : kAliases) {
        if (spec == alias) {
            find_named_model(name, json_out, st_out);
            return;
        }
    }
    // A path prefix whose files exist is taken as given.
    json_out = spec + ".json";
    st_out = spec + ".safetensors";
    std::error_code ec;
    if (std::filesystem::exists(json_out, ec) && std::filesystem::exists(st_out, ec)) return;
    // Any other bare name is looked up in the search path; a spec with a '/' is
    // used verbatim and checked when the model is loaded.
    if (spec.find('/') == std::string::npos) find_named_model(spec, json_out, st_out);
}

// The smaller of what device memory allows and where throughput stops
// improving, as pet-kokkos measures them; ask per chunk.
int PetEvaluator::recommended_batch_atoms() const {
    return impl_->calc.recommended_batch_atoms();
}

PetEvaluator::PetEvaluator()
    : PetEvaluator(default_model_json(), default_model_safetensors()) {}

PetEvaluator::PetEvaluator(const std::string& model_json,
                           const std::string& model_safetensors) {
    if (!Kokkos::is_initialized()) {
        throw std::runtime_error(
            "PetEvaluator: Kokkos must be initialized before constructing the "
            "PET model (klasp's main initializes Kokkos).");
    }
    impl_ = std::make_unique<Impl>(model_json, model_safetensors);
}

PetEvaluator::~PetEvaluator() = default;
PetEvaluator::PetEvaluator(PetEvaluator&&) noexcept = default;
PetEvaluator& PetEvaluator::operator=(PetEvaluator&&) noexcept = default;

PetResult PetEvaluator::evaluate(const CrystalGeometry& s, bool compute_forces) const {
    return evaluate(std::vector<CrystalGeometry>{s}, compute_forces);
}

// One structure goes to pet-kokkos's single-structure path, several to one
// batched device evaluation.
PetResult PetEvaluator::evaluate(const std::vector<CrystalGeometry>& structures,
                                 bool compute_forces) const {
    const int B = static_cast<int>(structures.size());
    std::vector<pet::System> systems;
    systems.reserve(B);
    for (const auto& s : structures) systems.push_back(to_pet_system(s));

    const pet::Results r = B == 1 ? impl_->calc.compute(systems[0], compute_forces)
                                  : impl_->calc.compute(systems, compute_forces);

    PetResult out;
    out.n_atoms = r.n_atoms;
    out.energy.resize(B);
    for (int b = 0; b < B; ++b) out.energy[b] = r.energy[b] * kHartreePerEv;
    if (!compute_forces) return out;

    out.forces.resize(r.forces.size());
    for (std::size_t i = 0; i < r.forces.size(); ++i) out.forces[i] = r.forces[i] * kForceEvAngToHaBohr;
    out.struct_id = r.struct_id;
    out.virial.resize(r.virial.size());
    for (std::size_t k = 0; k < r.virial.size(); ++k)
        out.virial[k] = r.virial[k] * kHartreePerEv * pet_virial_voigt_scale(static_cast<int>(k % 6));
    return out;
}

// Device-resident batched L-BFGS relaxation: the device analogue of
// run_lbfgs_batch + CellState (pet_relax.cpp), x = [p (3N), eps*cf (6)], with a
// batched-lockstep Armijo line search. Not bit-identical to the host path: it
// adds a model-Hessian preconditioner and a stall-recovery reset. Optimizer state
// is kept in host units (Bohr/Hartree), converting only at the network seam
// (gunit/f2hb/e2h), since the unit change rescales atomic and strain DOFs
// differently.

// Stage the batch in pet-kokkos, hand its geometry views to the shared relaxer,
// and evaluate whatever the relaxer has written there each round.
std::vector<PetRelaxResult> PetEvaluator::relax_batch_device(
    const std::vector<CrystalGeometry>& inits, const PetRelaxOptions& opts,
    const PetSymBatch* symb) const {
    const int B = static_cast<int>(inits.size());
    std::vector<PetRelaxResult> res(B);
    for (int b = 0; b < B; ++b) res[b].structure = inits[b];
    if (B < 1) return res;
    if (!opts.relax_cell)  // device path always carries the 6 cell-strain DOFs
        return pet_relax_batch(*this, inits, opts);

    pet::Calculator& calc = impl_->calc;
    std::vector<pet::System> systems;
    systems.reserve(B);
    for (const auto& s : inits) systems.push_back(to_pet_system(s));
    // Positions, cells and species are the relaxer's to write: it packs running
    // structures to the front each round and reports the order in `staged`, so
    // the neighbour cache follows. Charge and spin must be the same for every
    // structure.
    pet::DeviceGeom pg = calc.stage(systems);

    DeviceBackend be;
    be.geom = {pg.pos, pg.sid, pg.spec, pg.soff, pg.scnt, pg.sper, pg.scell, pg.staged, pg.Ntot, pg.B};
    be.h0 = devopt::RView2D("pet_h0", B, 9);
    be.spec_full = devopt::IView1D("pet_specf", pg.Ntot);
    Kokkos::deep_copy(be.spec_full, pg.spec);  // the staged order, before any packing
    be.gunit = kAngstromPerBohr;
    be.volunit = kAngstromPerBohr * kAngstromPerBohr * kAngstromPerBohr;
    be.f2hb = kForceEvAngToHaBohr; be.e2h = kHartreePerEv;
    auto hh0 = Kokkos::create_mirror_view(be.h0);
    for (int b = 0; b < B; ++b)  // PET frame: row i = lattice vector i (Angstrom)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) hh0(b, i * 3 + j) = inits[b].cell[j][i] * kAngstromPerBohr;
    Kokkos::deep_copy(be.h0, hh0);

    // Verlet-cache the neighbour topology across rounds (KLASP_PET_NOCACHE
    // disables). Only for this relaxation; evaluate() does not cache.
    struct CacheScope {
        pet::Options& o;
        bool was;
        CacheScope(pet::Options& opts, bool on) : o(opts), was(opts.cache_neighbors) { o.cache_neighbors = on; }
        ~CacheScope() { o.cache_neighbors = was; }
    } cache(calc.options(), std::getenv("KLASP_PET_NOCACHE") == nullptr);

    // PET fills the force-returning half of the backend seam; device_relax
    // applies the DOF policy's fold.
    be.force_fn = [&calc, pg](int Kc, int Mc) mutable -> BackendForces {
        pg.B = Kc;
        pg.Ntot = Mc;
        pet::BatchResult br = calc.compute_device(pg, /*compute_forces=*/true);
        // Symmetric -> full-tensor Voigt conversion (kPetVirialShearFactor), in
        // place; the buffer is regenerated every round.
        if (br.virial.extent(0) > 0) {
            auto W = br.virial;
            const int nb = static_cast<int>(W.extent(0));
            Kokkos::parallel_for(
                "virial_shear_to_strain", Kokkos::RangePolicy<pet::ExecSpace>(0, nb),
                KOKKOS_LAMBDA(int b) {
                    for (int v = 3; v < 6; ++v) W(b, v) *= kPetVirialShearFactor;
                });
        }
        return BackendForces{br.forces, br.virial, br.energy};
    };
    return device_relax(inits, opts, symb, be);
}

}  // namespace klasp
