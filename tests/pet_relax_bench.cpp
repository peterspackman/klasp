// Benchmark the batched PET relaxation to a force/stress tolerance and export
// the input and output geometries so an external implementation can be timed on
// identical work.
//
//   pet_relax_bench <model.json> <model.st> <landscape.cif> <atoms_per_mol>
//                   [N=32] [out_prefix]
//
// Reports wall time, batched rounds, total per-structure evaluations and the
// derived per-evaluation costs. With out_prefix, writes <prefix>_init.xyz and
// <prefix>_final.xyz (extxyz, Angstrom/eV).
//
// Env: KLASP_PET_MAXATOMS caps the atoms per sub-batch.
#include "klasp/cell.hpp"
#include "klasp/constants.hpp"
#include "klasp/element.hpp"
#include "klasp/pet_bridge.hpp"
#include "klasp/pet_relax.hpp"
#include "klasp/sgdata.hpp"

#include <Kokkos_Core.hpp>

#include "tool_cli.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace klasp;
using Clock = std::chrono::high_resolution_clock;

namespace {
struct Cand { CrystalGeometry s; int sg = 0, Z = 0; double ff = 0; };

const char* zsym(int z) {
    static const char* t[] = {"X","H","He","Li","Be","B","C","N","O","F","Ne",
        "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca"};
    return (z >= 0 && z <= 20) ? t[z] : "X";
}

// Landscape CIF written by `klasp extract`: one data_ block per structure,
// atoms listed in P1 but tagged with the space group they came from.
std::vector<Cand> parse_landscape(const char* path, int atoms_per_mol) {
    std::vector<Cand> all;
    std::ifstream f(path); std::string line;
    double cp[6] = {0,0,0,0,0,0}; int sg = 0; double ffe = 0;
    std::vector<std::array<double,3>> frac; std::vector<int> Zs;
    std::vector<std::string> loop_tags; bool in_loop = false, have = false;
    int ix=-1, iy=-1, iz=-1, itype=-1;
    auto flush = [&]() {
        if (!have || frac.empty()) return;
        Cand c; c.sg = sg; c.ff = ffe;
        RealType cpb[6] = {(RealType)(cp[0]*ANGSTROM_TO_BOHR),(RealType)(cp[1]*ANGSTROM_TO_BOHR),
                           (RealType)(cp[2]*ANGSTROM_TO_BOHR),(RealType)cp[3],(RealType)cp[4],(RealType)cp[5]};
        RealType M[3][3]; cell_params_to_matrix(cpb, M);
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) c.s.cell[i][j]=M[i][j];
        // The CIF lists the asymmetric unit; the space group expands it to P1.
        const SpaceGroupInfo& sgi = get_space_group(sg);
        for (int op=0; op<sgi.n_symops; ++op) {
            double R[3][3], t[3]; get_symop(sgi, op, R, t);
            for (std::size_t a=0;a<frac.size();++a) {
                double fi[3];
                for (int i=0;i<3;++i) fi[i]=R[i][0]*frac[a][0]+R[i][1]*frac[a][1]+R[i][2]*frac[a][2]+t[i];
                std::array<double,3> cart{0,0,0};
                for (int i=0;i<3;++i) for (int k=0;k<3;++k) cart[i]+=M[i][k]*fi[k];
                c.s.positions.push_back(cart); c.s.atomic_numbers.push_back(Zs[a]);
            }
        }
        c.Z = atoms_per_mol>0 ? (int)c.s.atomic_numbers.size()/atoms_per_mol : 1;
        if (c.Z>0) all.push_back(std::move(c));
    };
    while (std::getline(f, line)) {
        if (line.rfind("data_",0)==0) { flush(); for(double&x:cp)x=0; sg=0; ffe=0; frac.clear(); Zs.clear(); loop_tags.clear(); in_loop=false; have=true; continue; }
        if (!have) continue;
        if (line.rfind("loop_",0)==0) { in_loop=true; loop_tags.clear(); continue; }
        if (!line.empty() && line[0]=='_') {
            std::istringstream ss(line); std::string tag; ss>>tag;
            if (in_loop && tag.rfind("_atom_site_",0)==0) {
                if (tag=="_atom_site_type_symbol") itype=(int)loop_tags.size();
                else if (tag=="_atom_site_fract_x") ix=(int)loop_tags.size();
                else if (tag=="_atom_site_fract_y") iy=(int)loop_tags.size();
                else if (tag=="_atom_site_fract_z") iz=(int)loop_tags.size();
                loop_tags.push_back(tag); continue;
            }
            in_loop=false; double v;
            if (ss>>v) {
                if (tag=="_cell_length_a")cp[0]=v; else if(tag=="_cell_length_b")cp[1]=v;
                else if(tag=="_cell_length_c")cp[2]=v; else if(tag=="_cell_angle_alpha")cp[3]=v;
                else if(tag=="_cell_angle_beta")cp[4]=v; else if(tag=="_cell_angle_gamma")cp[5]=v;
                else if(tag=="_space_group_IT_number")sg=(int)v; else if(tag=="_klasp_energy_kj_per_mol")ffe=v;
            }
            continue;
        }
        if (in_loop && !loop_tags.empty() && ix>=0) {
            std::istringstream ss(line); std::vector<std::string> tok; std::string t;
            while (ss>>t) tok.push_back(t);
            if ((int)tok.size()>=(int)loop_tags.size()) {
                frac.push_back({std::atof(tok[ix].c_str()),std::atof(tok[iy].c_str()),std::atof(tok[iz].c_str())});
                Zs.push_back(element(itype>=0?tok[itype]:tok[1]).atomic_number);
            }
        }
    }
    flush();
    return all;
}

// One extxyz frame per structure, ASE conventions: Lattice rows are the lattice
// vectors (klasp stores them as columns), Angstrom, energy in eV.
void write_frames(const std::string& path, const std::vector<CrystalGeometry>& ss,
                  const std::vector<double>& e_hartree) {
    std::ofstream o(path);
    o.precision(10);
    for (std::size_t s = 0; s < ss.size(); ++s) {
        const CrystalGeometry& st = ss[s];
        const int na = (int)st.atomic_numbers.size();
        o << na << "\nLattice=\"";
        for (int k = 0; k < 3; ++k) for (int r = 0; r < 3; ++r)
            o << (st.cell[r][k]*BOHR_TO_ANGSTROM) << (k==2&&r==2?"":" ");
        o << "\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"";
        if (s < e_hartree.size()) o << " klasp_energy_eV=" << (e_hartree[s] / EV_TO_AU);
        o << " index=" << s << "\n";
        for (int i = 0; i < na; ++i)
            o << zsym(st.atomic_numbers[i]) << " "
              << st.positions[i][0]*BOHR_TO_ANGSTROM << " "
              << st.positions[i][1]*BOHR_TO_ANGSTROM << " "
              << st.positions[i][2]*BOHR_TO_ANGSTROM << "\n";
    }
}
}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Time a whole batched PET relaxation to tolerance, and optionally export the geometries.", "klasp_pet_relax_bench"};
    klasp::tools::tool_init(app);
    std::string model_json, model_weights, landscape_cif, prefix;
    int atoms_per_mol = 0, N = 32;
    app.add_option("model.json", model_json, "PET model architecture JSON")->required();
    app.add_option("model.st", model_weights, "PET model weights")->required();
    app.add_option("landscape.cif", landscape_cif, "Landscape to draw structures from")->required();
    app.add_option("atoms_per_mol", atoms_per_mol, "Atoms per molecule in the landscape")->required();
    app.add_option("N", N, "Structures to relax")->capture_default_str();
    app.add_option("out_prefix", prefix,
                   "Write <prefix>_init.xyz and <prefix>_final.xyz (extxyz, Angstrom/eV)");
    KLASP_TOOL_PARSE(app, argc, argv);

    Kokkos::ScopeGuard guard(argc, argv);

    auto all = parse_landscape(landscape_cif.c_str(), atoms_per_mol);
    // Top N by force-field energy, the same selection a CSP stage 3 makes.
    std::sort(all.begin(), all.end(), [](const Cand&a,const Cand&b){ return a.ff<b.ff; });
    if ((int)all.size() > N) all.resize(N);
    const int Nn = (int)all.size();
    if (Nn < 2) { std::fprintf(stderr, "need >= 2 structures (the device relaxer's batch minimum)\n"); return 2; }

    std::vector<CrystalGeometry> batch; long tot_atoms = 0;
    for (auto& c : all) { batch.push_back(c.s); tot_atoms += (long)c.s.atomic_numbers.size(); }

    PetEvaluator eval(model_json, model_weights);

    PetRelaxOptions opts;
    opts.device = true;
    opts.relax_cell = true;
    if (const char* it = std::getenv("KLASP_BENCH_ITERS")) opts.max_iter_full = std::atoi(it);
    if (const char* ft = std::getenv("KLASP_BENCH_FTOL"))  opts.force_tol = std::atof(ft);
    if (const char* st = std::getenv("KLASP_BENCH_STOL"))  opts.stress_tol = std::atof(st);

    const int max_atoms = eval.recommended_batch_atoms();
    std::printf("klasp PET batched relax: %d structures, %ld atoms (%.1f/struct), "
                "cap %d iters, force_tol %.1e Ha/Bohr, sub-batch <= %d atoms\n",
                Nn, tot_atoms, (double)tot_atoms/Nn, opts.max_iter_full, opts.force_tol, max_atoms);

    std::vector<double> e_init(Nn);
    { PetResult r0 = eval.evaluate(batch, true); Kokkos::fence();
      for (int i = 0; i < Nn; ++i) e_init[i] = r0.energy[i]; }
    if (!prefix.empty()) write_frames(prefix + "_init.xyz", batch, e_init);

    // Splitting the batch by a total-atom cap is exact (structures are independent).
    std::vector<PetRelaxResult> res;
    res.reserve(Nn);
    Kokkos::fence();
    auto t0 = Clock::now();
    // KLASP_BENCH_WINDOW=n: every structure in one call with n active at a time,
    // rather than fixed chunks.
    if (const char* w = std::getenv("KLASP_BENCH_WINDOW")) {
        opts.max_active = std::atoi(w);
        res = pet_relax_batch(eval, batch, opts);
    }
    for (int lo = res.empty() ? 0 : Nn; lo < Nn; ) {
        int hi = lo; long acc = 0;
        while (hi < Nn) {
            const long na = (long)batch[hi].atomic_numbers.size();
            if (hi > lo + 1 && acc + na > max_atoms) break;   // keep B >= 2
            acc += na; ++hi;
        }
        std::vector<CrystalGeometry> chunk(batch.begin()+lo, batch.begin()+hi);
        auto rc = pet_relax_batch(eval, chunk, opts);
        res.insert(res.end(), rc.begin(), rc.end());
        lo = hi;
    }
    Kokkos::fence();
    const double secs = std::chrono::duration<double>(Clock::now()-t0).count();

    // Rounds = the longest trajectory (what wall time tracks); evals = actual
    // per-structure force calls.
    long evals = 0; int rounds = 0, conv = 0; double emin = 1e300;
    long eval_atoms = 0;
    std::vector<CrystalGeometry> finals; std::vector<double> e_final;
    for (int i = 0; i < Nn; ++i) {
        evals += res[i].full_iterations;
        eval_atoms += (long)res[i].full_iterations * (long)batch[i].atomic_numbers.size();
        rounds = std::max(rounds, res[i].full_iterations);
        conv += res[i].converged ? 1 : 0;
        emin = std::min(emin, res[i].energy);
        finals.push_back(res[i].structure); e_final.push_back(res[i].energy);
    }
    if (!prefix.empty()) write_frames(prefix + "_final.xyz", finals, e_final);

    std::printf("  wall %.2f s | %.1f ms/struct | %d converged/%d | %d rounds (max traj), "
                "%ld struct-evals\n", secs, 1000.0*secs/Nn, conv, Nn, rounds, evals);
    std::printf("  %.2f ms per struct-eval | %.1f us per atom-eval | %.0f atom-evals/s\n",
                1000.0*secs/evals, 1e6*secs/eval_atoms, eval_atoms/secs);
    std::printf("  min E %.6f eV | median iters %d\n", emin/EV_TO_AU,
                res[Nn/2].full_iterations);

    // Per-structure detail, for row-by-row comparison.
    std::printf("# idx natoms sg iters converged E_init_eV E_final_eV maxF_HaBohr\n");
    for (int i = 0; i < Nn; ++i)
        std::printf("%d %d %d %d %d %.6f %.6f %.3e\n", i,
                    (int)batch[i].atomic_numbers.size(), all[i].sg,
                    res[i].full_iterations, res[i].converged ? 1 : 0,
                    e_init[i]/EV_TO_AU, res[i].energy/EV_TO_AU, res[i].max_force);
    return 0;
}
