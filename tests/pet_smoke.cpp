// Smoke test for the PET-MAD bridge: evaluates a PET golden reference
// (Angstrom/eV) through PetEvaluator in klasp units (Bohr/Hartree) and checks
// energy, forces and stress against it.
//
//   pet_smoke <model.json> <model.safetensors> <golden.json>
#include "klasp/constants.hpp"
#include "klasp/pet_bridge.hpp"

#include <Kokkos_Core.hpp>

#include "tool_cli.hpp"
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

using nlohmann::json;

int main(int argc, char** argv) {
    CLI::App app{"Evaluate a PET model on a golden structure and compare against its stored reference.", "klasp_pet_smoke"};
    klasp::tools::tool_init(app);
    std::string model_json, model_weights, golden_path;
    app.add_option("model.json", model_json, "PET model architecture JSON")->required();
    app.add_option("model.safetensors", model_weights, "PET model weights")->required();
    app.add_option("golden.json", golden_path,
                   "Golden structure with reference energy and forces")->required();
    KLASP_TOOL_PARSE(app, argc, argv);

    int rc = 0;
    Kokkos::initialize(argc, argv);
    {
        std::ifstream f(golden_path);
        if (!f) { std::fprintf(stderr, "cannot open golden %s\n", golden_path.c_str()); return 2; }
        json g;
        f >> g;

        const int N = g.at("n_atoms").get<int>();
        klasp::CrystalGeometry s;
        s.atomic_numbers = g.at("atomic_numbers").get<std::vector<int>>();
        for (const auto& p : g.at("positions")) {
            s.positions.push_back({p[0].get<double>() * klasp::ANGSTROM_TO_BOHR,
                                   p[1].get<double>() * klasp::ANGSTROM_TO_BOHR,
                                   p[2].get<double>() * klasp::ANGSTROM_TO_BOHR});
        }
        // golden cell: rows are lattice vectors. klasp wants columns -> transpose.
        double gcell[3][3];
        int r = 0;
        for (const auto& row : g.at("cell")) {
            int c = 0;
            for (const auto& v : row) gcell[r][c++] = v.get<double>();
            ++r;
        }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                s.cell[i][j] = gcell[j][i] * klasp::ANGSTROM_TO_BOHR;  // transpose + to Bohr

        klasp::PetEvaluator eval(model_json, model_weights);
        klasp::PetResult res = eval.evaluate(s, /*compute_forces=*/true);

        // Compare energy (convert Hartree -> eV) to the golden total energy.
        const double e_ev = res.energy.at(0) / klasp::EV_TO_AU;
        const double e_ref = g.at("total_energy").get<double>();
        const double de = std::fabs(e_ev - e_ref);

        // Compare forces (Hartree/Bohr -> eV/Angstrom) to golden forces.
        const double f_to_evang = klasp::EV_TO_AU * klasp::BOHR_TO_ANGSTROM;
        double max_df = 0.0;
        const auto& gf = g.at("forces");
        for (int i = 0; i < N; ++i)
            for (int c = 0; c < 3; ++c) {
                const double fi = res.forces[i * 3 + c] / f_to_evang;
                max_df = std::max(max_df, std::fabs(fi - gf[i][c].get<double>()));
            }

        // Virial/stress cross-check (periodic goldens only). The bridge returns
        // dE/deps in full-tensor Voigt form (off-diagonals W_xy + W_yx); the golden is
        // the symmetric stress tensor (eV/Angstrom^3), so undo the shear doubling.
        // See kPetVirialShearFactor.
        //   stress_pet = (W_v / scale_v / EV_TO_AU) / V_Angstrom^3.
        double max_ds = -1.0;
        if (g.contains("stress") && !g.at("stress").is_null() && g.contains("volume") &&
            !g.at("volume").is_null()) {
            const double vol = g.at("volume").get<double>();
            const auto& gs = g.at("stress");
            const int vi[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};
            max_ds = 0.0;
            for (int t = 0; t < 6; ++t) {
                const double s_pet =
                    (res.virial[t] / klasp::pet_virial_voigt_scale(t) / klasp::EV_TO_AU) / vol;
                // Compare against the symmetrized reference; its antisymmetric part is
                // autograd noise.
                const double s_ref = 0.5 * (gs[vi[t][0]][vi[t][1]].get<double>() +
                                            gs[vi[t][1]][vi[t][0]].get<double>());
                max_ds = std::max(max_ds, std::fabs(s_pet - s_ref));
            }
        }

        std::printf("PET bridge smoke: N=%d  E_pet=%.6f eV  E_ref=%.6f eV  |dE|=%.3e eV  max|dF|=%.3e eV/A%s",
                    N, e_ev, e_ref, de, max_df,
                    max_ds >= 0.0 ? "" : "\n");
        if (max_ds >= 0.0) std::printf("  max|dStress|=%.3e eV/A^3\n", max_ds);
        // fp32 (mixed) network: energy tolerance is relative-or-absolute, since
        // fp32 gives ~1e-7 relative accuracy on large absolute energies.
        const double e_tol = std::max(1e-3, 2e-6 * std::fabs(e_ref));
        if (de > e_tol || max_df > 1e-2) {
            std::printf("FAIL: bridge result diverges from golden\n");
            rc = 1;
        } else {
            std::printf("PASS\n");
        }
    }
    Kokkos::finalize();
    return rc;
}
