// FFT-based crystal structure generator. See fft_generator.hpp.

#include "klasp/fft_generator.hpp"
#include "klasp/sgdata.hpp"
#include "klasp/constants.hpp"
#include "klasp/kgf.hpp"
#include "klasp/rotation.hpp"
#include "klasp/sampling_prior.hpp"
#include "klasp/volume_budget.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <fmt/base.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <random>

#ifdef KLASP_HAS_KOKKOSFFT
#include <KokkosFFT.hpp>
#else
#include <gemmi/third_party/pocketfft_hdronly.h>
#endif
#include <gemmi/asumask.hpp>
#include <gemmi/symmetry.hpp>

namespace klasp {

using Complex = Kokkos::complex<double>;

KOKKOS_INLINE_FUNCTION
static double sphere_ff_gen(double g_sq, double radius) {
    double V = (4.0 / 3.0) * M_PI * radius * radius * radius;
    if (g_sq < 1e-20) return V;
    double g_mag = Kokkos::sqrt(g_sq);
    double u = 2.0 * M_PI * g_mag * radius;
    if (u < 0.01) {
        double u2 = u * u;
        return V * (1.0 - u2 / 10.0 + u2 * u2 / 280.0);
    }
    return V * 3.0 * (Kokkos::sin(u) - u * Kokkos::cos(u)) / (u * u * u);
}

KOKKOS_INLINE_FUNCTION
static double fft_freq_gen(int i, int n) {
    return (i <= n / 2) ? double(i) : double(i - n);
}

// Build upper-triangular cell: r_cart = cell @ f_frac. Columns = lattice vectors.
static void build_cell(double a, double b, double c,
                        double ca, double cb, double cg, double sg_val,
                        double cell[3][3]) {
    double sqrt_x = std::sqrt(std::max(0.0, 1.0 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg));
    double v = a * b * c * sqrt_x;
    cell[0][0] = a;    cell[0][1] = b*cg;           cell[0][2] = c*cb;
    cell[1][0] = 0;    cell[1][1] = b*sg_val;       cell[1][2] = c*(ca - cb*cg)/sg_val;
    cell[2][0] = 0;    cell[2][1] = 0;              cell[2][2] = v / (a * b * sg_val);
}

static double cell_det(const double cell[3][3]) {
    return cell[0][0] * cell[1][1] * cell[2][2];  // upper-triangular
}

// Upper-triangular inverse
static void cell_inverse(const double cell[3][3], double ci[3][3]) {
    double det = cell_det(cell);
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) ci[i][j] = 0;
    ci[0][0] = cell[1][1]*cell[2][2] / det;
    ci[0][1] = -cell[0][1]*cell[2][2] / det;
    ci[0][2] = (cell[0][1]*cell[1][2] - cell[0][2]*cell[1][1]) / det;
    ci[1][1] = cell[0][0]*cell[2][2] / det;
    ci[1][2] = -cell[0][0]*cell[1][2] / det;
    ci[2][2] = cell[0][0]*cell[1][1] / det;
}

// R_cart = cell @ W_frac @ cell_inv
static void frac_to_cart_rot(const double cell[3][3], const double W[3][3],
                              double R_cart[3][3]) {
    double ci[3][3];
    cell_inverse(cell, ci);
    double tmp[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                tmp[i][j] += W[i][k] * ci[k][j];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            R_cart[i][j] = 0;
            for (int k = 0; k < 3; ++k)
                R_cart[i][j] += cell[i][k] * tmp[k][j];
        }
}

// Reciprocal lattice: columns of cell are lattice vectors
static void compute_recip(const double cell[3][3], double recip[3][3]) {
    double V = cell_det(cell);
    for (int i = 0; i < 3; ++i) {
        int j = (i + 1) % 3, k = (i + 2) % 3;
        recip[i][0] = (cell[1][j]*cell[2][k] - cell[2][j]*cell[1][k]) / V;
        recip[i][1] = (cell[2][j]*cell[0][k] - cell[0][j]*cell[2][k]) / V;
        recip[i][2] = (cell[0][j]*cell[1][k] - cell[1][j]*cell[0][k]) / V;
    }
}

// Direct collision check: verify that a molecule at fractional position frac_com
// doesn't overlap any of its symmetry images (including periodic images ±2).
static bool direct_collision_ok(
    const double cell[3][3],
    const SpaceGroupInfo& sg,
    const double R_mol[3][3],
    const double frac_com[3],
    const double* body_ang,  // (n_sites, 3) Cartesian Angstrom COM-centered
    const double* radii,     // (n_sites,) Angstrom
    int n_sites,
    double buffer)
{
    int n_ops = sg.n_symops;

    // Generate Cartesian positions of all atoms in all copies
    std::vector<std::array<double,3>> all_pos(n_ops * n_sites);
    for (int s = 0; s < n_ops; ++s) {
        double R_cs[3][3];
        frac_to_cart_rot(cell, sg.symops[s].rot, R_cs);
        double Rc[3][3] = {};
        for (int i=0;i<3;++i) for (int j=0;j<3;++j)
            for (int k=0;k<3;++k) Rc[i][j] += R_cs[i][k] * R_mol[k][j];

        double fc[3];
        for (int d = 0; d < 3; ++d) {
            fc[d] = sg.symops[s].trans[d];
            for (int k = 0; k < 3; ++k)
                fc[d] += sg.symops[s].rot[d][k] * frac_com[k];
            fc[d] -= std::floor(fc[d]);
        }
        double com[3] = {0, 0, 0};
        for (int d = 0; d < 3; ++d)
            for (int k = 0; k < 3; ++k)
                com[d] += cell[d][k] * fc[k];

        for (int a = 0; a < n_sites; ++a) {
            const double* ba = body_ang + a * 3;
            all_pos[s*n_sites+a] = {
                com[0] + Rc[0][0]*ba[0] + Rc[0][1]*ba[1] + Rc[0][2]*ba[2],
                com[1] + Rc[1][0]*ba[0] + Rc[1][1]*ba[1] + Rc[1][2]*ba[2],
                com[2] + Rc[2][0]*ba[0] + Rc[2][1]*ba[1] + Rc[2][2]*ba[2]
            };
        }
    }

    // Every ordered pair of copies against +/-2 periodic images, including a
    // copy against its own images (a molecule too long for its cell).
    for (int si = 0; si < n_ops; ++si)
        for (int sj = si; sj < n_ops; ++sj) {
            const bool self_image = (si == sj);
            for (int ai = 0; ai < n_sites; ++ai)
                for (int aj = 0; aj < n_sites; ++aj) {
                    double thresh = radii[ai] + radii[aj] + buffer;
                    double thresh2 = thresh * thresh;
                    auto& pi = all_pos[si*n_sites + ai];
                    auto& pj = all_pos[sj*n_sites + aj];
                    for (int dx = -2; dx <= 2; ++dx)
                    for (int dy = -2; dy <= 2; ++dy)
                    for (int dz = -2; dz <= 2; ++dz) {
                        // A copy does not clash with itself at zero shift.
                        if (self_image && dx == 0 && dy == 0 && dz == 0) continue;
                        double shx = dx*cell[0][0] + dy*cell[0][1] + dz*cell[0][2];
                        double shy = dx*cell[1][0] + dy*cell[1][1] + dz*cell[1][2];
                        double shz = dx*cell[2][0] + dy*cell[2][1] + dz*cell[2][2];
                        double drx = pi[0] - (pj[0] + shx);
                        double dry = pi[1] - (pj[1] + shy);
                        double drz = pi[2] - (pj[2] + shz);
                        if (drx*drx + dry*dry + drz*drz < thresh2) return false;
                    }
                }
        }
    return true;
}

// The symmetry-overlap map on the FFT grid, in Angstrom^3 (see fft_generator.hpp
// for O_s(x)). The identity operation is excluded: its contribution is
// independent of x, and self_image_ok handles it once per trial.
// `out` is (nx, ny, nz) row-major. Angstrom throughout.
void fft_overlap_map(const SpaceGroupInfo& sg, int id_op,
                            const double cell[3][3], const double R_mol[3][3],
                            const View2D<RealType>& body_positions_ang,
                            const View1D<RealType>& cov_radii,
                            int n_sites, double radius_buffer,
                            int nx, int ny, int nz,
                            std::vector<double>& out) {
    const int n_symops = sg.n_symops;
    const double vol = std::abs(cell_det(cell));
    double recip[3][3];
    compute_recip(cell, recip);

    auto d_recip = Kokkos::View<double[3][3], ExecSpace::memory_space>("recip");
    auto d_rot = Kokkos::View<double[3][3], ExecSpace::memory_space>("rot");
    {
        auto h = Kokkos::create_mirror_view(d_recip);
        for (int i=0;i<3;++i) for (int d=0;d<3;++d) h(i,d)=recip[i][d];
        Kokkos::deep_copy(d_recip, h);
    }
    {
        auto h = Kokkos::create_mirror_view(d_rot);
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) h(i,j)=R_mol[i][j];
        Kokkos::deep_copy(d_rot, h);
    }

    // Allocate FFT grids for this trial
    Kokkos::View<Complex***, Kokkos::LayoutRight, ExecSpace::memory_space>
        d_f_id("f_id",nx,ny,nz), d_f_s("f_s",nx,ny,nz),
        d_product("product",nx,ny,nz), d_corr_s("corr_s",nx,ny,nz);
    Kokkos::View<double***, Kokkos::LayoutRight, ExecSpace::memory_space>
        d_overlap("overlap",nx,ny,nz);
    Kokkos::deep_copy(d_overlap, 0.0);

    // Compute f_id(G)
    {
        auto body=body_positions_ang; auto rad=cov_radii; auto rot=d_rot;
        auto f_id=d_f_id; auto rcp=d_recip;
        double buf = radius_buffer;
        int ns=n_sites, gnx=nx, gny=ny, gnz=nz;
        Kokkos::parallel_for("fft_gen_f_id",
            Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>>({0,0,0},{gnx,gny,gnz}),
            KOKKOS_LAMBDA(int ix,int iy,int iz) {
                double h=fft_freq_gen(ix,gnx), k=fft_freq_gen(iy,gny), l=fft_freq_gen(iz,gnz);
                double gx=h*rcp(0,0)+k*rcp(1,0)+l*rcp(2,0);
                double gy=h*rcp(0,1)+k*rcp(1,1)+l*rcp(2,1);
                double gz=h*rcp(0,2)+k*rcp(1,2)+l*rcp(2,2);
                double g_sq=gx*gx+gy*gy+gz*gz;
                double f_re=0, f_im=0;
                for (int j=0;j<ns;++j) {
                    double ff=sphere_ff_gen(g_sq, rad(j)+buf);
                    if (Kokkos::abs(ff)<1e-30) continue;
                    double rx=rot(0,0)*body(j,0)+rot(0,1)*body(j,1)+rot(0,2)*body(j,2);
                    double ry=rot(1,0)*body(j,0)+rot(1,1)*body(j,1)+rot(1,2)*body(j,2);
                    double rz=rot(2,0)*body(j,0)+rot(2,1)*body(j,1)+rot(2,2)*body(j,2);
                    double phase=-2.0*M_PI*(gx*rx+gy*ry+gz*rz);
                    f_re+=ff*Kokkos::cos(phase); f_im+=ff*Kokkos::sin(phase);
                }
                f_id(ix,iy,iz) = Complex(f_re, f_im);
            });
    }

    // For each non-identity symop: compute cross-correlation and accumulate
    for (int s_idx = 0; s_idx < n_symops; ++s_idx) {
        if (s_idx == id_op) continue;

        double R_cs[3][3];
        frac_to_cart_rot(cell, sg.symops[s_idx].rot, R_cs);
        double Rc[3][3] = {};
        for (int i=0;i<3;++i) for (int j=0;j<3;++j)
            for (int k=0;k<3;++k) Rc[i][j] += R_cs[i][k]*R_mol[k][j];

        {
            auto h = Kokkos::create_mirror_view(d_rot);
            for (int i=0;i<3;++i) for (int j=0;j<3;++j) h(i,j)=Rc[i][j];
            Kokkos::deep_copy(d_rot, h);
        }

        // Compute f_s(G) with combined rotation
        {
            auto body=body_positions_ang; auto rad=cov_radii; auto rot=d_rot;
            auto f_s=d_f_s; auto rcp=d_recip;
            double buf = radius_buffer;
            int ns=n_sites, gnx=nx, gny=ny, gnz=nz;
            Kokkos::parallel_for("fft_gen_f_s",
                Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>>({0,0,0},{gnx,gny,gnz}),
                KOKKOS_LAMBDA(int ix,int iy,int iz) {
                    double h=fft_freq_gen(ix,gnx), k=fft_freq_gen(iy,gny), l=fft_freq_gen(iz,gnz);
                    double gx=h*rcp(0,0)+k*rcp(1,0)+l*rcp(2,0);
                    double gy=h*rcp(0,1)+k*rcp(1,1)+l*rcp(2,1);
                    double gz=h*rcp(0,2)+k*rcp(1,2)+l*rcp(2,2);
                    double g_sq=gx*gx+gy*gy+gz*gz;
                    double f_re=0, f_im=0;
                    for (int j=0;j<ns;++j) {
                        double ff=sphere_ff_gen(g_sq, rad(j)+buf);
                        if (Kokkos::abs(ff)<1e-30) continue;
                        double rx=rot(0,0)*body(j,0)+rot(0,1)*body(j,1)+rot(0,2)*body(j,2);
                        double ry=rot(1,0)*body(j,0)+rot(1,1)*body(j,1)+rot(1,2)*body(j,2);
                        double rz=rot(2,0)*body(j,0)+rot(2,1)*body(j,1)+rot(2,2)*body(j,2);
                        double phase=-2.0*M_PI*(gx*rx+gy*ry+gz*rz);
                        f_re+=ff*Kokkos::cos(phase); f_im+=ff*Kokkos::sin(phase);
                    }
                    f_s(ix,iy,iz) = Complex(f_re, f_im);
                });
        }

        // Product with translation phase
        double w0=sg.symops[s_idx].trans[0];
        double w1=sg.symops[s_idx].trans[1];
        double w2=sg.symops[s_idx].trans[2];
        {
            auto f_id=d_f_id; auto f_s=d_f_s; auto product=d_product;
            int gnx=nx,gny=ny,gnz=nz;
            Kokkos::parallel_for("fft_gen_prod",
                Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>>({0,0,0},{gnx,gny,gnz}),
                KOKKOS_LAMBDA(int ix,int iy,int iz) {
                    double h=fft_freq_gen(ix,gnx), k=fft_freq_gen(iy,gny), l=fft_freq_gen(iz,gnz);
                    double phase=2.0*M_PI*(h*w0+k*w1+l*w2);
                    Complex shift(Kokkos::cos(phase), Kokkos::sin(phase));
                    // f conj(f_s), not conj(f) f_s: the latter gives the map
                    // reflected through the origin, still real-valued.
                    product(ix,iy,iz) = f_id(ix,iy,iz)*Kokkos::conj(f_s(ix,iy,iz))*shift;
                });
        }

        // IFFT
#ifdef KLASP_HAS_KOKKOSFFT
        KokkosFFT::ifftn(ExecSpace(), d_product, d_corr_s,
                         KokkosFFT::axis_type<3>{0,1,2},
                         KokkosFFT::Normalization::backward);
#else
        {
            pocketfft::shape_t sh={size_t(nx),size_t(ny),size_t(nz)};
            pocketfft::stride_t st={
                ptrdiff_t(ny*nz*sizeof(std::complex<double>)),
                ptrdiff_t(nz*sizeof(std::complex<double>)),
                ptrdiff_t(sizeof(std::complex<double>))};
            pocketfft::shape_t ax={0,1,2};
            auto hp=Kokkos::create_mirror_view(d_product);
            auto hc=Kokkos::create_mirror_view(d_corr_s);
            Kokkos::deep_copy(hp, d_product);
            int N=nx*ny*nz;
            pocketfft::c2c(sh,st,st,ax,false,
                reinterpret_cast<std::complex<double>*>(hp.data()),
                reinterpret_cast<std::complex<double>*>(hc.data()),
                1.0/N);
            Kokkos::deep_copy(d_corr_s, hc);
        }
#endif

        // Accumulate overlap via M·x lookup
        int M[3][3];
        for (int i=0;i<3;++i) for (int j=0;j<3;++j)
            M[i][j]=static_cast<int>(std::round(sg.symops[s_idx].rot[i][j]))-(i==j?1:0);
        auto olap=d_overlap; auto corr_s=d_corr_s;
        int M00=M[0][0],M01=M[0][1],M02=M[0][2];
        int M10=M[1][0],M11=M[1][1],M12=M[1][2];
        int M20=M[2][0],M21=M[2][1],M22=M[2][2];
        int gnx=nx,gny=ny,gnz=nz;
        Kokkos::parallel_for("fft_gen_accum",
            Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>>({0,0,0},{gnx,gny,gnz}),
            KOKKOS_LAMBDA(int ix,int iy,int iz) {
                int lx=((M00*ix+M01*iy+M02*iz)%gnx+gnx)%gnx;
                int ly=((M10*ix+M11*iy+M12*iz)%gny+gny)%gny;
                int lz=((M20*ix+M21*iy+M22*iz)%gnz+gnz)%gnz;
                olap(ix,iy,iz) += corr_s(lx,ly,lz).real();
            });
    }

    // Normalise: IFFT gives (1/N)Σ, physical overlap is (1/V)Σ
    {
        auto olap=d_overlap;
        double scale = double(nx*ny*nz) / vol;
        int gnx=nx,gny=ny,gnz=nz;
        Kokkos::parallel_for("fft_gen_norm",
            Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>>({0,0,0},{gnx,gny,gnz}),
            KOKKOS_LAMBDA(int ix,int iy,int iz) { olap(ix,iy,iz) *= scale; });
    }

    // Extract overlap values to host for position selection
    auto h_overlap = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_overlap);
    out.resize(static_cast<std::size_t>(nx) * ny * nz);
    for (int ix = 0; ix < nx; ++ix)
        for (int iy = 0; iy < ny; ++iy)
            for (int iz = 0; iz < nz; ++iz)
                out[(static_cast<std::size_t>(ix) * ny + iy) * nz + iz] = h_overlap(ix, iy, iz);
}

// Whether the cell can hold the molecule at all: the reference copy against its
// own lattice translates, independent of the centre.
static bool self_image_ok(const double cell[3][3], const double R_mol[3][3],
                          const double* body_ang, const double* radii, int n_sites) {
    std::vector<std::array<double,3>> pos(n_sites);
    for (int a = 0; a < n_sites; ++a) {
        const double* ba = body_ang + a * 3;
        for (int d = 0; d < 3; ++d)
            pos[a][d] = R_mol[d][0]*ba[0] + R_mol[d][1]*ba[1] + R_mol[d][2]*ba[2];
    }
    for (int ai = 0; ai < n_sites; ++ai)
        for (int aj = 0; aj < n_sites; ++aj) {
            const double thresh = radii[ai] + radii[aj];
            const double thresh2 = thresh * thresh;
            for (int dx = -2; dx <= 2; ++dx)
            for (int dy = -2; dy <= 2; ++dy)
            for (int dz = -2; dz <= 2; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                const double shx = dx*cell[0][0] + dy*cell[0][1] + dz*cell[0][2];
                const double shy = dx*cell[1][0] + dy*cell[1][1] + dz*cell[1][2];
                const double shz = dx*cell[2][0] + dy*cell[2][1] + dz*cell[2][2];
                const double drx = pos[ai][0] - (pos[aj][0] + shx);
                const double dry = pos[ai][1] - (pos[aj][1] + shy);
                const double drz = pos[ai][2] - (pos[aj][2] + shz);
                if (drx*drx + dry*dry + drz*drz < thresh2) return false;
            }
        }
    return true;
}

int fft_generate_batch(
    int space_group,
    int seed_start,
    int n_seeds,
    const FFTGeneratorConfig& config,
    const View2D<RealType>& body_positions_ang,
    const View1D<RealType>& cov_radii,
    const View1D<RealType>& vdw_radii,
    int n_sites,
    View2D<RealType> cell_params_out,
    View3D<RealType> translations_out,
    View3D<RealType> rot_vectors_out,
    View1D<int> seeds_out,
    int max_output,
    int rank, int nranks)
{
    (void)rank; (void)nranks;   // seed is a rank-independent global index now
    auto seed_of = [seed_start](int trial) {
        return seed_start + trial;
    };
    auto& sg_info = get_space_group(space_group);
    int n_symops = sg_info.n_symops;
    int id_op = find_identity_symop(sg_info);
    LatticeType lattice = sg_info.lattice;

    // ASU brick from gemmi
    auto gemmi_sg = gemmi::find_spacegroup_by_number(space_group);
    auto asu_brick = gemmi::find_asu_brick(gemmi_sg);
    double asu_frac[3];
    for (int d = 0; d < 3; ++d)
        asu_frac[d] = double(asu_brick.size[d]) / double(gemmi::AsuBrick::denom);

    // Copy body data to host for direct collision checks
    auto h_body = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), body_positions_ang);
    auto h_radii = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cov_radii);
    std::vector<double> body_flat(n_sites * 3), radii_flat(n_sites);
    for (int i = 0; i < n_sites; ++i) {
        body_flat[i*3+0] = h_body(i, 0);
        body_flat[i*3+1] = h_body(i, 1);
        body_flat[i*3+2] = h_body(i, 2);
        radii_flat[i] = h_radii(i) + config.collision_buffer * 0.5;  // effective radius
    }

    // The volume budget from volume_budget.hpp, shared with every other generator.
    auto h_vdw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vdw_radii);
    std::vector<double> vdw_flat(n_sites);
    for (int i = 0; i < n_sites; ++i) vdw_flat[i] = static_cast<double>(h_vdw(i));
    const int one_mol_sites = n_sites, one_mol_offset = 0;
    const VolumeBudget budget = volume_budget(body_flat.data(), vdw_flat.data(),
                                              &one_mol_sites, &one_mol_offset,
                                              1, n_symops, lattice, config.tvp);
    const double max_vol = budget.max_volume;

    // Grid cap
    double side_max = std::cbrt(max_vol * 2.0);
    int max_side = std::min(64, (int)std::ceil(side_max / config.grid_spacing));

    // Device views for FFT work (reused across trials)
    auto d_recip = Kokkos::View<double[3][3], ExecSpace::memory_space>("recip");
    auto d_rot = Kokkos::View<double[3][3], ExecSpace::memory_space>("rot");

    // The prior this landscape is drawn from. Z'=1 only, hence n_indep = 1.
    const SamplingPrior prior = make_sampling_prior(space_group, 1, max_vol);

    // Per-axis minimum cell length for a pose, measured as the quasi-random
    // kernel does: the rotated molecule's extent along each fractional direction
    // plus a vdW diameter.
    const double mv = budget.max_vdw;
    auto measure_s_mins = [&](const PriorDraw& d, RealType s_mins[3]) {
        const double ca = d.ca, cb = d.cb, cg = d.cg;
        const double sg_val = d.sg_val, sqrt_x = d.sqrt_x;
        double unit_inv[3][3] = {};
        unit_inv[0][0] = 1.0;
        unit_inv[1][0] = -cg / sg_val;
        unit_inv[1][1] = 1.0 / sg_val;
        unit_inv[2][0] = (ca * cg - cb) / (sqrt_x * sg_val);
        unit_inv[2][1] = (cb * cg - ca) / (sqrt_x * sg_val);
        unit_inv[2][2] = sg_val / sqrt_x;

        double f_min[3] = {1e30, 1e30, 1e30}, f_max[3] = {-1e30, -1e30, -1e30};
        for (int i = 0; i < n_sites; ++i) {
            double r[3];
            for (int k = 0; k < 3; ++k)
                r[k] = static_cast<double>(d.rot[0][k][0]) * body_flat[i*3+0]
                     + static_cast<double>(d.rot[0][k][1]) * body_flat[i*3+1]
                     + static_cast<double>(d.rot[0][k][2]) * body_flat[i*3+2];
            for (int dd = 0; dd < 3; ++dd) {
                double f = 0;
                for (int k = 0; k < 3; ++k) f += r[k] * unit_inv[k][dd];
                f_min[dd] = std::min(f_min[dd], f);
                f_max[dd] = std::max(f_max[dd], f);
            }
        }
        for (int dd = 0; dd < 3; ++dd)
            s_mins[dd] = static_cast<RealType>((f_max[dd] - f_min[dd]) + mv * 2);
    };

    int n_written = 0;

    // Host mirrors for output
    auto h_cp = Kokkos::create_mirror_view(cell_params_out);
    auto h_tr = Kokkos::create_mirror_view(translations_out);
    auto h_rv = Kokkos::create_mirror_view(rot_vectors_out);
    auto h_seeds = Kokkos::create_mirror_view(seeds_out);

    for (int trial = 0; trial < n_seeds && n_written < max_output; ++trial) {
        int seed = seed_of(trial);

        // Cell and orientation from the shared SamplingPrior, the same draw the
        // quasi-random generator makes for this seed. Its translations are ignored.
        RealType qr[PRIOR_MAX_DIM];
        PriorDraw dof;
        if (!prior_draw(prior, seed, qr, measure_s_mins, dof)) continue;

        double R_mol[3][3];
        {
            double rv[3];
            for (int i = 0; i < 3; ++i) {
                rv[i] = static_cast<double>(dof.rot_vec[0][i]);
                for (int j = 0; j < 3; ++j) R_mol[i][j] = static_cast<double>(dof.rot[0][i][j]);
            }

            const double alpha_r = dof.alpha, beta_r = dof.beta, gamma_r = dof.gamma;
            const double ca = dof.ca, cb = dof.cb, cg = dof.cg;
            const double sg_val = dof.sg_val;
            const double a_len = dof.a, b_len = dof.b, c_len = dof.c;

            // The prior covers angles, volume and per-axis floors; the aspect
            // ratio cap keeps the FFT grid within the grid cap.
            double ratio = std::max({a_len,b_len,c_len}) / std::min({a_len,b_len,c_len});
            if (ratio > config.max_aspect) continue;

            // Build cell matrix
            double cell[3][3];
            build_cell(a_len, b_len, c_len, ca, cb, cg, sg_val, cell);
            double recip[3][3];
            compute_recip(cell, recip);

            if (!self_image_ok(cell, R_mol, body_flat.data(), radii_flat.data(), n_sites))
                continue;

            // Grid dimensions. The M_s * i lookup is exact only when axes that a
            // symmetry operation mixes share a grid dimension, so each coupled
            // axis pair gets the coarser resolution of the two.
            int n_d[3] = {
                std::min(max_side, std::max(8, (int)std::ceil(a_len/config.grid_spacing))),
                std::min(max_side, std::max(8, (int)std::ceil(b_len/config.grid_spacing))),
                std::min(max_side, std::max(8, (int)std::ceil(c_len/config.grid_spacing)))};
            {
                int group[3] = {0, 1, 2};
                auto join = [&](int u, int v) {
                    const int gu = group[u], gv = group[v];
                    if (gu == gv) return;
                    for (int d = 0; d < 3; ++d) if (group[d] == gv) group[d] = gu;
                };
                for (int sx = 0; sx < n_symops; ++sx)
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            if (i != j &&
                                std::lround(sg_info.symops[sx].rot[i][j]) != 0) join(i, j);
                for (int g = 0; g < 3; ++g) {
                    int m = 0;
                    for (int d = 0; d < 3; ++d) if (group[d] == g) m = std::max(m, n_d[d]);
                    if (m) for (int d = 0; d < 3; ++d) if (group[d] == g) n_d[d] = m;
                }
            }
            const int nx = n_d[0], ny = n_d[1], nz = n_d[2];

            // ASU bounds in grid units
            int asu_ix = std::max(1, (int)(asu_frac[0] * nx));
            int asu_iy = std::max(1, (int)(asu_frac[1] * ny));
            int asu_iz = std::max(1, (int)(asu_frac[2] * nz));

            // Upload reciprocal lattice and rotation
            {
                auto h = Kokkos::create_mirror_view(d_recip);
                for (int i=0;i<3;++i) for (int d=0;d<3;++d) h(i,d)=recip[i][d];
                Kokkos::deep_copy(d_recip, h);
            }
            {
                auto h = Kokkos::create_mirror_view(d_rot);
                for (int i=0;i<3;++i) for (int j=0;j<3;++j) h(i,j)=R_mol[i][j];
                Kokkos::deep_copy(d_rot, h);
            }

            std::vector<double> omap;
            fft_overlap_map(sg_info, id_op, cell, R_mol, body_positions_ang,
                            cov_radii, n_sites, config.collision_buffer * 0.5,
                            nx, ny, nz, omap);

            // Collect valid positions within ASU
            struct Candidate { int ix, iy, iz; double overlap; };
            std::vector<Candidate> candidates;
            for (int ix=0; ix<asu_ix; ++ix)
                for (int iy=0; iy<asu_iy; ++iy)
                    for (int iz=0; iz<asu_iz; ++iz) {
                        double ov = omap[(static_cast<std::size_t>(ix) * ny + iy) * nz + iz];
                        if (ov < config.collision_tol)
                            candidates.push_back({ix, iy, iz, ov});
                    }

            if (candidates.empty()) continue;

            // Sort by overlap (best first) and pick up to k diverse positions
            std::sort(candidates.begin(), candidates.end(),
                      [](auto& a, auto& b) { return a.overlap < b.overlap; });

            // Greedy farthest-point selection for diversity
            int k = std::min(config.k_per_trial, (int)candidates.size());
            std::vector<int> selected;
            selected.push_back(0);  // best overlap position first

            std::vector<double> min_dist(candidates.size(), 1e30);
            for (int sel = 1; sel < k; ++sel) {
                auto& last = candidates[selected.back()];
                for (size_t i = 0; i < candidates.size(); ++i) {
                    double dx = (candidates[i].ix - last.ix) / double(nx);
                    double dy = (candidates[i].iy - last.iy) / double(ny);
                    double dz = (candidates[i].iz - last.iz) / double(nz);
                    double d = dx*dx + dy*dy + dz*dz;
                    min_dist[i] = std::min(min_dist[i], d);
                }
                for (int s : selected) min_dist[s] = -1;
                int best = 0;
                double best_d = -1;
                for (size_t i = 0; i < candidates.size(); ++i)
                    if (min_dist[i] > best_d) { best_d = min_dist[i]; best = i; }
                selected.push_back(best);
            }

            // Verify each selected position with a direct collision check
            for (int slot = 0; slot < (int)selected.size(); ++slot) {
                if (n_written >= max_output) break;
                auto& cand = candidates[selected[slot]];
                // The map was evaluated at grid index i, so the position is i/n.
                double fx = double(cand.ix) / nx;
                double fy = double(cand.iy) / ny;
                double fz = double(cand.iz) / nz;
                double frac[3] = {fx, fy, fz};

                if (!direct_collision_ok(cell, sg_info, R_mol, frac,
                                          body_flat.data(), radii_flat.data(),
                                          n_sites, 0.0))  // buffer already in radii
                    continue;

                // Write output (Bohr + degrees for cell_params)
                int idx = n_written;
                h_cp(idx, 0) = a_len * ANGSTROM_TO_BOHR;
                h_cp(idx, 1) = b_len * ANGSTROM_TO_BOHR;
                h_cp(idx, 2) = c_len * ANGSTROM_TO_BOHR;
                h_cp(idx, 3) = alpha_r * 180.0 / M_PI;
                h_cp(idx, 4) = beta_r * 180.0 / M_PI;
                h_cp(idx, 5) = gamma_r * 180.0 / M_PI;
                h_tr(idx, 0, 0) = fx;
                h_tr(idx, 0, 1) = fy;
                h_tr(idx, 0, 2) = fz;
                h_rv(idx, 0, 0) = rv[0];
                h_rv(idx, 0, 1) = rv[1];
                h_rv(idx, 0, 2) = rv[2];
                // This trial's slot in its own identity block (see fft_identity).
                h_seeds(idx) = fft_identity(seed, slot, config.k_per_trial);
                ++n_written;
            }
        }
    }

    // Copy results to device (only the written portion)
    if (n_written > 0) {
        auto d_cp_sub = Kokkos::subview(cell_params_out, std::make_pair(0, n_written), Kokkos::ALL);
        auto d_tr_sub = Kokkos::subview(translations_out, std::make_pair(0, n_written), Kokkos::ALL, Kokkos::ALL);
        auto d_rv_sub = Kokkos::subview(rot_vectors_out, std::make_pair(0, n_written), Kokkos::ALL, Kokkos::ALL);
        auto d_sd_sub = Kokkos::subview(seeds_out, std::make_pair(0, n_written));
        auto h_cp_sub = Kokkos::subview(h_cp, std::make_pair(0, n_written), Kokkos::ALL);
        auto h_tr_sub = Kokkos::subview(h_tr, std::make_pair(0, n_written), Kokkos::ALL, Kokkos::ALL);
        auto h_rv_sub = Kokkos::subview(h_rv, std::make_pair(0, n_written), Kokkos::ALL, Kokkos::ALL);
        auto h_sd_sub = Kokkos::subview(h_seeds, std::make_pair(0, n_written));
        Kokkos::deep_copy(d_cp_sub, h_cp_sub);
        Kokkos::deep_copy(d_tr_sub, h_tr_sub);
        Kokkos::deep_copy(d_rv_sub, h_rv_sub);
        Kokkos::deep_copy(d_sd_sub, h_sd_sub);
    }

    return n_written;
}

int FftGenerator::generate(const GeneratorRequest& req, DofBatch& out) const {
    if (out.n_indep != 1) return 0;   // FFT packing is Z'=1 only
    return fft_generate_batch(
        req.space_group, req.seed_start, req.n_try, config,
        body_positions_ang, cov_radii, vdw_radii, n_sites,
        out.cell_params, out.translations, out.rot_vectors, out.seeds,
        out.capacity, 0, 1);
}

}  // namespace klasp
