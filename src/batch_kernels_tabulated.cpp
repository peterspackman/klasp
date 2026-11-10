// The tabulated pair model's instantiations of the asymunit kernels.
//
// Kept in its own translation unit: instantiating a second PairModel next to
// the Buckingham one changes the latter's FMA contraction and breaks
// bit-exactness. A further model needs its own file too.

#include "klasp/asymunit_fold.hpp"

namespace klasp {

namespace {
template <typename Space>
TabulatedModelT<Space> tabulated_model(const ModelTablesT<Space>& tables) {
    return TabulatedModelT<Space>{tables.pair_tables, tables.n_types};
}
}  // namespace

template <int MaxRank, typename Space>
void asymunit_pair_pass_tabulated(const ModelTablesT<Space>& tables,
                                  BatchGeometryT<Space>& geom,
                                  RigidDofStateT<Space>& dofs,
                                  const SView3D<RealType, Space>& c_dgt,
                                  const SView3D<RealType, Space>& c_dgr,
                                  bool has_dipoles, const WolfDSF& wolf,
                                  const ForceField& ff,
                                  const EnergyModelParams& m, RealType mp_cut2) {
    asymunit_pair_pass<MaxRank, Space>(tables, geom, dofs, c_dgt, c_dgr,
                                       has_dipoles, wolf, ff, m, mp_cut2,
                                       tabulated_model<Space>(tables));
}

template <typename Space>
void asymunit_force_constants_tabulated(const ModelTablesT<Space>& tables,
                                        const BatchGeometryT<Space>& geom,
                                        TrustRegionScratchT<Space>& trs,
                                        const BatchOptimizeSettings& settings) {
    asymunit_force_constants(tables, geom, trs, settings,
                             tabulated_model<Space>(tables));
}

template void asymunit_pair_pass_tabulated<0, ExecSpace>(
    const ModelTables&, BatchGeometry&, RigidDofState&,
    const SView3D<RealType, ExecSpace>&, const SView3D<RealType, ExecSpace>&,
    bool, const WolfDSF&, const ForceField&, const EnergyModelParams&, RealType);
template void asymunit_pair_pass_tabulated<1, ExecSpace>(
    const ModelTables&, BatchGeometry&, RigidDofState&,
    const SView3D<RealType, ExecSpace>&, const SView3D<RealType, ExecSpace>&,
    bool, const WolfDSF&, const ForceField&, const EnergyModelParams&, RealType);
template void asymunit_pair_pass_tabulated<2, ExecSpace>(
    const ModelTables&, BatchGeometry&, RigidDofState&,
    const SView3D<RealType, ExecSpace>&, const SView3D<RealType, ExecSpace>&,
    bool, const WolfDSF&, const ForceField&, const EnergyModelParams&, RealType);
template void asymunit_pair_pass_tabulated<3, ExecSpace>(
    const ModelTables&, BatchGeometry&, RigidDofState&,
    const SView3D<RealType, ExecSpace>&, const SView3D<RealType, ExecSpace>&,
    bool, const WolfDSF&, const ForceField&, const EnergyModelParams&, RealType);
template void asymunit_pair_pass_tabulated<4, ExecSpace>(
    const ModelTables&, BatchGeometry&, RigidDofState&,
    const SView3D<RealType, ExecSpace>&, const SView3D<RealType, ExecSpace>&,
    bool, const WolfDSF&, const ForceField&, const EnergyModelParams&, RealType);

template void asymunit_force_constants_tabulated<ExecSpace>(
    const ModelTables&, const BatchGeometry&, TrustRegionScratch&,
    const BatchOptimizeSettings&);

}  // namespace klasp
