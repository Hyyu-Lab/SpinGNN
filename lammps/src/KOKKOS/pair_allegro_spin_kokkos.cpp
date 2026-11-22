/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
------------------------------------------------------------------------- */

#include "pair_allegro_spin_kokkos.h"

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "comm_kokkos.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>

using namespace LAMMPS_NS;

template<class DeviceType>
PairAllegroSpinKokkos<DeviceType>::PairAllegroSpinKokkos(LAMMPS *lmp) :
    PairAllegro_spin(lmp)
{
  kokkosable = 1;
  atomKK = static_cast<AtomKokkos *>(atom);
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | SP_MASK | FM_MASK;
  datamask_modify = F_MASK | FM_MASK;
  comm_forward = 3;
  comm_reverse = 3;
}

template<class DeviceType>
PairAllegroSpinKokkos<DeviceType>::~PairAllegroSpinKokkos()
= default;

template<class DeviceType>
void PairAllegroSpinKokkos<DeviceType>::coeff(int narg, char **arg)
{
  PairAllegro_spin::coeff(narg, arg);
  d_type_mapper = Kokkos::View<int *, DeviceType>(
      "PairAllegroSpinKokkos::type_mapper", type_mapper.size());
  auto h_type_mapper = Kokkos::create_mirror_view(d_type_mapper);
  for (size_t index = 0; index < type_mapper.size(); ++index)
    h_type_mapper(index) = type_mapper[index];
  Kokkos::deep_copy(d_type_mapper, h_type_mapper);
}

template<class DeviceType>
void PairAllegroSpinKokkos<DeviceType>::init_style()
{
  init_style_spin();
  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style allegro_spin/kk requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style allegro_spin/kk requires newton pair on");

  const int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->kokkos_host =
      std::is_same<DeviceType, LMPHostType>::value &&
      !std::is_same<DeviceType, LMPDeviceType>::value;
  neighbor->requests[irequest]->kokkos_device =
      std::is_same<DeviceType, LMPDeviceType>::value;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->ghost = 1;
}

template<class DeviceType>
void PairAllegroSpinKokkos<DeviceType>::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // This fork's legacy spin/kk atom communication only forwards positions.
  // Forward spin directions explicitly for regular force calls and SIB midpoints.
  atomKK->sync(Host, SP_MASK);
  auto *comm_kokkos = dynamic_cast<CommKokkos *>(comm);
  const bool restore_classic =
      comm_kokkos && !comm_kokkos->forward_pair_comm_classic;
  if (restore_classic) comm_kokkos->forward_pair_comm_classic = true;
  comm->forward_comm_pair(this);
  if (restore_classic) comm_kokkos->forward_pair_comm_classic = false;
  atomKK->modified(Host, SP_MASK);
  atomKK->sync(execution_space, datamask_read);

  const int nlocal = atom->nlocal;
  const int inum = list->inum;
  const int ntotal = inum + list->gnum;
  assert(inum == nlocal);

  if (nlocal_max < nlocal) {
    nlocal_max = nlocal;
    memory->grow(emag, nlocal_max, "pair/spin:emag");
  }

  auto *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  auto d_ilist = k_list->d_ilist;
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_x = atomKK->k_x.template view<DeviceType>();
  auto d_sp = atomKK->k_sp.template view<DeviceType>();
  auto d_type = atomKK->k_type.template view<DeviceType>();
  auto d_f = atomKK->k_f.template view<DeviceType>();
  auto d_fm = atomKK->k_fm.template view<DeviceType>();
  auto type_map = d_type_mapper;
  const double cutoff_sq = cutoff * cutoff;

  Kokkos::View<int *, DeviceType> edge_counts(
      "PairAllegroSpinKokkos::edge_counts", inum);
  Kokkos::parallel_for(
      "PairAllegroSpinKokkos::count_edges",
      Kokkos::RangePolicy<DeviceType>(0, inum), KOKKOS_LAMBDA(const int ii) {
        const int i = d_ilist(ii);
        int count = 0;
        const int jnum = d_numneigh(i);
        for (int jj = 0; jj < jnum; ++jj) {
          const int j = d_neighbors(i, jj) & NEIGHMASK;
          const double dx = d_x(i, 0) - d_x(j, 0);
          const double dy = d_x(i, 1) - d_x(j, 1);
          const double dz = d_x(i, 2) - d_x(j, 2);
          if (dx * dx + dy * dy + dz * dz < cutoff_sq) ++count;
        }
        edge_counts(ii) = count;
      });

  Kokkos::View<int *, DeviceType> edge_offsets(
      "PairAllegroSpinKokkos::edge_offsets", inum + 1);
  Kokkos::parallel_scan(
      "PairAllegroSpinKokkos::scan_edges",
      Kokkos::RangePolicy<DeviceType>(0, inum + 1),
      KOKKOS_LAMBDA(const int ii, int &update, const bool final) {
        if (final) edge_offsets(ii) = update;
        if (ii < inum) update += edge_counts(ii);
      });
  Kokkos::fence();
  int nedges = 0;
  Kokkos::deep_copy(nedges, Kokkos::subview(edge_offsets, inum));

  Kokkos::View<float *[3], Kokkos::LayoutRight, DeviceType> packed_pos(
      "PairAllegroSpinKokkos::pos", ntotal);
  Kokkos::View<float *[3], Kokkos::LayoutRight, DeviceType> packed_spin(
      "PairAllegroSpinKokkos::spin", ntotal);
  Kokkos::View<int64_t *, DeviceType> packed_types(
      "PairAllegroSpinKokkos::types", ntotal);
  Kokkos::View<int64_t **, Kokkos::LayoutRight, DeviceType> packed_edges(
      "PairAllegroSpinKokkos::edges", 2, nedges);

  Kokkos::parallel_for(
      "PairAllegroSpinKokkos::pack_atoms",
      Kokkos::RangePolicy<DeviceType>(0, ntotal), KOKKOS_LAMBDA(const int ii) {
        const int i = d_ilist(ii);
        packed_pos(i, 0) = static_cast<float>(d_x(i, 0));
        packed_pos(i, 1) = static_cast<float>(d_x(i, 1));
        packed_pos(i, 2) = static_cast<float>(d_x(i, 2));
        if (d_sp(i, 3) > 0) {
          packed_spin(i, 0) = static_cast<float>(d_sp(i, 0));
          packed_spin(i, 1) = static_cast<float>(d_sp(i, 1));
          packed_spin(i, 2) = static_cast<float>(d_sp(i, 2));
        } else {
          packed_spin(i, 0) = 0.0f;
          packed_spin(i, 1) = 0.0f;
          packed_spin(i, 2) = 0.0f;
        }
        packed_types(i) = static_cast<int64_t>(type_map(d_type(i) - 1));
      });
  Kokkos::parallel_for(
      "PairAllegroSpinKokkos::pack_edges",
      Kokkos::RangePolicy<DeviceType>(0, inum), KOKKOS_LAMBDA(const int ii) {
        const int i = d_ilist(ii);
        int edge = edge_offsets(ii);
        const int jnum = d_numneigh(i);
        for (int jj = 0; jj < jnum; ++jj) {
          const int j = d_neighbors(i, jj) & NEIGHMASK;
          const double dx = d_x(i, 0) - d_x(j, 0);
          const double dy = d_x(i, 1) - d_x(j, 1);
          const double dz = d_x(i, 2) - d_x(j, 2);
          if (dx * dx + dy * dy + dz * dz >= cutoff_sq) continue;
          packed_edges(0, edge) = static_cast<int64_t>(i);
          packed_edges(1, edge) = static_cast<int64_t>(j);
          ++edge;
        }
      });
  Kokkos::fence();

  auto h_pos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), packed_pos);
  auto h_spin = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), packed_spin);
  auto h_types = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), packed_types);
  auto h_edges = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), packed_edges);
  auto h_ilist = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_ilist);
  auto h_atom_sp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_sp);

  auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  torch::Tensor pos_tensor = torch::empty({ntotal, 3}, float_options);
  torch::Tensor spin_tensor = torch::empty({ntotal, 3}, float_options);
  torch::Tensor types_tensor = torch::empty({ntotal}, long_options);
  torch::Tensor edges_tensor = torch::empty({2, nedges}, long_options);
  std::memcpy(pos_tensor.data_ptr<float>(), h_pos.data(), ntotal * 3 * sizeof(float));
  std::memcpy(spin_tensor.data_ptr<float>(), h_spin.data(), ntotal * 3 * sizeof(float));
  std::memcpy(types_tensor.data_ptr<int64_t>(), h_types.data(), ntotal * sizeof(int64_t));
  std::memcpy(edges_tensor.data_ptr<int64_t>(), h_edges.data(), 2 * nedges * sizeof(int64_t));

  ModelOutputs output = run_model(
      pos_tensor, spin_tensor, edges_tensor, types_tensor, vflag_global,
      vflag_atom || cvflag_atom);
  output.forces = output.forces.contiguous();
  output.spin_forces = output.spin_forces.contiguous();
  output.atomic_energy = output.atomic_energy.contiguous();

  Kokkos::View<float *[3], Kokkos::LayoutRight, DeviceType> model_forces(
      "PairAllegroSpinKokkos::model_forces", ntotal);
  auto h_model_forces = Kokkos::create_mirror_view(model_forces);
  std::memcpy(
      h_model_forces.data(), output.forces.data_ptr<float>(),
      ntotal * 3 * sizeof(float));
  Kokkos::deep_copy(model_forces, h_model_forces);

  if (contribution_nmax < atom->nmax) {
    contribution_nmax = atom->nmax;
    memory->grow(force_contribution, contribution_nmax, 3,
                 "pair/allegro_spin/kk:force_contribution");
    memory->grow(spin_force_contribution, contribution_nmax, 3,
                 "pair/allegro_spin/kk:spin_force_contribution");
  }
  if (spin_force_reduced_nmax < atom->nmax) {
    spin_force_reduced_nmax = atom->nmax;
    memory->grow(spin_force_reduced, spin_force_reduced_nmax, 3,
                 "pair/allegro_spin/kk:spin_force_reduced");
  }
  auto forces = output.forces.accessor<float, 2>();
  auto spin_forces = output.spin_forces.accessor<float, 2>();
  for (int ii = 0; ii < ntotal; ++ii) {
    const int i = h_ilist(ii);
    for (int axis = 0; axis < 3; ++axis) {
      force_contribution[i][axis] = forces[i][axis];
      spin_force_contribution[i][axis] =
          h_atom_sp(i, 3) > 0 ? spin_forces[i][axis] / hbar : 0.0;
      spin_force_reduced[i][axis] = spin_force_contribution[i][axis];
    }
  }
  comm->reverse_comm_pair(this);

  Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType> reduced_spin_forces(
      "PairAllegroSpinKokkos::reduced_spin_forces", inum);
  auto h_reduced_spin_forces = Kokkos::create_mirror_view(reduced_spin_forces);
  for (int ii = 0; ii < inum; ++ii) {
    const int i = h_ilist(ii);
    for (int axis = 0; axis < 3; ++axis)
      h_reduced_spin_forces(ii, axis) = spin_force_reduced[i][axis];
  }
  Kokkos::deep_copy(reduced_spin_forces, h_reduced_spin_forces);
  Kokkos::parallel_for(
      "PairAllegroSpinKokkos::apply_forces",
      Kokkos::RangePolicy<DeviceType>(0, ntotal), KOKKOS_LAMBDA(const int ii) {
        const int i = d_ilist(ii);
        for (int axis = 0; axis < 3; ++axis) {
          d_f(i, axis) += model_forces(i, axis);
          if (ii < inum && d_sp(i, 3) > 0)
            d_fm(i, axis) += reduced_spin_forces(ii, axis);
        }
      });
  Kokkos::fence();
  atomKK->modified(execution_space, datamask_modify);

  auto atomic_energies = output.atomic_energy.accessor<float, 2>();
  eng_vdwl = 0.0;
  for (int ii = 0; ii < inum; ++ii) {
    const int i = h_ilist(ii);
    if (eflag_atom) eatom[i] = atomic_energies[i][0];
    eng_vdwl += atomic_energies[i][0];
    emag[i] = atomic_energies[i][0];
  }
  // constant_energy_offset is inherited from PairAllegro_spin and calibrated
  // once on a ghost-free probe graph in PairAllegro_spin::coeff, which
  // PairAllegroSpinKokkos::coeff delegates to.
  if (comm->me == 0) eng_vdwl += constant_energy_offset;

  if (vflag_global) {
    auto virial_tensor = output.virial.accessor<float, 3>();
    virial[0] += virial_tensor[0][0][0];
    virial[1] += virial_tensor[0][1][1];
    virial[2] += virial_tensor[0][2][2];
    virial[3] += virial_tensor[0][0][1];
    virial[4] += virial_tensor[0][0][2];
    virial[5] += virial_tensor[0][1][2];
  }

  if (vflag_atom || cvflag_atom) {
    auto atom_virial = output.atom_virial.accessor<float, 3>();
    for (int ii = 0; ii < ntotal; ++ii) {
      const int i = h_ilist(ii);
      if (vflag_atom) {
        vatom[i][0] += atom_virial[i][0][0];
        vatom[i][1] += atom_virial[i][1][1];
        vatom[i][2] += atom_virial[i][2][2];
        vatom[i][3] += atom_virial[i][0][1];
        vatom[i][4] += atom_virial[i][0][2];
        vatom[i][5] += atom_virial[i][1][2];
      }
      if (cvflag_atom) {
        cvatom[i][0] += atom_virial[i][0][0];
        cvatom[i][1] += atom_virial[i][1][1];
        cvatom[i][2] += atom_virial[i][2][2];
        cvatom[i][3] += atom_virial[i][0][1];
        cvatom[i][4] += atom_virial[i][0][2];
        cvatom[i][5] += atom_virial[i][1][2];
        cvatom[i][6] += atom_virial[i][1][0];
        cvatom[i][7] += atom_virial[i][2][0];
        cvatom[i][8] += atom_virial[i][2][1];
      }
    }
  }
}

template<class DeviceType>
int PairAllegroSpinKokkos<DeviceType>::pack_forward_comm(
    int n, int *list, double *buf, int, int *)
{
  int offset = 0;
  for (int index = 0; index < n; ++index) {
    const int i = list[index];
    for (int axis = 0; axis < 3; ++axis) buf[offset++] = atom->sp[i][axis];
  }
  return offset;
}

template<class DeviceType>
void PairAllegroSpinKokkos<DeviceType>::unpack_forward_comm(
    int n, int first, double *buf)
{
  int offset = 0;
  for (int i = first; i < first + n; ++i)
    for (int axis = 0; axis < 3; ++axis) atom->sp[i][axis] = buf[offset++];
}

namespace LAMMPS_NS {
template class PairAllegroSpinKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairAllegroSpinKokkos<LMPHostType>;
#endif
} // namespace LAMMPS_NS
