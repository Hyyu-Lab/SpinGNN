// SPDX-License-Identifier: GPL-2.0-only
#include "pair_spingnn2_kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm_kokkos.h"
#include "error.h"
#include "kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include <type_traits>

using namespace LAMMPS_NS;

PairSpinGNN2Kokkos::PairSpinGNN2Kokkos(LAMMPS *lmp) : PairSpinGNN2(lmp)
{
  kokkosable = reverse_comm_device = 1;
  execution_space = ExecutionSpaceFromDevice<Device>::space;
  datamask_read = X_MASK | TYPE_MASK | F_MASK;
  datamask_modify = F_MASK;
}

int PairSpinGNN2Kokkos::load_model(const char *path)
{
#ifdef KOKKOS_ENABLE_CUDA
  return sp2_model_load_device(path,precision,Device().cuda_device(),&model);
#else
  return PairSpinGNN2::load_model(path);
#endif
}

void PairSpinGNN2Kokkos::init_style()
{
  PairSpinGNN2::init_style();
  auto request = neighbor->find_request(this);
  request->set_kokkos_device(1);
  request->set_kokkos_host(0);
  datamask_modify = F_MASK | (magnetic ? FM_MASK : 0) | (bec_name.empty() ? 0 : BEC_MASK);
  // The pair overwrites only owned BEC entries. A host fix or legacy halo
  // exchange may have modified the view since the preceding evaluation.
  // Synchronize the whole field before claiming its device copy.
  datamask_read = X_MASK | TYPE_MASK | datamask_modify | (magnetic ? SP_MASK : 0);
  type_map = decltype(type_map)("spingnn:type_map",species.size());
  auto h = Kokkos::create_mirror_view(type_map);
  for (size_t i=0; i<species.size(); ++i) h(i) = species[i];
  Kokkos::deep_copy(type_map,h);
  bec_sum = decltype(bec_sum)("spingnn:bec sum");
  globals = decltype(globals)("spingnn:global tallies");
  h_globals = decltype(h_globals)("spingnn:host global tallies");
  h_bec_sum = decltype(h_bec_sum)("spingnn:host bec sum");
  last_device = {};
  last_build = -1;
  ++lattice_epoch;
}

void PairSpinGNN2Kokkos::collect_edges()
{
#ifndef KOKKOS_ENABLE_CUDA
  auto *nl = static_cast<NeighListKokkos<Device> *>(list);
  for (int ii=0; ii<nl->inum; ++ii) {
    int i = nl->d_ilist(ii);
    for (int k=0; k<nl->d_numneigh(i); ++k) add_neighbor(i,nl->d_neighbors(i,k));
  }
#endif
}

void PairSpinGNN2Kokkos::evaluate(bool midpoint, bool reuse)
{
  auto *atoms = static_cast<AtomKokkos *>(atom);
#ifndef KOKKOS_ENABLE_CUDA
  atoms->sync(Host,datamask_read);
  PairSpinGNN2::evaluate(midpoint,reuse);
  atoms->modified(Host,midpoint ? FM_MASK : datamask_modify);
  atoms->sync(execution_space,midpoint ? FM_MASK : datamask_modify);
#else
  const double start = MPI_Wtime(), comm_start = timing[1];
  atoms->sync(execution_space,datamask_read);
  auto *nl = static_cast<NeighListKokkos<Device> *>(list);
  if (nl->inum != atom->nlocal) error->all(FLERR,"spingnn2 requires all owned centers");
  auto x = atoms->k_x.view<Device>();
  auto sp = atoms->k_sp.view<Device>();
  sp2_neighbors graph{};
  graph.struct_size = sizeof(graph);
  graph.num_atoms = atom->nlocal + atom->nghost;
  graph.num_owned = atom->nlocal;
  graph.num_global = atom->natoms;
  graph.positions = x.data();
  graph.spins = magnetic ? sp.data() : nullptr;
  graph.position_precision = 8*sizeof(*x.data());
  graph.position_atom_stride = x.stride(0);
  graph.position_component_stride = x.stride(1);
  graph.spin_atom_stride = sp.stride(0); graph.spin_component_stride = sp.stride(1);
  graph.types = atoms->k_type.view<Device>().data();
  graph.type_map = type_map.data();
  graph.type_map_size = type_map.extent(0);
  graph.neighbor_mask = NEIGHMASK;
  graph.neighbor_slot_capacity = nl->d_neighbors.extent_int(1);
  graph.ilist = nl->d_ilist.data(); graph.ilist_stride = nl->d_ilist.stride(0);
  graph.numneigh = nl->d_numneigh.data(); graph.numneigh_stride = nl->d_numneigh.stride(0);
  graph.firstneigh = nl->d_neighbors.data();
  graph.neighbor_atom_stride = nl->d_neighbors.stride(0);
  graph.neighbor_slot_stride = nl->d_neighbors.stride(1);
  if (last_build != neighbor->ncalls) {
    last_build = neighbor->ncalls;
    ++topology_epoch;
    ++lattice_epoch;
  }
  graph.topology_epoch = topology_epoch;
  if (!reuse) ++lattice_epoch;
  // A zero epoch refreshes both compacted edges and Stage-1 geometry.
  graph.lattice_epoch = can_cache_sib_lattice() ? lattice_epoch : 0;
  graph.stream = Device().cuda_stream();
  graph.flags = cuda_graph_enabled ? SP2_NEIGHBOR_CUDA_GRAPH : 0;
  auto options = opts;
  if (midpoint && graph.lattice_epoch) options.eval_flags |= SP2_SKIP_LATTICE_FORCES;
  // Moving geometry is refreshed by the next regular force evaluation.
  // Only its spin-only midpoint can use this lattice cache.
  if (graph.lattice_epoch && !frozen) options.eval_flags |= SP2_CACHE_STAGE1_FORWARD_ONLY;
  sp2_device_result result{};
  result.struct_size = sizeof(result);
  checked(sp2_evaluate_neighbors(model,&graph,&options,&result));
  last_device = result;
  if (precision == SP2_FLOAT64) accumulate<double>(result,midpoint);
  else accumulate<float>(result,midpoint);
  atoms->modified(execution_space,midpoint ? FM_MASK : datamask_modify);
  timing[0] += MPI_Wtime()-start-(timing[1]-comm_start);
  timing[2] += result.diagnostics.stage1_seconds;
  timing[3] += result.diagnostics.stage2_seconds;
  timing[4] += result.diagnostics.stage1_cache_hits != 0;
  ++timing[5];
  if (result.cuda_graph_status == SP2_GRAPH_CAPTURED) ++graph_timing[0];
  if (result.cuda_graph_status == SP2_GRAPH_REPLAYED) ++graph_timing[1];
  if (result.cuda_graph_status == SP2_GRAPH_FALLBACK) ++graph_timing[2];
#endif
}

template<class Real> void PairSpinGNN2Kokkos::accumulate(const sp2_device_result &result,bool midpoint)
{
  const int owned = atom->nlocal, all = owned + atom->nghost;
  const auto *f = static_cast<const Real *>(result.forces);
  const auto *sf = static_cast<const Real *>(result.spin_forces);
  const double units = inv_hbar;
  const bool need_halo = comm->nprocs > 1 || atom->nghost > 0;
  auto endpoint = endpoints;
  if (need_halo) {
    if (endpoints.extent_int(0)<all) Kokkos::realloc(endpoints,atom->nmax);
    endpoint = endpoints;
    Kokkos::parallel_for("spingnn endpoint contributions",all,KOKKOS_LAMBDA(int i) {
      for (int k=0; k<3; ++k) {
        endpoint(i,k) = midpoint ? 0. : f[3*i+k];
        endpoint(i,3+k) = sf ? sf[3*i+k]*units : 0.;
      }
    });
    midpoint_active = midpoint;
    double t = MPI_Wtime();
    auto *halo = dynamic_cast<CommKokkos *>(comm);
    if (!halo) error->all(FLERR,"spingnn2/kk requires Kokkos brick communication");
    halo->reverse_comm_device<Device>(this,midpoint ? 3 : 6);
    timing[1] += MPI_Wtime()-t;
  }
  auto *atoms = static_cast<AtomKokkos *>(atom);
  auto force_view = atoms->k_f.view<Device>();
  auto fm = atoms->k_fm.view<Device>();
  auto spin = atoms->k_sp.view<Device>();
  auto bec = atoms->k_bec.view<Device>();
  auto sums = bec_sum;
  const auto *born = static_cast<const Real *>(result.bec);
  const bool project = !midpoint && hard_asr;
  if (project) {
    Kokkos::deep_copy(Device(),sums,0.);
    Kokkos::parallel_for("spingnn ASR sum",owned,KOKKOS_LAMBDA(int i) {
      for (int k=0; k<9; ++k) Kokkos::atomic_add(&sums(k),double(born[9*i+k]));
    });
    if (comm->nprocs>1) {
      double t = MPI_Wtime();
      if (lmp->kokkos->gpu_aware_flag) {
        Device().fence();
        MPI_Allreduce(MPI_IN_PLACE,sums.data(),9,MPI_DOUBLE,MPI_SUM,world);
      } else {
        // Non-GPU-aware MPI must receive the nine ASR sums in pinned host memory.
        Kokkos::deep_copy(Device(),h_bec_sum,sums);
        Device().fence();
        MPI_Allreduce(MPI_IN_PLACE,h_bec_sum.data(),9,MPI_DOUBLE,MPI_SUM,world);
        Kokkos::deep_copy(Device(),sums,h_bec_sum);
      }
      timing[1] += MPI_Wtime()-t;
    }
  }
  const bool eg = !midpoint && eflag_global, vg = !midpoint && vflag_global;
  const bool ea = !midpoint && eflag_atom, va = !midpoint && (vflag_atom || cvflag_atom);
  auto totals = globals;
  if (eg || vg) Kokkos::deep_copy(Device(),totals,0.);
  if ((ea || va) && atom_tallies.extent_int(0)<owned) Kokkos::realloc(atom_tallies,owned);
  auto tallies = atom_tallies;
  const auto *energy = static_cast<const Real *>(result.owned_energy);
  const auto *vir = static_cast<const Real *>(result.atom_virial);
  const double n = atom->natoms, shift = asr_shift;
  Kokkos::parallel_for("spingnn apply owned outputs",owned,KOKKOS_LAMBDA(int i) {
    for (int k=0; k<3; ++k) {
      if (!midpoint) force_view(i,k) += need_halo ? endpoint(i,k) : double(f[3*i+k]);
      if (sf) {
        const double sf_k = need_halo ? endpoint(i,3+k) : double(sf[3*i+k])*units;
        fm(i,k) = (midpoint ? 0. : fm(i,k)) + (spin(i,3) > 0. ? sf_k : 0.);
      }
    }
    if (midpoint) return;
    if (ea) tallies(i,0) = energy[i];
    if (eg) Kokkos::atomic_add(&totals(0),double(energy[i]));
    const int order[]{0,4,8,1,2,5,3,6,7};
    for (int k=0; k<9; ++k) {
      int p=order[k];
      if (va) tallies(i,1+k) = vir[9*i+p];
      if (vg && k<6) Kokkos::atomic_add(&totals(1+k),
        0.5*(double(vir[9*i+p])+double(vir[9*i+3*(p%3)+p/3])));
      if (born) bec(i,k) = born[9*i+k]-(project ? sums(k)/n-shift : 0.);
    }
  });
  if (eg || vg) {
    Kokkos::deep_copy(Device(),h_globals,totals);
    Device().fence();
    eng_vdwl += h_globals(0);
    for (int k=0; k<6; ++k) virial[k] += h_globals(1+k);
  }
  if (ea || va) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),tallies);
    for (int i=0; i<owned; ++i) {
      if (ea) eatom[i] = h(i,0);
      if (vflag_atom) for (int k=0; k<6; ++k) vatom[i][k] = h(i,k+1);
      if (cvflag_atom) for (int k=0; k<9; ++k) cvatom[i][k] = h(i,k+1);
    }
  }
}

int PairSpinGNN2Kokkos::pack_reverse_comm_kokkos(int n,int first,DAT::tdual_double_1d &buffer)
{
  auto out = buffer.view<Device>();
  auto ep = endpoints;
  const int width = midpoint_active ? 3 : 6;
  Kokkos::parallel_for("spingnn pack halo",n,KOKKOS_LAMBDA(int i) {
    for (int k=0; k<width; ++k) out(i*width+k) = ep(first+i,6-width+k);
  });
  return width*n;
}
void PairSpinGNN2Kokkos::unpack_reverse_comm_kokkos(int n,DAT::tdual_int_1d list,DAT::tdual_double_1d &buf)
{
  auto in = buf.view<Device>();
  auto ids = list.view<Device>();
  auto ep = endpoints;
  const int width = midpoint_active ? 3 : 6;
  Kokkos::parallel_for("spingnn unpack halo",n,KOKKOS_LAMBDA(int i) {
    for (int k=0; k<width; ++k) Kokkos::atomic_add(&ep(ids(i),6-width+k),in(width*i+k));
  });
}

void PairSpinGNN2Kokkos::read_field(const std::string &name,double *out,size_t count)
{
#ifdef KOKKOS_ENABLE_CUDA
  if (!count || !last_device.struct_size) return;
  auto pos = name.find('_');
  auto &v = name.substr(0,pos) == "sf" ? last_device.spin_evidence : last_device.force_evidence;
  const auto field = name.substr(pos+1);
  const void *raw = field == "sigma" ? v.sigma : field == "Uepi" ? v.uepi :
                    field == "nu" ? v.nu : v.kappa;
  if (!raw) error->one(FLERR,"Missing SpinGNN2 device readout {}",name);
  // Only explicit LAMMPS extraction materializes an evidential field on host.
  auto copy = [&](auto ptr) {
    using Scalar = std::remove_pointer_t<decltype(ptr)>;
    Kokkos::View<Scalar*,Device,Kokkos::MemoryUnmanaged> source(ptr,count);
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),source);
    std::copy_n(h.data(),count,out);
  };
  if (precision == SP2_FLOAT64) copy(static_cast<const double *>(raw));
  else copy(static_cast<const float *>(raw));
#else
  PairSpinGNN2::read_field(name,out,count);
#endif
}
