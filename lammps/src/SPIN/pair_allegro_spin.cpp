/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Anders Johansson (Harvard)
------------------------------------------------------------------------- */

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "pt2_loader.h"
#include "tokenizer.h"
#include "torch_determinism.h"
#include "torch_device.h"
#include "update.h"
#include "utils.h"
#include <pair_allegro_spin.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

// TODO: Only if MPI is available
#include <mpi.h>

// We have to do a backward compatability hack for <1.10
// https://discuss.pytorch.org/t/how-to-check-libtorch-version/77709/4
// Basically, the check in torch::jit::freeze
// (see https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp#L479)
// is wrong, and we have ro "reimplement" the function
// to get around that...
// it's broken in 1.8 and 1.9
// BUT the internal logic in the function is wrong in 1.10
// So we only use torch::jit::freeze in >=1.11
#if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
#define DO_TORCH_FREEZE_HACK
// For the hack, need more headers:
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/frozen_ops_to_mkldnn.h>
#endif

using namespace LAMMPS_NS;

namespace {

std::array<double, 12> current_box_snapshot(const Domain *domain)
{
  std::array<double, 12> snapshot;
  for (int axis = 0; axis < 3; ++axis)
  {
    snapshot[axis] = domain->boxlo[axis];
    snapshot[axis + 3] = domain->boxhi[axis];
  }
  for (int component = 0; component < 6; ++component)
    snapshot[component + 6] = domain->h[component];
  return snapshot;
}

} // namespace

PairAllegro_spin::PairAllegro_spin(LAMMPS *lmp) :
    PairSpin(lmp), contribution_nmax(0), force_contribution(nullptr),
    spin_force_contribution(nullptr), spin_force_reduced_nmax(0),
    spin_force_reduced(nullptr)
{
  restartinfo = 0;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
  centroidstressflag =
      CENTROID_AVAIL; // Allow the use of the centroid/stress/atom. - Added by Hongyu Yu
  comm_reverse = 3;

  if (const char *env_p = std::getenv("ALLEGRO_DEBUG"))
  {
    std::cout << "PairAllegro_spin is in DEBUG mode, since ALLEGRO_DEBUG is in env\n";
    debug_mode = 1;
  }

  const char *cache_value = std::getenv("SPINGNN_SIB_GEOM_CACHE");
  if (cache_value == nullptr || cache_value[0] == '\0' ||
      std::strcmp(cache_value, "0") == 0)
    sib_geometry_cache_enabled = false;
  else if (std::strcmp(cache_value, "1") == 0)
    sib_geometry_cache_enabled = true;
  else
    sib_geometry_cache_invalid_value = cache_value;

  device = spingnn_torch_device(world, error);
  std::cout << "Allegro_spin is using device " << device << "\n";
}

PairAllegro_spin::~PairAllegro_spin()
{
  pt2_model.reset();
  memory->destroy(force_contribution);
  memory->destroy(spin_force_contribution);
  memory->destroy(spin_force_reduced);
  if (allocated)
  {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(emag);
  }
}

void PairAllegro_spin::init_style()
{
  init_style_spin();

  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style PairAllegro_spin requires atom IDs");

  // need a full neighbor list
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;

  neighbor->requests[irequest]->ghost = 1;

  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style PairAllegro_spin requires newton pair on");
}

double PairAllegro_spin::init_one(int i, int j)
{
  return cutoff;
}

void PairAllegro_spin::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

void PairAllegro_spin::settings(int narg, char ** /*arg*/)
{
  // "allegro" should be the only word after "pair_style" in the input file.
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style command, too many arguments");

  check_spin_settings();
}

void PairAllegro_spin::coeff(int narg, char **arg)
{
  perf_state.validate(error);
  host_staging.validate(error);
  if (!sib_geometry_cache_invalid_value.empty())
    error->all(
        FLERR, "SPINGNN_SIB_GEOM_CACHE must be exactly 0 or 1; got '" +
                   sib_geometry_cache_invalid_value + "'");
  if (perf_state.enabled() && debug_mode)
    error->all(FLERR, "SPINGNN_PERF and ALLEGRO_DEBUG cannot be enabled together");

  if (!allocated)
    allocate();

  int ntypes = atom->ntypes;

  // Should be exactly 3 arguments following "pair_coeff" in the input file.
  if (narg != (3 + ntypes))
    error->all(FLERR,
               "Incorrect args for pair coefficients, should be * * <model>.pth <type1> <type2> "
               "... <typen>");

  // Ensure I,J args are "* *".
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  model_input.clear();
  model_input_vector.clear();
  pt2_inputs.clear();

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      setflag[i][j] = 0;

  std::vector<std::string> elements(ntypes);
  for (int i = 0; i < ntypes; i++)
  {
    elements[i] = arg[i + 1];
  }

  const std::string model_path(arg[2]);
  std::cout << "Allegro_spin: Loading model from " << model_path << "\n";

  std::unordered_map<std::string, std::string> torchscript_metadata = {{"config", ""},
                                                           {"nequip_version", ""},
                                                           {"r_max", ""},
                                                           {"n_species", ""},
                                                           {"type_names", ""},
                                                           {"_jit_bailout_depth", ""},
                                                           {"_jit_fusion_strategy", ""},
                                                           {"allow_tf32", ""},
                                                           {"schema_version", ""},
                                                           {"provided_outputs", ""},
                                                           {"required_inputs", ""},
                                                           {"default_dtype", ""},
                                                           {"energy_units", ""},
                                                           {"length_units", ""},
                                                           {"has_spin", ""},
                                                           {"has_bec", ""},
                                                           {"edge_prune", ""}};
  DeployMetadata metadata;
  use_pt2 = is_pt2_package_path(model_path);
  pt2_model.reset();
  pt2_input_signature.clear();
  pt2_output_indices.clear();
  if (use_pt2)
  {
    const std::vector<std::string> expected_inputs = {
        "pos", "spin", "edge_index", "atom_types"};
    LoadedPT2Model loaded = load_verified_pt2_model(
        error, model_path, "spin/allegro_spin", device, expected_inputs);
    pt2_model = std::move(loaded.model);
    metadata = std::move(loaded.manifest);
    pt2_input_signature = std::move(loaded.input_signature);
    pt2_output_indices = std::move(loaded.output_indices);
  }
  else
  {
    model = torch::jit::load(model_path, device, torchscript_metadata);
    model.eval();
    if (torchscript_metadata["nequip_version"].empty())
      error->all(
          FLERR,
          "The indicated TorchScript file does not appear to be a deployed NequIP model; did "
          "you forget to run `nequip-deploy`?");
    metadata = DeployMetadata(
        torchscript_metadata.begin(), torchscript_metadata.end());
  }

  const std::vector<std::string> available_inputs = {
      "pos", "spin", "edge_index", "atom_types"};
  validate_deploy_metadata(
      error, metadata, "spin/allegro_spin",
      {"total_energy", "forces", "spin_forces", "atomic_energy", "virial", "atom_virial"},
      available_inputs);

  // If the TorchScript model is not already frozen, freeze it.
  // This is the check used by PyTorch: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/api/module.cpp#L476
  if (!use_pt2 && model.hasattr("training"))
  {
    std::cout << "Allegro_spin: Freezing TorchScript model...\n";
#ifdef DO_TORCH_FREEZE_HACK
    // Do the hack
    // Copied from the implementation of torch::jit::freeze,
    // except without the broken check
    // See https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp
    bool optimize_numerics = true; // the default
    // the {} is preserved_attrs
    auto out_mod = freeze_module(model, {});
    // See 1.11 bugfix in https://github.com/pytorch/pytorch/pull/71436
    auto graph = out_mod.get_method("forward").graph();
    OptimizeFrozenGraph(graph, optimize_numerics);
    model = out_mod;
#else
    // Do it normally
    model = torch::jit::freeze(model);
#endif
  }

#if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
  if (!use_pt2)
  {
  // Set JIT bailout to avoid long recompilations for many steps
  size_t jit_bailout_depth;
  if (metadata["_jit_bailout_depth"].empty())
  {
    // This is the default used in the Python code
    jit_bailout_depth = 2;
  }
  else
  {
    jit_bailout_depth = std::stoi(metadata["_jit_bailout_depth"]);
  }
  torch::jit::getBailoutDepth() = jit_bailout_depth;
  }
#else
  if (!use_pt2)
  {
    // In PyTorch >=1.11, this is now set_fusion_strategy.
    torch::jit::FusionStrategy strategy;
    if (metadata["_jit_fusion_strategy"].empty())
      strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
    else
    {
      std::stringstream strat_stream(metadata["_jit_fusion_strategy"]);
      std::string fusion_type, fusion_depth;
      while (std::getline(strat_stream, fusion_type, ','))
      {
        std::getline(strat_stream, fusion_depth, ';');
        strategy.push_back({fusion_type == "STATIC" ? torch::jit::FusionBehavior::STATIC
                                                    : torch::jit::FusionBehavior::DYNAMIC,
                            std::stoi(fusion_depth)});
      }
    }
    torch::jit::setFusionStrategy(strategy);
  }
#endif

  // Set whether to allow TF32:
  bool allow_tf32;
  if (metadata["allow_tf32"].empty())
  {
    // Better safe than sorry
    allow_tf32 = false;
  }
  else
  {
    // It gets saved as an int 0/1
    allow_tf32 = std::stoi(metadata["allow_tf32"]);
  }
  // See https://pytorch.org/docs/stable/notes/cuda.html
  at::globalContext().setAllowTF32CuBLAS(allow_tf32);
  at::globalContext().setAllowTF32CuDNN(allow_tf32);
  if (configure_torch_determinism(error) && comm->me == 0)
    utils::logmesg(
        lmp,
        "Pair style spin/allegro_spin deterministic algorithms enabled "
        "(SPINGNN_DETERMINISTIC=1)\n");

  // std::cout << "Allegro_spin: Information from model: " << metadata.size() << " key-value pairs\n";
  // for( const auto& n : metadata ) {
  //   std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
  // }

  cutoff = std::stod(metadata["r_max"]);

  // LAMMPS atom types are 1-based, while this vector is 0-based.  Keep
  // unmapped entries at -1 so a missing model species cannot silently be
  // interpreted as model type 0.
  type_mapper.assign(ntypes, -1);
  std::stringstream ss;
  int n_species = std::stod(metadata["n_species"]);
  ss << metadata["type_names"];
  std::cout << "Type mapping:"
            << "\n";
  std::cout << "Allegro_spin type | Allegro_spin name | LAMMPS type | LAMMPS name"
            << "\n";
  for (int i = 0; i < n_species; i++)
  {
    std::string ele;
    ss >> ele;
    for (int itype = 1; itype <= ntypes; itype++)
    {
      if (ele.compare(arg[itype + 3 - 1]) == 0)
      {
        type_mapper[itype - 1] = i;
        std::cout << i << " | " << ele << " | " << itype << " | " << arg[itype + 3 - 1] << "\n";
      }
    }
  }

  for (int itype = 1; itype <= ntypes; itype++)
    if (type_mapper[itype - 1] < 0)
      error->all(FLERR, "Pair style PairAllegro_spin could not map all LAMMPS atom types");

  // set setflag i,j for type pairs where both are mapped to elements
  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      if ((type_mapper[i - 1] >= 0) && (type_mapper[j - 1] >= 0))
        setflag[i][j] = 1;

  char *batchstr = std::getenv("BATCHSIZE");
  if (batchstr != NULL)
  {
    batch_size = std::atoi(batchstr);
  }
  calibrate_energy_offset();
  host_staging.reset_generation();
  reset_sib_geometry_cache();
}

/* ----------------------------------------------------------------------
   Measure the model's constant total-energy offset once.

   The deployed model's total_energy is not guaranteed to equal the sum of
   its atomic_energy over the ghost-containing per-rank graph LAMMPS feeds
   it: rescale modules can attach global (per-configuration) terms to the
   total, and ghost nodes then make total - sum(atomic) depend on the
   processor decomposition and ghost count.  Adding that per-rank difference
   to eng_vdwl made the reported potential energy rank- and
   migration-dependent (bulk2 report: hundreds of eV between 1 and 4 ranks).
   Instead evaluate the offset once on a small ghost-free probe graph, where
   total - sum(atomic) is the model's global energy constant, and apply that
   constant once globally (rank 0) per configuration.
------------------------------------------------------------------------- */

void PairAllegro_spin::calibrate_energy_offset()
{
  // three type-0 atoms on a line; every pair is within the cutoff
  const double spacing = std::min(1.5, 0.35 * cutoff);
  torch::Tensor pos = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat));
  auto pos_a = pos.accessor<float, 2>();
  pos_a[1][0] = spacing;
  pos_a[2][0] = 2.0 * spacing;
  torch::Tensor spin = torch::zeros({3, 3}, pos.options());
  torch::Tensor edges = torch::tensor(
      {{0, 0, 1, 1, 2, 2}, {1, 2, 0, 2, 0, 1}},
      torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor types = torch::zeros({3}, torch::TensorOptions().dtype(torch::kInt64));

  ModelOutputs probe = run_model(pos, spin, edges, types, false, false);
  const double total = static_cast<double>(probe.total_energy.item<float>());
  const double atomic_sum =
      static_cast<double>(probe.atomic_energy.sum().item<float>());
  constant_energy_offset = total - atomic_sum;
  if (!std::isfinite(constant_energy_offset))
    error->one(FLERR, "Pair style spin/allegro_spin probe evaluation of the "
                      "total-energy offset was not finite");
  if (comm->me == 0 && std::fabs(constant_energy_offset) > 1.0e-6)
    utils::logmesg(lmp, "Allegro_spin: constant total-energy offset "
                        "{:.6g} eV (total_energy - sum(atomic_energy) on a "
                        "ghost-free probe graph)\n", constant_energy_offset);
}

void *PairAllegro_spin::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str, "cut") == 0)
    return (void *)&cutoff;
  return nullptr;
}

void PairAllegro_spin::set_sib_geometry_reuse_hint(bigint timestep)
{
  if (sib_geometry_cache_enabled)
    sib_geometry_hint_step = static_cast<long long>(timestep);
}

bool PairAllegro_spin::sib_geometry_cache_eligible() const
{
  return sib_geometry_cache_enabled && comm->nprocs == 1 && force->pair == this;
}

bool PairAllegro_spin::sib_geometry_matches(
    int ntotal, int *ilist, double **x, tagint *tag, int *type) const
{
  const SibGeometryGeneration &generation = sib_geometry_generation;
  if (!sib_geometry_cache_valid ||
      generation.timestep != static_cast<long long>(update->ntimestep) ||
      generation.neighbor_ncalls != static_cast<long long>(neighbor->ncalls) ||
      generation.neighbor_lastcall != static_cast<long long>(neighbor->lastcall) ||
      neighbor->ago != generation.neighbor_ago + 1 ||
      generation.nlocal != atom->nlocal || generation.nghost != atom->nghost ||
      generation.nmax != atom->nmax || generation.triclinic != domain->triclinic ||
      generation.cutoff != cutoff || list->inum != atom->nlocal ||
      list->gnum != atom->nghost ||
      type_mapper_snapshot != type_mapper || box_snapshot != current_box_snapshot(domain))
    return false;

  if (!sib_cached_pos_device.defined() || !sib_cached_edges_device.defined() ||
      !sib_cached_types_device.defined() ||
      sib_cached_pos_device.size(0) != ntotal ||
      sib_cached_pos_device.size(1) != 3 ||
      sib_cached_edges_device.size(0) != 2 ||
      sib_cached_edges_device.size(1) != generation.nedges ||
      sib_cached_types_device.size(0) != ntotal ||
      atom_index_snapshot.size() != static_cast<size_t>(ntotal) ||
      position_snapshot.size() != static_cast<size_t>(3 * ntotal) ||
      tag_snapshot.size() != static_cast<size_t>(ntotal) ||
      type_snapshot.size() != static_cast<size_t>(ntotal))
    return false;

  for (int ii = 0; ii < ntotal; ++ii)
  {
    const int i = ilist[ii];
    if (atom_index_snapshot[ii] != i ||
        tag_snapshot[ii] != static_cast<long long>(tag[i]) ||
        type_snapshot[ii] != type[i])
      return false;
    for (int axis = 0; axis < 3; ++axis)
      if (position_snapshot[3 * ii + axis] != x[i][axis]) return false;
  }
  return true;
}

void PairAllegro_spin::capture_sib_geometry(
    int ntotal, int nghost, int nedges, int *ilist, double **x,
    tagint *tag, int *type)
{
  sib_geometry_generation = {
      static_cast<long long>(update->ntimestep),
      static_cast<long long>(neighbor->ncalls),
      static_cast<long long>(neighbor->lastcall),
      neighbor->ago,
      atom->nlocal,
      nghost,
      atom->nmax,
      nedges,
      domain->triclinic,
      cutoff};
  box_snapshot = current_box_snapshot(domain);
  type_mapper_snapshot = type_mapper;
  atom_index_snapshot.resize(ntotal);
  position_snapshot.resize(3 * ntotal);
  tag_snapshot.resize(ntotal);
  type_snapshot.resize(ntotal);
  for (int ii = 0; ii < ntotal; ++ii)
  {
    const int i = ilist[ii];
    atom_index_snapshot[ii] = i;
    tag_snapshot[ii] = static_cast<long long>(tag[i]);
    type_snapshot[ii] = type[i];
    for (int axis = 0; axis < 3; ++axis)
      position_snapshot[3 * ii + axis] = x[i][axis];
  }
  sib_geometry_cache_valid = sib_cached_pos_device.defined() &&
      sib_cached_edges_device.defined() && sib_cached_types_device.defined();
}

void PairAllegro_spin::reset_sib_geometry_cache()
{
  sib_geometry_hint_step = std::numeric_limits<long long>::min();
  sib_geometry_cache_valid = false;
  sib_geometry_generation = SibGeometryGeneration();
  atom_index_snapshot.clear();
  position_snapshot.clear();
  tag_snapshot.clear();
  type_snapshot.clear();
  type_mapper_snapshot.clear();
  sib_cached_pos_device = torch::Tensor();
  sib_cached_edges_device = torch::Tensor();
  sib_cached_types_device = torch::Tensor();
}

template <bool Profile, bool Staged>
PairAllegro_spin::ModelOutputs PairAllegro_spin::run_model_impl(
    const torch::Tensor &pos_tensor, const torch::Tensor &spin_tensor,
    const torch::Tensor &edges_tensor, const torch::Tensor &types_tensor,
    bool need_global_virial, bool need_atom_virial,
    bool reuse_sib_geometry, bool capture_sib_geometry,
    SpingnnPerfContext<true> *perf, SpingnnStagingUse *staging_use)
{
  if constexpr (Profile) perf->h2d_start();
  torch::Tensor pos_device;
  torch::Tensor spin_device;
  torch::Tensor edges_device;
  torch::Tensor types_device;
  if (reuse_sib_geometry)
  {
    pos_device = sib_cached_pos_device.detach();
    edges_device = sib_cached_edges_device.detach();
    types_device = sib_cached_types_device.detach();
    if constexpr (Staged)
      spin_device = host_staging.upload_spin(spin_tensor.size(0));
    else
      spin_device = spin_tensor.to(device);
  }
  else if constexpr (Staged)
  {
    SpingnnStagedInputs staged =
        host_staging.upload_inputs(pos_tensor.size(0), edges_tensor.size(1), true);
    pos_device = staged.pos;
    spin_device = staged.spin;
    edges_device = staged.edges;
    types_device = staged.types;
  }
  else
  {
    pos_device = pos_tensor.to(device);
    spin_device = spin_tensor.to(device);
    edges_device = edges_tensor.to(device);
    types_device = types_tensor.to(device);
  }
  // Deterministic reductions also need a fixed edge order: GPU neighbor
  // construction can permute the same graph between otherwise identical runs.
  if (at::globalContext().deterministicAlgorithms() && edges_device.size(1) > 1)
  {
    const auto edge_keys =
        edges_device.select(0, 0) * pos_device.size(0) + edges_device.select(0, 1);
    edges_device = edges_device.index_select(1, torch::argsort(edge_keys));
  }
  if (capture_sib_geometry)
  {
    sib_cached_pos_device = pos_device.detach();
    sib_cached_edges_device = edges_device.detach();
    sib_cached_types_device = types_device.detach();
  }
  if constexpr (Profile) perf->h2d_end_model_start();
  const auto to_cpu = [perf](const torch::Tensor &tensor) {
    if constexpr (Profile) perf->d2h_start();
    torch::Tensor result = tensor.cpu();
    if constexpr (Profile) perf->d2h_end();
    return result;
  };
  ModelOutputs device_result;
  if (use_pt2)
  {
    if (pt2_inputs.capacity() < 4) pt2_inputs.reserve(4);
    pt2_inputs.clear();
    pt2_inputs.push_back(pos_device);
    pt2_inputs.push_back(spin_device);
    pt2_inputs.push_back(edges_device);
    pt2_inputs.push_back(types_device);
    std::vector<at::Tensor> outputs;
    try
    {
      outputs = pt2_model->run(pt2_inputs);
    }
    catch (const std::exception &exception)
    {
      error->one(
          FLERR, "Pair style spin/allegro_spin PT2 execution failed for n_atoms=" +
                     std::to_string(pos_tensor.size(0)) + ", n_edges=" +
                     std::to_string(edges_tensor.size(1)) +
                     " (export contract: n_atoms>=1, n_edges>=2): " + exception.what());
    }
    if constexpr (Profile) perf->model_end();
    if (outputs.size() != pt2_output_indices.size())
      error->one(
          FLERR, "Pair style spin/allegro_spin PT2 returned " +
                     std::to_string(outputs.size()) + " tensors but output_signature declares " +
                     std::to_string(pt2_output_indices.size()));
    const int64_t n_atoms = pos_tensor.size(0);
    device_result.total_energy = checked_pt2_output(
        error, "spin/allegro_spin", "total_energy", outputs,
        pt2_output_indices, {1, 1});
    device_result.forces = checked_pt2_output(
        error, "spin/allegro_spin", "forces", outputs,
        pt2_output_indices, {n_atoms, 3});
    device_result.spin_forces = checked_pt2_output(
        error, "spin/allegro_spin", "spin_forces", outputs,
        pt2_output_indices, {n_atoms, 3});
    device_result.atomic_energy = checked_pt2_output(
        error, "spin/allegro_spin", "atomic_energy", outputs,
        pt2_output_indices, {n_atoms, 1});
    if (need_global_virial)
      device_result.virial = checked_pt2_output(
          error, "spin/allegro_spin", "virial", outputs,
          pt2_output_indices, {1, 3, 3});
    if (need_atom_virial)
      device_result.atom_virial = checked_pt2_output(
          error, "spin/allegro_spin", "atom_virial", outputs,
          pt2_output_indices, {n_atoms, 3, 3});
  }
  else
  {
    model_input.insert_or_assign("pos", pos_device);
    model_input.insert_or_assign("spin", spin_device);
    model_input.insert_or_assign("edge_index", edges_device);
    model_input.insert_or_assign("atom_types", types_device);
    if (model_input_vector.empty()) model_input_vector.emplace_back(model_input);
    auto output = model.forward(model_input_vector).toGenericDict();
    if constexpr (Profile) perf->model_end();
    device_result.total_energy = output.at("total_energy").toTensor();
    device_result.forces = output.at("forces").toTensor();
    device_result.spin_forces = output.at("spin_forces").toTensor();
    device_result.atomic_energy = output.at("atomic_energy").toTensor();
    if (need_global_virial)
      device_result.virial = output.at("virial").toTensor();
    if (need_atom_virial)
      device_result.atom_virial = output.at("atom_virial").toTensor();
  }

  ModelOutputs result;
  if constexpr (Staged)
  {
    const int64_t output_values = device_result.total_energy.numel() +
        device_result.forces.numel() + device_result.spin_forces.numel() +
        device_result.atomic_energy.numel() +
        (device_result.virial.defined() ? device_result.virial.numel() : 0) +
        (device_result.atom_virial.defined() ? device_result.atom_virial.numel() : 0);
    staging_use->note_growth(host_staging.prepare_output(device, output_values));
    int64_t output_offset = 0;
    if constexpr (Profile) perf->d2h_start();
    result.total_energy = host_staging.pack_output(
        error, "spin/allegro_spin", "total_energy", device_result.total_energy,
        output_offset);
    output_offset += device_result.total_energy.numel();
    result.forces = host_staging.pack_output(
        error, "spin/allegro_spin", "forces", device_result.forces, output_offset);
    output_offset += device_result.forces.numel();
    result.spin_forces = host_staging.pack_output(
        error, "spin/allegro_spin", "spin_forces", device_result.spin_forces,
        output_offset);
    output_offset += device_result.spin_forces.numel();
    result.atomic_energy = host_staging.pack_output(
        error, "spin/allegro_spin", "atomic_energy", device_result.atomic_energy,
        output_offset);
    output_offset += device_result.atomic_energy.numel();
    if (device_result.virial.defined())
    {
      result.virial = host_staging.pack_output(
          error, "spin/allegro_spin", "virial", device_result.virial, output_offset);
      output_offset += device_result.virial.numel();
    }
    if (device_result.atom_virial.defined())
    {
      result.atom_virial = host_staging.pack_output(
          error, "spin/allegro_spin", "atom_virial", device_result.atom_virial,
          output_offset);
      output_offset += device_result.atom_virial.numel();
    }
    host_staging.download_output(device, output_offset);
    if constexpr (Profile) perf->d2h_end();
  }
  else
  {
    result.total_energy = to_cpu(device_result.total_energy);
    result.forces = to_cpu(device_result.forces);
    result.spin_forces = to_cpu(device_result.spin_forces);
    result.atomic_energy = to_cpu(device_result.atomic_energy);
    if (device_result.virial.defined()) result.virial = to_cpu(device_result.virial);
    if (device_result.atom_virial.defined())
      result.atom_virial = to_cpu(device_result.atom_virial);
  }
  return result;
}

PairAllegro_spin::ModelOutputs PairAllegro_spin::run_model(
    const torch::Tensor &pos_tensor, const torch::Tensor &spin_tensor,
    const torch::Tensor &edges_tensor, const torch::Tensor &types_tensor,
    bool need_global_virial, bool need_atom_virial)
{
  return run_model_impl<false, false>(
      pos_tensor, spin_tensor, edges_tensor, types_tensor,
      need_global_virial, need_atom_virial, false, false, nullptr, nullptr);
}

// Force and energy computation
template <bool Profile, bool Staged>
void PairAllegro_spin::compute_impl(int eflag, int vflag)
{
  SpingnnPerfContext<Profile> perf(
      perf_state, device, comm->me, static_cast<long long>(update->ntimestep),
      "spin/allegro_spin", use_pt2 ? "pt2" : "torchscript");
  ev_init(eflag, vflag);
  perf.graph_start();

  // Get info from lammps:

  // Atom positions, including ghost atoms
  double **x = atom->x;
  // Atom forces
  double **f = atom->f;
  // Add Atom spin
  double **fm = atom->fm; // force on spin
  double **sp = atom->sp; // spin vector
  // Atom IDs, unique, reproducible, the "real" indices
  // Probably 1-based
  tagint *tag = atom->tag;
  // Atom types, 1-based
  int *type = atom->type;
  // Number of local/real atoms
  int nlocal = atom->nlocal;
  // Whether Newton is on (i.e. reverse "communication" of forces on ghost atoms).
  // Should be on.
  int newton_pair = force->newton_pair;

  // Number of local/real atoms
  int inum = list->inum;
  assert(inum == nlocal); // This should be true, if my understanding is correct
  // Number of ghost atoms
  int nghost = list->gnum;
  // Total number of atoms
  int ntotal = inum + nghost;

  const long long current_step = static_cast<long long>(update->ntimestep);
  const bool capture_requested =
      sib_geometry_cache_enabled && sib_geometry_hint_step == current_step;
  if (sib_geometry_hint_step < current_step)
    sib_geometry_hint_step = std::numeric_limits<long long>::min();
  bool sib_geometry_attempted = false;
  bool sib_geometry_hit = false;
  bool capture_geometry = false;
  if (capture_requested)
  {
    sib_geometry_hint_step = std::numeric_limits<long long>::min();
    sib_geometry_cache_valid = false;
    sib_geometry_attempted = true;
    capture_geometry = sib_geometry_cache_eligible();
  }
  else if (sib_geometry_cache_eligible() && sib_geometry_cache_valid &&
           sib_geometry_generation.timestep == current_step)
  {
    sib_geometry_attempted = true;
    sib_geometry_hit = sib_geometry_matches(ntotal, list->ilist, x, tag, type);
    sib_geometry_cache_valid = false;
  }
  else if (sib_geometry_cache_valid &&
           sib_geometry_generation.timestep != current_step)
    sib_geometry_cache_valid = false;

  if (contribution_nmax < atom->nmax)
  {
    contribution_nmax = atom->nmax;
    memory->grow(force_contribution, contribution_nmax, 3,
                 "pair/allegro_spin:force_contribution");
    memory->grow(spin_force_contribution, contribution_nmax, 3,
                 "pair/allegro_spin:spin_force_contribution");
  }
  if (spin_force_reduced_nmax < atom->nmax)
  {
    spin_force_reduced_nmax = atom->nmax;
    memory->grow(spin_force_reduced, spin_force_reduced_nmax, 3,
                 "pair/allegro_spin:spin_force_reduced");
  }
  // Mapping from neigh list ordering to x/f ordering
  int *ilist = list->ilist;
  // Number of neighbors per atom
  int *numneigh = list->numneigh;
  // Neighbor list per atom
  int **firstneigh = list->firstneigh;

  // Total number of bonds (sum of number of neighbors)
  int nedges = sib_geometry_hit ? sib_geometry_generation.nedges : 0;

  // Number of bonds per atom
  std::vector<int> legacy_neigh_per_atom;
  std::vector<int> legacy_cumsum_neigh_per_atom;
  int *neigh_per_atom = nullptr;
  int *cumsum_neigh_per_atom = nullptr;
  bool graph_scratch_grew = false;
  if (!sib_geometry_hit)
  {
    if constexpr (Staged)
    {
      graph_scratch_grew = host_staging.prepare_graph_scratch(atom->nmax, nlocal);
      neigh_per_atom = host_staging.neighbor_counts();
      cumsum_neigh_per_atom = host_staging.neighbor_offsets();
    }
    else
    {
      legacy_neigh_per_atom.assign(nlocal, 0);
      legacy_cumsum_neigh_per_atom.assign(nlocal, 0);
      neigh_per_atom = legacy_neigh_per_atom.data();
      cumsum_neigh_per_atom = legacy_cumsum_neigh_per_atom.data();
    }
  }

  // checking size of emag

  if (nlocal_max < nlocal)
  { // grow emag lists if necessary
    nlocal_max = nlocal;
    memory->grow(emag, nlocal_max, "pair/spin:emag");
  }

  if (!sib_geometry_hit)
  {
#pragma omp parallel for reduction(+ \
                                   : nedges)
    for (int ii = 0; ii < nlocal; ii++)
    {
      int i = ilist[ii];

      int jnum = numneigh[i];
      int *jlist = firstneigh[i];
      for (int jj = 0; jj < jnum; jj++)
      {
        int j = jlist[jj];
        j &= NEIGHMASK;

        double dx = x[i][0] - x[j][0];
        double dy = x[i][1] - x[j][1];
        double dz = x[i][2] - x[j][2];

        double rsq = dx * dx + dy * dy + dz * dz;
        if (rsq < cutoff * cutoff)
        {
          neigh_per_atom[ii]++;
          nedges++;
        }

        if (ii < inum)
          emag[i] = 0.0;
      }
    }

    // Cumulative sum of neighbors, for knowing where to fill in the edges tensor
    for (int ii = 1; ii < nlocal; ii++)
    {
      cumsum_neigh_per_atom[ii] =
          cumsum_neigh_per_atom[ii - 1] + neigh_per_atom[ii - 1];
    }
  }

  perf.graph_end_pack_start();

  SpingnnStagingUse staging_use;
  torch::Tensor pos_tensor;
  torch::Tensor spin_tensor;
  torch::Tensor edges_tensor;
  torch::Tensor ij2type_tensor;
  if constexpr (Staged)
  {
    staging_use = host_staging.prepare_inputs(
        device, atom->nmax, ntotal, nedges, true,
        {static_cast<long long>(neighbor->ncalls),
         static_cast<long long>(neighbor->lastcall), atom->nlocal,
         atom->nghost, atom->nmax});
    staging_use.note_growth(graph_scratch_grew);
    pos_tensor = host_staging.host_pos(ntotal);
    spin_tensor = host_staging.host_spin(ntotal);
    edges_tensor = host_staging.host_edges(nedges);
    ij2type_tensor = host_staging.host_types(ntotal);
    spin_tensor.zero_();
    if (!sib_geometry_hit)
    {
      pos_tensor.zero_();
      edges_tensor.zero_();
      ij2type_tensor.zero_();
    }
  }
  else
  {
    spin_tensor = torch::zeros({ntotal, 3});
    if (sib_geometry_hit)
    {
      pos_tensor = sib_cached_pos_device;
      edges_tensor = sib_cached_edges_device;
      ij2type_tensor = sib_cached_types_device;
    }
    else
    {
      pos_tensor = torch::zeros({ntotal, 3});
      edges_tensor =
          torch::zeros({2, nedges}, torch::TensorOptions().dtype(torch::kInt64));
      ij2type_tensor =
          torch::zeros({ntotal}, torch::TensorOptions().dtype(torch::kInt64));
    }
  }

  // Loop over atoms and neighbors,
  // store edges and _cell_shifts
  // ii follows the order of the neighbor lists,
  // i follows the order of x, f, etc.
  auto spin = spin_tensor.accessor<float, 2>();
  if (sib_geometry_hit)
  {
#pragma omp parallel for
    for (int ii = 0; ii < ntotal; ii++)
    {
      const int i = ilist[ii];
      if (sp[i][3] > 0)
      {
        spin[i][0] = sp[i][0];
        spin[i][1] = sp[i][1];
        spin[i][2] = sp[i][2];
      }
      else
      {
        spin[i][0] = 0;
        spin[i][1] = 0;
        spin[i][2] = 0;
      }
    }
  }
  else
  {
    auto pos = pos_tensor.accessor<float, 2>();
    auto edges = edges_tensor.accessor<long, 2>();
    auto ij2type = ij2type_tensor.accessor<long, 1>();
    if (debug_mode)
      printf("Allegro_spin edges: i j rij\n");
#pragma omp parallel for
    for (int ii = 0; ii < ntotal; ii++)
    {
      int i = ilist[ii];
      int itag = tag[i];
      int itype = type[i];

      ij2type[i] = type_mapper[itype - 1];

      pos[i][0] = x[i][0];
      pos[i][1] = x[i][1];
      pos[i][2] = x[i][2];

      if (sp[i][3] > 0)
      {
        spin[i][0] = sp[i][0];
        spin[i][1] = sp[i][1];
        spin[i][2] = sp[i][2];
      }
      else
      {
        spin[i][0] = 0;
        spin[i][1] = 0;
        spin[i][2] = 0;
      }

      // if (debug_mode)
      //   printf("spin %.10g %.10g %.10g\n", spin[i][0], spin[i][1], spin[i][2]);

      if (ii >= nlocal)
      {
        continue;
      }

      int jnum = numneigh[i];
      int *jlist = firstneigh[i];

      int edge_counter = cumsum_neigh_per_atom[ii];
      for (int jj = 0; jj < jnum; jj++)
      {
        int j = jlist[jj];
        j &= NEIGHMASK;
        int jtag = tag[j];

        double dx = x[i][0] - x[j][0];
        double dy = x[i][1] - x[j][1];
        double dz = x[i][2] - x[j][2];

        double rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutoff * cutoff)
        {
          continue;
        }

        // TODO: double check order
        edges[0][edge_counter] = i;
        edges[1][edge_counter] = j;

        edge_counter++;

        if (debug_mode) printf("%d %d %.10g\n", itag - 1, jtag - 1, sqrt(rsq));
      }
    }
    if (debug_mode)
      printf("end Allegro_spin edges\n");
  }

  perf.pack_end();
  ModelOutputs model_output;
  if constexpr (Profile)
    model_output = run_model_impl<true, Staged>(
        pos_tensor, spin_tensor, edges_tensor, ij2type_tensor, vflag_global,
        vflag_atom || cvflag_atom, sib_geometry_hit, capture_geometry,
        &perf, Staged ? &staging_use : nullptr);
  else
    model_output = run_model_impl<false, Staged>(
        pos_tensor, spin_tensor, edges_tensor, ij2type_tensor, vflag_global,
        vflag_atom || cvflag_atom, sib_geometry_hit, capture_geometry,
        nullptr, Staged ? &staging_use : nullptr);
  if (capture_geometry)
    capture_sib_geometry(ntotal, nghost, nedges, ilist, x, tag, type);
  torch::Tensor &forces_tensor = model_output.forces;
  torch::Tensor &spin_forces_tensor = model_output.spin_forces;
  torch::Tensor &atomic_energy_tensor = model_output.atomic_energy;
  torch::Tensor &virial_tensor = model_output.virial;
  torch::Tensor &atom_virial_tensor = model_output.atom_virial;

  perf.writeback_start();
  auto forces = forces_tensor.accessor<float, 2>();

  auto spin_forces = spin_forces_tensor.accessor<float, 2>();

  // torch::Tensor total_energy_tensor = output.at("total_energy").toTensor().cpu(); WRONG WITH MPI

  auto atomic_energies = atomic_energy_tensor.accessor<float, 2>();
  perf.writeback_end();
  if (vflag_global)
  {
    perf.writeback_start();
    auto v = virial_tensor.accessor<float, 3>();
    // Convert from 3x3 symmetric tensor format, which NequIP outputs, to the flattened form LAMMPS expects
    // First [0] index on v is batch
    virial[0] += v[0][0][0];
    virial[1] += v[0][1][1];
    virial[2] += v[0][2][2];
    virial[3] += v[0][0][1];
    virial[4] += v[0][0][2];
    virial[5] += v[0][1][2];
    perf.writeback_end();
  }
  // std::cout << "Total energy: " << model_output.total_energy << "\n";
  // std::cout << "atomic energy shape: " << atomic_energy_tensor.sizes()[0] << "," << atomic_energy_tensor.sizes()[1] << std::endl;
  // std::cout << "atomic energies: " << atomic_energy_tensor << std::endl;

  // Write forces and per-atom energies (0-based tags here)

  perf.writeback_start();
  eng_vdwl = 0.0;
  // const double hbar = hbar;
#pragma omp parallel for reduction(+ \
                                   : eng_vdwl)
  for (int ii = 0; ii < ntotal; ii++)
  {
    int i = ilist[ii];

    force_contribution[i][0] = forces[i][0];
    force_contribution[i][1] = forces[i][1];
    force_contribution[i][2] = forces[i][2];
    f[i][0] += force_contribution[i][0];
    f[i][1] += force_contribution[i][1];
    f[i][2] += force_contribution[i][2];

    if (sp[i][3] > 0)
    {
      spin_force_contribution[i][0] = spin_forces[i][0] / hbar;
      spin_force_contribution[i][1] = spin_forces[i][1] / hbar;
      spin_force_contribution[i][2] = spin_forces[i][2] / hbar;
    }
    else
    {
      spin_force_contribution[i][0] = 0.0;
      spin_force_contribution[i][1] = 0.0;
      spin_force_contribution[i][2] = 0.0;
    }
    spin_force_reduced[i][0] = spin_force_contribution[i][0];
    spin_force_reduced[i][1] = spin_force_contribution[i][1];
    spin_force_reduced[i][2] = spin_force_contribution[i][2];
    if (eflag_atom && ii < inum)
      eatom[i] = atomic_energies[i][0];
    if (ii < inum)
      eng_vdwl += atomic_energies[i][0];
    if (ii < inum)
      emag[i] = atomic_energies[i][0];

    if (debug_mode)
    {
      std::cout << "Allegro model output:\n";
      std::cout << "forces: " << forces_tensor << "\n";
      std::cout << "atomic_energy: " << atomic_energy_tensor << "\n";
      if (vflag_global)
        std::cout << "virial: " << virial_tensor << std::endl;
      // BEC tensors are a persistent atom property loaded by read_bec;
      // the spin-only model is not required to provide them.
    }
    // if (debug_mode)
    // {
    //   if (ii < inum)
    //   {
    //     printf("i  %d\n", i);
    //     printf("atomic_energies  %.10g\n", atomic_energies[i][0]);
    //     printf("pos              %.10g %.10g %.10g\n", pos[i][0], pos[i][1], pos[i][2]);
    //     printf("f           %.10g %.10g %.10g\n", f[i][0], f[i][1], f[i][2]);
    //     printf("import sp        %.10g %.10g %.10g\n", sp[i][0], sp[i][1], sp[i][2]);
    //     printf("model spin       %.10g %.10g %.10g\n", spin[i][0], spin[i][1], spin[i][2]);
    //     printf("fm      %.10g %.10g %.10g\n", fm[i][0], fm[i][1], fm[i][2]);
    //     printf("virial      %.10g %.10g %.10g %.10g %.10g %.10g\n", virial[0], virial[1], virial[2],
    //            virial[3], virial[4], virial[5]);
    //   }
    // }
  }
  perf.writeback_end();

  perf.reverse_start();
  comm->reverse_comm_pair(this);
  perf.reverse_end();
  perf.writeback_start();
  for (int ii = 0; ii < inum; ++ii)
  {
    const int i = ilist[ii];
    fm[i][0] += spin_force_reduced[i][0];
    fm[i][1] += spin_force_reduced[i][1];
    fm[i][2] += spin_force_reduced[i][2];
  }
  if (comm->me == 0) eng_vdwl += constant_energy_offset;
  perf.writeback_end();

  if (vflag_atom || cvflag_atom)
  {
    perf.writeback_start();
    auto atomic_virial = atom_virial_tensor.accessor<float, 3>();
    for (int ii = 0; ii < ntotal; ii++)
    {
      int i = ilist[ii];
      if (vflag_atom)
      {
        vatom[i][0] += atomic_virial[i][0][0]; // xx
        vatom[i][1] += atomic_virial[i][1][1]; // yy
        vatom[i][2] += atomic_virial[i][2][2]; // zz
        vatom[i][3] += atomic_virial[i][0][1]; // xy
        vatom[i][4] += atomic_virial[i][0][2]; // xz
        vatom[i][5] += atomic_virial[i][1][2]; // yz
      }
      if (cvflag_atom)
      {
        cvatom[i][0] += atomic_virial[i][0][0]; // xx
        cvatom[i][1] += atomic_virial[i][1][1]; // yy
        cvatom[i][2] += atomic_virial[i][2][2]; // zz
        cvatom[i][3] += atomic_virial[i][0][1]; // xy
        cvatom[i][4] += atomic_virial[i][0][2]; // xz
        cvatom[i][5] += atomic_virial[i][1][2]; // yz
        cvatom[i][6] += atomic_virial[i][1][0]; // yx
        cvatom[i][7] += atomic_virial[i][2][0]; // zx
        cvatom[i][8] += atomic_virial[i][2][1]; // zy
      }
    }
    perf.writeback_end();
  }

  const long long geometry_h2d_bytes = spingnn_tensor_bytes(pos_tensor) +
      spingnn_tensor_bytes(edges_tensor) + spingnn_tensor_bytes(ij2type_tensor);
  const long long sib_geom_h2d_saved =
      sib_geometry_hit ? geometry_h2d_bytes : 0LL;
  const long long h2d_bytes = spingnn_tensor_bytes(spin_tensor) +
      (sib_geometry_hit ? 0LL : geometry_h2d_bytes);
  const long long d2h_bytes = spingnn_tensor_bytes(model_output.total_energy) +
      spingnn_tensor_bytes(forces_tensor) + spingnn_tensor_bytes(spin_forces_tensor) +
      spingnn_tensor_bytes(atomic_energy_tensor) + spingnn_tensor_bytes(virial_tensor) +
      spingnn_tensor_bytes(atom_virial_tensor);
  perf.finish(
      nlocal, nghost, nedges, vflag_global, vflag_atom || cvflag_atom,
      false, h2d_bytes, d2h_bytes,
      Staged ? staging_use.cache_hit() : 0,
      Staged ? staging_use.cache_miss() : 0,
      sib_geometry_hit ? 1 : 0,
      sib_geometry_attempted && !sib_geometry_hit ? 1 : 0,
      sib_geom_h2d_saved);
}

void PairAllegro_spin::compute(int eflag, int vflag)
{
  if (perf_state.enabled())
  {
    if (host_staging.active(device))
      compute_impl<true, true>(eflag, vflag);
    else
      compute_impl<true, false>(eflag, vflag);
  }
  else
  {
    if (host_staging.active(device))
      compute_impl<false, true>(eflag, vflag);
    else
      compute_impl<false, false>(eflag, vflag);
  }
}

void PairAllegro_spin::compute_single_pair(int ii, double fmi[3])
{
  fmi[0] += spin_force_reduced[ii][0];
  fmi[1] += spin_force_reduced[ii][1];
  fmi[2] += spin_force_reduced[ii][2];
}

int PairAllegro_spin::pack_reverse_comm(int n, int first, double *buffer)
{
  int offset = 0;
  for (int i = first; i < first + n; ++i)
    for (int axis = 0; axis < 3; ++axis)
      buffer[offset++] = spin_force_reduced[i][axis];
  return offset;
}

void PairAllegro_spin::unpack_reverse_comm(
    int n, int *indices, double *buffer)
{
  int offset = 0;
  for (int ii = 0; ii < n; ++ii)
    for (int axis = 0; axis < 3; ++axis)
      spin_force_reduced[indices[ii]][axis] += buffer[offset++];
}
