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

#include <pair_nequip_spin.h>
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
#include "nequip_communication.h"
#include "potential_file_reader.h"
#include "pt2_loader.h"
#include "tokenizer.h"
#include "torch_determinism.h"
#include "torch_device.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <array>
#include <cstring>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
// #include <c10/cuda/CUDACachingAllocator.h>

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

PairNEQUIP_spin::PairNEQUIP_spin(LAMMPS *lmp) : PairSpin(lmp)
{
  restartinfo = 0; 
  manybody_flag = 1;

  device = spingnn_torch_device(world, error);
  std::cout << "NEQUIP is using device " << device << "\n";

  if (const char *env_p = std::getenv("NEQUIP_DEBUG"))
  {
    std::cout << "PairNEQUIP_spin is in DEBUG mode, since NEQUIP_DEBUG is in env\n";
    debug_mode = 1;
  }
}

PairNEQUIP_spin::~PairNEQUIP_spin()
{
  pt2_model.reset();
  memory->destroy(force_contribution);
  memory->destroy(spin_force_contribution);
  if (allocated)
  {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(type_mapper);
    memory->destroy(emag);
  }
}

void PairNEQUIP_spin::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style NEQUIP_spin requires atom IDs");

  // need a full neighbor list
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;

  // NequIP message passing needs neighborhoods for visible ghost centers so
  // that forces on owned atoms include neighboring atomic-energy terms.
  neighbor->requests[irequest]->ghost = 1;

  // TODO: I think Newton should be off, enforce this.
  // The network should just directly compute the total forces
  // on the "real" atoms, with no need for reverse "communication".
  // May not matter, since f[j] will be 0 for the ghost atoms anyways.
  if (force->newton_pair == 1)
    error->all(FLERR, "Pair style NEQUIP requires newton pair off");
  configure_nequip_communication(
      error, comm, neighbor, num_interactions, cutoff, "spin/nequip_spin");
}

double PairNEQUIP_spin::init_one(int i, int j)
{
  return cutoff;
}

void PairNEQUIP_spin::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(type_mapper, n + 1, "pair:type_mapper");
}

void PairNEQUIP_spin::settings(int narg, char ** /*arg*/)
{
  // "flare" should be the only word after "pair_style" in the input file.
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style command");
}

void PairNEQUIP_spin::coeff(int narg, char **arg)
{

  if (!allocated)
    allocate();

  int ntypes = atom->ntypes;

  // Should be exactly 3 arguments following "pair_coeff" in the input file.
  if (narg != (3 + ntypes))
    error->all(FLERR, "Incorrect args for pair coefficients");

  // Ensure I,J args are "* *".
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      setflag[i][j] = 0;

  // Parse the definition of each atom type
  char **elements = new char *[ntypes + 1];
  for (int i = 1; i <= ntypes; i++)
  {
    elements[i] = new char[strlen(arg[i + 2]) + 1];
    strcpy(elements[i], arg[i + 2]);
    if (screen)
      fprintf(screen, "NequIP Coeff: type %d is element %s\n", i, elements[i]);
  }

  // Initiate type mapper
  for (int i = 1; i <= ntypes; i++)
  {
    type_mapper[i] = -1;
  }

  const std::string model_path(arg[2]);
  std::cout << "Loading model from " << model_path << "\n";

  std::unordered_map<std::string, std::string> torchscript_metadata = {
      {"config", ""},
      {"nequip_version", ""},
      {"r_max", ""},
      {"nequip_num_interactions", ""},
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
      {"has_bec", ""}};
  DeployMetadata metadata;
  use_pt2 = is_pt2_package_path(model_path);
  pt2_model.reset();
  pt2_input_signature.clear();
  pt2_output_indices.clear();
  if (use_pt2)
  {
    const std::vector<std::string> expected_inputs = {
        "pos", "spin", "edge_index", "edge_cell_shift", "cell", "pbc",
        "atom_types"};
    LoadedPT2Model loaded = load_verified_pt2_model(
        error, model_path, "spin/nequip_spin", device, expected_inputs);
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
          "The indicated TorchScript file does not appear to be a deployed NequIP model; "
          "did you forget to run `nequip-deploy`?");
    metadata = DeployMetadata(
        torchscript_metadata.begin(), torchscript_metadata.end());
  }

  std::vector<std::string> required_outputs = {
      "forces", "spin_forces", "total_energy", "atomic_energy", "virial"};
  if (use_pt2 && comm->nprocs > 1) required_outputs.push_back("atom_virial");
  if (atom->bec_flag)
    required_outputs.push_back("born_effective_charges");
  const std::vector<std::string> available_inputs = {
      "pos", "spin", "edge_index", "edge_cell_shift", "cell", "pbc", "atom_types"};
  validate_deploy_metadata(
      error, metadata, "spin/nequip_spin", required_outputs, available_inputs);

  // If the model is not already frozen, we should freeze it:
  // This is the check used by PyTorch: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/api/module.cpp#L476
  if (!use_pt2 && model.hasattr("training"))
  {
    std::cout << "Freezing TorchScript model...\n";
#ifdef DO_TORCH_FREEZE_HACK
    // Do the hack
    // Copied from the implementation of torch::jit::freeze,
    // except without the broken check
    // See https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp
    bool optimize_numerics = true; // the default
    // the {} is preserved_attrs
    auto out_mod = freeze_module(
        model, {});
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
    // Set JIT bailout to avoid long recompilations for many steps.
    size_t jit_bailout_depth;
    if (metadata["_jit_bailout_depth"].empty())
      jit_bailout_depth = 2;
    else
      jit_bailout_depth = std::stoi(metadata["_jit_bailout_depth"]);
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
        strategy.push_back(
            {fusion_type == "STATIC" ? torch::jit::FusionBehavior::STATIC
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
    std::cout << "Pair style spin/nequip_spin deterministic algorithms enabled "
                 "(SPINGNN_DETERMINISTIC=1)\n";

  // std::cout << "Information from model: " << metadata.size() << " key-value pairs\n";
  // for( const auto& n : metadata ) {
  //   std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
  // }

  cutoff = std::stod(metadata["r_max"]);
  num_interactions = read_nequip_num_interactions(error, metadata, "spin/nequip_spin");

  // match the type names in the pair_coeff to the metadata
  // to construct a type mapper from LAMMPS type to NequIP atom_types
  int n_species = std::stod(metadata["n_species"]);
  std::stringstream ss;
  ss << metadata["type_names"];
  for (int i = 0; i < n_species; i++)
  {
    char ele[100];
    ss >> ele;
    for (int itype = 1; itype <= ntypes; itype++)
      if (strcmp(elements[itype], ele) == 0)
        type_mapper[itype] = i;
  }

  // set setflag i,j for type pairs where both are mapped to elements
  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      if ((type_mapper[i] >= 0) && (type_mapper[j] >= 0))
        setflag[i][j] = 1;

  calibrate_energy_offset();

  if (elements)
  {
    for (int i = 1; i <= ntypes; i++)
      if (elements[i])
        delete[] elements[i];
    delete[] elements;
  }
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

void PairNEQUIP_spin::calibrate_energy_offset()
{
  // three type-0 atoms on a line; every pair is within the cutoff.  The
  // probe must be periodic with a non-singular cell: deployed models that
  // compute stress raise "Stress is undefined for non-periodic structures"
  // otherwise.
  const double spacing = std::min(1.5, 0.35 * cutoff);
  torch::Tensor pos = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat));
  auto pos_a = pos.accessor<float, 2>();
  pos_a[1][0] = spacing;
  pos_a[2][0] = 2.0 * spacing;
  torch::Tensor spin = torch::zeros({3, 3}, pos.options());
  torch::Tensor edges = torch::tensor(
      {{0, 0, 1, 1, 2, 2}, {1, 2, 0, 2, 0, 1}},
      torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor edge_cell_shifts = torch::zeros({6, 3}, pos.options());
  torch::Tensor cell = torch::zeros({3, 3}, pos.options());
  auto cell_a = cell.accessor<float, 2>();
  cell_a[0][0] = cell_a[1][1] = cell_a[2][2] =
      static_cast<float>(8.0 * spacing);
  torch::Tensor pbc = torch::ones(
      {1, 3}, torch::TensorOptions().dtype(torch::kBool));
  torch::Tensor types = torch::zeros({3}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor batch = torch::zeros({3}, torch::TensorOptions().dtype(torch::kInt64));

  ModelOutputs probe =
      run_model(pos, spin, edges, edge_cell_shifts, cell, pbc, types, batch,
                false, false);
  constant_energy_offset =
      static_cast<double>(probe.total_energy.item<float>()) -
      static_cast<double>(probe.atomic_energy.sum().item<float>());
  if (!std::isfinite(constant_energy_offset))
    error->one(FLERR, "Pair style spin/nequip_spin probe evaluation of the "
                      "total-energy offset was not finite");
  if (comm->me == 0 && std::fabs(constant_energy_offset) > 1.0e-6)
    utils::logmesg(lmp, "NequIP_spin: constant total-energy offset "
                        "{:.6g} eV (total_energy - sum(atomic_energy) on a "
                        "ghost-free probe graph)\n", constant_energy_offset);
}

void *PairNEQUIP_spin::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str, "cut") == 0)
    return (void *)&cutoff;
  return nullptr;
}

PairNEQUIP_spin::ModelOutputs PairNEQUIP_spin::run_model(
    const torch::Tensor &pos_tensor, const torch::Tensor &spin_tensor,
    const torch::Tensor &edges_tensor,
    const torch::Tensor &edge_cell_shifts_tensor,
    const torch::Tensor &cell_tensor, const torch::Tensor &pbc_tensor,
    const torch::Tensor &types_tensor, const torch::Tensor &batch_tensor,
    bool need_virial, bool need_bec)
{
  const torch::Tensor pos_device = pos_tensor.to(device);
  const torch::Tensor spin_device = spin_tensor.to(device);
  const torch::Tensor edges_device = edges_tensor.to(device);
  const torch::Tensor edge_cell_shifts_device = edge_cell_shifts_tensor.to(device);
  const torch::Tensor cell_device = cell_tensor.to(device);
  const torch::Tensor pbc_device = pbc_tensor.to(device);
  const torch::Tensor types_device = types_tensor.to(device);
  const torch::Tensor batch_device = batch_tensor.to(device);
  ModelOutputs result;
  if (use_pt2)
  {
    std::unordered_map<std::string, torch::Tensor> named_inputs = {
        {"pos", pos_device},
        {"spin", spin_device},
        {"edge_index", edges_device},
        {"edge_cell_shift", edge_cell_shifts_device},
        {"cell", cell_device},
        {"pbc", pbc_device},
        {"atom_types", types_device},
        // Not part of the PT2 input ABI yet; only signature names are read.
        {"batch", batch_device}};
    std::vector<at::Tensor> inputs;
    for (const auto &name : pt2_input_signature)
      inputs.push_back(named_inputs.at(name));
    std::vector<at::Tensor> outputs;
    try
    {
      outputs = pt2_model->run(inputs);
    }
    catch (const std::exception &exception)
    {
      error->one(
          FLERR, "Pair style spin/nequip_spin PT2 execution failed for n_atoms=" +
                     std::to_string(pos_tensor.size(0)) + ", n_edges=" +
                     std::to_string(edges_tensor.size(1)) +
                     " (export contract: n_atoms>=1, n_edges>=2): " + exception.what());
    }
    if (outputs.size() != pt2_output_indices.size())
      error->one(
          FLERR, "Pair style spin/nequip_spin PT2 returned " +
                     std::to_string(outputs.size()) + " tensors but output_signature declares " +
                     std::to_string(pt2_output_indices.size()));
    const int64_t n_atoms = pos_tensor.size(0);
    result.forces = checked_pt2_output(
        error, "spin/nequip_spin", "forces", outputs,
        pt2_output_indices, {n_atoms, 3}).cpu();
    result.spin_forces = checked_pt2_output(
        error, "spin/nequip_spin", "spin_forces", outputs,
        pt2_output_indices, {n_atoms, 3}).cpu();
    result.total_energy = checked_pt2_output(
        error, "spin/nequip_spin", "total_energy", outputs,
        pt2_output_indices, {1, 1}).cpu();
    result.atomic_energy = checked_pt2_output(
        error, "spin/nequip_spin", "atomic_energy", outputs,
        pt2_output_indices, {n_atoms, 1}).cpu();
    if (need_virial) {
      if (comm->nprocs > 1) {
        // The periodic PT2 ABI has no batch input. Reduce the existing
        // edge-centered atom virials by ownership to match TorchScript's
        // frame-zero virial without extending the input ABI.
        const torch::Tensor atom_virial = checked_pt2_output(
            error, "spin/nequip_spin", "atom_virial", outputs,
            pt2_output_indices, {n_atoms, 3, 3});
        const torch::Tensor owned = batch_device.eq(0).view({n_atoms, 1, 1});
        const torch::Tensor sum = (atom_virial * owned).sum(0, true);
        result.virial = ((sum + sum.transpose(-1, -2)) * 0.5).cpu();
      } else {
        result.virial = checked_pt2_output(
            error, "spin/nequip_spin", "virial", outputs,
            pt2_output_indices, {1, 3, 3}).cpu();
      }
    }
    if (need_bec)
      result.born_effective_charges = checked_pt2_output(
          error, "spin/nequip_spin", "born_effective_charges", outputs,
          pt2_output_indices, {n_atoms, 9}).cpu();
  }
  else
  {
    c10::Dict<std::string, torch::Tensor> input;
    input.insert("pos", pos_device);
    input.insert("spin", spin_device);
    input.insert("edge_index", edges_device);
    input.insert("edge_cell_shift", edge_cell_shifts_device);
    input.insert("cell", cell_device);
    input.insert("pbc", pbc_device);
    input.insert("atom_types", types_device);
    input.insert("batch", batch_device);
    std::vector<torch::IValue> input_vector(1, input);
    auto output = model.forward(input_vector).toGenericDict();
    result.forces = output.at("forces").toTensor().cpu();
    result.spin_forces = output.at("spin_forces").toTensor().cpu();
    result.total_energy = output.at("total_energy").toTensor().cpu();
    result.atomic_energy = output.at("atomic_energy").toTensor().cpu();
    if (need_virial) result.virial = output.at("virial").toTensor().cpu();
    if (need_bec)
      result.born_effective_charges =
          output.at("born_effective_charges").toTensor().cpu();
  }
  return result;
}

// Force and energy computation
void PairNEQUIP_spin::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  // All ranks must participate in Error::all, including empty owners.
  if (vflag_atom)
    error->all(FLERR, "Pair style NEQUIP does not support per-atom virial");
  if (atom->nlocal == 0) {
    eng_vdwl = comm->me == 0 ? constant_energy_offset : 0.0;
    return;
  }

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
  int newton_pair = force->newton_pair;
  // Should probably be off.
  if (newton_pair == 1)
    error->all(FLERR, "Pair style NEQUIP requires 'newton off'");

  // Number of local/real atoms
  int inum = list->inum;
  assert(inum == nlocal); // This should be true, if my understanding is correct
  // Number of local and ghost centers in the requested full neighbor list
  int ncenters = inum + list->gnum;
  // Mapping from neigh list ordering to x/f ordering
  int *ilist = list->ilist;
  // Number of neighbors per atom
  int *numneigh = list->numneigh;
  // Neighbor list per atom
  int **firstneigh = list->firstneigh;

  // checking size of emag

  if (nlocal_max < nlocal)
  { // grow emag lists if necessary
    nlocal_max = nlocal;
    memory->grow(emag, nlocal_max, "pair/spin:emag");
  }
  if (contribution_nmax < atom->nmax)
  {
    contribution_nmax = atom->nmax;
    memory->grow(
        force_contribution, contribution_nmax, 3,
        "pair/nequip_spin:force_contribution");
    memory->grow(
        spin_force_contribution, contribution_nmax, 3,
        "pair/nequip_spin:spin_force_contribution");
  }

  // LAMMPS indices are rank-local and contiguous, while tags are global and
  // may be sparse.  Periodic ghost images can share a tag, so first map every
  // visible LAMMPS index to one contiguous graph node per physical atom.
  const int nvisible = atom->nlocal + atom->nghost;
  std::unordered_map<tagint, int> tag_to_graph;
  tag_to_graph.reserve(nvisible);
  std::vector<int> atom_to_graph(nvisible, -1);
  std::vector<int> graph_to_atom;
  std::vector<char> graph_has_center;
  graph_to_atom.reserve(nvisible);
  graph_has_center.reserve(nvisible);

  auto add_visible_atom = [&](int i, bool has_center) {
    auto inserted = tag_to_graph.emplace(tag[i], tag_to_graph.size());
    int graph_index = inserted.first->second;
    atom_to_graph[i] = graph_index;
    if (inserted.second) {
      graph_to_atom.push_back(i);
      graph_has_center.push_back(has_center);
    } else if (has_center && nequip_prefer_center(
                   graph_to_atom[graph_index], graph_has_center[graph_index],
                   i, nlocal, x, numneigh, firstneigh, cutoff * cutoff)) {
      graph_to_atom[graph_index] = i;
      graph_has_center[graph_index] = 1;
    }
  };

  for (int ii = 0; ii < ncenters; ii++)
    add_visible_atom(ilist[ii], true);
  for (int i = 0; i < nvisible; i++)
    add_visible_atom(i, false);

  const int ngraph = graph_to_atom.size();

  // Split the model input into batch frame 0 (nodes whose representative
  // atom is owned by this rank) and frame 1 (ghost nodes). The model's
  // per-batch virial then gives this rank's owned-edge contribution only,
  // and LAMMPS sums the per-rank virials into the correct global virial.
  // Serial runs have no ghost-owned nodes, so the single all-zero frame
  // reproduces the previous single-graph behavior exactly.
  torch::Tensor batch_tensor =
      torch::zeros({ngraph}, torch::TensorOptions().dtype(torch::kInt64));
  auto node_batch = batch_tensor.accessor<long, 1>();
  bool has_ghost_batch = false;
  for (int graph_i = 0; graph_i < ngraph; graph_i++)
    if (graph_to_atom[graph_i] >= nlocal)
    {
      node_batch[graph_i] = 1;
      has_ghost_batch = true;
    }
  if (nlocal == 0 && has_ghost_batch)
  {
    // Batch frame indices must densely cover frame 0; with no owned atom
    // there is no frame 0, so fall back to a single graph and report no
    // owned virial from this rank.
    batch_tensor.zero_();
    has_ghost_batch = false;
  }
  const bool zero_owned_virial = nlocal == 0;
  std::vector<int> neigh_per_graph(ngraph, 0);
  int nedges = 0;
  for (int graph_i = 0; graph_i < ngraph; graph_i++) {
    if (!graph_has_center[graph_i]) continue;
    int i = graph_to_atom[graph_i];
    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];
      if (dx * dx + dy * dy + dz * dz < cutoff * cutoff) {
        neigh_per_graph[graph_i]++;
        nedges++;
      }
    }
  }

  std::vector<int> edge_offsets(ngraph, 0);
  for (int graph_i = 1; graph_i < ngraph; graph_i++)
    edge_offsets[graph_i] = edge_offsets[graph_i - 1] + neigh_per_graph[graph_i - 1];

  torch::Tensor pos_tensor = torch::zeros({ngraph, 3});
  torch::Tensor spin_tensor = torch::zeros({ngraph, 3});
  torch::Tensor tag2type_tensor = torch::zeros({ngraph}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor edges_tensor = torch::zeros({2, nedges}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor edge_cell_shifts_tensor = torch::zeros({nedges, 3});
  torch::Tensor cell_tensor = torch::zeros({3, 3});
  torch::Tensor pbc_tensor = torch::zeros(
      {1, 3}, torch::TensorOptions().dtype(torch::kBool));

  auto pos = pos_tensor.accessor<float, 2>();
  auto spin = spin_tensor.accessor<float, 2>();
  auto tag2type = tag2type_tensor.accessor<long, 1>();
  auto edges = edges_tensor.accessor<long, 2>();
  auto edge_cell_shifts = edge_cell_shifts_tensor.accessor<float, 2>();
  auto cell = cell_tensor.accessor<float, 2>();
  auto pbc = pbc_tensor.accessor<bool, 2>();

  pbc[0][0] = domain->xperiodic != 0;
  pbc[0][1] = domain->yperiodic != 0;
  pbc[0][2] = domain->zperiodic != 0;

  // Primary-cell positions must agree across ranks for the standard
  // strain-based stress wrapper as well as the edge-virial wrapper.
  std::vector<std::array<double, 3>> canonical_positions(ngraph);
  for (int graph_i = 0; graph_i < ngraph; graph_i++)
  {
    int i = graph_to_atom[graph_i];
    int itype = type[i];
    tag2type[graph_i] = type_mapper[itype];
    auto &canonical = canonical_positions[graph_i];
    std::copy(x[i], x[i] + 3, canonical.begin());
    domain->remap(canonical.data());
    for (int axis = 0; axis < 3; ++axis) pos[graph_i][axis] = canonical[axis];
    if (sp[i][3] > 0)
    {
      spin[graph_i][0] = sp[i][0];
      spin[graph_i][1] = sp[i][1];
      spin[graph_i][2] = sp[i][2];
    }
    else
    {
      spin[graph_i][0] = 0;
      spin[graph_i][1] = 0;
      spin[graph_i][2] = 0;
    }
  }

  // Get cell
  cell[0][0] = domain->boxhi[0] - domain->boxlo[0];

  cell[1][0] = domain->xy;
  cell[1][1] = domain->boxhi[1] - domain->boxlo[1];

  cell[2][0] = domain->xz;
  cell[2][1] = domain->yz;
  cell[2][2] = domain->boxhi[2] - domain->boxlo[2];

  // Loop over atoms and neighbors,
  // store edges and _cell_shifts
  // ii follows the order of the neighbor lists,
  // i follows the order of x, f, etc.
  if (debug_mode)
    printf("NEQUIP_spin edges: i j xi[:] xj[:] cell_shift[:] rij\n");
  for (int graph_i = 0; graph_i < ngraph; graph_i++)
  {
    if (!graph_has_center[graph_i]) continue;
    int i = graph_to_atom[graph_i];

    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    int edge_counter = edge_offsets[graph_i];
    for (int jj = 0; jj < jnum; jj++)
    {
      int j = jlist[jj] & NEIGHMASK;
      int graph_j = atom_to_graph[j];
      if (graph_j < 0)
        error->one(FLERR, "Pair style NEQUIP_spin could not map a ghost atom to the graph");

      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq < cutoff * cutoff)
      {
        nequip_periodic_shift(
            domain, x, i, j, canonical_positions[graph_i].data(),
            canonical_positions[graph_j].data(),
            &edge_cell_shifts[edge_counter][0]);
        edges[0][edge_counter] = graph_i;
        edges[1][edge_counter] = graph_j;
        edge_counter++;

        if (debug_mode)
        {
          printf("%d %d %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n", graph_i, graph_j,
                 pos[graph_i][0], pos[graph_i][1], pos[graph_i][2], pos[graph_j][0], pos[graph_j][1], pos[graph_j][2],
                 edge_cell_shifts[edge_counter-1][0], edge_cell_shifts[edge_counter-1][1], edge_cell_shifts[edge_counter-1][2], sqrt(rsq));
        }
      }
    }
  }
  if (debug_mode)
    printf("end NEQUIP edges\n");

  if (debug_mode)
  {
    std::cout << "NequIP model input:\n";
    std::cout << "pos:\n"
              << pos_tensor << "\n";
    std::cout << "spin:\n"
              << spin_tensor << "\n";
    std::cout << "edge_index:\n"
              << edges_tensor << "\n";
    std::cout << "edge_cell_shifts:\n"
              << edge_cell_shifts_tensor << "\n";
    std::cout << "cell:\n"
              << cell_tensor << "\n";
    std::cout << "pbc:\n"
              << pbc_tensor << "\n";
    std::cout << "atom_types:\n"
              << tag2type_tensor << "\n";
    std::cout << "batch:\n"
              << batch_tensor << "\n";
  }

  ModelOutputs model_output = run_model(
      pos_tensor, spin_tensor, edges_tensor, edge_cell_shifts_tensor,
      cell_tensor, pbc_tensor, tag2type_tensor, batch_tensor, vflag, atom->bec_flag);
  torch::Tensor &forces_tensor = model_output.forces;
  auto forces = forces_tensor.accessor<float, 2>();

  torch::Tensor &spin_forces_tensor = model_output.spin_forces;
  auto spin_forces = spin_forces_tensor.accessor<float, 2>();

  if (atom->bec_flag)
  {
    double **bec = atom->bec;
    torch::Tensor &bec_tensor = model_output.born_effective_charges;
    auto becs_nn = bec_tensor.accessor<float, 2>();

    for (int ii = 0; ii < inum; ii++)
    {
      int i = ilist[ii];
      int graph_i = atom_to_graph[i];
      bec[i][0] = becs_nn[graph_i][0];
      bec[i][1] = becs_nn[graph_i][1];
      bec[i][2] = becs_nn[graph_i][2];
      bec[i][3] = becs_nn[graph_i][3];
      bec[i][4] = becs_nn[graph_i][4];
      bec[i][5] = becs_nn[graph_i][5];
      bec[i][6] = becs_nn[graph_i][6];
      bec[i][7] = becs_nn[graph_i][7];
      bec[i][8] = becs_nn[graph_i][8];
    }
  }

  torch::Tensor &total_energy_tensor = model_output.total_energy;

  torch::Tensor &atomic_energy_tensor = model_output.atomic_energy;
  auto atomic_energies = atomic_energy_tensor.accessor<float, 2>();

  if (vflag)
  {
    torch::Tensor &v_tensor = model_output.virial;
    auto v = v_tensor.accessor<float, 3>();
    // Convert from 3x3 symmetric tensor format, which NequIP outputs, to the flattened form LAMMPS expects
    // First [0] index on v is batch; frame 0 holds the virial of the edges
    // centered on this rank's owned atoms, the only contribution this rank
    // may report because LAMMPS sums per-rank virials into the global one.
    if (zero_owned_virial)
    {
      virial[0] = 0.0;
      virial[1] = 0.0;
      virial[2] = 0.0;
      virial[3] = 0.0;
      virial[4] = 0.0;
      virial[5] = 0.0;
    }
    else
    {
      virial[0] = v[0][0][0];
      virial[1] = v[0][1][1];
      virial[2] = v[0][2][2];
      virial[3] = v[0][0][1];
      virial[4] = v[0][0][2];
      virial[5] = v[0][1][2];
    }
  }
  if (debug_mode)
  {
    std::cout << "NequIP_spin model output:\n";
    std::cout << "forces: " << forces_tensor << "\n";
    std::cout << "spin forces: " << spin_forces_tensor << "\n";
    std::cout << "total_energy: " << total_energy_tensor << "\n";
    std::cout << "atomic_energy: " << atomic_energy_tensor << "\n";
    if (atom->bec_flag)
      std::cout << "born_effective_charges: "
                << model_output.born_effective_charges << std::endl;
    if (vflag)
      std::cout << "virial: " << model_output.virial << std::endl;
  }

  // std::cout << "atomic energy sum: " << atomic_energy_sum << std::endl;
  // std::cout << "Total energy: " << total_energy_tensor << "\n";
  // std::cout << "atomic energy shape: " << atomic_energy_tensor.sizes()[0] << "," << atomic_energy_tensor.sizes()[1] << std::endl;
  // std::cout << "atomic energies: " << atomic_energy_tensor << std::endl;

  // Only owned atoms contribute to this rank's energy and output arrays.
  eng_vdwl = 0.0;
  for (int ii = 0; ii < inum; ii++)
  {
    int i = ilist[ii];
    int graph_i = atom_to_graph[i];
    force_contribution[i][0] = forces[graph_i][0];
    force_contribution[i][1] = forces[graph_i][1];
    force_contribution[i][2] = forces[graph_i][2];
    f[i][0] += force_contribution[i][0];
    f[i][1] += force_contribution[i][1];
    f[i][2] += force_contribution[i][2];
    if (sp[i][3] > 0)
    {
      spin_force_contribution[i][0] = spin_forces[graph_i][0] / hbar;
      spin_force_contribution[i][1] = spin_forces[graph_i][1] / hbar;
      spin_force_contribution[i][2] = spin_forces[graph_i][2] / hbar;
    }
    else
    {
      spin_force_contribution[i][0] = 0.0;
      spin_force_contribution[i][1] = 0.0;
      spin_force_contribution[i][2] = 0.0;
    }
    fm[i][0] += spin_force_contribution[i][0];
    fm[i][1] += spin_force_contribution[i][1];
    fm[i][2] += spin_force_contribution[i][2];
    eng_vdwl += atomic_energies[graph_i][0];
    if (eflag_atom)
      eatom[i] = atomic_energies[graph_i][0];
    emag[i] = atomic_energies[graph_i][0];
  }
  // Offset between the model's total energy and the sum of its atomic
  // energies (e.g. a non-size-extensive global shift), calibrated once on a
  // ghost-free probe graph in coeff() and applied once globally.
  if (comm->me == 0) eng_vdwl += constant_energy_offset;

  /*
  if(device.is_cuda()){
    //torch::cuda::empty_cache();
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
  */
}

void PairNEQUIP_spin::compute_single_pair(int ii, double fmi[3]){
  fmi[0] += spin_force_contribution[ii][0];
  fmi[1] += spin_force_contribution[ii][1];
  fmi[2] += spin_force_contribution[ii][2];
}
