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

#include <pair_allegro.h>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
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

#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>
#include <numeric>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

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

PairAllegro::PairAllegro(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
  centroidstressflag = CENTROID_AVAIL; // Allow the use of the centroid/stress/atom. - Added by Hongyu Yu

  if (const char *env_p = std::getenv("ALLEGRO_DEBUG"))
  {
    std::cout << "PairAllegro is in DEBUG mode, since ALLEGRO_DEBUG is in env\n";
    debug_mode = 1;
  }

  device = spingnn_torch_device(world, error);
  std::cout << "Allegro is using device " << device << "\n";
}

PairAllegro::~PairAllegro()
{
  if (allocated)
  {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairAllegro::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style Allegro requires atom IDs");

  // need a full neighbor list
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;

  neighbor->requests[irequest]->ghost = 1;

  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style Allegro requires newton pair on");
}

double PairAllegro::init_one(int i, int j)
{
  return cutoff;
}

void PairAllegro::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

void PairAllegro::settings(int narg, char ** /*arg*/)
{
  // "allegro" should be the only word after "pair_style" in the input file.
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style command, too many arguments");

  if (strcmp(update->unit_style, "metal") != 0)
    error->all(FLERR, "Pair style Allegro requires metal units because the model assumes eV and Angstrom");

  if (comm->me == 0)
    utils::logmesg(lmp, "Allegro model units: energy=eV, length=Angstrom, force=eV/Angstrom\n");
}

void PairAllegro::coeff(int narg, char **arg)
{
  perf_state.validate(error);
  host_staging.validate(error);
  if (perf_state.enabled() && debug_mode)
    error->all(FLERR, "SPINGNN_PERF and ALLEGRO_DEBUG cannot be enabled together");

  if (!allocated)
    allocate();

  int ntypes = atom->ntypes;

  // Should be exactly 3 arguments following "pair_coeff" in the input file.
  if (narg != (3 + ntypes))
    error->all(FLERR, "Incorrect args for pair coefficients, should be * * <model>.pth|<model>.pt2 <type1> <type2> ... <typen>");

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
  std::cout << "Allegro: Loading model from " << model_path << "\n";

  std::unordered_map<std::string, std::string> torchscript_metadata = {
      {"config", ""},
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
        "pos", "edge_index", "atom_types"};
    LoadedPT2Model loaded = load_verified_pt2_model(
        error, model_path, "allegro", device, expected_inputs);
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
      "total_energy", "forces", "atomic_energy", "virial", "atom_virial"};
  if (atom->bec_flag)
    required_outputs.push_back("born_effective_charges");
  const std::vector<std::string> available_inputs = {
      "pos", "edge_index", "atom_types"};
  validate_deploy_metadata(
      error, metadata, "allegro", required_outputs, available_inputs);

  // If the model is not already frozen, we should freeze it:
  // This is the check used by PyTorch: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/api/module.cpp#L476
  if (!use_pt2 && model.hasattr("training"))
  {
    std::cout << "Allegro: Freezing TorchScript model...\n";
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

  // In PyTorch >=1.11, this is now set_fusion_strategy
  if (!use_pt2)
  {
    torch::jit::FusionStrategy strategy;
    strategy = {{torch::jit::FusionBehavior::DYNAMIC, 10}};
  //strategy = {{torch::jit::FusionBehavior::STATIC, 100}, {torch::jit::FusionBehavior::DYNAMIC, 1}};

  //if (metadata["_jit_fusion_strategy"].empty()) { //TODO: respect model
  //  // This is the default used in the Python code
  //  strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
  //} else {
  //  std::stringstream strat_stream(metadata["_jit_fusion_strategy"]);
  //  std::string fusion_type, fusion_depth;
  //  while(std::getline(strat_stream, fusion_type, ',')) {
  //    std::getline(strat_stream, fusion_depth, ';');
  //    strategy.push_back({fusion_type == "STATIC" ? torch::jit::FusionBehavior::STATIC : torch::jit::FusionBehavior::DYNAMIC, std::stoi(fusion_depth)});
  //  }
  //}
    torch::jit::setFusionStrategy(strategy);
  }

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
        "Pair style allegro deterministic algorithms enabled "
        "(SPINGNN_DETERMINISTIC=1)\n");

  // std::cout << "Allegro: Information from model: " << metadata.size() << " key-value pairs\n";
  // for( const auto& n : metadata ) {
  //   std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
  // }

  cutoff = std::stod(metadata["r_max"]);

  type_mapper.assign(ntypes, -1);
  std::stringstream ss;
  int n_species = std::stod(metadata["n_species"]);
  ss << metadata["type_names"];
  std::cout << "Type mapping:"
            << "\n";
  std::cout << "Allegro type | Allegro name | LAMMPS type | LAMMPS name"
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
      error->all(FLERR, "Pair style Allegro could not map LAMMPS atom type {} named {} to a model species",
                 itype, arg[itype + 2]);

  // set setflag i,j for type pairs where both are mapped to elements
  for (int i = 1; i <= ntypes; i++) {
    for (int j = i; j <= ntypes; j++) {
        if ((type_mapper[i-1] >= 0) && (type_mapper[j-1] >= 0)) {
            setflag[i][j] = 1;
        }
    }
  }

  char *batchstr = std::getenv("BATCHSIZE");
  if (batchstr != NULL)
  {
    batch_size = std::atoi(batchstr);
  }
  calibrate_energy_offset();
  host_staging.reset_generation();
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

void PairAllegro::calibrate_energy_offset()
{
  // three type-0 atoms on a line; every pair is within the cutoff
  const double spacing = std::min(1.5, 0.35 * cutoff);
  torch::Tensor pos = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat));
  auto pos_a = pos.accessor<float, 2>();
  pos_a[1][0] = spacing;
  pos_a[2][0] = 2.0 * spacing;
  torch::Tensor edges = torch::tensor(
      {{0, 0, 1, 1, 2, 2}, {1, 2, 0, 2, 0, 1}},
      torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor types = torch::zeros({3}, torch::TensorOptions().dtype(torch::kInt64));

  torch::Tensor pos_device = pos.to(device);
  torch::Tensor edges_device = edges.to(device);
  torch::Tensor types_device = types.to(device);

  torch::Tensor total_energy_device;
  torch::Tensor atomic_energy_device;
  if (use_pt2)
  {
    std::vector<at::Tensor> probe_inputs;
    probe_inputs.reserve(3);
    probe_inputs.push_back(pos_device);
    probe_inputs.push_back(edges_device);
    probe_inputs.push_back(types_device);
    std::vector<at::Tensor> outputs;
    try
    {
      outputs = pt2_model->run(probe_inputs);
    }
    catch (const std::exception &exception)
    {
      error->one(FLERR, "Pair style allegro PT2 probe execution failed: " +
                            std::string(exception.what()));
    }
    total_energy_device = checked_pt2_output(
        error, "allegro", "total_energy", outputs,
        pt2_output_indices, {1, 1});
    atomic_energy_device = checked_pt2_output(
        error, "allegro", "atomic_energy", outputs,
        pt2_output_indices, {3, 1});
  }
  else
  {
    c10::Dict<std::string, torch::Tensor> probe_input;
    probe_input.insert_or_assign("pos", pos_device);
    probe_input.insert_or_assign("edge_index", edges_device);
    probe_input.insert_or_assign("atom_types", types_device);
    std::vector<torch::IValue> probe_input_vector;
    probe_input_vector.emplace_back(probe_input);
    auto output = model.forward(probe_input_vector).toGenericDict();
    total_energy_device = output.at("total_energy").toTensor();
    atomic_energy_device = output.at("atomic_energy").toTensor();
  }

  const double total = static_cast<double>(total_energy_device.item<float>());
  const double atomic_sum =
      static_cast<double>(atomic_energy_device.sum().item<float>());
  constant_energy_offset = total - atomic_sum;
  if (!std::isfinite(constant_energy_offset))
    error->one(FLERR, "Pair style allegro probe evaluation of the "
                      "total-energy offset was not finite");
  if (comm->me == 0 && std::fabs(constant_energy_offset) > 1.0e-6)
    utils::logmesg(lmp, "Allegro: constant total-energy offset "
                        "{:.6g} eV (total_energy - sum(atomic_energy) on a "
                        "ghost-free probe graph)\n", constant_energy_offset);
}

// Force and energy computation
template <bool Profile, bool Staged>
void PairAllegro::compute_impl(int eflag, int vflag)
{
  SpingnnPerfContext<Profile> perf(
      perf_state, device, comm->me, static_cast<long long>(update->ntimestep),
      "allegro", use_pt2 ? "pt2" : "torchscript");
  ev_init(eflag, vflag);
  perf.graph_start();

  // Get info from lammps:

  // Atom positions, including ghost atoms
  double **x = atom->x;
  // Atom forces
  double **f = atom->f;
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
  // Mapping from neigh list ordering to x/f ordering
  int *ilist = list->ilist;
  // Number of neighbors per atom
  int *numneigh = list->numneigh;
  // Neighbor list per atom
  int **firstneigh = list->firstneigh;

  // Total number of bonds (sum of number of neighbors)
  int nedges = 0;

  // Number of bonds per atom
  std::vector<int> legacy_neigh_per_atom;
  std::vector<int> legacy_cumsum_neigh_per_atom;
  int *neigh_per_atom;
  int *cumsum_neigh_per_atom;
  bool graph_scratch_grew = false;
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
    }
  }

  // Cumulative sum of neighbors, for knowing where to fill in the edges tensor
  for (int ii = 1; ii < nlocal; ii++)
  {
    cumsum_neigh_per_atom[ii] = cumsum_neigh_per_atom[ii - 1] + neigh_per_atom[ii - 1];
  }

  perf.graph_end_pack_start();

  SpingnnStagingUse staging_use;
  torch::Tensor pos_tensor;
  torch::Tensor edges_tensor;
  torch::Tensor ij2type_tensor;
  if constexpr (Staged)
  {
    staging_use = host_staging.prepare_inputs(
        device, atom->nmax, ntotal, nedges, false,
        {static_cast<long long>(neighbor->ncalls),
         static_cast<long long>(neighbor->lastcall), atom->nlocal,
         atom->nghost, atom->nmax});
    staging_use.note_growth(graph_scratch_grew);
    pos_tensor = host_staging.host_pos(ntotal);
    edges_tensor = host_staging.host_edges(nedges);
    ij2type_tensor = host_staging.host_types(ntotal);
    pos_tensor.zero_();
    edges_tensor.zero_();
    ij2type_tensor.zero_();
  }
  else
  {
    pos_tensor = torch::zeros({ntotal, 3});
    edges_tensor =
        torch::zeros({2, nedges}, torch::TensorOptions().dtype(torch::kInt64));
    ij2type_tensor =
        torch::zeros({ntotal}, torch::TensorOptions().dtype(torch::kInt64));
  }

  auto pos = pos_tensor.accessor<float, 2>();
  auto edges = edges_tensor.accessor<long, 2>();
  auto ij2type = ij2type_tensor.accessor<long, 1>();

  // Loop over atoms and neighbors,
  // store edges and _cell_shifts
  // ii follows the order of the neighbor lists,
  // i follows the order of x, f, etc.
  if (debug_mode)
    printf("Allegro edges: i j rij\n");
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
      int jtype = type[j];

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

      if (debug_mode)
        printf("%d %d %.10g\n", itag - 1, jtag - 1, sqrt(rsq));
    }
  }
  if (debug_mode)
    printf("end Allegro edges\n");

  perf.pack_end();
  perf.h2d_start();
  torch::Tensor pos_device;
  torch::Tensor edges_device;
  torch::Tensor types_device;
  if constexpr (Staged)
  {
    SpingnnStagedInputs staged = host_staging.upload_inputs(ntotal, nedges, false);
    pos_device = staged.pos;
    edges_device = staged.edges;
    types_device = staged.types;
  }
  else
  {
    pos_device = pos_tensor.to(device);
    edges_device = edges_tensor.to(device);
    types_device = ij2type_tensor.to(device);
  }
  perf.h2d_end_model_start();
  torch::Tensor total_energy_device;
  torch::Tensor forces_device;
  torch::Tensor atomic_energy_device;
  torch::Tensor virial_device;
  torch::Tensor atom_virial_device;
  torch::Tensor bec_device;
  if (use_pt2)
  {
    if (pt2_inputs.capacity() < 3) pt2_inputs.reserve(3);
    pt2_inputs.clear();
    pt2_inputs.push_back(pos_device);
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
          FLERR, "Pair style allegro PT2 execution failed for n_atoms=" +
                     std::to_string(ntotal) + ", n_edges=" +
                     std::to_string(nedges) +
                     " (export contract: n_atoms>=1, n_edges>=2): " + exception.what());
    }
    perf.model_end();
    if (outputs.size() != pt2_output_indices.size())
      error->one(
          FLERR, "Pair style allegro PT2 returned " +
                     std::to_string(outputs.size()) +
                     " tensors but output_signature declares " +
                     std::to_string(pt2_output_indices.size()));
    total_energy_device = checked_pt2_output(
        error, "allegro", "total_energy", outputs,
        pt2_output_indices, {1, 1});
    forces_device = checked_pt2_output(
        error, "allegro", "forces", outputs,
        pt2_output_indices, {ntotal, 3});
    atomic_energy_device = checked_pt2_output(
        error, "allegro", "atomic_energy", outputs,
        pt2_output_indices, {ntotal, 1});
    if (vflag_global)
      virial_device = checked_pt2_output(
          error, "allegro", "virial", outputs,
          pt2_output_indices, {1, 3, 3});
    if (vflag_atom || cvflag_atom)
      atom_virial_device = checked_pt2_output(
          error, "allegro", "atom_virial", outputs,
          pt2_output_indices, {ntotal, 3, 3});
    if (atom->bec_flag)
      bec_device = checked_pt2_output(
          error, "allegro", "born_effective_charges", outputs,
          pt2_output_indices, {ntotal, 9});
  }
  else
  {
    model_input.insert_or_assign("pos", pos_device);
    model_input.insert_or_assign("edge_index", edges_device);
    model_input.insert_or_assign("atom_types", types_device);
    if (model_input_vector.empty()) model_input_vector.emplace_back(model_input);
    auto output = model.forward(model_input_vector).toGenericDict();
    perf.model_end();
    total_energy_device = output.at("total_energy").toTensor();
    forces_device = output.at("forces").toTensor();
    atomic_energy_device = output.at("atomic_energy").toTensor();
    if (vflag_global) virial_device = output.at("virial").toTensor();
    if (vflag_atom || cvflag_atom)
      atom_virial_device = output.at("atom_virial").toTensor();
    if (atom->bec_flag)
      bec_device = output.at("born_effective_charges").toTensor();
  }

  torch::Tensor total_energy_tensor;
  torch::Tensor forces_tensor;
  torch::Tensor atomic_energy_tensor;
  torch::Tensor virial_tensor;
  torch::Tensor atom_virial_tensor;
  torch::Tensor bec_tensor;
  if constexpr (Staged)
  {
    const int64_t output_values = total_energy_device.numel() +
        forces_device.numel() + atomic_energy_device.numel() +
        (virial_device.defined() ? virial_device.numel() : 0) +
        (atom_virial_device.defined() ? atom_virial_device.numel() : 0) +
        (bec_device.defined() ? bec_device.numel() : 0);
    staging_use.note_growth(host_staging.prepare_output(device, output_values));
    int64_t output_offset = 0;
    perf.d2h_start();
    total_energy_tensor = host_staging.pack_output(
        error, "allegro", "total_energy", total_energy_device, output_offset);
    output_offset += total_energy_device.numel();
    forces_tensor = host_staging.pack_output(
        error, "allegro", "forces", forces_device, output_offset);
    output_offset += forces_device.numel();
    atomic_energy_tensor = host_staging.pack_output(
        error, "allegro", "atomic_energy", atomic_energy_device, output_offset);
    output_offset += atomic_energy_device.numel();
    if (virial_device.defined())
    {
      virial_tensor = host_staging.pack_output(
          error, "allegro", "virial", virial_device, output_offset);
      output_offset += virial_device.numel();
    }
    if (atom_virial_device.defined())
    {
      atom_virial_tensor = host_staging.pack_output(
          error, "allegro", "atom_virial", atom_virial_device, output_offset);
      output_offset += atom_virial_device.numel();
    }
    if (bec_device.defined())
    {
      bec_tensor = host_staging.pack_output(
          error, "allegro", "born_effective_charges", bec_device, output_offset);
      output_offset += bec_device.numel();
    }
    host_staging.download_output(device, output_offset);
    perf.d2h_end();
  }
  else
  {
    perf.d2h_start();
    total_energy_tensor = total_energy_device.cpu();
    perf.d2h_end();
    perf.d2h_start();
    forces_tensor = forces_device.cpu();
    perf.d2h_end();
    perf.d2h_start();
    atomic_energy_tensor = atomic_energy_device.cpu();
    perf.d2h_end();
    if (virial_device.defined())
    {
      perf.d2h_start();
      virial_tensor = virial_device.cpu();
      perf.d2h_end();
    }
  }

  perf.writeback_start();
  auto forces = forces_tensor.accessor<float, 2>();
  perf.writeback_end();

  perf.writeback_start();
  auto atomic_energies = atomic_energy_tensor.accessor<float, 2>();
  perf.writeback_end();

  if (vflag_global) {
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
  // std::cout << "atomic energy sum: " << atomic_energy_sum << std::endl;
  // std::cout << "Total energy: " << total_energy_tensor << "\n";
  // std::cout << "atomic energy shape: " << atomic_energy_tensor.sizes()[0] << "," << atomic_energy_tensor.sizes()[1] << std::endl;
  // std::cout << "atomic energies: " << atomic_energy_tensor << std::endl;

  if (debug_mode)
  {
    std::cout << "Allegro model output:\n";
    std::cout << "total_energy: " << total_energy_tensor << "\n";
    std::cout << "forces: " << forces_tensor << "\n";
    std::cout << "atomic_energy: " << atomic_energy_tensor << "\n";
    if (vflag_global)
      std::cout << "virial: " << virial_tensor << std::endl;
    if (atom->bec_flag)
    {
      if constexpr (!Staged) bec_tensor = bec_device.cpu();
      std::cout << "born_effective_charges: " << bec_tensor << std::endl;
    }
  }
  // Write forces and per-atom energies (0-based tags here)
  perf.writeback_start();
  eng_vdwl = 0.0;
#pragma omp parallel for reduction(+ \
                                   : eng_vdwl)
  for (int ii = 0; ii < ntotal; ii++)
  {
    int i = ilist[ii];

    f[i][0] += forces[i][0];
    f[i][1] += forces[i][1];
    f[i][2] += forces[i][2];
    if (eflag_atom && ii < inum)
      eatom[i] = atomic_energies[i][0];
    if (ii < inum)
      eng_vdwl += atomic_energies[i][0];
  }
  perf.writeback_end();
  if (comm->me == 0) eng_vdwl += constant_energy_offset;

  if (vflag_atom || cvflag_atom) {
    if constexpr (!Staged)
    {
      perf.d2h_start();
      atom_virial_tensor = atom_virial_device.cpu();
      perf.d2h_end();
    }
      perf.writeback_start();
      auto atomic_virial = atom_virial_tensor.accessor<float, 3>();
      for (int ii = 0; ii < ntotal; ii++)
      {
        int i = ilist[ii];
        if (vflag_atom) {
          vatom[i][0] += atomic_virial[i][0][0]; // xx
          vatom[i][1] += atomic_virial[i][1][1]; // yy
          vatom[i][2] += atomic_virial[i][2][2]; // zz
          vatom[i][3] += atomic_virial[i][0][1]; // xy
          vatom[i][4] += atomic_virial[i][0][2]; // xz
          vatom[i][5] += atomic_virial[i][1][2]; // yz
        }
        if (cvflag_atom) {
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
    } // Allow the use of the atomic virial. - Added by Hongyu Yu

  if (atom->bec_flag)
  {
    double **bec = atom->bec;
    if constexpr (!Staged)
    {
      perf.d2h_start();
      bec_tensor = bec_device.cpu();
      perf.d2h_end();
    }
    perf.writeback_start();
    auto becs_nn = bec_tensor.accessor<float, 2>();

    // Preserve the deployed model tensor exactly. The acoustic sum rule is a
    // training target and must not change predictions only on the LAMMPS path.
    for (int ii = 0; ii < nlocal; ii++)
    {
      int i = ilist[ii];
      bec[i][0] = becs_nn[i][0];
      bec[i][1] = becs_nn[i][1];
      bec[i][2] = becs_nn[i][2];
      bec[i][3] = becs_nn[i][3];
      bec[i][4] = becs_nn[i][4];
      bec[i][5] = becs_nn[i][5];
      bec[i][6] = becs_nn[i][6];
      bec[i][7] = becs_nn[i][7];
      bec[i][8] = becs_nn[i][8];
    }
    perf.writeback_end();
  }

  const long long h2d_bytes = spingnn_tensor_bytes(pos_tensor) +
      spingnn_tensor_bytes(edges_tensor) + spingnn_tensor_bytes(ij2type_tensor);
  const long long d2h_bytes = spingnn_tensor_bytes(total_energy_device) +
      spingnn_tensor_bytes(forces_device) +
      spingnn_tensor_bytes(atomic_energy_device) + spingnn_tensor_bytes(virial_device) +
      spingnn_tensor_bytes(atom_virial_device) + spingnn_tensor_bytes(bec_device);
  perf.finish(
      nlocal, nghost, nedges, vflag_global, vflag_atom || cvflag_atom,
      atom->bec_flag, h2d_bytes, d2h_bytes,
      Staged ? staging_use.cache_hit() : 0,
      Staged ? staging_use.cache_miss() : 0,
      0, 0, 0);
}

void PairAllegro::compute(int eflag, int vflag)
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
