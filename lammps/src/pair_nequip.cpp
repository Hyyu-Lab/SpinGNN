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

#include <pair_nequip.h>
#include "atom.h"
#include "comm.h"
#include "deploy_metadata.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "nequip_communication.h"
#include "potential_file_reader.h"
#include "tokenizer.h"
#include "torch_determinism.h"
#include "torch_device.h"

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
//#include <c10/cuda/CUDACachingAllocator.h>


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

PairNEQUIP::PairNEQUIP(LAMMPS *lmp) : Pair(lmp) {
  restartinfo = 0;
  manybody_flag = 1;

  device = spingnn_torch_device(world, error);
  std::cout << "NEQUIP is using device " << device << "\n";

  if(const char* env_p = std::getenv("NEQUIP_DEBUG")){
    std::cout << "PairNEQUIP is in DEBUG mode, since NEQUIP_DEBUG is in env\n";
    debug_mode = 1;
  }
}

PairNEQUIP::~PairNEQUIP(){
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(type_mapper);
  }
}

void PairNEQUIP::init_style(){
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style NEQUIP requires atom IDs");

  // need a full neighbor list
  int irequest = neighbor->request(this,instance_me);
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
    error->all(FLERR,"Pair style NEQUIP requires newton pair off");
  configure_nequip_communication(
      error, comm, neighbor, num_interactions, cutoff, "nequip");
}

double PairNEQUIP::init_one(int i, int j)
{
  return cutoff;
}

void PairNEQUIP::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(type_mapper, n+1, "pair:type_mapper");

}

void PairNEQUIP::settings(int narg, char ** /*arg*/) {
  // "flare" should be the only word after "pair_style" in the input file.
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style command");
}

void PairNEQUIP::coeff(int narg, char **arg) {

  if (!allocated)
    allocate();

  int ntypes = atom->ntypes;

  // Should be exactly 3 arguments following "pair_coeff" in the input file.
  if (narg != (3+ntypes))
    error->all(FLERR, "Incorrect args for pair coefficients");

  // Ensure I,J args are "* *".
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      setflag[i][j] = 0;

  // Parse the definition of each atom type
  char **elements = new char*[ntypes+1];
  for (int i = 1; i <= ntypes; i++){
      elements[i] = new char [strlen(arg[i+2])+1];
      strcpy(elements[i], arg[i+2]);
      if (screen) fprintf(screen, "NequIP Coeff: type %d is element %s\n", i, elements[i]);
  }

  // Initiate type mapper
  for (int i = 1; i<= ntypes; i++){
      type_mapper[i] = -1;
  }

  std::cout << "Loading model from " << arg[2] << "\n";

  std::unordered_map<std::string, std::string> metadata = {
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
    {"has_bec", ""}
  };
  model = torch::jit::load(std::string(arg[2]), device, metadata);
  model.eval();

  // Check if model is a NequIP model
  if (metadata["nequip_version"].empty()) {
    error->all(FLERR, "The indicated TorchScript file does not appear to be a deployed NequIP model; did you forget to run `nequip-deploy`?");
  }

  std::vector<std::string> required_outputs = {
      "forces", "total_energy", "atomic_energy", "virial"};
  if (atom->bec_flag)
    required_outputs.push_back("born_effective_charges");
  const std::vector<std::string> available_inputs = {
      "pos", "edge_index", "edge_cell_shift", "cell", "pbc", "atom_types"};
  validate_deploy_metadata(
      error, metadata, "nequip", required_outputs, available_inputs);

  // If the model is not already frozen, we should freeze it:
  // This is the check used by PyTorch: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/api/module.cpp#L476
  if (model.hasattr("training")) {
    std::cout << "Freezing TorchScript model...\n";
    #ifdef DO_TORCH_FREEZE_HACK
      // Do the hack
      // Copied from the implementation of torch::jit::freeze,
      // except without the broken check
      // See https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp
      bool optimize_numerics = true;  // the default
      // the {} is preserved_attrs
      auto out_mod = freeze_module(
        model, {}
      );
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
    // Set JIT bailout to avoid long recompilations for many steps
    size_t jit_bailout_depth;
    if (metadata["_jit_bailout_depth"].empty()) {
      // This is the default used in the Python code
      jit_bailout_depth = 2;
    } else {
      jit_bailout_depth = std::stoi(metadata["_jit_bailout_depth"]);
    }
    torch::jit::getBailoutDepth() = jit_bailout_depth;
  #else
    // In PyTorch >=1.11, this is now set_fusion_strategy
    torch::jit::FusionStrategy strategy;
    if (metadata["_jit_fusion_strategy"].empty()) {
      // This is the default used in the Python code
      strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
    } else {
      std::stringstream strat_stream(metadata["_jit_fusion_strategy"]);
      std::string fusion_type, fusion_depth;
      while(std::getline(strat_stream, fusion_type, ',')) {
        std::getline(strat_stream, fusion_depth, ';');
        strategy.push_back({fusion_type == "STATIC" ? torch::jit::FusionBehavior::STATIC : torch::jit::FusionBehavior::DYNAMIC, std::stoi(fusion_depth)});
      }
    }
    torch::jit::setFusionStrategy(strategy);
  #endif

  // Set whether to allow TF32:
  bool allow_tf32;
  if (metadata["allow_tf32"].empty()) {
    // Better safe than sorry
    allow_tf32 = false;
  } else {
    // It gets saved as an int 0/1
    allow_tf32 = std::stoi(metadata["allow_tf32"]);
  }
  // See https://pytorch.org/docs/stable/notes/cuda.html
  at::globalContext().setAllowTF32CuBLAS(allow_tf32);
  at::globalContext().setAllowTF32CuDNN(allow_tf32);
  if (configure_torch_determinism(error) && comm->me == 0)
    std::cout << "Pair style nequip deterministic algorithms enabled "
                 "(SPINGNN_DETERMINISTIC=1)\n";

  // std::cout << "Information from model: " << metadata.size() << " key-value pairs\n";
  // for( const auto& n : metadata ) {
  //   std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
  // }

  cutoff = std::stod(metadata["r_max"]);
  num_interactions = read_nequip_num_interactions(
      error, DeployMetadata(metadata.begin(), metadata.end()), "nequip");

  // match the type names in the pair_coeff to the metadata
  // to construct a type mapper from LAMMPS type to NequIP atom_types
  int n_species = std::stod(metadata["n_species"]);
  std::stringstream ss;
  ss << metadata["type_names"];
  for (int i = 0; i < n_species; i++){
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

  if (elements){
      for (int i=1; i<=ntypes; i++)
          if (elements[i]) delete [] elements[i];
      delete [] elements;
  }
  calibrate_energy_offset();
}

void PairNEQUIP::calibrate_energy_offset()
{
  // A total-only rescale shift is global, not one contribution per MPI rank.
  // Match the other ML pair styles by measuring it on a ghost-free graph.
  const double spacing = std::min(1.5, 0.35 * cutoff);
  torch::Tensor pos = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat));
  auto position = pos.accessor<float, 2>();
  position[1][0] = spacing;
  position[2][0] = 2.0 * spacing;
  torch::Tensor edges = torch::tensor(
      {{0, 0, 1, 1, 2, 2}, {1, 2, 0, 2, 0, 1}},
      torch::TensorOptions().dtype(torch::kInt64));
  c10::Dict<std::string, torch::Tensor> input;
  input.insert("pos", pos.to(device));
  input.insert("edge_index", edges.to(device));
  input.insert("edge_cell_shift", torch::zeros({6, 3}, pos.options()).to(device));
  input.insert("cell", (torch::eye(3, pos.options()) * (4.0 * cutoff)).to(device));
  // Stress-capable models require a valid periodic cell even though this
  // probe contains no periodic image edges or ghost atoms.
  input.insert("pbc", torch::ones({1, 3}, torch::TensorOptions().dtype(torch::kBool)).to(device));
  input.insert("atom_types", torch::zeros({3}, torch::TensorOptions().dtype(torch::kInt64)).to(device));
  std::vector<torch::IValue> inputs(1, input);
  auto output = model.forward(inputs).toGenericDict();
  constant_energy_offset =
      static_cast<double>(output.at("total_energy").toTensor().item<float>()) -
      static_cast<double>(output.at("atomic_energy").toTensor().sum().item<float>());
  if (!std::isfinite(constant_energy_offset))
    error->all(FLERR, "Pair style nequip has a nonfinite total-energy offset");
}

// Force and energy computation
void PairNEQUIP::compute(int eflag, int vflag){
  ev_init(eflag, vflag);
  // All ranks must participate in Error::all, including empty owners.
  if (vflag_atom)
    error->all(FLERR, "Pair style NEQUIP does not support per-atom virial");
  if (atom->nlocal == 0) {
    // Newton is off: an empty owner has no forces to contribute, even when
    // its ghost graph is empty and therefore outside the model's shape ABI.
    eng_vdwl = comm->me == 0 ? constant_energy_offset : 0.0;
    return;
  }

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
  int newton_pair = force->newton_pair;
  // Should probably be off.
  if (newton_pair==1)
    error->all(FLERR,"Pair style NEQUIP requires 'newton off'");

  // Number of local/real atoms
  int inum = list->inum;
  assert(inum==nlocal); // This should be true, if my understanding is correct
  // Number of local and ghost centers in the requested full neighbor list
  int ncenters = inum + list->gnum;
  // Mapping from neigh list ordering to x/f ordering
  int *ilist = list->ilist;
  // Number of neighbors per atom
  int *numneigh = list->numneigh;
  // Neighbor list per atom
  int **firstneigh = list->firstneigh;

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
  torch::Tensor batch_tensor =
      torch::zeros({ngraph}, torch::TensorOptions().dtype(torch::kInt64));
  auto node_batch = batch_tensor.accessor<long, 1>();
  for (int graph_i = 0; graph_i < ngraph; graph_i++)
    if (graph_to_atom[graph_i] >= nlocal) node_batch[graph_i] = 1;
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
  torch::Tensor tag2type_tensor = torch::zeros({ngraph}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor edges_tensor = torch::zeros({2, nedges}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor edge_cell_shifts_tensor = torch::zeros({nedges, 3});
  torch::Tensor cell_tensor = torch::zeros({3,3});
  torch::Tensor pbc_tensor = torch::zeros(
      {1, 3}, torch::TensorOptions().dtype(torch::kBool));

  auto pos = pos_tensor.accessor<float, 2>();
  auto tag2type = tag2type_tensor.accessor<long, 1>();
  auto edges = edges_tensor.accessor<long, 2>();
  auto edge_cell_shifts = edge_cell_shifts_tensor.accessor<float, 2>();
  auto cell = cell_tensor.accessor<float,2>();
  auto pbc = pbc_tensor.accessor<bool, 2>();

  pbc[0][0] = domain->xperiodic != 0;
  pbc[0][1] = domain->yperiodic != 0;
  pbc[0][2] = domain->zperiodic != 0;

  // A physical atom must use the same primary-cell origin on every rank.
  // Otherwise the per-batch strain derivative assigns ghost-image offsets
  // to a different frame and the summed StressOutput virial is incorrect.
  std::vector<std::array<double, 3>> canonical_positions(ngraph);
  for (int graph_i = 0; graph_i < ngraph; graph_i++) {
    int i = graph_to_atom[graph_i];
    int itype = type[i];
    tag2type[graph_i] = type_mapper[itype];
    auto &canonical = canonical_positions[graph_i];
    std::copy(x[i], x[i] + 3, canonical.begin());
    domain->remap(canonical.data());
    for (int axis = 0; axis < 3; ++axis) pos[graph_i][axis] = canonical[axis];
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
  if (debug_mode) printf("NEQUIP edges: i j xi[:] xj[:] cell_shift[:] rij\n");
  for (int graph_i = 0; graph_i < ngraph; graph_i++) {
    if (!graph_has_center[graph_i]) continue;
    int i = graph_to_atom[graph_i];

    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    int edge_counter = edge_offsets[graph_i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      int graph_j = atom_to_graph[j];
      if (graph_j < 0)
        error->one(FLERR, "Pair style NEQUIP could not map a ghost atom to the graph");

      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double rsq = dx*dx + dy*dy + dz*dz;
      if (rsq < cutoff*cutoff) {
          nequip_periodic_shift(
              domain, x, i, j, canonical_positions[graph_i].data(),
              canonical_positions[graph_j].data(),
              &edge_cell_shifts[edge_counter][0]);
          edges[0][edge_counter] = graph_i;
          edges[1][edge_counter] = graph_j;
          edge_counter++;

          if (debug_mode) {
              printf("%d %d %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n", graph_i, graph_j,
                pos[graph_i][0],pos[graph_i][1],pos[graph_i][2],pos[graph_j][0],pos[graph_j][1],pos[graph_j][2],
                edge_cell_shifts[edge_counter-1][0],edge_cell_shifts[edge_counter-1][1],edge_cell_shifts[edge_counter-1][2],sqrt(rsq));
          }
      }
    }
  }
  if (debug_mode) printf("end NEQUIP edges\n");


  c10::Dict<std::string, torch::Tensor> input;
  input.insert("pos", pos_tensor.to(device));
  input.insert("edge_index", edges_tensor.to(device));
  input.insert("edge_cell_shift", edge_cell_shifts_tensor.to(device));
  input.insert("cell", cell_tensor.to(device));
  input.insert("pbc", pbc_tensor.to(device));
  input.insert("atom_types", tag2type_tensor.to(device));
  input.insert("batch", batch_tensor.to(device));
  std::vector<torch::IValue> input_vector(1, input);

  if(debug_mode){
    std::cout << "NequIP model input:\n";
    std::cout << "pos:\n" << pos_tensor << "\n";
    std::cout << "edge_index:\n" << edges_tensor << "\n";
    std::cout << "edge_cell_shifts:\n" << edge_cell_shifts_tensor << "\n";
    std::cout << "cell:\n" << cell_tensor << "\n";
    std::cout << "pbc:\n" << pbc_tensor << "\n";
    std::cout << "atom_types:\n" << tag2type_tensor << "\n";
  }


  auto output = model.forward(input_vector).toGenericDict();

  torch::Tensor forces_tensor = output.at("forces").toTensor().cpu();
  auto forces = forces_tensor.accessor<float, 2>();

  if (atom->bec_flag){
    double **bec = atom->bec;
    torch::Tensor bec_tensor = output.at("born_effective_charges").toTensor().cpu();
    auto becs_nn = bec_tensor.accessor<float, 2>();

    for(int ii = 0; ii < inum; ii++){
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

  torch::Tensor total_energy_tensor = output.at("total_energy").toTensor().cpu();

  torch::Tensor atomic_energy_tensor = output.at("atomic_energy").toTensor().cpu();
  auto atomic_energies = atomic_energy_tensor.accessor<float, 2>();

  if(vflag){
    torch::Tensor v_tensor = output.at("virial").toTensor().cpu();
    auto v = v_tensor.accessor<float, 3>();
    // Convert from 3x3 symmetric tensor format, which NequIP outputs, to the flattened form LAMMPS expects
    // Batch zero contains edges centered on owned atoms. Ghost-centered
    // contributions belong to their owners when LAMMPS sums rank virials.
    virial[0] = v[0][0][0];
    virial[1] = v[0][1][1];
    virial[2] = v[0][2][2];
    virial[3] = v[0][0][1];
    virial[4] = v[0][0][2];
    virial[5] = v[0][1][2];
  }
  if(debug_mode){
    std::cout << "NequIP model output:\n";
    std::cout << "forces: " << forces_tensor << "\n";
    std::cout << "total_energy: " << total_energy_tensor << "\n";
    std::cout << "atomic_energy: " << atomic_energy_tensor << "\n";
    if (atom->bec_flag) std::cout << "born_effective_charges: " << output.at("born_effective_charges").toTensor().cpu() << std::endl;
    if(vflag) std::cout << "virial: " << output.at("virial").toTensor().cpu() << std::endl;
  }

  //std::cout << "atomic energy sum: " << atomic_energy_sum << std::endl;
  //std::cout << "Total energy: " << total_energy_tensor << "\n";
  //std::cout << "atomic energy shape: " << atomic_energy_tensor.sizes()[0] << "," << atomic_energy_tensor.sizes()[1] << std::endl;
  //std::cout << "atomic energies: " << atomic_energy_tensor << std::endl;

  // Only owned atoms contribute to this rank's energy and output arrays.
  eng_vdwl = 0.0;
  for(int ii = 0; ii < inum; ii++){
    int i = ilist[ii];
    int graph_i = atom_to_graph[i];
    f[i][0] += forces[graph_i][0];
    f[i][1] += forces[graph_i][1];
    f[i][2] += forces[graph_i][2];
    eng_vdwl += atomic_energies[graph_i][0];
    if (eflag_atom) eatom[i] = atomic_energies[graph_i][0];
  }
  if (comm->me == 0) eng_vdwl += constant_energy_offset;

  /*
  if(device.is_cuda()){
    //torch::cuda::empty_cache();
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
  */
}
