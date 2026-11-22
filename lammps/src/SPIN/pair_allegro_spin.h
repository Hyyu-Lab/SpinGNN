/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(spin/allegro_spin, PairAllegro_spin)

#else

#ifndef LMP_PAIR_ALLEGRO_SPIN_H
#define LMP_PAIR_ALLEGRO_SPIN_H

#include "host_staging.h"
#include "pair_spin.h"
#include "perf_utils.h"

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <array>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace LAMMPS_NS {

class PairAllegro_spin : public PairSpin {
 public:
  PairAllegro_spin(class LAMMPS *);
  virtual ~PairAllegro_spin();
  void *extract(const char *, int &);
  virtual void compute(int, int);
  void compute_single_pair(int, double *);
  void set_sib_geometry_reuse_hint(bigint) override;
  void settings(int, char **);
  virtual void coeff(int, char **);
  virtual double init_one(int, int);
  virtual void init_style();
  void allocate();
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  // void *extract(const char *, int &);?

  double cutoff;
  torch::jit::Module model;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> pt2_model;
  bool use_pt2 = false;
  std::vector<std::string> pt2_input_signature;
  std::unordered_map<std::string, size_t> pt2_output_indices;
  std::vector<at::Tensor> pt2_inputs;
  torch::Device device = torch::kCPU;
  std::vector<int> type_mapper;
  c10::Dict<std::string, torch::Tensor> model_input;
  std::vector<torch::IValue> model_input_vector;

  int batch_size = -1;

  int contribution_nmax;
  double **force_contribution;
  double **spin_force_contribution;
  int spin_force_reduced_nmax;
  double **spin_force_reduced;

  // Constant total-energy offset, calibrated once in coeff() on a small
  // ghost-free probe graph (see calibrate_energy_offset).
  double constant_energy_offset = 0.0;

  // void write_restart(FILE *);
  // void read_restart(FILE *);
  // void write_restart_settings(FILE *);
  // void read_restart_settings(FILE *);

 protected:
  struct ModelOutputs {
    torch::Tensor forces;
    torch::Tensor spin_forces;
    torch::Tensor total_energy;
    torch::Tensor atomic_energy;
    torch::Tensor virial;
    torch::Tensor atom_virial;
  };

  ModelOutputs run_model(const torch::Tensor &, const torch::Tensor &,
                         const torch::Tensor &, const torch::Tensor &, bool, bool);
  void calibrate_energy_offset();
  template <bool, bool>
  ModelOutputs run_model_impl(const torch::Tensor &, const torch::Tensor &,
                              const torch::Tensor &, const torch::Tensor &, bool, bool,
                              bool, bool, SpingnnPerfContext<true> *,
                              SpingnnStagingUse *);
  template <bool, bool> void compute_impl(int, int);

  struct SibGeometryGeneration {
    long long timestep = std::numeric_limits<long long>::min();
    long long neighbor_ncalls = -1;
    long long neighbor_lastcall = -1;
    int neighbor_ago = -1;
    int nlocal = -1;
    int nghost = -1;
    int nmax = -1;
    int nedges = -1;
    int triclinic = -1;
    double cutoff = -1.0;
  };

  bool sib_geometry_cache_eligible() const;
  bool sib_geometry_matches(int, int *, double **, tagint *, int *) const;
  void capture_sib_geometry(int, int, int, int *, double **, tagint *, int *);
  void reset_sib_geometry_cache();

  SpingnnPerfState perf_state;
  SpingnnHostStaging host_staging;
  bool sib_geometry_cache_enabled = false;
  std::string sib_geometry_cache_invalid_value;
  long long sib_geometry_hint_step = std::numeric_limits<long long>::min();
  bool sib_geometry_cache_valid = false;
  SibGeometryGeneration sib_geometry_generation;
  std::array<double, 12> box_snapshot{};
  std::vector<int> atom_index_snapshot;
  std::vector<double> position_snapshot;
  std::vector<long long> tag_snapshot;
  std::vector<int> type_snapshot;
  std::vector<int> type_mapper_snapshot;
  torch::Tensor sib_cached_pos_device;
  torch::Tensor sib_cached_edges_device;
  torch::Tensor sib_cached_types_device;
  int debug_mode = 0;
};

}    // namespace LAMMPS_NS

#endif
#endif

    /* ERROR/WARNING messages:

E: Incorrect args in pair_spin command

Self-explanatory.

E: Spin simulations require metal unit style

Self-explanatory.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair spin requires atom attribute spin

The atom style defined does not have these attributes.

*/
