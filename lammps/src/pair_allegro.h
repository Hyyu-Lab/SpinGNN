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

PairStyle(allegro, PairAllegro)

#else

#ifndef LMP_PAIR_ALLEGRO_H
#define LMP_PAIR_ALLEGRO_H

#include "host_staging.h"
#include "pair.h"
#include "perf_utils.h"

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace LAMMPS_NS {

class PairAllegro : public Pair {
 public:
  PairAllegro(class LAMMPS *);
  virtual ~PairAllegro();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  virtual double init_one(int, int);
  virtual void init_style();
  void allocate();

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

  // Constant total-energy offset, calibrated once in coeff() on a small
  // ghost-free probe graph (see calibrate_energy_offset).
  double constant_energy_offset = 0.0;

 protected:
  void calibrate_energy_offset();
  template <bool, bool> void compute_impl(int, int);
  SpingnnPerfState perf_state;
  SpingnnHostStaging host_staging;
  int debug_mode = 0;
};

}    // namespace LAMMPS_NS

#endif
#endif
