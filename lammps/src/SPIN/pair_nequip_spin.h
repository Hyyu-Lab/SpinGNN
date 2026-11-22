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

PairStyle(spin/nequip_spin, PairNEQUIP_spin)

#else

#ifndef LMP_PAIR_NEQUIP_SPIN_H
#define LMP_PAIR_NEQUIP_SPIN_H

#include "pair_spin.h"

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace LAMMPS_NS
{

  class PairNEQUIP_spin : public PairSpin
  {
  public:
    PairNEQUIP_spin(class LAMMPS *);
    virtual ~PairNEQUIP_spin();
    void *extract(const char *, int &);
    void compute(int, int);
    void compute_single_pair(int, double *);
    void settings(int, char **);
    void coeff(int, char **);
    double init_one(int, int);
    void init_style();
    void allocate();

    double cutoff;
    int num_interactions = 0;
    torch::jit::Module model;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> pt2_model;
    bool use_pt2 = false;
    std::vector<std::string> pt2_input_signature;
    std::unordered_map<std::string, size_t> pt2_output_indices;
    torch::Device device = torch::kCPU;
    int contribution_nmax = 0;
    double **force_contribution = nullptr;
    double **spin_force_contribution = nullptr;

    // Constant total-energy offset, calibrated once in coeff() on a small
    // ghost-free probe graph (see calibrate_energy_offset).
    double constant_energy_offset = 0.0;

  protected:
    struct ModelOutputs
    {
      torch::Tensor forces;
      torch::Tensor spin_forces;
      torch::Tensor total_energy;
      torch::Tensor atomic_energy;
      torch::Tensor virial;
      torch::Tensor born_effective_charges;
    };

    ModelOutputs run_model(
        const torch::Tensor &, const torch::Tensor &, const torch::Tensor &,
        const torch::Tensor &, const torch::Tensor &, const torch::Tensor &,
        const torch::Tensor &, const torch::Tensor &, bool, bool);
    void calibrate_energy_offset();
    int *type_mapper;
    int debug_mode = 0;
  };

}

#endif
#endif
