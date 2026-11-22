// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------
   Contributing authors: Julien Tranchida (SNL)
                         Aidan Thompson (SNL)

   Please cite the related publication:
   Tranchida, J., Plimpton, S. J., Thibaudeau, P., & Thompson, A. P. (2018).
   Massively parallel symplectic algorithm for coupled magnetic spin dynamics
   and molecular dynamics. Journal of Computational Physics.
------------------------------------------------------------------------- */

#include "fix_spin_sib.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_langevin_spin.h"
#include "fix_precession_spin.h"
#include "fix_setforce_spin.h"
#include "force.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "random_mars.h"
#include "pair_hybrid.h"
#include "pair_spin.h"
#include "update.h"
#include "timer.h"
#include "integrate.h"
#include "pair.h"

#include <cstring>
#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;

static const char cite_fix_nve_spin[] =
  "fix NVE/spin_sib command:\n\n"
  "@article{tranchida2018massively,\n"
  "title={Massively parallel symplectic algorithm for coupled magnetic spin "
  "dynamics and molecular dynamics},\n"
  "author={Tranchida, J and Plimpton, SJ and Thibaudeau, P and Thompson, AP},\n"
  "journal={Journal of Computational Physics},\n"
  "volume={372},\n"
  "pages={406-425},\n"
  "year={2018},\n"
  "publisher={Elsevier}\n"
  "doi={10.1016/j.jcp.2018.06.042}\n"
  "}\n\n";

enum{NONE};

/* ---------------------------------------------------------------------- */

FixNVESpinSIB::FixNVESpinSIB(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  nlocal_max(0), random_vectors(nullptr), original_spins(nullptr), half_step_spins(nullptr),
  pair(nullptr), spin_pairs(nullptr),
  locksetforcespin(nullptr), lockprecessionspin(nullptr)
{
  // if (lmp->citeme) lmp->citeme->add(cite_fix_nve_spin);

  if (narg != 8) error->all(FLERR,"Illegal fix/NVE/spin_sib command");

  temp = utils::numeric(FLERR,arg[3],false,lmp);
  alpha = utils::numeric(FLERR,arg[4],false,lmp);
  seed = utils::inumeric(FLERR,arg[5],false,lmp);

  if (alpha < 0.0) {
    error->all(FLERR,"Illegal langevin/spin command");
  }

  if (temp < 0.0) {
    error->all(FLERR,"Illegal langevin/spin command");
  }

  // example: fix 1 all spin/sib ${myTemp} ${Tdamp_spin} 5252 lattice frozen

  // defining lattice_flag

  // changing the lattice option, from (yes,no) -> (moving,frozen)
  // for now, (yes,no) still works (to avoid user's confusions).

  if (strcmp(arg[6],"lattice") == 0) {
    if (strcmp(arg[7],"no") == 0) lattice_flag = 0;
    else if (strcmp(arg[7],"frozen") == 0) lattice_flag = 0;
    else if (strcmp(arg[7],"yes") == 0) lattice_flag = 1;
    else if (strcmp(arg[7],"moving") == 0) lattice_flag = 1;
    else error->all(FLERR,"Illegal fix/NVE/spin_sib command");
  } else error->all(FLERR,"Illegal fix/NVE/spin_sib command");

  // initialize Marsaglia RNG with processor-unique seed

  // random = new RanPark(lmp,seed + comm->me);
  random = new RanMars(lmp,seed + comm->me);

  time_integrate = 1;
  npairs = 0;
  npairspin = 0;

  // test nprec
  nprecspin = nsetspin = 0;

  // checking if map array or hash is defined

  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR,"Fix NVE/spin_sib requires an atom map, see atom_modify");

  // check if the atom/spin style is defined

  if (!atom->sp_flag)
    error->all(FLERR,"Fix NVE/spin_sib requires atom/spin style");

  // initialize the magnetic interaction flags

  pair_spin_flag = 0;
  precession_spin_flag = 0;
  setforce_spin_flag = 0;
}

/* ---------------------------------------------------------------------- */

FixNVESpinSIB::~FixNVESpinSIB()
{
  memory->destroy(random_vectors);
  memory->destroy(original_spins);
  memory->destroy(half_step_spins);
  delete random;
  delete [] spin_pairs;
  delete [] lockprecessionspin;
}

/* ---------------------------------------------------------------------- */

int FixNVESpinSIB::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNVESpinSIB::init()
{

  // init() is called at the start of every run.  Release and reset cached
  // style lists before rebuilding them so consecutive run commands do not
  // leak memory or retain a stale interaction flag.
  delete [] spin_pairs;
  spin_pairs = nullptr;
  delete [] lockprecessionspin;
  lockprecessionspin = nullptr;
  pair = nullptr;
  locksetforcespin = nullptr;
  pair_spin_flag = 0;
  precession_spin_flag = 0;
  setforce_spin_flag = 0;

  // set timesteps

  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  npairs = npairspin = 0;

  // set ptrs on Pair/Spin styles

  // loop 1: obtain # of Pairs, and # of Pair/Spin styles

  npairspin = 0;
  PairHybrid *hybrid = (PairHybrid *)force->pair_match("^hybrid",0);
  if (hybrid) {
    npairs = hybrid->nstyles;
    for (int i = 0; i < npairs; i++)
      if (utils::strmatch(hybrid->keywords[i],"^spin")) npairspin++;
  } else {
    npairs = 1;
    pair = force->pair_match("^spin",0);
    if (pair) npairspin = 1;
  }

  // init length of vector of ptrs to Pair/Spin styles

  if (npairspin > 0) {
    spin_pairs = new PairSpin*[npairspin];
  }

  // loop 2: fill vector with ptrs to Pair/Spin styles

  int count1 = 0;
  if (hybrid) {
    for (int i = 0; i < npairs; i++) {
      if (utils::strmatch(hybrid->keywords[i],"^spin")) {
        PairSpin *spin_pair = dynamic_cast<PairSpin *>(hybrid->styles[i]);
        if (!spin_pair) error->all(FLERR,"Hybrid spin sub-style is not a PairSpin");
        if (count1 == 0) pair = spin_pair;
        spin_pairs[count1++] = spin_pair;
      }
    }
  } else if (pair) {
    PairSpin *spin_pair = dynamic_cast<PairSpin *>(pair);
    if (!spin_pair) error->all(FLERR,"Spin pair style is not a PairSpin");
    spin_pairs[count1++] = spin_pair;
  }

  if (count1 != npairspin)
    error->all(FLERR,"Incorrect number of spin pair styles");

  // set pair/spin and long/spin flags

  if (npairspin >= 1) pair_spin_flag = 1;

  // set ptrs for fix precession/spin styles

  // loop 1: obtain # of fix precession/spin styles

  int iforce;
  nprecspin = 0;
  for (iforce = 0; iforce < modify->nfix; iforce++) {
    if (utils::strmatch(modify->fix[iforce]->style,"^precession/spin")) {
      nprecspin++;
    }
  }

  // init length of vector of ptrs to precession/spin styles

  if (nprecspin > 0) {
    lockprecessionspin = new FixPrecessionSpin*[nprecspin];
  }

  // loop 2: fill vector with ptrs to precession/spin styles

  int count2 = 0;
  if (nprecspin > 0) {
    for (iforce = 0; iforce < modify->nfix; iforce++) {
      if (utils::strmatch(modify->fix[iforce]->style,"^precession/spin")) {
        precession_spin_flag = 1;
        lockprecessionspin[count2] = (FixPrecessionSpin *) modify->fix[iforce];
        count2++;
      }
    }
  }

  if (count2 != nprecspin)
    error->all(FLERR,"Incorrect number of precession/spin fixes");

  // ptrs FixSetForceSpin classes

  for (iforce = 0; iforce < modify->nfix; iforce++) {
    if (utils::strmatch(modify->fix[iforce]->style,"^setforce/spin")) {
      setforce_spin_flag = 1;
      locksetforcespin = (FixSetForceSpin *) modify->fix[iforce];
    }
  }

  // LangevinSpin init
  gil_factor = 1.0/(1.0+(alpha)*(alpha));

  hbar = force->hplanck/LAMMPS_NS::MathConst::MY_2PI;  // eV/(rad.THz)
  double kb = force->boltz;             // eV/K
  

  D = (alpha*gil_factor*kb*temp); // eV
  C_spin = sqrt(2.0*D);
  dts = 2 * update->dt * gil_factor / hbar ;//
}

/* ---------------------------------------------------------------------- */

void FixNVESpinSIB::initial_integrate(int /*vflag*/)
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int *type = atom->type;
  int *mask = atom->mask;

  if (nlocal_max < atom->nmax) {
    nlocal_max = atom->nmax;
    memory->grow(random_vectors,nlocal_max,3,"spin/sib:random_vectors");
    memory->grow(original_spins,nlocal_max,3,"spin/sib:original_spins");
    memory->grow(half_step_spins,nlocal_max,3,"spin/sib:half_step_spins");
  }

  double dtfm, dtspin;
  double Imat[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0},{0.0, 0.0, 1.0}};
  double Acar[3],Bcar[3],car_tmp[3],Fcar[3];
  double Amat[3][3],Bmat[3][3],Inv_mat[3][3],Pmat[3][3],Qmat[3][3];
  // Per-atom spin and random-vector workspaces are heap allocated members.

  if (lattice_flag) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if (rmass) dtfm = dtf / rmass[i];
        else dtfm = dtf / mass[type[i]];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }
    }
    // update bec_dipole
    if (atom->bec_flag){
      double **bec = atom->bec;
      double **bec_dipole = atom->bec_dipole;

      // bec_dipole
      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          bec_dipole[i][0] += dtv * v[i][0] * bec[i][0] + dtv * v[i][1] * bec[i][1] + dtv * v[i][2] * bec[i][2];
          bec_dipole[i][1] += dtv * v[i][0] * bec[i][3] + dtv * v[i][1] * bec[i][4] + dtv * v[i][2] * bec[i][5];
          bec_dipole[i][2] += dtv * v[i][0] * bec[i][6] + dtv * v[i][1] * bec[i][7] + dtv * v[i][2] * bec[i][8];
        }
      }
    }
  }


  // collect spin forces with potential, external magnetic field first
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      ComputeInteractionsSpin(i);
    }
  }
  
  double **fm = atom->fm;
  double **sp = atom->sp;

  // update s for all atoms with SIB

  // step 1: get S_h_car
  for (int i = 0; i < nlocal; i++) {
    if ((sp[i][3]>0) and (mask[i] & groupbit)){
      for (int j = 0; j < 3; j++) {
        Fcar[j] = fm[i][j]*hbar;
        original_spins[i][j] = sp[i][j];
        random_vectors[i][j] = random->gaussian();
      }
      dtspin = dts / sp[i][3];
      // rand_cut(dtspin, random_vectors[i]);
      as_vec(original_spins[i],Fcar,alpha,Acar);
      vec2mat(Acar,Amat);
      bs_vec(original_spins[i],random_vectors[i],alpha,C_spin,Bcar,sp[i][3]);
      vec2mat(Bcar,Bmat);

      for (int j1 = 0; j1 < 3; j1++) {
        for (int j2 = 0; j2 < 3; j2++) {
          Pmat[j1][j2] = Imat[j1][j2] - 0.5 * dtspin * Amat[j1][j2] - 0.5 * sqrt(dtspin) * Bmat[j1][j2];
          Qmat[j1][j2] = Imat[j1][j2] + 0.5 * dtspin * Amat[j1][j2] + 0.5 * sqrt(dtspin) * Bmat[j1][j2];
        }
      }
      invert_mat(Pmat, Inv_mat);
      MathExtra::matvec(Qmat,original_spins[i],car_tmp);
      MathExtra::matvec(Inv_mat,car_tmp,half_step_spins[i]);
      
      // update spin with half step
      for (int j = 0; j < 3; j++) {
        sp[i][j] = (original_spins[i][j] + half_step_spins[i][j]) * 0.5;
      }
      MathExtra::norm3(sp[i]);
    } 
  }

  // Synchronize the half-step spins before evaluating the midpoint force.
  // Pair styles such as spin/allegro_spin read both owned and ghost spins,
  // so calling pair->compute() directly here without a forward communication
  // mixes half-step owned spins with stale ghost spins.  With Newton pair on,
  // the ghost force contributions must also be returned to their owners just
  // as they are in the regular Verlet force path.
  comm->forward_comm();
  if (!lattice_flag)
    for (int k = 0; k < npairspin; k++)
      spin_pairs[k]->set_sib_geometry_reuse_hint(update->ntimestep);
  force->pair->compute(0,0);
  if (force->newton) comm->reverse_comm();

  // collect spin forces with potential, external magnetic field first
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      ComputeInteractionsSpin(i);
    }
  }

  // step 2: get final spin
  for (int i = 0; i < nlocal; i++) {
    if ((sp[i][3]>0) and (mask[i] & groupbit)){
      for (int j = 0; j < 3; j++) {
        Fcar[j] = fm[i][j]*hbar; // update fm from half step spin
        half_step_spins[i][j] = sp[i][j];
      }
      dtspin = dts / sp[i][3];
      as_vec(half_step_spins[i],Fcar,alpha,Acar);
      vec2mat(Acar,Amat);
      bs_vec(half_step_spins[i],random_vectors[i],alpha,C_spin,Bcar,sp[i][3]);
      vec2mat(Bcar,Bmat);

      for (int j1 = 0; j1 < 3; j1++) {
        for (int j2 = 0; j2 < 3; j2++) {
          Pmat[j1][j2] = Imat[j1][j2] - 0.5 * dtspin * Amat[j1][j2] - 0.5 * sqrt(dtspin) * Bmat[j1][j2];
          Qmat[j1][j2] = Imat[j1][j2] + 0.5 * dtspin * Amat[j1][j2] + 0.5 * sqrt(dtspin) * Bmat[j1][j2];
        }
      }
      invert_mat(Pmat, Inv_mat);
      MathExtra::matvec(Qmat,original_spins[i],car_tmp);
      MathExtra::matvec(Inv_mat,car_tmp,half_step_spins[i]);
      
      // update spin with half step
      for (int j = 0; j < 3; j++) {
        sp[i][j] = half_step_spins[i][j];
      }
      MathExtra::norm3(sp[i]);
    }
  }


  // if (lattice_flag) {
  //   // update system with new spin if lattice_flag
  //   // force->pair->compute(0,0);
  //   // if not lattice_flag, it will be update in verlet main
  // }

  if (lattice_flag) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
      }
    }
  }
}

/* ----------------------------------------------------------------------
   compute the magnetic torque for the spin ii
---------------------------------------------------------------------- */

void FixNVESpinSIB::ComputeInteractionsSpin(int i)
{
  double spi[3], fmi[3];

  double **sp = atom->sp;
  double **fm = atom->fm;

  // force computation for spin i

  spi[0] = sp[i][0];
  spi[1] = sp[i][1];
  spi[2] = sp[i][2];

  fmi[0] = fmi[1] = fmi[2] = 0.0;

  // update magnetic pair interactions

  if (pair_spin_flag) {
    for (int k = 0; k < npairspin; k++) {
      spin_pairs[k]->compute_single_pair(i,fmi);
      // fmi[0] += fm[i][0];
      // fmi[1] += fm[i][1];
      // fmi[2] += fm[i][2];
    }
  }

  // update magnetic precession interactions

  if (precession_spin_flag) {
    for (int k = 0; k < nprecspin; k++) {
      lockprecessionspin[k]->compute_single_precession(i,spi,fmi);
    }
  }

  // Skip langevin damping
  // update langevin damping and random force

  // if (maglangevin_flag) {               // mag. langevin
  //   for (int k = 0; k < nlangspin; k++) {
  //     locklangevinspin[k]->compute_single_langevin(i,spi,fmi);
  //   }
  // }

  // update setforce of magnetic interactions

  if (setforce_spin_flag) {
    locksetforcespin->single_setforce_spin(i,fmi);
  }

  // replace the magnetic force fm[i] by its new value fmi

  fm[i][0] = fmi[0];
  fm[i][1] = fmi[1];
  fm[i][2] = fmi[2];
}

void FixNVESpinSIB::final_integrate()
{
  // update half v for all particles

  if (lattice_flag) {
    double dtfm;

    double **v = atom->v;
    double **f = atom->f;
    double *rmass = atom->rmass;
    double *mass = atom->mass;
    int nlocal = atom->nlocal;
    if (igroup == atom->firstgroup) nlocal = atom->nfirst;
    int *type = atom->type;
    int *mask = atom->mask;

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if (rmass) dtfm = dtf / rmass[i];
        else dtfm = dtf / mass[type[i]];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }
    }
  }
}

void FixNVESpinSIB::reset_dt()
{
  double hbar = force->hplanck/LAMMPS_NS::MathConst::MY_2PI;  // eV/(rad.THz)

  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dts = 2 * update->dt * gil_factor / hbar;
}

void FixNVESpinSIB::as_vec(const double Scar[3], const double Fcar[3], const double alpha, double Acar[3])
{
  double Tcar[3];

  MathExtra::cross3(Scar, Fcar, Tcar);
  for (int i = 0; i < 3; i++){
    Acar[i] = -Fcar[i] - alpha * Tcar[i]; // +?
  }
}

void FixNVESpinSIB::bs_vec(const double Scar[3], const double Rcar[3], const double alpha, const double C_spin, double Bcar[3], const double spin_norm)
{
  double Tcar[3];

  MathExtra::cross3(Scar, Rcar, Tcar);
  for (int i = 0; i < 3; i++){
    Bcar[i] = C_spin * (-Rcar[i] - alpha * Tcar[i]); //* sqrt(spin_norm);
  }
}

void FixNVESpinSIB::rand_cut(const double delta_t, double R[3])
{
  double A_h;
  A_h = sqrt(2.0 * fabs(log(delta_t)));
  for (int i = 0; i < 3; i++){
    if(R[i]>A_h){
      R[i]=A_h;
    }else if (R[i]<-A_h)
    {
      R[i]=-A_h;
    }
  }
}

void FixNVESpinSIB::vec2mat(const double vec[3], double mat[3][3])
{
  mat[0][0]=0.0;
  mat[0][1]=vec[2];
  mat[0][2]=-vec[1];
  mat[1][0]=-vec[2];
  mat[1][1]=0.0;
  mat[1][2]=vec[0];
  mat[2][0]=vec[1];
  mat[2][1]=-vec[0];
  mat[2][2]=0.0;
}

void FixNVESpinSIB::invert_mat(const double mat[3][3], double mat_inv[3][3])
{
  // From lammps/src/RIGID/fix_rattle.cpp::void FixRattle::solve3x3exactly
  double determ, determinv;

  // calculate the determinant of the matrix

  determ = mat[0][0]*mat[1][1]*mat[2][2] + mat[0][1]*mat[1][2]*mat[2][0] +
    mat[0][2]*mat[1][0]*mat[2][1] - mat[0][0]*mat[1][2]*mat[2][1] -
    mat[0][1]*mat[1][0]*mat[2][2] - mat[0][2]*mat[1][1]*mat[2][0];

  // check if matrix is actually invertible

  if (determ == 0.0) error->one(FLERR,"Rattle determinant = 0.0");

  // calculate the inverse 3x3 matrix: A^(-1) = (ai_jk)

  determinv = 1.0/determ;
  mat_inv[0][0] =  determinv * (mat[1][1]*mat[2][2] - mat[1][2]*mat[2][1]);
  mat_inv[0][1] = -determinv * (mat[0][1]*mat[2][2] - mat[0][2]*mat[2][1]);
  mat_inv[0][2] =  determinv * (mat[0][1]*mat[1][2] - mat[0][2]*mat[1][1]);
  mat_inv[1][0] = -determinv * (mat[1][0]*mat[2][2] - mat[1][2]*mat[2][0]);
  mat_inv[1][1] =  determinv * (mat[0][0]*mat[2][2] - mat[0][2]*mat[2][0]);
  mat_inv[1][2] = -determinv * (mat[0][0]*mat[1][2] - mat[0][2]*mat[1][0]);
  mat_inv[2][0] =  determinv * (mat[1][0]*mat[2][1] - mat[1][1]*mat[2][0]);
  mat_inv[2][1] = -determinv * (mat[0][0]*mat[2][1] - mat[0][1]*mat[2][0]);
  mat_inv[2][2] =  determinv * (mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0]);
}
