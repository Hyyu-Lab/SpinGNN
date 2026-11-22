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

/* ----------------------------------------------------------------------
   Contributing author: Christina Payne (Vanderbilt U)
                        Stan Moore (Sandia) for dipole terms

   BEC coupling uses the accumulated per-atom bec_dipole as a path
   polarization increment P = integral Z(R) dR from the most recent
   read_bec or bec_dipole_reset reference.  Constant-field energy is -E.P.
   For configuration-dependent BECs this is an explicitly path-dependent
   definition, not a globally conservative polarization state function.
------------------------------------------------------------------------- */

#include "fix_efield.h"

#include <cstring>
#include "atom.h"
#include "update.h"
#include "domain.h"
#include "comm.h"
#include "modify.h"
#include "force.h"
#include "respa.h"
#include "input.h"
#include "variable.h"
#include "region.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NONE,CONSTANT,EQUAL,ATOM};
enum{COUPLING_AUTO,COUPLING_CHARGE,COUPLING_BEC,COUPLING_BOTH};

/* ---------------------------------------------------------------------- */

FixEfield::FixEfield(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), xstr(nullptr), ystr(nullptr), zstr(nullptr),
  estr(nullptr), idregion(nullptr), efield(nullptr)
{
  if (narg < 6) error->all(FLERR,"Illegal fix efield command");

  dynamic_group_allow = 1;
  vector_flag = 1;
  scalar_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extvector = 1;
  extscalar = 1;
  respa_level_support = 1;
  ilevel_respa = 0;
  energy_global_flag = 1;
  virial_global_flag = virial_peratom_flag = 1;

  qe2f = force->qe2f;
  xstr = ystr = zstr = nullptr;

  if (utils::strmatch(arg[3],"^v_")) {
    xstr = utils::strdup(arg[3]+2);
  } else {
    ex = qe2f * utils::numeric(FLERR,arg[3],false,lmp);
    xstyle = CONSTANT;
  }

  if (utils::strmatch(arg[4],"^v_")) {
    ystr = utils::strdup(arg[4]+2);
  } else {
    ey = qe2f * utils::numeric(FLERR,arg[4],false,lmp);
    ystyle = CONSTANT;
  }

  if (utils::strmatch(arg[5],"^v_")) {
    zstr = utils::strdup(arg[5]+2);
  } else {
    ez = qe2f * utils::numeric(FLERR,arg[5],false,lmp);
    zstyle = CONSTANT;
  }

  // optional args

  iregion = -1;
  idregion = nullptr;
  estr = nullptr;
  coupling = COUPLING_AUTO;

  int iarg = 6;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix efield command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix efield does not exist");
      idregion = utils::strdup(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"energy") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix efield command");
      if (utils::strmatch(arg[iarg+1],"^v_")) {
        estr = utils::strdup(arg[iarg+1]+2);
      } else error->all(FLERR,"Illegal fix efield command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"coupling") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix efield command");
      if (strcmp(arg[iarg+1],"auto") == 0) coupling = COUPLING_AUTO;
      else if (strcmp(arg[iarg+1],"charge") == 0) coupling = COUPLING_CHARGE;
      else if (strcmp(arg[iarg+1],"bec") == 0) coupling = COUPLING_BEC;
      else if (strcmp(arg[iarg+1],"both") == 0) coupling = COUPLING_BOTH;
      else error->all(FLERR,"Illegal fix efield command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"piezo") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix efield command");
      if (strcmp(arg[iarg+1],"yes") == 0)
        error->all(FLERR,"Fix efield piezo yes is unavailable: model no longer provides piezoelectric");
      else if (strcmp(arg[iarg+1],"no") != 0)
        error->all(FLERR,"Illegal fix efield command");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix efield command");
  }

  force_flag = 0;
  fsum[0] = fsum[1] = fsum[2] = fsum[3] = 0.0;

  maxatom = atom->nmax;
  memory->create(efield,maxatom,4,"efield:efield");
}

/* ---------------------------------------------------------------------- */

FixEfield::~FixEfield()
{
  delete [] xstr;
  delete [] ystr;
  delete [] zstr;
  delete [] estr;
  delete [] idregion;
  memory->destroy(efield);
}

/* ---------------------------------------------------------------------- */

int FixEfield::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixEfield::init()
{
  qflag = becflag = 0;
  muflag = 0;
  if (atom->mu_flag && atom->torque_flag) muflag = 1;

  if (coupling == COUPLING_AUTO || coupling == COUPLING_BOTH) {
    if (atom->q_flag) qflag = 1;
    if (atom->bec_flag) becflag = 1;
  } else if (coupling == COUPLING_CHARGE) {
    if (!atom->q_flag)
      error->all(FLERR,"Fix efield coupling charge requires atom attribute q");
    qflag = 1;
  } else if (coupling == COUPLING_BEC) {
    if (!atom->bec_flag)
      error->all(FLERR,"Fix efield coupling bec requires atom attribute bec");
    becflag = 1;
  }

  if (coupling == COUPLING_BOTH && (!atom->q_flag || !atom->bec_flag))
    error->all(FLERR,"Fix efield coupling both requires atom attributes q and bec");
  if (coupling == COUPLING_AUTO && qflag && becflag && comm->me == 0)
    error->warning(FLERR,"Fix efield coupling auto applies both q and BEC; "
                   "use coupling charge or coupling bec to select only one");
  if (!qflag && !muflag && !becflag)
    error->all(FLERR,"Fix efield requires atom attribute q or mu or bec");

  // check variables

  if (xstr) {
    xvar = input->variable->find(xstr);
    if (xvar < 0)
      error->all(FLERR,"Variable name for fix efield does not exist");
    if (input->variable->equalstyle(xvar)) xstyle = EQUAL;
    else if (input->variable->atomstyle(xvar)) xstyle = ATOM;
    else error->all(FLERR,"Variable for fix efield is invalid style");
  }
  if (ystr) {
    yvar = input->variable->find(ystr);
    if (yvar < 0)
      error->all(FLERR,"Variable name for fix efield does not exist");
    if (input->variable->equalstyle(yvar)) ystyle = EQUAL;
    else if (input->variable->atomstyle(yvar)) ystyle = ATOM;
    else error->all(FLERR,"Variable for fix efield is invalid style");
  }
  if (zstr) {
    zvar = input->variable->find(zstr);
    if (zvar < 0)
      error->all(FLERR,"Variable name for fix efield does not exist");
    if (input->variable->equalstyle(zvar)) zstyle = EQUAL;
    else if (input->variable->atomstyle(zvar)) zstyle = ATOM;
    else error->all(FLERR,"Variable for fix efield is invalid style");
  }
  if (estr) {
    evar = input->variable->find(estr);
    if (evar < 0)
      error->all(FLERR,"Variable name for fix efield does not exist");
    if (input->variable->atomstyle(evar)) estyle = ATOM;
    else error->all(FLERR,"Variable for fix efield is invalid style");
  } else estyle = NONE;


  // set index and check validity of region

  if (iregion >= 0) {
    iregion = domain->find_region(idregion);
    if (iregion == -1)
      error->all(FLERR,"Region ID for fix aveforce does not exist");
  }

  if (xstyle == ATOM || ystyle == ATOM || zstyle == ATOM)
    varflag = ATOM;
  else if (xstyle == EQUAL || ystyle == EQUAL || zstyle == EQUAL)
    varflag = EQUAL;
  else varflag = CONSTANT;

  if (muflag && varflag == ATOM)
    error->all(FLERR,"Fix efield with dipoles cannot use atom-style variables");

  if (muflag && update->whichflag == 2 && comm->me == 0)
    error->warning(FLERR,
                   "The minimizer does not re-orient dipoles "
                   "when using fix efield");

  if (varflag == CONSTANT && estyle != NONE)
    error->all(FLERR,"Cannot use variable energy with "
               "constant efield in fix efield");
  if ((varflag == EQUAL || varflag == ATOM) &&
      update->whichflag == 2 && estyle == NONE)
    error->all(FLERR,"Must use variable energy with fix efield");

  if (utils::strmatch(update->integrate_style,"^respa")) {
    ilevel_respa = ((Respa *) update->integrate)->nlevels-1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level,ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixEfield::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style,"^verlet"))
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag,ilevel_respa,0);
    ((Respa *) update->integrate)->copy_f_flevel(ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixEfield::min_setup(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   apply electric-field forces through q, BEC, and/or permanent dipoles
------------------------------------------------------------------------- */

void FixEfield::post_force(int vflag)
{
  double **f = atom->f;
  double *q = atom->q;
  int *mask = atom->mask;
  imageint *image = atom->image;
  int nlocal = atom->nlocal;

  // virial setup

  v_init(vflag);

  // reallocate efield array if necessary

  if (varflag == ATOM && atom->nmax > maxatom) {
    maxatom = atom->nmax;
    memory->destroy(efield);
    memory->create(efield,maxatom,4,"efield:efield");
  }

  // update region if necessary

  Region *region = nullptr;
  if (iregion >= 0) {
    region = domain->regions[iregion];
    region->prematch();
  }

  // fsum[0] = "potential energy" for added force
  // fsum[123] = extra force added to atoms

  fsum[0] = fsum[1] = fsum[2] = fsum[3] = 0.0;
  force_flag = 0;

  double **x = atom->x;
  double fx,fy,fz;
  double v[6];

  // constant efield

  if (varflag == CONSTANT) {
    double unwrap[3];

    // charge interactions
    // force = qE, potential energy = -F dot x in unwrapped coords

    if (qflag) {
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
          fx = q[i]*ex;
          fy = q[i]*ey;
          fz = q[i]*ez;
          f[i][0] += fx;
          f[i][1] += fy;
          f[i][2] += fz;

          domain->unmap(x[i],image[i],unwrap);
          fsum[0] -= fx*unwrap[0]+fy*unwrap[1]+fz*unwrap[2];
          fsum[1] += fx;
          fsum[2] += fy;
          fsum[3] += fz;
          if (evflag) {
            v[0] = fx*unwrap[0];
            v[1] = fy*unwrap[1];
            v[2] = fz*unwrap[2];
            v[3] = fx*unwrap[1];
            v[4] = fx*unwrap[2];
            v[5] = fy*unwrap[2];
            v_tally(i,v);
          }
        }
    }

    // dipole interactions
    // no force, torque = mu cross E, potential energy = -mu dot E

    if (muflag) {
      double **mu = atom->mu;
      double **t = atom->torque;
      double tx,ty,tz;
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
          tx = ez*mu[i][1] - ey*mu[i][2];
          ty = ex*mu[i][2] - ez*mu[i][0];
          tz = ey*mu[i][0] - ex*mu[i][1];
          t[i][0] += tx;
          t[i][1] += ty;
          t[i][2] += tz;
          fsum[0] -= mu[i][0]*ex + mu[i][1]*ey + mu[i][2]*ez;
        }
    }

    // Born effective charge interactions
    // force = transpose(bec) E, potential energy = -E dot integrated dipole

    if (becflag) {
      double **bec = atom->bec;
      double **bec_dipole = atom->bec_dipole;
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
          fx = bec[i][0]*ex+bec[i][3]*ey+bec[i][6]*ez;
          fy = bec[i][1]*ex+bec[i][4]*ey+bec[i][7]*ez;
          fz = bec[i][2]*ex+bec[i][5]*ey+bec[i][8]*ez;
          f[i][0] += fx;
          f[i][1] += fy;
          f[i][2] += fz;

          fsum[0] -= ex*bec_dipole[i][0] + ey*bec_dipole[i][1] +
            ez*bec_dipole[i][2];
          fsum[1] += fx;
          fsum[2] += fy;
          fsum[3] += fz;
          if (evflag) {
            domain->unmap(x[i],image[i],unwrap);
            v[0] = fx*unwrap[0];
            v[1] = fy*unwrap[1];
            v[2] = fz*unwrap[2];
            v[3] = fx*unwrap[1];
            v[4] = fx*unwrap[2];
            v[5] = fy*unwrap[2];
            v_tally(i,v);
          }
        }
    }

  // variable efield, wrap with clear/add
  // potential energy = evar if defined, else 0.0

  } else {

    modify->clearstep_compute();

    if (xstyle == EQUAL) ex = qe2f * input->variable->compute_equal(xvar);
    else if (xstyle == ATOM)
      input->variable->compute_atom(xvar,igroup,&efield[0][0],4,0);
    if (ystyle == EQUAL) ey = qe2f * input->variable->compute_equal(yvar);
    else if (ystyle == ATOM)
      input->variable->compute_atom(yvar,igroup,&efield[0][1],4,0);
    if (zstyle == EQUAL) ez = qe2f * input->variable->compute_equal(zvar);
    else if (zstyle == ATOM)
      input->variable->compute_atom(zvar,igroup,&efield[0][2],4,0);
    if (estyle == ATOM)
      input->variable->compute_atom(evar,igroup,&efield[0][3],4,0);

    modify->addstep_compute(update->ntimestep + 1);

    // charge interactions
    // force = qE

    if (qflag) {
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
          if (xstyle == ATOM) fx = qe2f * q[i]*efield[i][0];
          else fx = q[i]*ex;
          f[i][0] += fx;
          fsum[1] += fx;
          if (ystyle == ATOM) fy = qe2f * q[i]*efield[i][1];
          else fy = q[i]*ey;
          f[i][1] += fy;
          fsum[2] += fy;
          if (zstyle == ATOM) fz = qe2f * q[i]*efield[i][2];
          else fz = q[i]*ez;
          f[i][2] += fz;
          fsum[3] += fz;
          if (evflag) {
            double unwrap[3];
            domain->unmap(x[i],image[i],unwrap);
            v[0] = fx*unwrap[0];
            v[1] = fy*unwrap[1];
            v[2] = fz*unwrap[2];
            v[3] = fx*unwrap[1];
            v[4] = fx*unwrap[2];
            v[5] = fy*unwrap[2];
            v_tally(i,v);
          }
        }
    }

    // dipole interactions
    // no force, torque = mu cross E

    if (muflag) {
      double **mu = atom->mu;
      double **t = atom->torque;
      double tx,ty,tz;
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
          tx = ez*mu[i][1] - ey*mu[i][2];
          ty = ex*mu[i][2] - ez*mu[i][0];
          tz = ey*mu[i][0] - ex*mu[i][1];
          t[i][0] += tx;
          t[i][1] += ty;
          t[i][2] += tz;
        }
    }

    // Born effective charge interactions
    // force = transpose(bec) E
    if (becflag) {
      double **bec = atom->bec;
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
          fx = 0;
          fy = 0;
          fz = 0;
          if (xstyle == ATOM) {
            fx += qe2f * bec[i][0]*efield[i][0];
            fy += qe2f * bec[i][1]*efield[i][0];
            fz += qe2f * bec[i][2]*efield[i][0];
          }
          else {
            fx += bec[i][0]*ex;
            fy += bec[i][1]*ex;
            fz += bec[i][2]*ex;
          }
          if (ystyle == ATOM) {
            fx += qe2f * bec[i][3]*efield[i][1];
            fy += qe2f * bec[i][4]*efield[i][1];
            fz += qe2f * bec[i][5]*efield[i][1];
          }
          else {
            fx += bec[i][3]*ey;
            fy += bec[i][4]*ey;
            fz += bec[i][5]*ey;
          }
          if (zstyle == ATOM) {
            fx += qe2f * bec[i][6]*efield[i][2];
            fy += qe2f * bec[i][7]*efield[i][2];
            fz += qe2f * bec[i][8]*efield[i][2];
          }
          else {
            fx += bec[i][6]*ez;
            fy += bec[i][7]*ez;
            fz += bec[i][8]*ez;
          }
          f[i][0] += fx;
          fsum[1] += fx;
          f[i][1] += fy;
          fsum[2] += fy;
          f[i][2] += fz;
          fsum[3] += fz;
          if (evflag) {
            double unwrap[3];
            domain->unmap(x[i],image[i],unwrap);
            v[0] = fx*unwrap[0];
            v[1] = fy*unwrap[1];
            v[2] = fz*unwrap[2];
            v[3] = fx*unwrap[1];
            v[4] = fx*unwrap[2];
            v[5] = fy*unwrap[2];
            v_tally(i,v);
          }
        }
    }

    // An atom-style energy variable describes the energy of the complete
    // fix, so tally it once even when both charge and BEC are enabled.

    if (estyle == ATOM) {
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
          fsum[0] += efield[i][3];
        }
    }
  }
}

void FixEfield::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixEfield::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixEfield::memory_usage()
{
  double bytes = 0.0;
  if (varflag == ATOM) bytes = atom->nmax*4 * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   return energy added by fix
------------------------------------------------------------------------- */

double FixEfield::compute_scalar()
{
  if (force_flag == 0) {
    MPI_Allreduce(fsum,fsum_all,4,MPI_DOUBLE,MPI_SUM,world);
    force_flag = 1;
  }
  return fsum_all[0];
}

/* ----------------------------------------------------------------------
   return total extra force due to fix
------------------------------------------------------------------------- */

double FixEfield::compute_vector(int n)
{
  if (force_flag == 0) {
    MPI_Allreduce(fsum,fsum_all,4,MPI_DOUBLE,MPI_SUM,world);
    force_flag = 1;
  }
  return fsum_all[n+1];
}
