// SPDX-License-Identifier: GPL-2.0-only
#include "pair_spingnn2.h"

#include "atom.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "json.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string_view>

using namespace LAMMPS_NS;

PairSpinGNN2::PairSpinGNN2(LAMMPS *lmp) : Pair(lmp)
{
  manybody_flag = one_coeff = no_virial_fdotr_compute = 1;
  restartinfo = single_enable = 0;
  centroidstressflag = CENTROID_AVAIL;
  comm_reverse = comm_reverse_off = 6;
  opts.struct_size = sizeof(opts);
  opts.execution_mode = SP2_DIRECT;
}

PairSpinGNN2::~PairSpinGNN2()
{
  if (copymode) return;
  sp2_cache_free(cache);
  sp2_model_free(model);
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairSpinGNN2::checked(int status)
{
  if (status) error->one(FLERR, "SpinGNN2: {}", sp2_last_error());
}

void PairSpinGNN2::settings(int narg, char **arg)
{
  if (narg % 2) error->all(FLERR,"Expected key/value arguments to pair_style spingnn2");
  for (int i=0; i<narg; i+=2) {
    std::string key(arg[i]), value(arg[i+1]);
    if (key == "precision" && (value == "float32" || value == "float64"))
      precision = value == "float32" ? SP2_FLOAT32 : SP2_FLOAT64;
    else if (key == "math" && (value == "ieee" || value == "tf32"))
      opts.math_mode = value == "tf32" ? SP2_MATH_TF32 : SP2_MATH_IEEE;
    else if (key == "profile") opts.profile = utils::logical(FLERR,arg[i+1],false,lmp);
    else if (key == "cache") cache_enabled = utils::logical(FLERR,arg[i+1],false,lmp);
    else if (key == "cuda_graph") cuda_graph_enabled = utils::logical(FLERR,arg[i+1],false,lmp);
    else if (key == "tile") opts.tile_atoms = utils::inumeric(FLERR,arg[i+1],false,lmp);
    else error->all(FLERR,"Invalid spingnn2 option {} {}",key,value);
  }
  if (opts.math_mode && (precision != SP2_FLOAT32 || std::string(sp2_backend()) != "cuda"))
    error->all(FLERR,"TF32 requires a CUDA float32 model");
}

void PairSpinGNN2::coeff(int narg, char **arg)
{
  int n = atom->ntypes;
  if (narg != 4+n || strcmp(arg[0],"*") || strcmp(arg[1],"*"))
    error->all(FLERR,"Expected pair_coeff * * model.sp2.json artifact.json type1 ...");
  sp2_cache_free(cache); cache = nullptr;
  sp2_model_free(model); model = nullptr;
  checked(load_model(arg[2]));
  checked(sp2_model_load_jit_artifact(model,arg[3]));
  const auto info = json::parse(sp2_model_info(model));
  cutoff = info.at("r_max");
  evidence.clear();
  last_result.reset();
  for (const auto &value : info.at("prefixes")) {
    const auto prefix = value.get<std::string>();
    for (const auto *field : {"_sigma","_Uepi","_nu","_kappa"})
      evidence.emplace(prefix + field,Evidence{std::string(field) == "_nu" || std::string(field) == "_kappa" ? 1 : 9,{},{}});
  }
  magnetic = info.at("magnetic");
  bec_name = info.value("bec",std::string{});
  hard_asr = info.value("bec_asr",std::string{}) == "hard";
  asr_shift = info.value("bec_asr_shift",0.0);
  auto names = info.value("type_names",std::vector<std::string>{});
  species.resize(n+1);
  for (int i=1; i<=n; ++i) {
    std::string name(arg[3+i]);
    auto it = std::find(names.begin(),names.end(),name);
    species[i] = it != names.end() ? int(it-names.begin()) :
      utils::inumeric(FLERR,arg[3+i],false,lmp);
    if (species[i] < 0 || species[i] >= info.at("num_types").get<int>())
      error->all(FLERR,"Unknown SpinGNN2 model species {}",name);
  }
  if (!allocated) {
    allocated = 1;
    memory->create(setflag,n+1,n+1,"spingnn2:setflag");
    memory->create(cutsq,n+1,n+1,"spingnn2:cutsq");
  }
  for (int i=1; i<=n; ++i)
    for (int j=i; j<=n; ++j) setflag[i][j] = 1;
}

double PairSpinGNN2::init_one(int, int) { return cutoff; }

void PairSpinGNN2::init_style()
{
  if (strcmp(update->unit_style,"metal"))
    error->all(FLERR,"spingnn2 models require metal units");
  if (magnetic && !atom->sp_flag) error->all(FLERR,"Magnetic model requires atom_style spin");
  if (!bec_name.empty() && !atom->bec_flag) error->all(FLERR,"DREAM requires atom_style bec");
  inv_hbar = MathConst::MY_2PI / force->hplanck;
  neighbor->add_request(this,NeighConst::REQ_FULL);
  sib_cache_owner = nullptr;
  frozen = false;
  sp2_cache_free(cache); cache = nullptr;
  std::fill_n(timing,6,0.);
  std::fill_n(graph_timing,3,0.);
}

void PairSpinGNN2::set_sib_lattice(Fix *owner, bool fixed_lattice)
{
  sib_cache_owner = owner;
  frozen = fixed_lattice;
}

bool PairSpinGNN2::can_cache_sib_lattice() const
{
  if (!cache_enabled || !sib_cache_owner) return false;

  using namespace FixConst;
  constexpr int run_hooks = INITIAL_INTEGRATE | POST_INTEGRATE | PRE_EXCHANGE | PRE_NEIGHBOR |
    POST_NEIGHBOR | PRE_FORCE | PRE_REVERSE | POST_FORCE | FINAL_INTEGRATE | END_OF_STEP | POST_RUN;
  for (int i = 0; i < modify->nfix; ++i) {
    const auto *fix = modify->fix[i];
    if (fix == sib_cache_owner) continue;
    if (fix->time_integrate || fix->box_change != Fix::NO_BOX_CHANGE) return false;
    if (fix->datamask_modify != ALL_MASK && (fix->datamask_modify & X_MASK)) return false;

    // Legacy host fixes inherit ALL_MASK. These built-ins only sample pair
    // data or update forces and spins, even though they register run hooks.
    const std::string_view style(fix->style);
    if (style == "pair" || style == "langevin" || style == "langevin/kk" ||
        style == "langevin/kk/device" || style == "langevin/kk/host" ||
        style == "precession/spin" || style == "setforce/spin")
      continue;
    if ((fix->datamask_modify & X_MASK) || (modify->fmask[i] & run_hooks)) return false;
  }
  return true;
}

void PairSpinGNN2::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);
  evaluate(false,frozen);
}

void PairSpinGNN2::compute_sib_midpoint(bool reuse_lattice)
{
  evaluate(true,reuse_lattice);
}

void PairSpinGNN2::add_neighbor(int i, int raw)
{
  int j = raw & NEIGHMASK;
  double rsq = 0.;
  for (int k=0; k<3; ++k) {
    const double d = atom->x[j][k]-atom->x[i][k];
    rsq += d*d;
  }
  if (rsq < cutoff*cutoff) {
    centers.push_back(i);
    neighbors.push_back(j);
  }
}

void PairSpinGNN2::collect_edges()
{
  for (int ii=0; ii<list->inum; ++ii) {
    const int i = list->ilist[ii];
    for (int k=0; k<list->numneigh[i]; ++k) add_neighbor(i,list->firstneigh[i][k]);
  }
}

void PairSpinGNN2::evaluate(bool midpoint, bool reuse)
{
  if (precision == SP2_FLOAT64) evaluate_host<double>(midpoint,reuse);
  else evaluate_host<float>(midpoint,reuse);
}

template<class Real> void PairSpinGNN2::evaluate_host(bool midpoint, bool reuse)
{
  const double start = MPI_Wtime(), comm_start = timing[1];
  const int owned = atom->nlocal, all = owned + atom->nghost;
  centers.clear(); neighbors.clear();
  collect_edges();
  int edges = centers.size();
  centers.insert(centers.end(),neighbors.begin(),neighbors.end());
  std::vector<Real> x(3ULL*all), sp(magnetic ? std::max(3*all,1) : 0);
  std::vector<int> types(all);
  for (int i=0; i<all; ++i) {
    types[i] = species[atom->type[i]];
    for (int k=0; k<3; ++k) {
      x[3*i+k] = atom->x[i][k];
      if (magnetic) sp[3*i+k] = atom->sp[i][3]>0 ? atom->sp[i][k] : 0;
    }
  }
  sp2_graph graph{};
  graph.struct_size = sizeof(graph);
  graph.num_frames = 1;
  graph.num_atoms = all;
  graph.num_owned = owned;
  graph.num_global = atom->natoms;
  graph.num_edges = edges;
  graph.positions = x.data(); graph.spins = sp.data();
  graph.types = types.data(); graph.edge_index = centers.data();
  bool use_cache = can_cache_sib_lattice();
  bool hit = use_cache && reuse && cache && cached_types == types && cached_edges == centers &&
    cached_owned == owned && cached_global == atom->natoms && cached_positions.size() == x.size() &&
    std::equal(x.begin(),x.end(),cached_positions.begin());
  if (use_cache && !hit) {
    sp2_cache_free(cache); cache = nullptr;
    const double t = MPI_Wtime();
    checked(sp2_stage1_cache_create(model,&graph,&opts,&cache));
    timing[2] += MPI_Wtime()-t;
    cached_owned = owned; cached_global = atom->natoms;
    cached_positions.assign(x.begin(),x.end());
    cached_types = types; cached_edges = centers;
  }
  sp2_result *raw = nullptr;
  if (use_cache)
    checked(sp2_evaluate_stage2_cached_flags(model,cache,sp.data(),
            midpoint ? SP2_SKIP_LATTICE_FORCES : 0,&raw));
  else checked(sp2_evaluate(model,&graph,&opts,&raw));
  std::unique_ptr<sp2_result,decltype(&sp2_result_free)> result(raw,sp2_result_free);
  sp2_diagnostics diag{}; diag.struct_size = sizeof(diag);
  checked(sp2_result_diagnostics(raw,&diag));
  timing[2] += diag.stage1_seconds; timing[3] += diag.stage2_seconds;
  timing[4] += hit; timing[5]++;
  const Real *f=nullptr, *sf=nullptr, *energy=nullptr, *va=nullptr, *bec=nullptr;
  for (int i=0; i<sp2_result_count(raw); ++i) {
    const char *name; const void *data; int32_t rank; const int64_t *shape;
    checked(sp2_result_field(raw,i,&name,&data,&rank,&shape));
    auto ptr = static_cast<const Real *>(data);
    if (!strcmp(name,"forces")) f=ptr;
    else if (!strcmp(name,"spin_forces")) sf=ptr;
    else if (!strcmp(name,"owned_energy")) energy=ptr;
    else if (!strcmp(name,"atom_virial")) va=ptr;
    else if (name == bec_name) bec=ptr;
  }
  endpoints.resize(all);
  midpoint_call = midpoint;
  for (int i=0; i<all; ++i)
    for (int k=0; k<3; ++k) {
      endpoints[i][k] = midpoint ? 0. : f[3*i+k];
      endpoints[i][3+k] = sf ? sf[3*i+k]*inv_hbar : 0.;
    }
  // A center's energy depends on ghost endpoints even with newton off.
  // Reverse only this pair's scratch, then update owned atoms exactly once.
  double t = MPI_Wtime();
  comm->reverse_comm(this,midpoint ? 3 : 6);
  timing[1] += MPI_Wtime()-t;
  double mean[9]{};
  if (!midpoint && hard_asr) {
    for (int i=0; i<owned; ++i)
      for (int k=0; k<9; ++k) mean[k] += bec[9*i+k];
    t = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE,mean,9,MPI_DOUBLE,MPI_SUM,world);
    timing[1] += MPI_Wtime()-t;
    for (double &v : mean) v = v / atom->natoms - asr_shift;
  }
  const int order[9]{0,4,8,1,2,5,3,6,7};
  for (int i=0; i<owned; ++i) {
    for (int k=0; k<3; ++k) {
      if (!midpoint) atom->f[i][k] += endpoints[i][k];
      if (magnetic) {
        if (midpoint) atom->fm[i][k] = 0.;
        if (atom->sp[i][3] > 0.) atom->fm[i][k] += endpoints[i][3+k];
      }
    }
    if (midpoint) continue;
    if (eflag_global) eng_vdwl += energy[i];
    if (eflag_atom) eatom[i] = energy[i];
    for (int k=0; k<9; ++k) {
      int p = order[k];
      if (vflag_global && k<6) virial[k] += 0.5*(va[9*i+p]+va[9*i+3*(p%3)+p/3]);
      if (vflag_atom && k<6) vatom[i][k] = va[9*i+p];
      if (cvflag_atom) cvatom[i][k] = va[9*i+p];
      if (bec) atom->bec[i][k] = bec[9*i+k]-mean[k];
    }
  }
  if (!midpoint && !evidence.empty()) last_result = std::move(result);
  timing[0] += MPI_Wtime()-start-(timing[1]-comm_start);
}

int PairSpinGNN2::pack_reverse_comm(int n, int first, double *buf)
{
  int m=0;
  for (int i=first; i<first+n; ++i)
    for (int k=midpoint_call ? 3 : 0; k<6; ++k) buf[m++] = endpoints[i][k];
  return m;
}
void PairSpinGNN2::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m=0;
  for (int i=0; i<n; ++i)
    for (int k=midpoint_call ? 3 : 0; k<6; ++k) endpoints[list[i]][k] += buf[m++];
}

void *PairSpinGNN2::extract_peratom(const char *name,int &columns)
{
  auto found = evidence.find(name);
  if (found == evidence.end()) return nullptr;
  auto &field = found->second;
  columns = field.columns == 1 ? 0 : field.columns;
  field.values.resize(size_t(atom->nlocal)*field.columns);
  read_field(name,field.values.data(),field.values.size());
  if (!columns) return field.values.data();
  field.rows.resize(atom->nlocal);
  for (int i=0; i<atom->nlocal; ++i) field.rows[i] = field.values.data()+size_t(i)*columns;
  return field.rows.data();
}

void PairSpinGNN2::read_field(const std::string &key,double *out,size_t size)
{
  // fix pair queries the column count before the first force evaluation.
  if (!last_result || !size) return;
  for (int i=0; i<sp2_result_count(last_result.get()); ++i) {
    const char *name; const void *data; int32_t rank; const int64_t *shape;
    checked(sp2_result_field(last_result.get(),i,&name,&data,&rank,&shape));
    if (name != key) continue;
    if (precision == SP2_FLOAT64) std::copy_n(static_cast<const double *>(data),size,out);
    else std::copy_n(static_cast<const float *>(data),size,out);
    return;
  }
  error->one(FLERR,"Missing SpinGNN2 readout {}",key);
}
