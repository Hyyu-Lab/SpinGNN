// SPDX-License-Identifier: GPL-2.0-only
#ifdef PAIR_CLASS
PairStyle(spingnn2,PairSpinGNN2);
#else
#ifndef LMP_PAIR_SPINGNN2_H
#define LMP_PAIR_SPINGNN2_H

#include "pair.h"
#include "spingnn2_c_api.h"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace LAMMPS_NS {
class Fix;

class PairSpinGNN2 : public Pair {
 public:
  explicit PairSpinGNN2(LAMMPS *);
  ~PairSpinGNN2() override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  void compute(int, int) override;
  virtual void compute_sib_midpoint(bool reuse_lattice = true);
  void set_sib_lattice(Fix *, bool frozen);
  void *extract_peratom(const char *, int &) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

  // Exclusive pair wall time, halo communication, stage one, stage two,
  // midpoint/frozen cache hits, and evaluations since the last init.
  double timing[6]{};
  // Successful captures, replays, and capture/replay fallback calls.
  double graph_timing[3]{};
 protected:
  virtual void evaluate(bool midpoint, bool reuse);
  virtual void collect_edges();
  virtual int load_model(const char *path) { return sp2_model_load(path,precision,&model); }
  bool can_cache_sib_lattice() const;
  void add_neighbor(int, int);
  void checked(int);
  virtual void read_field(const std::string &, double *, size_t);
  std::unique_ptr<sp2_result,decltype(&sp2_result_free)> last_result{nullptr,sp2_result_free};
  struct Evidence {
    int columns;
    std::vector<double> values;
    std::vector<double *> rows;
  };
  std::map<std::string,Evidence> evidence;
  sp2_model *model = nullptr;
  sp2_options opts{};
  std::vector<int> species;
  std::string bec_name;
  double cutoff = 0, inv_hbar = 0, asr_shift = 0;
  int precision = SP2_FLOAT32;
  bool magnetic = false, hard_asr = false, frozen = false;
  bool cache_enabled = true, midpoint_call = false;
  bool cuda_graph_enabled = false;
  std::vector<int> centers, neighbors;
 private:
  template<class Real> void evaluate_host(bool, bool);
  Fix *sib_cache_owner = nullptr;
  sp2_cache *cache = nullptr;
  int cached_owned = -1;
  int64_t cached_global = 0;
  std::vector<int> cached_types, cached_edges;
  std::vector<double> cached_positions;
  std::vector<std::array<double,6>> endpoints;
};
}
#endif
#endif
