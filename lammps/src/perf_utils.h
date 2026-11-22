/* -*- c++ -*- ----------------------------------------------------------
   Opt-in, observation-only performance records for SpinGNN pair styles.
------------------------------------------------------------------------- */

#ifndef LMP_SPINGNN_PERF_UTILS_H
#define LMP_SPINGNN_PERF_UTILS_H

#include "error.h"

#include <torch/torch.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#ifdef SPINGNN_TORCH_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#endif

namespace LAMMPS_NS {

struct SpingnnPerfRecord {
  int rank = 0;
  long long step = 0;
  int call = 0;
  const char *pair = nullptr;
  const char *backend = nullptr;
  long long nlocal = 0;
  long long nghost = 0;
  long long nedges = 0;
  int global_virial = 0;
  int atom_virial = 0;
  int bec = 0;
  double graph_ms = 0.0;
  double pack_ms = 0.0;
  double h2d_ms = 0.0;
  double model_ms = 0.0;
  double d2h_ms = 0.0;
  double writeback_ms = 0.0;
  double pair_reverse_ms = 0.0;
  long long h2d_bytes = 0;
  long long d2h_bytes = 0;
  int cache_hit = 0;
  int cache_miss = 0;
  int sib_geom_cache_hit = 0;
  int sib_geom_cache_miss = 0;
  long long sib_geom_h2d_saved = 0;
  long long torch_alloc_peak = 0;
  long long torch_reserved_peak = 0;
  long long host_rss_peak = 0;
};

class SpingnnPerfState {
 public:
  SpingnnPerfState()
  {
    const char *value = std::getenv("SPINGNN_PERF");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0)
      enabled_ = false;
    else if (std::strcmp(value, "1") == 0)
      enabled_ = true;
    else
      invalid_value_ = value;
  }

  bool enabled() const { return enabled_; }

  void validate(Error *error) const
  {
    if (!invalid_value_.empty())
      error->all(FLERR, "SPINGNN_PERF must be exactly 0 or 1; got '" + invalid_value_ + "'");
  }

  int next_call(long long step)
  {
    if (step != last_step_)
    {
      last_step_ = step;
      calls_at_step_ = 0;
    }
    return ++calls_at_step_;
  }

 private:
  bool enabled_ = false;
  std::string invalid_value_;
  long long last_step_ = std::numeric_limits<long long>::min();
  int calls_at_step_ = 0;
};

inline long long spingnn_host_rss_peak_bytes()
{
#if defined(__linux__)
  // getrusage keeps the pre-exec RSS high-water mark of the Python launcher.
  // /proc reports the current executable's address space instead, so model
  // export allocations in the parent cannot inflate LAMMPS memory evidence.
  FILE *status = std::fopen("/proc/self/status", "r");
  if (status != nullptr) {
    char line[256];
    unsigned long long peak_kib = 0;
    while (std::fgets(line, sizeof(line), status) != nullptr) {
      if (std::sscanf(line, "VmHWM: %llu kB", &peak_kib) == 1 && peak_kib > 0 &&
          peak_kib <= static_cast<unsigned long long>(
              std::numeric_limits<long long>::max() / 1024LL)) {
        std::fclose(status);
        return static_cast<long long>(peak_kib) * 1024LL;
      }
    }
    std::fclose(status);
  }
#endif
  // Keep the portable fallback visible in captured logs. It is not directly
  // comparable to the Linux current-exec measurement after a large parent.
  static const bool fallback_reported = []() {
#if defined(_WIN32)
    const char *source = "unsupported";
#else
    const char *source = "getrusage";
#endif
    std::fprintf(stderr,
                 "SPINGNN_PERF_RSS_FALLBACK source=%s "
                 "reason=current_exec_VmHWM_unavailable\n", source);
    return true;
  }();
  (void) fallback_reported;
#if defined(_WIN32)
  return 0;
#else
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
  return static_cast<long long>(usage.ru_maxrss);
#else
  return static_cast<long long>(usage.ru_maxrss) * 1024LL;
#endif
#endif
}

class SpingnnPerfDeviceTimer {
 public:
  using Clock = std::chrono::steady_clock;

  explicit SpingnnPerfDeviceTimer(const torch::Device &device) :
      use_cuda_(device.is_cuda()), device_index_(device.has_index() ? device.index() : 0)
  {}

  void begin_h2d()
  {
    h2d_host_start_ = Clock::now();
#ifdef SPINGNN_TORCH_CUDA
    if (use_cuda_) h2d_start_.record(c10::cuda::getCurrentCUDAStream(device_index_));
#endif
  }

  void end_h2d_begin_model()
  {
    h2d_host_end_ = Clock::now();
    model_host_start_ = h2d_host_end_;
#ifdef SPINGNN_TORCH_CUDA
    if (use_cuda_) h2d_end_.record(c10::cuda::getCurrentCUDAStream(device_index_));
#endif
  }

  void end_model()
  {
    model_host_end_ = Clock::now();
#ifdef SPINGNN_TORCH_CUDA
    if (use_cuda_) model_end_.record(c10::cuda::getCurrentCUDAStream(device_index_));
#endif
  }

  void reset_allocator_peak()
  {
#ifdef SPINGNN_TORCH_CUDA
    if (use_cuda_) c10::cuda::CUDACachingAllocator::resetPeakStats(device_index_);
#endif
  }

  void resolve(SpingnnPerfRecord &record) const
  {
    record.h2d_ms = milliseconds(h2d_host_start_, h2d_host_end_);
    record.model_ms = milliseconds(model_host_start_, model_host_end_);
#ifdef SPINGNN_TORCH_CUDA
    if (use_cuda_)
    {
      // Model outputs have already crossed the existing .cpu() consumption
      // boundary, so these events are complete without adding a synchronization.
      record.h2d_ms = static_cast<double>(h2d_start_.elapsed_time(h2d_end_));
      record.model_ms = static_cast<double>(h2d_end_.elapsed_time(model_end_));
      const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_index_);
      const auto aggregate =
          static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
      record.torch_alloc_peak = stats.allocated_bytes[aggregate].peak;
      record.torch_reserved_peak = stats.reserved_bytes[aggregate].peak;
    }
#endif
  }

  static double milliseconds(const Clock::time_point &start, const Clock::time_point &end)
  {
    return std::chrono::duration<double, std::milli>(end - start).count();
  }

 private:
  bool use_cuda_;
  c10::DeviceIndex device_index_;
  Clock::time_point h2d_host_start_;
  Clock::time_point h2d_host_end_;
  Clock::time_point model_host_start_;
  Clock::time_point model_host_end_;
#ifdef SPINGNN_TORCH_CUDA
  c10::cuda::CUDAEvent h2d_start_{cudaEventDefault};
  c10::cuda::CUDAEvent h2d_end_{cudaEventDefault};
  c10::cuda::CUDAEvent model_end_{cudaEventDefault};
#endif
};

inline void spingnn_emit_perf_record(const SpingnnPerfRecord &record)
{
  std::fprintf(
      stderr,
      "SPINGNN_PERF schema=1 rank=%d step=%lld call=%d pair=%s backend=%s "
      "nlocal=%lld nghost=%lld nedges=%lld forwards=1 global_virial=%d "
      "atom_virial=%d bec=%d graph_ms=%.9f pack_ms=%.9f h2d_ms=%.9f "
      "model_ms=%.9f d2h_ms=%.9f writeback_ms=%.9f pair_reverse_ms=%.9f "
      "h2d_bytes=%lld d2h_bytes=%lld cache_hit=%d cache_miss=%d "
      "sib_geom_cache_hit=%d sib_geom_cache_miss=%d sib_geom_h2d_saved=%lld "
      "torch_alloc_peak=%lld torch_reserved_peak=%lld host_rss_peak=%lld\n",
      record.rank, record.step, record.call, record.pair, record.backend,
      record.nlocal, record.nghost, record.nedges, record.global_virial,
      record.atom_virial, record.bec, record.graph_ms, record.pack_ms,
      record.h2d_ms, record.model_ms, record.d2h_ms, record.writeback_ms,
      record.pair_reverse_ms, record.h2d_bytes, record.d2h_bytes,
      record.cache_hit, record.cache_miss,
      record.sib_geom_cache_hit, record.sib_geom_cache_miss,
      record.sib_geom_h2d_saved,
      record.torch_alloc_peak, record.torch_reserved_peak, record.host_rss_peak);
  std::fflush(stderr);
}

template <bool Enabled> class SpingnnPerfContext;

template <> class SpingnnPerfContext<false> {
 public:
  SpingnnPerfContext() = default;
  template <typename... Args> explicit SpingnnPerfContext(Args &&...) {}
  void graph_start() {}
  void graph_end_pack_start() {}
  void pack_end() {}
  void h2d_start() {}
  void h2d_end_model_start() {}
  void model_end() {}
  void d2h_start() {}
  void d2h_end() {}
  void writeback_start() {}
  void writeback_end() {}
  void reverse_start() {}
  void reverse_end() {}
  void finish(
      long long, long long, long long, bool, bool, bool, long long, long long,
      int, int, int, int, long long) {}
};

template <> class SpingnnPerfContext<true> {
 public:
  using Clock = SpingnnPerfDeviceTimer::Clock;

  SpingnnPerfContext(SpingnnPerfState &state, const torch::Device &device,
                     int rank, long long step, const char *pair, const char *backend) :
      device_timer_(device)
  {
    record_.rank = rank;
    record_.step = step;
    record_.call = state.next_call(step);
    record_.pair = pair;
    record_.backend = backend;
  }

  void graph_start()
  {
    device_timer_.reset_allocator_peak();
    phase_start_ = Clock::now();
  }

  void graph_end_pack_start()
  {
    const auto now = Clock::now();
    record_.graph_ms += SpingnnPerfDeviceTimer::milliseconds(phase_start_, now);
    phase_start_ = now;
  }

  void pack_end()
  {
    record_.pack_ms += SpingnnPerfDeviceTimer::milliseconds(phase_start_, Clock::now());
  }

  void h2d_start() { device_timer_.begin_h2d(); }
  void h2d_end_model_start() { device_timer_.end_h2d_begin_model(); }
  void model_end() { device_timer_.end_model(); }

  void d2h_start() { phase_start_ = Clock::now(); }
  void d2h_end()
  {
    record_.d2h_ms += SpingnnPerfDeviceTimer::milliseconds(phase_start_, Clock::now());
  }

  void writeback_start() { phase_start_ = Clock::now(); }
  void writeback_end()
  {
    record_.writeback_ms +=
        SpingnnPerfDeviceTimer::milliseconds(phase_start_, Clock::now());
  }

  void reverse_start() { phase_start_ = Clock::now(); }
  void reverse_end()
  {
    record_.pair_reverse_ms +=
        SpingnnPerfDeviceTimer::milliseconds(phase_start_, Clock::now());
  }

  void finish(long long nlocal, long long nghost, long long nedges,
              bool global_virial, bool atom_virial, bool bec,
              long long h2d_bytes, long long d2h_bytes,
              int cache_hit, int cache_miss, int sib_geom_cache_hit,
              int sib_geom_cache_miss, long long sib_geom_h2d_saved)
  {
    record_.nlocal = nlocal;
    record_.nghost = nghost;
    record_.nedges = nedges;
    record_.global_virial = global_virial ? 1 : 0;
    record_.atom_virial = atom_virial ? 1 : 0;
    record_.bec = bec ? 1 : 0;
    record_.h2d_bytes = h2d_bytes;
    record_.d2h_bytes = d2h_bytes;
    record_.cache_hit = cache_hit;
    record_.cache_miss = cache_miss;
    record_.sib_geom_cache_hit = sib_geom_cache_hit;
    record_.sib_geom_cache_miss = sib_geom_cache_miss;
    record_.sib_geom_h2d_saved = sib_geom_h2d_saved;
    device_timer_.resolve(record_);
    record_.host_rss_peak = spingnn_host_rss_peak_bytes();
    spingnn_emit_perf_record(record_);
  }

 private:
  SpingnnPerfRecord record_;
  SpingnnPerfDeviceTimer device_timer_;
  Clock::time_point phase_start_;
};

} // namespace LAMMPS_NS

#endif
