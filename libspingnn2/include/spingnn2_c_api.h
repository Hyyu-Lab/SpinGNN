#ifndef SPINGNN2_C_API_H
#define SPINGNN2_C_API_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define SP2_API __attribute__((visibility("default")))
#else
#define SP2_API
#endif

#define SP2_ABI_VERSION 1u
typedef struct sp2_model sp2_model;
typedef struct sp2_cache sp2_cache;
typedef struct sp2_result sp2_result;

enum sp2_precision { SP2_MODEL_PRECISION = 0, SP2_FLOAT32 = 32, SP2_FLOAT64 = 64 };
enum sp2_radial { SP2_ANALYTIC = 0, SP2_SPLINE = 1 };
enum sp2_execution_mode { SP2_AUTO = 0, SP2_GENERIC = 1, SP2_DIRECT = 2 };
enum sp2_execution_profile { SP2_CAPACITY = 0, SP2_SPEED = 1 };
enum sp2_eval_flag {
    SP2_SKIP_LATTICE_FORCES = 1u,
    /* Device neighbors only. Retain forward tensors for spin-only replay. */
    SP2_CACHE_STAGE1_FORWARD_ONLY = 2u
};
enum sp2_math_mode { SP2_MATH_IEEE = 0, SP2_MATH_TF32 = 1 };
enum sp2_neighbor_flag { SP2_NEIGHBOR_CUDA_GRAPH = 1u };
enum sp2_cuda_graph_status {
    SP2_GRAPH_EAGER = 0, SP2_GRAPH_CAPTURED = 1, SP2_GRAPH_REPLAYED = 2, SP2_GRAPH_FALLBACK = 3
};

/* All pointers designate contiguous host memory, borrowed for the call. Real
 * arrays use the model's selected precision. Index arrays are signed int32.
 * edge_index is [2,E], with all centers followed by all neighbors. The cell
 * contains row lattice vectors. Its first dimension is 1 or num_frames.
 * Omit the cell for open boundaries. Each supplied cell must be fully periodic.
 * Across ABI v1, initialize structs to zero and set struct_size = sizeof(...).
 */
typedef struct {
    uint32_t struct_size;
    int32_t num_atoms, num_edges, num_frames, num_cells;
    const void *positions, *spins;
    const int32_t *types, *edge_index, *batch;
    const void *cells, *edge_cell_shifts;
    /* Global atom count > 0 selects a partition of one frame. Owned atoms
     * precede ghosts; edges originate at owned atoms only. Forces include
     * ghost endpoint contributions. Center outputs span the owned atoms.
     * Hard BEC ASR needs the application's global mean subtraction. */
    int32_t num_owned;
    int64_t num_global;
} sp2_graph;

typedef struct {
    uint32_t struct_size;
    int32_t tile_atoms, radial_mode, profile;
    /* Zero selects the default bounded device cache, UINT64_MAX selects host storage. */
    uint64_t cache_device_limit_bytes;
    uint32_t eval_flags;
    int32_t execution_mode, execution_profile;
    /* Occupies the tail padding of the previous 40-byte structure. */
    uint32_t reserved;
    /* TF32 applies only to explicitly opted-in CUDA FP32 evaluations. */
    uint32_t math_mode;
} sp2_options;

typedef struct {
    uint32_t struct_size;
    uint64_t workspace_bytes, cache_host_bytes, peak_tile_edges;
    double geometry_seconds, mlp_seconds, tp_linear_seconds, reverse_seconds;
    uint64_t cache_device_bytes;
    double stage1_seconds, stage2_seconds;
    uint64_t stage1_cache_hits;
    uint64_t topology_metadata_bytes;
} sp2_diagnostics;

/* CUDA neighbor list. All array pointers reside on the active CUDA device.
 * Position and spin views have position_precision and up to two strides, in
 * elements. Spin components are (unit direction, moment); nonpositive moments
 * have zero direction. Atom types index the device type_map, which maps to
 * zero-based model types. ilist contains each owned atom once. numneigh and
 * firstneigh are indexed by atom index. neighbor_mask removes list flag bits,
 * and neighbor_slot_capacity is the number of accessible slots per row.
 * Increment topology_epoch when the list, atom order, or type map changes.
 * Reuse a nonzero lattice_epoch only while all owned and ghost positions are
 * fixed. A zero lattice_epoch disables tape caching. stream is the caller's
 * cudaStream_t (null selects CUDA's default stream). The optional flags
 * field requests CUDA Graphs for steady-state evaluation. */
typedef struct {
    uint32_t struct_size;
    int32_t num_atoms, num_owned, position_precision;
    int64_t num_global;
    const void *positions, *spins;
    int64_t position_atom_stride, position_component_stride;
    int64_t spin_atom_stride, spin_component_stride;
    const int32_t *types, *type_map, *ilist, *numneigh, *firstneigh;
    int32_t type_map_size, neighbor_mask, neighbor_slot_capacity;
    int64_t ilist_stride, numneigh_stride, neighbor_atom_stride, neighbor_slot_stride;
    uint64_t topology_epoch, lattice_epoch;
    void *stream;
    uint32_t flags;
} sp2_neighbors;

typedef struct {
    const void *sigma, *uepi, *nu, *kappa;
} sp2_device_evidence;

/* Model-precision views valid until the next call on this model. Read them on
 * the caller's stream. Forces and spin forces are [num_atoms,3], owned energy
 * is [num_owned], and virial and BEC are [num_owned,9] in row-major order.
 * Evidential sigma and uepi use nine values per owner; nu and kappa use one.
 * Unavailable fields are null. For hard BEC ASR the caller subtracts the global
 * mean and adds bec_asr_shift from the model metadata. Stage timings are
 * measured only with options.profile enabled. */
typedef struct {
    uint32_t struct_size;
    const void *forces, *spin_forces, *owned_energy, *atom_virial, *bec;
    sp2_diagnostics diagnostics;
    sp2_device_evidence force_evidence, spin_evidence;
    uint32_t cuda_graph_status;
} sp2_device_result;

/* Functions returning int report zero on success. Error text is thread local
 * and valid until the next failed call on that thread. Handles belong to
 * their creating library. Calls on a single model serialize workspace use.
 * The result owns each field view until sp2_result_free.
 */
SP2_API uint32_t sp2_abi_version(void);
SP2_API const char *sp2_last_error(void);
SP2_API const char *sp2_backend(void);
SP2_API int sp2_model_load(const char *manifest, int32_t precision, sp2_model **out);
/* Bind the CUDA backend to the caller's device ordinal. A process may use one
 * device ordinal per library instance. The standard load selects the default. */
SP2_API int sp2_model_load_device(const char *manifest, int32_t precision,
                                  int32_t device, sp2_model **out);
SP2_API void sp2_model_free(sp2_model *model);
SP2_API const char *sp2_model_info(const sp2_model *model);
/* Only prepared, matching artifacts may be attached. Both strings returned by
 * the plan API are owned by the library (thread-local for execution_plan). */
SP2_API int sp2_model_load_jit_artifact(sp2_model *model, const char *artifact_manifest);
SP2_API const char *sp2_model_execution_plan(sp2_model *model);
SP2_API int sp2_evaluate(sp2_model *model, const sp2_graph *graph, const sp2_options *options,
                         sp2_result **out);
SP2_API int sp2_evaluate_neighbors(sp2_model *model, const sp2_neighbors *neighbors,
                                   const sp2_options *options, sp2_device_result *out);
/* Set SP2_SKIP_LATTICE_FORCES to retain only the fields needed for a spin
 * update. A subsequent request for lattice derivatives recomputes the tape. */
SP2_API int sp2_stage1_cache_create(sp2_model *model, const sp2_graph *graph,
                                    const sp2_options *options, sp2_cache **out);
SP2_API int sp2_evaluate_stage2_cached(sp2_model *model, const sp2_cache *cache, const void *spins,
                                       sp2_result **out);
/* Override cache creation's eval_flags for this call. A frozen-lattice call
 * returns energies, spin forces and readouts; it omits lattice derivatives. */
SP2_API int sp2_evaluate_stage2_cached_flags(sp2_model *model, const sp2_cache *cache,
                                             const void *spins, uint32_t eval_flags,
                                             sp2_result **out);
SP2_API void sp2_cache_free(sp2_cache *cache);
SP2_API int32_t sp2_result_count(const sp2_result *result);
SP2_API int sp2_result_field(const sp2_result *result, int32_t index, const char **name,
                             const void **data, int32_t *rank, const int64_t **shape);
SP2_API int sp2_result_diagnostics(const sp2_result *result, sp2_diagnostics *out);
SP2_API void sp2_result_free(sp2_result *result);
#ifdef __cplusplus
}
#endif
#endif
