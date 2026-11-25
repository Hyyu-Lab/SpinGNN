#ifndef SPINGNN2_JIT_PREPARE_H
#define SPINGNN2_JIT_PREPARE_H

#include "spingnn2_c_api.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Preparation-only API. Inference APIs load already compiled images. The
 * caller owns all paths and source bytes. On failure the diagnostic string is
 * thread-local and valid until the next preparation call on that thread. */
SP2_API int sp2_jit_compile_cuda(const char* source, const char* architecture,
                                const char* output);
SP2_API const char* sp2_jit_prepare_error(void);
SP2_API const char* sp2_jit_cuda_toolchain(void);
/* Architecture selector also exported by libspingnn2, independent of NVRTC.
 * The CPU library reports an empty string. */
SP2_API const char* sp2_jit_target_arch(void);

#ifdef __cplusplus
}
#endif
#endif
