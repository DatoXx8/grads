#ifndef COMPILER_CODEGEN_H
#define COMPILER_CODEGEN_H

#include "compile.h"
#include <stdint.h>

/* TODO: Option to choose smaller local and global size as an optimization or default? Does that even enable any
 * optimizations? */
/* I think optimization_memory may be the only place where local_size matters */

/* Bitwise or these together to decide the optimization options. */
/* Values located in `codegen.c` */
extern const uint64_t optimization_none;   /* No optimizations */
extern const uint64_t optimization_unroll; /* Unroll loops */
extern const uint64_t optimization_inline; /* Inline ops i.e. `a += b; a *= c;` -> `a = (a + b) * c;` */
extern const uint64_t optimization_split;  /* Split singular ops between kernels */
extern const uint64_t optimization_fuse_a; /* Aggregate along a-axis using simd types like float4 */
extern const uint64_t optimization_fuse_z; /* Aggregate along z-axis using simd types like float4 */
extern const uint64_t optimization_fuse_y; /* Aggregate along y-axis using simd types like float4 */
extern const uint64_t optimization_fuse_x; /* Aggregate along x-axis using simd types like float4 */
extern const uint64_t optimization_memory; /* Try to optimize memory accesses and cache. Expensive to compile */
extern const uint64_t optimization_kernel; /* Reduce number of kernels being sent to the GPU */
extern const uint64_t optimization_all;    /* All optimizations */
extern void compile_op_group(kernel_t *kernel, const op_group_t *group, const uint64_t size_global,
                             const uint64_t size_local, const uint64_t optimization);

#endif
