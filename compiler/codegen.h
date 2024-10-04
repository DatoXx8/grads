#ifndef COMPILER_CODEGEN_H
#define COMPILER_CODEGEN_H

#include "compile.h"
#include <stdint.h>

/* Bitwise or these together to decide the optimization options. Optimizations may be ignored if they are likely to
 * introduce bugs into the kernel */
const uint64_t optimization_none = 0;
const uint64_t optimization_unroll = (1 << 0); /* Unroll loops */
const uint64_t optimization_inline = (1 << 1); /* Inline ops i.e. `a += b; a *= c` -> `a = (a + b) * c` */
const uint64_t optimization_split = (1 << 2);  /* Split singular ops between kernels */
const uint64_t optimization_fuse_a = (1 << 3); /* Aggregate along axis a using simd types like float4 */
const uint64_t optimization_fuse_z = (1 << 4); /* Aggregate along axis z using simd types like float4 */
const uint64_t optimization_fuse_y = (1 << 5); /* Aggregate along axis y using simd types like float4 */
const uint64_t optimization_fuse_x = (1 << 6); /* Aggregate along axis x using simd types like float4 */
const uint64_t optimization_memory = (1 << 7); /* Try to optimize memory accesses and cache. May be slow initially */
const uint64_t optimization_kernel = (1 << 8); /* Reduce number of kernels being sent to the GPU */
const uint64_t optimization_all =
    UINT64_MAX; /* All bits set to 1 -> All optimizations on. Significantly icreases compile times */
extern char *compile_op_group(op_group_t *group, uint64_t optimization);

#endif
