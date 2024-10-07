#ifndef COMPILER_CODEGEN_H
#define COMPILER_CODEGEN_H

#include "compile.h"
#include <stdint.h>

/* Bitwise or these together to decide the optimization options. Optimizations may be ignored if they are likely to
 * introduce bugs into the kernel */
const uint64_t optimization_none = 0;
const uint64_t optimization_unroll = (1 << 0); /* Unroll loops */
const uint64_t optimization_inline = (1 << 1); /* Inline ops i.e. `a += b; a *= c;` -> `a = (a + b) * c;` */
const uint64_t optimization_split = (1 << 2);  /* Split singular ops between kernels */
const uint64_t optimization_fuse_a = (1 << 3); /* Aggregate along a-axis using simd types like float4 */
const uint64_t optimization_fuse_z = (1 << 4); /* Aggregate along z-axis using simd types like float4 */
const uint64_t optimization_fuse_y = (1 << 5); /* Aggregate along y-axis using simd types like float4 */
const uint64_t optimization_fuse_x = (1 << 6); /* Aggregate along x-axis using simd types like float4 */
const uint64_t optimization_memory = (1 << 7); /* Try to optimize memory accesses and cache. Expensive to compile */
const uint64_t optimization_kernel = (1 << 8); /* Reduce number of kernels being sent to the GPU */
/* All bits set to 1 -> All optimizations on. May significantly increase compile times */
const uint64_t optimization_all = UINT64_MAX;
extern void compile_op_group(kernel_t *kernel, const op_group_t *group, const uint64_t optimization);

#endif
