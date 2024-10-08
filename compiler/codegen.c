#include "codegen.h"
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include "compile.h"

const uint64_t optimization_none = 0;          /* No optimizations */
const uint64_t optimization_unroll = (1 << 0); /* Unroll loops */
const uint64_t optimization_inline = (1 << 1); /* Inline ops i.e. `a += b; a *= c;` -> `a = (a + b) * c;` */
const uint64_t optimization_split = (1 << 2);  /* Split singular ops between kernels */
const uint64_t optimization_fuse_a = (1 << 3); /* Aggregate along a-axis using simd types like float4 */
const uint64_t optimization_fuse_z = (1 << 4); /* Aggregate along z-axis using simd types like float4 */
const uint64_t optimization_fuse_y = (1 << 5); /* Aggregate along y-axis using simd types like float4 */
const uint64_t optimization_fuse_x = (1 << 6); /* Aggregate along x-axis using simd types like float4 */
const uint64_t optimization_memory = (1 << 7); /* Try to optimize memory accesses and cache. Expensive to compile */
const uint64_t optimization_kernel = (1 << 8); /* Reduce number of kernels being sent to the GPU */
const uint64_t optimization_all = UINT64_MAX;  /* All optimizations */

/* TODO: Choose reasonable limit based on some data and not just a gut feeling */
const uint64_t padding = 1024;
/* The length of the source can just be calculated so there is no need to store that */
static inline void source_expand(char **source, char **source_curr, uint64_t *source_cap) {
    /* MAYBE: Not sure if I should allow `*source` and things like that to be NULL so that this could be used to
     * initially allocate the source aswell */
    /* TODO: Choose reasonable max size for source. Like maybe a few GiB? That feels like way to much but IDK. */
    assert(source);
    assert(*source);
    assert(source_curr);
    assert(*source_curr);
    assert(*source <= *source_curr);
    assert(source_cap);
    assert(*source_cap);

    const uint64_t source_len = *source_curr - *source;
    if(*source_cap - source_len < padding) {
        *source_cap *= 2;
        *source = reallocarray(*source, *source_cap, sizeof(char));
        assert(*source);
        *source_curr = *source + source_len;
    }
}

static void source_index_scheme(char **source, char **source_curr, uint64_t *source_cap) {
    assert(source);
    assert(*source);
    assert(source_curr);
    assert(*source_curr);
    assert(*source <= *source_curr);
    assert(source_cap);
    assert(*source_cap);
}

void compile_op_group(kernel_t *kernel, const op_group_t *group, const uint64_t optimization) {
    assert(kernel);
    assert(group);
    if(optimization != optimization_none) {
        TODO();
    }
}
