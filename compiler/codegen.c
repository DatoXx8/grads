#include "codegen.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
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
const uint64_t write_len_max = padding;
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

static void source_append_kernel_start(char **source, char **source_curr, uint64_t *source_cap, const char **arg,
                                       const uint64_t arg_num) {
    assert(source);
    assert(*source);
    assert(source_curr);
    assert(*source_curr);
    assert(*source <= *source_curr);
    assert(source_cap);
    assert(*source_cap);
    assert(arg);
    /* MAYBE: Assert all the strings in `arg` */
    assert(arg_num);

    *source_curr += snprintf(*source_curr, write_len_max, "__kernel void " KERNEL_NAME "(");
    source_expand(source, source_curr, source_cap);
    for(uint64_t arg_idx = 0; arg_idx < arg_num; arg_idx++) {
        /* There might be some optimization cases where it's not a `__global` or `double` */
        if(arg_idx) {
            *source_curr += snprintf(*source_curr, write_len_max, ", __global double *%s", arg[arg_idx]);
        } else {
            *source_curr += snprintf(*source_curr, write_len_max, "__global double *%s", arg[arg_idx]);
        }
        source_expand(source, source_curr, source_cap);
    }
    *source_curr += snprintf(*source_curr, write_len_max, ") {\n");
    source_expand(source, source_curr, source_cap);
}

static void source_append_kernel_end(char **source, char **source_curr, uint64_t *source_cap) {
    assert(source);
    assert(*source);
    assert(source_curr);
    assert(*source_curr);
    assert(*source <= *source_curr);
    assert(source_cap);
    assert(*source_cap);

    *source_curr += snprintf(*source_curr, write_len_max, "}\n");
    source_expand(source, source_curr, source_cap);
}

static void source_append_head(char **source, char **source_curr, uint64_t *source_cap) {
    assert(source);
    assert(*source);
    assert(source_curr);
    assert(*source_curr);
    assert(*source <= *source_curr);
    assert(source_cap);
    assert(*source_cap);

    *source_curr += snprintf(*source_curr, write_len_max, "__local const int gid = get_global_id(0);\n");
    source_expand(source, source_curr, source_cap);
    *source_curr += snprintf(*source_curr, write_len_max, "__local int id;\n");
    source_expand(source, source_curr, source_cap);
}

static void source_append_index(char **source, char **source_curr, uint64_t *source_cap, const op_t *op,
                                const uint64_t op_idx, const uint64_t loop_idx) {
    assert(source);
    assert(*source);
    assert(source_curr);
    assert(*source_curr);
    assert(*source <= *source_curr);
    assert(source_cap);
    assert(*source_cap);
    assert(op);

    /* TODO: Choose a more accurate name for what this is */
    const uint64_t x_wait_out = 1;
    const uint64_t y_wait_out = x_wait_out * op->buffer_out.x_sze;
    const uint64_t z_wait_out = y_wait_out * op->buffer_out.y_sze;
    const uint64_t a_wait_out = z_wait_out * op->buffer_out.z_sze;
    *source_curr +=
        snprintf(*source_curr, write_len_max,
                 "__local int %s_%lu_%lu = %lu+id%%%lu/%lu*%lu+id%%%lu/%lu*%lu+id%%%lu/%lu*%lu+id%%%lu/%lu*%lu;\n",
                 op->buffer_out.name, loop_idx, op_idx, op->buffer_out.off, op->buffer_out.a_sze * a_wait_out,
                 a_wait_out, op->buffer_out.a_str, a_wait_out, z_wait_out, op->buffer_out.z_str, z_wait_out, y_wait_out,
                 op->buffer_out.y_str, y_wait_out, x_wait_out, op->buffer_out.x_str);
    source_expand(source, source_curr, source_cap);
    if(op->type_op != op_unary) {
        const uint64_t x_wait_in = 1;
        const uint64_t y_wait_in = x_wait_in * op->buffer_in.x_sze;
        const uint64_t z_wait_in = y_wait_in * op->buffer_in.y_sze;
        const uint64_t a_wait_in = z_wait_in * op->buffer_in.z_sze;
        *source_curr +=
            snprintf(*source_curr, write_len_max,
                     "__local int %s_%lu_%lu = %lu+id%%%lu/%lu*%lu+id%%%lu/%lu*%lu+id%%%lu/%lu*%lu+id%%%lu/%lu*%lu;\n",
                     op->buffer_in.name, loop_idx, op_idx, op->buffer_in.off, op->buffer_in.a_sze * a_wait_in,
                     a_wait_in, op->buffer_in.a_str, a_wait_in, z_wait_in, op->buffer_in.z_str, z_wait_in, y_wait_in,
                     op->buffer_in.y_str, y_wait_in, x_wait_in, op->buffer_in.x_str);
        source_expand(source, source_curr, source_cap);
    }
}
void compile_op_group(kernel_t *kernel, const op_group_t *group, const uint64_t size_global, const uint64_t size_local,
                      const uint64_t optimization) {
    assert(kernel);
    assert(group);
    assert(size_global);
    assert(size_local);
    assert(size_local <= size_global);

    if(optimization != optimization_none) {
        TODO();
    }
    uint64_t source_cap = padding;
    char *source = calloc(source_cap, sizeof(char));
    char *source_curr = source;

    source_append_kernel_start(&source, &source_curr, &source_cap, (const char **) kernel->arg_name, kernel->arg_num);
    source_append_head(&source, &source_curr, &source_cap);

    const uint64_t loop_leftover = group->repeat_num % size_global;
    const uint64_t loop_num = group->repeat_num / size_global + loop_leftover ? 1 : 0;
    for(uint64_t loop_idx = 0; loop_idx < loop_num; loop_idx++) {
        for(uint64_t op_idx = 0; op_idx < group->op_num; op_idx++) {
            source_append_index(&source, &source_curr, &source_cap, &group->op[op_idx], op_idx, loop_idx);
        }
    }

    source_append_kernel_end(&source, &source_curr, &source_cap);

    kernel->source = source;
    kernel->source_cap = source_cap;
}
