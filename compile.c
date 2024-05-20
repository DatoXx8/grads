#include <CL/cl.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compile.h"
#include "tensor.h"

#define INDEX(buffer, a, z, y, x)                                                                                      \
    ((buffer).str_a_sim * (a) + (buffer).str_z_sim * (z) + (buffer).str_y_sim * (y) + (buffer).str_x_sim * (x))
#define INDEX_(buffer, a, z, y, x)                                                                                     \
    ((buffer)->str_a_sim * (a) + (buffer)->str_z_sim * (z) + (buffer)->str_y_sim * (y) + (buffer)->str_x_sim * (x))
static void simple_loop_free(simple_loop_t *simple) {
    assert(simple);
    assert(simple->op);
    assert(simple->dim_info);
    free(simple->op);
    for(int64_t op_idx = 0; op_idx < simple->loop_len; op_idx++) {
        free(simple->dim_info[op_idx].off_out);
        free(simple->dim_info[op_idx].off_in);
    }
    free(simple->dim_info);
}
/* Has to have the same input and output tensors, with the same shape and be the same op type. Offsets however should be
 * irrelevant */
static int64_t op_equal(op_t *starting, op_t *compared) {
    assert(starting);
    assert(compared);
    if(starting->type != compared->type) { return 0; }
    if(starting->type_unary != compared->type_unary) { return 0; }
    if(starting->type_binary != compared->type_binary) { return 0; }
    if(starting->type_reduce != compared->type_reduce) { return 0; }

    if(strncmp(starting->buffer_out.name, compared->buffer_out.name, BUFFER_NAME_SIZE)) { return 0; }
    if(starting->buffer_out.sze_a_sim != compared->buffer_out.sze_a_sim) { return 0; }
    if(starting->buffer_out.sze_z_sim != compared->buffer_out.sze_z_sim) { return 0; }
    if(starting->buffer_out.sze_y_sim != compared->buffer_out.sze_y_sim) { return 0; }
    if(starting->buffer_out.sze_x_sim != compared->buffer_out.sze_x_sim) { return 0; }
    if(starting->type != op_unary) {
        if(strncmp(starting->buffer_in.name, compared->buffer_in.name, BUFFER_NAME_SIZE)) { return 0; }
        if(starting->buffer_in.sze_a_sim != compared->buffer_in.sze_a_sim) { return 0; }
        if(starting->buffer_in.sze_z_sim != compared->buffer_in.sze_z_sim) { return 0; }
        if(starting->buffer_in.sze_y_sim != compared->buffer_in.sze_y_sim) { return 0; }
        if(starting->buffer_in.sze_x_sim != compared->buffer_in.sze_x_sim) { return 0; }
    }
    return 1;
}
static void simple_loop_configure(simple_loop_t *loop, op_t **op, int64_t loop_len, int64_t loop_num) {
    assert(loop);
    assert(op);
    assert(loop_len > 0);
    assert(loop_num > 0);
    for(int64_t i = 0; i < loop_num; i++) { assert(op[i]); }

    if(loop->op) { simple_loop_free(loop); }
    loop->loop_num = loop_num;
    loop->loop_len = loop_len;
    loop->op = calloc(loop_len, sizeof(op_t));
    assert(loop->op);
    loop->dim_info = calloc(loop_len, sizeof(dim_info_t));
    assert(loop->dim_info);
    for(int64_t i = 0; i < loop_len; i++) {
        loop->op[i] = op[0][i];
        loop->dim_info[i].off_out = calloc(loop_num, sizeof(int64_t));
        loop->dim_info[i].off_in = calloc(loop_num, sizeof(int64_t));
    }
    for(int64_t i = 0; i < loop_num; i++) {
        for(int64_t j = 0; j < loop_len; j++) {
            loop->dim_info[j].off_out[i] = op[i][j].buffer_out.off_sim;
            if(op[i][j].type != op_unary) { loop->dim_info[j].off_in[i] = op[i][j].buffer_in.off_sim; }
        }
    }
}
/* Returns the amount of ops in all the iterations of the loop combined, which makes it possible to use like `snprintf`
 * for format-string appending */
static int64_t simple_loop_from_linearized_index(simple_loop_t *simple, linearized_t *linearized, int64_t start_idx) {
    assert(simple);
    assert(linearized);
    assert(start_idx >= 0 && start_idx < linearized->op_len);
    int64_t loop_length = 0;
    int64_t loop_number = 0;
    int64_t diff;
    op_t starting_op = linearized->op[start_idx];
    for(int64_t i = start_idx + 1; i < linearized->op_len; i++) {
        if(op_equal(&starting_op, &linearized->op[i])) {
            /* TODO: This could probably just be done in the `for` statement */
            if(2 * i - start_idx < linearized->op_len) {
                diff = 0;
                for(int64_t j = 0; j < i - start_idx; j++) {
                    if(!op_equal(&linearized->op[start_idx + j], &linearized->op[i + j])) {
                        diff = 1;
                        break;
                    }
                }
                if(!diff) {
                    loop_length = i - start_idx;
                    break;
                }
            } else {
                break;
            }
        }
    }
    if(!loop_length) { /* Could not find loop */
        op_t **loop_instances = calloc(1, sizeof(op_t *));
        assert(loop_instances);
        loop_instances[0] = calloc(1, sizeof(op_t));
        assert(loop_instances[0]);
        loop_instances[0][0] = linearized->op[start_idx];
        simple_loop_configure(simple, loop_instances, 1, 1);
        free(loop_instances[0]);
        free(loop_instances);
        return 1;
    }
    for(int64_t i = start_idx; i < linearized->op_len; i += loop_length) {
        if(op_equal(&starting_op, &linearized->op[i])) {
            loop_number++;
        } else {
            break;
        }
    }
    assert(loop_number > 0);

    op_t **loop_instances = calloc(loop_number, sizeof(op_t *));
    assert(loop_instances);
    for(int64_t i = 0; i < loop_number; i++) {
        loop_instances[i] = calloc(loop_length, sizeof(op_t));
        assert(loop_instances[i]);
    }

    for(int64_t i = 0; i < loop_number; i++) {
        for(int64_t j = 0; j < loop_length; j++) {
            loop_instances[i][j] = linearized->op[start_idx + (loop_length * i) + j];
        }
    }
    simple_loop_configure(simple, loop_instances, loop_length, loop_number);

    for(int64_t i = 0; i < loop_number; i++) { free(loop_instances[i]); }
    free(loop_instances);

    return loop_length * loop_number;
}
static const int64_t INITIAL_CAP = 4;
#define OVERRIDES_OUTPUT(op)                                                                                           \
    ((op.type == op_unary && (op.type_unary == unary_set)) ||                                                          \
     (op.type == op_binary && (op.type_binary == binary_copy || op.type_binary == binary_copy_like)) ||                \
     (op.type == op_reduce))
static void compile_loop_optimize(compile_loop_t *compile, uint64_t optim) {
    assert(compile);
    assert(optim <= OPTIMIZE_ALL);
    if(optim & OPTIMIZE_INLINE) {
        int64_t inline_cap = INITIAL_CAP;
        int64_t inline_num = 0;
        op_t *inlined = calloc(INITIAL_CAP, sizeof(op_t));
        dim_info_t *inlined_dim_info = calloc(INITIAL_CAP, sizeof(dim_info_t));
        assert(inlined);
        assert(inlined_dim_info);

        for(int64_t i = 0; i < compile->loop_len; i++) {
            if(compile->op[i][0].type == op_binary && compile->op[i][0].type_binary == binary_copy) {
                inline_num = 1;
                inlined[0] = compile->op[i][0];
                inlined_dim_info[0] = compile->dim_info[i][0];
                for(int64_t j = 1; j < compile->loop_len - i; j++) {
                    assert(compile->op_num[i + j] == 1);
                    if(!strncmp(compile->op[i][0].buffer_out.name, compile->op[i + j][0].buffer_out.name,
                                BUFFER_NAME_SIZE)) {
                        if(OVERRIDES_OUTPUT(compile->op[i + j][0])) {
                            break;
                        } else {
                            compile->op_num[i + j] = compile->op_cap[i + j];
                            inline_num++;
                            if(inline_num == inline_cap) {
                                inline_cap *= 2;
                                inlined = reallocarray(inlined, inline_cap, sizeof(op_t));
                                assert(inlined);
                                inlined_dim_info = reallocarray(inlined_dim_info, inline_cap, sizeof(dim_info_t));
                                assert(inlined_dim_info);
                            }
                            inlined[inline_num - 1] = compile->op[i + j][0];
                            inlined_dim_info[inline_num - 1] = compile->dim_info[i + j][0];
                        }
                    } else if(!strncmp(compile->op[i][0].buffer_out.name, compile->op[i + j][0].buffer_in.name,
                                       BUFFER_NAME_SIZE)) {
                        compile->op_num[i] = compile->op_cap[i];
                        compile->op_num[i + j] += inline_num;
                        if(compile->op_num[i + j] >= compile->op_cap[i + j]) {
                            compile->op_cap[i + j] *= 2;
                            compile->op[i + j] = reallocarray(compile->op[i + j], compile->op_cap[i + j], sizeof(op_t));
                            assert(compile->op[i + j]);
                            compile->dim_info[i + j] =
                                reallocarray(compile->dim_info[i + j], compile->op_cap[i + j], sizeof(dim_info_t));
                            assert(compile->dim_info[i + j]);
                        }
                        for(int64_t k = 0; k < inline_num; k++) {
                            compile->op[i + j][k + 1] = inlined[k];
                            compile->dim_info[i + j][k + 1] = inlined_dim_info[k];
                        }
                    }
                }
            }
        }
        free(inlined);
        free(inlined_dim_info);
        int64_t count = 0;
        int64_t new_len = compile->loop_len;
        for(int64_t i = 0; i < compile->loop_len; i++) {
            if(compile->op_num[i] == compile->op_cap[i]) {
                free(compile->op[i]);
                free(compile->dim_info[i]);
                new_len--;
            } else {
                compile->op_cap[count] = compile->op_cap[i];
                compile->op_num[count] = compile->op_num[i];
                compile->op[count] = compile->op[i];
                compile->dim_info[count] = compile->dim_info[i];
                count++;
            }
        }
        compile->loop_len = new_len;
    }
    if(optim & OPTIMIZE_FUSE) { printf("Optimizing: Fuse\n"); }
}
static void compile_loop_free(compile_loop_t *compile) {
    assert(compile);
    assert(compile->op);
    assert(compile->dim_info);
    assert(compile->op_num);
    assert(compile->op_cap);
    for(int64_t i = 0; i < compile->loop_len; i++) {
        assert(compile->op[i]);
        assert(compile->dim_info[i]);
        for(int64_t j = 0; j < compile->op_num[i]; j++) {
            if(compile->dim_info[i][j].off_out) { free(compile->dim_info[i][j].off_out); }
            if(compile->dim_info[i][j].off_in) { free(compile->dim_info[i][j].off_in); }
        }
        free(compile->op[i]);
        free(compile->dim_info[i]);
    }
    free(compile->op);
    free(compile->op_num);
    free(compile->op_cap);
    free(compile->dim_info);
}
static compile_loop_t compile_loop_alloc(simple_loop_t *simple, uint64_t optim) {
    assert(simple);
    assert(simple->loop_len > 0);
    assert(simple->loop_num > 0);
    assert(optim <= OPTIMIZE_ALL);
    compile_loop_t compile = {
        .loop_len = simple->loop_len,
        .loop_num = simple->loop_num,
        .optim = optim,
        .op = NULL,
        .op_num = NULL,
    };
    compile.op_num = calloc(compile.loop_len, sizeof(int64_t));
    assert(compile.op_num);
    compile.op_cap = calloc(compile.loop_len, sizeof(int64_t));
    assert(compile.op_cap);
    compile.op = calloc(compile.loop_len, sizeof(op_t *));
    assert(compile.op);
    compile.dim_info = calloc(compile.loop_len, sizeof(dim_info_t *));
    assert(compile.dim_info);
    for(int64_t i = 0; i < compile.loop_len; i++) { /* This loops can be merged */
        compile.dim_info[i] = calloc(INITIAL_CAP, sizeof(dim_info_t));
        assert(compile.dim_info[i]);
    }
    for(int64_t i = 0; i < compile.loop_len; i++) {
        compile.op_num[i] = 1;
        compile.op_cap[i] = INITIAL_CAP;
        compile.op[i] = calloc(INITIAL_CAP, sizeof(op_t));
        assert(compile.op[i]);
        compile.op[i][0] = simple->op[i];
        compile.dim_info[i][0].off_out = calloc(compile.loop_num, sizeof(int64_t));
        assert(compile.dim_info[i][0].off_out);
        compile.dim_info[i][0].off_in = calloc(compile.loop_num, sizeof(int64_t));
        assert(compile.dim_info[i][0].off_in);
        for(int64_t j = 0; j < compile.loop_num; j++) {
            compile.dim_info[i][0].off_out[j] = simple->dim_info[i].off_out[j];
            compile.dim_info[i][0].off_in[j] = simple->dim_info[i].off_in[j];
        }
    }
    compile_loop_optimize(&compile, optim);
    return compile;
}
const int64_t INITIAL_SOURCE_SIZE = 1024;
const int64_t MAX_ARG_SIZE = 24;
const int64_t MAX_INDEX_DIGITS = 9;
/* Biggest I found was 131 for `max` or `min` binary ops */
const int64_t MAX_OP_SIZE = 512;
#define EXPAND_SOURCE_IF_NEEDED(curr, source, source_size, max_op_size)                                                \
    if(source_size - (curr - source) <= max_op_size) {                                                                 \
        source_size *= 2;                                                                                              \
        offset = curr - source;                                                                                        \
        source = reallocarray(source, source_size, sizeof(char));                                                      \
        assert(source);                                                                                                \
        curr = source + offset;                                                                                        \
    }
#define IS_PREFIX(op)                                                                                                  \
    ((op)->type == op_unary &&                                                                                         \
     ((op)->type_unary == unary_exp || (op)->type_unary == unary_log || (op)->type_unary == unary_sqrt ||              \
      (op)->type_unary == unary_reciprocal || (op)->type_unary == unary_tanh || (op)->type_unary == unary_absolute ||  \
      (op)->type_unary == unary_max || (op)->type_unary == unary_min)) ||                                              \
        ((op)->type == op_binary) && ((op)->type_binary == binary_min || (op)->type_binary == binary_max)

/* Pointers for the last 3 cuz they need to be modified, which is kinda horrible but you can't have multiple return
 * types in C */
static void compile_single_op_to_cl(op_t *op, dim_info_t *dim_info, int64_t op_num, int64_t compile_idx,
                                    int64_t loop_idx, int64_t op_idx, char **source, char **curr, int64_t *source_cap) {
    assert(op);
    assert(dim_info);
    assert(op_num > 0);
    assert(compile_idx >= 0);
    assert(loop_idx >= 0);
    assert(op_idx >= 0);
    assert(source);
    assert(source[0]);
    assert(curr);
    assert(curr[0]);
    assert(source_cap);
    int64_t offset;
    int64_t temp_cap = INITIAL_SOURCE_SIZE;
    char *temp = calloc(INITIAL_SOURCE_SIZE, sizeof(char));
    char *temp_c = temp;
    int64_t max_a = op[0].type == op_reduce ? op[0].buffer_in.sze_a_sim : op[0].buffer_out.sze_a_sim;
    int64_t max_z = op[0].type == op_reduce ? op[0].buffer_in.sze_z_sim : op[0].buffer_out.sze_z_sim;
    int64_t max_y = op[0].type == op_reduce ? op[0].buffer_in.sze_y_sim : op[0].buffer_out.sze_y_sim;
    int64_t max_x = op[0].type == op_reduce ? op[0].buffer_in.sze_x_sim : op[0].buffer_out.sze_x_sim;
    /* TODO: This needs a really big refactor */
    /* This is very, very sus. A lot of things could go wrong just from thinking about it. I haven't found a case where
     * it breaks, but be cautious! */
    if(op[0].type == op_reduce) {
        switch(op[0].type_reduce) {
            case reduce_sum: {
                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu]=0;\n", op[0].buffer_out.name,
                                   op[0].buffer_out.name, compile_idx, op_idx, loop_idx);
                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                break;
            }
            case reduce_avg: {
                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu]=0;\n", op[0].buffer_out.name,
                                   op[0].buffer_out.name, compile_idx, op_idx, loop_idx);
                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                break;
            }
            case reduce_max: {
                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu]=-INFINITY;\n", op[0].buffer_out.name,
                                   op[0].buffer_out.name, compile_idx, op_idx, loop_idx);
                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                break;
            }
            case reduce_min: {
                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu]=INFINITY;\n", op[0].buffer_out.name,
                                   op[0].buffer_out.name, compile_idx, op_idx, loop_idx);
                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                break;
            }
        }
    }
    while(*source_cap - (*curr - *source) - (temp_c - temp) <= MAX_OP_SIZE) {
        *source_cap *= 2;
        offset = *curr - *source;
        *source = reallocarray(*source, *source_cap, sizeof(char));
        assert(*source);
        *curr = *source + offset;
    }
    *curr += snprintf(*curr, temp_cap, "%s", temp);
    EXPAND_SOURCE_IF_NEEDED(*curr, *source, *source_cap, MAX_OP_SIZE);
    memset(temp, 0, temp_cap);
    temp_c = temp;
    int64_t op_offset = 0;
    char *super_temp;
    for(int64_t a = 0; a < max_a; a++) {
        for(int64_t z = 0; z < max_z; z++) {
            for(int64_t y = 0; y < max_y; y++) {
                for(int64_t x = 0; x < max_x; x++) {
                    switch(op[0].type) {
                        case op_unary: {
                            switch(op[0].type_unary) {
                                case unary_add: {
                                    super_temp = temp_c;
                                    temp_c += snprintf(temp_c, MAX_OP_SIZE,
                                                       "%s[%s%lu_%luoff%lu+%lu]+=", op[0].buffer_out.name,
                                                       op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                       INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_subtract: {
                                    super_temp = temp_c;
                                    temp_c += snprintf(temp_c, MAX_OP_SIZE,
                                                       "%s[%s%lu_%luoff%lu+%lu]-=", op[0].buffer_out.name,
                                                       op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                       INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_multiply: {
                                    super_temp = temp_c;
                                    temp_c += snprintf(temp_c, MAX_OP_SIZE,
                                                       "%s[%s%lu_%luoff%lu+%lu]*=", op[0].buffer_out.name,
                                                       op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                       INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_divide: {
                                    super_temp = temp_c;
                                    temp_c += snprintf(temp_c, MAX_OP_SIZE,
                                                       "%s[%s%lu_%luoff%lu+%lu]/=", op[0].buffer_out.name,
                                                       op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                       INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_exp: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                 INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_log: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                 INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_square: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                 INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_sqrt: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                 INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_reciprocal: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                 INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_max: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                 INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_min: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                 INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_set: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                 INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_random: {
                                    ERROR("RNG not supported for OpenCL");
                                    break;
                                }
                                case unary_tanh: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                 INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_absolute: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                 INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_sign: {
                                    TODO();
                                    break;
                                }
                            }
                            break;
                        }
                        case op_binary: {
                            switch(op[0].type_binary) {
                                case binary_add: {
                                    super_temp = temp_c;
                                    temp_c += snprintf(temp_c, MAX_OP_SIZE,
                                                       "%s[%s%lu_%luoff%lu+%lu]+=", op[0].buffer_out.name,
                                                       op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                       INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_subtract: {
                                    super_temp = temp_c;
                                    temp_c += snprintf(temp_c, MAX_OP_SIZE,
                                                       "%s[%s%lu_%luoff%lu+%lu]-=", op[0].buffer_out.name,
                                                       op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                       INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_multiply: {
                                    super_temp = temp_c;
                                    temp_c += snprintf(temp_c, MAX_OP_SIZE,
                                                       "%s[%s%lu_%luoff%lu+%lu]*=", op[0].buffer_out.name,
                                                       op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                       INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_divide: {
                                    super_temp = temp_c;
                                    temp_c += snprintf(temp_c, MAX_OP_SIZE,
                                                       "%s[%s%lu_%luoff%lu+%lu]/=", op[0].buffer_out.name,
                                                       op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                       INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_max: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                 INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_min: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                 INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_copy: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                 INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_add_like: {
                                    super_temp = temp_c;
                                    temp_c += snprintf(temp_c, MAX_OP_SIZE,
                                                       "%s[%s%lu_%luoff%lu+%lu]+=", op[0].buffer_out.name,
                                                       op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                       INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_subtract_like: {
                                    super_temp = temp_c;
                                    temp_c += snprintf(temp_c, MAX_OP_SIZE,
                                                       "%s[%s%lu_%luoff%lu+%lu]-=", op[0].buffer_out.name,
                                                       op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                       INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_multiply_like: {
                                    super_temp = temp_c;
                                    temp_c += snprintf(temp_c, MAX_OP_SIZE,
                                                       "%s[%s%lu_%luoff%lu+%lu]*=", op[0].buffer_out.name,
                                                       op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                       INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_divide_like: {
                                    super_temp = temp_c;
                                    temp_c += snprintf(temp_c, MAX_OP_SIZE,
                                                       "%s[%s%lu_%luoff%lu+%lu]/=", op[0].buffer_out.name,
                                                       op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                       INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_max_like: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                 INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_min_like: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                 INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_copy_like: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                                 INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                            }
                            break;
                        }
                        case op_reduce: {
                            switch(op[0].type_reduce) {
                                case reduce_sum: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu]+=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx);
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case reduce_avg: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu]+=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx);
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case reduce_max: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx);
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case reduce_min: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, compile_idx, op_idx, loop_idx);
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                            }
                            break;
                        }
                        case op_move: {
                            ERROR("Tried to compile move operation to OpenCL at index %lu\n", op_idx);
                        }
                    }
                    if(op_num == 1) {
                        switch(op[0].type) {
                            case op_unary: {
                                switch(op[0].type_unary) {
                                    case unary_add: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%lf", op[0].var_unary);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_subtract: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%lf", op[0].var_unary);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_multiply: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%lf", op[0].var_unary);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_divide: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%lf", op[0].var_unary);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_exp: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "exp(%s[%s%lu_%luoff%lu+%lu])",
                                                           op[0].buffer_out.name, op[0].buffer_out.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_log: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "log(%s[%s%lu_%luoff%lu+%lu])",
                                                           op[0].buffer_out.name, op[0].buffer_out.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_square: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE,
                                                           "%s[%s%lu_%luoff%lu+%lu]*%s[%s%lu_%luoff%lu+%lu]",
                                                           op[0].buffer_out.name, op[0].buffer_out.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_out, a, z, y, x),
                                                           op[0].buffer_out.name, op[0].buffer_out.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_sqrt: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "sqrt(%s[%s%lu_%luoff%lu+%lu])",
                                                           op[0].buffer_out.name, op[0].buffer_out.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_reciprocal: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "1/(%s[%s%lu_%luoff%lu+%lu])",
                                                           op[0].buffer_out.name, op[0].buffer_out.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_max: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "fmax(%s[%s%lu_%luoff%lu+%lu],%lf)",
                                                     op[0].buffer_out.name, op[0].buffer_out.name, compile_idx, op_idx,
                                                     loop_idx, INDEX(op[0].buffer_out, a, z, y, x), op[0].var_unary);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_min: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "fmin(%s[%s%lu_%luoff%lu+%lu],%lf)",
                                                     op[0].buffer_out.name, op[0].buffer_out.name, compile_idx, op_idx,
                                                     loop_idx, INDEX(op[0].buffer_out, a, z, y, x), op[0].var_unary);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_set: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%lf", op[0].var_unary);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_random: {
                                        ERROR("RNG not supported for OpenCL");
                                        break;
                                    }
                                    case unary_tanh: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "tanh(%s[%s%lu_%luoff%lu+%lu])",
                                                           op[0].buffer_out.name, op[0].buffer_out.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_absolute: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "fabs(%s[%s%lu_%luoff%lu+%lu])",
                                                           op[0].buffer_out.name, op[0].buffer_out.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_sign: {
                                        TODO();
                                        break;
                                    }
                                }
                                break;
                            }
                            case op_binary: {
                                switch(op[0].type_binary) {
                                    case binary_add: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]",
                                                           op[0].buffer_in.name, op[0].buffer_in.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_in, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_subtract: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]",
                                                           op[0].buffer_in.name, op[0].buffer_in.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_in, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_multiply: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]",
                                                           op[0].buffer_in.name, op[0].buffer_in.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_in, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_divide: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]",
                                                           op[0].buffer_in.name, op[0].buffer_in.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_in, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_max: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE,
                                                           "fmax(%s[%s%lu_%luoff%lu+%lu],%s[%s%lu_%luoff%lu+%lu])",
                                                           op[0].buffer_out.name, op[0].buffer_out.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_out, a, z, y, x),
                                                           op[0].buffer_in.name, op[0].buffer_in.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_in, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_min: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE,
                                                           "fmin(%s[%s%lu_%luoff%lu+%lu],%s[%s%lu_%luoff%lu+%lu])",
                                                           op[0].buffer_out.name, op[0].buffer_out.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_out, a, z, y, x),
                                                           op[0].buffer_in.name, op[0].buffer_in.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_in, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_copy: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]",
                                                           op[0].buffer_in.name, op[0].buffer_in.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_in, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_add_like: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu]", op[0].buffer_in.name,
                                                     op[0].buffer_in.name, compile_idx, op_idx, loop_idx);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_subtract_like: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu]", op[0].buffer_in.name,
                                                     op[0].buffer_in.name, compile_idx, op_idx, loop_idx);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_multiply_like: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu]", op[0].buffer_in.name,
                                                     op[0].buffer_in.name, compile_idx, op_idx, loop_idx);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_divide_like: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu]", op[0].buffer_in.name,
                                                     op[0].buffer_in.name, compile_idx, op_idx, loop_idx);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_max_like: {
                                        temp_c += snprintf(
                                            temp_c, MAX_OP_SIZE, "fmax(%s[%s%lu_%luoff%lu],%s[%s%lu_%luoff%lu+%lu])",
                                            op[0].buffer_in.name, op[0].buffer_in.name, compile_idx, op_idx, loop_idx,
                                            op[0].buffer_out.name, op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                            INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_min_like: {
                                        temp_c += snprintf(
                                            temp_c, MAX_OP_SIZE, "fmin(%s[%s%lu_%luoff%lu],%s[%s%lu_%luoff%lu+%lu])",
                                            op[0].buffer_in.name, op[0].buffer_in.name, compile_idx, op_idx, loop_idx,
                                            op[0].buffer_out.name, op[0].buffer_out.name, compile_idx, op_idx, loop_idx,
                                            INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_copy_like: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu]", op[0].buffer_in.name,
                                                     op[0].buffer_in.name, compile_idx, op_idx, loop_idx);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                }
                                break;
                            }
                            case op_reduce: {
                                switch(op[0].type_reduce) {
                                    case reduce_sum: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]",
                                                           op[0].buffer_in.name, op[0].buffer_in.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_in, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case reduce_avg: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu]",
                                                           op[0].buffer_in.name, op[0].buffer_in.name, compile_idx,
                                                           op_idx, loop_idx, INDEX(op[0].buffer_in, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case reduce_max: {
                                        temp_c += snprintf(
                                            temp_c, MAX_OP_SIZE, "fmax(%s[%s%lu_%luoff%lu+%lu],%s[%s%lu_%luoff%lu])",
                                            op[0].buffer_in.name, op[0].buffer_in.name, compile_idx, op_idx, loop_idx,
                                            INDEX(op[0].buffer_in, a, z, y, x), op[0].buffer_out.name,
                                            op[0].buffer_out.name, compile_idx, op_idx, loop_idx);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case reduce_min: {
                                        temp_c += snprintf(
                                            temp_c, MAX_OP_SIZE, "fmin(%s[%s%lu_%luoff%lu+%lu],%s[%s%lu_%luoff%lu])",
                                            op[0].buffer_in.name, op[0].buffer_in.name, compile_idx, op_idx, loop_idx,
                                            INDEX(op[0].buffer_in, a, z, y, x), op[0].buffer_out.name,
                                            op[0].buffer_out.name, compile_idx, op_idx, loop_idx);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                }
                                break;
                            }
                            case op_move: {
                                ERROR("Tried to compile move operation to OpenCL\n");
                            }
                        }
                    } else {
                        for(int64_t i = 1; i < op_num; i++) {
                            if(IS_PREFIX(op + i)) {
                                if(op[i].type == op_unary) {
                                    switch(op[i].type_unary) {
                                        case unary_exp: {
                                            temp_c += snprintf(temp_c, MAX_OP_SIZE, "exp(");
                                            EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                            break;
                                        }
                                        case unary_log: {
                                            temp_c += snprintf(temp_c, MAX_OP_SIZE, "log(");
                                            EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                            break;
                                        }
                                        case unary_sqrt: {
                                            temp_c += snprintf(temp_c, MAX_OP_SIZE, "sqrt(");
                                            EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                            break;
                                        }
                                        case unary_reciprocal: {
                                            temp_c += snprintf(temp_c, MAX_OP_SIZE, "1/(");
                                            EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                            break;
                                        }
                                        case unary_max: {
                                            temp_c += snprintf(temp_c, MAX_OP_SIZE, "fmax(");
                                            EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                            break;
                                        }
                                        case unary_min: {
                                            temp_c += snprintf(temp_c, MAX_OP_SIZE, "fmin(");
                                            EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                            break;
                                        }
                                        case unary_tanh: {
                                            temp_c += snprintf(temp_c, MAX_OP_SIZE, "tanh(");
                                            EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                            break;
                                        }
                                        case unary_absolute: {
                                            temp_c += snprintf(temp_c, MAX_OP_SIZE, "fabs(");
                                            EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                            break;
                                        }
                                        default: {
                                            ERROR("Tried to compile non-prefix operation as prefix\n");
                                        }
                                    }
                                } else if(op[i].type == op_binary) {
                                    switch(op[i].type_binary) {
                                        case binary_max: {
                                            temp_c += snprintf(temp_c, MAX_OP_SIZE, "fmax(");
                                            EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                            break;
                                        }
                                        case binary_min: {
                                            temp_c += snprintf(temp_c, MAX_OP_SIZE, "fmin(");
                                            EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                            break;
                                        }
                                        default: {
                                            ERROR("Tried to compile non-prefix operation as prefix\n");
                                        }
                                    }
                                }
                            } else {
                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "(");
                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                            }
                        }
                        for(int64_t i = 1; i < op_num; i++) {
                            if(IS_PREFIX(op + i)) {
                                assert(op[i].type == op_unary);
                                temp_c += snprintf(temp_c, MAX_OP_SIZE, ")");
                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                            } else {
                                switch(op[i].type) {
                                    case op_unary: {
                                        switch(op[i].type_unary) {
                                            case unary_add: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "+%.16lf)", op[i].var_unary);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case unary_subtract: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "-%.16lf)", op[i].var_unary);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case unary_multiply: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "*%.16lf)", op[i].var_unary);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case unary_divide: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "/%.16lf)", op[i].var_unary);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case unary_square: {
                                                /* TODO: REEEEFACTOR THIS SUCKER!!! */
                                                super_temp = temp + op_offset + op_num - i;
                                                char *dup = calloc(temp_c - super_temp + 1, sizeof(char));
                                                assert(dup);
                                                for(int64_t dup_idx = 0; dup_idx < temp_c - super_temp; dup_idx++) {
                                                    dup[dup_idx] = super_temp[dup_idx];
                                                }
                                                temp_c += snprintf(temp_c, 3 + (temp_c - super_temp), "*%s)", dup);
                                                free(dup);
                                                break;
                                            }
                                            case unary_max: {
                                                super_temp = temp + op_offset + op_num - i;
                                                char *dup = calloc(temp_c - super_temp + 1, sizeof(char));
                                                assert(dup);
                                                for(int64_t dup_idx = 0; dup_idx < temp_c - super_temp; dup_idx++) {
                                                    dup[dup_idx] = super_temp[dup_idx];
                                                }
                                                temp_c += snprintf(temp_c, 3 + (temp_c - super_temp), "%s, %lf)", dup,
                                                                   op[i].var_unary);
                                                free(dup);
                                                break;
                                            }
                                            case unary_min: {
                                                super_temp = temp + op_offset + op_num - i;
                                                char *dup = calloc(temp_c - super_temp + 1, sizeof(char));
                                                assert(dup);
                                                for(int64_t dup_idx = 0; dup_idx < temp_c - super_temp; dup_idx++) {
                                                    dup[dup_idx] = super_temp[dup_idx];
                                                }
                                                temp_c += snprintf(temp_c, 3 + (temp_c - super_temp), "%s, %lf)", dup,
                                                                   op[i].var_unary);
                                                free(dup);
                                                break;
                                            }
                                            case unary_set: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%.16lf)", op[i].var_unary);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case unary_sign: {
                                                TODO();
                                                break;
                                            }
                                            default: {
                                                ERROR("Tried to compile prefix op as non-prefix\n");
                                            }
                                        }
                                        break;
                                    }
                                    case op_binary: {
                                        switch(op[i].type_binary) {
                                            case binary_add: {
                                                temp_c +=
                                                    snprintf(temp_c, MAX_OP_SIZE, "+%s[%s%lu_%luoff%lu+%lu])",
                                                             op[i].buffer_in.name, op[i].buffer_in.name, compile_idx,
                                                             op_idx, loop_idx, INDEX(op[i].buffer_in, a, z, y, x));
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case binary_subtract: {
                                                temp_c +=
                                                    snprintf(temp_c, MAX_OP_SIZE, "-%s[%s%lu_%luoff%lu+%lu])",
                                                             op[i].buffer_in.name, op[i].buffer_in.name, compile_idx,
                                                             op_idx, loop_idx, INDEX(op[i].buffer_in, a, z, y, x));
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case binary_multiply: {
                                                temp_c +=
                                                    snprintf(temp_c, MAX_OP_SIZE, "*%s[%s%lu_%luoff%lu+%lu])",
                                                             op[i].buffer_in.name, op[i].buffer_in.name, compile_idx,
                                                             op_idx, loop_idx, INDEX(op[i].buffer_in, a, z, y, x));
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case binary_divide: {
                                                temp_c +=
                                                    snprintf(temp_c, MAX_OP_SIZE, "/%s[%s%lu_%luoff%lu+%lu])",
                                                             op[i].buffer_in.name, op[i].buffer_in.name, compile_idx,
                                                             op_idx, loop_idx, INDEX(op[i].buffer_in, a, z, y, x));
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case binary_max: {
                                                super_temp = temp + op_offset + op_num - i;
                                                char *dup = calloc(temp_c - super_temp + 1, sizeof(char));
                                                assert(dup);
                                                for(int64_t dup_idx = 0; dup_idx < temp_c - super_temp; dup_idx++) {
                                                    dup[dup_idx] = super_temp[dup_idx];
                                                }
                                                temp_c += snprintf(
                                                    temp_c, 3 + (temp_c - super_temp), "%s, %s[%s%lu_%luoff%lu+%lu])",
                                                    dup, op[i].buffer_in.name, op[i].buffer_in.name, compile_idx,
                                                    op_idx, loop_idx, INDEX(op[i].buffer_in, a, z, y, x));
                                                free(dup);
                                                break;
                                            }
                                            case binary_min: {
                                                super_temp = temp + op_offset + op_num - i;
                                                char *dup = calloc(temp_c - super_temp + 1, sizeof(char));
                                                assert(dup);
                                                for(int64_t dup_idx = 0; dup_idx < temp_c - super_temp; dup_idx++) {
                                                    dup[dup_idx] = super_temp[dup_idx];
                                                }
                                                temp_c += snprintf(
                                                    temp_c, 3 + (temp_c - super_temp), "%s, %s[%s%lu_%luoff%lu+%lu])",
                                                    dup, op[i].buffer_in.name, op[i].buffer_in.name, compile_idx,
                                                    op_idx, loop_idx, INDEX(op[i].buffer_in, a, z, y, x));
                                                free(dup);
                                                break;
                                            }
                                            case binary_copy: {
                                                temp_c +=
                                                    snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu+%lu])",
                                                             op[i].buffer_in.name, op[i].buffer_in.name, compile_idx,
                                                             op_idx, loop_idx, INDEX(op[i].buffer_in, a, z, y, x));
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case binary_add_like: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "+%s[%s%lu_%luoff%lu])",
                                                                   op[i].buffer_in.name, op[i].buffer_in.name,
                                                                   compile_idx, op_idx, loop_idx);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case binary_subtract_like: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "-%s[%s%lu_%luoff%lu])",
                                                                   op[i].buffer_in.name, op[i].buffer_in.name,
                                                                   compile_idx, op_idx, loop_idx);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case binary_multiply_like: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "*%s[%s%lu_%luoff%lu])",
                                                                   op[i].buffer_in.name, op[i].buffer_in.name,
                                                                   compile_idx, op_idx, loop_idx);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case binary_divide_like: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "/%s[%s%lu_%luoff%lu])",
                                                                   op[i].buffer_in.name, op[i].buffer_in.name,
                                                                   compile_idx, op_idx, loop_idx);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case binary_max_like: {
                                                super_temp = temp + op_offset + op_num - i;
                                                char *dup = calloc(temp_c - super_temp + 1, sizeof(char));
                                                assert(dup);
                                                for(int64_t dup_idx = 0; dup_idx < temp_c - super_temp; dup_idx++) {
                                                    dup[dup_idx] = super_temp[dup_idx];
                                                }
                                                temp_c +=
                                                    snprintf(temp_c, 3 + (temp_c - super_temp),
                                                             "%s, %s[%s%lu_%luoff%lu])", dup, op[i].buffer_in.name,
                                                             op[i].buffer_in.name, compile_idx, op_idx, loop_idx);
                                                free(dup);
                                                break;
                                            }
                                            case binary_min_like: {
                                                super_temp = temp + op_offset + op_num - i;
                                                char *dup = calloc(temp_c - super_temp + 1, sizeof(char));
                                                assert(dup);
                                                for(int64_t dup_idx = 0; dup_idx < temp_c - super_temp; dup_idx++) {
                                                    dup[dup_idx] = super_temp[dup_idx];
                                                }
                                                temp_c +=
                                                    snprintf(temp_c, 3 + (temp_c - super_temp),
                                                             "%s, %s[%s%lu_%luoff%lu])", dup, op[i].buffer_in.name,
                                                             op[i].buffer_in.name, compile_idx, op_idx, loop_idx);
                                                free(dup);
                                                break;
                                            }
                                            case binary_copy_like: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu])",
                                                                   op[i].buffer_in.name, op[i].buffer_in.name,
                                                                   compile_idx, op_idx, loop_idx);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                        }
                                        break;
                                    }
                                    case op_reduce: {
                                        ERROR("Tried to inline reduce operation!\n");
                                        break;
                                    }
                                    case op_move: {
                                        ERROR("Tried to inline move operation!\n");
                                    }
                                }
                            }
                        }
                    }
                    temp_c += snprintf(temp_c, MAX_OP_SIZE, ";\n");
                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                    while(*source_cap - (*curr - *source) - (temp_c - temp) <= MAX_OP_SIZE) {
                        *source_cap *= 2;
                        offset = *curr - *source;
                        *source = reallocarray(*source, *source_cap, sizeof(char));
                        assert(*source);
                        *curr = *source + offset;
                    }
                    *curr += snprintf(*curr, temp_cap, "%s", temp);
                    EXPAND_SOURCE_IF_NEEDED(*curr, *source, *source_cap, MAX_OP_SIZE);
                    memset(temp, 0, temp_cap);
                    temp_c = temp;
                }
            }
        }
    }
    if(op[0].type == op_reduce && op[0].type_reduce == reduce_avg) {
        temp_c +=
            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%lu_%luoff%lu]/=%lf;\n", op[0].buffer_out.name, op[0].buffer_out.name,
                     compile_idx, op_idx, loop_idx, (double) max_a * max_z * max_y * max_x);
        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
        while(*source_cap - (*curr - *source) - (temp_c - temp) <= MAX_OP_SIZE) {
            *source_cap *= 2;
            offset = *curr - *source;
            *source = reallocarray(*source, *source_cap, sizeof(char));
            assert(*source);
            *curr = *source + offset;
        }
        *curr += snprintf(*curr, temp_cap, "%s", temp);
        EXPAND_SOURCE_IF_NEEDED(*curr, *source, *source_cap, MAX_OP_SIZE);
        memset(temp, 0, temp_cap);
        temp_c = temp;
    }

    free(temp);
}
static void compile_loops_to_cl(program_t *program, compile_loop_t *compile, int64_t global_size, int64_t local_size,
                                int64_t compile_loops) {
    assert(compile);
    assert(global_size);
    /* TODO: Support `local_size > 1`*/
    assert(local_size == 1);

    char *func_name = KERNEL_NAME;
    assert(func_name);

    int64_t source_cap = INITIAL_SOURCE_SIZE;
    char *source = calloc(source_cap, sizeof(char));
    assert(source);
    char *curr = source;
    int64_t offset;

    int64_t arg_num = 0;
    char **arg = NULL;
    cl_mem *arg_cl = NULL;
    int64_t found;
    for(int64_t compile_idx = 0; compile_idx < compile_loops; compile_idx++) {
        for(int64_t loop_op_idx = 0; loop_op_idx < compile[compile_idx].loop_len; loop_op_idx++) {
            if(compile[compile_idx].op_num[loop_op_idx] == 1) {
                found = 0;
                for(int64_t arg_idx = 0; arg_idx < arg_num; arg_idx++) {
                    if(!strncmp(compile[compile_idx].op[loop_op_idx][0].buffer_out.name, arg[arg_idx],
                                BUFFER_NAME_SIZE)) {
                        found = 1;
                        break;
                    }
                }
                if(!found) {
                    arg_num++;
                    arg = reallocarray(arg, arg_num, sizeof(char *));
                    arg_cl = reallocarray(arg_cl, arg_num, sizeof(cl_mem));
                    assert(arg);
                    assert(arg_cl);
                    arg[arg_num - 1] = calloc(BUFFER_NAME_SIZE + 1, sizeof(char));
                    assert(arg[arg_num - 1]);
                    strncpy(arg[arg_num - 1], compile[compile_idx].op[loop_op_idx][0].buffer_out.name,
                            BUFFER_NAME_SIZE);
                    arg_cl[arg_num - 1] = compile[compile_idx].op[loop_op_idx][0].buffer_out.val_cl;
                }
                if(compile[compile_idx].op[loop_op_idx][0].type != op_unary) {
                    found = 0;
                    for(int64_t j = 0; j < arg_num; j++) {
                        if(!strncmp(compile[compile_idx].op[loop_op_idx][0].buffer_in.name, arg[j], BUFFER_NAME_SIZE)) {
                            found = 1;
                            break;
                        }
                    }
                    if(!found) {
                        arg_num++;
                        arg = reallocarray(arg, arg_num, sizeof(char *));
                        arg_cl = reallocarray(arg_cl, arg_num, sizeof(cl_mem));
                        assert(arg);
                        assert(arg_cl);
                        arg[arg_num - 1] = calloc(BUFFER_NAME_SIZE + 1, sizeof(char));
                        assert(arg[arg_num - 1]);
                        strncpy(arg[arg_num - 1], compile[compile_idx].op[loop_op_idx][0].buffer_in.name,
                                BUFFER_NAME_SIZE);
                        arg_cl[arg_num - 1] = compile[compile_idx].op[loop_op_idx][0].buffer_in.val_cl;
                    }
                }
            } else {
                for(int64_t op_idx = 0; op_idx < compile[compile_idx].op_num[loop_op_idx]; op_idx++) {
                    if(op_idx) {
                        if(compile[compile_idx].op[loop_op_idx][op_idx].type != op_unary) {
                            found = 0;
                            for(int64_t k = 0; k < arg_num; k++) {
                                if(!strncmp(compile[compile_idx].op[loop_op_idx][op_idx].buffer_in.name, arg[k],
                                            BUFFER_NAME_SIZE)) {
                                    found = 1;
                                    break;
                                }
                            }
                            if(!found) {
                                arg_num++;
                                arg = reallocarray(arg, arg_num, sizeof(char *));
                                arg_cl = reallocarray(arg_cl, arg_num, sizeof(cl_mem));
                                assert(arg);
                                assert(arg_cl);
                                arg[arg_num - 1] = calloc(BUFFER_NAME_SIZE + 1, sizeof(char));
                                assert(arg[arg_num - 1]);
                                strncpy(arg[arg_num - 1], compile[compile_idx].op[loop_op_idx][op_idx].buffer_in.name,
                                        BUFFER_NAME_SIZE);
                                arg_cl[arg_num - 1] = compile[compile_idx].op[loop_op_idx][op_idx].buffer_in.val_cl;
                            }
                        }
                    } else {
                        found = 0;
                        for(int64_t k = 0; k < arg_num; k++) {
                            if(!strncmp(compile[compile_idx].op[loop_op_idx][op_idx].buffer_out.name, arg[k],
                                        BUFFER_NAME_SIZE)) {
                                found = 1;
                                break;
                            }
                        }
                        if(!found) {
                            arg_num++;
                            arg = reallocarray(arg, arg_num, sizeof(char *));
                            arg_cl = reallocarray(arg_cl, arg_num, sizeof(cl_mem));
                            assert(arg);
                            assert(arg_cl);
                            arg[arg_num - 1] = calloc(BUFFER_NAME_SIZE + 1, sizeof(char));
                            assert(arg[arg_num - 1]);
                            strncpy(arg[arg_num - 1], compile[compile_idx].op[loop_op_idx][op_idx].buffer_out.name,
                                    BUFFER_NAME_SIZE);
                            arg_cl[arg_num - 1] = compile[compile_idx].op[loop_op_idx][op_idx].buffer_out.val_cl;
                        }
                    }
                }
            }
        }
    }
    curr += snprintf(curr, MAX_OP_SIZE, "int gid0 = get_global_id(0);\n");
    EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
    for(int64_t compile_idx = 0; compile_idx < compile_loops; compile_idx++) {
        if(compile_idx) {
            curr += snprintf(curr, MAX_OP_SIZE, "id = gid0;\n");
            EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
        } else {
            curr += snprintf(curr, MAX_OP_SIZE, "int id = gid0;\n");
            EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
        }
        for(int64_t op_idx = 0; op_idx < compile[compile_idx].loop_len; op_idx++) {
            if(compile[compile_idx].op_num[op_idx] == 1) {
                /* TODO: Add static / const for better perf?  */
                curr += snprintf(curr, MAX_OP_SIZE, "int %s%lu_%luoff[]={",
                                 compile[compile_idx].op[op_idx][0].buffer_out.name, compile_idx, op_idx);
                EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                for(int64_t loop_idx = 0; loop_idx < compile[compile_idx].loop_num; loop_idx++) {
                    if(loop_idx) {
                        curr += snprintf(curr, MAX_OP_SIZE, ",%lu",
                                         compile[compile_idx].dim_info[op_idx][0].off_out[loop_idx]);
                        EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                    } else {
                        curr += snprintf(curr, MAX_OP_SIZE, "%lu",
                                         compile[compile_idx].dim_info[op_idx][0].off_out[loop_idx]);
                        EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                    }
                }
                curr += snprintf(curr, MAX_OP_SIZE, "};\n");
                EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                if(compile[compile_idx].op[op_idx][0].type != op_unary) {
                    curr += snprintf(curr, MAX_OP_SIZE, "int %s%lu_%luoff[]={",
                                     compile[compile_idx].op[op_idx][0].buffer_in.name, compile_idx, op_idx);
                    EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                    for(int64_t loop_idx = 0; loop_idx < compile[compile_idx].loop_num; loop_idx++) {
                        if(loop_idx) {
                            curr += snprintf(curr, MAX_OP_SIZE, ",%lu",
                                             compile[compile_idx].dim_info[op_idx][0].off_in[loop_idx]);
                        } else {
                            curr += snprintf(curr, MAX_OP_SIZE, "%lu",
                                             compile[compile_idx].dim_info[op_idx][0].off_in[loop_idx]);
                        }
                    }
                    curr += snprintf(curr, MAX_OP_SIZE, "};\n");
                    EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                }
            } else {
                for(int64_t inline_op_idx = 0; inline_op_idx < compile[compile_idx].op_num[op_idx]; inline_op_idx++) {
                    if(inline_op_idx) {
                        if(compile[compile_idx].op[op_idx][inline_op_idx].type != op_unary) {
                            curr += snprintf(curr, MAX_OP_SIZE, "int %s%lu_%luoff[]={",
                                             compile[compile_idx].op[op_idx][inline_op_idx].buffer_in.name, compile_idx,
                                             op_idx);
                            EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                            for(int64_t loop_idx = 0; loop_idx < compile[compile_idx].loop_num; loop_idx++) {
                                if(loop_idx) {
                                    curr +=
                                        snprintf(curr, MAX_OP_SIZE, ",%lu",
                                                 compile[compile_idx].dim_info[op_idx][inline_op_idx].off_in[loop_idx]);
                                } else {
                                    curr +=
                                        snprintf(curr, MAX_OP_SIZE, "%lu",
                                                 compile[compile_idx].dim_info[op_idx][inline_op_idx].off_in[loop_idx]);
                                }
                            }
                            curr += snprintf(curr, MAX_OP_SIZE, "};\n");
                            EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                        }
                    } else {
                        curr += snprintf(curr, MAX_OP_SIZE, "int %s%lu_%luoff[]={",
                                         compile[compile_idx].op[op_idx][inline_op_idx].buffer_out.name, compile_idx,
                                         op_idx);
                        EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                        for(int64_t loop_idx = 0; loop_idx < compile[compile_idx].loop_num; loop_idx++) {
                            if(loop_idx) {
                                curr +=
                                    snprintf(curr, MAX_OP_SIZE, ",%lu",
                                             compile[compile_idx].dim_info[op_idx][inline_op_idx].off_out[loop_idx]);
                            } else {
                                curr +=
                                    snprintf(curr, MAX_OP_SIZE, "%lu",
                                             compile[compile_idx].dim_info[op_idx][inline_op_idx].off_out[loop_idx]);
                                EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                            }
                        }
                        curr += snprintf(curr, MAX_OP_SIZE, "};\n");
                        EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                    }
                }
            }
        }
        EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
        int64_t loops_leftover = compile[compile_idx].loop_num % global_size;
        int64_t loops_assigned = (compile[compile_idx].loop_num - loops_leftover) / global_size;
        int64_t loops_needed;
        if(loops_leftover) {
            loops_needed = loops_assigned + 1;
        } else {
            loops_needed = loops_assigned;
        }
        for(int64_t loop_idx = 0; loop_idx < loops_needed; loop_idx++) {
            if(loop_idx == loops_assigned) {
                curr += snprintf(curr, MAX_OP_SIZE, "if(gid0 < %lu) {\n", loops_leftover);
                EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
            }
            if(loop_idx) {
                curr += snprintf(curr, MAX_OP_SIZE, "id += %lu;\n", global_size);
                EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
            }
            for(int64_t loop_op_idx = 0; loop_op_idx < compile[compile_idx].loop_len; loop_op_idx++) {
                if(compile[compile_idx].op_num[loop_op_idx] == 1) {
                    curr += snprintf(curr, MAX_OP_SIZE, "int %s%lu_%luoff%lu=%s%lu_%luoff[id];\n",
                                     compile[compile_idx].op[loop_op_idx][0].buffer_out.name, compile_idx, loop_op_idx,
                                     loop_idx, compile[compile_idx].op[loop_op_idx][0].buffer_out.name, compile_idx,
                                     loop_op_idx);
                    EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                    if(compile[compile_idx].op[loop_op_idx][0].type != op_unary) {
                        curr += snprintf(curr, MAX_OP_SIZE, "int %s%lu_%luoff%lu=%s%lu_%luoff[id];\n",
                                         compile[compile_idx].op[loop_op_idx][0].buffer_in.name, compile_idx,
                                         loop_op_idx, loop_idx, compile[compile_idx].op[loop_op_idx][0].buffer_in.name,
                                         compile_idx, loop_op_idx);
                        EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                    }
                } else {
                    for(int64_t inline_op_idx = 0; inline_op_idx < compile[compile_idx].op_num[loop_op_idx];
                        inline_op_idx++) {
                        if(inline_op_idx) {
                            if(compile[compile_idx].op[loop_op_idx][inline_op_idx].type != op_unary) {
                                curr += snprintf(curr, MAX_OP_SIZE, "int %s%lu_%luoff%lu=%s%lu_%luoff[id];\n",
                                                 compile[compile_idx].op[loop_op_idx][inline_op_idx].buffer_in.name,
                                                 compile_idx, loop_op_idx, loop_idx,
                                                 compile[compile_idx].op[loop_op_idx][inline_op_idx].buffer_in.name,
                                                 compile_idx, loop_op_idx);
                                EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                            }
                        } else {
                            curr += snprintf(curr, MAX_OP_SIZE, "int %s%lu_%luoff%lu=%s%lu_%luoff[id];\n",
                                             compile[compile_idx].op[loop_op_idx][inline_op_idx].buffer_out.name,
                                             compile_idx, loop_op_idx, loop_idx,
                                             compile[compile_idx].op[loop_op_idx][inline_op_idx].buffer_out.name,
                                             compile_idx, loop_op_idx);
                            EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                        }
                    }
                }
                compile_single_op_to_cl(compile[compile_idx].op[loop_op_idx],
                                        compile[compile_idx].dim_info[loop_op_idx],
                                        compile[compile_idx].op_num[loop_op_idx], compile_idx, loop_idx, loop_op_idx,
                                        &source, &curr, &source_cap);
            }
            if(loop_idx == loops_assigned) {
                curr += snprintf(curr, MAX_OP_SIZE, "}\n");
                EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
            }
        }
        /* TODO: Figure out a smarter way of doing this because it only needs to happen if the values are dependant on
         * eachother */
        if(compile_idx != compile_loops - 1) {
            curr += snprintf(curr, MAX_OP_SIZE, "barrier(CLK_GLOBAL_MEM_FENCE);\n");
            EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
        }
    }

    assert(arg_num != 0);
    /* That `+ 3` is pure magic. I have no clue where it comes from */
    int64_t kernel_size = strlen("__kernel void ") + strlen(KERNEL_NAME "(") +
                          (strlen("__global double *") + BUFFER_NAME_SIZE) * arg_num + strlen(", ") * (arg_num - 1) +
                          strlen(") {\n") + (curr - source) + strlen("}\n") + 3;
    char *kernel_source = calloc(kernel_size, sizeof(char));
    assert(kernel_source);
    char *kernel_i = kernel_source;
    kernel_i += snprintf(kernel_i, 3 + strlen("__kernel void " KERNEL_NAME "("), "__kernel void " KERNEL_NAME "(");
    for(int64_t arg_idx = 0; arg_idx < arg_num; arg_idx++) {
        if(arg_idx != arg_num - 1) {
            kernel_i += snprintf(kernel_i, 1 + BUFFER_NAME_SIZE + strnlen("__global double *, ", 20),
                                 "__global double *%s, ", arg[arg_idx]);
        } else {
            kernel_i += snprintf(kernel_i, 1 + BUFFER_NAME_SIZE + strnlen("__global double *) {\n", 22),
                                 "__global double *%s) {\n", arg[arg_idx]);
        }
    }
    /* This one is very sus. Extremely sus. Why in the world do I need to do the `+ 1` here? */
    kernel_i += snprintf(kernel_i, curr - source + 1, "%s", source);
    EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
    kernel_i += snprintf(kernel_i, 3, "}\n");
    EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);

    program->arg_num = arg_num;
    program->arg_mem = arg_cl;
    program->arg_name = arg;
    program->local_size = local_size;
    program->global_size = global_size;
    program->source = kernel_source;
    program->source_cap = source_cap;
    program->source_len = kernel_i - kernel_source;

    free(source);
}
void program_compile(program_t *program, linearized_t *linearized, cl_device_id *device_id, cl_context *context,
                     cl_command_queue *command_queue) {
    assert(program);
    assert(linearized);
    assert(device_id);
    assert(context);
    assert(command_queue);
    if(!linearized->op_len) { return; }
    simple_loop_t simple = {0};
    int64_t global_size = 9;
    int64_t local_size = 1;
    compile_loop_t *compile = calloc(INITIAL_CAP, sizeof(compile_loop_t));
    int64_t compile_num = 0;
    int64_t compile_cap = INITIAL_CAP;
    int64_t op_idx = 0;
    while(op_idx < linearized->op_len) {
        op_idx += simple_loop_from_linearized_index(&simple, linearized, op_idx);
        compile[compile_num] = compile_loop_alloc(&simple, OPTIMIZE_INLINE);
        compile_num++;
        if(compile_num == compile_cap) {
            compile_cap *= 2;
            compile = reallocarray(compile, compile_cap, sizeof(compile_loop_t));
            assert(compile);
        }
    }
    compile_loops_to_cl(program, compile, global_size, local_size, compile_num);
    simple_loop_free(&simple);
    for(int64_t i = 0; i < compile_num; i++) { compile_loop_free(&compile[i]); }
    program->cl_device_id = device_id;
    program->cl_context = context;
    program->cl_command_queue = command_queue;
    free(compile);
}
void program_free(program_t *program) {
    for(int64_t arg_idx = 0; arg_idx < program->arg_num; arg_idx++) { free(program->arg_name[arg_idx]); }
    free(program->arg_name);
    program->arg_name = NULL;
    free(program->arg_mem);
    program->arg_mem = NULL;
    free(program->source);
    program->source = NULL;
    if(program->cl_kernel) {
        clReleaseKernel(program->cl_kernel);
        program->cl_kernel = NULL;
    }
    /* This is a very disgusting fix, but I suppose it works for now. TODO: Make this nicer */
    if(program->cl_program) {
        if(*program->cl_program) {
            clReleaseProgram(*program->cl_program);
            *program->cl_program = NULL;
            free(*program->cl_program);
        }
        free(program->cl_program);
        program->cl_program = NULL;
    }
    if(program->cl_device_id) {
        if(*program->cl_device_id) {
            clReleaseDevice(*program->cl_device_id);
            *program->cl_device_id = NULL;
        }
        program->cl_device_id = NULL;
    }
    if(program->cl_context) {
        if(*program->cl_context) {
            clReleaseContext(*program->cl_context);
            *program->cl_context = NULL;
        }
        program->cl_context = NULL;
    }
    if(program->cl_command_queue) {
        if(*program->cl_command_queue) {
            clReleaseCommandQueue(*program->cl_command_queue);
            *program->cl_command_queue = NULL;
        }
        program->cl_command_queue = NULL;
    }
}
