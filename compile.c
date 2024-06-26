#include <CL/cl.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compile.h"
#include "tensor.h"
#include "utils.h"

#define INDEX(buffer, a, z, y, x)                                                                                      \
    ((buffer).str_a * (a) + (buffer).str_z * (z) + (buffer).str_y * (y) + (buffer).str_x * (x))
#define INDEX_(buffer, a, z, y, x)                                                                                     \
    ((buffer)->str_a * (a) + (buffer)->str_z * (z) + (buffer)->str_y * (y) + (buffer)->str_x * (x))
static void simple_loop_free(simple_loop_t *simple) {
    assert(simple);
    assert(simple->op);
    assert(simple->dim_info);
    for(int64_t op_idx = 0; op_idx < simple->loop_len; op_idx++) {
        free(simple->dim_info[op_idx].off_out);
        free(simple->dim_info[op_idx].off_in);
    }
    free(simple->dim_info);
    free(simple->op);
    simple->dim_info = NULL;
    simple->op = NULL;
}
/* Has to have the same input and output tensors, with the same shape and be the same op type. Offsets however should be
 * irrelevant */
static int64_t op_equal(const op_t *starting, const op_t *compared) {
    assert(starting);
    assert(compared);
    if(starting->type_op != compared->type_op) {
        return 0;
    }
    if(starting->type_unary != compared->type_unary) {
        return 0;
    }
    if(starting->type_binary != compared->type_binary) {
        return 0;
    }
    if(starting->type_reduce != compared->type_reduce) {
        return 0;
    }
    if(starting->buffer_out.name_off != compared->buffer_out.name_off) {
        return 0;
    }
    if(starting->buffer_out.sze_a != compared->buffer_out.sze_a) {
        return 0;
    }
    if(starting->buffer_out.sze_z != compared->buffer_out.sze_z) {
        return 0;
    }
    if(starting->buffer_out.sze_y != compared->buffer_out.sze_y) {
        return 0;
    }
    if(starting->buffer_out.sze_x != compared->buffer_out.sze_x) {
        return 0;
    }
    if(starting->type_op != op_unary) {
        if(starting->buffer_in.name_off != compared->buffer_in.name_off) {
            return 0;
        }
        if(starting->buffer_in.sze_a != compared->buffer_in.sze_a) {
            return 0;
        }
        if(starting->buffer_in.sze_z != compared->buffer_in.sze_z) {
            return 0;
        }
        if(starting->buffer_in.sze_y != compared->buffer_in.sze_y) {
            return 0;
        }
        if(starting->buffer_in.sze_x != compared->buffer_in.sze_x) {
            return 0;
        }
    }
    return 1;
}
static void simple_loop_configure(simple_loop_t *loop, const op_t **op, const int64_t loop_len,
                                  const int64_t loop_num) {
    assert(loop);
    assert(op);
    assert(loop_len > 0);
    assert(loop_num > 0);
    for(int64_t i = 0; i < loop_num; i++) {
        assert(op[i]);
    }

    if(loop->op) {
        simple_loop_free(loop);
    }
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
            loop->dim_info[j].off_out[i] = op[i][j].buffer_out.off;
            if(op[i][j].type_op != op_unary) {
                loop->dim_info[j].off_in[i] = op[i][j].buffer_in.off;
            }
        }
    }
}
/* Returns the amount of ops in all the iterations of the loop combined, which makes it possible to use like `snprintf`
 * for format string appending */
static int64_t simple_loop_from_linearized_index(simple_loop_t *simple, const linearized_t *linearized,
                                                 const int64_t start_idx) {
    assert(simple);
    assert(linearized);
    assert(start_idx >= 0);
    assert(start_idx < linearized->op_len);
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
        simple_loop_configure(simple, (const op_t **) loop_instances, 1, 1);
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
    simple_loop_configure(simple, (const op_t **) loop_instances, loop_length, loop_number);

    for(int64_t i = 0; i < loop_number; i++) {
        free(loop_instances[i]);
    }
    free(loop_instances);

    return loop_length * loop_number;
}
const int64_t INITIAL_CAP = 4;
#define INLINABLE_OVERRIDE(op)                                                                                         \
    ((op).type_op == op_unary && ((op).type_unary == unary_set)) ||                                                    \
        ((op).type_op == op_binary && ((op).type_binary == binary_copy || (op).type_binary == binary_copy_like))
#define INLINABLE_OVERRIDE_(op)                                                                                        \
    ((op)->type_op == op_unary && ((op)->type_unary == unary_set)) ||                                                  \
        ((op)->type_op == op_binary && ((op)->type_binary == binary_copy || (op)->type_binary == binary_copy_like))

#define OVERRIDES_OUTPUT(op)                                                                                           \
    (((op).type_op == op_unary && ((op).type_unary == unary_set)) ||                                                   \
     ((op).type_op == op_binary && ((op).type_binary == binary_copy || (op).type_binary == binary_copy_like)) ||       \
     ((op).type_op == op_reduce))
#define OVERRIDES_OUTPUT_(op)                                                                                          \
    (((op)->op_type == op_unary && ((op)->type_unary == unary_set)) ||                                                 \
     ((op)->op_type == op_binary && ((op)->type_binary == binary_copy || (op)->type_binary == binary_copy_like)) ||    \
     ((op)->op_type == op_reduce))

static void compile_loop_optimize(compile_loop_t *compile) {
    assert(compile);
    /* Inline */
    int64_t *delete = calloc(compile->op_num, sizeof(int64_t));
    assert(delete);
    for(int64_t op_idx = 0; op_idx < compile->op_num; op_idx++) {
        assert(compile->inline_num[op_idx] == 1);
        if(delete[op_idx]) {
            continue;
        }
        if(INLINABLE_OVERRIDE(compile->op[op_idx][0])) {
            for(int64_t search_off = 1; search_off < compile->op_num - op_idx; search_off++) {
                assert(compile->inline_num[op_idx + search_off] == 1);
                if(compile->op[op_idx][0].buffer_out.name_off ==
                   compile->op[op_idx + search_off][0].buffer_out.name_off) {
                    if(OVERRIDES_OUTPUT(compile->op[op_idx + search_off][0])) {
                        break;
                    } else {
                        int64_t inline_cap = compile->inline_cap[op_idx];
                        int64_t inline_num = compile->inline_num[op_idx];
                        compile->op[op_idx][inline_num] = compile->op[op_idx + search_off][0];
                        compile->dim_info[op_idx][inline_num] = compile->dim_info[op_idx + search_off][0];
                        compile->inline_type[op_idx][inline_num] = inline_op_out;
                        delete[op_idx + search_off] = 1;
                        inline_num++;
                        if(inline_num == inline_cap) {
                            inline_cap *= 2;
                            compile->op[op_idx] = reallocarray(compile->op[op_idx], inline_cap, sizeof(op_t));
                            compile->dim_info[op_idx] =
                                reallocarray(compile->dim_info[op_idx], inline_cap, sizeof(dim_info_t));
                            compile->inline_type[op_idx] =
                                reallocarray(compile->inline_type[op_idx], inline_cap, sizeof(inline_op_e));
                            delete = reallocarray(delete, inline_cap, sizeof(int64_t));
                            assert(compile->op[op_idx]);
                            assert(compile->dim_info[op_idx]);
                            assert(compile->inline_type[op_idx]);
                            assert(delete);
                        }
                        compile->inline_cap[op_idx] = inline_cap;
                        compile->inline_num[op_idx] = inline_num;
                    }
                } else if(compile->op[op_idx][0].buffer_out.name_off ==
                          compile->op[op_idx + search_off][0].buffer_in.name_off) {
                    int64_t inline_cap = compile->inline_cap[op_idx];
                    int64_t inline_num = compile->inline_num[op_idx];
                    compile->op[op_idx][inline_num] = compile->op[op_idx + search_off][0];
                    compile->dim_info[op_idx][inline_num] = compile->dim_info[op_idx + search_off][0];
                    compile->inline_type[op_idx][inline_num] = inline_op_in;
                    delete[op_idx + search_off] = 1;
                    inline_num++;
                    if(inline_num == inline_cap) {
                        inline_cap *= 2;
                        compile->op[op_idx] = reallocarray(compile->op[op_idx], inline_cap, sizeof(op_t));
                        compile->dim_info[op_idx] =
                            reallocarray(compile->dim_info[op_idx], inline_cap, sizeof(dim_info_t));
                        compile->inline_type[op_idx] =
                            reallocarray(compile->inline_type[op_idx], inline_cap, sizeof(inline_op_e));
                        delete = reallocarray(delete, inline_cap, sizeof(int64_t));
                        assert(compile->op[op_idx]);
                        assert(compile->dim_info[op_idx]);
                        assert(compile->inline_type[op_idx]);
                        assert(delete);
                    }
                    compile->inline_cap[op_idx] = inline_cap;
                    compile->inline_num[op_idx] = inline_num;
                }
            }
        }
    }
    int64_t new_idx = 0;
    int64_t new_num = compile->op_num;
    for(int64_t op_idx = 0; op_idx < compile->op_num; op_idx++) {
        if(delete[op_idx]) {
            free(compile->op[op_idx]);
            free(compile->dim_info[op_idx]);
            free(compile->inline_type[op_idx]);
            new_num--;
        } else {
            compile->op[new_idx] = compile->op[op_idx];
            compile->dim_info[new_idx] = compile->dim_info[op_idx];
            compile->inline_type[new_idx] = compile->inline_type[op_idx];
            compile->inline_num[new_idx] = compile->inline_num[op_idx];
            compile->inline_cap[new_idx] = compile->inline_cap[op_idx];
            op_t *op = calloc(compile->inline_num[new_idx], sizeof(op_t));
            dim_info_t *dim_info = calloc(compile->inline_num[new_idx], sizeof(dim_info_t));
            inline_op_e *inline_type = calloc(compile->inline_num[new_idx], sizeof(dim_info_t));
            for(int64_t swap_idx = 0; swap_idx < compile->inline_num[new_idx]; swap_idx++) {
                op[swap_idx] = compile->op[new_idx][swap_idx];
                dim_info[swap_idx] = compile->dim_info[new_idx][swap_idx];
                inline_type[swap_idx] = compile->inline_type[new_idx][swap_idx];
            }
            for(int64_t swap_idx = 0; swap_idx < compile->inline_num[new_idx]; swap_idx++) {
                compile->op[new_idx][swap_idx] = op[compile->inline_num[new_idx] - swap_idx - 1];
                compile->dim_info[new_idx][swap_idx] = dim_info[compile->inline_num[new_idx] - swap_idx - 1];
                compile->inline_type[new_idx][swap_idx] = inline_type[compile->inline_num[new_idx] - swap_idx - 1];
            }
            new_idx++;
            free(op);
            free(dim_info);
            free(inline_type);
        }
    }
    compile->op_num = new_num;
    free(delete);
    /* Fuse */
}
static void compile_loop_print(compile_loop_t *compile, int padding, int offset, const char *name) {
    assert(compile);
    if(strncmp(name, "", 1) != 0) {
        printf("%*s%s\n", offset, "", name);
    } else {
        printf("%*scompile loop with %lu iterations\n", offset, "", compile->loop_num);
    }
    for(int64_t op_idx = 0; op_idx < compile->op_num; op_idx++) {
        printf("%*s %d ", offset + padding, "", compile->inline_type[op_idx][0]);
        op_print(&compile->op[op_idx][0], 0, 0, "");
        for(int64_t inline_idx = 1; inline_idx < compile->inline_num[op_idx]; inline_idx++) {
            printf("%*s %d ", offset + 2 * padding, "", compile->inline_type[op_idx][inline_idx]);
            op_print(&compile->op[op_idx][inline_idx], 0, 0, "");
        }
    }
}
static void compile_loop_free(compile_loop_t *compile) {
    assert(compile);
    assert(compile->op);
    assert(compile->dim_info);
    assert(compile->inline_type);
    assert(compile->inline_num);
    assert(compile->inline_cap);
    for(int64_t i = 0; i < compile->op_num; i++) {
        assert(compile->op[i]);
        assert(compile->inline_type[i]);
        assert(compile->dim_info[i]);
        for(int64_t j = 0; j < compile->inline_num[i]; j++) {
            if(compile->dim_info[i][j].off_out) {
                free(compile->dim_info[i][j].off_out);
            }
            if(compile->dim_info[i][j].off_in) {
                free(compile->dim_info[i][j].off_in);
            }
        }
        free(compile->op[i]);
        free(compile->dim_info[i]);
        free(compile->inline_type[i]);
    }
    free(compile->op);
    free(compile->inline_num);
    free(compile->inline_cap);
    free(compile->dim_info);
    free(compile->inline_type);
}
static compile_loop_t compile_loop_alloc(const simple_loop_t *simple) {
    assert(simple);
    assert(simple->loop_len > 0);
    assert(simple->loop_num > 0);
    compile_loop_t compile = {
        .op_num = simple->loop_len,
        .loop_num = simple->loop_num,
        .op = NULL,
        .inline_num = NULL,
    };
    compile.inline_num = calloc(compile.op_num, sizeof(int64_t));
    compile.inline_cap = calloc(compile.op_num, sizeof(int64_t));
    compile.op = calloc(compile.op_num, sizeof(op_t *));
    compile.dim_info = calloc(compile.op_num, sizeof(dim_info_t *));
    compile.inline_type = calloc(compile.op_num, sizeof(inline_op_e *));
    assert(compile.inline_num);
    assert(compile.inline_cap);
    assert(compile.op);
    assert(compile.dim_info);
    assert(compile.inline_type);
    for(int64_t i = 0; i < compile.op_num; i++) {
        compile.inline_num[i] = 1;
        compile.inline_cap[i] = INITIAL_CAP;
        compile.op[i] = calloc(INITIAL_CAP, sizeof(op_t));
        compile.dim_info[i] = calloc(INITIAL_CAP, sizeof(dim_info_t));
        compile.inline_type[i] = calloc(INITIAL_CAP, sizeof(inline_op_e));
        assert(compile.dim_info[i]);
        assert(compile.op[i]);
        assert(compile.inline_type[i]);
        compile.op[i][0] = simple->op[i];
        compile.inline_type[i][0] = inline_op_none;
        compile.dim_info[i][0].off_out = calloc(compile.loop_num, sizeof(int64_t));
        compile.dim_info[i][0].off_in = calloc(compile.loop_num, sizeof(int64_t));
        assert(compile.dim_info[i][0].off_in);
        assert(compile.dim_info[i][0].off_out);
        for(int64_t j = 0; j < compile.loop_num; j++) {
            compile.dim_info[i][0].off_out[j] = simple->dim_info[i].off_out[j];
            compile.dim_info[i][0].off_in[j] = simple->dim_info[i].off_in[j];
        }
    }
    compile_loop_optimize(&compile);
    return compile;
}
static const int64_t INITIAL_SOURCE_SIZE = 12500;
static const int64_t MAX_OP_SIZE = 1000;
static inline void compile_expand_source(char **source, char **source_curr, int64_t *source_cap,
                                         const int64_t padding) {
    int64_t source_off = *source_curr - *source;
    while(*source_cap - (*source_curr - *source) <= padding) {
        *source_cap *= 2;
    }
    *source = reallocarray(*source, *source_cap, sizeof(char *));
    assert(*source);
    *source_curr = *source + source_off;
}
static void compile_loop_gather_args(kernel_t *kernel, const compile_loop_t *compile) {
    assert(kernel);
    assert(compile);
    int64_t arg_num = 0;
    int64_t arg_cap = INITIAL_CAP;
    char **arg_name = calloc(INITIAL_CAP, sizeof(char *));
    int64_t *arg_name_off = calloc(INITIAL_CAP, sizeof(int64_t));
    cl_mem *arg_mem = calloc(INITIAL_CAP, sizeof(cl_mem));
    assert(arg_name);
    assert(arg_name_off);
    assert(arg_mem);
    for(int64_t op_idx = 0; op_idx < compile->op_num; op_idx++) {
        int64_t found;
        /* Out */
        found = 0;
        for(int64_t arg_idx = 0; arg_idx < arg_num; arg_idx++) {
            if(arg_name_off[arg_idx] == compile->op[op_idx][0].buffer_out.name_off) {
                found = 1;
                break;
            }
        }
        if(!found) {
            if(arg_num == arg_cap) {
                arg_cap *= 2;
                arg_name = reallocarray(arg_name, arg_cap, sizeof(char *));
                arg_name_off = reallocarray(arg_name_off, arg_cap, sizeof(int64_t));
                arg_mem = reallocarray(arg_mem, arg_cap, sizeof(cl_mem));
                assert(arg_name);
                assert(arg_name_off);
                assert(arg_mem);
            }
            arg_name[arg_num] = strndup(compile->op[op_idx][0].buffer_out.name, BUFFER_NAME_SIZE + 1);
            arg_name_off[arg_num] = compile->op[op_idx][0].buffer_out.name_off;
            assert(arg_name[arg_num]);
            arg_mem[arg_num] = compile->op[op_idx][0].buffer_out.val_cl;
            arg_num++;
        }
        if(compile->op[op_idx][0].type_op != op_unary) {
            /* In */
            found = 0;
            for(int64_t arg_idx = 0; arg_idx < arg_num; arg_idx++) {
                if(arg_name_off[arg_idx] == compile->op[op_idx][0].buffer_in.name_off) {
                    found = 1;
                    break;
                }
            }
            if(!found) {
                if(arg_num == arg_cap) {
                    arg_cap *= 2;
                    arg_name = reallocarray(arg_name, arg_cap, sizeof(char *));
                    arg_name_off = reallocarray(arg_name_off, arg_cap, sizeof(int64_t));
                    arg_mem = reallocarray(arg_mem, arg_cap, sizeof(cl_mem));
                    assert(arg_name);
                    assert(arg_name_off);
                    assert(arg_mem);
                }
                arg_name[arg_num] = strndup(compile->op[op_idx][0].buffer_in.name, BUFFER_NAME_SIZE + 1);
                arg_name_off[arg_num] = compile->op[op_idx][0].buffer_in.name_off;
                assert(arg_name[arg_num]);
                arg_mem[arg_num] = compile->op[op_idx][0].buffer_in.val_cl;
                arg_num++;
            }
        }
        for(int64_t inline_idx = 1; inline_idx < compile->inline_num[op_idx]; inline_idx++) {
            if(compile->op[op_idx][inline_idx].type_op == op_unary) {
                /* Out */
                found = 0;
                for(int64_t arg_idx = 0; arg_idx < arg_num; arg_idx++) {
                    if(arg_name_off[arg_idx] != compile->op[op_idx][inline_idx].buffer_out.name_off) {
                        found = 1;
                        break;
                    }
                }
                if(!found) {
                    if(arg_num == arg_cap) {
                        arg_cap *= 2;
                        arg_name = reallocarray(arg_name, arg_cap, sizeof(char *));
                        arg_name_off = reallocarray(arg_name_off, arg_cap, sizeof(int64_t));
                        arg_mem = reallocarray(arg_mem, arg_cap, sizeof(cl_mem));
                        assert(arg_name);
                        assert(arg_name_off);
                        assert(arg_mem);
                    }
                    arg_name[arg_num] = strndup(compile->op[op_idx][inline_idx].buffer_out.name, BUFFER_NAME_SIZE + 1);
                    arg_name_off[arg_num] = compile->op[op_idx][inline_idx].buffer_out.name_off;
                    assert(arg_name[arg_num]);
                    arg_mem[arg_num] = compile->op[op_idx][inline_idx].buffer_out.val_cl;
                    arg_num++;
                }
            } else {
                /* In */
                found = 0;
                for(int64_t arg_idx = 0; arg_idx < arg_num; arg_idx++) {
                    if(arg_name_off[arg_idx] == compile->op[op_idx][inline_idx].buffer_in.name_off) {
                        found = 1;
                        break;
                    }
                }
                if(!found) {
                    if(arg_num == arg_cap) {
                        arg_cap *= 2;
                        arg_name = reallocarray(arg_name, arg_cap, sizeof(char *));
                        arg_name_off = reallocarray(arg_name_off, arg_cap, sizeof(int64_t));
                        arg_mem = reallocarray(arg_mem, arg_cap, sizeof(cl_mem));
                        assert(arg_name);
                        assert(arg_name_off);
                        assert(arg_mem);
                    }
                    arg_name[arg_num] = strndup(compile->op[op_idx][inline_idx].buffer_in.name, BUFFER_NAME_SIZE + 1);
                    arg_name_off[arg_num] = compile->op[op_idx][inline_idx].buffer_in.name_off;
                    assert(arg_name[arg_num]);
                    arg_mem[arg_num] = compile->op[op_idx][inline_idx].buffer_in.val_cl;
                    arg_num++;
                }
            }
        }
    }
    kernel->arg_name = arg_name;
    kernel->arg_mem = arg_mem;
    kernel->arg_num = arg_num;
    kernel->arg_cap = arg_cap;
    free(arg_name_off);
    assert(kernel->arg_num > 0);
}
extern void compile_append_index_table_cl(char **source, char **source_curr, int64_t *source_cap,
                                          const compile_loop_t *loop, const int64_t compile_loop_idx,
                                          const int64_t op_idx, const int64_t inline_num) {
    assert(source);
    assert(*source);
    assert(source_curr);
    assert(*source_curr);
    assert(*source_curr > *source);
    assert(source_cap);
    assert(*source_cap >= INITIAL_SOURCE_SIZE);
    assert(loop);
    assert(op_idx >= 0);
    assert(inline_num > 0);
    *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "__const int %s_%lu_%lu_%d[]={",
                             loop->op[op_idx][0].buffer_out.name, compile_loop_idx, op_idx, 0);
    compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
    for(int64_t loop_idx = 0; loop_idx < loop->loop_num; loop_idx++) {
        if(!loop_idx) {
            *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "%ld", loop->dim_info[op_idx][0].off_out[loop_idx]);
        } else {
            *source_curr += snprintf(*source_curr, MAX_OP_SIZE, ",%ld", loop->dim_info[op_idx][0].off_out[loop_idx]);
        }
        compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
    }
    *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "};\n");
    compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
    if(loop->op[op_idx][0].type_op != op_unary) {
        *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "__const int %s_%lu_%lu_%d[]={",
                                 loop->op[op_idx][0].buffer_in.name, compile_loop_idx, op_idx, 0);
        compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
        for(int64_t loop_idx = 0; loop_idx < loop->loop_num; loop_idx++) {
            if(!loop_idx) {
                *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "%ld", loop->dim_info[op_idx][0].off_in[loop_idx]);
            } else {
                *source_curr += snprintf(*source_curr, MAX_OP_SIZE, ",%ld", loop->dim_info[op_idx][0].off_in[loop_idx]);
            }
            compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
        }
        *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "};\n");
        compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
    }
    for(int64_t inline_idx = 1; inline_idx < inline_num; inline_idx++) {
        if(loop->op[op_idx][inline_idx].type_op == op_unary) {
            *source_curr +=
                snprintf(*source_curr, MAX_OP_SIZE, "__const int %s_%lu_%lu_%lu[]={",
                         loop->op[op_idx][inline_idx].buffer_out.name, compile_loop_idx, op_idx, inline_idx);
            compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
            for(int64_t loop_idx = 0; loop_idx < loop->loop_num; loop_idx++) {
                if(!loop_idx) {
                    *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "%ld",
                                             loop->dim_info[op_idx][inline_idx].off_out[loop_idx]);
                } else {
                    *source_curr += snprintf(*source_curr, MAX_OP_SIZE, ",%ld",
                                             loop->dim_info[op_idx][inline_idx].off_out[loop_idx]);
                }
                compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
            }
            *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "};\n");
            compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
        } else {
            *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "__const int %s_%lu_%lu_%lu[]={",
                                     loop->op[op_idx][inline_idx].buffer_in.name, compile_loop_idx, op_idx, inline_idx);
            compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
            for(int64_t loop_idx = 0; loop_idx < loop->loop_num; loop_idx++) {
                if(!loop_idx) {
                    *source_curr +=
                        snprintf(*source_curr, MAX_OP_SIZE, "%ld", loop->dim_info[op_idx][inline_idx].off_in[loop_idx]);
                } else {
                    *source_curr += snprintf(*source_curr, MAX_OP_SIZE, ",%ld",
                                             loop->dim_info[op_idx][inline_idx].off_in[loop_idx]);
                }
                compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
            }
            *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "};\n");
            compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
        }
    }
}
static void compile_append_op_index(char **source, char **source_curr, int64_t *source_cap, const op_t *op,
                                    const int64_t inline_num, const int64_t compile_loop_idx, const int64_t op_idx,
                                    const int64_t loop_idx) {
    assert(source);
    assert(*source);
    assert(source_curr);
    assert(*source_curr);
    assert(source_cap);
    assert(*source_cap >= INITIAL_SOURCE_SIZE);
    *source_curr +=
        snprintf(*source_curr, MAX_OP_SIZE, "__const int %s_%lu_%lu_%d_%lu=%s_%lu_%lu_%d[id];\n", op[0].buffer_out.name,
                 compile_loop_idx, op_idx, 0, loop_idx, op[0].buffer_out.name, compile_loop_idx, op_idx, 0);
    compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
    if(op[0].type_op != op_unary) {
        *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "__const int %s_%lu_%lu_%d_%lu=%s_%lu_%lu_%d[id];\n",
                                 op[0].buffer_in.name, compile_loop_idx, op_idx, 0, loop_idx, op->buffer_in.name,
                                 compile_loop_idx, op_idx, 0);
        compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
    }
    for(int64_t inline_idx = 1; inline_idx < inline_num; inline_idx++) {
        if(op[inline_idx].type_op == op_unary) {
            *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "__const int %s_%lu_%lu_%lu_%lu=%s_%lu_%lu_%lu[id];\n",
                                     op[inline_idx].buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx,
                                     op[inline_idx].buffer_out.name, compile_loop_idx, op_idx, inline_idx);
        } else {
            *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "__const int %s_%lu_%lu_%lu_%lu=%s_%lu_%lu_%lu[id];\n",
                                     op[inline_idx].buffer_in.name, compile_loop_idx, op_idx, inline_idx, loop_idx,
                                     op[inline_idx].buffer_in.name, compile_loop_idx, op_idx, inline_idx);
        }
        compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
    }
}
static void compile_append_header(char **source, char **source_curr, int64_t *source_cap, const op_t *op,
                                  const int64_t compile_loop_idx, const int64_t op_idx, const int64_t loop_idx) {
    assert(source);
    assert(*source);
    assert(source_curr);
    assert(*source_curr);
    assert(source_cap);
    assert(*source_cap >= INITIAL_SOURCE_SIZE);
    if(op->type_op == op_reduce) {
        char *name = (char *) op->buffer_out.name;
        switch(op->type_reduce) {
            case reduce_sum: {
                *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%d_%lu]=0;\n", name, name,
                                         compile_loop_idx, op_idx, 0, loop_idx);
                break;
            }
            case reduce_avg: {
                *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%d_%lu]=0;\n", name, name,
                                         compile_loop_idx, op_idx, 0, loop_idx);
                break;
            }
            case reduce_max: {
                *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%d_%lu]=-INFINITY;\n", name, name,
                                         compile_loop_idx, op_idx, 0, loop_idx);
                break;
            }
            case reduce_min: {
                *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%d_%lu]=INFINITY;\n", name, name,
                                         compile_loop_idx, op_idx, 0, loop_idx);
                break;
            }
        }
        compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
    }
}
static void compile_append_footer(char **source, char **source_curr, int64_t *source_cap, const op_t *op,
                                  const int64_t compile_loop_idx, const int64_t op_idx, const int64_t loop_idx,
                                  double avg_divisor) {
    assert(source);
    assert(*source);
    assert(source_curr);
    assert(*source_curr);
    assert(source_cap);
    assert(*source_cap >= INITIAL_SOURCE_SIZE);
    if(op->type_op == op_reduce) {
        switch(op->type_reduce) {
            case reduce_sum: {
                break;
            }
            case reduce_avg: {
                char *name = (char *) op->buffer_out.name;
                *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%d_%lu]/=%lf;\n", name, name,
                                         compile_loop_idx, op_idx, 0, loop_idx, avg_divisor);
                break;
            }
            case reduce_max: {
                break;
            }
            case reduce_min: {
                break;
            }
        }
        compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
    }
}
static void compile_append_assign(char **temp, char **temp_curr, int64_t *temp_cap, const op_t *op,
                                  const int64_t op_num, const int64_t compile_loop_idx, const int64_t op_idx,
                                  const int64_t inline_idx, const int64_t loop_idx, const int64_t offset) {
    assert(temp);
    assert(*temp);
    assert(temp_curr);
    assert(*temp_curr);
    assert(temp_cap);
    assert(*temp_cap);
    assert(op);
    assert(compile_loop_idx >= 0);
    assert(op_idx >= 0);
    assert(loop_idx >= 0);
    assert(!inline_idx);
    assert(offset >= 0);
    char *name = (char *) op->buffer_out.name;
    switch(op->type_op) {
        case op_unary: {
            switch(op->type_unary) {
                case unary_add: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]+=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case unary_subtract: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]-=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case unary_multiply: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]*=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case unary_divide: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]/=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case unary_exp: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case unary_log: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case unary_square: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case unary_sqrt: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case unary_reciprocal: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case unary_max: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case unary_min: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case unary_set: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case unary_random: {
                    ERROR("Tried to compile unary_random");
                    break;
                }
                case unary_tanh: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case unary_sign: {
                    TODO();
                    break;
                }
                case unary_absolute: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
            }
            break;
        }
        case op_binary: {
            switch(op->type_binary) {
                case binary_add: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]+=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case binary_subtract: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]-=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case binary_multiply: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]*=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case binary_divide: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]/=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case binary_max: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case binary_min: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case binary_copy: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case binary_add_like: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]+=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case binary_subtract_like: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]-=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case binary_multiply_like: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]*=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case binary_divide_like: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]/=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case binary_max_like: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case binary_min_like: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case binary_copy_like: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
            }
            break;
        }
        case op_reduce: {
            switch(op->type_reduce) {
                case reduce_sum: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu]+=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx);
                    break;
                }
                case reduce_avg: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu]+=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx);
                    break;
                }
                case reduce_max: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx);
                    break;
                }
                case reduce_min: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu]=", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx);
                    break;
                }
            }
            break;
        }
        case op_move: {
            ERROR("Tried to append an assign for a move op");
        }
    }
    compile_expand_source(temp, temp_curr, temp_cap, MAX_OP_SIZE);
}
static void compile_append_prefix(char **temp, char **temp_curr, int64_t *temp_cap, const op_t *op,
                                  const int64_t op_num, const int64_t compile_loop_idx, const int64_t op_idx,
                                  const int64_t inline_idx, const int64_t loop_idx, const int64_t offset,
                                  const inline_op_e inline_type) {
    assert(temp);
    assert(*temp);
    assert(temp_curr);
    assert(*temp_curr);
    assert(temp_cap);
    assert(*temp_cap);
    assert(op);
    assert(compile_loop_idx >= 0);
    assert(op_idx >= 0);
    assert(inline_idx >= 0);
    assert(loop_idx >= 0);
    assert(offset >= 0);
    switch(op->type_op) {
        case op_unary: {
            switch(op->type_unary) {
                case unary_add: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                    break;
                }
                case unary_subtract: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                    break;
                }
                case unary_multiply: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                    break;
                }
                case unary_divide: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                    break;
                }
                case unary_exp: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "exp(");
                    break;
                }
                case unary_log: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "log(");
                    break;
                }
                case unary_square: {
                    /* TODO: Do this with `x*x` for less compilated x instead of `pow(x,2)` cuz that's prolly faster
                     * than a universal `pow` algorithm */
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "pow(");
                    break;
                }
                case unary_sqrt: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "sqrt(");
                    break;
                }
                case unary_reciprocal: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "1/(");
                    break;
                }
                case unary_max: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmax(%lf,", op->var_unary);
                    break;
                }
                case unary_min: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmin(%lf,", op->var_unary);
                    break;
                }
                case unary_set: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                    break;
                }
                case unary_random: {
                    ERROR("Tried to compile unary_random");
                    break;
                }
                case unary_tanh: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "tanh(");
                    break;
                }
                case unary_sign: {
                    TODO();
                    break;
                }
                case unary_absolute: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fabs(");
                    break;
                }
            }
            break;
        }
        case op_binary: {
            char *name;
            if(inline_type == inline_op_none) {
                name = (char *) op->buffer_out.name;
            } else if(inline_type == inline_op_out) {
                name = (char *) op->buffer_in.name;
            } else {
                name = (char *) op->buffer_out.name;
            }
            switch(op->type_binary) {
                case binary_add: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu+%lu]+", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case binary_subtract: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu+%lu]-", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case binary_multiply: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu+%lu]*", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case binary_divide: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu+%lu]/", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case binary_max: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmax(%s[%s_%lu_%lu_%lu_%lu+%lu],", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case binary_min: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmin(%s[%s_%lu_%lu_%lu_%lu+%lu],", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case binary_copy: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                    break;
                }
                case binary_add_like: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu+%lu]+", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case binary_subtract_like: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu+%lu]-", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case binary_multiply_like: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu+%lu]*", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case binary_divide_like: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu+%lu]/", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case binary_max_like: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmax(%s[%s_%lu_%lu_%lu_%lu+%lu],", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case binary_min_like: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmin(%s[%s_%lu_%lu_%lu_%lu+%lu],", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case binary_copy_like: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                    break;
                }
            }
            break;
        }
        case op_reduce: {
            assert(!inline_idx);
            char *name = (char *) op->buffer_out.name;
            switch(op->type_reduce) {
                case reduce_sum: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                    break;
                }
                case reduce_avg: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                    break;
                }
                case reduce_max: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmax(%s[%s_%lu_%lu_%lu_%lu],", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx);
                    break;
                }
                case reduce_min: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmin(%s[%s_%lu_%lu_%lu_%lu],", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx);
                    break;
                }
            }
            break;
        }
        case op_move: {
            ERROR("Tried to append prefix for a move op");
        }
    }
    compile_expand_source(temp, temp_curr, temp_cap, MAX_OP_SIZE);
}
static void compile_append_inner(char **temp, char **temp_curr, int64_t *temp_cap, const op_t *op, const int64_t op_num,
                                 const int64_t compile_loop_idx, const int64_t op_idx, const int64_t inline_idx,
                                 const int64_t loop_idx, const int64_t offset) {
    assert(temp);
    assert(*temp);
    assert(temp_curr);
    assert(*temp_curr);
    assert(temp_cap);
    assert(*temp_cap);
    assert(op);
    assert(compile_loop_idx >= 0);
    assert(op_idx >= 0);
    assert(loop_idx >= 0);
    assert(inline_idx >= 0);
    assert(offset >= 0);
    switch(op->type_op) {
        case op_unary: {
            switch(op->type_unary) {
                case unary_add: {
                    if(op_num != 1) {
                        *temp_curr +=
                            snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", op->buffer_out.name,
                                     op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case unary_subtract: {
                    if(op_num != 1) {
                        *temp_curr +=
                            snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", op->buffer_out.name,
                                     op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case unary_multiply: {
                    if(op_num != 1) {
                        *temp_curr +=
                            snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", op->buffer_out.name,
                                     op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case unary_divide: {
                    if(op_num != 1) {
                        *temp_curr +=
                            snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", op->buffer_out.name,
                                     op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    }
                    break;
                }
                case unary_exp: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", op->buffer_out.name,
                                           op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case unary_log: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", op->buffer_out.name,
                                           op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case unary_square: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", op->buffer_out.name,
                                           op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case unary_sqrt: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", op->buffer_out.name,
                                           op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case unary_reciprocal: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", op->buffer_out.name,
                                           op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case unary_max: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", op->buffer_out.name,
                                           op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case unary_min: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", op->buffer_out.name,
                                           op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case unary_set: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%lf", op->var_unary);
                    break;
                }
                case unary_random: {
                    ERROR("Tried to compile unary_random");
                    break;
                }
                case unary_tanh: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", op->buffer_out.name,
                                           op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case unary_sign: {
                    TODO();
                    break;
                }
                case unary_absolute: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", op->buffer_out.name,
                                           op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
            }
            break;
        }
        case op_binary: {
            char *name = (char *) op->buffer_in.name;
            switch(op->type_binary) {
                case binary_add: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case binary_subtract: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case binary_multiply: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case binary_divide: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case binary_max: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case binary_min: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case binary_copy: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
                    break;
                }
                case binary_add_like: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu]", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx);
                    break;
                }
                case binary_subtract_like: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu]", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx);
                    break;
                }
                case binary_multiply_like: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu]", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx);
                    break;
                }
                case binary_divide_like: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu]", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx);
                    break;
                }
                case binary_max_like: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu]", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx);
                    break;
                }
                case binary_min_like: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu]", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx);
                    break;
                }
                case binary_copy_like: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu]", name, name,
                                           compile_loop_idx, op_idx, inline_idx, loop_idx);
                    break;
                }
            }
            break;
        }
        case op_reduce: {
            assert(!inline_idx);
            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu+%lu]", op->buffer_in.name,
                                   op->buffer_in.name, compile_loop_idx, op_idx, inline_idx, loop_idx, offset);
            break;
        }
        case op_move: {
            ERROR("Tried to append innermost for a move op");
        }
    }
    compile_expand_source(temp, temp_curr, temp_cap, MAX_OP_SIZE);
}
static void compile_append_postfix(char **temp, char **temp_curr, int64_t *temp_cap, const op_t *op,
                                   const int64_t op_num) {
    assert(temp);
    assert(*temp);
    assert(temp_curr);
    assert(*temp_curr);
    assert(temp_cap);
    assert(*temp_cap);
    assert(op);
    switch(op->type_op) {
        case op_unary: {
            switch(op->type_unary) {
                case unary_add: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%lf)", op->var_unary);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "+(%lf))", op->var_unary);
                    }
                    break;
                }
                case unary_subtract: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%lf)", op->var_unary);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "-(%lf))", op->var_unary);
                    }
                    break;
                }
                case unary_multiply: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%lf)", op->var_unary);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "*(%lf))", op->var_unary);
                    }
                    break;
                }
                case unary_divide: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%lf)", op->var_unary);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "/(%lf))", op->var_unary);
                    }
                    break;
                }
                case unary_exp: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
                case unary_log: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
                case unary_square: {
                    /* TODO: Do this with `x*x` for less compilated x instead of `pow(x,2)` cuz that's prolly faster
                     * than a universal `pow` algorithm */
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ",2)");
                    break;
                }
                case unary_sqrt: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
                case unary_reciprocal: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
                case unary_max: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
                case unary_min: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
                case unary_set: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
                case unary_random: {
                    ERROR("Tried to compile unary_random");
                    break;
                }
                case unary_tanh: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
                case unary_sign: {
                    TODO();
                    break;
                }
                case unary_absolute: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
            }
            break;
        }
        case op_binary: {
            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
            break;
        }
        case op_reduce: {
            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
            break;
        }
        case op_move: {
            ERROR("Tried to append prefix for a move op");
        }
    }
    compile_expand_source(temp, temp_curr, temp_cap, MAX_OP_SIZE);
}
static void compile_append_op(char **source, char **source_curr, int64_t *source_cap, const op_t *op,
                              const dim_info_t *dim_info, const inline_op_e *inline_info, const int64_t op_num,
                              const int64_t compile_loop_idx, const int64_t op_idx, const int64_t loop_idx) {
    assert(source);
    assert(*source);
    assert(source_curr);
    assert(*source_curr);
    assert(source_cap);
    assert(*source_cap >= INITIAL_SOURCE_SIZE);
    assert(op);
    assert(dim_info);
    assert(op_num > 0);
    assert(compile_loop_idx >= 0);
    assert(op_idx >= 0);
    assert(loop_idx >= 0);
    int64_t temp_cap = INITIAL_SOURCE_SIZE;
    char *temp = calloc(INITIAL_SOURCE_SIZE, sizeof(char));
    assert(temp);
    char *temp_curr = temp;

    int64_t a_max = op->type_op == op_reduce ? op->buffer_in.sze_a : op->buffer_out.sze_a;
    int64_t z_max = op->type_op == op_reduce ? op->buffer_in.sze_z : op->buffer_out.sze_z;
    int64_t y_max = op->type_op == op_reduce ? op->buffer_in.sze_y : op->buffer_out.sze_y;
    int64_t x_max = op->type_op == op_reduce ? op->buffer_in.sze_x : op->buffer_out.sze_x;
    compile_append_header(source, source_curr, source_cap, op, compile_loop_idx, op_idx, loop_idx);
    for(int64_t a_idx = 0; a_idx < a_max; a_idx++) {
        for(int64_t z_idx = 0; z_idx < z_max; z_idx++) {
            for(int64_t y_idx = 0; y_idx < y_max; y_idx++) {
                for(int64_t x_idx = 0; x_idx < x_max; x_idx++) {
                    int64_t offset = INDEX(op[0].buffer_out, a_idx, z_idx, y_idx, x_idx);
                    compile_append_assign(&temp, &temp_curr, &temp_cap, op, op_num, compile_loop_idx, op_idx, 0,
                                          loop_idx, offset);
                    for(int64_t inline_idx = 0; inline_idx < op_num; inline_idx++) {
                        offset = INDEX(op[inline_idx].buffer_out, a_idx, z_idx, y_idx, x_idx);
                        compile_append_prefix(&temp, &temp_curr, &temp_cap, &op[inline_idx], op_num, compile_loop_idx,
                                              op_idx, inline_idx, loop_idx, offset, inline_info[inline_idx]);
                    }

                    int64_t inner_idx = op_num - 1;
                    offset = op[inner_idx].type_op == op_unary
                                 ? INDEX(op[inner_idx].buffer_out, a_idx, z_idx, y_idx, x_idx)
                                 : INDEX(op[inner_idx].buffer_in, a_idx, z_idx, y_idx, x_idx);
                    compile_append_inner(&temp, &temp_curr, &temp_cap, &op[inner_idx], op_num, compile_loop_idx, op_idx,
                                         inner_idx, loop_idx, offset);

                    for(int64_t inline_idx = op_num - 1; inline_idx >= 0; inline_idx--) {
                        offset = INDEX(op[inner_idx].buffer_out, a_idx, z_idx, y_idx, x_idx);
                        compile_append_postfix(&temp, &temp_curr, &temp_cap, &op[inline_idx], op_num);
                    }

                    *source_curr += snprintf(*source_curr, temp_cap, "%s;\n", temp);
                    compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
                    temp_curr = temp;
                    memset(temp, '\0', temp_cap);
                }
            }
        }
    }
    compile_append_footer(source, source_curr, source_cap, op, compile_loop_idx, op_idx, loop_idx,
                          a_max * z_max * y_max * x_max);
    free(temp);
}
static void compile_loop_to_cl(kernel_t *kernel, const compile_loop_t *compile_loop, const int64_t global_size,
                               const int64_t local_size) {
    assert(kernel);
    assert(compile_loop);
    assert(global_size > 0);
    assert(local_size > 0);
    assert(global_size % local_size == 0);
    compile_loop_gather_args(kernel, compile_loop);

    char *source = calloc(INITIAL_SOURCE_SIZE, sizeof(char));
    assert(source);
    char *source_curr = source;
    int64_t source_cap = INITIAL_SOURCE_SIZE;
    source_curr += snprintf(source_curr, MAX_OP_SIZE, "__kernel void " KERNEL_NAME "(");
    compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
    for(int64_t arg_idx = 0; arg_idx < kernel->arg_num; arg_idx++) {
        if(!arg_idx) {
            source_curr += snprintf(source_curr, MAX_OP_SIZE, "__global double *%s", kernel->arg_name[arg_idx]);
        } else {
            source_curr += snprintf(source_curr, MAX_OP_SIZE, ",__global double *%s", kernel->arg_name[arg_idx]);
        }
        compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
    }
    source_curr += snprintf(source_curr, MAX_OP_SIZE, ") {\n");
    compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
    source_curr += snprintf(source_curr, MAX_OP_SIZE, "__const int gid = get_global_id(0);\n");
    compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
    source_curr += snprintf(source_curr, MAX_OP_SIZE, "int id;\n");
    compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
    for(int64_t op_idx = 0; op_idx < compile_loop->op_num; op_idx++) {
        compile_append_index_table_cl(&source, &source_curr, &source_cap, compile_loop, 0, op_idx,
                                      compile_loop->inline_num[op_idx]);
    }
    int64_t loops_left = compile_loop->loop_num % global_size;
    int64_t loops_per_kernel =
        !loops_left ? compile_loop->loop_num / global_size : compile_loop->loop_num / global_size + 1;
    source_curr += snprintf(source_curr, MAX_OP_SIZE, "id = gid;\n");
    compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
    for(int64_t loop_idx = 0; loop_idx < loops_per_kernel; loop_idx++) {
        if(loop_idx) {
            source_curr += snprintf(source_curr, MAX_OP_SIZE, "id += %lu;\n", global_size);
            compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
        }
        if(loop_idx == loops_per_kernel - 1 && loops_left) {
            source_curr += snprintf(source_curr, MAX_OP_SIZE, "if(gid < %lu) {\n", loops_left);
            compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
        }
        for(int64_t op_idx = 0; op_idx < compile_loop->op_num; op_idx++) {
            compile_append_op_index(&source, &source_curr, &source_cap, compile_loop->op[op_idx],
                                    compile_loop->inline_num[op_idx], 0, op_idx, loop_idx);
            compile_append_op(&source, &source_curr, &source_cap, compile_loop->op[op_idx],
                              compile_loop->dim_info[op_idx], compile_loop->inline_type[op_idx],
                              compile_loop->inline_num[op_idx], 0, op_idx, loop_idx);
        }
        if(loop_idx == loops_per_kernel - 1 && loops_left) {
            source_curr += snprintf(source_curr, MAX_OP_SIZE, "}\n");
            compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
        }
    }
    source_curr += snprintf(source_curr, MAX_OP_SIZE, "}\n");
    compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
    kernel->source = source;
    kernel->source_cap = source_cap;
}
void program_compile(program_t *program, const linearized_t *linearized, const cl_device_id *device_id,
                     const cl_context *context, const cl_command_queue *command_queue, const int64_t global_size,
                     const int64_t local_size) {
    assert(program);
    assert(linearized);
    assert(device_id);
    assert(context);
    assert(command_queue);
    assert(*device_id);
    assert(*context);
    assert(*command_queue);
    /* Having a global or local size of 1 is really stupid but it should be supported. */
    assert(global_size > 0);
    assert(local_size > 0);
    assert(!(global_size % local_size));
    if(!linearized->op_len) {
        return;
    }
    simple_loop_t simple = {0};
    compile_loop_t compile = {0};
    int64_t kernel_num = 0;
    int64_t kernel_cap = INITIAL_CAP;
    program->kernel = calloc(INITIAL_CAP, sizeof(kernel_t));
    int64_t op_idx = 0;
    while(op_idx < linearized->op_len) {
        op_idx += simple_loop_from_linearized_index(&simple, linearized, op_idx);
        compile = compile_loop_alloc(&simple);
        compile_loop_to_cl(&program->kernel[kernel_num], &compile, global_size, local_size);
        program->kernel[kernel_num].cl_program = NULL;
        kernel_num++;
        if(kernel_num == kernel_cap) {
            kernel_cap *= 2;
            program->kernel = reallocarray(program->kernel, kernel_cap, sizeof(kernel_t));
            assert(program->kernel);
        }
        compile_loop_free(&compile);
    }
    simple_loop_free(&simple);
    program->kernel_num = kernel_num;
    program->kernel_cap = kernel_cap;
    program->local_size = local_size;
    program->global_size = global_size;
    program->cl_device_id = (cl_device_id *) device_id;
    program->cl_context = (cl_context *) context;
    program->cl_command_queue = (cl_command_queue *) command_queue;
}
void program_free(program_t *program) {
    for(int64_t kernel_idx = 0; kernel_idx < program->kernel_num; kernel_idx++) {
        for(int64_t arg_idx = 0; arg_idx < program->kernel[kernel_idx].arg_num; arg_idx++) {
            free(program->kernel[kernel_idx].arg_name[arg_idx]);
        }
        free(program->kernel[kernel_idx].arg_name);
        free(program->kernel[kernel_idx].arg_mem);
        free(program->kernel[kernel_idx].source);
        clReleaseKernel(program->kernel[kernel_idx].cl_kernel);
        clReleaseProgram(program->kernel[kernel_idx].cl_program);
    }
    free(program->kernel);
    program->kernel = NULL;
    program->kernel_num = 0;
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
