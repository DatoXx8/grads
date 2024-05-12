#include <CL/cl.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "compile.h"
#include "linearize.h"
#include "tensor.h"

#define SIMPLE_INDEX(simple, a, z, y, x)                                                                               \
    ((simple).str_a * (a + (simple).off_a) + (simple).str_z * (z + (simple).off_z) +                                   \
     (simple).str_y * (y + (simple).off_y) + (simple).str_x * (x + (simple).off_x))
#define SIMPLE_INDEX_(simple, a, z, y, x)                                                                              \
    ((simple)->str_a * (a + (simple)->off_a) + (simple)->str_z * (z + (simple)->off_z) +                               \
     (simple)->str_y * (y + (simple)->off_y) + (simple)->str_x * (x + (simple)->off_x))
static void simple_loop_free(simple_loop_t *simple) {
    assert(simple);
    assert(simple->op);
    assert(simple->dim_info);
    free(simple->op);
    free(simple->dim_info);
}
/* TODO: Don't pass all the loops and just check earlier, that it they are all valid repetitions of
 * each other. */
static void simple_loop_configure(simple_loop_t *loop, simple_op_t **op, int64_t loop_len, int64_t loop_num) {
    assert(loop);
    assert(op);
    assert(loop_len > 0);
    assert(loop_num > 0);
    for(int64_t i = 0; i < loop_num; i++) { assert(op[i]); }

    if(loop->op) { simple_loop_free(loop); }
    loop->loop_num = loop_num;
    loop->loop_len = loop_len;
    loop->op = calloc(loop_len, sizeof(simple_op_t));
    assert(loop->op);
    for(int64_t i = 0; i < loop_len; i++) { loop->op[i] = op[0][i]; }
    loop->dim_info = calloc(loop_len, sizeof(dim_info_t));
    assert(loop->dim_info);
    /* FIX: Currently we assume that the initial op necessarily has the lowest indices, this should
     * *really* be fixed to have some sorting. This would also fix potentially negative strides. */
    int64_t found_a_o, found_a_i;
    int64_t found_z_o, found_z_i;
    int64_t found_y_o, found_y_i;
    int64_t found_x_o, found_x_i;
    for(int64_t i = 0; i < loop_len; i++) {
        found_a_o = 0;
        found_z_o = 0;
        found_y_o = 0;
        found_x_o = 0;
        found_a_i = 0;
        found_z_i = 0;
        found_y_i = 0;
        found_x_i = 0;
        for(int64_t j = 1; j < loop_num; j++) {
            if((!found_a_o) && op[j][i].buffer_out.off_a != op[0][i].buffer_out.off_a) {
                loop->dim_info[i].str_a_out = op[j][i].buffer_out.off_a - op[0][i].buffer_out.off_a;
                loop->dim_info[i].wai_a_out = j;
                assert(loop->dim_info[i].str_a_out > 0);
                assert(loop->dim_info[i].wai_a_out > 0);
                found_a_o = 1;
            }
            if((!found_z_o) && op[j][i].buffer_out.off_z != op[0][i].buffer_out.off_z) {
                loop->dim_info[i].str_z_out = op[j][i].buffer_out.off_z - op[0][i].buffer_out.off_z;
                loop->dim_info[i].wai_z_out = j;
                assert(loop->dim_info[i].str_z_out > 0);
                assert(loop->dim_info[i].wai_z_out > 0);
                found_z_o = 1;
            }
            if((!found_y_o) && op[j][i].buffer_out.off_y != op[0][i].buffer_out.off_y) {
                loop->dim_info[i].str_y_out = op[j][i].buffer_out.off_y - op[0][i].buffer_out.off_y;
                loop->dim_info[i].wai_y_out = j;
                assert(loop->dim_info[i].str_y_out > 0);
                assert(loop->dim_info[i].wai_y_out > 0);
                found_y_o = 1;
            }
            if((!found_x_o) && op[j][i].buffer_out.off_x != op[0][i].buffer_out.off_x) {
                loop->dim_info[i].str_x_out = op[j][i].buffer_out.off_x - op[0][i].buffer_out.off_x;
                loop->dim_info[i].wai_x_out = j;
                assert(loop->dim_info[i].str_x_out > 0);
                assert(loop->dim_info[i].wai_x_out > 0);
                found_x_o = 1;
            }
            if(loop->op[i].type != operation_unary) {
                if((!found_a_i) && op[j][i].buffer_in.off_a != op[0][i].buffer_in.off_a) {
                    loop->dim_info[i].str_a_in = op[j][i].buffer_in.off_a - op[0][i].buffer_in.off_a;
                    loop->dim_info[i].wai_a_in = j;
                    assert(loop->dim_info[i].str_a_in > 0);
                    assert(loop->dim_info[i].wai_a_in > 0);
                    found_a_i = 1;
                }
                if((!found_z_i) && op[j][i].buffer_in.off_z != op[0][i].buffer_in.off_z) {
                    loop->dim_info[i].str_z_in = op[j][i].buffer_in.off_z - op[0][i].buffer_in.off_z;
                    loop->dim_info[i].wai_z_in = j;
                    assert(loop->dim_info[i].str_z_in > 0);
                    assert(loop->dim_info[i].wai_z_in > 0);
                    found_z_i = 1;
                }
                if((!found_y_i) && op[j][i].buffer_in.off_y != op[0][i].buffer_in.off_y) {
                    loop->dim_info[i].str_y_in = op[j][i].buffer_in.off_y - op[0][i].buffer_in.off_y;
                    loop->dim_info[i].wai_y_in = j;
                    assert(loop->dim_info[i].str_y_in > 0);
                    assert(loop->dim_info[i].wai_y_in > 0);
                    found_y_i = 1;
                }
                if((!found_x_i) && op[j][i].buffer_in.off_x != op[0][i].buffer_in.off_x) {
                    loop->dim_info[i].str_x_in = op[j][i].buffer_in.off_x - op[0][i].buffer_in.off_x;
                    loop->dim_info[i].wai_x_in = j;
                    assert(loop->dim_info[i].str_x_in > 0);
                    assert(loop->dim_info[i].wai_x_in > 0);
                    found_x_i = 1;
                }
            }
        }
        if(!found_a_o) {
            loop->dim_info[i].str_a_out = 0;
            loop->dim_info[i].wai_a_out = loop_num;
        }
        if(!found_z_o) {
            loop->dim_info[i].str_z_out = 0;
            loop->dim_info[i].wai_z_out = loop_num;
        }
        if(!found_y_o) {
            loop->dim_info[i].str_y_out = 0;
            loop->dim_info[i].wai_y_out = loop_num;
        }
        if(!found_x_o) {
            loop->dim_info[i].str_x_out = 0;
            loop->dim_info[i].wai_x_out = loop_num;
        }
        if(loop->op[i].type != operation_unary) {
            if(!found_a_i) {
                loop->dim_info[i].str_a_in = 0;
                loop->dim_info[i].wai_a_in = loop_num;
            }
            if(!found_z_i) {
                loop->dim_info[i].str_z_in = 0;
                loop->dim_info[i].wai_z_in = loop_num;
            }
            if(!found_y_i) {
                loop->dim_info[i].str_y_in = 0;
                loop->dim_info[i].wai_y_in = loop_num;
            }
            if(!found_x_i) {
                loop->dim_info[i].str_x_in = 0;
                loop->dim_info[i].wai_x_in = loop_num;
            }
        }
    }
    int64_t left_a_o = 0;
    int64_t left_z_o = 0;
    int64_t left_y_o = 0;
    int64_t left_x_o = 0;
    int64_t left_a_i = 0;
    int64_t left_z_i = 0;
    int64_t left_y_i = 0;
    int64_t left_x_i = 0;
    for(int64_t i = 0; i < loop_len; i++) {
        found_a_o = 0;
        found_z_o = 0;
        found_y_o = 0;
        found_x_o = 0;
        found_a_i = 0;
        found_z_i = 0;
        found_y_i = 0;
        found_x_i = 0;
        left_a_o = 0;
        left_z_o = 0;
        left_y_o = 0;
        left_x_o = 0;
        left_a_i = 0;
        left_z_i = 0;
        left_y_i = 0;
        left_x_i = 0;
        for(int64_t j = 1; j < loop_num; j++) {
            if((!left_a_o) && (!found_a_o) && op[j][i].buffer_out.off_a != op[0][i].buffer_out.off_a) { left_a_o = 1; }
            if(left_a_o && (!found_a_o) && op[j][i].buffer_out.off_a == op[0][i].buffer_out.off_a) {
                loop->dim_info[i].res_a_out = j;
                assert(loop->dim_info[i].res_a_out > 0);
                found_a_o = 1;
            }
            if((!left_z_o) && (!found_z_o) && op[j][i].buffer_out.off_z != op[0][i].buffer_out.off_z) { left_z_o = 1; }
            if(left_z_o && (!found_z_o) && op[j][i].buffer_out.off_z == op[0][i].buffer_out.off_z) {
                loop->dim_info[i].res_z_out = j;
                assert(loop->dim_info[i].res_z_out > 0);
                found_z_o = 1;
            }
            if((!left_y_o) && (!found_y_o) && op[j][i].buffer_out.off_y != op[0][i].buffer_out.off_y) { left_y_o = 1; }
            if(left_y_o && (!found_y_o) && op[j][i].buffer_out.off_y == op[0][i].buffer_out.off_y) {
                loop->dim_info[i].res_y_out = j;
                assert(loop->dim_info[i].res_y_out > 0);
                found_y_o = 1;
            }
            if((!left_x_o) && (!found_x_o) && op[j][i].buffer_out.off_x != op[0][i].buffer_out.off_x) { left_x_o = 1; }
            if(left_x_o && (!found_x_o) && op[j][i].buffer_out.off_x == op[0][i].buffer_out.off_x) {
                loop->dim_info[i].res_x_out = j;
                assert(loop->dim_info[i].res_x_out > 0);
                found_x_o = 1;
            }
            if(loop->op[i].type != operation_unary) {
                if((!left_a_i) && (!found_a_i) && op[j][i].buffer_in.off_a != op[0][i].buffer_in.off_a) {
                    left_a_i = 1;
                }
                if(left_a_i && (!found_a_i) && op[j][i].buffer_in.off_a == op[0][i].buffer_in.off_a) {
                    loop->dim_info[i].res_a_in = j;
                    assert(loop->dim_info[i].res_a_in > 0);
                    found_a_i = 1;
                }
                if((!left_z_i) && (!found_z_i) && op[j][i].buffer_in.off_z != op[0][i].buffer_in.off_z) {
                    left_z_i = 1;
                }
                if(left_z_i && (!found_z_i) && op[j][i].buffer_in.off_z == op[0][i].buffer_in.off_z) {
                    loop->dim_info[i].res_z_in = j;
                    assert(loop->dim_info[i].res_z_in > 0);
                    found_z_i = 1;
                }
                if((!left_y_i) && (!found_y_i) && op[j][i].buffer_in.off_y != op[0][i].buffer_in.off_y) {
                    left_y_i = 1;
                }
                if(left_y_i && (!found_y_i) && op[j][i].buffer_in.off_y == op[0][i].buffer_in.off_y) {
                    loop->dim_info[i].res_y_in = j;
                    assert(loop->dim_info[i].res_y_in > 0);
                    found_y_i = 1;
                }
                if((!left_x_i) && (!found_x_i) && op[j][i].buffer_in.off_x != op[0][i].buffer_in.off_x) {
                    left_x_i = 1;
                }
                if(left_x_i && (!found_x_i) && op[j][i].buffer_in.off_x == op[0][i].buffer_in.off_x) {
                    loop->dim_info[i].res_x_in = j;
                    assert(loop->dim_info[i].res_x_in > 0);
                    found_x_i = 1;
                }
            }
        }
        if(!found_a_o) { loop->dim_info[i].res_a_out = loop_num; }
        if(!found_z_o) { loop->dim_info[i].res_z_out = loop_num; }
        if(!found_y_o) { loop->dim_info[i].res_y_out = loop_num; }
        if(!found_x_o) { loop->dim_info[i].res_x_out = loop_num; }
        if(loop->op[i].type != operation_unary) {
            if(!found_a_i) { loop->dim_info[i].res_a_in = loop_num; }
            if(!found_z_i) { loop->dim_info[i].res_z_in = loop_num; }
            if(!found_y_i) { loop->dim_info[i].res_y_in = loop_num; }
            if(!found_x_i) { loop->dim_info[i].res_x_in = loop_num; }
        }
    }
}
/* This does *not* free the `cl_mem` fields */
static void kernel_free(kernel_t *kernel) {
    assert(kernel);
    for(int64_t i = 0; i < kernel->arg_num; i++) { free(kernel->arg_name[i]); }
    free(kernel->arg_name);
    free((void *) kernel->name);
    free(kernel->source);
    if(kernel->cl_kernel) {
        clReleaseKernel(*kernel->cl_kernel);
        free(kernel->cl_kernel);
    }
}
/* Has to have the same input and output tensors, with the same shape and be the same op type.
 * Offsets however should be irrelevant. */
static bool simple_loop_simple_op_equal(simple_op_t *starting, simple_op_t *compared) {
    assert(starting);
    assert(compared);
    if(starting->type != compared->type) { return false; }
    if(starting->type_unary != compared->type_unary) { return false; }
    if(starting->type_binary != compared->type_binary) { return false; }
    if(starting->type_reduce != compared->type_reduce) { return false; }

    if(strncmp(starting->buffer_out.name, compared->buffer_out.name, BUFFER_NAME_SIZE)) { return false; }
    if(starting->buffer_out.sze_a != compared->buffer_out.sze_a) { return false; }
    if(starting->buffer_out.sze_z != compared->buffer_out.sze_z) { return false; }
    if(starting->buffer_out.sze_y != compared->buffer_out.sze_y) { return false; }
    if(starting->buffer_out.sze_x != compared->buffer_out.sze_x) { return false; }
    if(starting->type != operation_unary) {
        if(strncmp(starting->buffer_in.name, compared->buffer_in.name, BUFFER_NAME_SIZE)) { return false; }
        if(starting->buffer_in.sze_a != compared->buffer_in.sze_a) { return false; }
        if(starting->buffer_in.sze_z != compared->buffer_in.sze_z) { return false; }
        if(starting->buffer_in.sze_y != compared->buffer_in.sze_y) { return false; }
        if(starting->buffer_in.sze_x != compared->buffer_in.sze_x) { return false; }
    }
    return true;
}
/* Returns the amount of ops in all the iterations of the loop combined, which makes it possible to
 * use like `snprintf` for format-string appending. */
static int64_t simple_loop_from_linearized_index(simple_loop_t *simple, linearized_t *linearized, int64_t start_idx) {
    assert(simple);
    assert(linearized);
    assert(start_idx >= 0 && start_idx < linearized->op_len);
    int64_t loop_length = 0;
    int64_t loop_number = 0;
    int64_t diff;
    simple_op_t starting_op = linearized->simple[start_idx];
    for(int64_t i = start_idx + 1; i < linearized->op_len; i++) {
        if(simple_loop_simple_op_equal(&starting_op, &linearized->simple[i])) {
            /* TODO: This could probably just be done in the `for` statement. */
            if(2 * i - start_idx < linearized->op_len) {
                diff = 0;
                for(int64_t j = 0; j < i - start_idx; j++) {
                    if(!simple_loop_simple_op_equal(&linearized->simple[start_idx + j], &linearized->simple[i + j])) {
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
    if(!loop_length) { /* Could not find loop. */
        simple_op_t **loop_instances = calloc(1, sizeof(simple_op_t *));
        assert(loop_instances);
        loop_instances[0] = calloc(1, sizeof(simple_op_t));
        assert(loop_instances[0]);
        loop_instances[0][0] = linearized->simple[start_idx];
        simple_loop_configure(simple, loop_instances, 1, 1);
        free(loop_instances[0]);
        free(loop_instances);
        return 1;
    }
    for(int64_t i = start_idx; i < linearized->op_len; i += loop_length) {
        if(simple_loop_simple_op_equal(&starting_op, &linearized->simple[i])) {
            loop_number++;
        } else {
            break;
        }
    }
    assert(loop_number > 0);

    simple_op_t **loop_instances = calloc(loop_number, sizeof(simple_op_t *));
    assert(loop_instances);
    for(int64_t i = 0; i < loop_number; i++) {
        loop_instances[i] = calloc(loop_length, sizeof(simple_op_t));
        assert(loop_instances[i]);
    }

    for(int64_t i = 0; i < loop_number; i++) {
        for(int64_t j = 0; j < loop_length; j++) {
            loop_instances[i][j] = linearized->simple[start_idx + (loop_length * i) + j];
        }
    }
    simple_loop_configure(simple, loop_instances, loop_length, loop_number);

    for(int64_t i = 0; i < loop_number; i++) { free(loop_instances[i]); }
    free(loop_instances);

    return loop_length * loop_number;
}
const int64_t INITIAL_CAP = 4;
#define OVERRIDES_OUTPUT(op)                                                                                           \
    ((op.type == operation_unary && (op.type_unary == unary_set)) ||                                                   \
     (op.type == operation_binary && (op.type_binary == binary_copy || op.type_binary == binary_copy_like)) ||         \
     (op.type == operation_reduce))
static void compile_loop_optimize(compile_loop_t *compile, uint64_t optim) {
    assert(compile);
    assert(optim <= OPTIMIZE_ALL);
    if(optim & OPTIMIZE_INLINE) {
        // printf("Optimizing: Inline\n");
        int64_t inline_cap = INITIAL_CAP;
        int64_t inline_num = 0;
        simple_op_t *inlined = calloc(INITIAL_CAP, sizeof(simple_op_t));
        dim_info_t *inlined_dim_info = calloc(INITIAL_CAP, sizeof(dim_info_t));
        assert(inlined);
        assert(inlined_dim_info);

        for(int64_t i = 0; i < compile->loop_len; i++) {
            if(compile->op[i][0].type == operation_binary && compile->op[i][0].type_binary == binary_copy) {
                inline_num = 1;
                inlined[0] = compile->op[i][0];
                inlined_dim_info[0] = compile->dim_info[i][0];
                // simple_op_print(&inlined[0], 4, 0, "");
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
                                inlined = reallocarray(inlined, inline_cap, sizeof(simple_op_t));
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
                            compile->op[i + j] =
                                reallocarray(compile->op[i + j], compile->op_cap[i + j], sizeof(simple_op_t));
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
        int64_t c = 0;
        int64_t new_len = compile->loop_len;
        for(int64_t i = 0; i < compile->loop_len; i++) {
            if(compile->op_num[i] == compile->op_cap[i]) {
                free(compile->op[i]);
                free(compile->dim_info[i]);
                new_len--;
            } else {
                compile->op_cap[c] = compile->op_cap[i];
                compile->op_num[c] = compile->op_num[i];
                compile->op[c] = compile->op[i];
                compile->dim_info[c] = compile->dim_info[i];
                c++;
            }
        }
        compile->loop_len = new_len;
    }
    if(optim & OPTIMIZE_FUSE) { printf("Optimizing: Fuse\n"); }
}
// static void compile_loop_print(compile_loop_t *compile, int padding, int offset, const char *name) {
//     assert(compile);
//     if(!strncmp(name, "", 1)) {
//         printf("%*scompile loop repetitions %lu\n", offset, "", compile->loop_num);
//     } else {
//         printf("%*s%s %lu repetitions\n", offset, "", name, compile->loop_num);
//     }
//     for(int64_t i = 0; i < compile->loop_len; i++) {
//         for(int64_t j = 0; j < compile->op_num[i]; j++) {
//             if(j) {
//                 printf("%*s[%lu, %lu] ", 2 * padding + offset, "", i, j);
//             } else {
//                 printf("%*s[%lu, 0] ", padding + offset, "", i);
//             }
//             simple_op_print(&compile->op[i][j], 0, 0, "");
//         }
//     }
//     printf("str\n");
//     for(int64_t i = 0; i < compile->loop_len; i++) {
//         for(int64_t j = 0; j < compile->op_num[i]; j++) {
//             if(j) {
//                 printf("%*s[%lu, %lu] ", 2 * padding + offset, "", i, j);
//             } else {
//                 printf("%*s[%lu, 0] ", padding + offset, "", i);
//             }
//             if(compile->op[i][j].type == operation_unary) {
//                 printf("{%lu, %lu, %lu, %lu}\n", compile->dim_info[i][j].str_a_out,
//                 compile->dim_info[i][j].str_z_out,
//                        compile->dim_info[i][j].str_y_out, compile->dim_info[i][j].str_x_out);
//             } else {
//                 printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", compile->dim_info[i][j].str_a_out,
//                        compile->dim_info[i][j].str_z_out, compile->dim_info[i][j].str_y_out,
//                        compile->dim_info[i][j].str_x_out, compile->dim_info[i][j].str_a_in,
//                        compile->dim_info[i][j].str_z_in, compile->dim_info[i][j].str_y_in,
//                        compile->dim_info[i][j].str_x_in);
//             }
//         }
//     }
//     printf("reset\n");
//     for(int64_t i = 0; i < compile->loop_len; i++) {
//         for(int64_t j = 0; j < compile->op_num[i]; j++) {
//             if(j) {
//                 printf("%*s[%lu, %lu] ", 2 * padding + offset, "", i, j);
//             } else {
//                 printf("%*s[%lu, 0] ", padding + offset, "", i);
//             }
//             if(compile->op[i][j].type == operation_unary) {
//                 printf("{%lu, %lu, %lu, %lu}\n", compile->dim_info[i][j].res_a_out,
//                 compile->dim_info[i][j].res_z_out,
//                        compile->dim_info[i][j].res_y_out, compile->dim_info[i][j].res_x_out);
//             } else {
//                 printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", compile->dim_info[i][j].res_a_out,
//                        compile->dim_info[i][j].res_z_out, compile->dim_info[i][j].res_y_out,
//                        compile->dim_info[i][j].res_x_out, compile->dim_info[i][j].res_a_in,
//                        compile->dim_info[i][j].res_z_in, compile->dim_info[i][j].res_y_in,
//                        compile->dim_info[i][j].res_x_in);
//             }
//         }
//     }
//     printf("wait\n");
//     for(int64_t i = 0; i < compile->loop_len; i++) {
//         for(int64_t j = 0; j < compile->op_num[i]; j++) {
//             if(j) {
//                 printf("%*s[%lu, %lu] ", 2 * padding + offset, "", i, j);
//             } else {
//                 printf("%*s[%lu, 0] ", padding + offset, "", i);
//             }
//             if(compile->op[i][j].type == operation_unary) {
//                 printf("{%lu, %lu, %lu, %lu}\n", compile->dim_info[i][j].wai_a_out,
//                 compile->dim_info[i][j].wai_z_out,
//                        compile->dim_info[i][j].wai_y_out, compile->dim_info[i][j].wai_x_out);
//             } else {
//                 printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", compile->dim_info[i][j].wai_a_out,
//                        compile->dim_info[i][j].wai_z_out, compile->dim_info[i][j].wai_y_out,
//                        compile->dim_info[i][j].wai_x_out, compile->dim_info[i][j].wai_a_in,
//                        compile->dim_info[i][j].wai_z_in, compile->dim_info[i][j].wai_y_in,
//                        compile->dim_info[i][j].wai_x_in);
//             }
//         }
//     }
// }
static void compile_loop_free(compile_loop_t *compile) {
    assert(compile);
    assert(compile->op);
    assert(compile->dim_info);
    assert(compile->op_num);
    assert(compile->op_cap);
    for(int64_t i = 0; i < compile->loop_len; i++) {
        assert(compile->op[i]);
        assert(compile->dim_info[i]);
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
    compile.op = calloc(compile.loop_len, sizeof(simple_op_t *));
    assert(compile.op);
    compile.dim_info = calloc(compile.loop_len, sizeof(dim_info_t *));
    assert(compile.dim_info);
    for(int64_t i = 0; i < compile.loop_len; i++) {
        compile.dim_info[i] = calloc(INITIAL_CAP, sizeof(dim_info_t));
        assert(compile.dim_info[i]);
    }
    for(int64_t i = 0; i < compile.loop_len; i++) {
        compile.op_num[i] = 1;
        compile.op_cap[i] = INITIAL_CAP;
        compile.op[i] = calloc(INITIAL_CAP, sizeof(simple_op_t));
        assert(compile.op[i]);
        compile.op[i][0] = simple->op[i];
        compile.dim_info[i][0] = simple->dim_info[i];
    }
    compile_loop_optimize(&compile, optim);
    return compile;
}
const int64_t INITIAL_SOURCE_SIZE = 1024;
const int64_t MAX_ARG_SIZE = 24;
const int64_t MAX_INDEX_DIGITS = 9;
/* Biggest I found was 131 for `max` or `min` binary ops. */
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
    ((op)->type == operation_unary &&                                                                                  \
     ((op)->type_unary == unary_exp || (op)->type_unary == unary_log || (op)->type_unary == unary_sqrt ||              \
      (op)->type_unary == unary_reciprocal || (op)->type_unary == unary_tanh || (op)->type_unary == unary_absolute ||  \
      (op)->type_unary == unary_max || (op)->type_unary == unary_min)) ||                                              \
        ((op)->type == operation_binary) && ((op)->type_binary == binary_min || (op)->type_binary == binary_max)

/* Pointers for the last 3 cuz they need to be modified, which is kinda horrible but you can't have
 * multiple return types in C. */
static void compile_single_op_to_cl(simple_op_t *op, dim_info_t *dim_info, int64_t op_num, int64_t loop_idx,
                                    int64_t op_idx, char **source, char **curr, int64_t *source_cap) {
    assert(op);
    assert(dim_info);
    assert(op_num > 0);
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
    int64_t max_a = op[0].type == operation_reduce ? op[0].buffer_in.sze_a : op[0].buffer_out.sze_a;
    int64_t max_z = op[0].type == operation_reduce ? op[0].buffer_in.sze_z : op[0].buffer_out.sze_z;
    int64_t max_y = op[0].type == operation_reduce ? op[0].buffer_in.sze_y : op[0].buffer_out.sze_y;
    int64_t max_x = op[0].type == operation_reduce ? op[0].buffer_in.sze_x : op[0].buffer_out.sze_x;
    /* TODO: This needs a really big refactor. */
    /* This is very, very sus. A lot of things could go wrong just from thinking about it. I haven't found a case where
     * it breaks, but be cautious! */
    if(op[0].type == operation_reduce) {
        switch(op[0].type_reduce) {
            case reduce_sum: {
                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]=0;\n", op[0].buffer_out.name,
                                   op[0].buffer_out.name, loop_idx, op_idx);
                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                break;
            }
            case reduce_avg: {
                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]=0;\n", op[0].buffer_out.name,
                                   op[0].buffer_out.name, loop_idx, op_idx);
                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                break;
            }
            case reduce_max: {
                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]=-INFINITY;\n", op[0].buffer_out.name,
                                   op[0].buffer_out.name, loop_idx, op_idx);
                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                break;
            }
            case reduce_min: {
                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]=INFINITY;\n", op[0].buffer_out.name,
                                   op[0].buffer_out.name, loop_idx, op_idx);
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
                        case operation_unary: {
                            switch(op[0].type_unary) {
                                case unary_add: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]+=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_subtract: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]-=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_multiply: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]*=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_divide: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]/=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_exp: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_log: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_square: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_sqrt: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_reciprocal: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_max: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_min: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_set: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
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
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case unary_absolute: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
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
                        case operation_binary: {
                            switch(op[0].type_binary) {
                                case binary_add: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]+=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_subtract: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]-=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_multiply: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]*=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_divide: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]/=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_max: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_min: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_copy: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_add_like: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]+=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_subtract_like: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]-=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_multiply_like: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]*=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_divide_like: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]/=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_max_like: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_min_like: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case binary_copy_like: {
                                    super_temp = temp_c;
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].buffer_out.name,
                                                 op[0].buffer_out.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                            }
                            break;
                        }
                        case operation_reduce: {
                            switch(op[0].type_reduce) {
                                case reduce_sum: {
                                    super_temp = temp_c;
                                    temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]+=", op[0].buffer_out.name,
                                                       op[0].buffer_out.name, loop_idx, op_idx);
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case reduce_avg: {
                                    super_temp = temp_c;
                                    temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]+=", op[0].buffer_out.name,
                                                       op[0].buffer_out.name, loop_idx, op_idx);
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case reduce_max: {
                                    super_temp = temp_c;
                                    temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]=", op[0].buffer_out.name,
                                                       op[0].buffer_out.name, loop_idx, op_idx);
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                                case reduce_min: {
                                    super_temp = temp_c;
                                    temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]=", op[0].buffer_out.name,
                                                       op[0].buffer_out.name, loop_idx, op_idx);
                                    op_offset = temp_c - super_temp;
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                    break;
                                }
                            }
                            break;
                        }
                        case operation_move: {
                            ERROR("Tried to compile move operation to OpenCL at index %lu\n", op_idx);
                        }
                    }
                    if(op_num == 1) {
                        switch(op[0].type) {
                            case operation_unary: {
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
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "exp(%s[%s%luoff%lu+%lu])",
                                                           op[0].buffer_out.name, op[0].buffer_out.name, loop_idx,
                                                           op_idx, SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_log: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "log(%s[%s%luoff%lu+%lu])",
                                                           op[0].buffer_out.name, op[0].buffer_out.name, loop_idx,
                                                           op_idx, SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_square: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]*%s[%s%luoff%lu+%lu]",
                                                     op[0].buffer_out.name, op[0].buffer_out.name, loop_idx, op_idx,
                                                     SIMPLE_INDEX(op[0].buffer_out, a, z, y, x), op[0].buffer_out.name,
                                                     op[0].buffer_out.name, loop_idx, op_idx,
                                                     SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_sqrt: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "sqrt(%s[%s%luoff%lu+%lu])",
                                                           op[0].buffer_out.name, op[0].buffer_out.name, loop_idx,
                                                           op_idx, SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_reciprocal: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "1/(%s[%s%luoff%lu+%lu])",
                                                           op[0].buffer_out.name, op[0].buffer_out.name, loop_idx,
                                                           op_idx, SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_max: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "fmax(%s[%s%luoff%lu+%lu],%lf)",
                                                     op[0].buffer_out.name, op[0].buffer_out.name, loop_idx, op_idx,
                                                     SIMPLE_INDEX(op[0].buffer_out, a, z, y, x), op[0].var_unary);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_min: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "fmin(%s[%s%luoff%lu+%lu],%lf)",
                                                     op[0].buffer_out.name, op[0].buffer_out.name, loop_idx, op_idx,
                                                     SIMPLE_INDEX(op[0].buffer_out, a, z, y, x), op[0].var_unary);
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
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "tanh(%s[%s%luoff%lu+%lu])",
                                                           op[0].buffer_out.name, op[0].buffer_out.name, loop_idx,
                                                           op_idx, SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case unary_absolute: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "exp(%s[%s%luoff%lu+%lu])",
                                                           op[0].buffer_out.name, op[0].buffer_out.name, loop_idx,
                                                           op_idx, SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
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
                            case operation_binary: {
                                switch(op[0].type_binary) {
                                    case binary_add: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]",
                                                           op[0].buffer_in.name, op[0].buffer_in.name, loop_idx, op_idx,
                                                           SIMPLE_INDEX(op[0].buffer_in, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_subtract: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]",
                                                           op[0].buffer_in.name, op[0].buffer_in.name, loop_idx, op_idx,
                                                           SIMPLE_INDEX(op[0].buffer_in, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_multiply: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]",
                                                           op[0].buffer_in.name, op[0].buffer_in.name, loop_idx, op_idx,
                                                           SIMPLE_INDEX(op[0].buffer_in, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_divide: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]",
                                                           op[0].buffer_in.name, op[0].buffer_in.name, loop_idx, op_idx,
                                                           SIMPLE_INDEX(op[0].buffer_in, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_max: {
                                        temp_c += snprintf(
                                            temp_c, MAX_OP_SIZE,
                                            "%s[%s%luoff%lu+%lu]<%s[%s%luoff%lu+%lu]?%s[%s%luoff%lu+%lu]:%s[%s%luoff%lu+%lu]",
                                            op[0].buffer_out.name, op[0].buffer_out.name, loop_idx, op_idx,
                                            SIMPLE_INDEX(op[0].buffer_out, a, z, y, x), op[0].buffer_in.name,
                                            op[0].buffer_in.name, loop_idx, op_idx,
                                            SIMPLE_INDEX(op[0].buffer_in, a, z, y, x), op[0].buffer_in.name,
                                            op[0].buffer_in.name, loop_idx, op_idx,
                                            SIMPLE_INDEX(op[0].buffer_in, a, z, y, x), op[0].buffer_out.name,
                                            op[0].buffer_out.name, loop_idx, op_idx,
                                            SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_min: {
                                        temp_c += snprintf(
                                            temp_c, MAX_OP_SIZE,
                                            "%s[%s%luoff%lu+%lu]>%s[%s%luoff%lu+%lu]?%s[%s%luoff%lu+%lu]:%s[%s%luoff%lu+%lu]",
                                            op[0].buffer_out.name, op[0].buffer_out.name, loop_idx, op_idx,
                                            SIMPLE_INDEX(op[0].buffer_out, a, z, y, x), op[0].buffer_in.name,
                                            op[0].buffer_in.name, loop_idx, op_idx,
                                            SIMPLE_INDEX(op[0].buffer_in, a, z, y, x), op[0].buffer_in.name,
                                            op[0].buffer_in.name, loop_idx, op_idx,
                                            SIMPLE_INDEX(op[0].buffer_in, a, z, y, x), op[0].buffer_out.name,
                                            op[0].buffer_out.name, loop_idx, op_idx,
                                            SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_copy: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]",
                                                           op[0].buffer_in.name, op[0].buffer_in.name, loop_idx, op_idx,
                                                           SIMPLE_INDEX(op[0].buffer_in, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_add_like: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]", op[0].buffer_in.name,
                                                           op[0].buffer_in.name, loop_idx, op_idx);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_subtract_like: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]", op[0].buffer_in.name,
                                                           op[0].buffer_in.name, loop_idx, op_idx);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_multiply_like: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]", op[0].buffer_in.name,
                                                           op[0].buffer_in.name, loop_idx, op_idx);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_divide_like: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]", op[0].buffer_in.name,
                                                           op[0].buffer_in.name, loop_idx, op_idx);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_max_like: {
                                        temp_c += snprintf(
                                            temp_c, MAX_OP_SIZE,
                                            "%s[%s%luoff%lu+%lu]<%s[%s%luoff%lu]?%s[%s%luoff%lu]:%s[%s%luoff%lu+%lu]",
                                            op[0].buffer_out.name, op[0].buffer_out.name, loop_idx, op_idx,
                                            SIMPLE_INDEX(op[0].buffer_out, a, z, y, x), op[0].buffer_in.name,
                                            op[0].buffer_in.name, loop_idx, op_idx, op[0].buffer_in.name,
                                            op[0].buffer_in.name, loop_idx, op_idx, op[0].buffer_out.name,
                                            op[0].buffer_out.name, loop_idx, op_idx,
                                            SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_min_like: {
                                        temp_c += snprintf(
                                            temp_c, MAX_OP_SIZE,
                                            "%s[%s%luoff%lu+%lu]>%s[%s%luoff%lu]?%s[%s%luoff%lu]:%s[%s%luoff%lu+%lu]",
                                            op[0].buffer_out.name, op[0].buffer_out.name, loop_idx, op_idx,
                                            SIMPLE_INDEX(op[0].buffer_out, a, z, y, x), op[0].buffer_in.name,
                                            op[0].buffer_in.name, loop_idx, op_idx, op[0].buffer_in.name,
                                            op[0].buffer_in.name, loop_idx, op_idx, op[0].buffer_out.name,
                                            op[0].buffer_out.name, loop_idx, op_idx,
                                            SIMPLE_INDEX(op[0].buffer_out, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case binary_copy_like: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]", op[0].buffer_in.name,
                                                           op[0].buffer_in.name, loop_idx, op_idx);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                }
                                break;
                            }
                            case operation_reduce: {
                                switch(op[0].type_reduce) {
                                    case reduce_sum: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]",
                                                           op[0].buffer_in.name, op[0].buffer_in.name, loop_idx, op_idx,
                                                           SIMPLE_INDEX(op[0].buffer_in, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case reduce_avg: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]",
                                                           op[0].buffer_in.name, op[0].buffer_in.name, loop_idx, op_idx,
                                                           SIMPLE_INDEX(op[0].buffer_in, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case reduce_max: {
                                        temp_c += snprintf(
                                            temp_c, MAX_OP_SIZE,
                                            "%s[%s%luoff%lu+%lu]>%s[%s%luoff%lu]?%s[%s%luoff%lu+%lu]:%s[%s%luoff%lu]",
                                            op[0].buffer_in.name, op[0].buffer_in.name, loop_idx, op_idx,
                                            SIMPLE_INDEX(op[0].buffer_in, a, z, y, x), op[0].buffer_out.name,
                                            op[0].buffer_out.name, loop_idx, op_idx, op[0].buffer_in.name,
                                            op[0].buffer_in.name, loop_idx, op_idx,
                                            SIMPLE_INDEX(op[0].buffer_in, a, z, y, x), op[0].buffer_out.name,
                                            op[0].buffer_out.name, loop_idx, op_idx);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                    case reduce_min: {
                                        temp_c += snprintf(
                                            temp_c, MAX_OP_SIZE,
                                            "%s[%s%luoff%lu+%lu]<%s[%s%luoff%lu]?%s[%s%luoff%lu+%lu]:%s[%s%luoff%lu]",
                                            op[0].buffer_in.name, op[0].buffer_in.name, loop_idx, op_idx,
                                            SIMPLE_INDEX(op[0].buffer_in, a, z, y, x), op[0].buffer_out.name,
                                            op[0].buffer_out.name, loop_idx, op_idx, op[0].buffer_in.name,
                                            op[0].buffer_in.name, loop_idx, op_idx,
                                            SIMPLE_INDEX(op[0].buffer_in, a, z, y, x), op[0].buffer_out.name,
                                            op[0].buffer_out.name, loop_idx, op_idx);
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                        break;
                                    }
                                }
                                break;
                            }
                            case operation_move: {
                                ERROR("Tried to compile move operation to OpenCL\n");
                            }
                        }
                    } else {
                        for(int64_t i = 1; i < op_num; i++) {
                            if(IS_PREFIX(op + i)) {
                                if(op[i].type == operation_unary) {
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
                                } else if(op[i].type == operation_binary) {
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
                                assert(op[i].type == operation_unary);
                                temp_c += snprintf(temp_c, MAX_OP_SIZE, ")");
                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                            } else {
                                switch(op[i].type) {
                                    case operation_unary: {
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
                                    case operation_binary: {
                                        switch(op[i].type_binary) {
                                            case binary_add: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "+%s[%s%luoff%lu+%lu])",
                                                                   op[i].buffer_in.name, op[i].buffer_in.name, loop_idx,
                                                                   op_idx, SIMPLE_INDEX(op[i].buffer_in, a, z, y, x));
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case binary_subtract: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "-%s[%s%luoff%lu+%lu])",
                                                                   op[i].buffer_in.name, op[i].buffer_in.name, loop_idx,
                                                                   op_idx, SIMPLE_INDEX(op[i].buffer_in, a, z, y, x));
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case binary_multiply: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "*%s[%s%luoff%lu+%lu])",
                                                                   op[i].buffer_in.name, op[i].buffer_in.name, loop_idx,
                                                                   op_idx, SIMPLE_INDEX(op[i].buffer_in, a, z, y, x));
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case binary_divide: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "/%s[%s%luoff%lu+%lu])",
                                                                   op[i].buffer_in.name, op[i].buffer_in.name, loop_idx,
                                                                   op_idx, SIMPLE_INDEX(op[i].buffer_in, a, z, y, x));
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
                                                temp_c += snprintf(temp_c, 3 + (temp_c - super_temp),
                                                                   "%s, %s[%s%luoff%lu+%lu])", dup,
                                                                   op[i].buffer_in.name, op[i].buffer_in.name, loop_idx,
                                                                   op_idx, SIMPLE_INDEX(op[i].buffer_in, a, z, y, x));
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
                                                temp_c += snprintf(temp_c, 3 + (temp_c - super_temp),
                                                                   "%s, %s[%s%luoff%lu+%lu])", dup,
                                                                   op[i].buffer_in.name, op[i].buffer_in.name, loop_idx,
                                                                   op_idx, SIMPLE_INDEX(op[i].buffer_in, a, z, y, x));
                                                free(dup);
                                                break;
                                            }
                                            case binary_copy: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu])",
                                                                   op[i].buffer_in.name, op[i].buffer_in.name, loop_idx,
                                                                   op_idx, SIMPLE_INDEX(op[i].buffer_in, a, z, y, x));
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case binary_add_like: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "+%s[%s%luoff%lu])",
                                                                   op[i].buffer_in.name, op[i].buffer_in.name, loop_idx,
                                                                   op_idx);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case binary_subtract_like: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "-%s[%s%luoff%lu])",
                                                                   op[i].buffer_in.name, op[i].buffer_in.name, loop_idx,
                                                                   op_idx);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case binary_multiply_like: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "*%s[%s%luoff%lu])",
                                                                   op[i].buffer_in.name, op[i].buffer_in.name, loop_idx,
                                                                   op_idx);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                            case binary_divide_like: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "/%s[%s%luoff%lu])",
                                                                   op[i].buffer_in.name, op[i].buffer_in.name, loop_idx,
                                                                   op_idx);
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
                                                temp_c += snprintf(temp_c, 3 + (temp_c - super_temp),
                                                                   "%s, %s[%s%luoff%lu])", dup, op[i].buffer_in.name,
                                                                   op[i].buffer_in.name, loop_idx, op_idx);
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
                                                temp_c += snprintf(temp_c, 3 + (temp_c - super_temp),
                                                                   "%s, %s[%s%luoff%lu])", dup, op[i].buffer_in.name,
                                                                   op[i].buffer_in.name, loop_idx, op_idx);
                                                free(dup);
                                                break;
                                            }
                                            case binary_copy_like: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu])",
                                                                   op[i].buffer_in.name, op[i].buffer_in.name, loop_idx,
                                                                   op_idx);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap, MAX_OP_SIZE);
                                                break;
                                            }
                                        }
                                        break;
                                    }
                                    case operation_reduce: {
                                        ERROR("Tried to inline reduce operation!\n");
                                        break;
                                    }
                                    case operation_move: {
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
    if(op[0].type == operation_reduce && op[0].type_reduce == reduce_avg) {
        temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]/=%lf;\n", op[0].buffer_out.name, op[0].buffer_out.name,
                           loop_idx, op_idx, (double) max_a * max_z * max_y * max_x);
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
int64_t kernel_counter = 0;
static kernel_t compile_loop_to_cl(compile_loop_t *compile, int64_t size_global, int64_t size_local) {
    assert(compile);
    assert(size_global);
    /* TODO: Support `local_size > 1`. */
    assert(size_local == 1);

    char *func_name = calloc(1 + log10(kernel_counter + 1) + 3, sizeof(char));
    assert(func_name);
    snprintf(func_name, 1 + log10(kernel_counter + 1) + 3, "k%lu", kernel_counter);
    kernel_counter++;
    int64_t loops_leftover = compile->loop_num % size_global;
    int64_t loops_assigned = (compile->loop_num - loops_leftover) / size_global;
    int64_t loops_needed;
    if(loops_leftover) {
        loops_needed = loops_assigned + 1;
    } else {
        loops_needed = loops_assigned;
    }

    int64_t source_cap = INITIAL_SOURCE_SIZE;
    char *source = calloc(source_cap, sizeof(char));
    assert(source);
    char *curr = source;
    int64_t offset;

    int64_t arg_num = 0;
    char **arg = NULL;
    cl_mem *arg_cl = NULL;
    int64_t found;
    for(int64_t loop_op_idx = 0; loop_op_idx < compile->loop_len; loop_op_idx++) {
        if(compile->op_num[loop_op_idx] == 1) {
            found = 0;
            for(int64_t arg_idx = 0; arg_idx < arg_num; arg_idx++) {
                if(!strncmp(compile->op[loop_op_idx][0].buffer_out.name, arg[arg_idx], BUFFER_NAME_SIZE)) {
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
                strncpy(arg[arg_num - 1], compile->op[loop_op_idx][0].buffer_out.name, BUFFER_NAME_SIZE);
                arg_cl[arg_num - 1] = compile->op[loop_op_idx][0].buffer_out.val_cl;
            }
            if(compile->op[loop_op_idx][0].type != operation_unary) {
                found = 0;
                for(int64_t j = 0; j < arg_num; j++) {
                    if(!strncmp(compile->op[loop_op_idx][0].buffer_in.name, arg[j], BUFFER_NAME_SIZE)) {
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
                    strncpy(arg[arg_num - 1], compile->op[loop_op_idx][0].buffer_in.name, BUFFER_NAME_SIZE);
                    arg_cl[arg_num - 1] = compile->op[loop_op_idx][0].buffer_in.val_cl;
                }
            }
        } else {
            for(int64_t op_idx = 0; op_idx < compile->op_num[loop_op_idx]; op_idx++) {
                if(op_idx) {
                    if(compile->op[loop_op_idx][op_idx].type != operation_unary) {
                        found = 0;
                        for(int64_t k = 0; k < arg_num; k++) {
                            if(!strncmp(compile->op[loop_op_idx][op_idx].buffer_in.name, arg[k], BUFFER_NAME_SIZE)) {
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
                            strncpy(arg[arg_num - 1], compile->op[loop_op_idx][op_idx].buffer_in.name,
                                    BUFFER_NAME_SIZE);
                            arg_cl[arg_num - 1] = compile->op[loop_op_idx][op_idx].buffer_in.val_cl;
                        }
                    }
                } else {
                    found = 0;
                    for(int64_t k = 0; k < arg_num; k++) {
                        if(!strncmp(compile->op[loop_op_idx][op_idx].buffer_out.name, arg[k], BUFFER_NAME_SIZE)) {
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
                        strncpy(arg[arg_num - 1], compile->op[loop_op_idx][op_idx].buffer_out.name, BUFFER_NAME_SIZE);
                        arg_cl[arg_num - 1] = compile->op[loop_op_idx][op_idx].buffer_out.val_cl;
                    }
                }
            }
        }
    }
    int64_t gid_cap = INITIAL_CAP;
    int64_t gid_len;
    char **gid = calloc(gid_cap, sizeof(char *));
    curr += snprintf(curr, MAX_OP_SIZE, "int gid0 = get_global_id(0);\nint id = gid0;\n");
    EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
    for(int64_t loop_idx = 0; loop_idx < loops_needed; loop_idx++) {
        gid_len = 0;
        if(loop_idx == loops_assigned) {
            curr += snprintf(curr, MAX_OP_SIZE, "if(gid0 < %lu) {\n", loops_leftover);
            EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
        }
        if(loop_idx) {
            curr += snprintf(curr, MAX_OP_SIZE, "id += %lu;\n", size_global);
            EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
        }
        for(int64_t loop_op_idx = 0; loop_op_idx < compile->loop_len; loop_op_idx++) {
            if(compile->op_num[loop_op_idx] == 1) {
                curr += snprintf(
                    curr, MAX_OP_SIZE,
                    "int %s%luoff%lu=(((id%%%lu)/%lu)*%lu)+(((id%%%lu)/%lu)*%lu)+(((id%%%lu)/%lu)*%lu)+(((id%%%lu)/%lu)*%lu);\n",
                    compile->op[loop_op_idx][0].buffer_out.name, loop_idx, loop_op_idx,
                    compile->dim_info[loop_op_idx][0].res_a_out, compile->dim_info[loop_op_idx][0].wai_a_out,
                    compile->dim_info[loop_op_idx][0].str_a_out, compile->dim_info[loop_op_idx][0].res_z_out,
                    compile->dim_info[loop_op_idx][0].wai_z_out, compile->dim_info[loop_op_idx][0].str_z_out,
                    compile->dim_info[loop_op_idx][0].res_y_out, compile->dim_info[loop_op_idx][0].wai_y_out,
                    compile->dim_info[loop_op_idx][0].str_y_out, compile->dim_info[loop_op_idx][0].res_x_out,
                    compile->dim_info[loop_op_idx][0].wai_x_out, compile->dim_info[loop_op_idx][0].str_x_out);
                EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                if(compile->op[loop_op_idx]->type != operation_unary) {
                    curr += snprintf(
                        curr, MAX_OP_SIZE,
                        "int %s%luoff%lu=(((id%%%lu)/%lu)*%lu)+(((id%%%lu)/%lu)*%lu)+(((id%%%lu)/%lu)*%lu)+(((id%%%lu)/%lu)*%lu);\n",
                        compile->op[loop_op_idx][0].buffer_in.name, loop_idx, loop_op_idx,
                        compile->dim_info[loop_op_idx][0].res_a_in, compile->dim_info[loop_op_idx][0].wai_a_in,
                        compile->dim_info[loop_op_idx][0].str_a_in, compile->dim_info[loop_op_idx][0].res_z_in,
                        compile->dim_info[loop_op_idx][0].wai_z_in, compile->dim_info[loop_op_idx][0].str_z_in,
                        compile->dim_info[loop_op_idx][0].res_y_in, compile->dim_info[loop_op_idx][0].wai_y_in,
                        compile->dim_info[loop_op_idx][0].str_y_in, compile->dim_info[loop_op_idx][0].res_x_in,
                        compile->dim_info[loop_op_idx][0].wai_x_in, compile->dim_info[loop_op_idx][0].str_x_in);
                    EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                }
            } else {
                for(int64_t op_idx = 0; op_idx < compile->op_num[loop_op_idx]; op_idx++) {
                    if(op_idx) {
                        if(compile->op[loop_op_idx][op_idx].type != operation_unary) {
                            curr += snprintf(
                                curr, MAX_OP_SIZE,
                                "int %s%luoff%lu=(((id%%%lu)/%lu)*%lu)+(((id%%%lu)/%lu)*%lu)+(((id%%%lu)/%lu)*%lu)+(((id%%%lu)/%lu)*%lu);\n",
                                compile->op[loop_op_idx][op_idx].buffer_in.name, loop_idx, loop_op_idx,
                                compile->dim_info[loop_op_idx][op_idx].res_a_in,
                                compile->dim_info[loop_op_idx][op_idx].wai_a_in,
                                compile->dim_info[loop_op_idx][op_idx].str_a_in,
                                compile->dim_info[loop_op_idx][op_idx].res_z_in,
                                compile->dim_info[loop_op_idx][op_idx].wai_z_in,
                                compile->dim_info[loop_op_idx][op_idx].str_z_in,
                                compile->dim_info[loop_op_idx][op_idx].res_y_in,
                                compile->dim_info[loop_op_idx][op_idx].wai_y_in,
                                compile->dim_info[loop_op_idx][op_idx].str_y_in,
                                compile->dim_info[loop_op_idx][op_idx].res_x_in,
                                compile->dim_info[loop_op_idx][op_idx].wai_x_in,
                                compile->dim_info[loop_op_idx][op_idx].str_x_in);
                            EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                        }
                    } else {
                        curr += snprintf(
                            curr, MAX_OP_SIZE,
                            "int %s%luoff%lu=(((id%%%lu)/%lu)*%lu)+(((id%%%lu)/%lu)*%lu)+(((id%%%lu)/%lu)*%lu)+(((id%%%lu)/%lu)*%lu);\n",
                            compile->op[loop_op_idx][op_idx].buffer_out.name, loop_idx, loop_op_idx,
                            compile->dim_info[loop_op_idx][op_idx].res_a_out,
                            compile->dim_info[loop_op_idx][op_idx].wai_a_out,
                            compile->dim_info[loop_op_idx][op_idx].str_a_out,
                            compile->dim_info[loop_op_idx][op_idx].res_z_out,
                            compile->dim_info[loop_op_idx][op_idx].wai_z_out,
                            compile->dim_info[loop_op_idx][op_idx].str_z_out,
                            compile->dim_info[loop_op_idx][op_idx].res_y_out,
                            compile->dim_info[loop_op_idx][op_idx].wai_y_out,
                            compile->dim_info[loop_op_idx][op_idx].str_y_out,
                            compile->dim_info[loop_op_idx][op_idx].res_x_out,
                            compile->dim_info[loop_op_idx][op_idx].wai_x_out,
                            compile->dim_info[loop_op_idx][op_idx].str_x_out);
                        EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
                    }
                }
            }
            compile_single_op_to_cl(compile->op[loop_op_idx], compile->dim_info[loop_op_idx],
                                    compile->op_num[loop_op_idx], loop_idx, loop_op_idx, &source, &curr, &source_cap);
        }
        for(int64_t gix_idx = 0; gix_idx < gid_len; gix_idx++) { free(gid[gix_idx]); }
        if(loop_idx == loops_assigned) {
            curr += snprintf(curr, MAX_OP_SIZE, "}\n");
            EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap, MAX_OP_SIZE);
        }
    }

    assert(arg_num != 0);
    /* That `+ 3` is pure magic. I have no clue where it comes from. */
    int64_t kernel_size = strlen("__kernel void ") + strlen(func_name) +
                          (strlen("__global double *") + BUFFER_NAME_SIZE) * arg_num + strlen(", ") * (arg_num - 1) +
                          strlen(") {\n") + (curr - source) + strlen("}\n") + 3;
    char *kernel_source = calloc(kernel_size, sizeof(char));
    assert(kernel_source);
    char *kernel_i = kernel_source;
    kernel_i += snprintf(kernel_i, 1 + log10(kernel_counter + 1) + 3 + strnlen("__kernel void(", 15),
                         "__kernel void %s(", func_name);
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

    kernel_t kernel = {
        .arg_num = arg_num,
        .arg_mem = calloc(arg_num, sizeof(cl_mem)),
        .arg_name = calloc(arg_num, sizeof(char *)),
        .name = strndup(func_name, strlen(func_name)),
        .size_local = size_local,
        .size_global = size_global,
        .source = kernel_source,
        .source_cap = source_cap,
        .source_len = kernel_i - kernel_source,
    };
    assert(kernel.arg_name);
    for(int64_t arg_idx = 0; arg_idx < arg_num; arg_idx++) {
        kernel.arg_mem[arg_idx] = arg_cl[arg_idx];
        kernel.arg_name[arg_idx] = strndup(arg[arg_idx], BUFFER_NAME_SIZE + 1);
        free(arg[arg_idx]);
    }
    free(arg_cl);
    free(arg);
    free(source);
    free(func_name);
    free(gid);
    return kernel;
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
    compile_loop_t compile;
    int64_t op_idx = 0;
    kernel_t kernel;
    while(op_idx < linearized->op_len) {
        op_idx += simple_loop_from_linearized_index(&simple, linearized, op_idx);
        compile = compile_loop_alloc(&simple, OPTIMIZE_INLINE);
        kernel = compile_loop_to_cl(&compile, global_size, local_size);
        program->kernel_num++;
        program->kernel = reallocarray(program->kernel, program->kernel_num, sizeof(kernel_t));
        assert(program->kernel);
        program->kernel[program->kernel_num - 1] = kernel;
        compile_loop_free(&compile);
    }
    simple_loop_free(&simple);

    int64_t source_len_cumulative = 0;
    for(int64_t kernel_idx = 0; kernel_idx < program->kernel_num; kernel_idx++) {
        source_len_cumulative += program->kernel[kernel_idx].source_len;
    }
    program->source_len = source_len_cumulative;
    program->source = calloc(source_len_cumulative + 1, sizeof(char)); /* `+1` for '\0'. */
    assert(program->source);
    program->cl_device_id = device_id;
    program->cl_context = context;
    program->cl_command_queue = command_queue;
    char *source_curr = program->source;
    for(int64_t kernel_idx = 0; kernel_idx < program->kernel_num; kernel_idx++) {
        source_curr +=
            snprintf(source_curr, program->kernel[kernel_idx].source_cap, "%s", program->kernel[kernel_idx].source);
    }
}
void program_free(program_t *program) {
    for(int64_t kernel_idx = 0; kernel_idx < program->kernel_num; kernel_idx++) {
        kernel_free(&program->kernel[kernel_idx]);
    }
    free(program->kernel);
    free((void *) program->source);
    /* This is a very disgusting fix, but I suppose it works for now. TODO: Make this nicer. */
    if(program->cl_program) {
        if(*program->cl_program) {
            clReleaseProgram(*program->cl_program);
            *program->cl_program = NULL;
            free(*program->cl_program);
        }
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
