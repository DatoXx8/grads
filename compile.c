#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compile.h"
#include "linearize.h"
#include "tensor.h"
#include "utils.h"

/* TODO: THIS NEEDS TO BE REFACTORS *SO* BAD!!!! THIS IS THE WORST CODE I HAVE EVER WRITTEN!!! */

#define SIMPLE_INDEX(simple, a, z, y, x) ((simple).a_str * (a) + (simple).z_str * (z) + (simple).y_str * (y) + (simple).x_str * (x) + (simple).off)
#define SIMPLE_INDEX_(simple, a, z, y, x) ((simple)->a_str * (a) + (simple)->z_str * (z) + (simple)->y_str * (y) + (simple)->x_str * (x) + (simple)->off)
static void simple_loop_free(simple_loop_t *simple) {
    free(simple->op);
    free(simple->dim_info);
}
/* TODO: Don't pass all the loops and just check earlier, that it they are all valid repetitions of each other. */
static void simple_loop_configure(simple_loop_t *loop, simple_op_t **op, int64_t loop_len, int64_t loop_num) {
    if(loop->op) { simple_loop_free(loop); }
    loop->loop_num = loop_num;
    loop->loop_len = loop_len;
    loop->op = calloc(loop_len, sizeof(simple_op_t));
    assert(loop->op);
    for(int64_t i = 0; i < loop_len; i++) { loop->op[i] = op[0][i]; }
    loop->dim_info = calloc(loop_len, sizeof(dim_info_t));
    assert(loop->dim_info);
    /* FIX: Currently we assume that the initial op necessarily has the lowest indices, this should *really* be fixed to have some sorting. This would also fix
     * potentially negative strides. */
    for(int64_t i = 0; i < loop_len; i++) {
        loop->dim_info[i].off_a_out = loop->op[i].out_buffer.a_off;
        loop->dim_info[i].off_a_out = loop->op[i].out_buffer.z_off;
        loop->dim_info[i].off_a_out = loop->op[i].out_buffer.y_off;
        loop->dim_info[i].off_a_out = loop->op[i].out_buffer.x_off;
        if(loop->op[i].type != operation_unary) {
            loop->dim_info[i].off_a_in = loop->op[i].in_buffer.a_off;
            loop->dim_info[i].off_z_in = loop->op[i].in_buffer.z_off;
            loop->dim_info[i].off_y_in = loop->op[i].in_buffer.y_off;
            loop->dim_info[i].off_x_in = loop->op[i].in_buffer.x_off;
        }
    }
    int64_t found_a_o, found_a_i;
    int64_t found_z_o, found_z_i;
    int64_t found_y_o, found_y_i;
    int64_t found_x_o, found_x_i;
    /* FIX: These are all hacks and most likely don't work. */
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
            if((!found_a_o) && op[j][i].out_buffer.a_off != loop->dim_info[i].off_a_out) {
                loop->dim_info[i].str_a_out = op[j][i].out_buffer.a_off - loop->dim_info[i].off_a_out;
                loop->dim_info[i].wai_a_out = j;
                found_a_o = 1;
            }
            if((!found_z_o) && op[j][i].out_buffer.z_off != loop->dim_info[i].off_z_out) {
                loop->dim_info[i].str_z_out = op[j][i].out_buffer.z_off - loop->dim_info[i].off_z_out;
                loop->dim_info[i].wai_z_out = j;
                found_z_o = 1;
            }
            if((!found_y_o) && op[j][i].out_buffer.y_off != loop->dim_info[i].off_y_out) {
                loop->dim_info[i].str_y_out = op[j][i].out_buffer.y_off - loop->dim_info[i].off_y_out;
                loop->dim_info[i].wai_y_out = j;
                found_y_o = 1;
            }
            if((!found_x_o) && op[j][i].out_buffer.x_off != loop->dim_info[i].off_x_out) {
                loop->dim_info[i].str_x_out = op[j][i].out_buffer.x_off - loop->dim_info[i].off_x_out;
                loop->dim_info[i].wai_x_out = j;
                found_x_o = 1;
            }
            if(loop->op[i].type != operation_unary) {
                if((!found_a_i) && op[j][i].in_buffer.a_off != loop->dim_info[i].off_a_in) {
                    loop->dim_info[i].str_a_in = op[j][i].in_buffer.a_off - loop->dim_info[i].off_a_in;
                    loop->dim_info[i].wai_a_in = j;
                    found_a_i = 1;
                }
                if((!found_z_i) && op[j][i].in_buffer.z_off != loop->dim_info[i].off_z_in) {
                    loop->dim_info[i].str_z_in = op[j][i].in_buffer.z_off - loop->dim_info[i].off_z_in;
                    loop->dim_info[i].wai_z_in = j;
                    found_z_i = 1;
                }
                if((!found_y_i) && op[j][i].in_buffer.y_off != loop->dim_info[i].off_y_in) {
                    loop->dim_info[i].str_y_in = op[j][i].in_buffer.y_off - loop->dim_info[i].off_y_in;
                    loop->dim_info[i].wai_y_in = j;
                    found_y_i = 1;
                }
                if((!found_x_i) && op[j][i].in_buffer.x_off != loop->dim_info[i].off_x_in) {
                    loop->dim_info[i].str_x_in = op[j][i].in_buffer.x_off - loop->dim_info[i].off_x_in;
                    loop->dim_info[i].wai_x_in = j;
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
            if((!left_a_o) && (!found_a_o) && op[j][i].out_buffer.a_off != loop->dim_info[i].off_a_out) { left_a_o = 1; }
            if(left_a_o && (!found_a_o) && op[j][i].out_buffer.a_off == loop->dim_info[i].off_a_out) {
                loop->dim_info[i].res_a_out = j;
                found_a_o = 1;
            }
            if((!left_z_o) && (!found_z_o) && op[j][i].out_buffer.z_off != loop->dim_info[i].off_z_out) { left_z_o = 1; }
            if(left_z_o && (!found_z_o) && op[j][i].out_buffer.z_off == loop->dim_info[i].off_z_out) {
                loop->dim_info[i].res_z_out = j;
                found_z_o = 1;
            }
            if((!left_y_o) && (!found_y_o) && op[j][i].out_buffer.y_off != loop->dim_info[i].off_y_out) { left_y_o = 1; }
            if(left_y_o && (!found_y_o) && op[j][i].out_buffer.y_off == loop->dim_info[i].off_y_out) {
                loop->dim_info[i].res_y_out = j;
                found_y_o = 1;
            }
            if((!left_x_o) && (!found_x_o) && op[j][i].out_buffer.x_off != loop->dim_info[i].off_x_out) { left_x_o = 1; }
            if(left_x_o && (!found_x_o) && op[j][i].out_buffer.x_off == loop->dim_info[i].off_x_out) {
                loop->dim_info[i].res_x_out = j;
                found_x_o = 1;
            }
            if(loop->op[i].type != operation_unary) {
                if((!left_a_i) && (!found_a_i) && op[j][i].in_buffer.a_off != loop->dim_info[i].off_a_in) { left_a_i = 1; }
                if(left_a_i && (!found_a_i) && op[j][i].in_buffer.a_off == loop->dim_info[i].off_a_in) {
                    loop->dim_info[i].res_a_in = j;
                    found_a_i = 1;
                }
                if((!left_z_i) && (!found_z_i) && op[j][i].in_buffer.z_off != loop->dim_info[i].off_z_in) { left_z_i = 1; }
                if(left_z_i && (!found_z_i) && op[j][i].in_buffer.z_off == loop->dim_info[i].off_z_in) {
                    loop->dim_info[i].res_z_in = j;
                    found_z_i = 1;
                }
                if((!left_y_i) && (!found_y_i) && op[j][i].in_buffer.y_off != loop->dim_info[i].off_y_in) { left_y_i = 1; }
                if(left_y_i && (!found_y_i) && op[j][i].in_buffer.y_off == loop->dim_info[i].off_y_in) {
                    loop->dim_info[i].res_y_in = j;
                    found_y_i = 1;
                }
                if((!left_x_i) && (!found_x_i) && op[j][i].in_buffer.x_off != loop->dim_info[i].off_x_in) { left_x_i = 1; }
                if(left_x_i && (!found_x_i) && op[j][i].in_buffer.x_off == loop->dim_info[i].off_x_in) {
                    loop->dim_info[i].res_x_in = j;
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
// static void simple_loop_print(simple_loop_t *simple, int padding, int offset, const char *name) {
//     if(!strncmp(name, "", 1)) {
//         printf("%*ssimple loop %lu repetitions\n", offset, "", simple->loop_num);
//     } else {
//         printf("%*s%s %lu repetitions\n", offset, "", name, simple->loop_num);
//     }
//     for(int64_t i = 0; i < simple->loop_len; i++) {
//         printf("%*s[%lu] ", offset + padding, "", i);
//         simple_op_print(&simple->op[i], 0, 0, "");
//     }
//     printf("off\n");
//     for(int64_t i = 0; i < simple->loop_len; i++) {
//         if(simple->op[i].type == operation_unary) {
//             printf("{%lu, %lu, %lu, %lu}\n", simple->dim_info[i].off_a_out, simple->dim_info[i].off_z_out, simple->dim_info[i].off_y_out,
//                    simple->dim_info[i].off_x_out);
//         } else {
//             printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", simple->dim_info[i].off_a_out, simple->dim_info[i].off_z_out, simple->dim_info[i].off_y_out,
//                    simple->dim_info[i].off_x_out, simple->dim_info[i].off_a_in, simple->dim_info[i].off_z_in, simple->dim_info[i].off_y_in,
//                    simple->dim_info[i].off_x_in);
//         }
//     }
//     printf("str\n");
//     for(int64_t i = 0; i < simple->loop_len; i++) {
//         if(simple->op[i].type == operation_unary) {
//             printf("{%lu, %lu, %lu, %lu}\n", simple->dim_info[i].str_a_out, simple->dim_info[i].str_z_out, simple->dim_info[i].str_y_out,
//                    simple->dim_info[i].str_x_out);
//         } else {
//             printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", simple->dim_info[i].str_a_out, simple->dim_info[i].str_z_out, simple->dim_info[i].str_y_out,
//                    simple->dim_info[i].str_x_out, simple->dim_info[i].str_a_in, simple->dim_info[i].str_z_in, simple->dim_info[i].str_y_in,
//                    simple->dim_info[i].str_x_in);
//         }
//     }
//     printf("res\n");
//     for(int64_t i = 0; i < simple->loop_len; i++) {
//         if(simple->op[i].type == operation_unary) {
//             printf("{%lu, %lu, %lu, %lu}\n", simple->dim_info[i].res_a_out, simple->dim_info[i].res_z_out, simple->dim_info[i].res_y_out,
//                    simple->dim_info[i].res_x_out);
//         } else {
//             printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", simple->dim_info[i].res_a_out, simple->dim_info[i].res_z_out, simple->dim_info[i].res_y_out,
//                    simple->dim_info[i].res_x_out, simple->dim_info[i].res_a_in, simple->dim_info[i].res_z_in, simple->dim_info[i].res_y_in,
//                    simple->dim_info[i].res_x_in);
//         }
//     }
//     printf("wai\n");
//     for(int64_t i = 0; i < simple->loop_len; i++) {
//         if(simple->op[i].type == operation_unary) {
//             printf("{%lu, %lu, %lu, %lu}\n", simple->dim_info[i].wai_a_out, simple->dim_info[i].wai_z_out, simple->dim_info[i].wai_y_out,
//                    simple->dim_info[i].wai_x_out);
//         } else {
//             printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", simple->dim_info[i].wai_a_out, simple->dim_info[i].wai_z_out, simple->dim_info[i].wai_y_out,
//                    simple->dim_info[i].wai_x_out, simple->dim_info[i].wai_a_in, simple->dim_info[i].wai_z_in, simple->dim_info[i].wai_y_in,
//                    simple->dim_info[i].wai_x_in);
//         }
//     }
// }
static void cl_kernel_free(cl_kernel_t *kernel) {
    for(int64_t i = 0; i < kernel->arg_num; i++) { free(kernel->args[i]); }
    free(kernel->args);
    free((void *) kernel->name);
}
// static void cl_kernel_print(cl_kernel_t *kernel, int padding, int offset, const char *name) {
//     if(strncmp(name, "", 1)) {
//         printf("%*s%s %s\n", offset, "", name, kernel->name);
//     } else {
//         printf("%*scl kernel %s\n", offset, "", kernel->name);
//     }
//     for(int64_t i = 0; i < kernel->arg_num; i++) { printf("%*s[%lu] %s\n", padding + offset, "", i, kernel->args[i]); }
// }
/* Has to have the same input and output tensors, with the same shape and be the same op type. Offsets however should be irrelevant. */
static ALWAYS_INLINE bool simple_loop_simple_op_equal(simple_op_t *starting, simple_op_t *compared) {
    /* NOTE: This comparison is probably not needed technically. */
    if(starting->type != compared->type) { return false; }
    /* NOTE: Always checking every single one cuz it probably takes longer to go to the different cases. */
    if(starting->unary_type != compared->unary_type) { return false; }
    if(starting->binary_type != compared->binary_type) { return false; }
    if(starting->reduce_type != compared->reduce_type) { return false; }

    if(strncmp(starting->out_buffer.name, compared->out_buffer.name, BUFFER_NAME_SIZE)) { return false; }
    if(starting->out_buffer.a_sze != compared->out_buffer.a_sze) { return false; }
    if(starting->out_buffer.z_sze != compared->out_buffer.z_sze) { return false; }
    if(starting->out_buffer.y_sze != compared->out_buffer.y_sze) { return false; }
    if(starting->out_buffer.x_sze != compared->out_buffer.x_sze) { return false; }
    if(starting->type != operation_unary) {
        if(strncmp(starting->in_buffer.name, compared->in_buffer.name, BUFFER_NAME_SIZE)) { return false; }
        if(starting->in_buffer.a_sze != compared->in_buffer.a_sze) { return false; }
        if(starting->in_buffer.z_sze != compared->in_buffer.z_sze) { return false; }
        if(starting->in_buffer.y_sze != compared->in_buffer.y_sze) { return false; }
        if(starting->in_buffer.x_sze != compared->in_buffer.x_sze) { return false; }
    }
    return true;
}
/* Returns the amount of ops in all the iterations of the loop combined, which makes it possible to use like `snprintf` for format-string appending. */
static int64_t simple_loop_from_linearized_index(simple_loop_t *simple, linearized_t *linearized, int64_t start_idx) {
    int64_t loop_length = 0;
    int64_t loop_number = 0;
    int64_t diff;
    simple_op_t starting_op = linearized->simple[start_idx];
    for(int64_t i = start_idx + 1; i < linearized->op_count; i++) {
        if(simple_loop_simple_op_equal(&starting_op, &linearized->simple[i])) {
            /* TODO: This could probably just be done in the `for` statement. */
            if(2 * i - start_idx < linearized->op_count) {
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
    for(int64_t i = start_idx; i < linearized->op_count; i += loop_length) {
        if(simple_loop_simple_op_equal(&starting_op, &linearized->simple[i])) {
            loop_number++;
        } else {
            break;
        }
    }

    simple_op_t **loop_instances = calloc(loop_number, sizeof(simple_op_t *));
    assert(loop_instances);
    for(int64_t i = 0; i < loop_number; i++) {
        loop_instances[i] = calloc(loop_length, sizeof(simple_op_t));
        assert(loop_instances[i]);
    }

    for(int64_t i = 0; i < loop_number; i++) {
        for(int64_t j = 0; j < loop_length; j++) { loop_instances[i][j] = linearized->simple[start_idx + (loop_length * i) + j]; }
    }
    simple_loop_configure(simple, loop_instances, loop_length, loop_number);

    for(int64_t i = 0; i < loop_number; i++) { free(loop_instances[i]); }
    free(loop_instances);

    return loop_length * loop_number;
}
const int64_t INITIAL_CAP = 4;
#define OVERRIDES_OUTPUT(op)                                                                                                                                   \
    ((op.type == operation_unary && (op.unary_type == unary_set)) ||                                                                                           \
     (op.type == operation_binary && (op.binary_type == binary_copy || op.binary_type == binary_copy_like)) || (op.type == operation_reduce))
static void compile_loop_optimize(compile_loop_t *compile, uint64_t optim) {
    if(optim & OPTIMIZE_INLINE) {
        printf("Optimizing: Inline\n");
        int64_t inline_cap = INITIAL_CAP;
        int64_t inline_num = 0;
        simple_op_t *inlined = calloc(INITIAL_CAP, sizeof(simple_op_t));
        dim_info_t *inlined_dim_info = calloc(INITIAL_CAP, sizeof(dim_info_t));
        assert(inlined);

        /* FIX: If the op at [i][0] already has stuff inlined then that will get lost. This should not be the case. */
        for(int64_t i = 0; i < compile->loop_len; i++) {
            if(compile->op[i][0].type == operation_binary && compile->op[i][0].binary_type == binary_copy) {
                inline_num = 1;
                inlined[0] = compile->op[i][0];
                inlined_dim_info[0] = compile->dim_info[i][0];
                simple_op_print(&inlined[0], 4, 0, "");
                for(int64_t j = 1; j < compile->loop_len - i; j++) {
                    if(!strncmp(compile->op[i][0].out_buffer.name, compile->op[i + j][0].out_buffer.name, BUFFER_NAME_SIZE)) {
                        if(OVERRIDES_OUTPUT(compile->op[i + j][0])) {
                            break;
                        } else {
                            compile->op_num[i + j] = compile->op_cap[i + j];
                            inline_num++;
                            if(inline_num == inline_cap) {
                                inline_cap *= 2;
                                inlined = reallocarray(inlined, inline_cap, sizeof(simple_op_t));
                                inlined_dim_info = reallocarray(inlined_dim_info, inline_cap, sizeof(dim_info_t));
                            }
                            inlined[inline_num - 1] = compile->op[i + j][0];
                            inlined_dim_info[inline_num - 1] = compile->dim_info[i + j][0];
                        }
                    } else if(!strncmp(compile->op[i][0].out_buffer.name, compile->op[i + j][0].in_buffer.name, BUFFER_NAME_SIZE)) {
                        compile->op_num[i] = compile->op_cap[i];
                        compile->op_num[i + j] += inline_num;
                        if(compile->op_num[i + j] >= compile->op_cap[i + j]) {
                            compile->op_cap[i + j] *= 2;
                            compile->op[i + j] = reallocarray(compile->op[i + j], compile->op_cap[i + j], sizeof(simple_op_t));
                            compile->dim_info[i + j] = reallocarray(compile->dim_info[i + j], compile->op_cap[i + j], sizeof(dim_info_t));
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
        /* Kinda stupid to do it here. */
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
static void compile_loop_print(compile_loop_t *compile, int padding, int offset, const char *name) {
    if(!strncmp(name, "", 1)) {
        printf("%*scompile loop repetitions %lu\n", offset, "", compile->loop_num);
    } else {
        printf("%*s%s %lu repetitions\n", offset, "", name, compile->loop_num);
    }
    for(int64_t i = 0; i < compile->loop_len; i++) {
        for(int64_t j = 0; j < compile->op_num[i]; j++) {
            if(j) {
                printf("%*s[%lu, %lu] ", 2 * padding + offset, "", i, j);
            } else {
                printf("%*s[%lu, 0] ", padding + offset, "", i);
            }
            simple_op_print(&compile->op[i][j], 0, 0, "");
        }
    }
    printf("off\n");
    for(int64_t i = 0; i < compile->loop_len; i++) {
        for(int64_t j = 0; j < compile->op_num[i]; j++) {
            if(j) {
                printf("%*s[%lu, %lu] ", 2 * padding + offset, "", i, j);
            } else {
                printf("%*s[%lu, 0] ", padding + offset, "", i);
            }
            if(compile->op[i][j].type == operation_unary) {
                printf("{%lu, %lu, %lu, %lu}\n", compile->dim_info[i][j].off_a_out, compile->dim_info[i][j].off_z_out, compile->dim_info[i][j].off_y_out,
                       compile->dim_info[i][j].off_x_out);
            } else {
                printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", compile->dim_info[i][j].off_a_out, compile->dim_info[i][j].off_z_out,
                       compile->dim_info[i][j].off_y_out, compile->dim_info[i][j].off_x_out, compile->dim_info[i][j].off_a_in, compile->dim_info[i][j].off_z_in,
                       compile->dim_info[i][j].off_y_in, compile->dim_info[i][j].off_x_in);
            }
        }
    }
    printf("str\n");
    for(int64_t i = 0; i < compile->loop_len; i++) {
        for(int64_t j = 0; j < compile->op_num[i]; j++) {
            if(j) {
                printf("%*s[%lu, %lu] ", 2 * padding + offset, "", i, j);
            } else {
                printf("%*s[%lu, 0] ", padding + offset, "", i);
            }
            if(compile->op[i][j].type == operation_unary) {
                printf("{%lu, %lu, %lu, %lu}\n", compile->dim_info[i][j].str_a_out, compile->dim_info[i][j].str_z_out, compile->dim_info[i][j].str_y_out,
                       compile->dim_info[i][j].str_x_out);
            } else {
                printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", compile->dim_info[i][j].str_a_out, compile->dim_info[i][j].str_z_out,
                       compile->dim_info[i][j].str_y_out, compile->dim_info[i][j].str_x_out, compile->dim_info[i][j].str_a_in, compile->dim_info[i][j].str_z_in,
                       compile->dim_info[i][j].str_y_in, compile->dim_info[i][j].str_x_in);
            }
        }
    }
    printf("reset\n");
    for(int64_t i = 0; i < compile->loop_len; i++) {
        for(int64_t j = 0; j < compile->op_num[i]; j++) {
            if(j) {
                printf("%*s[%lu, %lu] ", 2 * padding + offset, "", i, j);
            } else {
                printf("%*s[%lu, 0] ", padding + offset, "", i);
            }
            if(compile->op[i][j].type == operation_unary) {
                printf("{%lu, %lu, %lu, %lu}\n", compile->dim_info[i][j].res_a_out, compile->dim_info[i][j].res_z_out, compile->dim_info[i][j].res_y_out,
                       compile->dim_info[i][j].res_x_out);
            } else {
                printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", compile->dim_info[i][j].res_a_out, compile->dim_info[i][j].res_z_out,
                       compile->dim_info[i][j].res_y_out, compile->dim_info[i][j].res_x_out, compile->dim_info[i][j].res_a_in, compile->dim_info[i][j].res_z_in,
                       compile->dim_info[i][j].res_y_in, compile->dim_info[i][j].res_x_in);
            }
        }
    }
    printf("wait\n");
    for(int64_t i = 0; i < compile->loop_len; i++) {
        for(int64_t j = 0; j < compile->op_num[i]; j++) {
            if(j) {
                printf("%*s[%lu, %lu] ", 2 * padding + offset, "", i, j);
            } else {
                printf("%*s[%lu, 0] ", padding + offset, "", i);
            }
            if(compile->op[i][j].type == operation_unary) {
                printf("{%lu, %lu, %lu, %lu}\n", compile->dim_info[i][j].wai_a_out, compile->dim_info[i][j].wai_z_out, compile->dim_info[i][j].wai_y_out,
                       compile->dim_info[i][j].wai_x_out);
            } else {
                printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", compile->dim_info[i][j].wai_a_out, compile->dim_info[i][j].wai_z_out,
                       compile->dim_info[i][j].wai_y_out, compile->dim_info[i][j].wai_x_out, compile->dim_info[i][j].wai_a_in, compile->dim_info[i][j].wai_z_in,
                       compile->dim_info[i][j].wai_y_in, compile->dim_info[i][j].wai_x_in);
            }
        }
    }
}
static void compile_loop_free(compile_loop_t *compile) {
    for(int64_t i = 0; i < compile->loop_len; i++) {
        free(compile->op[i]);
        free(compile->dim_info[i]);
    }
    free(compile->op);
    free(compile->op_num);
    free(compile->op_cap);
    free(compile->dim_info);
}
static compile_loop_t compile_loop_alloc(simple_loop_t *simple, int64_t optim) {
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
/* NOTE: Biggest I found was 131 for `max` or `min` binary ops. */
const int64_t MAX_OP_SIZE = 512;
#define EXPAND_SOURCE_IF_NEEDED(curr, source, source_size)                                                                                                     \
    if(source_size - (curr - source) <= MAX_OP_SIZE) {                                                                                                         \
        source_size *= 2;                                                                                                                                      \
        offset = curr - source;                                                                                                                                \
        source = reallocarray(source, source_size, sizeof(char));                                                                                              \
        assert(source);                                                                                                                                        \
        curr = source + offset;                                                                                                                                \
    }
#define IS_PREFIX(op)                                                                                                                                          \
    (op)->type == operation_unary && ((op)->unary_type == unary_exp || (op)->unary_type == unary_log || (op)->unary_type == unary_sqrt ||                      \
                                      (op)->unary_type == unary_reciprocal || (op)->unary_type == unary_tanh || (op)->unary_type == unary_absolute)

/* Pointers for the last 3 cuz they need to be modified, which is kinda horrible but you can't have multiple return types in C. */
static void compile_single_op_to_cl(simple_op_t *op, dim_info_t *dim_info, int64_t op_num, int64_t loop_idx, int64_t op_idx, char **source, char **curr,
                                    int64_t *source_cap) {
    int64_t offset;
    int64_t temp_cap = INITIAL_SOURCE_SIZE;
    char *temp = calloc(INITIAL_SOURCE_SIZE, sizeof(char));
    char *temp_c = temp;
    int64_t max_a = op[0].type == operation_reduce ? op[0].in_buffer.a_sze : op[0].out_buffer.a_sze;
    int64_t max_z = op[0].type == operation_reduce ? op[0].in_buffer.z_sze : op[0].out_buffer.z_sze;
    int64_t max_y = op[0].type == operation_reduce ? op[0].in_buffer.y_sze : op[0].out_buffer.y_sze;
    int64_t max_x = op[0].type == operation_reduce ? op[0].in_buffer.x_sze : op[0].out_buffer.x_sze;
    /* TODO: This needs a really big refactor. */
    /* WARN: This is very, very sus. A lot of things could go wrong just from thinking about it. I haven't found a case where it breaks, but be cautious! */
    if(op[0].type == operation_reduce) {
        switch(op[0].reduce_type) {
            case reduce_sum: {
                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]=0;\n", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx);
                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                break;
            }
            case reduce_avg: {
                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]=0;\n", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx);
                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                break;
            }
            case reduce_max: {
                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]=-INFINITY;\n", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx);
                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                TODO();
                break;
            }
            case reduce_min: {
                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]=INFINITY;\n", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx);
                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                TODO();
                break;
            }
        }
    }
    for(int64_t a = 0; a < max_a; a++) {
        for(int64_t z = 0; z < max_z; z++) {
            for(int64_t y = 0; y < max_y; y++) {
                for(int64_t x = 0; x < max_x; x++) {
                    switch(op[0].type) {
                        case operation_unary: {
                            switch(op[0].unary_type) {
                                case unary_add: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]+=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                              (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                    break;
                                }
                                case unary_subtract: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]-=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                              (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                    break;
                                }
                                case unary_multiply: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]*=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                              (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                    break;
                                }
                                case unary_divide: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]/=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                              (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                    break;
                                }
                                case unary_exp: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                              (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                    break;
                                }
                                case unary_log: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                              (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                    break;
                                }
                                case unary_square: {
                                    TODO();
                                    break;
                                }
                                case unary_sqrt: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                              (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                    break;
                                }
                                case unary_reciprocal: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                              (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                    break;
                                }
                                case unary_max: {
                                    TODO();
                                    break;
                                }
                                case unary_min: {
                                    TODO();
                                    break;
                                }
                                case unary_set: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                              (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                    break;
                                }
                                case unary_random: {
                                    TODO();
                                    break;
                                }
                                case unary_tanh: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                              (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                    break;
                                }
                                case unary_absolute: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                              (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
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
                            switch(op[0].binary_type) {
                                case binary_add: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]+=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                              (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                    break;
                                }
                                case binary_subtract: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]-=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                              (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                    break;
                                }
                                case binary_multiply: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]*=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                              (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                    break;
                                }
                                case binary_divide: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]/=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                              (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                    break;
                                }
                                case binary_max: {
                                    TODO();
                                    break;
                                }
                                case binary_min: {
                                    TODO();
                                    break;
                                }
                                case binary_copy: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                 SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                              (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                    break;
                                }
                                case binary_add_like: {
                                    ERROR("THIS IS SUS!");
                                    break;
                                }
                                case binary_subtract_like: {
                                    ERROR("THIS IS SUS!");
                                    break;
                                }
                                case binary_multiply_like: {
                                    ERROR("THIS IS SUS!");
                                    break;
                                }
                                case binary_divide_like: {
                                    ERROR("THIS IS SUS!");
                                    break;
                                }
                                case binary_max_like: {
                                    TODO();
                                    break;
                                }
                                case binary_min_like: {
                                    TODO();
                                    break;
                                }
                                case binary_copy_like: {
                                    ERROR("THIS IS SUS!");
                                    break;
                                }
                            }
                            break;
                        }
                        case operation_reduce: {
                            switch(op[0].reduce_type) {
                                case reduce_sum: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]+=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx);
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                    break;
                                }
                                case reduce_avg: {
                                    temp_c +=
                                        snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu]+=", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx);
                                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                    break;
                                }
                                case reduce_max: {
                                    TODO();
                                    break;
                                }
                                case reduce_min: {
                                    TODO();
                                    break;
                                }
                            }
                            break;
                        }
                        case operation_move: {
                            ERROR("ERROR: Tried to compile move operation to OpenCL at index %lu\n", op_idx);
                        }
                    }
                    if(op_num == 1) {
                        switch(op[0].type) {
                            case operation_unary: {
                                switch(op[0].unary_type) {
                                    case unary_add: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]", op[0].in_buffer.name, op[0].in_buffer.name, loop_idx, op_idx,
                                                     SIMPLE_INDEX(op[0].in_buffer, (a - dim_info[0].off_a_in), (z - dim_info[0].off_z_in),
                                                                  (y - dim_info[0].off_y_in), (x - dim_info[0].off_x_in)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case unary_subtract: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                     SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                                  (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case unary_multiply: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                     SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                                  (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case unary_divide: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                     SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                                  (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case unary_exp: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "exp(%s[%s%luoff%lu+%lu])", op[0].out_buffer.name, op[0].out_buffer.name,
                                                           loop_idx, op_idx,
                                                           SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                                        (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case unary_log: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "log(%s[%s%luoff%lu+%lu])", op[0].out_buffer.name, op[0].out_buffer.name,
                                                           loop_idx, op_idx,
                                                           SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                                        (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case unary_square: {
                                        TODO();
                                        break;
                                    }
                                    case unary_sqrt: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "sqrt(%s[%s%luoff%lu+%lu])", op[0].out_buffer.name, op[0].out_buffer.name,
                                                           loop_idx, op_idx,
                                                           SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                                        (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case unary_reciprocal: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "1/(%s[%s%luoff%lu+%lu])", op[0].out_buffer.name, op[0].out_buffer.name,
                                                           loop_idx, op_idx,
                                                           SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                                        (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case unary_max: {
                                        TODO();
                                        break;
                                    }
                                    case unary_min: {
                                        TODO();
                                        break;
                                    }
                                    case unary_set: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]", op[0].out_buffer.name, op[0].out_buffer.name, loop_idx, op_idx,
                                                     SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                                  (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case unary_random: {
                                        TODO();
                                        break;
                                    }
                                    case unary_tanh: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "tanh(%s[%s%luoff%lu+%lu])", op[0].out_buffer.name, op[0].out_buffer.name,
                                                           loop_idx, op_idx,
                                                           SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                                        (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case unary_absolute: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "exp(%s[%s%luoff%lu+%lu])", op[0].out_buffer.name, op[0].out_buffer.name,
                                                           loop_idx, op_idx,
                                                           SIMPLE_INDEX(op[0].out_buffer, (a - dim_info[0].off_a_out), (z - dim_info[0].off_z_out),
                                                                        (y - dim_info[0].off_y_out), (x - dim_info[0].off_x_out)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
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
                                switch(op[0].binary_type) {
                                    case binary_add: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]", op[0].in_buffer.name, op[0].in_buffer.name, loop_idx, op_idx,
                                                     SIMPLE_INDEX(op[0].in_buffer, (a - dim_info[0].off_a_in), (z - dim_info[0].off_z_in),
                                                                  (y - dim_info[0].off_y_in), (x - dim_info[0].off_x_in)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case binary_subtract: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]", op[0].in_buffer.name, op[0].in_buffer.name, loop_idx, op_idx,
                                                     SIMPLE_INDEX(op[0].in_buffer, (a - dim_info[0].off_a_in), (z - dim_info[0].off_z_in),
                                                                  (y - dim_info[0].off_y_in), (x - dim_info[0].off_x_in)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case binary_multiply: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]", op[0].in_buffer.name, op[0].in_buffer.name, loop_idx, op_idx,
                                                     SIMPLE_INDEX(op[0].in_buffer, (a - dim_info[0].off_a_in), (z - dim_info[0].off_z_in),
                                                                  (y - dim_info[0].off_y_in), (x - dim_info[0].off_x_in)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case binary_divide: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]", op[0].in_buffer.name, op[0].in_buffer.name, loop_idx, op_idx,
                                                     SIMPLE_INDEX(op[0].in_buffer, (a - dim_info[0].off_a_in), (z - dim_info[0].off_z_in),
                                                                  (y - dim_info[0].off_y_in), (x - dim_info[0].off_x_in)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case binary_max: {
                                        TODO();
                                        break;
                                    }
                                    case binary_min: {
                                        TODO();
                                        break;
                                    }
                                    case binary_copy: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]", op[0].in_buffer.name, op[0].in_buffer.name, loop_idx, op_idx,
                                                     SIMPLE_INDEX(op[0].in_buffer, (a - dim_info[0].off_a_in), (z - dim_info[0].off_z_in),
                                                                  (y - dim_info[0].off_y_in), (x - dim_info[0].off_x_in)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case binary_add_like: {
                                        ERROR("THIS IS SUS!");
                                        break;
                                    }
                                    case binary_subtract_like: {
                                        ERROR("THIS IS SUS!");
                                        break;
                                    }
                                    case binary_multiply_like: {
                                        ERROR("THIS IS SUS!");
                                        break;
                                    }
                                    case binary_divide_like: {
                                        ERROR("THIS IS SUS!");
                                        break;
                                    }
                                    case binary_max_like: {
                                        TODO();
                                        break;
                                    }
                                    case binary_min_like: {
                                        TODO();
                                        break;
                                    }
                                    case binary_copy_like: {
                                        ERROR("THIS IS SUS!");
                                        break;
                                    }
                                }
                                break;
                            }
                            case operation_reduce: {
                                switch(op[0].reduce_type) {
                                    case reduce_sum: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]", op[0].in_buffer.name, op[0].in_buffer.name, loop_idx, op_idx,
                                                     SIMPLE_INDEX(op[0].in_buffer, (a - dim_info[0].off_a_in), (z - dim_info[0].off_z_in),
                                                                  (y - dim_info[0].off_y_in), (x - dim_info[0].off_x_in)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case reduce_avg: {
                                        temp_c +=
                                            snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu]", op[0].in_buffer.name, op[0].in_buffer.name, loop_idx, op_idx,
                                                     SIMPLE_INDEX(op[0].in_buffer, (a - dim_info[0].off_a_in), (z - dim_info[0].off_z_in),
                                                                  (y - dim_info[0].off_y_in), (x - dim_info[0].off_x_in)));
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case reduce_max: {
                                        TODO();
                                        break;
                                    }
                                    case reduce_min: {
                                        TODO();
                                        break;
                                    }
                                }
                                break;
                            }
                            case operation_move: {
                                ERROR("ERROR: Tried to compile move operation to OpenCL at index %lu\n", op_idx);
                            }
                        }
                    } else {
                        for(int64_t i = 1; i < op_num; i++) {
                            if(IS_PREFIX(op + i)) {
                                simple_op_print(op + i, 4, 0, "");
                                switch(op[i].unary_type) {
                                    case unary_exp: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "exp(");
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case unary_log: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "log(");
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case unary_sqrt: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "sqrt(");
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case unary_reciprocal: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "1/(");
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case unary_tanh: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "tanh(");
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    case unary_absolute: {
                                        temp_c += snprintf(temp_c, MAX_OP_SIZE, "fabs(");
                                        EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                        break;
                                    }
                                    default: {
                                        ERROR("This should not ever happen.\n");
                                    }
                                }
                            } else {
                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "(");
                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                            }
                        }
                        for(int64_t i = 1; i < op_num; i++) {
                            if(IS_PREFIX(op + i)) {
                                assert(op[i].type == operation_unary);
                                temp_c += snprintf(temp_c, MAX_OP_SIZE, ")");
                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                            } else {
                                switch(op[i].type) {
                                    case operation_unary: {
                                        switch(op[i].unary_type) {
                                            case unary_add: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "+%.16lf)", op[i].var_unary);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                                break;
                                            }
                                            case unary_subtract: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "-%.16lf)", op[i].var_unary);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                                break;
                                            }
                                            case unary_multiply: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "*%.16lf)", op[i].var_unary);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                                break;
                                            }
                                            case unary_divide: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "/%.16lf)", op[i].var_unary);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                                break;
                                            }
                                            case unary_square: {
                                                TODO();
                                                break;
                                            }
                                            case unary_max: {
                                                TODO();
                                                break;
                                            }
                                            case unary_min: {
                                                TODO();
                                                break;
                                            }
                                            case unary_set: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%.16lf)", op[i].var_unary);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                                break;
                                            }
                                            case unary_sign: {
                                                TODO();
                                                break;
                                            }
                                            default: {
                                                ERROR("???");
                                            }
                                        }
                                        break;
                                    }
                                    case operation_binary: {
                                        switch(op[i].binary_type) {
                                            case binary_add: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "+%s[%s%luoff%lu+%lu])", op[i].in_buffer.name, op[i].in_buffer.name,
                                                                   loop_idx, op_idx,
                                                                   SIMPLE_INDEX(op[i].in_buffer, (a - dim_info[i].off_a_in), (z - dim_info[i].off_z_in),
                                                                                (y - dim_info[i].off_y_in), (x - dim_info[i].off_x_in)));
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                                break;
                                            }
                                            case binary_subtract: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "-%s[%s%luoff%lu+%lu])", op[i].in_buffer.name, op[i].in_buffer.name,
                                                                   loop_idx, op_idx,
                                                                   SIMPLE_INDEX(op[i].in_buffer, (a - dim_info[i].off_a_in), (z - dim_info[i].off_z_in),
                                                                                (y - dim_info[i].off_y_in), (x - dim_info[i].off_x_in)));
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                                break;
                                            }
                                            case binary_multiply: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "*%s[%s%luoff%lu+%lu])", op[i].in_buffer.name, op[i].in_buffer.name,
                                                                   loop_idx, op_idx,
                                                                   SIMPLE_INDEX(op[i].in_buffer, (a - dim_info[i].off_a_in), (z - dim_info[i].off_z_in),
                                                                                (y - dim_info[i].off_y_in), (x - dim_info[i].off_x_in)));
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                                break;
                                            }
                                            case binary_divide: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "/%s[%s%luoff%lu+%lu])", op[i].in_buffer.name, op[i].in_buffer.name,
                                                                   loop_idx, op_idx,
                                                                   SIMPLE_INDEX(op[i].in_buffer, (a - dim_info[i].off_a_in), (z - dim_info[i].off_z_in),
                                                                                (y - dim_info[i].off_y_in), (x - dim_info[i].off_x_in)));
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                                break;
                                            }
                                            case binary_max: {
                                                TODO();
                                                break;
                                            }
                                            case binary_min: {
                                                TODO();
                                                break;
                                            }
                                            case binary_copy: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu+%lu])", op[i].in_buffer.name, op[i].in_buffer.name,
                                                                   loop_idx, op_idx,
                                                                   SIMPLE_INDEX(op[i].in_buffer, (a - dim_info[i].off_a_in), (z - dim_info[i].off_z_in),
                                                                                (y - dim_info[i].off_y_in), (x - dim_info[i].off_x_in)));
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                                break;
                                            }
                                            case binary_add_like: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "+%s[%s%luoff%lu])", op[i].in_buffer.name, op[i].in_buffer.name,
                                                                   loop_idx, op_idx);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                                break;
                                            }
                                            case binary_subtract_like: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "-%s[%s%luoff%lu])", op[i].in_buffer.name, op[i].in_buffer.name,
                                                                   loop_idx, op_idx);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                                break;
                                            }
                                            case binary_multiply_like: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "*%s[%s%luoff%lu])", op[i].in_buffer.name, op[i].in_buffer.name,
                                                                   loop_idx, op_idx);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                                break;
                                            }
                                            case binary_divide_like: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "/%s[%s%luoff%lu])", op[i].in_buffer.name, op[i].in_buffer.name,
                                                                   loop_idx, op_idx);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                                break;
                                            }
                                            case binary_max_like: {
                                                TODO();
                                                break;
                                            }
                                            case binary_min_like: {
                                                TODO();
                                                break;
                                            }
                                            case binary_copy_like: {
                                                temp_c += snprintf(temp_c, MAX_OP_SIZE, "%s[%s%luoff%lu])", op[i].in_buffer.name, op[i].in_buffer.name,
                                                                   loop_idx, op_idx);
                                                EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                                                break;
                                            }
                                        }
                                        break;
                                    }
                                    case operation_reduce: {
                                        ERROR("ERROR: Tried to inline reduce operation!\n");
                                        break;
                                    }
                                    case operation_move: {
                                        ERROR("ERROR: Tried to inline move operation!\n");
                                    }
                                }
                            }
                        }
                    }
                    temp_c += snprintf(temp_c, MAX_OP_SIZE, ";\n");
                    EXPAND_SOURCE_IF_NEEDED(temp_c, temp, temp_cap);
                    while(*source_cap - (*curr - *source) - (temp_c - temp) <= MAX_OP_SIZE) {
                        *source_cap *= 2;
                        offset = *curr - *source;
                        *source = reallocarray(*source, *source_cap, sizeof(char));
                        assert(*source);
                        *curr = *source + offset;
                    }
                    *curr += snprintf(*curr, temp_cap, "%s", temp);
                    EXPAND_SOURCE_IF_NEEDED(*curr, *source, *source_cap);
                    memset(temp, 0, temp_cap);
                    temp_c = temp;
                }
            }
        }
    }

    free(temp);
}
int64_t kernel_counter = 0;
static cl_kernel_t compile_loop_to_cl(const char *filename, compile_loop_t *compile, int64_t global_size, int64_t local_size) {
    assert(filename);
    assert(compile);
    assert(global_size);
    /* TODO: Support `local_size > 1`. */
    assert(local_size == 1);

    char *func_name = calloc(1 + log10(kernel_counter + 1) + 3, sizeof(char));
    assert(func_name);
    snprintf(func_name, 1 + log10(kernel_counter + 1) + 3, "k%lu", kernel_counter);
    kernel_counter++;
    int64_t leftover_loops = compile->loop_num % global_size;
    int64_t assigned_loops = (compile->loop_num - leftover_loops) / global_size;
    int64_t needed_loops;
    if(leftover_loops) {
        needed_loops = assigned_loops + 1;
    } else {
        needed_loops = assigned_loops;
    }

    int64_t source_cap = INITIAL_SOURCE_SIZE;
    char *source = calloc(source_cap, sizeof(char));
    assert(source);
    char *curr = source;
    int64_t offset;

    int64_t arg_num = 0;
    char **args = NULL;
    int64_t found;
    for(int64_t i = 0; i < compile->loop_len; i++) {
        if(compile->op_num[i] == 1) {
            found = 0;
            for(int64_t j = 0; j < arg_num; j++) {
                if(!strncmp(compile->op[i][0].out_buffer.name, args[j], BUFFER_NAME_SIZE)) {
                    found = 1;
                    break;
                }
            }
            if(!found) {
                arg_num++;
                args = reallocarray(args, arg_num, sizeof(char *));
                assert(args);
                args[arg_num - 1] = calloc(BUFFER_NAME_SIZE + 1, sizeof(char));
                assert(args[arg_num - 1]);
                strncpy(args[arg_num - 1], compile->op[i][0].out_buffer.name, BUFFER_NAME_SIZE);
            }
            if(compile->op[i][0].type != operation_unary) {
                found = 0;
                for(int64_t j = 0; j < arg_num; j++) {
                    if(!strncmp(compile->op[i][0].in_buffer.name, args[j], BUFFER_NAME_SIZE)) {
                        found = 1;
                        break;
                    }
                }
                if(!found) {
                    arg_num++;
                    args = reallocarray(args, arg_num, sizeof(char *));
                    assert(args);
                    args[arg_num - 1] = calloc(BUFFER_NAME_SIZE + 1, sizeof(char));
                    assert(args[arg_num - 1]);
                    strncpy(args[arg_num - 1], compile->op[i][0].in_buffer.name, BUFFER_NAME_SIZE);
                }
            }
        } else {
            for(int64_t j = 0; j < compile->op_num[i]; j++) {
                if(j) {
                    found = 0;
                    for(int64_t k = 0; k < arg_num; k++) {
                        if(!strncmp(compile->op[i][j].in_buffer.name, args[k], BUFFER_NAME_SIZE)) {
                            found = 1;
                            break;
                        }
                    }
                    if(!found) {
                        arg_num++;
                        args = reallocarray(args, arg_num, sizeof(char *));
                        assert(args);
                        args[arg_num - 1] = calloc(BUFFER_NAME_SIZE + 1, sizeof(char));
                        assert(args[arg_num - 1]);
                        strncpy(args[arg_num - 1], compile->op[i][j].in_buffer.name, BUFFER_NAME_SIZE);
                    }
                } else {
                    found = 0;
                    for(int64_t k = 0; k < arg_num; k++) {
                        if(!strncmp(compile->op[i][j].out_buffer.name, args[k], BUFFER_NAME_SIZE)) {
                            found = 1;
                            break;
                        }
                    }
                    if(!found) {
                        arg_num++;
                        args = reallocarray(args, arg_num, sizeof(char *));
                        assert(args);
                        args[arg_num - 1] = calloc(BUFFER_NAME_SIZE + 1, sizeof(char));
                        assert(args[arg_num - 1]);
                        strncpy(args[arg_num - 1], compile->op[i][j].out_buffer.name, BUFFER_NAME_SIZE);
                    }
                }
            }
        }
    }
    int64_t gid_cap = INITIAL_CAP;
    int64_t gid_len;
    char **gid = calloc(gid_cap, sizeof(char *));
    curr += snprintf(curr, MAX_OP_SIZE, "int gid0 = get_global_id(0);\nint id = gid0;\n");
    EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap);
    for(int64_t i = 0; i < needed_loops; i++) {
        gid_len = 0;
        if(i) {
            curr += snprintf(curr, MAX_OP_SIZE, "id += %lu;\n", assigned_loops);
            EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap);
        }
        if(i == assigned_loops) {
            curr += snprintf(curr, MAX_OP_SIZE, "if(gid0 < %lu) {\n", leftover_loops);
            EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap);
        }
        for(int64_t j = 0; j < compile->loop_len; j++) {
            if(compile->op_num[j] == 1) {
                if(compile->op[j]->type == operation_unary) {
                    curr +=
                        snprintf(curr, MAX_OP_SIZE,
                                 "int %s%luoff%lu=(((id%%%lu)/%lu+%lu)*%lu)+(((id%%%lu)/%lu+%lu)*%lu)+(((id%%%lu)/%lu+%lu)*%lu)+(((id%%%lu)/%lu+%lu)*%lu);\n",
                                 compile->op[j][0].out_buffer.name, i, j, compile->dim_info[j][0].res_a_out, compile->dim_info[j][0].wai_a_out,
                                 compile->dim_info[j][0].str_a_out, compile->dim_info[j][0].off_a_out, compile->dim_info[j][0].res_z_out,
                                 compile->dim_info[j][0].wai_z_out, compile->dim_info[j][0].str_z_out, compile->dim_info[j][0].off_z_out,
                                 compile->dim_info[j][0].res_y_out, compile->dim_info[j][0].wai_y_out, compile->dim_info[j][0].str_y_out,
                                 compile->dim_info[j][0].off_y_out, compile->dim_info[j][0].res_x_out, compile->dim_info[j][0].wai_x_out,
                                 compile->dim_info[j][0].off_x_out, compile->dim_info[j][0].str_x_out);
                    EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap);
                } else {
                    curr +=
                        snprintf(curr, MAX_OP_SIZE,
                                 "int %s%luoff%lu=(((id%%%lu)/%lu+%lu)*%lu)+(((id%%%lu)/%lu+%lu)*%lu)+(((id%%%lu)/%lu+%lu)*%lu)+(((id%%%lu)/%lu+%lu)*%lu);\n",
                                 compile->op[j][0].out_buffer.name, i, j, compile->dim_info[j][0].res_a_out, compile->dim_info[j][0].wai_a_out,
                                 compile->dim_info[j][0].str_a_out, compile->dim_info[j][0].off_a_out, compile->dim_info[j][0].res_z_out,
                                 compile->dim_info[j][0].wai_z_out, compile->dim_info[j][0].str_z_out, compile->dim_info[j][0].off_z_out,
                                 compile->dim_info[j][0].res_y_out, compile->dim_info[j][0].wai_y_out, compile->dim_info[j][0].str_y_out,
                                 compile->dim_info[j][0].off_y_out, compile->dim_info[j][0].res_x_out, compile->dim_info[j][0].wai_x_out,
                                 compile->dim_info[j][0].off_x_out, compile->dim_info[j][0].str_x_out);
                    EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap);
                    curr += snprintf(
                        curr, MAX_OP_SIZE,
                        "int %s%luoff%lu=(((id%%%lu)/%lu+%lu)*%lu)+(((id%%%lu)/%lu+%lu)*%lu)+(((id%%%lu)/%lu+%lu)*%lu)+(((id%%%lu)/%lu+%lu)*%lu);\n",
                        compile->op[j][0].in_buffer.name, i, j, compile->dim_info[j][0].res_a_in, compile->dim_info[j][0].wai_a_in,
                        compile->dim_info[j][0].str_a_in, compile->dim_info[j][0].off_a_in, compile->dim_info[j][0].res_z_in, compile->dim_info[j][0].wai_z_in,
                        compile->dim_info[j][0].str_z_in, compile->dim_info[j][0].off_z_in, compile->dim_info[j][0].res_y_in, compile->dim_info[j][0].wai_y_in,
                        compile->dim_info[j][0].str_y_in, compile->dim_info[j][0].off_y_in, compile->dim_info[j][0].res_x_in, compile->dim_info[j][0].wai_x_in,
                        compile->dim_info[j][0].off_x_in, compile->dim_info[j][0].str_x_in);
                    EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap);
                }
            } else {
                for(int64_t k = 0; k < compile->op_num[j]; k++) {
                    if(k) {
                        curr += snprintf(
                            curr, MAX_OP_SIZE,
                            "int %s%luoff%lu=(((id%%%lu)/%lu+%lu)*%lu)+(((id%%%lu)/%lu+%lu)*%lu)+(((id%%%lu)/%lu+%lu)*%lu)+(((id%%%lu)/%lu+%lu)*%lu);\n",
                            compile->op[j][k].in_buffer.name, i, j, compile->dim_info[j][k].res_a_in, compile->dim_info[j][k].wai_a_in,
                            compile->dim_info[j][k].str_a_in, compile->dim_info[j][k].off_a_in, compile->dim_info[j][k].res_z_in,
                            compile->dim_info[j][k].wai_z_in, compile->dim_info[j][k].str_z_in, compile->dim_info[j][k].off_z_in,
                            compile->dim_info[j][k].res_y_in, compile->dim_info[j][k].wai_y_in, compile->dim_info[j][k].str_y_in,
                            compile->dim_info[j][k].off_y_in, compile->dim_info[j][k].res_x_in, compile->dim_info[j][k].wai_x_in,
                            compile->dim_info[j][k].off_x_in, compile->dim_info[j][k].str_x_in);
                        EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap);
                    } else {
                        curr += snprintf(
                            curr, MAX_OP_SIZE,
                            "int %s%luoff%lu=(((id%%%lu)/%lu+%lu)*%lu)+(((id%%%lu)/%lu+%lu)*%lu)+(((id%%%lu)/%lu+%lu)*%lu)+(((id%%%lu)/%lu+%lu)*%lu);\n",
                            compile->op[j][k].out_buffer.name, i, j, compile->dim_info[j][k].res_a_out, compile->dim_info[j][k].wai_a_out,
                            compile->dim_info[j][k].str_a_out, compile->dim_info[j][k].off_a_out, compile->dim_info[j][k].res_z_out,
                            compile->dim_info[j][k].wai_z_out, compile->dim_info[j][k].str_z_out, compile->dim_info[j][k].off_z_out,
                            compile->dim_info[j][k].res_y_out, compile->dim_info[j][k].wai_y_out, compile->dim_info[j][k].str_y_out,
                            compile->dim_info[j][k].off_y_out, compile->dim_info[j][k].res_x_out, compile->dim_info[j][k].wai_x_out,
                            compile->dim_info[j][k].off_x_out, compile->dim_info[j][k].str_x_out);
                        EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap);
                    }
                }
            }
            compile_single_op_to_cl(compile->op[j], compile->dim_info[j], compile->op_num[j], i, j, &source, &curr, &source_cap);
        }
        for(int64_t i = 0; i < gid_len; i++) { free(gid[i]); }
        if(i == assigned_loops) {
            curr += snprintf(curr, MAX_OP_SIZE, "}\n");
            EXPAND_SOURCE_IF_NEEDED(curr, source, source_cap);
        }
    }

    assert(arg_num != 0);
    /* NOTE: That `+ 3` is pure magic. I have no clue where it comes from. */
    int64_t kernel_size = strlen("__kernel void ") + strlen(func_name) + (strlen("__global double *") + BUFFER_NAME_SIZE) * arg_num +
                          strlen(", ") * (arg_num - 1) + strlen(") {\n") + (curr - source) + strlen("}\n") + 3;
    char *kernel_source = calloc(kernel_size, sizeof(char));
    assert(kernel_source);
    char *kernel_i = kernel_source;
    kernel_i += snprintf(kernel_i, 1 + log10(kernel_counter + 1) + 3 + strnlen("__kernel void(", 20), "__kernel void %s(", func_name);
    for(int64_t i = 0; i < arg_num; i++) {
        if(i != arg_num - 1) {
            kernel_i += snprintf(kernel_i, 1 + BUFFER_NAME_SIZE + strnlen("__global double *, ", 30), "__global double *%s, ", args[i]);
        } else {
            kernel_i += snprintf(kernel_i, 1 + BUFFER_NAME_SIZE + strnlen("__global double *) {\n", 30), "__global double *%s) {\n", args[i]);
        }
    }
    /* This one is very sus. Extremely sus. Why in the world do I need to do the `+ 1` here? */
    kernel_i += snprintf(kernel_i, curr - source + 1, "%s", source);
    kernel_i += snprintf(kernel_i, 3, "}\n");

    FILE *f = fopen(filename, "a");
    assert(f);
    fwrite(kernel_source, sizeof(char), kernel_i - kernel_source, f);
    fclose(f);

    cl_kernel_t kernel = {
        .arg_num = arg_num,
        .args = calloc(arg_num, sizeof(char *)),
        .name = strndup(func_name, strlen(func_name)),
        .local_size = local_size,
        .global_size = global_size,
    };
    assert(kernel.args);
    for(int64_t i = 0; i < arg_num; i++) {
        kernel.args[i] = strndup(args[i], BUFFER_NAME_SIZE + 1);
        free(args[i]);
    }
    free(args);
    free(source);
    free(kernel_source);
    free(func_name);
    free(gid);
    return kernel;
}
int compile_linearized_to_cl(cl_program_t *program, const char *filename, linearized_t *linearized) {
    if(!linearized->op_count) { return 1; }
    simple_loop_t simple = {0};
    int64_t global_size = 4;
    int64_t local_size = 1;
    compile_loop_t compile;
    /* Clears file. */
    FILE *f = fopen(filename, "w");
    if(!f) { return 1; }
    fclose(f);
    int64_t i = 0;
    cl_kernel_t kernel;
    while(i < linearized->op_count) {
        i += simple_loop_from_linearized_index(&simple, linearized, i);
        compile = compile_loop_alloc(&simple, OPTIMIZE_INLINE);
        // compile = compile_loop_alloc(&simple, OPTIMIZE_NONE);
        compile_loop_print(&compile, 4, 0, "");
        kernel = compile_loop_to_cl(filename, &compile, global_size, local_size);
        program->kernel_num++;
        program->kernel = reallocarray(program->kernel, program->kernel_num, sizeof(cl_kernel_t));
        assert(program->kernel);
        program->kernel[program->kernel_num - 1] = kernel;
        compile_loop_free(&compile);
    }

    simple_loop_free(&simple);
    return 0;
}
void cl_program_free(cl_program_t *program) {
    for(int64_t i = 0; i < program->kernel_num; i++) { cl_kernel_free(&program->kernel[i]); }
    free(program->kernel);
}
