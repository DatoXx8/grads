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

#define SIMPLE_INDEX(simple, a, z, y, x)                                                                                                                       \
    ((simple).a_stride * (a) + (simple).z_stride * (z) + (simple).y_stride * (y) + (simple).x_stride * (x) + (simple).offset)
#define SIMPLE_INDEX_(simple, a, z, y, x)                                                                                                                      \
    ((simple)->a_stride * (a) + (simple)->z_stride * (z) + (simple)->y_stride * (y) + (simple)->x_stride * (x) + (simple)->offset)
static void compile_loop_free(compile_loop_t *compile_loop) {
    for(uint64_t i = 0; i < compile_loop->loop_number; i++) { free(compile_loop->loop_instance[i]); }
    free(compile_loop->loop_instance);
    free(compile_loop->per_dim_off_a);
    free(compile_loop->per_dim_off_z);
    free(compile_loop->per_dim_off_y);
    free(compile_loop->per_dim_off_x);
    free(compile_loop->per_dim_str_a);
    free(compile_loop->per_dim_str_z);
    free(compile_loop->per_dim_str_y);
    free(compile_loop->per_dim_str_x);
    free(compile_loop->per_dim_wait_a);
    free(compile_loop->per_dim_wait_z);
    free(compile_loop->per_dim_wait_y);
    free(compile_loop->per_dim_wait_x);
    free(compile_loop->per_dim_reset_a);
    free(compile_loop->per_dim_reset_z);
    free(compile_loop->per_dim_reset_y);
    free(compile_loop->per_dim_reset_x);
}
/* TODO: Check if `compile_loop` has already been configured, by checking pointers for NULL. */
static void compile_loop_configure(compile_loop_t *compile_loop, simple_op_t **simple_op, uint64_t loop_length, uint64_t loop_number) {
    /* Can not currently re-use compile loops, but it should be a pretty quick fix. */
    compile_loop->loop_length = loop_length;
    compile_loop->loop_number = loop_number;
    compile_loop->loop_instance = calloc(loop_number, sizeof(simple_op_t *));
    for(uint64_t i = 0; i < loop_number; i++) {
        compile_loop->loop_instance[i] = calloc(loop_length, sizeof(simple_op_t));
        for(uint64_t j = 0; j < loop_length; j++) { compile_loop->loop_instance[i][j] = simple_op[i][j]; }
    }
    /* TODO: this one. */
    /* Gather per_dim_off, per_dim_str, per_dim_loop_reset, per_dim_loop_wait. */
    compile_loop->per_dim_off_a = calloc(loop_length * 2, sizeof(uint64_t *));
    compile_loop->per_dim_off_z = calloc(loop_length * 2, sizeof(uint64_t *));
    compile_loop->per_dim_off_y = calloc(loop_length * 2, sizeof(uint64_t *));
    compile_loop->per_dim_off_x = calloc(loop_length * 2, sizeof(uint64_t *));
    compile_loop->per_dim_str_a = calloc(loop_length * 2, sizeof(uint64_t *));
    compile_loop->per_dim_str_z = calloc(loop_length * 2, sizeof(uint64_t *));
    compile_loop->per_dim_str_y = calloc(loop_length * 2, sizeof(uint64_t *));
    compile_loop->per_dim_str_x = calloc(loop_length * 2, sizeof(uint64_t *));
    compile_loop->per_dim_reset_a = calloc(loop_length * 2, sizeof(uint64_t *));
    compile_loop->per_dim_reset_z = calloc(loop_length * 2, sizeof(uint64_t *));
    compile_loop->per_dim_reset_y = calloc(loop_length * 2, sizeof(uint64_t *));
    compile_loop->per_dim_reset_x = calloc(loop_length * 2, sizeof(uint64_t *));
    compile_loop->per_dim_wait_a = calloc(loop_length * 2, sizeof(uint64_t *));
    compile_loop->per_dim_wait_z = calloc(loop_length * 2, sizeof(uint64_t *));
    compile_loop->per_dim_wait_y = calloc(loop_length * 2, sizeof(uint64_t *));
    compile_loop->per_dim_wait_x = calloc(loop_length * 2, sizeof(uint64_t *));
    /* FIX: Currently we assume that the initial op necessarily has the lowest indices, this should *really* be fixed to have some sorting. This would also fix
     * potentially negative strides. */
    for(uint64_t i = 0; i < loop_length; i++) {
        compile_loop->per_dim_off_a[2 * i] = compile_loop->loop_instance[0][i].out_buffer.a_offset;
        compile_loop->per_dim_off_z[2 * i] = compile_loop->loop_instance[0][i].out_buffer.z_offset;
        compile_loop->per_dim_off_y[2 * i] = compile_loop->loop_instance[0][i].out_buffer.y_offset;
        compile_loop->per_dim_off_x[2 * i] = compile_loop->loop_instance[0][i].out_buffer.x_offset;
        if(compile_loop->loop_instance[0][i].type != operation_unary) {
            compile_loop->per_dim_off_a[2 * i + 1] = compile_loop->loop_instance[0][i].in_buffer.a_offset;
            compile_loop->per_dim_off_z[2 * i + 1] = compile_loop->loop_instance[0][i].in_buffer.z_offset;
            compile_loop->per_dim_off_y[2 * i + 1] = compile_loop->loop_instance[0][i].in_buffer.y_offset;
            compile_loop->per_dim_off_x[2 * i + 1] = compile_loop->loop_instance[0][i].in_buffer.x_offset;
        }
    }
    uint64_t found_a_o, found_a_i;
    uint64_t found_z_o, found_z_i;
    uint64_t found_y_o, found_y_i;
    uint64_t found_x_o, found_x_i;
    /* Order flipped intentionally, since it's per op and not really per loop. */
    /* FIX: These are all hacks and most likely don't work. */
    /* TODO: Do these per dimension instead of a huge loop, so that it's more readable and I can unify all the `found_x_y`s together. */
    for(uint64_t i = 0; i < loop_length; i++) {
        found_a_o = 0;
        found_z_o = 0;
        found_y_o = 0;
        found_x_o = 0;
        found_a_i = 0;
        found_z_i = 0;
        found_y_i = 0;
        found_x_i = 0;
        for(uint64_t j = 1; j < loop_number; j++) {
            if((!found_a_o) && compile_loop->loop_instance[j][i].out_buffer.a_offset != compile_loop->per_dim_off_a[2 * i]) {
                compile_loop->per_dim_str_a[2 * i] = compile_loop->loop_instance[j][i].out_buffer.a_offset - compile_loop->per_dim_off_a[2 * i];
                compile_loop->per_dim_wait_a[2 * i] = j;
                found_a_o = 1;
            }
            if((!found_z_o) && compile_loop->loop_instance[j][i].out_buffer.z_offset != compile_loop->per_dim_off_z[2 * i]) {
                compile_loop->per_dim_str_z[2 * i] = compile_loop->loop_instance[j][i].out_buffer.z_offset - compile_loop->per_dim_off_z[2 * i];
                compile_loop->per_dim_wait_z[2 * i] = j;
                found_z_o = 1;
            }
            if((!found_y_o) && compile_loop->loop_instance[j][i].out_buffer.y_offset != compile_loop->per_dim_off_y[2 * i]) {
                compile_loop->per_dim_str_y[2 * i] = compile_loop->loop_instance[j][i].out_buffer.y_offset - compile_loop->per_dim_off_y[2 * i];
                compile_loop->per_dim_wait_y[2 * i] = j;
                found_y_o = 1;
            }
            if((!found_x_o) && compile_loop->loop_instance[j][i].out_buffer.x_offset != compile_loop->per_dim_off_x[2 * i]) {
                compile_loop->per_dim_str_x[2 * i] = compile_loop->loop_instance[j][i].out_buffer.x_offset - compile_loop->per_dim_off_x[2 * i];
                compile_loop->per_dim_wait_x[2 * i] = j;
                found_x_o = 1;
            }
            if(compile_loop->loop_instance[0][i].type != operation_unary) {
                if((!found_a_i) && compile_loop->loop_instance[j][i].in_buffer.a_offset != compile_loop->per_dim_off_a[2 * i + 1]) {
                    compile_loop->per_dim_str_a[2 * i + 1] = compile_loop->loop_instance[j][i].in_buffer.a_offset - compile_loop->per_dim_off_a[2 * i + 1];
                    compile_loop->per_dim_wait_a[2 * i + 1] = j;
                    found_a_i = 1;
                }
                if((!found_z_i) && compile_loop->loop_instance[j][i].in_buffer.z_offset != compile_loop->per_dim_off_z[2 * i + 1]) {
                    compile_loop->per_dim_str_z[2 * i + 1] = compile_loop->loop_instance[j][i].in_buffer.z_offset - compile_loop->per_dim_off_z[2 * i + 1];
                    compile_loop->per_dim_wait_z[2 * i + 1] = j;
                    found_z_i = 1;
                }
                if((!found_y_i) && compile_loop->loop_instance[j][i].in_buffer.y_offset != compile_loop->per_dim_off_y[2 * i + 1]) {
                    compile_loop->per_dim_str_y[2 * i + 1] = compile_loop->loop_instance[j][i].in_buffer.y_offset - compile_loop->per_dim_off_y[2 * i + 1];
                    compile_loop->per_dim_wait_y[2 * i + 1] = j;
                    found_y_i = 1;
                }
                if((!found_x_i) && compile_loop->loop_instance[j][i].in_buffer.x_offset != compile_loop->per_dim_off_x[2 * i + 1]) {
                    compile_loop->per_dim_str_x[2 * i + 1] = compile_loop->loop_instance[j][i].in_buffer.x_offset - compile_loop->per_dim_off_x[2 * i + 1];
                    compile_loop->per_dim_wait_x[2 * i + 1] = j;
                    found_x_i = 1;
                }
            }
        }
        if(!found_a_o) {
            compile_loop->per_dim_str_a[2 * i] = 0;
            compile_loop->per_dim_wait_a[2 * i] = loop_number;
        }
        if(!found_z_o) {
            compile_loop->per_dim_str_z[2 * i] = 0;
            compile_loop->per_dim_wait_z[2 * i] = loop_number;
        }
        if(!found_y_o) {
            compile_loop->per_dim_str_y[2 * i] = 0;
            compile_loop->per_dim_wait_y[2 * i] = loop_number;
        }
        if(!found_x_o) {
            compile_loop->per_dim_str_x[2 * i] = 0;
            compile_loop->per_dim_wait_x[2 * i] = loop_number;
        }
        if(compile_loop->loop_instance[0][i].type != operation_unary) {
            if(!found_a_i) {
                compile_loop->per_dim_str_a[2 * i + 1] = 0;
                compile_loop->per_dim_wait_a[2 * i + 1] = loop_number;
            }
            if(!found_z_i) {
                compile_loop->per_dim_str_z[2 * i + 1] = 0;
                compile_loop->per_dim_wait_z[2 * i + 1] = loop_number;
            }
            if(!found_y_i) {
                compile_loop->per_dim_str_y[2 * i + 1] = 0;
                compile_loop->per_dim_wait_y[2 * i + 1] = loop_number;
            }
            if(!found_x_i) {
                compile_loop->per_dim_str_x[2 * i + 1] = 0;
                compile_loop->per_dim_wait_x[2 * i + 1] = loop_number;
            }
        }
    }
    uint64_t left_a_o = 0;
    uint64_t left_z_o = 0;
    uint64_t left_y_o = 0;
    uint64_t left_x_o = 0;
    uint64_t left_a_i = 0;
    uint64_t left_z_i = 0;
    uint64_t left_y_i = 0;
    uint64_t left_x_i = 0;
    for(uint64_t i = 0; i < loop_length; i++) {
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
        for(uint64_t j = 1; j < loop_number; j++) {
            if((!left_a_o) && (!found_a_o) && compile_loop->loop_instance[j][i].out_buffer.a_offset != compile_loop->per_dim_off_a[2 * i]) { left_a_o = 1; }
            if(left_a_o && (!found_a_o) && compile_loop->loop_instance[j][i].out_buffer.a_offset == compile_loop->per_dim_off_a[2 * i]) {
                compile_loop->per_dim_reset_a[2 * i] = j;
                found_a_o = 1;
            }
            if((!left_z_o) && (!found_z_o) && compile_loop->loop_instance[j][i].out_buffer.z_offset != compile_loop->per_dim_off_z[2 * i]) { left_z_o = 1; }
            if(left_z_o && (!found_z_o) && compile_loop->loop_instance[j][i].out_buffer.z_offset == compile_loop->per_dim_off_z[2 * i]) {
                compile_loop->per_dim_reset_z[2 * i] = j;
                found_z_o = 1;
            }
            if((!left_y_o) && (!found_y_o) && compile_loop->loop_instance[j][i].out_buffer.y_offset != compile_loop->per_dim_off_y[2 * i]) { left_y_o = 1; }
            if(left_y_o && (!found_y_o) && compile_loop->loop_instance[j][i].out_buffer.y_offset == compile_loop->per_dim_off_y[2 * i]) {
                compile_loop->per_dim_reset_y[2 * i] = j;
                found_y_o = 1;
            }
            if((!left_x_o) && (!found_x_o) && compile_loop->loop_instance[j][i].out_buffer.x_offset != compile_loop->per_dim_off_x[2 * i]) { left_x_o = 1; }
            if(left_x_o && (!found_x_o) && compile_loop->loop_instance[j][i].out_buffer.x_offset == compile_loop->per_dim_off_x[2 * i]) {
                compile_loop->per_dim_reset_x[2 * i] = j;
                found_x_o = 1;
            }
            if(compile_loop->loop_instance[0][i].type != operation_unary) {
                if((!left_a_i) && (!found_a_i) && compile_loop->loop_instance[j][i].in_buffer.a_offset != compile_loop->per_dim_off_a[2 * i + 1]) {
                    left_a_i = 1;
                }
                if(left_a_i && (!found_a_i) && compile_loop->loop_instance[j][i].in_buffer.a_offset == compile_loop->per_dim_off_a[2 * i + 1]) {
                    compile_loop->per_dim_reset_a[2 * i + 1] = j;
                    found_a_i = 1;
                }
                if((!left_z_i) && (!found_z_i) && compile_loop->loop_instance[j][i].in_buffer.z_offset != compile_loop->per_dim_off_z[2 * i + 1]) {
                    left_z_i = 1;
                }
                if(left_z_i && (!found_z_i) && compile_loop->loop_instance[j][i].in_buffer.z_offset == compile_loop->per_dim_off_z[2 * i + 1]) {
                    compile_loop->per_dim_reset_z[2 * i + 1] = j;
                    found_z_i = 1;
                }
                if((!left_y_i) && (!found_y_i) && compile_loop->loop_instance[j][i].in_buffer.y_offset != compile_loop->per_dim_off_y[2 * i + 1]) {
                    left_y_i = 1;
                }
                if(left_y_i && (!found_y_i) && compile_loop->loop_instance[j][i].in_buffer.y_offset == compile_loop->per_dim_off_y[2 * i + 1]) {
                    compile_loop->per_dim_reset_y[2 * i + 1] = j;
                    found_y_i = 1;
                }
                if((!left_x_i) && (!found_x_i) && compile_loop->loop_instance[j][i].in_buffer.x_offset != compile_loop->per_dim_off_x[2 * i + 1]) {
                    left_x_i = 1;
                }
                if(left_x_i && (!found_x_i) && compile_loop->loop_instance[j][i].in_buffer.x_offset == compile_loop->per_dim_off_x[2 * i + 1]) {
                    compile_loop->per_dim_reset_x[2 * i + 1] = j;
                    found_x_i = 1;
                }
            }
        }
        if(!found_a_o) { compile_loop->per_dim_reset_a[2 * i] = loop_number; }
        if(!found_z_o) { compile_loop->per_dim_reset_z[2 * i] = loop_number; }
        if(!found_y_o) { compile_loop->per_dim_reset_y[2 * i] = loop_number; }
        if(!found_x_o) { compile_loop->per_dim_reset_x[2 * i] = loop_number; }
        if(compile_loop->loop_instance[0][i].type != operation_unary) {
            if(!found_a_i) { compile_loop->per_dim_reset_a[2 * i + 1] = loop_number; }
            if(!found_z_i) { compile_loop->per_dim_reset_z[2 * i + 1] = loop_number; }
            if(!found_y_i) { compile_loop->per_dim_reset_y[2 * i + 1] = loop_number; }
            if(!found_x_i) { compile_loop->per_dim_reset_x[2 * i + 1] = loop_number; }
        }
    }
}
static void compile_loop_print(compile_loop_t *compile_loop, int padding, int offset, const char *name) {
    if(!strncmp(name, "", 1)) {
        printf("%*scompile loop\n", offset, "");
    } else {
        printf("%*s %s\n", offset, "", name);
    }
    for(uint64_t i = 0; i < compile_loop->loop_number; i++) {
        if(i) { printf("\n"); }
        for(uint64_t j = 0; j < compile_loop->loop_length; j++) {
            printf("%*s[%lu, %lu] ", offset + padding, "", i, j);
            simple_op_print(&compile_loop->loop_instance[i][j], 0, 0, "");
        }
    }
    printf("off\n");
    for(uint64_t i = 0; i < compile_loop->loop_length; i++) {
        if(compile_loop->loop_instance[0][i].type == operation_unary) {
            printf("{%lu, %lu, %lu, %lu}\n", compile_loop->per_dim_off_a[2 * i], compile_loop->per_dim_off_z[2 * i], compile_loop->per_dim_off_y[2 * i],
                   compile_loop->per_dim_off_x[2 * i]);
        } else {
            printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", compile_loop->per_dim_off_a[2 * i], compile_loop->per_dim_off_z[2 * i],
                   compile_loop->per_dim_off_y[2 * i], compile_loop->per_dim_off_x[2 * i], compile_loop->per_dim_off_a[2 * i + 1],
                   compile_loop->per_dim_off_z[2 * i + 1], compile_loop->per_dim_off_y[2 * i + 1], compile_loop->per_dim_off_x[2 * i + 1]);
        }
    }
    printf("str\n");
    for(uint64_t i = 0; i < compile_loop->loop_length; i++) {
        if(compile_loop->loop_instance[0][i].type == operation_unary) {
            printf("{%lu, %lu, %lu, %lu}\n", compile_loop->per_dim_str_a[2 * i], compile_loop->per_dim_str_z[2 * i], compile_loop->per_dim_str_y[2 * i],
                   compile_loop->per_dim_str_x[2 * i]);
        } else {
            printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", compile_loop->per_dim_str_a[2 * i], compile_loop->per_dim_str_z[2 * i],
                   compile_loop->per_dim_str_y[2 * i], compile_loop->per_dim_str_x[2 * i], compile_loop->per_dim_str_a[2 * i + 1],
                   compile_loop->per_dim_str_z[2 * i + 1], compile_loop->per_dim_str_y[2 * i + 1], compile_loop->per_dim_str_x[2 * i + 1]);
        }
    }
    printf("res\n");
    for(uint64_t i = 0; i < compile_loop->loop_length; i++) {
        if(compile_loop->loop_instance[0][i].type == operation_unary) {
            printf("{%lu, %lu, %lu, %lu}\n", compile_loop->per_dim_reset_a[2 * i], compile_loop->per_dim_reset_z[2 * i], compile_loop->per_dim_reset_y[2 * i],
                   compile_loop->per_dim_reset_x[2 * i]);
        } else {
            printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", compile_loop->per_dim_reset_a[2 * i], compile_loop->per_dim_reset_z[2 * i],
                   compile_loop->per_dim_reset_y[2 * i], compile_loop->per_dim_reset_x[2 * i], compile_loop->per_dim_reset_a[2 * i + 1],
                   compile_loop->per_dim_reset_z[2 * i + 1], compile_loop->per_dim_reset_y[2 * i + 1], compile_loop->per_dim_reset_x[2 * i + 1]);
        }
    }
    printf("wat\n");
    for(uint64_t i = 0; i < compile_loop->loop_length; i++) {
        if(compile_loop->loop_instance[0][i].type == operation_unary) {
            printf("{%lu, %lu, %lu, %lu}\n", compile_loop->per_dim_wait_a[2 * i], compile_loop->per_dim_wait_z[2 * i], compile_loop->per_dim_wait_y[2 * i],
                   compile_loop->per_dim_wait_x[2 * i]);
        } else {
            printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", compile_loop->per_dim_wait_a[2 * i], compile_loop->per_dim_wait_z[2 * i],
                   compile_loop->per_dim_wait_y[2 * i], compile_loop->per_dim_wait_x[2 * i], compile_loop->per_dim_wait_a[2 * i + 1],
                   compile_loop->per_dim_wait_z[2 * i + 1], compile_loop->per_dim_wait_y[2 * i + 1], compile_loop->per_dim_wait_x[2 * i + 1]);
        }
    }
}
/* Has to have the same input and output tensors, with the same shape and be the same op type. Offsets however should be irrelevant. */
static ALWAYS_INLINE bool compile_loop_simple_op_equal(simple_op_t *starting_op, simple_op_t *compared_op) {
    /* NOTE: This comparison is probably not needed technically. */
    if(starting_op->type != compared_op->type) { return false; }
    /* NOTE: Always checking every single one cuz it probably takes longer to go to the different cases. */
    if(starting_op->unary_type != compared_op->unary_type) { return false; }
    if(starting_op->binary_type != compared_op->binary_type) { return false; }
    if(starting_op->reduce_type != compared_op->reduce_type) { return false; }

    if(strncmp(starting_op->out_buffer.name, compared_op->out_buffer.name, BUFFER_NAME_SIZE)) { return false; }
    if(starting_op->out_buffer.a_size != compared_op->out_buffer.a_size) { return false; }
    if(starting_op->out_buffer.z_size != compared_op->out_buffer.z_size) { return false; }
    if(starting_op->out_buffer.y_size != compared_op->out_buffer.y_size) { return false; }
    if(starting_op->out_buffer.x_size != compared_op->out_buffer.x_size) { return false; }
    if(starting_op->type != operation_unary) {
        if(strncmp(starting_op->in_buffer.name, compared_op->in_buffer.name, BUFFER_NAME_SIZE)) { return false; }
        if(starting_op->in_buffer.a_size != compared_op->in_buffer.a_size) { return false; }
        if(starting_op->in_buffer.z_size != compared_op->in_buffer.z_size) { return false; }
        if(starting_op->in_buffer.y_size != compared_op->in_buffer.y_size) { return false; }
        if(starting_op->in_buffer.x_size != compared_op->in_buffer.x_size) { return false; }
    }
    return true;
}
/* Returns the amount of ops in all the iterations of the loop combined, which makes it possible to use like `snprintf` for format-string appending. */
static uint64_t compile_loop_from_linearized_index(compile_loop_t *compile_loop, linearized_t *linearized, uint64_t start_index) {
    uint64_t loop_length = 0;
    uint64_t loop_number = 0;
    simple_op_t starting_op = linearized->simple[start_index];
    for(uint64_t i = start_index + 1; i < linearized->op_count; i++) {
        /* This is kind of a heuristic. I am 99% sure it works tho. */
        if(compile_loop_simple_op_equal(&starting_op, &linearized->simple[i])) {
            loop_length = i - start_index;
            break;
        }
    }
    if(!loop_length) { /* Could not find loop. */
        simple_op_t **loop_instances = calloc(1, sizeof(simple_op_t *));
        loop_instances[0] = calloc(1, sizeof(simple_op_t));
        loop_instances[0][0] = linearized->simple[start_index];
        compile_loop_configure(compile_loop, loop_instances, 1, 1);
        free(loop_instances[0]);
        free(loop_instances);
        return 1;
    }
    for(uint64_t i = start_index; i < linearized->op_count; i += loop_length) {
        if(compile_loop_simple_op_equal(&starting_op, &linearized->simple[i])) {
            loop_number++;
        } else {
            break;
        }
    }

    simple_op_t **loop_instances = calloc(loop_number, sizeof(simple_op_t *));
    for(uint64_t i = 0; i < loop_number; i++) { loop_instances[i] = calloc(loop_length, sizeof(simple_op_t)); }

    for(uint64_t i = 0; i < loop_number; i++) {
        for(uint64_t j = 0; j < loop_length; j++) { loop_instances[i][j] = linearized->simple[start_index + (loop_length * i) + j]; }
    }
    compile_loop_configure(compile_loop, loop_instances, loop_length, loop_number);

    for(uint64_t i = 0; i < loop_number; i++) { free(loop_instances[i]); }
    free(loop_instances);

    return loop_length * loop_number;
}
const uint64_t initial_source_size = 1000;
const uint64_t max_arg_size = 24;
const uint64_t max_index_digits = 9;
/* NOTE: Biggest I found was 131 for `max` or `min` binary ops. */
const uint64_t max_op_size = 256;
#define EXPAND_SOURCE_IF_NEEDED()                                                                                                                              \
    if(source_size - (curr - source) < max_op_size) {                                                                                                          \
        source_size *= 2;                                                                                                                                      \
        offset = curr - source;                                                                                                                                \
        source = realloc(source, source_size);                                                                                                                 \
        curr = source + offset;                                                                                                                                \
    }
/* TODO: Make use of multiple work-items per workgroup. */
/* Appends code for kernel that computes `compile_loop` with the specified global and local size. */
static void compile_loop_to_cl(const char *filename, compile_loop_t *compile_loop, uint64_t global_size, uint64_t local_size) {
    char *func_name = "money";
    /* TODO: Remove this after initial testing. */
    assert(local_size == 1);
    uint64_t leftover_loops = compile_loop->loop_number % global_size;
    uint64_t assigned_loops = (compile_loop->loop_number - leftover_loops) / global_size;
    uint64_t needed_loops;
    if(leftover_loops) {
        needed_loops = assigned_loops + 1;
    } else {
        needed_loops = assigned_loops;
    }

    uint64_t source_size = initial_source_size;
    char *source = malloc(initial_source_size);
    char *curr = source;
    uint64_t offset;

    uint64_t arg_num = 0;
    char **args = NULL;
    uint64_t found;
    for(uint64_t i = 0; i < compile_loop->loop_length; i++) {
        found = 0;
        for(uint64_t j = 0; j < arg_num; j++) {
            if(!strncmp(compile_loop->loop_instance[0][i].out_buffer.name, args[j], BUFFER_NAME_SIZE)) {
                found = 1;
                break;
            }
        }
        if(!found) {
            arg_num++;
            args = realloc(args, arg_num * sizeof(char *));
            args[arg_num - 1] = calloc(BUFFER_NAME_SIZE + 1, sizeof(char));
            strncpy(args[arg_num - 1], compile_loop->loop_instance[0][i].out_buffer.name, BUFFER_NAME_SIZE);
        }
        if(compile_loop->loop_instance[0][i].type != operation_unary) {
            found = 0;
            for(uint64_t j = 0; j < arg_num; j++) {
                if(!strncmp(compile_loop->loop_instance[0][i].in_buffer.name, args[j], BUFFER_NAME_SIZE)) {
                    found = 1;
                    break;
                }
            }
            if(!found) {
                arg_num++;
                args = realloc(args, arg_num * sizeof(char *));
                args[arg_num - 1] = calloc(BUFFER_NAME_SIZE + 1, sizeof(char));
                strncpy(args[arg_num - 1], compile_loop->loop_instance[0][i].in_buffer.name, BUFFER_NAME_SIZE);
            }
        }
    }

    /* TODO: Fix this by binding loops to work groups, this way an actual useful number of work-items can get spawned by using reasonable global dimensions and
     * it will also be possible then to split the remaining, non evenly dividing, loops up with a simple if statement. Each work-groups then has to split up
     * into the right number of work items. */
    curr += snprintf(curr, max_op_size, "int gid0 = get_global_id(0);\nint id = gid0;\n");
    EXPAND_SOURCE_IF_NEEDED();
    for(uint64_t i = 0; i < needed_loops; i++) {
        if(i) {
            curr += snprintf(curr, max_op_size, "id += %lu;\n", assigned_loops);
            EXPAND_SOURCE_IF_NEEDED();
        }
        if(i == assigned_loops) {
            curr += snprintf(curr, max_op_size, "if(gid0 < %lu) {\n", leftover_loops);
            EXPAND_SOURCE_IF_NEEDED();
        }
        for(uint64_t j = 0; j < compile_loop->loop_length; j++) {
            switch(compile_loop->loop_instance[0][j].type) {
                case(operation_unary): {
                    curr += snprintf(
                        curr, max_op_size,
                        "int %s%luoff%lu = (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) "
                        "+ (((id %% %lu) / %lu) * %lu + %lu);\n",
                        compile_loop->loop_instance[0][j].out_buffer.name, i, j, compile_loop->per_dim_reset_a[2 * j], compile_loop->per_dim_wait_a[2 * j],
                        compile_loop->per_dim_str_a[2 * j], compile_loop->per_dim_off_a[2 * j], compile_loop->per_dim_reset_z[2 * j],
                        compile_loop->per_dim_wait_z[2 * j], compile_loop->per_dim_str_z[2 * j], compile_loop->per_dim_off_z[2 * j],
                        compile_loop->per_dim_reset_y[2 * j], compile_loop->per_dim_wait_y[2 * j], compile_loop->per_dim_str_y[2 * j],
                        compile_loop->per_dim_off_y[2 * j], compile_loop->per_dim_reset_x[2 * j], compile_loop->per_dim_wait_x[2 * j],
                        compile_loop->per_dim_str_x[2 * j], compile_loop->per_dim_off_x[2 * j]);
                    EXPAND_SOURCE_IF_NEEDED();
                    switch(compile_loop->loop_instance[0][j].unary_type) {
                        case(unary_add): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] += %.16lf;\n",
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][j].var_unary);
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(unary_subtract): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] -= %.16lf;\n",
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][j].var_unary);
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(unary_multiply): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] *= %.16lf;\n",
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][j].var_unary);
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(unary_divide): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] /= %.16lf;\n",
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][j].var_unary);
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(unary_exp): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] = exp(%s[%s_%luoff%lu + %lu]);\n",
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x));
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(unary_log): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] = log(%s[%s_%luoff%lu + %lu]);\n",
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x));
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(unary_square): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] *= %s[%s_%luoff%lu + %lu];\n",
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x));
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(unary_sqrt): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] = sqrt(%s[%s_%luoff%lu + %lu]);\n",
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x));
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(unary_negate): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] *= -1;\n", compile_loop->loop_instance[0][j].out_buffer.name,
                                                         compile_loop->loop_instance[0][j].out_buffer.name, i, j,
                                                         SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x));
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(unary_reciprocal): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] = 1 / %s[%s_%luoff%lu + %lu];\n",
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x));
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(unary_max): {
                            TODO();
                            break;
                        }
                        case(unary_min): {
                            TODO();
                            break;
                        }
                        case(unary_set): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] = %.16lf;\n",
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][j].var_unary);
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        /* Never *ever* use this for things like encryption, where the randomnes of the numbers is important! */
                        case(unary_random): {
                            /*
                         this isn't really random at all but it should be sufficient for initializing
                         seed here is like the grid id or something
                         have to make sure that 1/epsilon isn't nan or inf
                         double random(int seed) {
                            double epsilon = 1e-4;
                            double modified = (sin(seed) / 100) + epsilon;
                            return(sin(1/modified));
                         }
                         */
                            TODO();
                            break;
                        }
                        case(unary_tanh): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] = tanh(%s[%s_%luoff%lu + %lu]);\n",
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x));
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(unary_absolute): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] = fabs(%s[%s_%luoff%lu + %lu]);\n",
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x));
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(unary_sign): {
                            TODO();
                            break;
                        }
                    }
                    break;
                }
                case(operation_binary): {
                    curr += snprintf(
                        curr, max_op_size,
                        "int %s%luoff%lu = (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) "
                        "+ (((id %% %lu) / %lu) * %lu + %lu);\n",
                        compile_loop->loop_instance[0][j].out_buffer.name, i, j, compile_loop->per_dim_reset_a[2 * j], compile_loop->per_dim_wait_a[2 * j],
                        compile_loop->per_dim_str_a[2 * j], compile_loop->per_dim_off_a[2 * j], compile_loop->per_dim_reset_z[2 * j],
                        compile_loop->per_dim_wait_z[2 * j], compile_loop->per_dim_str_z[2 * j], compile_loop->per_dim_off_z[2 * j],
                        compile_loop->per_dim_reset_y[2 * j], compile_loop->per_dim_wait_y[2 * j], compile_loop->per_dim_str_y[2 * j],
                        compile_loop->per_dim_off_y[2 * j], compile_loop->per_dim_reset_x[2 * j], compile_loop->per_dim_wait_x[2 * j],
                        compile_loop->per_dim_str_x[2 * j], compile_loop->per_dim_off_x[2 * j]);
                    EXPAND_SOURCE_IF_NEEDED();
                    curr += snprintf(
                        curr, max_op_size,
                        "int %s%luoff%lu = (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) "
                        "+ (((id %% %lu) / %lu) * %lu + %lu);\n",
                        compile_loop->loop_instance[0][j].in_buffer.name, i, j, compile_loop->per_dim_reset_a[2 * j + 1],
                        compile_loop->per_dim_wait_a[2 * j + 1], compile_loop->per_dim_str_a[2 * j + 1], compile_loop->per_dim_off_a[2 * j + 1],
                        compile_loop->per_dim_reset_z[2 * j + 1], compile_loop->per_dim_wait_z[2 * j + 1], compile_loop->per_dim_str_z[2 * j + 1],
                        compile_loop->per_dim_off_z[2 * j + 1], compile_loop->per_dim_reset_y[2 * j + 1], compile_loop->per_dim_wait_y[2 * j + 1],
                        compile_loop->per_dim_str_y[2 * j + 1], compile_loop->per_dim_off_y[2 * j + 1], compile_loop->per_dim_reset_x[2 * j + 1],
                        compile_loop->per_dim_wait_x[2 * j + 1], compile_loop->per_dim_str_x[2 * j + 1], compile_loop->per_dim_off_x[2 * j + 1]);
                    EXPAND_SOURCE_IF_NEEDED();
                    switch(compile_loop->loop_instance[0][j].binary_type) {
                        case(binary_add): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] += %s[%s%luoff%lu + %lu];\n",
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][j].in_buffer.name, compile_loop->loop_instance[0][j].in_buffer.name, i,
                                                         j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].in_buffer, a, z, y, x));
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(binary_subtract): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] -= %s[%s%luoff%lu + %lu];\n",
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][j].in_buffer.name, compile_loop->loop_instance[0][j].in_buffer.name, i,
                                                         j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].in_buffer, a, z, y, x));
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(binary_multiply): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] *= %s[%s%luoff%lu + %lu];\n",
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][j].in_buffer.name, compile_loop->loop_instance[0][j].in_buffer.name, i,
                                                         j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].in_buffer, a, z, y, x));
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(binary_divide): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] /= %s[%s%luoff%lu + %lu];\n",
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][j].in_buffer.name, compile_loop->loop_instance[0][j].in_buffer.name, i,
                                                         j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].in_buffer, a, z, y, x));
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(binary_max): {
                            TODO();
                            break;
                        }
                        case(binary_min): {
                            TODO();
                            break;
                        }
                        case(binary_copy): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr +=
                                                snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] = %s[%s%luoff%lu + %lu];\n",
                                                         compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name,
                                                         i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][j].in_buffer.name, compile_loop->loop_instance[0][j].in_buffer.name, i,
                                                         j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].in_buffer, a, z, y, x));
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(binary_add_like): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr += snprintf(
                                                curr, max_op_size, "%s[%s%luoff%lu + %lu] += %s[%s%luoff%lu];\n",
                                                compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name, i, j,
                                                SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                compile_loop->loop_instance[0][j].in_buffer.name, compile_loop->loop_instance[0][j].in_buffer.name, i, j);
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(binary_subtract_like): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr += snprintf(
                                                curr, max_op_size, "%s[%s%luoff%lu + %lu] -= %s[%s%luoff%lu];\n",
                                                compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name, i, j,
                                                SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                compile_loop->loop_instance[0][j].in_buffer.name, compile_loop->loop_instance[0][j].in_buffer.name, i, j);
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(binary_multiply_like): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr += snprintf(
                                                curr, max_op_size, "%s[%s%luoff%lu + %lu] *= %s[%s%luoff%lu];\n",
                                                compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name, i, j,
                                                SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                compile_loop->loop_instance[0][j].in_buffer.name, compile_loop->loop_instance[0][j].in_buffer.name, i, j);
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(binary_divide_like): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr += snprintf(
                                                curr, max_op_size, "%s[%s%luoff%lu + %lu] /= %s[%s%luoff%lu];\n",
                                                compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name, i, j,
                                                SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                compile_loop->loop_instance[0][j].in_buffer.name, compile_loop->loop_instance[0][j].in_buffer.name, i, j);
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(binary_max_like): {
                            TODO();
                            break;
                        }
                        case(binary_min_like): {
                            TODO();
                            break;
                        }
                        case(binary_copy_like): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].out_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].out_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].out_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].out_buffer.x_size; x++) {
                                            curr += snprintf(
                                                curr, max_op_size, "%s[%s%luoff%lu + %lu] = %s[%s%luoff%lu];\n",
                                                compile_loop->loop_instance[0][j].out_buffer.name, compile_loop->loop_instance[0][j].out_buffer.name, i, j,
                                                SIMPLE_INDEX(compile_loop->loop_instance[0][j].out_buffer, a, z, y, x),
                                                compile_loop->loop_instance[0][j].in_buffer.name, compile_loop->loop_instance[0][j].in_buffer.name, i, j);
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                    }
                    break;
                }
                case(operation_reduce): {
                    curr += snprintf(
                        curr, max_op_size,
                        "int %s%luoff%lu = (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) "
                        "+ (((id %% %lu) / %lu) * %lu + %lu);\n",
                        compile_loop->loop_instance[0][j].out_buffer.name, i, j, compile_loop->per_dim_reset_a[2 * j], compile_loop->per_dim_wait_a[2 * j],
                        compile_loop->per_dim_str_a[2 * j], compile_loop->per_dim_off_a[2 * j], compile_loop->per_dim_reset_z[2 * j],
                        compile_loop->per_dim_wait_z[2 * j], compile_loop->per_dim_str_z[2 * j], compile_loop->per_dim_off_z[2 * j],
                        compile_loop->per_dim_reset_y[2 * j], compile_loop->per_dim_wait_y[2 * j], compile_loop->per_dim_str_y[2 * j],
                        compile_loop->per_dim_off_y[2 * j], compile_loop->per_dim_reset_x[2 * j], compile_loop->per_dim_wait_x[2 * j],
                        compile_loop->per_dim_str_x[2 * j], compile_loop->per_dim_off_x[2 * j]);
                    EXPAND_SOURCE_IF_NEEDED();
                    curr += snprintf(
                        curr, max_op_size,
                        "int %s%luoff%lu = (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) "
                        "+ (((id %% %lu) / %lu) * %lu + %lu);\n",
                        compile_loop->loop_instance[0][j].in_buffer.name, i, j, compile_loop->per_dim_reset_a[2 * j + 1],
                        compile_loop->per_dim_wait_a[2 * j + 1], compile_loop->per_dim_str_a[2 * j + 1], compile_loop->per_dim_off_a[2 * j + 1],
                        compile_loop->per_dim_reset_z[2 * j + 1], compile_loop->per_dim_wait_z[2 * j + 1], compile_loop->per_dim_str_z[2 * j + 1],
                        compile_loop->per_dim_off_z[2 * j + 1], compile_loop->per_dim_reset_y[2 * j + 1], compile_loop->per_dim_wait_y[2 * j + 1],
                        compile_loop->per_dim_str_y[2 * j + 1], compile_loop->per_dim_off_y[2 * j + 1], compile_loop->per_dim_reset_x[2 * j + 1],
                        compile_loop->per_dim_wait_x[2 * j + 1], compile_loop->per_dim_str_x[2 * j + 1], compile_loop->per_dim_off_x[2 * j + 1]);
                    EXPAND_SOURCE_IF_NEEDED();
                    switch(compile_loop->loop_instance[0][j].reduce_type) {
                        case(reduce_sum): {
                            curr += snprintf(curr, max_op_size, "%s[%s%luoff%lu] = %lf;\n", compile_loop->loop_instance[0][j].out_buffer.name,
                                             compile_loop->loop_instance[0][j].out_buffer.name, i, j, 0.0);
                            EXPAND_SOURCE_IF_NEEDED();
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].in_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].in_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].in_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].in_buffer.x_size; x++) {
                                            curr += snprintf(curr, max_op_size, "%s[%s%luoff%lu] += %s[%s%luoff%lu + %lu];\n",
                                                             compile_loop->loop_instance[0][j].out_buffer.name,
                                                             compile_loop->loop_instance[0][j].out_buffer.name, i, j,
                                                             compile_loop->loop_instance[0][j].in_buffer.name, compile_loop->loop_instance[0][j].in_buffer.name,
                                                             i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].in_buffer, a, z, y, x));
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case(reduce_avg): {
                            for(uint64_t a = 0; a < compile_loop->loop_instance[0][j].in_buffer.a_size; a++) {
                                for(uint64_t z = 0; z < compile_loop->loop_instance[0][j].in_buffer.z_size; z++) {
                                    for(uint64_t y = 0; y < compile_loop->loop_instance[0][j].in_buffer.y_size; y++) {
                                        for(uint64_t x = 0; x < compile_loop->loop_instance[0][j].in_buffer.x_size; x++) {
                                            curr += snprintf(curr, max_op_size, "%s[%s%luoff%lu] += %s[%s%luoff%lu + %lu];\n",
                                                             compile_loop->loop_instance[0][j].out_buffer.name,
                                                             compile_loop->loop_instance[0][j].out_buffer.name, i, j,
                                                             compile_loop->loop_instance[0][j].in_buffer.name, compile_loop->loop_instance[0][j].in_buffer.name,
                                                             i, j, SIMPLE_INDEX(compile_loop->loop_instance[0][j].in_buffer, a, z, y, x));
                                            EXPAND_SOURCE_IF_NEEDED();
                                        }
                                    }
                                }
                            }
                            curr += snprintf(curr, max_op_size, "%s[%s%luoff%lu] /= %lf;\n", compile_loop->loop_instance[0][j].out_buffer.name,
                                             compile_loop->loop_instance[0][j].out_buffer.name, i, j,
                                             (double) compile_loop->loop_instance[0][j].in_buffer.a_size * compile_loop->loop_instance[0][j].in_buffer.z_size *
                                                 compile_loop->loop_instance[0][j].in_buffer.y_size * compile_loop->loop_instance[0][j].in_buffer.x_size);
                            EXPAND_SOURCE_IF_NEEDED();
                            break;
                        }
                        case(reduce_max): {
                            TODO();
                            break;
                        }
                        case(reduce_min): {
                            TODO();
                            break;
                        }
                    }
                    break;
                }
                case(operation_move): {
                    ERROR("ERROR: Tried to compile move operation to OpenCL at index %lu\n", j);
                }
            }
        }
        if(i == assigned_loops) {
            curr += snprintf(curr, max_op_size, "}\n");
            EXPAND_SOURCE_IF_NEEDED();
        }
    }

    assert(arg_num != 0);
    /* This formula is very jank, but it makes sense, if you think about it. */
    uint64_t kernel_size = strlen("__kernel void ") + strlen(func_name) + (strlen("__global double *") + BUFFER_NAME_SIZE) * arg_num +
                           strlen(", ") * (arg_num - 1) + strlen(") {\n") + (curr - source) + strlen("}\n");
    char *kernel = malloc(kernel_size);
    char *kernel_i = kernel;
    kernel_i += sprintf(kernel_i, "__kernel void %s(", func_name);
    for(uint64_t i = 0; i < arg_num; i++) {
        if(i != arg_num - 1) {
            kernel_i += sprintf(kernel_i, "__global double *%s, ", args[i]);
        } else {
            kernel_i += sprintf(kernel_i, "__global double *%s) {\n", args[i]);
        }
    }
    /* This one is very sus. Doing sprintf crashes the program and I have no clue why. This desperatly needs to be investigated. */
    kernel_i += sprintf(kernel_i, "%s}\n", source);

    FILE *f = fopen(filename, "a");
    fwrite(kernel, sizeof(char), kernel_size, f);
    fclose(f);

    printf("%lu %lu\n", strlen(source), curr - source);
    free(source);
    free(kernel);
    for(uint64_t i = 0; i < arg_num; i++) { free(args[i]); }
    free(args);
}
void compile_linearized_to_cl(const char *filename, linearized_t *linearized) {
    compile_loop_t compile_loop = {0};
    /* Clears file. */
    FILE *f = fopen(filename, "w");
    fclose(f);
    // uint64_t i = compile_loop_from_linearized_index(&compile_loop, linearized, 1);
    uint64_t i = compile_loop_from_linearized_index(&compile_loop, linearized, 0);
    compile_loop_print(&compile_loop, 4, 0, "");
    uint64_t global_size = 9;
    uint64_t local_size = 1;
    compile_loop_to_cl(filename, &compile_loop, global_size, local_size);
}
