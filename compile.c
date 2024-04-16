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
static void simple_loop_free(simple_loop_t *simple_loop) {
    free(simple_loop->loop_instance);
    free(simple_loop->per_dim_off_a);
    free(simple_loop->per_dim_off_z);
    free(simple_loop->per_dim_off_y);
    free(simple_loop->per_dim_off_x);
    free(simple_loop->per_dim_str_a);
    free(simple_loop->per_dim_str_z);
    free(simple_loop->per_dim_str_y);
    free(simple_loop->per_dim_str_x);
    free(simple_loop->per_dim_wait_a);
    free(simple_loop->per_dim_wait_z);
    free(simple_loop->per_dim_wait_y);
    free(simple_loop->per_dim_wait_x);
    free(simple_loop->per_dim_reset_a);
    free(simple_loop->per_dim_reset_z);
    free(simple_loop->per_dim_reset_y);
    free(simple_loop->per_dim_reset_x);
}
/* TODO: Check if `simple_loop` has already been configured, by checking pointers for NULL. */
/* TODO: Don't pass all the loops and just check earlier, that it they are all valid repetitions of each other. */
static void simple_loop_configure(simple_loop_t *simple_loop, simple_op_t **simple_op, uint64_t loop_length, uint64_t loop_number) {
    if(simple_loop->loop_instance) { simple_loop_free(simple_loop); }
    simple_loop->loop_num = loop_number;
    simple_loop->loop_len = loop_length;
    simple_loop->loop_instance = calloc(loop_length, sizeof(simple_op_t));
    assert(simple_loop->loop_instance);
    for(uint64_t i = 0; i < loop_length; i++) { simple_loop->loop_instance[i] = simple_op[0][i]; }
    simple_loop->per_dim_off_a = calloc(loop_length * 2, sizeof(uint64_t));
    assert(simple_loop->per_dim_off_a);
    simple_loop->per_dim_off_z = calloc(loop_length * 2, sizeof(uint64_t));
    assert(simple_loop->per_dim_off_z);
    simple_loop->per_dim_off_y = calloc(loop_length * 2, sizeof(uint64_t));
    assert(simple_loop->per_dim_off_y);
    simple_loop->per_dim_off_x = calloc(loop_length * 2, sizeof(uint64_t));
    assert(simple_loop->per_dim_off_x);
    simple_loop->per_dim_str_a = calloc(loop_length * 2, sizeof(uint64_t));
    assert(simple_loop->per_dim_str_a);
    simple_loop->per_dim_str_z = calloc(loop_length * 2, sizeof(uint64_t));
    assert(simple_loop->per_dim_str_z);
    simple_loop->per_dim_str_y = calloc(loop_length * 2, sizeof(uint64_t));
    assert(simple_loop->per_dim_str_y);
    simple_loop->per_dim_str_x = calloc(loop_length * 2, sizeof(uint64_t));
    assert(simple_loop->per_dim_str_x);
    simple_loop->per_dim_reset_a = calloc(loop_length * 2, sizeof(uint64_t));
    assert(simple_loop->per_dim_reset_a);
    simple_loop->per_dim_reset_z = calloc(loop_length * 2, sizeof(uint64_t));
    assert(simple_loop->per_dim_reset_z);
    simple_loop->per_dim_reset_y = calloc(loop_length * 2, sizeof(uint64_t));
    assert(simple_loop->per_dim_reset_y);
    simple_loop->per_dim_reset_x = calloc(loop_length * 2, sizeof(uint64_t));
    assert(simple_loop->per_dim_reset_x);
    simple_loop->per_dim_wait_a = calloc(loop_length * 2, sizeof(uint64_t));
    assert(simple_loop->per_dim_wait_a);
    simple_loop->per_dim_wait_z = calloc(loop_length * 2, sizeof(uint64_t));
    assert(simple_loop->per_dim_wait_z);
    simple_loop->per_dim_wait_y = calloc(loop_length * 2, sizeof(uint64_t));
    assert(simple_loop->per_dim_wait_y);
    simple_loop->per_dim_wait_x = calloc(loop_length * 2, sizeof(uint64_t));
    assert(simple_loop->per_dim_wait_x);
    /* FIX: Currently we assume that the initial op necessarily has the lowest indices, this should *really* be fixed to have some sorting. This would also fix
     * potentially negative strides. */
    for(uint64_t i = 0; i < loop_length; i++) {
        simple_loop->per_dim_off_a[2 * i] = simple_loop->loop_instance[i].out_buffer.a_offset;
        simple_loop->per_dim_off_z[2 * i] = simple_loop->loop_instance[i].out_buffer.z_offset;
        simple_loop->per_dim_off_y[2 * i] = simple_loop->loop_instance[i].out_buffer.y_offset;
        simple_loop->per_dim_off_x[2 * i] = simple_loop->loop_instance[i].out_buffer.x_offset;
        if(simple_loop->loop_instance[i].type != operation_unary) {
            simple_loop->per_dim_off_a[2 * i + 1] = simple_loop->loop_instance[i].in_buffer.a_offset;
            simple_loop->per_dim_off_z[2 * i + 1] = simple_loop->loop_instance[i].in_buffer.z_offset;
            simple_loop->per_dim_off_y[2 * i + 1] = simple_loop->loop_instance[i].in_buffer.y_offset;
            simple_loop->per_dim_off_x[2 * i + 1] = simple_loop->loop_instance[i].in_buffer.x_offset;
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
            if((!found_a_o) && simple_op[j][i].out_buffer.a_offset != simple_loop->per_dim_off_a[2 * i]) {
                simple_loop->per_dim_str_a[2 * i] = simple_op[j][i].out_buffer.a_offset - simple_loop->per_dim_off_a[2 * i];
                simple_loop->per_dim_wait_a[2 * i] = j;
                found_a_o = 1;
            }
            if((!found_z_o) && simple_op[j][i].out_buffer.z_offset != simple_loop->per_dim_off_z[2 * i]) {
                simple_loop->per_dim_str_z[2 * i] = simple_op[j][i].out_buffer.z_offset - simple_loop->per_dim_off_z[2 * i];
                simple_loop->per_dim_wait_z[2 * i] = j;
                found_z_o = 1;
            }
            if((!found_y_o) && simple_op[j][i].out_buffer.y_offset != simple_loop->per_dim_off_y[2 * i]) {
                simple_loop->per_dim_str_y[2 * i] = simple_op[j][i].out_buffer.y_offset - simple_loop->per_dim_off_y[2 * i];
                simple_loop->per_dim_wait_y[2 * i] = j;
                found_y_o = 1;
            }
            if((!found_x_o) && simple_op[j][i].out_buffer.x_offset != simple_loop->per_dim_off_x[2 * i]) {
                simple_loop->per_dim_str_x[2 * i] = simple_op[j][i].out_buffer.x_offset - simple_loop->per_dim_off_x[2 * i];
                simple_loop->per_dim_wait_x[2 * i] = j;
                found_x_o = 1;
            }
            if(simple_loop->loop_instance[i].type != operation_unary) {
                if((!found_a_i) && simple_op[j][i].in_buffer.a_offset != simple_loop->per_dim_off_a[2 * i + 1]) {
                    simple_loop->per_dim_str_a[2 * i + 1] = simple_op[j][i].in_buffer.a_offset - simple_loop->per_dim_off_a[2 * i + 1];
                    simple_loop->per_dim_wait_a[2 * i + 1] = j;
                    found_a_i = 1;
                }
                if((!found_z_i) && simple_op[j][i].in_buffer.z_offset != simple_loop->per_dim_off_z[2 * i + 1]) {
                    simple_loop->per_dim_str_z[2 * i + 1] = simple_op[j][i].in_buffer.z_offset - simple_loop->per_dim_off_z[2 * i + 1];
                    simple_loop->per_dim_wait_z[2 * i + 1] = j;
                    found_z_i = 1;
                }
                if((!found_y_i) && simple_op[j][i].in_buffer.y_offset != simple_loop->per_dim_off_y[2 * i + 1]) {
                    simple_loop->per_dim_str_y[2 * i + 1] = simple_op[j][i].in_buffer.y_offset - simple_loop->per_dim_off_y[2 * i + 1];
                    simple_loop->per_dim_wait_y[2 * i + 1] = j;
                    found_y_i = 1;
                }
                if((!found_x_i) && simple_op[j][i].in_buffer.x_offset != simple_loop->per_dim_off_x[2 * i + 1]) {
                    simple_loop->per_dim_str_x[2 * i + 1] = simple_op[j][i].in_buffer.x_offset - simple_loop->per_dim_off_x[2 * i + 1];
                    simple_loop->per_dim_wait_x[2 * i + 1] = j;
                    found_x_i = 1;
                }
            }
        }
        if(!found_a_o) {
            simple_loop->per_dim_str_a[2 * i] = 0;
            simple_loop->per_dim_wait_a[2 * i] = loop_number;
        }
        if(!found_z_o) {
            simple_loop->per_dim_str_z[2 * i] = 0;
            simple_loop->per_dim_wait_z[2 * i] = loop_number;
        }
        if(!found_y_o) {
            simple_loop->per_dim_str_y[2 * i] = 0;
            simple_loop->per_dim_wait_y[2 * i] = loop_number;
        }
        if(!found_x_o) {
            simple_loop->per_dim_str_x[2 * i] = 0;
            simple_loop->per_dim_wait_x[2 * i] = loop_number;
        }
        if(simple_loop->loop_instance[i].type != operation_unary) {
            if(!found_a_i) {
                simple_loop->per_dim_str_a[2 * i + 1] = 0;
                simple_loop->per_dim_wait_a[2 * i + 1] = loop_number;
            }
            if(!found_z_i) {
                simple_loop->per_dim_str_z[2 * i + 1] = 0;
                simple_loop->per_dim_wait_z[2 * i + 1] = loop_number;
            }
            if(!found_y_i) {
                simple_loop->per_dim_str_y[2 * i + 1] = 0;
                simple_loop->per_dim_wait_y[2 * i + 1] = loop_number;
            }
            if(!found_x_i) {
                simple_loop->per_dim_str_x[2 * i + 1] = 0;
                simple_loop->per_dim_wait_x[2 * i + 1] = loop_number;
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
            if((!left_a_o) && (!found_a_o) && simple_op[j][i].out_buffer.a_offset != simple_loop->per_dim_off_a[2 * i]) { left_a_o = 1; }
            if(left_a_o && (!found_a_o) && simple_op[j][i].out_buffer.a_offset == simple_loop->per_dim_off_a[2 * i]) {
                simple_loop->per_dim_reset_a[2 * i] = j;
                found_a_o = 1;
            }
            if((!left_z_o) && (!found_z_o) && simple_op[j][i].out_buffer.z_offset != simple_loop->per_dim_off_z[2 * i]) { left_z_o = 1; }
            if(left_z_o && (!found_z_o) && simple_op[j][i].out_buffer.z_offset == simple_loop->per_dim_off_z[2 * i]) {
                simple_loop->per_dim_reset_z[2 * i] = j;
                found_z_o = 1;
            }
            if((!left_y_o) && (!found_y_o) && simple_op[j][i].out_buffer.y_offset != simple_loop->per_dim_off_y[2 * i]) { left_y_o = 1; }
            if(left_y_o && (!found_y_o) && simple_op[j][i].out_buffer.y_offset == simple_loop->per_dim_off_y[2 * i]) {
                simple_loop->per_dim_reset_y[2 * i] = j;
                found_y_o = 1;
            }
            if((!left_x_o) && (!found_x_o) && simple_op[j][i].out_buffer.x_offset != simple_loop->per_dim_off_x[2 * i]) { left_x_o = 1; }
            if(left_x_o && (!found_x_o) && simple_op[j][i].out_buffer.x_offset == simple_loop->per_dim_off_x[2 * i]) {
                simple_loop->per_dim_reset_x[2 * i] = j;
                found_x_o = 1;
            }
            if(simple_loop->loop_instance[i].type != operation_unary) {
                if((!left_a_i) && (!found_a_i) && simple_op[j][i].in_buffer.a_offset != simple_loop->per_dim_off_a[2 * i + 1]) { left_a_i = 1; }
                if(left_a_i && (!found_a_i) && simple_op[j][i].in_buffer.a_offset == simple_loop->per_dim_off_a[2 * i + 1]) {
                    simple_loop->per_dim_reset_a[2 * i + 1] = j;
                    found_a_i = 1;
                }
                if((!left_z_i) && (!found_z_i) && simple_op[j][i].in_buffer.z_offset != simple_loop->per_dim_off_z[2 * i + 1]) { left_z_i = 1; }
                if(left_z_i && (!found_z_i) && simple_op[j][i].in_buffer.z_offset == simple_loop->per_dim_off_z[2 * i + 1]) {
                    simple_loop->per_dim_reset_z[2 * i + 1] = j;
                    found_z_i = 1;
                }
                if((!left_y_i) && (!found_y_i) && simple_op[j][i].in_buffer.y_offset != simple_loop->per_dim_off_y[2 * i + 1]) { left_y_i = 1; }
                if(left_y_i && (!found_y_i) && simple_op[j][i].in_buffer.y_offset == simple_loop->per_dim_off_y[2 * i + 1]) {
                    simple_loop->per_dim_reset_y[2 * i + 1] = j;
                    found_y_i = 1;
                }
                if((!left_x_i) && (!found_x_i) && simple_op[j][i].in_buffer.x_offset != simple_loop->per_dim_off_x[2 * i + 1]) { left_x_i = 1; }
                if(left_x_i && (!found_x_i) && simple_op[j][i].in_buffer.x_offset == simple_loop->per_dim_off_x[2 * i + 1]) {
                    simple_loop->per_dim_reset_x[2 * i + 1] = j;
                    found_x_i = 1;
                }
            }
        }
        if(!found_a_o) { simple_loop->per_dim_reset_a[2 * i] = loop_number; }
        if(!found_z_o) { simple_loop->per_dim_reset_z[2 * i] = loop_number; }
        if(!found_y_o) { simple_loop->per_dim_reset_y[2 * i] = loop_number; }
        if(!found_x_o) { simple_loop->per_dim_reset_x[2 * i] = loop_number; }
        if(simple_loop->loop_instance[i].type != operation_unary) {
            if(!found_a_i) { simple_loop->per_dim_reset_a[2 * i + 1] = loop_number; }
            if(!found_z_i) { simple_loop->per_dim_reset_z[2 * i + 1] = loop_number; }
            if(!found_y_i) { simple_loop->per_dim_reset_y[2 * i + 1] = loop_number; }
            if(!found_x_i) { simple_loop->per_dim_reset_x[2 * i + 1] = loop_number; }
        }
    }
}
static void simple_loop_print(simple_loop_t *simple_loop, int padding, int offset, const char *name) {
    if(!strncmp(name, "", 1)) {
        printf("%*ssimple loop %lu repetitions\n", offset, "", simple_loop->loop_num);
    } else {
        printf("%*s%s %lu repetitions\n", offset, "", name, simple_loop->loop_num);
    }
    for(uint64_t i = 0; i < simple_loop->loop_len; i++) {
        printf("%*s[%lu] ", offset + padding, "", i);
        simple_op_print(&simple_loop->loop_instance[i], 0, 0, "");
    }
    printf("off\n");
    for(uint64_t i = 0; i < simple_loop->loop_len; i++) {
        if(simple_loop->loop_instance[i].type == operation_unary) {
            printf("{%lu, %lu, %lu, %lu}\n", simple_loop->per_dim_off_a[2 * i], simple_loop->per_dim_off_z[2 * i], simple_loop->per_dim_off_y[2 * i],
                   simple_loop->per_dim_off_x[2 * i]);
        } else {
            printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", simple_loop->per_dim_off_a[2 * i], simple_loop->per_dim_off_z[2 * i],
                   simple_loop->per_dim_off_y[2 * i], simple_loop->per_dim_off_x[2 * i], simple_loop->per_dim_off_a[2 * i + 1],
                   simple_loop->per_dim_off_z[2 * i + 1], simple_loop->per_dim_off_y[2 * i + 1], simple_loop->per_dim_off_x[2 * i + 1]);
        }
    }
    printf("str\n");
    for(uint64_t i = 0; i < simple_loop->loop_len; i++) {
        if(simple_loop->loop_instance[i].type == operation_unary) {
            printf("{%lu, %lu, %lu, %lu}\n", simple_loop->per_dim_str_a[2 * i], simple_loop->per_dim_str_z[2 * i], simple_loop->per_dim_str_y[2 * i],
                   simple_loop->per_dim_str_x[2 * i]);
        } else {
            printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", simple_loop->per_dim_str_a[2 * i], simple_loop->per_dim_str_z[2 * i],
                   simple_loop->per_dim_str_y[2 * i], simple_loop->per_dim_str_x[2 * i], simple_loop->per_dim_str_a[2 * i + 1],
                   simple_loop->per_dim_str_z[2 * i + 1], simple_loop->per_dim_str_y[2 * i + 1], simple_loop->per_dim_str_x[2 * i + 1]);
        }
    }
    printf("res\n");
    for(uint64_t i = 0; i < simple_loop->loop_len; i++) {
        if(simple_loop->loop_instance[i].type == operation_unary) {
            printf("{%lu, %lu, %lu, %lu}\n", simple_loop->per_dim_reset_a[2 * i], simple_loop->per_dim_reset_z[2 * i], simple_loop->per_dim_reset_y[2 * i],
                   simple_loop->per_dim_reset_x[2 * i]);
        } else {
            printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", simple_loop->per_dim_reset_a[2 * i], simple_loop->per_dim_reset_z[2 * i],
                   simple_loop->per_dim_reset_y[2 * i], simple_loop->per_dim_reset_x[2 * i], simple_loop->per_dim_reset_a[2 * i + 1],
                   simple_loop->per_dim_reset_z[2 * i + 1], simple_loop->per_dim_reset_y[2 * i + 1], simple_loop->per_dim_reset_x[2 * i + 1]);
        }
    }
    printf("wat\n");
    for(uint64_t i = 0; i < simple_loop->loop_len; i++) {
        if(simple_loop->loop_instance[i].type == operation_unary) {
            printf("{%lu, %lu, %lu, %lu}\n", simple_loop->per_dim_wait_a[2 * i], simple_loop->per_dim_wait_z[2 * i], simple_loop->per_dim_wait_y[2 * i],
                   simple_loop->per_dim_wait_x[2 * i]);
        } else {
            printf("{%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", simple_loop->per_dim_wait_a[2 * i], simple_loop->per_dim_wait_z[2 * i],
                   simple_loop->per_dim_wait_y[2 * i], simple_loop->per_dim_wait_x[2 * i], simple_loop->per_dim_wait_a[2 * i + 1],
                   simple_loop->per_dim_wait_z[2 * i + 1], simple_loop->per_dim_wait_y[2 * i + 1], simple_loop->per_dim_wait_x[2 * i + 1]);
        }
    }
}
static void cl_kernel_free(cl_kernel_t *kernel) {
    for(uint64_t i = 0; i < kernel->arg_num; i++) { free(kernel->args[i]); }
    free(kernel->args);
    free((void *) kernel->name);
}
// static void cl_kernel_print(cl_kernel_t *kernel, int padding, int offset, const char *name) {
//     if(strncmp(name, "", 1)) {
//         printf("%*s%s %s\n", offset, "", name, kernel->name);
//     } else {
//         printf("%*scl kernel %s\n", offset, "", kernel->name);
//     }
//     for(uint64_t i = 0; i < kernel->arg_num; i++) { printf("%*s[%lu] %s\n", padding + offset, "", i, kernel->args[i]); }
// }
/* Has to have the same input and output tensors, with the same shape and be the same op type. Offsets however should be irrelevant. */
static ALWAYS_INLINE bool simple_loop_simple_op_equal(simple_op_t *starting_op, simple_op_t *compared_op) {
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
static uint64_t simple_loop_from_linearized_index(simple_loop_t *simple_loop, linearized_t *linearized, uint64_t start_index) {
    uint64_t loop_length = 0;
    uint64_t loop_number = 0;
    uint64_t diff;
    simple_op_t starting_op = linearized->simple[start_index];
    for(uint64_t i = start_index + 1; i < linearized->op_count; i++) {
        if(simple_loop_simple_op_equal(&starting_op, &linearized->simple[i])) {
            /* TODO: This could probably just be done in the `for` statement. */
            if(2 * i - start_index < linearized->op_count) {
                diff = 0;
                for(uint64_t j = 0; j < i - start_index; j++) {
                    if(!simple_loop_simple_op_equal(&linearized->simple[start_index + j], &linearized->simple[i + j])) {
                        diff = 1;
                        break;
                    }
                }
                if(!diff) {
                    loop_length = i - start_index;
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
        loop_instances[0][0] = linearized->simple[start_index];
        simple_loop_configure(simple_loop, loop_instances, 1, 1);
        free(loop_instances[0]);
        free(loop_instances);
        return 1;
    }
    for(uint64_t i = start_index; i < linearized->op_count; i += loop_length) {
        if(simple_loop_simple_op_equal(&starting_op, &linearized->simple[i])) {
            loop_number++;
        } else {
            break;
        }
    }

    simple_op_t **loop_instances = calloc(loop_number, sizeof(simple_op_t *));
    assert(loop_instances);
    for(uint64_t i = 0; i < loop_number; i++) {
        loop_instances[i] = calloc(loop_length, sizeof(simple_op_t));
        assert(loop_instances[i]);
    }

    for(uint64_t i = 0; i < loop_number; i++) {
        for(uint64_t j = 0; j < loop_length; j++) { loop_instances[i][j] = linearized->simple[start_index + (loop_length * i) + j]; }
    }
    simple_loop_configure(simple_loop, loop_instances, loop_length, loop_number);

    for(uint64_t i = 0; i < loop_number; i++) { free(loop_instances[i]); }
    free(loop_instances);

    return loop_length * loop_number;
}
const uint64_t initial_cap = 4;
#define OVERRIDES_OUTPUT(op)                                                                                                                                   \
    ((op.type == operation_unary && op.binary_type == unary_set) ||                                                                                            \
     (op.type == operation_binary && (op.binary_type == binary_copy || op.binary_type == binary_copy_like)) || (op.type == operation_reduce))
static void compile_loop_optimize(compile_loop_t *compile, uint64_t optim) {
    if(optim & OPTIMIZE_INLINE) {
        printf("Optimizing: Inline\n");
        uint64_t inline_cap = initial_cap;
        uint64_t inline_num = 0;
        simple_op_t *inlined = calloc(initial_cap, sizeof(simple_op_t));
        assert(inlined);

        for(uint64_t i = 0; i < compile->loop_len; i++) {
            if(compile->op[i][0].type == operation_binary && compile->op[i][0].binary_type == binary_copy) {
                inline_num = 1;
                inlined[0] = compile->op[i][0];
                simple_op_print(&inlined[0], 4, 0, "");
                for(uint64_t j = 1; j < compile->loop_len - i; j++) {
                    if(!strncmp(compile->op[i][0].out_buffer.name, compile->op[i + j][0].out_buffer.name, BUFFER_NAME_SIZE)) {
                        if(OVERRIDES_OUTPUT(compile->op[i + j][0])) {
                            break;
                        } else {
                            compile->op_num[i + j] = compile->op_cap[i + j];
                            inline_num++;
                            if(inline_num == inline_cap) {
                                inline_cap *= 2;
                                inlined = realloc(inlined, inline_cap * sizeof(simple_op_t));
                            }
                            inlined[inline_num - 1] = compile->op[i + j][0];
                        }
                    } else if(!strncmp(compile->op[i][0].out_buffer.name, compile->op[i + j][0].in_buffer.name, BUFFER_NAME_SIZE)) {
                        compile->op_num[i] = compile->op_cap[i];
                        compile->op_num[i + j] += inline_num;
                        if(compile->op_num[i + j] >= compile->op_cap[i + j]) {
                            compile->op_cap[i + j] *= 2;
                            compile->op[i + j] = realloc(compile->op[i + j], compile->op_cap[i + j] * sizeof(simple_op_t));
                        }
                        for(uint64_t k = 0; k < inline_num; k++) { compile->op[i + j][k + 1] = inlined[k]; }
                    }
                }
            }
        }
        free(inlined);
        uint64_t c = 0;
        uint64_t new_len = compile->loop_len;
        /* Kinda stupid to do it here. */
        for(uint64_t i = 0; i < compile->loop_len; i++) {
            if(compile->op_num[i] == compile->op_cap[i]) {
                free(compile->op[i]);
                free(compile->per_dim_off_a[2 * i]);
                free(compile->per_dim_off_z[2 * i]);
                free(compile->per_dim_off_y[2 * i]);
                free(compile->per_dim_off_x[2 * i]);
                free(compile->per_dim_off_a[2 * i + 1]);
                free(compile->per_dim_off_z[2 * i + 1]);
                free(compile->per_dim_off_y[2 * i + 1]);
                free(compile->per_dim_off_x[2 * i + 1]);
                free(compile->per_dim_str_a[2 * i]);
                free(compile->per_dim_str_z[2 * i]);
                free(compile->per_dim_str_y[2 * i]);
                free(compile->per_dim_str_x[2 * i]);
                free(compile->per_dim_str_a[2 * i + 1]);
                free(compile->per_dim_str_z[2 * i + 1]);
                free(compile->per_dim_str_y[2 * i + 1]);
                free(compile->per_dim_str_x[2 * i + 1]);
                free(compile->per_dim_reset_a[2 * i]);
                free(compile->per_dim_reset_z[2 * i]);
                free(compile->per_dim_reset_y[2 * i]);
                free(compile->per_dim_reset_x[2 * i]);
                free(compile->per_dim_reset_a[2 * i + 1]);
                free(compile->per_dim_reset_z[2 * i + 1]);
                free(compile->per_dim_reset_y[2 * i + 1]);
                free(compile->per_dim_reset_x[2 * i + 1]);
                free(compile->per_dim_wait_a[2 * i]);
                free(compile->per_dim_wait_z[2 * i]);
                free(compile->per_dim_wait_y[2 * i]);
                free(compile->per_dim_wait_x[2 * i]);
                free(compile->per_dim_wait_a[2 * i + 1]);
                free(compile->per_dim_wait_z[2 * i + 1]);
                free(compile->per_dim_wait_y[2 * i + 1]);
                free(compile->per_dim_wait_x[2 * i + 1]);
                new_len--;
            } else {
                compile->op_cap[c] = compile->op_cap[i];
                compile->op_num[c] = compile->op_num[i];
                compile->op[c] = compile->op[i];
                compile->per_dim_off_a[2 * c] = compile->per_dim_off_a[2 * i];
                compile->per_dim_off_z[2 * c] = compile->per_dim_off_z[2 * i];
                compile->per_dim_off_y[2 * c] = compile->per_dim_off_y[2 * i];
                compile->per_dim_off_x[2 * c] = compile->per_dim_off_x[2 * i];
                compile->per_dim_off_a[2 * c + 1] = compile->per_dim_off_a[2 * i + 1];
                compile->per_dim_off_z[2 * c + 1] = compile->per_dim_off_z[2 * i + 1];
                compile->per_dim_off_y[2 * c + 1] = compile->per_dim_off_y[2 * i + 1];
                compile->per_dim_off_x[2 * c + 1] = compile->per_dim_off_x[2 * i + 1];
                compile->per_dim_str_a[2 * c] = compile->per_dim_str_a[2 * i];
                compile->per_dim_str_z[2 * c] = compile->per_dim_str_z[2 * i];
                compile->per_dim_str_y[2 * c] = compile->per_dim_str_y[2 * i];
                compile->per_dim_str_x[2 * c] = compile->per_dim_str_x[2 * i];
                compile->per_dim_str_a[2 * c + 1] = compile->per_dim_str_a[2 * i + 1];
                compile->per_dim_str_z[2 * c + 1] = compile->per_dim_str_z[2 * i + 1];
                compile->per_dim_str_y[2 * c + 1] = compile->per_dim_str_y[2 * i + 1];
                compile->per_dim_str_x[2 * c + 1] = compile->per_dim_str_x[2 * i + 1];
                compile->per_dim_reset_a[2 * c] = compile->per_dim_reset_a[2 * i];
                compile->per_dim_reset_z[2 * c] = compile->per_dim_reset_z[2 * i];
                compile->per_dim_reset_y[2 * c] = compile->per_dim_reset_y[2 * i];
                compile->per_dim_reset_x[2 * c] = compile->per_dim_reset_x[2 * i];
                compile->per_dim_reset_a[2 * c + 1] = compile->per_dim_reset_a[2 * i + 1];
                compile->per_dim_reset_z[2 * c + 1] = compile->per_dim_reset_z[2 * i + 1];
                compile->per_dim_reset_y[2 * c + 1] = compile->per_dim_reset_y[2 * i + 1];
                compile->per_dim_reset_x[2 * c + 1] = compile->per_dim_reset_x[2 * i + 1];
                compile->per_dim_wait_a[2 * c] = compile->per_dim_wait_a[2 * i];
                compile->per_dim_wait_z[2 * c] = compile->per_dim_wait_z[2 * i];
                compile->per_dim_wait_y[2 * c] = compile->per_dim_wait_y[2 * i];
                compile->per_dim_wait_x[2 * c] = compile->per_dim_wait_x[2 * i];
                compile->per_dim_wait_a[2 * c + 1] = compile->per_dim_wait_a[2 * i + 1];
                compile->per_dim_wait_z[2 * c + 1] = compile->per_dim_wait_z[2 * i + 1];
                compile->per_dim_wait_y[2 * c + 1] = compile->per_dim_wait_y[2 * i + 1];
                compile->per_dim_wait_x[2 * c + 1] = compile->per_dim_wait_x[2 * i + 1];
                c++;
            }
        }
        compile->loop_len = new_len;
    }
    if(optim & OPTIMIZE_FUSE) { printf("Optimizing: Fuse\n"); }
}
static void compile_loop_print(compile_loop_t *compile, int padding, int offset, const char *name) {
    if(!strncmp(name, "", 1)) {
        printf("%*scompile loop\n", offset, "");
    } else {
        printf("%*s%s\n", offset, "", name);
    }
    for(uint64_t i = 0; i < compile->loop_len; i++) {
        for(uint64_t j = 0; j < compile->op_num[i]; j++) {
            if(j) {
                printf("%*s[%lu, %lu] ", 2 * padding + offset, "", i, j);
            } else {
                printf("%*s[%lu, 0] ", padding + offset, "", i);
            }
            simple_op_print(&compile->op[i][j], 0, 0, "");
        }
    }
}
static void compile_loop_free(compile_loop_t *compile) {
    for(uint64_t i = 0; i < compile->loop_len; i++) { free(compile->op[i]); }
    printf("ll %lu\n", compile->loop_len);
    for(uint64_t i = 0; i < 2 * compile->loop_len; i++) {
        free(compile->per_dim_off_a[i]);
        free(compile->per_dim_off_z[i]);
        free(compile->per_dim_off_y[i]);
        free(compile->per_dim_off_x[i]);
        free(compile->per_dim_str_a[i]);
        free(compile->per_dim_str_z[i]);
        free(compile->per_dim_str_y[i]);
        free(compile->per_dim_str_x[i]);
        free(compile->per_dim_wait_a[i]);
        free(compile->per_dim_wait_z[i]);
        free(compile->per_dim_wait_y[i]);
        free(compile->per_dim_wait_x[i]);
        free(compile->per_dim_reset_a[i]);
        free(compile->per_dim_reset_z[i]);
        free(compile->per_dim_reset_x[i]);
        free(compile->per_dim_reset_y[i]);
    }
    free(compile->op);
    free(compile->op_num);
    free(compile->op_cap);
    free(compile->per_dim_off_a);
    free(compile->per_dim_off_z);
    free(compile->per_dim_off_y);
    free(compile->per_dim_off_x);
    free(compile->per_dim_str_a);
    free(compile->per_dim_str_z);
    free(compile->per_dim_str_y);
    free(compile->per_dim_str_x);
    free(compile->per_dim_wait_a);
    free(compile->per_dim_wait_z);
    free(compile->per_dim_wait_y);
    free(compile->per_dim_wait_x);
    free(compile->per_dim_reset_a);
    free(compile->per_dim_reset_z);
    free(compile->per_dim_reset_x);
    free(compile->per_dim_reset_y);
}
static compile_loop_t compile_loop_alloc(simple_loop_t *simple, uint64_t optim) {
    compile_loop_t compile = {
        .loop_len = simple->loop_len,
        .loop_num = simple->loop_num,
        .optimizations = optim,
        .op = NULL,
        .op_num = NULL,
    };
    compile.op_num = calloc(compile.loop_len, sizeof(uint64_t));
    assert(compile.op_num);
    compile.op_cap = calloc(compile.loop_len, sizeof(uint64_t));
    assert(compile.op_cap);
    compile.op = calloc(compile.loop_len, sizeof(simple_op_t *));
    assert(compile.op);
    for(uint64_t i = 0; i < compile.loop_num; i++) {
        compile.op_num[i] = 1;
        compile.op_cap[i] = initial_cap;
        compile.op[i] = calloc(initial_cap, sizeof(simple_op_t));
        assert(compile.op[i]);
        compile.op[i][0] = simple->loop_instance[i];
    }
    compile.per_dim_off_a = calloc(compile.loop_len * 2, sizeof(uint64_t *));
    assert(compile.per_dim_off_a);
    compile.per_dim_off_z = calloc(compile.loop_len * 2, sizeof(uint64_t *));
    assert(compile.per_dim_off_z);
    compile.per_dim_off_y = calloc(compile.loop_len * 2, sizeof(uint64_t *));
    assert(compile.per_dim_off_y);
    compile.per_dim_off_x = calloc(compile.loop_len * 2, sizeof(uint64_t *));
    assert(compile.per_dim_off_x);
    compile.per_dim_str_a = calloc(compile.loop_len * 2, sizeof(uint64_t *));
    assert(compile.per_dim_str_a);
    compile.per_dim_str_z = calloc(compile.loop_len * 2, sizeof(uint64_t *));
    assert(compile.per_dim_str_z);
    compile.per_dim_str_y = calloc(compile.loop_len * 2, sizeof(uint64_t *));
    assert(compile.per_dim_str_y);
    compile.per_dim_str_x = calloc(compile.loop_len * 2, sizeof(uint64_t *));
    assert(compile.per_dim_str_x);
    compile.per_dim_wait_a = calloc(compile.loop_len * 2, sizeof(uint64_t *));
    assert(compile.per_dim_wait_a);
    compile.per_dim_wait_z = calloc(compile.loop_len * 2, sizeof(uint64_t *));
    assert(compile.per_dim_wait_z);
    compile.per_dim_wait_y = calloc(compile.loop_len * 2, sizeof(uint64_t *));
    assert(compile.per_dim_wait_y);
    compile.per_dim_wait_x = calloc(compile.loop_len * 2, sizeof(uint64_t *));
    assert(compile.per_dim_wait_x);
    compile.per_dim_reset_a = calloc(compile.loop_len * 2, sizeof(uint64_t *));
    assert(compile.per_dim_reset_a);
    compile.per_dim_reset_z = calloc(compile.loop_len * 2, sizeof(uint64_t *));
    assert(compile.per_dim_reset_z);
    compile.per_dim_reset_y = calloc(compile.loop_len * 2, sizeof(uint64_t *));
    assert(compile.per_dim_reset_y);
    compile.per_dim_reset_x = calloc(compile.loop_len * 2, sizeof(uint64_t *));
    assert(compile.per_dim_reset_x);
    for(uint64_t i = 0; i < compile.loop_len * 2; i++) {
        compile.per_dim_off_a[i] = calloc(initial_cap, sizeof(uint64_t));
        assert(compile.per_dim_off_a[i]);
        compile.per_dim_off_z[i] = calloc(initial_cap, sizeof(uint64_t));
        assert(compile.per_dim_off_z[i]);
        compile.per_dim_off_y[i] = calloc(initial_cap, sizeof(uint64_t));
        assert(compile.per_dim_off_y[i]);
        compile.per_dim_off_x[i] = calloc(initial_cap, sizeof(uint64_t));
        assert(compile.per_dim_off_x[i]);
        compile.per_dim_str_a[i] = calloc(initial_cap, sizeof(uint64_t));
        assert(compile.per_dim_str_a[i]);
        compile.per_dim_str_z[i] = calloc(initial_cap, sizeof(uint64_t));
        assert(compile.per_dim_str_z[i]);
        compile.per_dim_str_y[i] = calloc(initial_cap, sizeof(uint64_t));
        assert(compile.per_dim_str_y[i]);
        compile.per_dim_str_x[i] = calloc(initial_cap, sizeof(uint64_t));
        assert(compile.per_dim_str_x[i]);
        compile.per_dim_wait_a[i] = calloc(initial_cap, sizeof(uint64_t));
        assert(compile.per_dim_wait_a[i]);
        compile.per_dim_wait_z[i] = calloc(initial_cap, sizeof(uint64_t));
        assert(compile.per_dim_wait_z[i]);
        compile.per_dim_wait_y[i] = calloc(initial_cap, sizeof(uint64_t));
        assert(compile.per_dim_wait_y[i]);
        compile.per_dim_wait_x[i] = calloc(initial_cap, sizeof(uint64_t));
        assert(compile.per_dim_wait_x[i]);
        compile.per_dim_reset_a[i] = calloc(initial_cap, sizeof(uint64_t));
        assert(compile.per_dim_reset_a[i]);
        compile.per_dim_reset_z[i] = calloc(initial_cap, sizeof(uint64_t));
        assert(compile.per_dim_reset_z[i]);
        compile.per_dim_reset_y[i] = calloc(initial_cap, sizeof(uint64_t));
        assert(compile.per_dim_reset_y[i]);
        compile.per_dim_reset_x[i] = calloc(initial_cap, sizeof(uint64_t));
        assert(compile.per_dim_reset_x[i]);
    }

    simple_loop_print(simple, 4, 0, "");
    compile_loop_print(&compile, 4, 0, "");
    compile_loop_optimize(&compile, optim);
    compile_loop_print(&compile, 4, 0, "");

    return compile;
}
const uint64_t initial_source_size = 1024;
const uint64_t max_arg_size = 24;
const uint64_t max_index_digits = 9;
/* NOTE: Biggest I found was 131 for `max` or `min` binary ops. */
const uint64_t max_op_size = 256;
#define EXPAND_SOURCE_IF_NEEDED()                                                                                                                              \
    if(source_size - (curr - source) <= max_op_size) {                                                                                                         \
        source_size *= 2;                                                                                                                                      \
        offset = curr - source;                                                                                                                                \
        source = realloc(source, source_size);                                                                                                                 \
        assert(source);                                                                                                                                        \
        curr = source + offset;                                                                                                                                \
    }
int kernel_counter = 0;
/* TODO: Make use of multiple work-items per workgroup. */
/* Appends code for kernel that computes `simple_loop` with the specified global and local size. */
// static cl_kernel_t simple_loop_to_cl(const char *filename, simple_loop_t *simple_loop, uint64_t global_size, uint64_t local_size) {
//     // /* TODO: Support `local_size > 1`. */
//     char *func_name = calloc(1 + log10(kernel_counter + 1) + 3, sizeof(char));
//     assert(func_name);
//     snprintf(func_name, 1 + log10(kernel_counter + 1) + 3, "k%d", kernel_counter);
//     kernel_counter++;
//     assert(local_size == 1);
//     uint64_t leftover_loops = simple_loop->loop_number % global_size;
//     uint64_t assigned_loops = (simple_loop->loop_number - leftover_loops) / global_size;
//     uint64_t needed_loops;
//     if(leftover_loops) {
//         needed_loops = assigned_loops + 1;
//     } else {
//         needed_loops = assigned_loops;
//     }
//
//     uint64_t source_size = initial_source_size;
//     /* Calloc is needed here to make strlen and sprintf work. */
//     char *source = calloc(source_size, sizeof(char));
//     assert(source);
//     char *curr = source;
//     uint64_t offset;
//
//     uint64_t arg_num = 0;
//     char **args = NULL;
//     uint64_t found;
//     for(uint64_t i = 0; i < simple_loop->loop_length; i++) {
//         found = 0;
//         for(uint64_t j = 0; j < arg_num; j++) {
//             if(!strncmp(simple_loop->loop_instance[0][i].out_buffer.name, args[j], BUFFER_NAME_SIZE)) {
//                 found = 1;
//                 break;
//             }
//         }
//         if(!found) {
//             arg_num++;
//             args = realloc(args, arg_num * sizeof(char *));
//             assert(args);
//             args[arg_num - 1] = calloc(BUFFER_NAME_SIZE + 1, sizeof(char));
//             assert(args[arg_num - 1]);
//             strncpy(args[arg_num - 1], simple_loop->loop_instance[0][i].out_buffer.name, BUFFER_NAME_SIZE);
//         }
//         if(simple_loop->loop_instance[0][i].type != operation_unary) {
//             found = 0;
//             for(uint64_t j = 0; j < arg_num; j++) {
//                 if(!strncmp(simple_loop->loop_instance[0][i].in_buffer.name, args[j], BUFFER_NAME_SIZE)) {
//                     found = 1;
//                     break;
//                 }
//             }
//             if(!found) {
//                 arg_num++;
//                 args = realloc(args, arg_num * sizeof(char *));
//                 assert(args);
//                 args[arg_num - 1] = calloc(BUFFER_NAME_SIZE + 1, sizeof(char));
//                 assert(args[arg_num - 1]);
//                 strncpy(args[arg_num - 1], simple_loop->loop_instance[0][i].in_buffer.name, BUFFER_NAME_SIZE);
//             }
//         }
//     }
//
//     /* TODO: Fix this by binding loops to work groups, this way an actual useful number of work-items can get spawned by using reasonable global dimensions
//     and
//      * it will also be possible then to split the remaining, non evenly dividing, loops up with a simple if statement. Each work-groups then has to split up
//      * into the right number of work items. */
//     curr += snprintf(curr, max_op_size, "int gid0 = get_global_id(0);\nint id = gid0;\n");
//     EXPAND_SOURCE_IF_NEEDED();
//     for(uint64_t i = 0; i < needed_loops; i++) {
//         if(i) {
//             curr += snprintf(curr, max_op_size, "id += %lu;\n", assigned_loops);
//             EXPAND_SOURCE_IF_NEEDED();
//         }
//         if(i == assigned_loops) {
//             curr += snprintf(curr, max_op_size, "if(gid0 < %lu) {\n", leftover_loops);
//             EXPAND_SOURCE_IF_NEEDED();
//         }
//         for(uint64_t j = 0; j < simple_loop->loop_length; j++) {
//             switch(simple_loop->loop_instance[0][j].type) {
//                 case operation_unary: {
//                     curr += snprintf(
//                         curr, max_op_size,
//                         "int %s%luoff%lu = (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) + (((id
//                         %% %lu) / %lu) * %lu + %lu);\n", simple_loop->loop_instance[0][j].out_buffer.name, i, j, simple_loop->per_dim_reset_a[2 * j],
//                         simple_loop->per_dim_wait_a[2 * j], simple_loop->per_dim_str_a[2 * j], simple_loop->per_dim_off_a[2 * j],
//                         simple_loop->per_dim_reset_z[2 * j], simple_loop->per_dim_wait_z[2 * j], simple_loop->per_dim_str_z[2 * j],
//                         simple_loop->per_dim_off_z[2 * j], simple_loop->per_dim_reset_y[2 * j], simple_loop->per_dim_wait_y[2 * j],
//                         simple_loop->per_dim_str_y[2 * j], simple_loop->per_dim_off_y[2 * j], simple_loop->per_dim_reset_x[2 * j],
//                         simple_loop->per_dim_wait_x[2 * j], simple_loop->per_dim_str_x[2 * j], simple_loop->per_dim_off_x[2 * j]);
//                     EXPAND_SOURCE_IF_NEEDED();
//                     switch(simple_loop->loop_instance[0][j].unary_type) {
//                         case unary_add: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] += %.16lf;\n",
//                                                              simple_loop->loop_instance[0][j].out_buffer.name,
//                                                              simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                              SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 *
//                                                              j]),
//                                                                           (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                                           (x - simple_loop->per_dim_off_x[2 * j])),
//                                                              simple_loop->loop_instance[0][j].var_unary);
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case unary_subtract: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] -= %.16lf;\n",
//                                                              simple_loop->loop_instance[0][j].out_buffer.name,
//                                                              simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                              SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 *
//                                                              j]),
//                                                                           (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                                           (x - simple_loop->per_dim_off_x[2 * j])),
//                                                              simple_loop->loop_instance[0][j].var_unary);
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case unary_multiply: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] *= %.16lf;\n",
//                                                              simple_loop->loop_instance[0][j].out_buffer.name,
//                                                              simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                              SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 *
//                                                              j]),
//                                                                           (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                                           (x - simple_loop->per_dim_off_x[2 * j])),
//                                                              simple_loop->loop_instance[0][j].var_unary);
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case unary_divide: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] /= %.16lf;\n",
//                                                              simple_loop->loop_instance[0][j].out_buffer.name,
//                                                              simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                              SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 *
//                                                              j]),
//                                                                           (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                                           (x - simple_loop->per_dim_off_x[2 * j])),
//                                                              simple_loop->loop_instance[0][j].var_unary);
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case unary_exp: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu + %lu] = exp(%s[%s%luoff%lu + %lu]);\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case unary_log: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu + %lu] = log(%s[%s%luoff%lu + %lu]);\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case unary_square: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu + %lu] *= %s[%s%luoff%lu + %lu];\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case unary_sqrt: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu + %lu] = sqrt(%s[%s%luoff%lu + %lu]);\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case unary_negate: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr +=
//                                                 snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] *= -1;\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name,
//                                                          simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                          SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                                       (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                                       (x - simple_loop->per_dim_off_x[2 * j])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case unary_reciprocal: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu + %lu] = 1 / %s[%s%luoff%lu + %lu];\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case unary_max: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(curr, max_op_size, "if(%s[%s%luoff%lu + %lu] < %lf) { %s[%s%luoff%lu + %lu] = %lf; }\n",
//                                                              simple_loop->loop_instance[0][j].out_buffer.name,
//                                                              simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                              SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 *
//                                                              j]),
//                                                                           (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                                           (x - simple_loop->per_dim_off_x[2 * j])),
//                                                              simple_loop->loop_instance[0][j].var_unary, simple_loop->loop_instance[0][j].out_buffer.name,
//                                                              simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                              SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 *
//                                                              j]),
//                                                                           (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                                           (x - simple_loop->per_dim_off_x[2 * j])),
//                                                              simple_loop->loop_instance[0][j].var_unary);
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case unary_min: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(curr, max_op_size, "if(%s[%s%luoff%lu + %lu] < %lf) { %s[%s%luoff%lu + %lu] = %lf; }\n",
//                                                              simple_loop->loop_instance[0][j].out_buffer.name,
//                                                              simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                              SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 *
//                                                              j]),
//                                                                           (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                                           (x - simple_loop->per_dim_off_x[2 * j])),
//                                                              simple_loop->loop_instance[0][j].var_unary, simple_loop->loop_instance[0][j].out_buffer.name,
//                                                              simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                              SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 *
//                                                              j]),
//                                                                           (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                                           (x - simple_loop->per_dim_off_x[2 * j])),
//                                                              simple_loop->loop_instance[0][j].var_unary);
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case unary_set: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(curr, max_op_size, "%s[%s%luoff%lu + %lu] = %.16lf;\n",
//                                                              simple_loop->loop_instance[0][j].out_buffer.name,
//                                                              simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                              SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 *
//                                                              j]),
//                                                                           (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                                           (x - simple_loop->per_dim_off_x[2 * j])),
//                                                              simple_loop->loop_instance[0][j].var_unary);
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         /* Never *ever* use this for things like encryption, where the randomnes of the numbers is important! */
//                         case unary_random: {
//                             /*
//                             Actually I might not support this for OpenCL
//
//                             this isn't really random at all but it should be sufficient for initializing
//                             seed here is like the grid id or something
//                             have to make sure that 1/epsilon isn't nan or inf
//                             double random(int seed) {
//                                 double epsilon = 1e-4;
//                                 double modified = (sin(seed) / 100) + epsilon;
//                                 return(sin(1/modified));
//                             }
//                          */
//                             TODO();
//                             break;
//                         }
//                         case unary_tanh: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu + %lu] = tanh(%s[%s%luoff%lu + %lu]);\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case unary_absolute: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu + %lu] = fabs(%s[%s%luoff%lu + %lu]);\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case unary_sign: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size,
//                                                 "if(%s[%s%luoff%lu + %lu] < 0) { %s[%s%luoff%lu + %lu] = -1; } else if(%s[%s%luoff%lu + %lu] > 0) {
//                                                 %s[%s%luoff%lu + %lu] = 1; } else { %s[%s%luoff%lu + %lu] = 0; }\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                     }
//                     break;
//                 }
//                 case operation_binary: {
//                     curr += snprintf(
//                         curr, max_op_size,
//                         "int %s%luoff%lu = (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) + (((id
//                         %% %lu) / %lu) * %lu + %lu);\n", simple_loop->loop_instance[0][j].out_buffer.name, i, j, simple_loop->per_dim_reset_a[2 * j],
//                         simple_loop->per_dim_wait_a[2 * j], simple_loop->per_dim_str_a[2 * j], simple_loop->per_dim_off_a[2 * j],
//                         simple_loop->per_dim_reset_z[2 * j], simple_loop->per_dim_wait_z[2 * j], simple_loop->per_dim_str_z[2 * j],
//                         simple_loop->per_dim_off_z[2 * j], simple_loop->per_dim_reset_y[2 * j], simple_loop->per_dim_wait_y[2 * j],
//                         simple_loop->per_dim_str_y[2 * j], simple_loop->per_dim_off_y[2 * j], simple_loop->per_dim_reset_x[2 * j],
//                         simple_loop->per_dim_wait_x[2 * j], simple_loop->per_dim_str_x[2 * j], simple_loop->per_dim_off_x[2 * j]);
//                     EXPAND_SOURCE_IF_NEEDED();
//                     curr += snprintf(
//                         curr, max_op_size,
//                         "int %s%luoff%lu = (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) + (((id
//                         %% %lu) / %lu) * %lu + %lu);\n", simple_loop->loop_instance[0][j].in_buffer.name, i, j, simple_loop->per_dim_reset_a[2 * j + 1],
//                         simple_loop->per_dim_wait_a[2 * j + 1], simple_loop->per_dim_str_a[2 * j + 1], simple_loop->per_dim_off_a[2 * j + 1],
//                         simple_loop->per_dim_reset_z[2 * j + 1], simple_loop->per_dim_wait_z[2 * j + 1], simple_loop->per_dim_str_z[2 * j + 1],
//                         simple_loop->per_dim_off_z[2 * j + 1], simple_loop->per_dim_reset_y[2 * j + 1], simple_loop->per_dim_wait_y[2 * j + 1],
//                         simple_loop->per_dim_str_y[2 * j + 1], simple_loop->per_dim_off_y[2 * j + 1], simple_loop->per_dim_reset_x[2 * j + 1],
//                         simple_loop->per_dim_wait_x[2 * j + 1], simple_loop->per_dim_str_x[2 * j + 1], simple_loop->per_dim_off_x[2 * j + 1]);
//                     EXPAND_SOURCE_IF_NEEDED();
//                     switch(simple_loop->loop_instance[0][j].binary_type) {
//                         case binary_add: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu + %lu] += %s[%s%luoff%lu + %lu];\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].in_buffer, (a - simple_loop->per_dim_off_a[2 * j + 1]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j + 1]), (y - simple_loop->per_dim_off_y[2 * j + 1]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j + 1])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case binary_subtract: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu + %lu] -= %s[%s%luoff%lu + %lu];\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].in_buffer, (a - simple_loop->per_dim_off_a[2 * j + 1]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j + 1]), (y - simple_loop->per_dim_off_y[2 * j + 1]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j + 1])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case binary_multiply: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu + %lu] *= %s[%s%luoff%lu + %lu];\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].in_buffer, (a - simple_loop->per_dim_off_a[2 * j + 1]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j + 1]), (y - simple_loop->per_dim_off_y[2 * j + 1]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j + 1])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case binary_divide: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu + %lu] /= %s[%s%luoff%lu + %lu];\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].in_buffer, (a - simple_loop->per_dim_off_a[2 * j + 1]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j + 1]), (y - simple_loop->per_dim_off_y[2 * j + 1]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j + 1])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case binary_max: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size,
//                                                 "if(%s[%s%luoff%lu + %lu] < %s[%s%luoff%lu + %lu]) { %s[%s%luoff%lu + %lu] = %s[%s%luoff%lu + %lu]; }\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].in_buffer, (a - simple_loop->per_dim_off_a[2 * j + 1]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j + 1]), (y - simple_loop->per_dim_off_y[2 * j + 1]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j + 1])),
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].in_buffer, (a - simple_loop->per_dim_off_a[2 * j + 1]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j + 1]), (y - simple_loop->per_dim_off_y[2 * j + 1]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j + 1])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case binary_min: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size,
//                                                 "if(%s[%s%luoff%lu + %lu] > %s[%s%luoff%lu + %lu]) { %s[%s%luoff%lu + %lu] = %s[%s%luoff%lu + %lu]; }\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].in_buffer, (a - simple_loop->per_dim_off_a[2 * j + 1]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j + 1]), (y - simple_loop->per_dim_off_y[2 * j + 1]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j + 1])),
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].in_buffer, (a - simple_loop->per_dim_off_a[2 * j + 1]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j + 1]), (y - simple_loop->per_dim_off_y[2 * j + 1]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j + 1])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case binary_copy: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu + %lu] = %s[%s%luoff%lu + %lu];\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].in_buffer, (a - simple_loop->per_dim_off_a[2 * j + 1]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j + 1]), (y - simple_loop->per_dim_off_y[2 * j + 1]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j + 1])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case binary_add_like: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu + %lu] += %s[%s%luoff%lu];\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i, j);
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case binary_subtract_like: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu + %lu] -= %s[%s%luoff%lu];\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i, j);
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case binary_multiply_like: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu + %lu] *= %s[%s%luoff%lu];\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i, j);
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case binary_divide_like: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu + %lu] /= %s[%s%luoff%lu];\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i, j);
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case binary_max_like: {
//                             TODO()
//                             break;
//                         }
//                         case binary_min_like: {
//                             TODO();
//                             break;
//                         }
//                         case binary_copy_like: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].out_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].out_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].out_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].out_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu + %lu] = %s[%s%luoff%lu];\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].out_buffer, (a - simple_loop->per_dim_off_a[2 * j]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j]), (y - simple_loop->per_dim_off_y[2 * j]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j])),
//                                                 simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i, j);
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                     }
//                     break;
//                 }
//                 case operation_reduce: {
//                     curr += snprintf(
//                         curr, max_op_size,
//                         "int %s%luoff%lu = (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) + (((id
//                         %% %lu) / %lu) * %lu + %lu);\n", simple_loop->loop_instance[0][j].out_buffer.name, i, j, simple_loop->per_dim_reset_a[2 * j],
//                         simple_loop->per_dim_wait_a[2 * j], simple_loop->per_dim_str_a[2 * j], simple_loop->per_dim_off_a[2 * j],
//                         simple_loop->per_dim_reset_z[2 * j], simple_loop->per_dim_wait_z[2 * j], simple_loop->per_dim_str_z[2 * j],
//                         simple_loop->per_dim_off_z[2 * j], simple_loop->per_dim_reset_y[2 * j], simple_loop->per_dim_wait_y[2 * j],
//                         simple_loop->per_dim_str_y[2 * j], simple_loop->per_dim_off_y[2 * j], simple_loop->per_dim_reset_x[2 * j],
//                         simple_loop->per_dim_wait_x[2 * j], simple_loop->per_dim_str_x[2 * j], simple_loop->per_dim_off_x[2 * j]);
//                     EXPAND_SOURCE_IF_NEEDED();
//                     curr += snprintf(
//                         curr, max_op_size,
//                         "int %s%luoff%lu = (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) + (((id %% %lu) / %lu) * %lu + %lu) + (((id
//                         %% %lu) / %lu) * %lu + %lu);\n", simple_loop->loop_instance[0][j].in_buffer.name, i, j, simple_loop->per_dim_reset_a[2 * j + 1],
//                         simple_loop->per_dim_wait_a[2 * j + 1], simple_loop->per_dim_str_a[2 * j + 1], simple_loop->per_dim_off_a[2 * j + 1],
//                         simple_loop->per_dim_reset_z[2 * j + 1], simple_loop->per_dim_wait_z[2 * j + 1], simple_loop->per_dim_str_z[2 * j + 1],
//                         simple_loop->per_dim_off_z[2 * j + 1], simple_loop->per_dim_reset_y[2 * j + 1], simple_loop->per_dim_wait_y[2 * j + 1],
//                         simple_loop->per_dim_str_y[2 * j + 1], simple_loop->per_dim_off_y[2 * j + 1], simple_loop->per_dim_reset_x[2 * j + 1],
//                         simple_loop->per_dim_wait_x[2 * j + 1], simple_loop->per_dim_str_x[2 * j + 1], simple_loop->per_dim_off_x[2 * j + 1]);
//                     EXPAND_SOURCE_IF_NEEDED();
//                     switch(simple_loop->loop_instance[0][j].reduce_type) {
//                         case reduce_sum: {
//                             curr += snprintf(curr, max_op_size, "%s[%s%luoff%lu] = %lf;\n", simple_loop->loop_instance[0][j].out_buffer.name,
//                                              simple_loop->loop_instance[0][j].out_buffer.name, i, j, 0.0);
//                             EXPAND_SOURCE_IF_NEEDED();
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].in_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].in_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].in_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].in_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu] += %s[%s%luoff%lu + %lu];\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].in_buffer, (a - simple_loop->per_dim_off_a[2 * j + 1]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j + 1]), (y - simple_loop->per_dim_off_y[2 * j + 1]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j + 1])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             break;
//                         }
//                         case reduce_avg: {
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].in_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].in_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].in_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].in_buffer.x_size; x++) {
//                                             curr += snprintf(
//                                                 curr, max_op_size, "%s[%s%luoff%lu] += %s[%s%luoff%lu + %lu];\n",
//                                                 simple_loop->loop_instance[0][j].out_buffer.name, simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                                 simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i, j,
//                                                 SIMPLE_INDEX(simple_loop->loop_instance[0][j].in_buffer, (a - simple_loop->per_dim_off_a[2 * j + 1]),
//                                                              (z - simple_loop->per_dim_off_z[2 * j + 1]), (y - simple_loop->per_dim_off_y[2 * j + 1]),
//                                                              (x - simple_loop->per_dim_off_x[2 * j + 1])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             curr += snprintf(curr, max_op_size, "%s[%s%luoff%lu] /= %lf;\n", simple_loop->loop_instance[0][j].out_buffer.name,
//                                              simple_loop->loop_instance[0][j].out_buffer.name, i, j,
//                                              (double) simple_loop->loop_instance[0][j].in_buffer.a_size * simple_loop->loop_instance[0][j].in_buffer.z_size *
//                                                  simple_loop->loop_instance[0][j].in_buffer.y_size * simple_loop->loop_instance[0][j].in_buffer.x_size);
//                             EXPAND_SOURCE_IF_NEEDED();
//                             break;
//                         }
//                         case reduce_max: {
//                             curr += snprintf(curr, max_op_size, "double _%lumax%lu = -INFINITY;\n", i, j);
//                             EXPAND_SOURCE_IF_NEEDED();
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].in_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].in_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].in_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].in_buffer.x_size; x++) {
//                                             curr +=
//                                                 snprintf(curr, max_op_size, "if(%s[%s%luoff%lu + %lu] < _%lumax%lu) { _%lumax%lu = %s[%s%luoff%lu + %lu];
//                                                 }\n",
//                                                          simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i,
//                                                          j, SIMPLE_INDEX(simple_loop->loop_instance[0][j].in_buffer, (a - simple_loop->per_dim_off_a[2 * j +
//                                                          1]),
//                                                                       (z - simple_loop->per_dim_off_z[2 * j + 1]), (y - simple_loop->per_dim_off_y[2 * j +
//                                                                       1]), (x - simple_loop->per_dim_off_x[2 * j + 1])),
//                                                          i, j, i, j, simple_loop->loop_instance[0][j].in_buffer.name,
//                                                          simple_loop->loop_instance[0][j].in_buffer.name, i, j,
//                                                          SIMPLE_INDEX(simple_loop->loop_instance[0][j].in_buffer, (a - simple_loop->per_dim_off_a[2 * j +
//                                                          1]),
//                                                                       (z - simple_loop->per_dim_off_z[2 * j + 1]), (y - simple_loop->per_dim_off_y[2 * j +
//                                                                       1]), (x - simple_loop->per_dim_off_x[2 * j + 1])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             curr += snprintf(curr, max_op_size, "%s[%s%luoff%lu] = _%lumax%lu;\n", simple_loop->loop_instance[0][j].out_buffer.name,
//                                              simple_loop->loop_instance[0][j].out_buffer.name, i, j, i, j);
//                             EXPAND_SOURCE_IF_NEEDED();
//                             break;
//                         }
//                         case reduce_min: {
//                             curr += snprintf(curr, max_op_size, "double _%lumin%lu = INFINITY;\n", i, j);
//                             EXPAND_SOURCE_IF_NEEDED();
//                             for(uint64_t a = 0; a < simple_loop->loop_instance[0][j].in_buffer.a_size; a++) {
//                                 for(uint64_t z = 0; z < simple_loop->loop_instance[0][j].in_buffer.z_size; z++) {
//                                     for(uint64_t y = 0; y < simple_loop->loop_instance[0][j].in_buffer.y_size; y++) {
//                                         for(uint64_t x = 0; x < simple_loop->loop_instance[0][j].in_buffer.x_size; x++) {
//                                             curr +=
//                                                 snprintf(curr, max_op_size, "if(%s[%s%luoff%lu + %lu] > _%lumin%lu) { _%lumin%lu = %s[%s%luoff%lu + %lu];
//                                                 }\n",
//                                                          simple_loop->loop_instance[0][j].in_buffer.name, simple_loop->loop_instance[0][j].in_buffer.name, i,
//                                                          j, SIMPLE_INDEX(simple_loop->loop_instance[0][j].in_buffer, (a - simple_loop->per_dim_off_a[2 * j +
//                                                          1]),
//                                                                       (z - simple_loop->per_dim_off_z[2 * j + 1]), (y - simple_loop->per_dim_off_y[2 * j +
//                                                                       1]), (x - simple_loop->per_dim_off_x[2 * j + 1])),
//                                                          i, j, i, j, simple_loop->loop_instance[0][j].in_buffer.name,
//                                                          simple_loop->loop_instance[0][j].in_buffer.name, i, j,
//                                                          SIMPLE_INDEX(simple_loop->loop_instance[0][j].in_buffer, (a - simple_loop->per_dim_off_a[2 * j +
//                                                          1]),
//                                                                       (z - simple_loop->per_dim_off_z[2 * j + 1]), (y - simple_loop->per_dim_off_y[2 * j +
//                                                                       1]), (x - simple_loop->per_dim_off_x[2 * j + 1])));
//                                             EXPAND_SOURCE_IF_NEEDED();
//                                         }
//                                     }
//                                 }
//                             }
//                             curr += snprintf(curr, max_op_size, "%s[%s%luoff%lu] = _%lumin%lu;\n", simple_loop->loop_instance[0][j].out_buffer.name,
//                                              simple_loop->loop_instance[0][j].out_buffer.name, i, j, i, j);
//                             EXPAND_SOURCE_IF_NEEDED();
//                             break;
//                         }
//                     }
//                     break;
//                 }
//                 case operation_move: {
//                     ERROR("ERROR: Tried to compile move operation to OpenCL at index %lu\n", j);
//                 }
//             }
//         }
//         if(i == assigned_loops) {
//             curr += snprintf(curr, max_op_size, "}\n");
//             EXPAND_SOURCE_IF_NEEDED();
//         }
//     }
//
//     assert(arg_num != 0);
//     // /* This formula is very jank, but it makes sense, if you think about it except for the ` + 3` that is just magic. */
//     uint64_t kernel_size = strlen("__kernel void ") + strlen(func_name) + (strlen("__global double *") + BUFFER_NAME_SIZE) * arg_num +
//                            strlen(", ") * (arg_num - 1) + strlen(") {\n") + (curr - source) + strlen("}\n") + 3;
//     char *kernel_source = calloc(kernel_size, sizeof(char));
//     assert(kernel_source);
//     char *kernel_i = kernel_source;
//     kernel_i += sprintf(kernel_i, "__kernel void %s(", func_name);
//     for(uint64_t i = 0; i < arg_num; i++) {
//         if(i != arg_num - 1) {
//             kernel_i += sprintf(kernel_i, "__global double *%s, ", args[i]);
//         } else {
//             kernel_i += sprintf(kernel_i, "__global double *%s) {\n", args[i]);
//         }
//     }
//     /* This one is very sus. Extremely sus. Why in the world do I need to do the `+ 1` here? */
//     kernel_i += snprintf(kernel_i, curr - source + 1, "%s", source);
//     kernel_i += snprintf(kernel_i, 3, "}\n");
//
//     FILE *f = fopen(filename, "a");
//     assert(f);
//     fwrite(kernel_source, sizeof(char), kernel_i - kernel_source, f);
//     fclose(f);
//
//     cl_kernel_t kernel = {
//         .arg_num = arg_num,
//         .args = calloc(arg_num, sizeof(char *)),
//         .name = strndup(func_name, strlen(func_name)),
//         .local_size = local_size,
//         .global_size = global_size,
//     };
//     assert(kernel.args);
//     for(uint64_t i = 0; i < arg_num; i++) {
//         kernel.args[i] = strndup(args[i], BUFFER_NAME_SIZE + 1);
//         free(args[i]);
//     }
//     free(args);
//     free(source);
//     free(kernel_source);
//     free(func_name);
//     return kernel;
// }
/* Could also be called `cl_program_alloc()`. */
cl_program_t compile_linearized_to_cl(const char *filename, linearized_t *linearized) {
    cl_program_t program = {
        .filename = filename,
        .kernel_num = 0,
        .kernel = NULL,
    };

    simple_loop_t simple = {0};
    // uint64_t global_size = 4;
    // uint64_t local_size = 1;
    simple_loop_from_linearized_index(&simple, linearized, 0);
    compile_loop_t compile = compile_loop_alloc(&simple, OPTIMIZE_INLINE);
    /* Clears file. */
    // FILE *f = fopen(filename, "w");
    // fclose(f);
    // uint64_t i = 0;
    // cl_kernel_t kernel;
    // while(i < linearized->op_count) {
    //     i += simple_loop_from_linearized_index(&simple_loop, linearized, i);
    //     // simple_loop_print(&simple_loop, 4, 0, "");
    //     kernel = simple_loop_to_cl(filename, &simple_loop, global_size, local_size);
    //     program.kernel_num++;
    //     program.kernel = realloc(program.kernel, program.kernel_num * sizeof(cl_kernel_t));
    //     assert(program.kernel);
    //     program.kernel[program.kernel_num - 1] = kernel;
    // }

    compile_loop_free(&compile);
    simple_loop_free(&simple);
    return program;
}
void cl_program_free(cl_program_t *program) {
    for(uint64_t i = 0; i < program->kernel_num; i++) { cl_kernel_free(&program->kernel[i]); }
    free(program->kernel);
}
