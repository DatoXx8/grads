#include <CL/cl.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"

static char name[BUFFER_NAME_SIZE + 1] = {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
                                          'a', 'a', 'a', 'a', 'a', 'a', 'a', '\0'};
static uint64_t name_off = 0;
static void name_update(void) {
    for(uint64_t i = 0; i < BUFFER_NAME_SIZE; i++) {
        assert(name[i] >= 'a' && name[i] <= 'z');
        assert(name_off >= 0 && name_off <= INT64_MAX);
        if(name[i] != 'z') {
            name[i]++;
            name_off++;
            return;
        } else {
            assert(i < BUFFER_NAME_SIZE - 1); /* This would be a wrap around back to "aaa..." */
            name[i] = 'a';
            name_off++;
        }
    }
}
void buffer_sync_realize(buffer_t *buffer, cl_command_queue command_queue) {
    assert(command_queue);
    assert(buffer);
    switch(buffer->sync) {
        case sync_none: {
            break;
        }
        case sync_to_device: {
            uint64_t size = buffer->a_inh * buffer->z_inh * buffer->y_inh * buffer->x_inh * sizeof(double);
            clEnqueueWriteBuffer(command_queue, buffer->val_cl, CL_TRUE, 0, size, buffer->val, 0, NULL, NULL);
            buffer->sync = sync_none;
            break;
        }
        case sync_to_host: {
            uint64_t size = buffer->a_inh * buffer->z_inh * buffer->y_inh * buffer->x_inh * sizeof(double);
            clEnqueueReadBuffer(command_queue, buffer->val_cl, CL_TRUE, 0, size, buffer->val, 0, NULL, NULL);
            buffer->sync = sync_none;
            break;
        }
    }
}
void buffer_sync_update(buffer_t *buffer, const sync_e sync) {
    if(buffer->sync == sync_none) {
        assert(sync != sync_none);
        buffer->sync = sync;
    } else {
        assert(buffer->sync == sync);
    }
}

buffer_t buffer_alloc(const uint64_t a, const uint64_t z, const uint64_t y, const uint64_t x, cl_context context) {
    assert(a > 0);
    assert(z > 0);
    assert(y > 0);
    assert(x > 0);
    buffer_t buffer = {
        .a_inh = a,
        .z_inh = z,
        .y_inh = y,
        .x_inh = x,
        .a_sze = a,
        .z_sze = z,
        .y_sze = y,
        .x_sze = x,
        .a_str = z * y * x,
        .z_str = y * x,
        .y_str = x,
        .x_str = 1,
        .val = calloc(a * z * y * x, sizeof(double)),
        .val_cl = NULL,
        .sync = sync_none,
        .name_off = name_off,
    };
    if(context) {
        int err;
        buffer.val_cl = clCreateBuffer(context, CL_MEM_READ_WRITE, a * z * y * x * sizeof(double), NULL, &err);
        assert(!err);
    }
    assert(buffer.val);
    strncpy(buffer.name, name, BUFFER_NAME_SIZE + 1);
    name_update();
    return buffer;
}
void buffer_free(buffer_t *buffer) {
    free(buffer->val);
    buffer->val = NULL;
    if(buffer->val_cl) {
        clReleaseMemObject(buffer->val_cl);
        buffer->val_cl = NULL;
    }
}

void op_print(const op_t *op, const int padding, const int offset, const char *name) {
    if(strncmp(name, "", 1) != 0) {
        printf("%*s%s\n", offset, "", name);
    }
    printf("%*s<%p> ", offset + padding, "", (void *) op);
    switch(op->type_op) {
        case op_unary: {
            switch(op->type_unary) {
                case unary_add: {
                    printf("U add {%lu, %lu, %lu, %lu} %lu %lf [%s]\n", op->buffer_out.a_sze, op->buffer_out.z_sze,
                           op->buffer_out.y_sze, op->buffer_out.x_sze, op->buffer_out.off, op->u_var,
                           op->buffer_out.name);
                    break;
                }
                case unary_subtract: {
                    printf("U sub {%lu, %lu, %lu, %lu} %lu %lf [%s]\n", op->buffer_out.a_sze, op->buffer_out.z_sze,
                           op->buffer_out.y_sze, op->buffer_out.x_sze, op->buffer_out.off, op->u_var,
                           op->buffer_out.name);
                    break;
                }
                case unary_multiply: {
                    printf("U mul {%lu, %lu, %lu, %lu} %lu %lf [%s]\n", op->buffer_out.a_sze, op->buffer_out.z_sze,
                           op->buffer_out.y_sze, op->buffer_out.x_sze, op->buffer_out.off, op->u_var,
                           op->buffer_out.name);
                    break;
                }
                case unary_divide: {
                    printf("U div {%lu, %lu, %lu, %lu} %lu %lf [%s]\n", op->buffer_out.a_sze, op->buffer_out.z_sze,
                           op->buffer_out.y_sze, op->buffer_out.x_sze, op->buffer_out.off, op->u_var,
                           op->buffer_out.name);
                    break;
                }
                case unary_exp: {
                    printf("U exp {%lu, %lu, %lu, %lu} %lu [%s]\n", op->buffer_out.a_sze, op->buffer_out.z_sze,
                           op->buffer_out.y_sze, op->buffer_out.x_sze, op->buffer_out.off, op->buffer_out.name);
                    break;
                }
                case unary_log: {
                    printf("U log {%lu, %lu, %lu, %lu} %lu [%s]\n", op->buffer_out.a_sze, op->buffer_out.z_sze,
                           op->buffer_out.y_sze, op->buffer_out.x_sze, op->buffer_out.off, op->buffer_out.name);
                    break;
                }
                case unary_square: {
                    printf("U sqr {%lu, %lu, %lu, %lu} %lu [%s]\n", op->buffer_out.a_sze, op->buffer_out.z_sze,
                           op->buffer_out.y_sze, op->buffer_out.x_sze, op->buffer_out.off, op->buffer_out.name);
                    break;
                }
                case unary_sqrt: {
                    printf("U sqt {%lu, %lu, %lu, %lu} %lu [%s]\n", op->buffer_out.a_sze, op->buffer_out.z_sze,
                           op->buffer_out.y_sze, op->buffer_out.x_sze, op->buffer_out.off, op->buffer_out.name);
                    break;
                }
                case unary_reciprocal: {
                    printf("U rcp {%lu, %lu, %lu, %lu} %lu [%s]\n", op->buffer_out.a_sze, op->buffer_out.z_sze,
                           op->buffer_out.y_sze, op->buffer_out.x_sze, op->buffer_out.off, op->buffer_out.name);
                    break;
                }
                case unary_max: {
                    printf("U max {%lu, %lu, %lu, %lu} %lu %lf [%s]\n", op->buffer_out.a_sze, op->buffer_out.z_sze,
                           op->buffer_out.y_sze, op->buffer_out.x_sze, op->buffer_out.off, op->u_var,
                           op->buffer_out.name);
                    break;
                }
                case unary_min: {
                    printf("U min {%lu, %lu, %lu, %lu} %lu %lf [%s]\n", op->buffer_out.a_sze, op->buffer_out.z_sze,
                           op->buffer_out.y_sze, op->buffer_out.x_sze, op->buffer_out.off, op->u_var,
                           op->buffer_out.name);
                    break;
                }
                case unary_set: {
                    printf("U set {%lu, %lu, %lu, %lu} %lu %lf [%s]\n", op->buffer_out.a_sze, op->buffer_out.z_sze,
                           op->buffer_out.y_sze, op->buffer_out.x_sze, op->buffer_out.off, op->u_var,
                           op->buffer_out.name);
                    break;
                }
                case unary_random: {
                    printf("U ran {%lu, %lu, %lu, %lu} %lu [%s]\n", op->buffer_out.a_sze, op->buffer_out.z_sze,
                           op->buffer_out.y_sze, op->buffer_out.x_sze, op->buffer_out.off, op->buffer_out.name);
                    break;
                }
                case unary_tanh: {
                    printf("U tnh {%lu, %lu, %lu, %lu} %lu [%s]\n", op->buffer_out.a_sze, op->buffer_out.z_sze,
                           op->buffer_out.y_sze, op->buffer_out.x_sze, op->buffer_out.off, op->buffer_out.name);
                    break;
                }
                case unary_absolute: {
                    printf("U abs {%lu, %lu, %lu, %lu} %lu [%s]\n", op->buffer_out.a_sze, op->buffer_out.z_sze,
                           op->buffer_out.y_sze, op->buffer_out.x_sze, op->buffer_out.off, op->buffer_out.name);
                    break;
                }
                case unary_sign: {
                    printf("U sgn {%lu, %lu, %lu, %lu} %lu [%s]\n", op->buffer_out.a_sze, op->buffer_out.z_sze,
                           op->buffer_out.y_sze, op->buffer_out.x_sze, op->buffer_out.off, op->buffer_out.name);
                    break;
                }
            }
            break;
        }
        case op_binary: {
            switch(op->type_binary) {
                case binary_add: {
                    printf("B add {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
                case binary_subtract: {
                    printf("B sub {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
                case binary_multiply: {
                    printf("B mul {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
                case binary_divide: {
                    printf("B div {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
                case binary_max: {
                    printf("B max {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
                case binary_min: {
                    printf("B min {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
                case binary_copy: {
                    printf("B cpy {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
                case binary_add_like: {
                    printf("L add {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
                case binary_subtract_like: {
                    printf("L sub {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
                case binary_multiply_like: {
                    printf("L mul {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
                case binary_divide_like: {
                    printf("L div {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
                case binary_max_like: {
                    printf("L max {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
                case binary_min_like: {
                    printf("L min {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
                case binary_copy_like: {
                    printf("L cpy {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
            }
            break;
        }
        case op_reduce: {
            switch(op->type_reduce) {
                case reduce_sum: {
                    printf("R sum {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
                case reduce_avg: {
                    printf("R avg {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
                case reduce_max: {
                    printf("R max {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
                case reduce_min: {
                    printf("R min {%lu, %lu, %lu, %lu} %lu < {%lu, %lu, %lu, %lu} %lu [%s] [%s]\n",
                           op->buffer_out.a_sze, op->buffer_out.z_sze, op->buffer_out.y_sze, op->buffer_out.x_sze,
                           op->buffer_out.off, op->buffer_in.a_sze, op->buffer_in.z_sze, op->buffer_in.y_sze,
                           op->buffer_in.x_sze, op->buffer_in.off, op->buffer_out.name, op->buffer_in.name);
                    break;
                }
            }
            break;
        }
    }
}
/* TODO: Not sure this should be const since we are modifying the state of the buffers in the op */
void op_realize(const op_t *op) {
    switch(op->type_op) {
        case op_unary: {
            switch(op->type_unary) {
                case unary_add: {
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) += op->u_var;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_subtract: {
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) -= op->u_var;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_multiply: {
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) *= op->u_var;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_divide: {
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) /= op->u_var;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_exp: {
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) = exp(BUFFER_AT(op->buffer_out, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_log: {
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) = log(BUFFER_AT(op->buffer_out, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_square: {
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) *= BUFFER_AT(op->buffer_out, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_sqrt: {
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) = sqrt(BUFFER_AT(op->buffer_out, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_reciprocal: {
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) = 1 / BUFFER_AT(op->buffer_out, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_max: {
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) =
                                        fmax(BUFFER_AT(op->buffer_out, a, z, y, x), op->u_var);
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_min: {
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) =
                                        fmin(BUFFER_AT(op->buffer_out, a, z, y, x), op->u_var);
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_set: {
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) = op->u_var;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_random: {
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) = ((double) rand() / RAND_MAX) * 2 - 1;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_tanh: {
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) = tanh(BUFFER_AT(op->buffer_out, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_absolute: {
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) = fabs(BUFFER_AT(op->buffer_out, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_sign: {
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    if(BUFFER_AT(op->buffer_out, a, z, y, x) < 0) {
                                        BUFFER_AT(op->buffer_out, a, z, y, x) = -1;
                                    } else if(BUFFER_AT(op->buffer_out, a, z, y, x) == 0) {
                                        BUFFER_AT(op->buffer_out, a, z, y, x) = 0;
                                    } else {
                                        BUFFER_AT(op->buffer_out, a, z, y, x) = 1;
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
            }
            break;
        }
        case op_binary: {
            switch(op->type_binary) {
                case binary_add: {
                    assert(op->buffer_out.a_sze == op->buffer_in.a_sze);
                    assert(op->buffer_out.z_sze == op->buffer_in.z_sze);
                    assert(op->buffer_out.y_sze == op->buffer_in.y_sze);
                    assert(op->buffer_out.x_sze == op->buffer_in.x_sze);
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) += BUFFER_AT(op->buffer_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_subtract: {
                    assert(op->buffer_out.a_sze == op->buffer_in.a_sze);
                    assert(op->buffer_out.z_sze == op->buffer_in.z_sze);
                    assert(op->buffer_out.y_sze == op->buffer_in.y_sze);
                    assert(op->buffer_out.x_sze == op->buffer_in.x_sze);
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) -= BUFFER_AT(op->buffer_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_multiply: {
                    assert(op->buffer_out.a_sze == op->buffer_in.a_sze);
                    assert(op->buffer_out.z_sze == op->buffer_in.z_sze);
                    assert(op->buffer_out.y_sze == op->buffer_in.y_sze);
                    assert(op->buffer_out.x_sze == op->buffer_in.x_sze);
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) *= BUFFER_AT(op->buffer_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_divide: {
                    assert(op->buffer_out.a_sze == op->buffer_in.a_sze);
                    assert(op->buffer_out.z_sze == op->buffer_in.z_sze);
                    assert(op->buffer_out.y_sze == op->buffer_in.y_sze);
                    assert(op->buffer_out.x_sze == op->buffer_in.x_sze);
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) /= BUFFER_AT(op->buffer_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_max: {
                    assert(op->buffer_out.a_sze == op->buffer_in.a_sze);
                    assert(op->buffer_out.z_sze == op->buffer_in.z_sze);
                    assert(op->buffer_out.y_sze == op->buffer_in.y_sze);
                    assert(op->buffer_out.x_sze == op->buffer_in.x_sze);
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) = fmax(BUFFER_AT(op->buffer_out, a, z, y, x),
                                                                                 BUFFER_AT(op->buffer_in, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_min: {
                    assert(op->buffer_out.a_sze == op->buffer_in.a_sze);
                    assert(op->buffer_out.z_sze == op->buffer_in.z_sze);
                    assert(op->buffer_out.y_sze == op->buffer_in.y_sze);
                    assert(op->buffer_out.x_sze == op->buffer_in.x_sze);
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) = fmin(BUFFER_AT(op->buffer_out, a, z, y, x),
                                                                                 BUFFER_AT(op->buffer_in, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_copy: {
                    assert(op->buffer_out.a_sze == op->buffer_in.a_sze);
                    assert(op->buffer_out.z_sze == op->buffer_in.z_sze);
                    assert(op->buffer_out.y_sze == op->buffer_in.y_sze);
                    assert(op->buffer_out.x_sze == op->buffer_in.x_sze);
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) = BUFFER_AT(op->buffer_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_add_like: {
                    assert(op->buffer_in.a_sze == 1);
                    assert(op->buffer_in.z_sze == 1);
                    assert(op->buffer_in.y_sze == 1);
                    assert(op->buffer_in.x_sze == 1);
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) += BUFFER_AT(op->buffer_in, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_subtract_like: {
                    assert(op->buffer_in.a_sze == 1);
                    assert(op->buffer_in.z_sze == 1);
                    assert(op->buffer_in.y_sze == 1);
                    assert(op->buffer_in.x_sze == 1);
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) -= BUFFER_AT(op->buffer_in, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_multiply_like: {
                    assert(op->buffer_in.a_sze == 1);
                    assert(op->buffer_in.z_sze == 1);
                    assert(op->buffer_in.y_sze == 1);
                    assert(op->buffer_in.x_sze == 1);
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) *= BUFFER_AT(op->buffer_in, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_divide_like: {
                    assert(op->buffer_in.a_sze == 1);
                    assert(op->buffer_in.z_sze == 1);
                    assert(op->buffer_in.y_sze == 1);
                    assert(op->buffer_in.x_sze == 1);
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) /= BUFFER_AT(op->buffer_in, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_max_like: {
                    assert(op->buffer_in.a_sze == 1);
                    assert(op->buffer_in.z_sze == 1);
                    assert(op->buffer_in.y_sze == 1);
                    assert(op->buffer_in.x_sze == 1);
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) = fmax(BUFFER_AT(op->buffer_out, a, z, y, x),
                                                                                 BUFFER_AT(op->buffer_in, 0, 0, 0, 0));
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_min_like: {
                    assert(op->buffer_in.a_sze == 1);
                    assert(op->buffer_in.z_sze == 1);
                    assert(op->buffer_in.y_sze == 1);
                    assert(op->buffer_in.x_sze == 1);
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) = fmin(BUFFER_AT(op->buffer_out, a, z, y, x),
                                                                                 BUFFER_AT(op->buffer_in, 0, 0, 0, 0));
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_copy_like: {
                    assert(op->buffer_in.a_sze == 1);
                    assert(op->buffer_in.z_sze == 1);
                    assert(op->buffer_in.y_sze == 1);
                    assert(op->buffer_in.x_sze == 1);
                    for(uint64_t a = 0; a < op->buffer_out.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_out.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_out.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_out.x_sze; x++) {
                                    BUFFER_AT(op->buffer_out, a, z, y, x) = BUFFER_AT(op->buffer_in, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
            }
            break;
        }
        case op_reduce: {
            switch(op->type_reduce) {
                case reduce_sum: {
                    assert(op->buffer_out.a_sze == 1);
                    assert(op->buffer_out.z_sze == 1);
                    assert(op->buffer_out.y_sze == 1);
                    assert(op->buffer_out.x_sze == 1);
                    double temp = 0;
                    for(uint64_t a = 0; a < op->buffer_in.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_in.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_in.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_in.x_sze; x++) {
                                    temp += BUFFER_AT(op->buffer_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    BUFFER_AT(op->buffer_out, 0, 0, 0, 0) = temp;
                    break;
                }
                case reduce_max: {
                    assert(op->buffer_out.a_sze == 1);
                    assert(op->buffer_out.z_sze == 1);
                    assert(op->buffer_out.y_sze == 1);
                    assert(op->buffer_out.x_sze == 1);
                    double temp = -INFINITY;
                    for(uint64_t a = 0; a < op->buffer_in.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_in.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_in.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_in.x_sze; x++) {
                                    temp = fmax(temp, BUFFER_AT(op->buffer_in, a, z, y, x));
                                }
                            }
                        }
                    }
                    BUFFER_AT(op->buffer_out, 0, 0, 0, 0) = temp;
                    break;
                }
                case reduce_avg: {
                    assert(op->buffer_out.a_sze == 1);
                    assert(op->buffer_out.z_sze == 1);
                    assert(op->buffer_out.y_sze == 1);
                    assert(op->buffer_out.x_sze == 1);
                    double temp = 0;
                    for(uint64_t a = 0; a < op->buffer_in.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_in.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_in.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_in.x_sze; x++) {
                                    temp += BUFFER_AT(op->buffer_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    BUFFER_AT(op->buffer_out, 0, 0, 0, 0) =
                        temp / (op->buffer_in.a_sze * op->buffer_in.z_sze * op->buffer_in.y_sze * op->buffer_in.x_sze);
                    break;
                }
                case reduce_min: {
                    assert(op->buffer_out.a_sze == 1);
                    assert(op->buffer_out.z_sze == 1);
                    assert(op->buffer_out.y_sze == 1);
                    assert(op->buffer_out.x_sze == 1);
                    double temp = INFINITY;
                    for(uint64_t a = 0; a < op->buffer_in.a_sze; a++) {
                        for(uint64_t z = 0; z < op->buffer_in.z_sze; z++) {
                            for(uint64_t y = 0; y < op->buffer_in.y_sze; y++) {
                                for(uint64_t x = 0; x < op->buffer_in.x_sze; x++) {
                                    temp = fmin(temp, BUFFER_AT(op->buffer_in, a, z, y, x));
                                }
                            }
                        }
                    }
                    BUFFER_AT(op->buffer_out, 0, 0, 0, 0) = temp;
                    break;
                }
            }
            break;
        }
    }
}

/* Completely made up value. No reasoning behind it at all */
const uint64_t INITIAL_OP_CAP = 25;
linearized_t linearized_alloc(void) {
    linearized_t linearized = {
        .op_len = 0,
        .op_cap = INITIAL_OP_CAP,
        .op = calloc(INITIAL_OP_CAP, sizeof(op_t)),
    };
    assert(linearized.op);

    return linearized;
}
void linearized_free(linearized_t *linearized) {
    assert(linearized);
    free(linearized->op);
    linearized->op = NULL;
    linearized->op_len = 0;
    linearized->op_cap = 0;
}
void linearized_clear(linearized_t *linearized) {
    assert(linearized);
    linearized->op_len = 0;
}
void linearized_run(const linearized_t *linearized) {
    assert(linearized);
    for(uint64_t op_idx = 0; op_idx < linearized->op_len; op_idx++) {
        op_realize(&linearized->op[op_idx]);
    }
}
void linearized_add_op(linearized_t *linearized, const op_t *op) {
    assert(linearized);
    assert(op);
    linearized->op_len++;
    if(linearized->op_len >= linearized->op_cap) {
        linearized->op_cap *= 2;
        linearized->op = reallocarray(linearized->op, linearized->op_cap, sizeof(op_t));
    }
    linearized->op[linearized->op_len - 1] = *op;
}
void linearized_append(linearized_t *linearized1, linearized_t *linearized2) {
    while(linearized1->op_len + linearized2->op_len >= linearized1->op_cap) {
        linearized1->op_cap *= 2;
        linearized1->op = reallocarray(linearized1->op, linearized1->op_cap, sizeof(op_t));
        assert(linearized1->op);
    }
    for(uint64_t op_idx = 0; op_idx < linearized2->op_len; op_idx++) {
        linearized1->op[linearized1->op_len + op_idx] = linearized2->op[op_idx];
    }
    linearized1->op_len += linearized2->op_len;
    linearized_clear(linearized2);
}
void linearized_print(const linearized_t *linearized, const int padding, const int offset, const char *name) {
    assert(linearized);
    if(linearized == NULL) {
        return;
    }
    if(strncmp(name, "", 1) != 0) {
        printf("%*slen %lu, cap %lu %s\n", offset, "", linearized->op_len, linearized->op_cap, name);
    } else {
        printf("%*slen %lu, cap %lu\n", offset, "", linearized->op_len, linearized->op_cap);
    }
    if(!linearized->op_len) {
        printf("%*sEmpty\n", padding + offset, "");
    }
    /* Kind of a nice allignment for printing */
    // uint64_t max = log10(linearized->op_count);
    // for(uint64_t i = 0; i < linearized->op_count; i++) {
    //     printf("%*s[%*s%lu] ", padding + offset, "", (int) (max - (uint64_t) log10(i)), "", i);
    //     op_print(linearized->simple + i, 0, 0, "");
    // }
    /* This one is not alligned */
    for(uint64_t i = 0; i < linearized->op_len; i++) {
        printf("%*s[%lu] ", padding + offset, "", i);
        op_print(&linearized->op[i], 0, 0, "");
    }
}

tensor_t tensor_alloc(const uint64_t a, const uint64_t z, const uint64_t y, const uint64_t x, cl_context context) {
    assert(a > 0);
    assert(z > 0);
    assert(y > 0);
    assert(x > 0);
    tensor_t tensor = {
        .buffer = calloc(1, sizeof(buffer_t)),
        .linearized = calloc(1, sizeof(linearized_t)),
    };
    assert(tensor.buffer);
    assert(tensor.linearized);
    *tensor.buffer = buffer_alloc(a, z, y, x, context);
    *tensor.linearized = linearized_alloc();
    return tensor;
}
void tensor_free(tensor_t *tensor) {
    assert(tensor);
    assert(tensor->buffer);
    if(tensor->buffer) {
        buffer_free(tensor->buffer);
        free(tensor->buffer);
        tensor->buffer = NULL;
    }
    if(tensor->linearized) {
        linearized_free(tensor->linearized);
        free(tensor->linearized);
        tensor->linearized = NULL;
    }
}

void tensor_unary_add(tensor_t *tensor, const double value) {
    assert(tensor);
    op_t new = {
        .type_op = op_unary,
        .type_unary = unary_add,
        .u_var = value,
        .buffer_out = *tensor->buffer,
    };
    linearized_add_op(tensor->linearized, &new);
}
void tensor_unary_subtract(tensor_t *tensor, const double value) {
    assert(tensor);
    op_t new = {
        .type_op = op_unary,
        .type_unary = unary_subtract,
        .u_var = value,
        .buffer_out = *tensor->buffer,
    };
    linearized_add_op(tensor->linearized, &new);
}
void tensor_unary_multiply(tensor_t *tensor, const double value) {
    assert(tensor);
    op_t new = {
        .type_op = op_unary,
        .type_unary = unary_multiply,
        .u_var = value,
        .buffer_out = *tensor->buffer,
    };
    linearized_add_op(tensor->linearized, &new);
}
void tensor_unary_divide(tensor_t *tensor, const double value) {
    assert(tensor);
    assert(value != 0);
    op_t new = {
        .type_op = op_unary,
        .type_unary = unary_divide,
        .u_var = value,
        .buffer_out = *tensor->buffer,
    };
    linearized_add_op(tensor->linearized, &new);
}
void tensor_unary_set(tensor_t *tensor, const double value) {
    assert(tensor);
    op_t new = {
        .type_op = op_unary,
        .type_unary = unary_set,
        .u_var = value,
        .buffer_out = *tensor->buffer,
    };
    linearized_add_op(tensor->linearized, &new);
}
void tensor_unary_exp(tensor_t *tensor) {
    assert(tensor);
    op_t new = {
        .type_op = op_unary,
        .type_unary = unary_exp,
        .buffer_out = *tensor->buffer,
    };
    linearized_add_op(tensor->linearized, &new);
}
void tensor_unary_log(tensor_t *tensor) {
    assert(tensor);
    op_t new = {
        .type_op = op_unary,
        .type_unary = unary_log,
        .buffer_out = *tensor->buffer,
    };
    linearized_add_op(tensor->linearized, &new);
}
void tensor_unary_square(tensor_t *tensor) {
    assert(tensor);
    op_t new = {
        .type_op = op_unary,
        .type_unary = unary_square,
        .buffer_out = *tensor->buffer,
    };
    linearized_add_op(tensor->linearized, &new);
}
void tensor_unary_sqrt(tensor_t *tensor) {
    assert(tensor);
    op_t new = {
        .type_op = op_unary,
        .type_unary = unary_sqrt,
        .buffer_out = *tensor->buffer,
    };
    linearized_add_op(tensor->linearized, &new);
}
void tensor_unary_reciprocal(tensor_t *tensor) {
    assert(tensor);
    op_t new = {
        .type_op = op_unary,
        .type_unary = unary_reciprocal,
        .buffer_out = *tensor->buffer,
    };
    linearized_add_op(tensor->linearized, &new);
}
void tensor_unary_random(tensor_t *tensor) {
    assert(tensor);
    op_t new = {
        .type_op = op_unary,
        .type_unary = unary_random,
        .buffer_out = *tensor->buffer,
    };
    linearized_add_op(tensor->linearized, &new);
}
void tensor_unary_tanh(tensor_t *tensor) {
    assert(tensor);
    op_t new = {
        .type_op = op_unary,
        .type_unary = unary_tanh,
        .buffer_out = *tensor->buffer,
    };
    linearized_add_op(tensor->linearized, &new);
}
void tensor_unary_max(tensor_t *tensor, const double value) {
    assert(tensor);
    op_t new = {
        .type_op = op_unary,
        .type_unary = unary_max,
        .u_var = value,
        .buffer_out = *tensor->buffer,
    };
    linearized_add_op(tensor->linearized, &new);
}
void tensor_unary_min(tensor_t *tensor, const double value) {
    assert(tensor);
    op_t new = {
        .type_op = op_unary,
        .type_unary = unary_min,
        .u_var = value,
        .buffer_out = *tensor->buffer,
    };
    linearized_add_op(tensor->linearized, &new);
}
void tensor_unary_absolute(tensor_t *tensor) {
    assert(tensor);
    op_t new = {
        .type_op = op_unary,
        .type_unary = unary_absolute,
        .buffer_out = *tensor->buffer,
    };
    linearized_add_op(tensor->linearized, &new);
}
void tensor_unary_sign(tensor_t *tensor) {
    assert(tensor);
    op_t new = {
        .type_op = op_unary,
        .type_unary = unary_sign,
        .buffer_out = *tensor->buffer,
    };
    linearized_add_op(tensor->linearized, &new);
}

void tensor_binary_add(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_binary,
        .type_binary = binary_add,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}
void tensor_binary_subtract(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_binary,
        .type_binary = binary_subtract,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}
void tensor_binary_multiply(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_binary,
        .type_binary = binary_multiply,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}
void tensor_binary_divide(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_binary,
        .type_binary = binary_divide,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}
void tensor_binary_max(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_binary,
        .type_binary = binary_max,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}
void tensor_binary_min(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_binary,
        .type_binary = binary_min,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}
void tensor_binary_copy(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_binary,
        .type_binary = binary_copy,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}
void tensor_lbinary_add(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_binary,
        .type_binary = binary_add_like,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}
void tensor_lbinary_subtract(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_binary,
        .type_binary = binary_subtract_like,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}
void tensor_lbinary_multiply(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_binary,
        .type_binary = binary_multiply_like,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}
void tensor_lbinary_divide(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_binary,
        .type_binary = binary_divide_like,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}
void tensor_lbinary_max(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_binary,
        .type_binary = binary_max_like,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}
void tensor_lbinary_min(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_binary,
        .type_binary = binary_min_like,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}
void tensor_lbinary_copy(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_binary,
        .type_binary = binary_copy_like,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}

void tensor_reduce_sum(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_reduce,
        .type_reduce = reduce_sum,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}
void tensor_reduce_avg(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_reduce,
        .type_reduce = reduce_avg,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}
void tensor_reduce_min(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_reduce,
        .type_reduce = reduce_min,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}
void tensor_reduce_max(tensor_t *out, tensor_t *in) {
    assert(out);
    assert(in);
    op_t new = {
        .type_op = op_reduce,
        .type_reduce = reduce_max,
        .buffer_out = *out->buffer,
        .buffer_in = *in->buffer,
    };
    linearized_append(out->linearized, in->linearized);
    linearized_add_op(out->linearized, &new);
}

void tensor_move_resize(tensor_t *tensor, const uint64_t a, const uint64_t z, const uint64_t y, const uint64_t x) {
    assert(tensor);
    assert(a > 0);
    assert(z > 0);
    assert(y > 0);
    assert(x > 0);
    tensor->buffer->a_sze = a;
    tensor->buffer->z_sze = z;
    tensor->buffer->y_sze = y;
    tensor->buffer->x_sze = x;
}
void tensor_move_reshape(tensor_t *tensor, const uint64_t a, const uint64_t z, const uint64_t y, const uint64_t x) {
    assert(tensor);
    tensor->buffer->a_sze = a;
    tensor->buffer->z_sze = z;
    tensor->buffer->y_sze = y;
    tensor->buffer->x_sze = x;
    tensor->buffer->a_str = z * y * x;
    tensor->buffer->z_str = y * x;
    tensor->buffer->y_str = x;
}
void tensor_move_offset(tensor_t *tensor, const uint64_t a, const uint64_t z, const uint64_t y, const uint64_t x) {
    assert(tensor);
    assert(a >= 0);
    assert(z >= 0);
    assert(y >= 0);
    assert(x >= 0);
    tensor->buffer->a_off = a;
    tensor->buffer->z_off = z;
    tensor->buffer->y_off = y;
    tensor->buffer->x_off = x;
    tensor->buffer->off =
        tensor->buffer->a_str * a + tensor->buffer->z_str * z + tensor->buffer->y_str * y + tensor->buffer->x_str * x;
}

void tensor_realize(tensor_t *tensor) {
    assert(tensor);
    if(tensor->buffer->val_cl) {
        buffer_sync_update(tensor->buffer, sync_to_device);
    }
    linearized_run(tensor->linearized);
    linearized_clear(tensor->linearized);
}

void tensor_print(const tensor_t *tensor, const int padding, const int offset, const char *name) {
    assert(tensor);
    if(strncmp(name, "", 1) != 0) {
        printf("%*s%s NAME: %s %lu %u\n", offset, "", name, tensor->buffer->name, tensor->buffer->name_off,
               tensor->buffer->sync);
    } else {
        printf("%*sNAME: %s %lu %u\n", offset, "", tensor->buffer->name, tensor->buffer->name_off,
               tensor->buffer->sync);
    }
    for(uint64_t a = 0; a < tensor->buffer->a_sze; a++) {
        if(a) {
            printf("\n");
            printf("\n");
        }
        for(uint64_t z = 0; z < tensor->buffer->z_sze; z++) {
            if(z) {
                printf("\n");
            }
            for(uint64_t y = 0; y < tensor->buffer->y_sze; y++) {
                printf("%*s[ ", offset + padding, "");
                for(uint64_t x = 0; x < tensor->buffer->x_sze; x++) {
                    printf("% lf ", BUFFER_AT_(tensor->buffer, a, z, y, x));
                }
                printf("]\n");
            }
        }
    }
}
const uint64_t A_MAX = 2;
const uint64_t Z_MAX = 2;
const uint64_t Y_MAX = 4;
const uint64_t X_MAX = 4;
/* Just prints a `{2, 2, 4, 4}` subsection of the tensor. If name is `""` it doesn't print a new empty line where
 * the name would have been */
void tensor_preview(const tensor_t *tensor, const int padding, const int offset, const char *name) {
    assert(tensor);
    if(strncmp(name, "", 1) != 0) {
        printf("%*s%s NAME: %s %lu %u\n", offset, "", name, tensor->buffer->name, tensor->buffer->name_off,
               tensor->buffer->sync);
    } else {
        printf("%*sNAME: %s %lu %u\n", offset, "", tensor->buffer->name, tensor->buffer->name_off,
               tensor->buffer->sync);
    }
    for(uint64_t a = 0; a < tensor->buffer->a_sze; a++) {
        if(a >= A_MAX) {
            printf("%*s...\n\n", offset, "");
            break;
        }
        if(a) {
            printf("\n\n");
        }
        for(uint64_t z = 0; z < tensor->buffer->z_sze; z++) {
            if(z >= Z_MAX) {
                printf("%*s...\n", offset, "");
                break;
            }
            if(z) {
                printf("\n");
            }
            for(uint64_t y = 0; y < tensor->buffer->y_sze; y++) {
                if(y >= Y_MAX) {
                    printf("%*s...\n", offset + padding, "");
                    break;
                }
                printf("%*s[ ", offset + padding, "");
                for(uint64_t x = 0; x < tensor->buffer->x_sze; x++) {
                    if(x >= X_MAX) {
                        printf("...");
                        break;
                    }
                    printf("% lf ", BUFFER_AT_(tensor->buffer, a, z, y, x));
                }
                printf("]\n");
            }
        }
    }
}
