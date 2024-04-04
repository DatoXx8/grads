#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "tensor.h"
#include "linearize.h"
#include "utils.h"

void ALWAYS_INLINE simple_op_simulate_move(op_t *op) {
    switch(op->move_type) {
        case(move_reshape): {
            op->out_buffer->cl_a_size = op->var_a;
            op->out_buffer->cl_z_size = op->var_z;
            op->out_buffer->cl_y_size = op->var_y;
            op->out_buffer->cl_x_size = op->var_x;
            op->out_buffer->cl_a_stride = op->var_z * op->var_y * op->var_x;
            op->out_buffer->cl_z_stride = op->var_y * op->var_x;
            op->out_buffer->cl_y_stride = op->var_x;
            op->out_buffer->cl_x_stride = 1;
            break;
        }
        case(move_resize): {
            op->out_buffer->cl_a_size = op->var_a;
            op->out_buffer->cl_z_size = op->var_z;
            op->out_buffer->cl_y_size = op->var_y;
            op->out_buffer->cl_x_size = op->var_x;
            break;
        }
        case(move_offset): {
            op->out_buffer->cl_offset = op->out_buffer->cl_a_stride * op->var_a + op->out_buffer->cl_z_stride * op->var_z + op->out_buffer->cl_y_stride * op->var_y + op->out_buffer->cl_x_stride * op->var_x;
            break;
        }
    }
}
void ALWAYS_INLINE simple_op_convert(simple_op_t *simple_op, op_t *op) {
    simple_op->type = op->type;
    simple_op->unary_type = op->unary_type;
    simple_op->binary_type = op->binary_type;
    simple_op->reduce_type = op->reduce_type;
    simple_op->var_unary = op->var_unary;
    simple_op->out_buffer.a_size = op->out_buffer->cl_a_size;
    simple_op->out_buffer.z_size = op->out_buffer->cl_z_size;
    simple_op->out_buffer.y_size = op->out_buffer->cl_y_size;
    simple_op->out_buffer.x_size = op->out_buffer->cl_x_size;
    simple_op->out_buffer.a_stride = op->out_buffer->cl_a_stride;
    simple_op->out_buffer.z_stride = op->out_buffer->cl_z_stride;
    simple_op->out_buffer.y_stride = op->out_buffer->cl_y_stride;
    simple_op->out_buffer.x_stride = op->out_buffer->cl_x_stride;
    simple_op->out_buffer.offset = op->out_buffer->cl_offset;
    simple_op->out_buffer.values = op->out_buffer->values;
    strncpy(simple_op->out_buffer.name, op->out_buffer->cl_name, BUFFER_NAME_SIZE + 1);
    if((op->type == operation_binary) || (op->type == operation_reduce)) {
        simple_op->in_buffer.a_size = op->in_buffer->cl_a_size;
        simple_op->in_buffer.z_size = op->in_buffer->cl_z_size;
        simple_op->in_buffer.y_size = op->in_buffer->cl_y_size;
        simple_op->in_buffer.x_size = op->in_buffer->cl_x_size;
        simple_op->in_buffer.a_stride = op->in_buffer->cl_a_stride;
        simple_op->in_buffer.z_stride = op->in_buffer->cl_z_stride;
        simple_op->in_buffer.y_stride = op->in_buffer->cl_y_stride;
        simple_op->in_buffer.x_stride = op->in_buffer->cl_x_stride;
        simple_op->in_buffer.offset = op->in_buffer->cl_offset;
        simple_op->in_buffer.values = op->in_buffer->values;
        strncpy(simple_op->in_buffer.name, op->in_buffer->cl_name, BUFFER_NAME_SIZE + 1);
    }
}
void simple_op_print(simple_op_t *simple_op, int padding, int offset, const char *name) {
    if(strcmp(name, "")) {
        printf("%*s%s\n", offset, "", name);
    }
    printf("%*s<%p> ", offset + padding, "", (void *) simple_op);
    switch(simple_op->type) {
        case(operation_unary): {
            switch(simple_op->unary_type) {
                case(unary_add): {
                    printf("U add {%lu, %lu, %lu, %lu} %lu %lf %s\n", simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->var_unary, simple_op->out_buffer.name);
                    break;
                }
                case(unary_subtract): {
                    printf("U sub {%lu, %lu, %lu, %lu} %lu %lf %s\n", simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->var_unary, simple_op->out_buffer.name);
                    break;
                }
                case(unary_multiply): {
                    printf("U mul {%lu, %lu, %lu, %lu} %lu %lf %s\n", simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->var_unary, simple_op->out_buffer.name);
                    break;
                }
                case(unary_divide): {
                    printf("U div {%lu, %lu, %lu, %lu} %lu %lf %s\n", simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->var_unary, simple_op->out_buffer.name);
                    break;
                }
                case(unary_exp): {
                    printf("U exp {%lu, %lu, %lu, %lu} %lu %s\n", simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->out_buffer.name);
                    break;
                }
                case(unary_log): {
                    printf("U log {%lu, %lu, %lu, %lu} %lu %s\n", simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->out_buffer.name);
                    break;
                }
                case(unary_square): {
                    printf("U sqr {%lu, %lu, %lu, %lu} %lu %s\n", simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->out_buffer.name);
                    break;
                }
                case(unary_sqrt): {
                    printf("U sqt {%lu, %lu, %lu, %lu} %lu %s\n", simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->out_buffer.name);
                    break;
                }
                case(unary_negate): {
                    printf("U neg {%lu, %lu, %lu, %lu} %lu %s\n", simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->out_buffer.name);
                    break;
                }
                case(unary_reciprocal): {
                    printf("U rcp {%lu, %lu, %lu, %lu} %lu %s\n", simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->out_buffer.name);
                    break;
                }
                case(unary_max): {
                    printf("U max {%lu, %lu, %lu, %lu} %lu %lf %s\n", simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->var_unary, simple_op->out_buffer.name);
                    break;
                }
                case(unary_min): {
                    printf("U min {%lu, %lu, %lu, %lu} %lu %lf %s\n", simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->var_unary, simple_op->out_buffer.name);
                    break;
                }
                case(unary_set): {
                    printf("U set {%lu, %lu, %lu, %lu} %lu %lf %s\n", simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->var_unary, simple_op->out_buffer.name);
                    break;
                }
                case(unary_random): {
                    printf("U ran {%lu, %lu, %lu, %lu} %lu %s\n", simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->out_buffer.name);
                    break;
                }
                case(unary_tanh): {
                    printf("U tnh {%lu, %lu, %lu, %lu} %lu %s\n", simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->out_buffer.name);
                    break;
                }
                case(unary_absolute): {
                    printf("U abs {%lu, %lu, %lu, %lu} %lu %s\n", simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->out_buffer.name);
                    break;
                }
                case(unary_sign): {
                    printf("U sgn {%lu, %lu, %lu, %lu} %lu %s\n", simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->out_buffer.name);
                    break;
                }
            }
            break;
        }
        case(operation_binary): {
            switch(simple_op->binary_type) {
                case(binary_add): {
                    printf("B add {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
                case(binary_subtract): {
                    printf("B sub {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
                case(binary_multiply): {
                    printf("B mul {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
                case(binary_divide): {
                    printf("B div {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
                case(binary_max): {
                    printf("B max {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
                case(binary_min): {
                    printf("B min {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
                case(binary_copy): {
                    printf("B cpy {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
                case(binary_add_like): {
                    printf("L add {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
                case(binary_subtract_like): {
                    printf("L sub {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
                case(binary_multiply_like): {
                    printf("L mul {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
                case(binary_divide_like): {
                    printf("L div {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
                case(binary_max_like): {
                    printf("L max {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
                case(binary_min_like): {
                    printf("L min {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
                case(binary_copy_like): {
                    printf("L cpy {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
            }
            break;
        }
        case(operation_reduce): {
            switch(simple_op->reduce_type) {
                case(reduce_sum): {
                    printf("R sum {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
                case(reduce_avg): {
                    printf("R avg {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
                case(reduce_max): {
                    printf("R max {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
                case(reduce_min): {
                    printf("R min {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu %s %s\n", simple_op->in_buffer.a_size, simple_op->in_buffer.z_size, simple_op->in_buffer.y_size, simple_op->in_buffer.x_size, simple_op->in_buffer.offset, simple_op->out_buffer.a_size, simple_op->out_buffer.z_size, simple_op->out_buffer.y_size, simple_op->out_buffer.x_size, simple_op->out_buffer.offset, simple_op->in_buffer.name, simple_op->out_buffer.name);
                    break;
                }
            }
            break;
        }
        case(operation_move): {
            fprintf(stderr, "ERROR: simple_op should not be a move operation!\n");
            exit(1);
        }
    }
}
ALWAYS_INLINE void simple_op_realize(simple_op_t *simple_op) {
    switch(simple_op->type) {
        case(operation_unary): {
            switch(simple_op->unary_type) {
                case(unary_add): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) += simple_op->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_subtract): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) -= simple_op->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_multiply): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) *= simple_op->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_divide): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) /= simple_op->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_exp): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = exp(SIMPLE_AT(simple_op->out_buffer, a, z, y ,x));
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_log): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = log(SIMPLE_AT(simple_op->out_buffer, a, z, y ,x));
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_square): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) *= SIMPLE_AT(simple_op->out_buffer, a, z, y ,x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_sqrt): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = sqrt(SIMPLE_AT(simple_op->out_buffer, a, z, y ,x));
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_negate): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = - SIMPLE_AT(simple_op->out_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_reciprocal): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = 1 / SIMPLE_AT(simple_op->out_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_max): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    if(SIMPLE_AT(simple_op->out_buffer, a, z, y, x) < simple_op->var_unary) {
                                        SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = simple_op->var_unary;
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_min): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    if(SIMPLE_AT(simple_op->out_buffer, a, z, y, x) > simple_op->var_unary) {
                                        SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = simple_op->var_unary;
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_set): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = simple_op->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_random): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = ((double) rand() / RAND_MAX) * 2 - 1;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_tanh): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = tanh(SIMPLE_AT(simple_op->out_buffer, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_absolute): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = fabs(SIMPLE_AT(simple_op->out_buffer, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_sign): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    if(SIMPLE_AT(simple_op->out_buffer, a, z, y, x) < 0) {
                                        SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = -1;
                                        /* Better perf but kinda ugly */
                                        // } else if(!SIMPLE_AT(simple_op->out_buffer, a, z, y, x)) {
                                    } else if(SIMPLE_AT(simple_op->out_buffer, a, z, y, x) == 0) {
                                        SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = 0;
                                    } else {
                                        SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = 1;
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
        case(operation_binary): {
            switch(simple_op->binary_type) {
                case(binary_add): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) += SIMPLE_AT(simple_op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_subtract): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) -= SIMPLE_AT(simple_op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_multiply): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) *= SIMPLE_AT(simple_op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_divide): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) /= SIMPLE_AT(simple_op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_max): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    if(SIMPLE_AT(simple_op->out_buffer, a, z, y, x) < SIMPLE_AT(simple_op->in_buffer, a, z, y, x)) {
                                        SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = SIMPLE_AT(simple_op->in_buffer, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_min): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    if(SIMPLE_AT(simple_op->out_buffer, a, z, y, x) > SIMPLE_AT(simple_op->in_buffer, a, z, y, x)) {
                                        SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = SIMPLE_AT(simple_op->in_buffer, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_copy): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = SIMPLE_AT(simple_op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_add_like): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) += SIMPLE_AT(simple_op->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_subtract_like): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) -= SIMPLE_AT(simple_op->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_multiply_like): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) *= SIMPLE_AT(simple_op->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_divide_like): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) /= SIMPLE_AT(simple_op->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_max_like): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    if(SIMPLE_AT(simple_op->out_buffer, a, z, y, x) < SIMPLE_AT(simple_op->in_buffer, 0, 0, 0, 0)) {
                                        SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = SIMPLE_AT(simple_op->in_buffer, 0, 0, 0, 0);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_min_like): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    if(SIMPLE_AT(simple_op->out_buffer, a, z, y, x) > SIMPLE_AT(simple_op->in_buffer, 0, 0, 0, 0)) {
                                        SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = SIMPLE_AT(simple_op->in_buffer, 0, 0, 0, 0);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_copy_like): {
                    for(uint64_t a = 0; a < simple_op->out_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer.x_size; x++) {
                                    SIMPLE_AT(simple_op->out_buffer, a, z, y, x) = SIMPLE_AT(simple_op->in_buffer, 0, 0, 0, 0);
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
            switch(simple_op->reduce_type) {
                case(reduce_sum): {
                    double temp = 0;
                    for(uint64_t a = 0; a < simple_op->in_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->in_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->in_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->in_buffer.x_size; x++) {
                                    temp += SIMPLE_AT(simple_op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    SIMPLE_AT(simple_op->out_buffer, 0, 0, 0, 0) = temp;
                    break;
                }
                case(reduce_max): {
                    double temp = - INFINITY;
                    for(uint64_t a = 0; a < simple_op->in_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->in_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->in_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->in_buffer.x_size; x++) {
                                    if(temp < SIMPLE_AT(simple_op->in_buffer, a, z, y, x)) {
                                        temp = SIMPLE_AT(simple_op->in_buffer, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    SIMPLE_AT(simple_op->out_buffer, 0, 0, 0, 0) = temp;
                    break;
                }
                case(reduce_avg): {
                    double temp = 0;
                    for(uint64_t a = 0; a < simple_op->in_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->in_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->in_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->in_buffer.x_size; x++) {
                                    temp += SIMPLE_AT(simple_op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    SIMPLE_AT(simple_op->out_buffer, 0, 0, 0, 0) = temp / (simple_op->in_buffer.x_size * simple_op->in_buffer.y_size * simple_op->in_buffer.z_size * simple_op->in_buffer.a_size);
                    break;
                }
                case(reduce_min): {
                    double temp = INFINITY;
                    for(uint64_t a = 0; a < simple_op->in_buffer.a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->in_buffer.z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->in_buffer.y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->in_buffer.x_size; x++) {
                                    if(temp > SIMPLE_AT(simple_op->in_buffer, a, z, y, x)) {
                                        temp = SIMPLE_AT(simple_op->in_buffer, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    SIMPLE_AT(simple_op->out_buffer, 0, 0, 0, 0) = temp;
                    break;
                }
            }
            break;
        }
        case(operation_move): {
            fprintf(stderr, "ERROR: simple_op should not be a move operation!\n");
            exit(1);
        }
    }
}

/* NOTE: Completely made up value. No reasoning behind it at all. */
const uint64_t initial_simple_op_capactity = 25;
linearized_t linearized_alloc(void) {
    linearized_t linearized = {
        .op_count = 0,
        .op_capacity = initial_simple_op_capactity,
        .simple = calloc(initial_simple_op_capactity, sizeof(simple_op_t)),
    };
    assert(linearized.simple);

    return(linearized);
}
/* NOTE: Does `not` override the linearized ops instead appends ops. */
void linearized_from_op(linearized_t *linearized, op_t *op) {
    /* TODO: Maybe remove this eventually. */
    if(!op) {
        return;
    }
    while(op->parent_count > 0) {
        linearized_from_op(linearized, op->parent[0]);
    }
    if(linearized->op_capacity == linearized->op_count) {
        linearized->op_capacity *= 2;
        linearized->simple = realloc(linearized->simple, linearized->op_capacity * sizeof(simple_op_t));
    }
    if(op->type == operation_move) {
        simple_op_simulate_move(op);
    } else {
        simple_op_convert(&linearized->simple[linearized->op_count++], op);
    }
    op_cleanup(op);
    op_free(op);
    free(op);
}
void linearized_free(linearized_t *linearized) {
    free(linearized->simple);
}
void linearized_clear(linearized_t *linearized) {
    linearized->op_count = 0;
}
void linearized_run(linearized_t *linearized) {
    for(uint64_t i = 0; i < linearized->op_count; i++) {
        simple_op_realize(&linearized->simple[i]);
    }
}
void linearized_print(linearized_t *linearized, int padding, int offset, const char *name) {
    if(!linearized) {
        return;
    }
    if(strcmp(name, "")) {
        printf("%*slen %lu, cap %lu %s\n", offset, "", linearized->op_count, linearized->op_capacity, name);
    } else {
        printf("%*slen %lu, cap %lu\n", offset, "", linearized->op_count, linearized->op_capacity);
    }
    /* NOTE: Kind of a nice allignment for printing */
    // uint64_t max = log10(linearized->op_count);
    // for(uint64_t i = 0; i < linearized->op_count; i++) {
    //     printf("%*s[%*s%lu] ", padding + offset, "", (int) (max - (uint64_t) log10(i)), "", i);
    //     simple_op_print(linearized->simple + i, 0, 0, "");
    // }
    /* This one is not alligned. */
    for(uint64_t i = 0; i < linearized->op_count; i++) {
        printf("%*s[%lu] ", padding + offset, "", i);
        simple_op_print(linearized->simple + i, 0, 0, "");
    }
}
