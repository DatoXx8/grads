#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#include "tensor.h"
#include "utils.h"

ALWAYS_INLINE buffer_t buffer_alloc(uint64_t a, uint64_t z, uint64_t y, uint64_t x) {
    buffer_t buffer = {
        .a_inherent = a,
        .z_inherent = z,
        .y_inherent = y,
        .x_inherent = x,
        .a_size = a,
        .z_size = z,
        .y_size = y,
        .x_size = x,
        .a_stride = z * y * x,
        .z_stride = y * x,
        .y_stride = x,
        .x_stride = 1,
        .values = calloc(a * z * y * x, sizeof(double))
    };
    return(buffer);
}
ALWAYS_INLINE void buffer_free(buffer_t *buffer) {
    free(buffer->values);
}

/* Not a special or tested value at all. This is pure intuition. */
const uint64_t initial_child_number = 8;
/* However there is a max of two parents per lazyop. */
const uint64_t max_parent_number = 2;
op_t op_alloc(op_t *output_parent, op_t *input_parent) {
    op_t op = {0};
    op.parent = calloc(max_parent_number, sizeof(op_t *));
    op.child_capacity = initial_child_number;
    op.child = calloc(initial_child_number, sizeof(op_t *));
    if(output_parent) {
        if(output_parent->child_capacity == output_parent->child_count) {
            output_parent->child_capacity *= 2;
            output_parent->child = realloc(output_parent->child, output_parent->child_capacity * sizeof(op_t *));
        }
        output_parent->child[output_parent->child_count++] = &op;

        op.parent[op.parent_count++] = output_parent;
    }
    if(input_parent) {
        if(input_parent->child_capacity == input_parent->child_count) {
            input_parent->child_capacity *= 2;
            input_parent->child = realloc(input_parent->child, input_parent->child_capacity * sizeof(op_t *));
        }
        input_parent->child[input_parent->child_count++] = &op;

        op.parent[op.parent_count++] = input_parent;
    }
    return(op);
}
void op_free(op_t *op) {
    free(op->parent);
    free(op->child);
}
void op_cleanup(op_t *op) {
    if(op->tensor_base) {
        tensor_t *tensor = (tensor_t *) op->tensor_base;
        tensor->op = NULL;
    }
    uint64_t found;
    for(uint64_t i = 0; i < op->child_count; i++) {
        found = 0;
        for(uint64_t j = 0; j < op->child[i]->parent_count; j++) {
            if(op->child[i]->parent[j] == op) {
                found = 1;
            }
            if(found) {
                if(j == op->child[i]->parent_count - 1) {
                    op->child[i]->parent[j] = NULL;
                } else {
                    op->child[i]->parent[j] = op->child[i]->parent[j + 1];
                }
            }
        }
        op->child[i]->parent_count--;
    }
}
void op_single_op_print(op_t *op) {
    switch(op->type) {
        case(operation_unary): {
            switch(op->unary_type) {
                case(unary_add): {
                    printf("U add [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf\n", op->unary_buffer->a_inherent, op->unary_buffer->z_inherent, op->unary_buffer->y_inherent, op->unary_buffer->x_inherent, op->unary_buffer->a_size, op->unary_buffer->z_size, op->unary_buffer->y_size, op->unary_buffer->x_size, op->unary_buffer->offset, op->unary_value);
                    break;
                }
                case(unary_subtract): {
                    printf("U sub [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf\n", op->unary_buffer->a_inherent, op->unary_buffer->z_inherent, op->unary_buffer->y_inherent, op->unary_buffer->x_inherent, op->unary_buffer->a_size, op->unary_buffer->z_size, op->unary_buffer->y_size, op->unary_buffer->x_size, op->unary_buffer->offset, op->unary_value);
                    break;
                }
                case(unary_multiply): {
                    printf("U mul [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf\n", op->unary_buffer->a_inherent, op->unary_buffer->z_inherent, op->unary_buffer->y_inherent, op->unary_buffer->x_inherent, op->unary_buffer->a_size, op->unary_buffer->z_size, op->unary_buffer->y_size, op->unary_buffer->x_size, op->unary_buffer->offset, op->unary_value);
                    break;
                }
                case(unary_divide): {
                    printf("U div [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf\n", op->unary_buffer->a_inherent, op->unary_buffer->z_inherent, op->unary_buffer->y_inherent, op->unary_buffer->x_inherent, op->unary_buffer->a_size, op->unary_buffer->z_size, op->unary_buffer->y_size, op->unary_buffer->x_size, op->unary_buffer->offset, op->unary_value);
                    break;
                }
                case(unary_exp): {
                    printf("U exp [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf\n", op->unary_buffer->a_inherent, op->unary_buffer->z_inherent, op->unary_buffer->y_inherent, op->unary_buffer->x_inherent, op->unary_buffer->a_size, op->unary_buffer->z_size, op->unary_buffer->y_size, op->unary_buffer->x_size, op->unary_buffer->offset, op->unary_value);
                    break;
                }
                case(unary_log): {
                    printf("U log [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf\n", op->unary_buffer->a_inherent, op->unary_buffer->z_inherent, op->unary_buffer->y_inherent, op->unary_buffer->x_inherent, op->unary_buffer->a_size, op->unary_buffer->z_size, op->unary_buffer->y_size, op->unary_buffer->x_size, op->unary_buffer->offset, op->unary_value);
                    break;
                }
                case(unary_square): {
                    printf("U sqr [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf\n", op->unary_buffer->a_inherent, op->unary_buffer->z_inherent, op->unary_buffer->y_inherent, op->unary_buffer->x_inherent, op->unary_buffer->a_size, op->unary_buffer->z_size, op->unary_buffer->y_size, op->unary_buffer->x_size, op->unary_buffer->offset, op->unary_value);
                    break;
                }
                case(unary_sqrt): {
                    printf("U sqt [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf\n", op->unary_buffer->a_inherent, op->unary_buffer->z_inherent, op->unary_buffer->y_inherent, op->unary_buffer->x_inherent, op->unary_buffer->a_size, op->unary_buffer->z_size, op->unary_buffer->y_size, op->unary_buffer->x_size, op->unary_buffer->offset, op->unary_value);
                    break;
                }
                case(unary_negate): {
                    printf("U ngt [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf\n", op->unary_buffer->a_inherent, op->unary_buffer->z_inherent, op->unary_buffer->y_inherent, op->unary_buffer->x_inherent, op->unary_buffer->a_size, op->unary_buffer->z_size, op->unary_buffer->y_size, op->unary_buffer->x_size, op->unary_buffer->offset, op->unary_value);
                    break;
                }
                case(unary_reciprocal): {
                    printf("U rcp [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf\n", op->unary_buffer->a_inherent, op->unary_buffer->z_inherent, op->unary_buffer->y_inherent, op->unary_buffer->x_inherent, op->unary_buffer->a_size, op->unary_buffer->z_size, op->unary_buffer->y_size, op->unary_buffer->x_size, op->unary_buffer->offset, op->unary_value);
                    break;
                }
                case(unary_max): {
                    printf("U max [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf\n", op->unary_buffer->a_inherent, op->unary_buffer->z_inherent, op->unary_buffer->y_inherent, op->unary_buffer->x_inherent, op->unary_buffer->a_size, op->unary_buffer->z_size, op->unary_buffer->y_size, op->unary_buffer->x_size, op->unary_buffer->offset, op->unary_value);
                    break;
                }
                case(unary_min): {
                    printf("U min [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf\n", op->unary_buffer->a_inherent, op->unary_buffer->z_inherent, op->unary_buffer->y_inherent, op->unary_buffer->x_inherent, op->unary_buffer->a_size, op->unary_buffer->z_size, op->unary_buffer->y_size, op->unary_buffer->x_size, op->unary_buffer->offset, op->unary_value);
                    break;
                }
                case(unary_set): {
                    printf("U set [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf\n", op->unary_buffer->a_inherent, op->unary_buffer->z_inherent, op->unary_buffer->y_inherent, op->unary_buffer->x_inherent, op->unary_buffer->a_size, op->unary_buffer->z_size, op->unary_buffer->y_size, op->unary_buffer->x_size, op->unary_buffer->offset, op->unary_value);
                    break;
                }
                case(unary_zero): {
                    printf("U zer [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf\n", op->unary_buffer->a_inherent, op->unary_buffer->z_inherent, op->unary_buffer->y_inherent, op->unary_buffer->x_inherent, op->unary_buffer->a_size, op->unary_buffer->z_size, op->unary_buffer->y_size, op->unary_buffer->x_size, op->unary_buffer->offset, op->unary_value);
                    break;
                }
            }
            break;
        }
        case(operation_binary): {
            switch(op->binary_type) {
                case(binary_add): {
                    printf("B add {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu\n", op->binary_out->a_size, op->binary_out->z_size, op->binary_out->y_size, op->binary_out->x_size, op->binary_out->offset, op->binary_in->a_size, op->binary_in->z_size, op->binary_in->y_size, op->binary_in->x_size, op->binary_in->offset);
                    break;
                }
                case(binary_subtract): {
                    printf("B sub {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu\n", op->binary_out->a_size, op->binary_out->z_size, op->binary_out->y_size, op->binary_out->x_size, op->binary_out->offset, op->binary_in->a_size, op->binary_in->z_size, op->binary_in->y_size, op->binary_in->x_size, op->binary_in->offset);
                    break;
                }
                case(binary_multiply): {
                    printf("B mul {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu\n", op->binary_out->a_size, op->binary_out->z_size, op->binary_out->y_size, op->binary_out->x_size, op->binary_out->offset, op->binary_in->a_size, op->binary_in->z_size, op->binary_in->y_size, op->binary_in->x_size, op->binary_in->offset);
                    break;
                }
                case(binary_divide): {
                    printf("B div {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu\n", op->binary_out->a_size, op->binary_out->z_size, op->binary_out->y_size, op->binary_out->x_size, op->binary_out->offset, op->binary_in->a_size, op->binary_in->z_size, op->binary_in->y_size, op->binary_in->x_size, op->binary_in->offset);
                    break;
                }
                case(binary_max): {
                    printf("B max {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu\n", op->binary_out->a_size, op->binary_out->z_size, op->binary_out->y_size, op->binary_out->x_size, op->binary_out->offset, op->binary_in->a_size, op->binary_in->z_size, op->binary_in->y_size, op->binary_in->x_size, op->binary_in->offset);
                    break;
                }
                case(binary_min): {
                    printf("B min {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu\n", op->binary_out->a_size, op->binary_out->z_size, op->binary_out->y_size, op->binary_out->x_size, op->binary_out->offset, op->binary_in->a_size, op->binary_in->z_size, op->binary_in->y_size, op->binary_in->x_size, op->binary_in->offset);
                    break;
                }
                case(binary_copy): {
                    printf("B cpy {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu\n", op->binary_out->a_size, op->binary_out->z_size, op->binary_out->y_size, op->binary_out->x_size, op->binary_out->offset, op->binary_in->a_size, op->binary_in->z_size, op->binary_in->y_size, op->binary_in->x_size, op->binary_in->offset);
                    break;
                }
            }
            break;
        }
        case(operation_reduce): {
            switch(op->reduce_type) {
                case(reduce_sum): {
                    printf("R sum {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu\n", op->reduce_out->a_size, op->reduce_out->z_size, op->reduce_out->y_size, op->reduce_out->x_size, op->reduce_out->offset, op->reduce_in->a_size, op->reduce_in->z_size, op->reduce_in->y_size, op->reduce_in->x_size, op->reduce_in->offset);
                    break;
                }
                case(reduce_max): {
                    printf("R max {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu\n", op->reduce_out->a_size, op->reduce_out->z_size, op->reduce_out->y_size, op->reduce_out->x_size, op->reduce_out->offset, op->reduce_in->a_size, op->reduce_in->z_size, op->reduce_in->y_size, op->reduce_in->x_size, op->reduce_in->offset);
                    break;
                }
                case(reduce_avg): {
                    printf("R avg {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu\n", op->reduce_out->a_size, op->reduce_out->z_size, op->reduce_out->y_size, op->reduce_out->x_size, op->reduce_out->offset, op->reduce_in->a_size, op->reduce_in->z_size, op->reduce_in->y_size, op->reduce_in->x_size, op->reduce_in->offset);
                    break;
                }
                case(reduce_min): {
                    printf("R min {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu\n", op->reduce_out->a_size, op->reduce_out->z_size, op->reduce_out->y_size, op->reduce_out->x_size, op->reduce_out->offset, op->reduce_in->a_size, op->reduce_in->z_size, op->reduce_in->y_size, op->reduce_in->x_size, op->reduce_in->offset);
                    break;
                }
            }
            break;
        }
        case(operation_move): {
            switch(op->move_type) {
                case(move_reshape): {
                    printf("M rsp {%lu, %lu, %lu, %lu} %lu - {%lu, %lu, %lu, %lu} %lu\n", op->move_buffer->a_size, op->move_buffer->z_size, op->move_buffer->y_size, op->move_buffer->x_size, op->move_buffer->offset, op->move_a, op->move_z, op->move_y, op->move_x, op->move_buffer->offset);
                    break;
                }
                case(move_offset): {
                    printf("M off {%lu, %lu, %lu, %lu} %lu - {%lu, %lu, %lu, %lu} %lu\n", op->move_buffer->a_size, op->move_buffer->z_size, op->move_buffer->y_size, op->move_buffer->x_size, op->move_buffer->offset, op->move_buffer->a_size, op->move_buffer->z_size, op->move_buffer->y_size, op->move_buffer->x_size, op->move_a);
                    break;
                }
            }
            break;
        }
    }
}
ALWAYS_INLINE void op_single_op_cpu_realize(op_t *op) {
    switch(op->type) {
        case(operation_unary): {
            switch(op->unary_type) {
                case(unary_add): {
                    for(uint64_t a = 0; a < op->unary_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->unary_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->unary_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->unary_buffer->x_size; x++) {
                                    BUFFER_AT_(op->unary_buffer, a, z, y, x) += op->unary_value;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_subtract): {
                    for(uint64_t a = 0; a < op->unary_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->unary_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->unary_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->unary_buffer->x_size; x++) {
                                    BUFFER_AT_(op->unary_buffer, a, z, y, x) -= op->unary_value;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_multiply): {
                    for(uint64_t a = 0; a < op->unary_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->unary_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->unary_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->unary_buffer->x_size; x++) {
                                    BUFFER_AT_(op->unary_buffer, a, z, y, x) *= op->unary_value;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_divide): {
                    for(uint64_t a = 0; a < op->unary_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->unary_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->unary_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->unary_buffer->x_size; x++) {
                                    BUFFER_AT_(op->unary_buffer, a, z, y, x) /= op->unary_value;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_exp): {
                    for(uint64_t a = 0; a < op->unary_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->unary_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->unary_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->unary_buffer->x_size; x++) {
                                    BUFFER_AT_(op->unary_buffer, a, z, y, x) = exp(BUFFER_AT_(op->unary_buffer, a, z, y ,x));
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_log): {
                    for(uint64_t a = 0; a < op->unary_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->unary_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->unary_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->unary_buffer->x_size; x++) {
                                    BUFFER_AT_(op->unary_buffer, a, z, y, x) = log(BUFFER_AT_(op->unary_buffer, a, z, y ,x));
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_square): {
                    for(uint64_t a = 0; a < op->unary_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->unary_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->unary_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->unary_buffer->x_size; x++) {
                                    BUFFER_AT_(op->unary_buffer, a, z, y, x) *= BUFFER_AT_(op->unary_buffer, a, z, y ,x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_sqrt): {
                    for(uint64_t a = 0; a < op->unary_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->unary_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->unary_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->unary_buffer->x_size; x++) {
                                    BUFFER_AT_(op->unary_buffer, a, z, y, x) = sqrt(BUFFER_AT_(op->unary_buffer, a, z, y ,x));
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_negate): {
                    for(uint64_t a = 0; a < op->unary_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->unary_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->unary_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->unary_buffer->x_size; x++) {
                                    BUFFER_AT_(op->unary_buffer, a, z, y, x) = - BUFFER_AT_(op->unary_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_reciprocal): {
                    for(uint64_t a = 0; a < op->unary_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->unary_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->unary_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->unary_buffer->x_size; x++) {
                                    BUFFER_AT_(op->unary_buffer, a, z, y, x) = 1 / BUFFER_AT_(op->unary_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_max): {
                    for(uint64_t a = 0; a < op->unary_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->unary_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->unary_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->unary_buffer->x_size; x++) {
                                    if(BUFFER_AT_(op->unary_buffer, a, z, y, x) < op->unary_value) {
                                        BUFFER_AT_(op->unary_buffer, a, z, y, x) = op->unary_value;
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_min): {
                    for(uint64_t a = 0; a < op->unary_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->unary_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->unary_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->unary_buffer->x_size; x++) {
                                    if(BUFFER_AT_(op->unary_buffer, a, z, y, x) > op->unary_value) {
                                        BUFFER_AT_(op->unary_buffer, a, z, y, x) = op->unary_value;
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_set): {
                    for(uint64_t a = 0; a < op->unary_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->unary_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->unary_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->unary_buffer->x_size; x++) {
                                    BUFFER_AT_(op->unary_buffer, a, z, y, x) = op->unary_value;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_zero): {
                    explicit_bzero(op->unary_buffer->values, op->unary_buffer->a_size * op->unary_buffer->z_size * op->unary_buffer->y_size * op->unary_buffer->x_size * sizeof(double));
                    break;
                }
            }
            break;
        }
        case(operation_binary): {
            switch(op->binary_type) {
                case(binary_add): {
                    for(uint64_t a = 0; a < op->binary_out->a_size; a++) {
                        for(uint64_t z = 0; z < op->binary_out->z_size; z++) {
                            for(uint64_t y = 0; y < op->binary_out->y_size; y++) {
                                for(uint64_t x = 0; x < op->binary_out->x_size; x++) {
                                    BUFFER_AT_(op->binary_out, a, z, y, x) += BUFFER_AT_(op->binary_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_subtract): {
                    for(uint64_t a = 0; a < op->binary_out->a_size; a++) {
                        for(uint64_t z = 0; z < op->binary_out->z_size; z++) {
                            for(uint64_t y = 0; y < op->binary_out->y_size; y++) {
                                for(uint64_t x = 0; x < op->binary_out->x_size; x++) {
                                    BUFFER_AT_(op->binary_out, a, z, y, x) -= BUFFER_AT_(op->binary_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_multiply): {
                    for(uint64_t a = 0; a < op->binary_out->a_size; a++) {
                        for(uint64_t z = 0; z < op->binary_out->z_size; z++) {
                            for(uint64_t y = 0; y < op->binary_out->y_size; y++) {
                                for(uint64_t x = 0; x < op->binary_out->x_size; x++) {
                                    BUFFER_AT_(op->binary_out, a, z, y, x) *= BUFFER_AT_(op->binary_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_divide): {
                    for(uint64_t a = 0; a < op->binary_out->a_size; a++) {
                        for(uint64_t z = 0; z < op->binary_out->z_size; z++) {
                            for(uint64_t y = 0; y < op->binary_out->y_size; y++) {
                                for(uint64_t x = 0; x < op->binary_out->x_size; x++) {
                                    BUFFER_AT_(op->binary_out, a, z, y, x) /= BUFFER_AT_(op->binary_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_max): {
                    for(uint64_t a = 0; a < op->binary_out->a_size; a++) {
                        for(uint64_t z = 0; z < op->binary_out->z_size; z++) {
                            for(uint64_t y = 0; y < op->binary_out->y_size; y++) {
                                for(uint64_t x = 0; x < op->binary_out->x_size; x++) {
                                    if(BUFFER_AT_(op->binary_out, a, z, y, x) < BUFFER_AT_(op->binary_in, a, z, y, x)) {
                                        BUFFER_AT_(op->binary_out, a, z, y, x) = BUFFER_AT_(op->binary_in, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_min): {
                    for(uint64_t a = 0; a < op->binary_out->a_size; a++) {
                        for(uint64_t z = 0; z < op->binary_out->z_size; z++) {
                            for(uint64_t y = 0; y < op->binary_out->y_size; y++) {
                                for(uint64_t x = 0; x < op->binary_out->x_size; x++) {
                                    if(BUFFER_AT_(op->binary_out, a, z, y, x) > BUFFER_AT_(op->binary_in, a, z, y, x)) {
                                        BUFFER_AT_(op->binary_out, a, z, y, x) = BUFFER_AT_(op->binary_in, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_copy): {
                    for(uint64_t a = 0; a < op->binary_out->a_size; a++) {
                        for(uint64_t z = 0; z < op->binary_out->z_size; z++) {
                            for(uint64_t y = 0; y < op->binary_out->y_size; y++) {
                                for(uint64_t x = 0; x < op->binary_out->x_size; x++) {
                                    BUFFER_AT_(op->binary_out, a, z, y, x) = BUFFER_AT_(op->binary_in, a, z, y, x);
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
            switch(op->reduce_type) {
                case(reduce_sum): {
                    double temp = 0;
                    for(uint64_t a = 0; a < op->reduce_in->a_size; a++) {
                        for(uint64_t z = 0; z < op->reduce_in->z_size; z++) {
                            for(uint64_t y = 0; y < op->reduce_in->y_size; y++) {
                                for(uint64_t x = 0; x < op->reduce_in->x_size; x++) {
                                    temp += BUFFER_AT_(op->reduce_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    BUFFER_AT_(op->reduce_out, 0, 0, 0, 0) = temp;
                    break;
                }
                case(reduce_max): {
                    double temp = - INFINITY;
                    for(uint64_t a = 0; a < op->reduce_in->a_size; a++) {
                        for(uint64_t z = 0; z < op->reduce_in->z_size; z++) {
                            for(uint64_t y = 0; y < op->reduce_in->y_size; y++) {
                                for(uint64_t x = 0; x < op->reduce_in->x_size; x++) {
                                    if(temp < BUFFER_AT_(op->reduce_in, a, z, y, x)) {
                                        temp = BUFFER_AT_(op->reduce_in, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    BUFFER_AT_(op->reduce_out, 0, 0, 0, 0) = temp;
                    break;
                }
                case(reduce_avg): {
                    double temp = 0;
                    for(uint64_t a = 0; a < op->reduce_in->a_size; a++) {
                        for(uint64_t z = 0; z < op->reduce_in->z_size; z++) {
                            for(uint64_t y = 0; y < op->reduce_in->y_size; y++) {
                                for(uint64_t x = 0; x < op->reduce_in->x_size; x++) {
                                    temp += BUFFER_AT_(op->reduce_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    BUFFER_AT_(op->reduce_out, 0, 0, 0, 0) = temp / (op->reduce_in->x_size * op->reduce_in->y_size * op->reduce_in->z_size * op->reduce_in->a_size);
                    break;
                }
                case(reduce_min): {
                    double temp = INFINITY;
                    for(uint64_t a = 0; a < op->reduce_in->a_size; a++) {
                        for(uint64_t z = 0; z < op->reduce_in->z_size; z++) {
                            for(uint64_t y = 0; y < op->reduce_in->y_size; y++) {
                                for(uint64_t x = 0; x < op->reduce_in->x_size; x++) {
                                    if(temp < BUFFER_AT_(op->reduce_in, a, z, y, x)) {
                                        temp = BUFFER_AT_(op->reduce_in, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    BUFFER_AT_(op->reduce_out, 0, 0, 0, 0) = temp;
                    break;
                }
            }
            break;
        }
        case(operation_move): {
            switch(op->move_type) {
                case(move_reshape): {
                    op->move_buffer->a_size = op->move_a;
                    op->move_buffer->z_size = op->move_z;
                    op->move_buffer->y_size = op->move_y;
                    op->move_buffer->x_size = op->move_x;
                    break;
                }
                case(move_offset): {
                    op->move_buffer->offset = op->move_buffer->a_inherent * op->move_a + op->move_buffer->z_inherent * op->move_z + op->move_buffer->y_inherent * op->move_y + op->move_buffer->x_inherent * op->move_x;
                    break;
                }
            }
            break;
        }
    }
}
void op_cpu_realize(op_t *op) {
    while(op->parent_count > 0) {
        op_cpu_realize(op->parent[0]);
    }
    op_single_op_cpu_realize(op);
    op_cleanup(op);
    op_free(op);
    free(op);
}
// void op_cl_realize(op_t *op) {
// }

tensor_t tensor_alloc(uint64_t a, uint64_t z, uint64_t y, uint64_t x) {
}
void tensor_free(tensor_t *tensor) {
}
