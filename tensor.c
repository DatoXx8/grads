#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"
#include "utils.h"

uint64_t name_start = 0;
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
        .values = calloc(a * z * y * x, sizeof(double)),
        .cl_a_size = a,
        .cl_z_size = z,
        .cl_y_size = y,
        .cl_x_size = x,
        .cl_a_stride = z * y * x,
        .cl_z_stride = y * x,
        .cl_y_stride = x,
        .cl_x_stride = 1,
    };
    assert(buffer.values);
    for(uint64_t i = 0; i < BUFFER_NAME_SIZE; i++) { buffer.cl_name[i] = 'a'; }
    uint64_t name_offset = name_start++;
    for(int64_t i = BUFFER_NAME_SIZE - 1; i >= 0; i--) {
        buffer.cl_name[i] += name_offset / (uint64_t)pow(26, i);
        name_offset = name_offset % (uint64_t)pow(26, i);
    }
    buffer.cl_name[BUFFER_NAME_SIZE] = '\0';
    return (buffer);
}
void buffer_free(buffer_t *buffer) {
    free(buffer->values);
}

/* Not a special or tested value at all. This is pure intuition. */
const uint64_t initial_child_number = 8;
/* However there is a max of two parents per lazyop. */
const uint64_t max_parent_number = 2;
op_t op_alloc(void) {
    op_t op = {0};
    op.parent = calloc(max_parent_number, sizeof(op_t *));
    assert(op.parent);
    op.parent_capacity = max_parent_number;
    op.child_capacity = initial_child_number;
    op.child = calloc(initial_child_number, sizeof(op_t *));
    assert(op.child);
    return (op);
}
void op_add_parents(op_t *op, op_t *output_parent, op_t *input_parent) {
    if(output_parent) {
        if(output_parent->child_capacity == output_parent->child_count) {
            output_parent->child_capacity *= 2;
            output_parent->child = realloc(output_parent->child, output_parent->child_capacity * sizeof(op_t *));
        }
        output_parent->child[output_parent->child_count++] = op;

        op->parent[op->parent_count++] = output_parent;
    }
    if(input_parent) {
        if(input_parent->child_capacity == input_parent->child_count) {
            input_parent->child_capacity *= 2;
            input_parent->child = realloc(input_parent->child, input_parent->child_capacity * sizeof(op_t *));
        }
        input_parent->child[input_parent->child_count++] = op;

        op->parent[op->parent_count++] = input_parent;
    }
}
void op_free(op_t *op) {
    free(op->parent);
    free(op->child);
}
void op_cleanup(op_t *op) {
    if(op->tensor_base) {
        tensor_t *tensor = (tensor_t *)op->tensor_base;
        tensor->op = NULL;
    }
    uint64_t found;
    for(uint64_t i = 0; i < op->child_count; i++) {
        found = 0;
        for(uint64_t j = 0; j < op->child[i]->parent_count; j++) {
            if(op->child[i]->parent[j] == op) { found = 1; }
            if(found) {
                if(j == op->child[i]->parent_count - 1) {
                    op->child[i]->parent[j] = NULL;
                } else {
                    op->child[i]->parent[j] = op->child[i]->parent[j + 1];
                }
            }
        }
        op->child[i]->parent_count--;
        op->child[i] = NULL;
    }
    op->child_count = 0;
}
void op_single_print(op_t *op, int padding, int offset, const char *name) {
    if(strcmp(name, "")) { printf("%*s%s\n", offset, "", name); }
    printf("%*s<%p> ", offset + padding, "", (void *)op);
    switch(op->type) {
        case(operation_unary): {
            switch(op->unary_type) {
                case(unary_add): {
                    printf("U add [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf [%p]\n", op->out_buffer->a_inherent, op->out_buffer->z_inherent,
                           op->out_buffer->y_inherent, op->out_buffer->x_inherent, op->out_buffer->a_size, op->out_buffer->z_size, op->out_buffer->y_size,
                           op->out_buffer->x_size, op->out_buffer->offset, op->var_unary, (void *)op->out_buffer);
                    break;
                }
                case(unary_subtract): {
                    printf("U sub [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf [%p]\n", op->out_buffer->a_inherent, op->out_buffer->z_inherent,
                           op->out_buffer->y_inherent, op->out_buffer->x_inherent, op->out_buffer->a_size, op->out_buffer->z_size, op->out_buffer->y_size,
                           op->out_buffer->x_size, op->out_buffer->offset, op->var_unary, (void *)op->out_buffer);
                    break;
                }
                case(unary_multiply): {
                    printf("U mul [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf [%p]\n", op->out_buffer->a_inherent, op->out_buffer->z_inherent,
                           op->out_buffer->y_inherent, op->out_buffer->x_inherent, op->out_buffer->a_size, op->out_buffer->z_size, op->out_buffer->y_size,
                           op->out_buffer->x_size, op->out_buffer->offset, op->var_unary, (void *)op->out_buffer);
                    break;
                }
                case(unary_divide): {
                    printf("U div [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf [%p]\n", op->out_buffer->a_inherent, op->out_buffer->z_inherent,
                           op->out_buffer->y_inherent, op->out_buffer->x_inherent, op->out_buffer->a_size, op->out_buffer->z_size, op->out_buffer->y_size,
                           op->out_buffer->x_size, op->out_buffer->offset, op->var_unary, (void *)op->out_buffer);
                    break;
                }
                case(unary_exp): {
                    printf("U exp [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p]\n", op->out_buffer->a_inherent, op->out_buffer->z_inherent,
                           op->out_buffer->y_inherent, op->out_buffer->x_inherent, op->out_buffer->a_size, op->out_buffer->z_size, op->out_buffer->y_size,
                           op->out_buffer->x_size, op->out_buffer->offset, (void *)op->out_buffer);
                    break;
                }
                case(unary_log): {
                    printf("U log [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p]\n", op->out_buffer->a_inherent, op->out_buffer->z_inherent,
                           op->out_buffer->y_inherent, op->out_buffer->x_inherent, op->out_buffer->a_size, op->out_buffer->z_size, op->out_buffer->y_size,
                           op->out_buffer->x_size, op->out_buffer->offset, (void *)op->out_buffer);
                    break;
                }
                case(unary_square): {
                    printf("U sqr [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p]\n", op->out_buffer->a_inherent, op->out_buffer->z_inherent,
                           op->out_buffer->y_inherent, op->out_buffer->x_inherent, op->out_buffer->a_size, op->out_buffer->z_size, op->out_buffer->y_size,
                           op->out_buffer->x_size, op->out_buffer->offset, (void *)op->out_buffer);
                    break;
                }
                case(unary_sqrt): {
                    printf("U sqt [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p]\n", op->out_buffer->a_inherent, op->out_buffer->z_inherent,
                           op->out_buffer->y_inherent, op->out_buffer->x_inherent, op->out_buffer->a_size, op->out_buffer->z_size, op->out_buffer->y_size,
                           op->out_buffer->x_size, op->out_buffer->offset, (void *)op->out_buffer);
                    break;
                }
                case(unary_negate): {
                    printf("U ngt [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p]\n", op->out_buffer->a_inherent, op->out_buffer->z_inherent,
                           op->out_buffer->y_inherent, op->out_buffer->x_inherent, op->out_buffer->a_size, op->out_buffer->z_size, op->out_buffer->y_size,
                           op->out_buffer->x_size, op->out_buffer->offset, (void *)op->out_buffer);
                    break;
                }
                case(unary_reciprocal): {
                    printf("U rcp [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p]\n", op->out_buffer->a_inherent, op->out_buffer->z_inherent,
                           op->out_buffer->y_inherent, op->out_buffer->x_inherent, op->out_buffer->a_size, op->out_buffer->z_size, op->out_buffer->y_size,
                           op->out_buffer->x_size, op->out_buffer->offset, (void *)op->out_buffer);
                    break;
                }
                case(unary_max): {
                    printf("U max [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf [%p]\n", op->out_buffer->a_inherent, op->out_buffer->z_inherent,
                           op->out_buffer->y_inherent, op->out_buffer->x_inherent, op->out_buffer->a_size, op->out_buffer->z_size, op->out_buffer->y_size,
                           op->out_buffer->x_size, op->out_buffer->offset, op->var_unary, (void *)op->out_buffer);
                    break;
                }
                case(unary_min): {
                    printf("U min [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf [%p]\n", op->out_buffer->a_inherent, op->out_buffer->z_inherent,
                           op->out_buffer->y_inherent, op->out_buffer->x_inherent, op->out_buffer->a_size, op->out_buffer->z_size, op->out_buffer->y_size,
                           op->out_buffer->x_size, op->out_buffer->offset, op->var_unary, (void *)op->out_buffer);
                    break;
                }
                case(unary_set): {
                    printf("U set [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf [%p]\n", op->out_buffer->a_inherent, op->out_buffer->z_inherent,
                           op->out_buffer->y_inherent, op->out_buffer->x_inherent, op->out_buffer->a_size, op->out_buffer->z_size, op->out_buffer->y_size,
                           op->out_buffer->x_size, op->out_buffer->offset, op->var_unary, (void *)op->out_buffer);
                    break;
                }
                case(unary_random): {
                    printf("U ran [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p]\n", op->out_buffer->a_inherent, op->out_buffer->z_inherent,
                           op->out_buffer->y_inherent, op->out_buffer->x_inherent, op->out_buffer->a_size, op->out_buffer->z_size, op->out_buffer->y_size,
                           op->out_buffer->x_size, op->out_buffer->offset, (void *)op->out_buffer);
                    break;
                }
                case(unary_tanh): {
                    printf("U tnh [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p]\n", op->out_buffer->a_inherent, op->out_buffer->z_inherent,
                           op->out_buffer->y_inherent, op->out_buffer->x_inherent, op->out_buffer->a_size, op->out_buffer->z_size, op->out_buffer->y_size,
                           op->out_buffer->x_size, op->out_buffer->offset, (void *)op->out_buffer);
                    break;
                }
                case(unary_absolute): {
                    printf("U abs [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p]\n", op->out_buffer->a_inherent, op->out_buffer->z_inherent,
                           op->out_buffer->y_inherent, op->out_buffer->x_inherent, op->out_buffer->a_size, op->out_buffer->z_size, op->out_buffer->y_size,
                           op->out_buffer->x_size, op->out_buffer->offset, (void *)op->out_buffer);
                    break;
                }
                case(unary_sign): {
                    printf("U sgn [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p]\n", op->out_buffer->a_inherent, op->out_buffer->z_inherent,
                           op->out_buffer->y_inherent, op->out_buffer->x_inherent, op->out_buffer->a_size, op->out_buffer->z_size, op->out_buffer->y_size,
                           op->out_buffer->x_size, op->out_buffer->offset, (void *)op->out_buffer);
                    break;
                }
            }
            break;
        }
        case(operation_binary): {
            switch(op->binary_type) {
                case(binary_add): {
                    printf("B add {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)op->out_buffer);
                    break;
                }
                case(binary_subtract): {
                    printf("B sub {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)op->out_buffer);
                    break;
                }
                case(binary_multiply): {
                    printf("B mul {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)op->out_buffer);
                    break;
                }
                case(binary_divide): {
                    printf("B div {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)op->out_buffer);
                    break;
                }
                case(binary_max): {
                    printf("B max {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)op->out_buffer);
                    break;
                }
                case(binary_min): {
                    printf("B min {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)op->out_buffer);
                    break;
                }
                case(binary_copy): {
                    printf("B cpy {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)op->out_buffer);
                    break;
                }
                case(binary_add_like): {
                    printf("B ldd {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)op->out_buffer);
                    break;
                }
                case(binary_subtract_like): {
                    printf("B lub {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)op->out_buffer);
                    break;
                }
                case(binary_multiply_like): {
                    printf("B lul {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)op->out_buffer);
                    break;
                }
                case(binary_divide_like): {
                    printf("B liv {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)op->out_buffer);
                    break;
                }
                case(binary_max_like): {
                    printf("B lax {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)op->out_buffer);
                    break;
                }
                case(binary_min_like): {
                    printf("B lin {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)op->out_buffer);
                    break;
                }
                case(binary_copy_like): {
                    printf("B lpy {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)op->out_buffer);
                    break;
                }
            }
            break;
        }
        case(operation_reduce): {
            switch(op->reduce_type) {
                case(reduce_sum): {
                    printf("R sum {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)(void *)op->out_buffer);
                    break;
                }
                case(reduce_avg): {
                    printf("R avg {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)op->out_buffer);
                    break;
                }
                case(reduce_max): {
                    printf("R max {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)op->out_buffer);
                    break;
                }
                case(reduce_min): {
                    printf("R min {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu [%p] [%p]\n", op->in_buffer->a_size, op->in_buffer->z_size,
                           op->in_buffer->y_size, op->in_buffer->x_size, op->in_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, (void *)op->in_buffer, (void *)op->out_buffer);
                    break;
                }
            }
            break;
        }
        case(operation_move): {
            switch(op->move_type) {
                case(move_reshape): {
                    printf("M rsp {%lu, %lu, %lu, %lu} %lu - {%lu, %lu, %lu, %lu} %lu [%p]\n", op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, op->var_a, op->var_z, op->var_y, op->var_x,
                           op->out_buffer->offset, (void *)op->out_buffer);
                    break;
                }
                case(move_resize): {
                    printf("M rsz {%lu, %lu, %lu, %lu} %lu - {%lu, %lu, %lu, %lu} %lu [%p]\n", op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, op->var_a, op->var_z, op->var_y, op->var_x,
                           op->out_buffer->offset, (void *)op->out_buffer);
                    break;
                }
                case(move_offset): {
                    printf("M off {%lu, %lu, %lu, %lu} %lu - {%lu, %lu, %lu, %lu} %lu [%p]\n", op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size, op->out_buffer->offset, op->out_buffer->a_size, op->out_buffer->z_size,
                           op->out_buffer->y_size, op->out_buffer->x_size,
                           op->out_buffer->a_stride * op->var_a + op->out_buffer->z_stride * op->var_z + op->out_buffer->y_stride * op->var_y +
                               op->out_buffer->x_stride * op->var_x,
                           (void *)op->out_buffer);
                    break;
                }
            }
            break;
        }
    }
}
void op_print(op_t *op, int padding, int offset, const char *name) {
    if(!op) { return; }
    if(strcmp(name, "")) { printf("%*s%s\n", offset, "", name); }
    if(op == NULL) {
        printf("%*sNULL\n", offset + padding, "");
    } else {
        op_single_print(op, padding, offset, "");
        for(uint64_t i = 0; i < op->parent_count; i++) { op_print(op->parent[i], padding, offset + padding, ""); }
    }
}
ALWAYS_INLINE void op_single_op_cpu_realize(op_t *op) {
    switch(op->type) {
        case(operation_unary): {
            switch(op->unary_type) {
                case(unary_add): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) { BUFFER_AT_(op->out_buffer, a, z, y, x) += op->var_unary; }
                            }
                        }
                    }
                    break;
                }
                case(unary_subtract): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) { BUFFER_AT_(op->out_buffer, a, z, y, x) -= op->var_unary; }
                            }
                        }
                    }
                    break;
                }
                case(unary_multiply): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) { BUFFER_AT_(op->out_buffer, a, z, y, x) *= op->var_unary; }
                            }
                        }
                    }
                    break;
                }
                case(unary_divide): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) { BUFFER_AT_(op->out_buffer, a, z, y, x) /= op->var_unary; }
                            }
                        }
                    }
                    break;
                }
                case(unary_exp): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) = exp(BUFFER_AT_(op->out_buffer, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_log): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) = log(BUFFER_AT_(op->out_buffer, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_square): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) *= BUFFER_AT_(op->out_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_sqrt): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) = sqrt(BUFFER_AT_(op->out_buffer, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_negate): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) = -BUFFER_AT_(op->out_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_reciprocal): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) = 1 / BUFFER_AT_(op->out_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_max): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    if(BUFFER_AT_(op->out_buffer, a, z, y, x) < op->var_unary) { BUFFER_AT_(op->out_buffer, a, z, y, x) = op->var_unary; }
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_min): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    if(BUFFER_AT_(op->out_buffer, a, z, y, x) > op->var_unary) { BUFFER_AT_(op->out_buffer, a, z, y, x) = op->var_unary; }
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_set): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) { BUFFER_AT_(op->out_buffer, a, z, y, x) = op->var_unary; }
                            }
                        }
                    }
                    break;
                }
                case(unary_random): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) = ((double)rand() / RAND_MAX) * 2 - 1;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_tanh): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) = tanh(BUFFER_AT_(op->out_buffer, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_absolute): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) = fabs(BUFFER_AT_(op->out_buffer, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_sign): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    if(BUFFER_AT_(op->out_buffer, a, z, y, x) < 0) {
                                        BUFFER_AT_(op->out_buffer, a, z, y, x) = -1;
                                    } else if(BUFFER_AT_(op->out_buffer, a, z, y, x) == 0) {
                                        BUFFER_AT_(op->out_buffer, a, z, y, x) = 0;
                                    } else {
                                        BUFFER_AT_(op->out_buffer, a, z, y, x) = 1;
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
            switch(op->binary_type) {
                case(binary_add): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) += BUFFER_AT_(op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_subtract): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) -= BUFFER_AT_(op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_multiply): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) *= BUFFER_AT_(op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_divide): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) /= BUFFER_AT_(op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_max): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    if(BUFFER_AT_(op->out_buffer, a, z, y, x) < BUFFER_AT_(op->in_buffer, a, z, y, x)) {
                                        BUFFER_AT_(op->out_buffer, a, z, y, x) = BUFFER_AT_(op->in_buffer, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_min): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    if(BUFFER_AT_(op->out_buffer, a, z, y, x) > BUFFER_AT_(op->in_buffer, a, z, y, x)) {
                                        BUFFER_AT_(op->out_buffer, a, z, y, x) = BUFFER_AT_(op->in_buffer, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_copy): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) = BUFFER_AT_(op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_add_like): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) += BUFFER_AT_(op->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_subtract_like): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) -= BUFFER_AT_(op->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_multiply_like): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) *= BUFFER_AT_(op->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_divide_like): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) /= BUFFER_AT_(op->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_max_like): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    if(BUFFER_AT_(op->out_buffer, a, z, y, x) < BUFFER_AT_(op->in_buffer, 0, 0, 0, 0)) {
                                        BUFFER_AT_(op->out_buffer, a, z, y, x) = BUFFER_AT_(op->in_buffer, 0, 0, 0, 0);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_min_like): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    if(BUFFER_AT_(op->out_buffer, a, z, y, x) > BUFFER_AT_(op->in_buffer, 0, 0, 0, 0)) {
                                        BUFFER_AT_(op->out_buffer, a, z, y, x) = BUFFER_AT_(op->in_buffer, 0, 0, 0, 0);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_copy_like): {
                    for(uint64_t a = 0; a < op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(op->out_buffer, a, z, y, x) = BUFFER_AT_(op->in_buffer, 0, 0, 0, 0);
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
                    for(uint64_t a = 0; a < op->in_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->in_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->in_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->in_buffer->x_size; x++) { temp += BUFFER_AT_(op->in_buffer, a, z, y, x); }
                            }
                        }
                    }
                    BUFFER_AT_(op->out_buffer, 0, 0, 0, 0) = temp;
                    break;
                }
                case(reduce_max): {
                    double temp = -INFINITY;
                    for(uint64_t a = 0; a < op->in_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->in_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->in_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->in_buffer->x_size; x++) {
                                    if(temp < BUFFER_AT_(op->in_buffer, a, z, y, x)) { temp = BUFFER_AT_(op->in_buffer, a, z, y, x); }
                                }
                            }
                        }
                    }
                    BUFFER_AT_(op->out_buffer, 0, 0, 0, 0) = temp;
                    break;
                }
                case(reduce_avg): {
                    double temp = 0;
                    for(uint64_t a = 0; a < op->in_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->in_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->in_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->in_buffer->x_size; x++) { temp += BUFFER_AT_(op->in_buffer, a, z, y, x); }
                            }
                        }
                    }
                    BUFFER_AT_(op->out_buffer, 0, 0, 0, 0) =
                        temp / (op->in_buffer->x_size * op->in_buffer->y_size * op->in_buffer->z_size * op->in_buffer->a_size);
                    break;
                }
                case(reduce_min): {
                    double temp = INFINITY;
                    for(uint64_t a = 0; a < op->in_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < op->in_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < op->in_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < op->in_buffer->x_size; x++) {
                                    if(temp > BUFFER_AT_(op->in_buffer, a, z, y, x)) { temp = BUFFER_AT_(op->in_buffer, a, z, y, x); }
                                }
                            }
                        }
                    }
                    BUFFER_AT_(op->out_buffer, 0, 0, 0, 0) = temp;
                    break;
                }
            }
            break;
        }
        case(operation_move): {
            switch(op->move_type) {
                case(move_reshape): {
                    op->out_buffer->a_size = op->var_a;
                    op->out_buffer->z_size = op->var_z;
                    op->out_buffer->y_size = op->var_y;
                    op->out_buffer->x_size = op->var_x;
                    op->out_buffer->a_stride = op->var_z * op->var_y * op->var_x;
                    op->out_buffer->z_stride = op->var_y * op->var_x;
                    op->out_buffer->y_stride = op->var_x;
                    op->out_buffer->x_stride = 1;
                    break;
                }
                case(move_resize): {
                    op->out_buffer->a_size = op->var_a;
                    op->out_buffer->z_size = op->var_z;
                    op->out_buffer->y_size = op->var_y;
                    op->out_buffer->x_size = op->var_x;
                    break;
                }
                case(move_offset): {
                    op->out_buffer->offset = op->out_buffer->a_stride * op->var_a + op->out_buffer->z_stride * op->var_z +
                                             op->out_buffer->y_stride * op->var_y + op->out_buffer->x_stride * op->var_x;
                    break;
                }
            }
            break;
        }
    }
}
void op_cpu_realize(op_t *op) {
    while(op->parent_count > 0) { op_cpu_realize(op->parent[0]); }
    op_single_op_cpu_realize(op);
    op_cleanup(op);
    op_free(op);
    free(op);
}

tensor_t tensor_alloc(uint64_t a, uint64_t z, uint64_t y, uint64_t x) {
    tensor_t tensor = {
        .op = NULL,
        .buffer = malloc(sizeof(buffer_t)),
    };
    assert(tensor.buffer);
    *tensor.buffer = buffer_alloc(a, z, y, x);
    return (tensor);
}
void tensor_free(tensor_t *tensor) {
    if(tensor->op) {
        op_free(tensor->op);
        free(tensor->op);
    }
    buffer_free(tensor->buffer);
    free(tensor->buffer);
}

/* NOTE: You can remove all these asserts if you want optimal performance. I don't recommend you do, because it makes the program less safe. */

void tensor_add_unary(tensor_t *tensor, double value) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_unary;
    tensor->op->unary_type = unary_add;
    tensor->op->var_unary = value;
    tensor->op->out_buffer = tensor->buffer;
}
void tensor_subtract_unary(tensor_t *tensor, double value) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_unary;
    tensor->op->unary_type = unary_subtract;
    tensor->op->var_unary = value;
    tensor->op->out_buffer = tensor->buffer;
}
void tensor_multiply_unary(tensor_t *tensor, double value) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_unary;
    tensor->op->unary_type = unary_multiply;
    tensor->op->var_unary = value;
    tensor->op->out_buffer = tensor->buffer;
}
void tensor_divide_unary(tensor_t *tensor, double value) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_unary;
    tensor->op->unary_type = unary_divide;
    tensor->op->var_unary = value;
    tensor->op->out_buffer = tensor->buffer;
}
void tensor_exp_unary(tensor_t *tensor) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_unary;
    tensor->op->unary_type = unary_exp;
    tensor->op->out_buffer = tensor->buffer;
}
void tensor_log_unary(tensor_t *tensor) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_unary;
    tensor->op->unary_type = unary_log;
    tensor->op->out_buffer = tensor->buffer;
}
void tensor_square_unary(tensor_t *tensor) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_unary;
    tensor->op->unary_type = unary_square;
    tensor->op->out_buffer = tensor->buffer;
}
void tensor_sqrt_unary(tensor_t *tensor) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_unary;
    tensor->op->unary_type = unary_sqrt;
    tensor->op->out_buffer = tensor->buffer;
}
void tensor_negate_unary(tensor_t *tensor) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_unary;
    tensor->op->unary_type = unary_negate;
    tensor->op->out_buffer = tensor->buffer;
}
void tensor_reciprocal_unary(tensor_t *tensor) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_unary;
    tensor->op->unary_type = unary_reciprocal;
    tensor->op->out_buffer = tensor->buffer;
}
void tensor_max_unary(tensor_t *tensor, double value) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_unary;
    tensor->op->unary_type = unary_max;
    tensor->op->var_unary = value;
    tensor->op->out_buffer = tensor->buffer;
}
void tensor_min_unary(tensor_t *tensor, double value) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_unary;
    tensor->op->unary_type = unary_min;
    tensor->op->var_unary = value;
    tensor->op->out_buffer = tensor->buffer;
}
void tensor_set_unary(tensor_t *tensor, double value) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_unary;
    tensor->op->unary_type = unary_set;
    tensor->op->var_unary = value;
    tensor->op->out_buffer = tensor->buffer;
}
void tensor_random_unary(tensor_t *tensor) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_unary;
    tensor->op->unary_type = unary_random;
    tensor->op->out_buffer = tensor->buffer;
}
void tensor_tanh_unary(tensor_t *tensor) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_unary;
    tensor->op->unary_type = unary_tanh;
    tensor->op->out_buffer = tensor->buffer;
}
void tensor_absolute_unary(tensor_t *tensor) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_unary;
    tensor->op->unary_type = unary_absolute;
    tensor->op->out_buffer = tensor->buffer;
}
void tensor_sign_unary(tensor_t *tensor) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_unary;
    tensor->op->unary_type = unary_sign;
    tensor->op->out_buffer = tensor->buffer;
}

void tensor_add_binary(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_binary;
    out->op->binary_type = binary_add;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}
void tensor_subtract_binary(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_binary;
    out->op->binary_type = binary_subtract;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}
void tensor_multiply_binary(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_binary;
    out->op->binary_type = binary_multiply;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}
void tensor_divide_binary(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_binary;
    out->op->binary_type = binary_divide;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}
void tensor_max_binary(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_binary;
    out->op->binary_type = binary_max;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}
void tensor_min_binary(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_binary;
    out->op->binary_type = binary_min;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}
void tensor_copy_binary(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_binary;
    out->op->binary_type = binary_copy;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}
void tensor_add_like_binary(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_binary;
    out->op->binary_type = binary_add_like;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}
void tensor_subtract_like_binary(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_binary;
    out->op->binary_type = binary_subtract_like;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}
void tensor_multiply_like_binary(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_binary;
    out->op->binary_type = binary_multiply_like;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}
void tensor_divide_like_binary(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_binary;
    out->op->binary_type = binary_divide_like;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}
void tensor_max_like_binary(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_binary;
    out->op->binary_type = binary_max_like;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}
void tensor_min_like_binary(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_binary;
    out->op->binary_type = binary_min_like;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}
void tensor_copy_like_binary(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_binary;
    out->op->binary_type = binary_copy_like;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}

/* Since reduce always overwrites `out` an optimizer should remove all the parent ops of out. This is a little complex to make sure that needed reduces are stored in the `in` op structure. */
void tensor_sum_reduce(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_reduce;
    out->op->reduce_type = reduce_sum;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}
/* Since reduce always overwrites `out` an optimizer should remove all the parent ops of out. This is a little complex to make sure that needed reduces are stored in the `in` op structure. */
void tensor_max_reduce(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_reduce;
    out->op->reduce_type = reduce_max;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}
/* Since reduce always overwrites `out` an optimizer should remove all the parent ops of out. This is a little complex to make sure that needed reduces are stored in the `in` op structure. */
void tensor_avg_reduce(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_reduce;
    out->op->reduce_type = reduce_avg;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}
/* Since reduce always overwrites `out` an optimizer should remove all the parent ops of out. This is a little complex to make sure that needed reduces are stored in the `in` op structure. */
void tensor_min_reduce(tensor_t *out, tensor_t *in) {
    op_t *out_parent = out->op;
    out->op = malloc(sizeof(op_t));
    assert(out->op);
    *out->op = op_alloc();
    op_add_parents(out->op, out_parent, in->op);
    out->op->tensor_base = out;
    if(out_parent) { out_parent->tensor_base = NULL; }
    out->op->type = operation_reduce;
    out->op->reduce_type = reduce_min;
    out->op->out_buffer = out->buffer;
    out->op->in_buffer = in->buffer;
}

void tensor_reshape_move(tensor_t *tensor, uint64_t a, uint64_t z, uint64_t y, uint64_t x) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_move;
    tensor->op->move_type = move_reshape;
    tensor->op->out_buffer = tensor->buffer;
    tensor->op->var_a = a;
    tensor->op->var_z = z;
    tensor->op->var_y = y;
    tensor->op->var_x = x;
}
void tensor_resize_move(tensor_t *tensor, uint64_t a, uint64_t z, uint64_t y, uint64_t x) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    *tensor->op = op_alloc();
    assert(tensor->op);
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_move;
    tensor->op->move_type = move_resize;
    tensor->op->out_buffer = tensor->buffer;
    tensor->op->var_a = a;
    tensor->op->var_z = z;
    tensor->op->var_y = y;
    tensor->op->var_x = x;
}
void tensor_offset_move(tensor_t *tensor, uint64_t a, uint64_t z, uint64_t y, uint64_t x) {
    op_t *parent = tensor->op;
    tensor->op = malloc(sizeof(op_t));
    assert(tensor->op);
    *tensor->op = op_alloc();
    op_add_parents(tensor->op, parent, NULL);
    tensor->op->tensor_base = tensor;
    if(parent) { parent->tensor_base = NULL; }
    tensor->op->type = operation_move;
    tensor->op->move_type = move_offset;
    tensor->op->out_buffer = tensor->buffer;
    tensor->op->var_a = a;
    tensor->op->var_z = z;
    tensor->op->var_y = y;
    tensor->op->var_x = x;
}

void tensor_cpu_realize(tensor_t *tensor) {
    if(tensor->op) { op_cpu_realize(tensor->op); }
}

/* If name is `""` it doesn't print a new empty line where the name would have been. */
void tensor_print(tensor_t *tensor, int padding, int offset, const char *name) {
    if(strcmp(name, "")) {
        printf("%*s%s CL_NAME: %s\n", offset, "", name, tensor->buffer->cl_name);
    } else {
        printf("%*s CL_NAME: %s\n", offset, "", tensor->buffer->cl_name);
    }
    for(uint64_t a = 0; a < tensor->buffer->a_size; a++) {
        if(a) {
            printf("\n");
            printf("\n");
        }
        for(uint64_t z = 0; z < tensor->buffer->z_size; z++) {
            if(z) { printf("\n"); }
            for(uint64_t y = 0; y < tensor->buffer->y_size; y++) {
                printf("%*s[ ", offset + padding, "");
                for(uint64_t x = 0; x < tensor->buffer->x_size; x++) { printf("% lf ", BUFFER_AT_(tensor->buffer, a, z, y, x)); }
                printf("]\n");
            }
        }
    }
}
const uint64_t a_max = 2;
const uint64_t z_max = 2;
const uint64_t y_max = 4;
const uint64_t x_max = 4;
/* Just prints a `{2, 2, 4, 4}` subsection of the tensor. If name is `""` it doesn't print a new empty line where the name would have been. */
void tensor_preview(tensor_t *tensor, int padding, int offset, const char *name) {
    if(strcmp(name, "")) {
        printf("%*s%s CL_NAME: %s\n", offset, "", name, tensor->buffer->cl_name);
    } else {
        printf("%*s CL_NAME: %s\n", offset, "", tensor->buffer->cl_name);
    }
    for(uint64_t a = 0; a < tensor->buffer->a_size; a++) {
        if(a >= a_max) {
            printf("%*s...\n\n", offset, "");
            break;
        }
        if(a) { printf("\n\n"); }
        for(uint64_t z = 0; z < tensor->buffer->z_size; z++) {
            if(z >= z_max) {
                printf("%*s...\n", offset, "");
                break;
            }
            if(z) { printf("\n"); }
            for(uint64_t y = 0; y < tensor->buffer->y_size; y++) {
                if(y >= y_max) {
                    printf("%*s...\n", offset + padding, "");
                    break;
                }
                printf("%*s[ ", offset + padding, "");
                for(uint64_t x = 0; x < tensor->buffer->x_size; x++) {
                    if(x >= x_max) {
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
