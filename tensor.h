#ifndef TENSOR_H_
#define TENSOR_H_

#include <stdint.h>

#include "utils.h"

typedef struct {
    uint64_t a_inherent;
    uint64_t z_inherent;
    uint64_t y_inherent;
    uint64_t x_inherent;
    uint64_t a_stride;
    uint64_t z_stride;
    uint64_t y_stride;
    uint64_t x_stride;
    uint64_t a_size;
    uint64_t z_size;
    uint64_t y_size;
    uint64_t x_size;
    uint64_t offset;
    double *values;
} buffer_t;

extern buffer_t buffer_alloc(uint64_t a, uint64_t z, uint64_t y, uint64_t x);
extern void buffer_free(buffer_t *buffer);

#define BUFFER_AT(buffer, a, z, y, x) ((buffer).values[(buffer).a_stride * a + (buffer).z_stride * z + (buffer).y_stride * y + (buffer).x_stride * x + (buffer).offset])
#define BUFFER_AT_(buffer, a, z, y, x) ((buffer)->values[(buffer)->a_stride * a + (buffer)->z_stride * z + (buffer)->y_stride * y + (buffer)->x_stride * x + (buffer)->offset])


enum operation_e : uint64_t {
    operation_unary, operation_binary, operation_reduce, operation_move
};
enum unary_e : uint64_t {
    unary_add, unary_subtract, unary_multiply, unary_divide,
    unary_exp, unary_log, unary_square, unary_sqrt,
    unary_negate, unary_reciprocal, unary_max, unary_min,
    unary_set, unary_zero /* Zero is made another seperate op for performance reasons. As explicit_bzero is *really* fast */
};
enum binary_e : uint64_t {
    binary_add, binary_subtract, binary_multiply, binary_divide,
    binary_max, binary_min, binary_copy
};
enum reduce_e : uint64_t {
    reduce_sum, reduce_max, reduce_avg, reduce_min
};
enum move_e : uint64_t {
    move_reshape, move_offset
};

typedef struct op {
    uint64_t parent_count;
    uint64_t parent_capacity;
    struct op **parent;
    uint64_t child_count;
    uint64_t child_capacity;
    struct op **child;
    enum operation_e type;
    enum unary_e unary_type;
    buffer_t *unary_buffer;
    double unary_value;
    enum binary_e binary_type;
    buffer_t *binary_out;
    buffer_t *binary_in;
    enum reduce_e reduce_type;
    buffer_t *reduce_out;
    buffer_t *reduce_in;
    enum move_e move_type;
    buffer_t *move_buffer;
    void *tensor_base;
    uint64_t move_a;
    uint64_t move_z;
    uint64_t move_y;
    uint64_t move_x;
} op_t;

extern op_t op_alloc(op_t *output_parent, op_t *input_parent);
extern void op_free(op_t *op);
extern void op_cleanup(op_t *op);
extern void op_single_op_cpu_realize(op_t *op);
extern void op_cpu_realize(op_t *op);
// extern void op_cl_realize(op_t *op); Need to have seperate linearize.h file for this and then a calc.cl that takes in the linearized operations instead of as a tree
extern void op_tree(op_t *op);

typedef struct {
    buffer_t *buffer;
    op_t *op;
} tensor_t;

extern tensor_t tensor_alloc(uint64_t a, uint64_t z, uint64_t y, uint64_t x);
extern void tensor_free(tensor_t *tensor);

#endif
