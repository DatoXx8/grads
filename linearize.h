#ifndef LINEARIZE_H_
#define LINEARIZE_H_

/* The purpose of this is to turn the op tree structure into an array of ops that can then be fed to the GPU via OpenCL. */

#include <stdint.h>

#include "tensor.h"

/* NOTE: Actually fusing in is not so basic given my way of compiling things. */
typedef struct {
    enum operation_e type;
    enum unary_e unary_type;
    enum binary_e binary_type;
    enum reduce_e reduce_type;
    enum move_e move_type;
    double var_unary;
    uint64_t var_a;
    uint64_t var_z;
    uint64_t var_y;
    uint64_t var_x;
    buffer_t *out_buffer;
    buffer_t *in_buffer;
} simple_op_t;

extern void simple_op_convert(simple_op_t *simple_op, op_t *op);
extern void simple_op_print(simple_op_t *simple_op, int padding, int offset, const char *name);
extern void simple_op_realize(simple_op_t *simple_op);

typedef struct {
    uint64_t op_count;
    uint64_t op_capacity;
    simple_op_t *simple;
} linearized_t;

extern linearized_t linearized_alloc(void);
/* NOTE: `op` should be the root of the tree. */
extern void linearized_from_op(linearized_t *linearized, op_t *op);
extern void linearized_free(linearized_t *linearized);
extern void linearized_clear(linearized_t *linearized);
extern void linearized_print(linearized_t *linearized, int padding, int offset, const char *name);

#endif
