#ifndef LINEARIZE_H_
#define LINEARIZE_H_

/* The purpose of this is to turn the op tree structure into an array of ops that can then be fed to the GPU via OpenCL. */

#include <stdint.h>

#include "tensor.h"

typedef struct {
    uint64_t a_sze;
    uint64_t z_sze;
    uint64_t y_sze;
    uint64_t x_sze;
    uint64_t a_str;
    uint64_t z_str;
    uint64_t y_str;
    uint64_t x_str;
    uint64_t off;
    uint64_t a_off;
    uint64_t z_off;
    uint64_t y_off;
    uint64_t x_off;
    double *val;
    char name[BUFFER_NAME_SIZE + 1];
} simple_buffer_t;

#define SIMPLE_AT(simple, a, z, y, x)                                                                                                                          \
    (simple).val[a * (simple).a_str + z * (simple).z_str + y * (simple).y_str + x * (simple).x_str + (simple).off]
#define SIMPLE_AT_(simple, a, z, y, x)                                                                                                                         \
    (simple)->val[a * (simple)->a_str + z * (simple)->z_str + y * (simple)->y_str + x * (simple)->x_str + (simple)->off]

/* NOTE: Actually fusing in is not so basic given my way of compiling things. */
typedef struct {
    enum operation_e type;
    enum unary_e unary_type;
    enum binary_e binary_type;
    enum reduce_e reduce_type;
    double var_unary;
    simple_buffer_t out_buffer;
    simple_buffer_t in_buffer;
} simple_op_t;

extern void simple_op_convert(simple_op_t *simple, op_t *op);
extern void simple_op_print(simple_op_t *simple, int padding, int offset, const char *name);
extern void simple_op_realize(simple_op_t *simple);

#define SIMPLE_OP_PRINT(simple) (simple_op_print(&(simple_op), 4, 0, (#simple)))
#define SIMPLE_OP_PRINT_(simple) (simple_op_print((simple_op), 4, 0, (#simple)))

typedef struct {
    uint64_t op_count;
    uint64_t op_capacity;
    simple_op_t *simple;
} linearized_t;

extern linearized_t linearized_alloc(void);
/* NOTE: `op` should be the root of the tree. */
extern void linearized_free(linearized_t *linearized);
extern void linearized_from_op(linearized_t *linearized, op_t *op);
extern void linearized_run(linearized_t *linearized);
extern void linearized_clear(linearized_t *linearized);
extern void linearized_print(linearized_t *linearized, int padding, int offset, const char *name);

#define LINEARIZED_PRINT(linearized) (linearized_print(&(linearized), 4, 0, (#linearized)))
#define LINEARIZED_PRINT_(linearized) (linearized_print((linearized), 4, 0, (#linearized)))

#endif
