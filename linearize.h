#ifndef LINEARIZE_H_
#define LINEARIZE_H_

/* The purpose of this is to turn the op tree structure into an array of ops that can then be fed to the GPU via OpenCL. */

#include <stdint.h>

#include "tensor.h"

typedef struct {
    int64_t sze_a;
    int64_t sze_z;
    int64_t sze_y;
    int64_t sze_x;
    int64_t str_a;
    int64_t str_z;
    int64_t str_y;
    int64_t str_x;
    int64_t off_a;
    int64_t off_z;
    int64_t off_y;
    int64_t off_x;
    int64_t off;
    double *val;
    char name[BUFFER_NAME_SIZE + 1];
} simple_buffer_t;

#define SIMPLE_AT(simple, a, z, y, x)                                                                                                                          \
    (simple).val[a * (simple).str_a + z * (simple).str_z + y * (simple).str_y + x * (simple).str_x + (simple).off]
#define SIMPLE_AT_(simple, a, z, y, x)                                                                                                                         \
    (simple)->val[a * (simple)->str_a + z * (simple)->str_z + y * (simple)->str_y + x * (simple)->str_x + (simple)->off]

/* NOTE: Actually fusing in is not so basic given my way of compiling things. */
typedef struct {
    enum operation_e type;
    enum unary_e type_unary;
    enum binary_e type_binary;
    enum reduce_e type_reduce;
    double var_unary;
    simple_buffer_t buffer_out;
    simple_buffer_t buffer_in;
} simple_op_t;

extern void simple_op_convert(simple_op_t *simple, op_t *op);
extern void simple_op_print(simple_op_t *simple, int padding, int offset, const char *name);
extern void simple_op_realize(simple_op_t *simple);

#define SIMPLE_OP_PRINT(simple) (simple_op_print(&(simple_op), 4, 0, (#simple)))
#define SIMPLE_OP_PRINT_(simple) (simple_op_print((simple_op), 4, 0, (#simple)))

typedef struct {
    int64_t op_count;
    int64_t op_capacity;
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
