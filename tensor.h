#ifndef TENSOR_H_
#define TENSOR_H_

#include <stdint.h>

#include "utils.h"

#define BUFFER_NAME_SIZE 16
typedef struct {
    int64_t inh_a;
    int64_t inh_z;
    int64_t inh_y;
    int64_t inh_x;
    int64_t str_a;
    int64_t str_z;
    int64_t str_y;
    int64_t str_x;
    int64_t sze_a;
    int64_t sze_z;
    int64_t sze_y;
    int64_t sze_x;
    int64_t off;
    double *val;
    char name[BUFFER_NAME_SIZE + 1];
    int64_t str_a_sim;
    int64_t str_z_sim;
    int64_t str_y_sim;
    int64_t str_x_sim;
    int64_t sze_a_sim;
    int64_t sze_z_sim;
    int64_t sze_y_sim;
    int64_t sze_x_sim;
    int64_t off_a_sim;
    int64_t off_z_sim;
    int64_t off_y_sim;
    int64_t off_x_sim;
    int64_t off_sim;
} buffer_t;

extern buffer_t buffer_alloc(int64_t a, int64_t z, int64_t y, int64_t x);
extern void buffer_free(buffer_t *buffer);

#define BUFFER_AT(buffer, a, z, y, x) ((buffer).val[(buffer).str_a * (a) + (buffer).str_z * (z) + (buffer).str_y * (y) + (buffer).str_x * (x) + (buffer).off])
#define BUFFER_AT_(buffer, a, z, y, x)                                                                                                                         \
    ((buffer)->val[(buffer)->str_a * (a) + (buffer)->str_z * (z) + (buffer)->str_y * (y) + (buffer)->str_x * (x) + (buffer)->off])

enum operation_e { operation_unary, operation_binary, operation_reduce, operation_move };
enum unary_e {
    unary_add,
    unary_subtract,
    unary_multiply,
    unary_divide,
    unary_exp,
    unary_log,
    unary_square,
    unary_sqrt,
    unary_reciprocal,
    unary_max,
    unary_min,
    unary_set,
    /* Never *ever* use this for things like encryption, where the randomnes of the numbers is important! */
    unary_random,
    unary_tanh,
    unary_absolute,
    unary_sign
};
enum binary_e {
    binary_add,
    binary_subtract,
    binary_multiply,
    binary_divide,
    binary_max,
    binary_min,
    binary_copy,
    /* NOTE: Use these as their respective unary ops, but the unary_value is not constant and instead provided by the in_buffer, that has to have a shape of
       `{1, 1, 1, 1}`*/
    binary_add_like,
    binary_subtract_like,
    binary_multiply_like,
    binary_divide_like,
    binary_max_like,
    binary_min_like,
    binary_copy_like
};
enum reduce_e { reduce_sum, reduce_max, reduce_avg, reduce_min };
/* NOTE: Move ops have 0 cost at runtime. */
enum move_e { move_reshape, move_resize, move_offset };

#define MAX_DEPTH 1000000
/* TODO: Could maybe merge all the enums for a smaller op_t struct. */
typedef struct op {
    void *tensor_base;
    int64_t parent_count;
    int64_t parent_capacity;
    struct op **parent;
    int64_t child_count;
    int64_t child_capacity;
    struct op **child;
    enum operation_e type;
    enum unary_e type_unary;
    enum binary_e type_binary;
    enum reduce_e type_reduce;
    enum move_e type_move;
    double var_unary;
    int64_t var_a;
    int64_t var_z;
    int64_t var_y;
    int64_t var_x;
    buffer_t *buffer_out;
    buffer_t *buffer_in;
} op_t;

extern op_t op_alloc(void);
extern void op_add_parents(op_t *op, op_t *output_parent, op_t *input_parent);
extern void op_free(op_t *op);
extern void op_cleanup(op_t *op);
extern void op_single_print(op_t *op, int padding, int offset, const char *name);
extern void op_print(op_t *op, int padding, int offset, const char *name);
extern void op_single_op_cpu_realize(op_t *op);
extern void op_cpu_realize(op_t *op);
extern void op_tree(op_t *op);

#define OP_PRINT(op) op_print(&op, 4, 0, (#op))
#define OP_PRINT_(op) op_print(op, 4, 0, (#op))

typedef struct {
    buffer_t *buffer;
    op_t *op;
} tensor_t;

extern tensor_t tensor_alloc(int64_t a, int64_t z, int64_t y, int64_t x);
extern void tensor_free(tensor_t *tensor);

extern void tensor_unary_set(tensor_t *tensor, double value);
extern void tensor_unary_add(tensor_t *tensor, double value);
extern void tensor_unary_subtract(tensor_t *tensor, double value);
extern void tensor_unary_multiply(tensor_t *tensor, double value);
extern void tensor_unary_divide(tensor_t *tensor, double value);
extern void tensor_unary_exp(tensor_t *tensor);
extern void tensor_unary_log(tensor_t *tensor);
extern void tensor_unary_square(tensor_t *tensor);
extern void tensor_unary_sqrt(tensor_t *tensor);
extern void tensor_unary_reciprocal(tensor_t *tensor);
/* Never *ever* use this for things like encryption, where the randomnes of the numbers is important! I don't know why you would do that in a ML framework but I digress. */
extern void tensor_unary_random(tensor_t *tensor);
extern void tensor_unary_tanh(tensor_t *tensor);
extern void tensor_unary_max(tensor_t *tensor, double value);
extern void tensor_unary_min(tensor_t *tensor, double value);
extern void tensor_unary_absolute(tensor_t *tensor);
extern void tensor_unary_sign(tensor_t *tensor);

extern void tensor_binary_add(tensor_t *out, tensor_t *in);
extern void tensor_binary_subtract(tensor_t *out, tensor_t *in);
extern void tensor_binary_multiply(tensor_t *out, tensor_t *in);
extern void tensor_binary_divide(tensor_t *out, tensor_t *in);
extern void tensor_binary_max(tensor_t *out, tensor_t *in);
extern void tensor_binary_min(tensor_t *out, tensor_t *in);
extern void tensor_binary_copy(tensor_t *out, tensor_t *in);
extern void tensor_lbinary_add(tensor_t *out, tensor_t *in);
extern void tensor_lbinary_subtract(tensor_t *out, tensor_t *in);
extern void tensor_lbinary_multiply(tensor_t *out, tensor_t *in);
extern void tensor_lbinary_divide(tensor_t *out, tensor_t *in);
extern void tensor_lbinary_max(tensor_t *out, tensor_t *in);
extern void tensor_lbinary_min(tensor_t *out, tensor_t *in);
extern void tensor_lbinary_copy(tensor_t *out, tensor_t *in);

extern void tensor_reduce_sum(tensor_t *out, tensor_t *in);
extern void tensor_reduce_max(tensor_t *out, tensor_t *in);
extern void tensor_reduce_avg(tensor_t *out, tensor_t *in);
extern void tensor_reduce_min(tensor_t *out, tensor_t *in);

extern void tensor_move_reshape(tensor_t *tensor, int64_t a, int64_t z, int64_t y, int64_t x);
extern void tensor_move_resize(tensor_t *tensor, int64_t a, int64_t z, int64_t y, int64_t x);
extern void tensor_move_offset(tensor_t *tensor, int64_t a, int64_t z, int64_t y, int64_t x);

extern void tensor_realize(tensor_t *tensor);

extern void tensor_print(tensor_t *tensor, int padding, int offset, const char *name);
extern void tensor_preview(tensor_t *tensor, int padding, int offset, const char *name);

#define TENSOR_PRINT(tensor) tensor_print(&tensor, 4, 0, (#tensor))
#define TENSOR_PRINT_(tensor) tensor_print(tensor, 4, 0, (#tensor))
#define TENSOR_PREVIEW(tensor) tensor_preview(&tensor, 4, 0, (#tensor))
#define TENSOR_PREVIEW_(tensor) tensor_preview(tensor, 4, 0, (#tensor))

#endif
