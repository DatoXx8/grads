#ifndef TENSOR_H_
#define TENSOR_H_

typedef enum {
    sync_none = 0,
    sync_to_host,
    sync_to_device
} sync_e;

#include <CL/cl.h>
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
    int64_t off_a;
    int64_t off_z;
    int64_t off_y;
    int64_t off_x;
    int64_t off;
    double *val;
    cl_mem val_cl;
    sync_e sync;
    char name[BUFFER_NAME_SIZE + 1];
    /* 0 for "aaa...", 1 for "baa..." etc. Used for avoiding a bunch of string comparisons. Could maybe refactor to
     * generate the names based on these i.e. have `name_off == 12` generate the name `t0...012` */
    int64_t name_off;
} buffer_t;

extern buffer_t buffer_alloc(int64_t a, int64_t z, int64_t y, int64_t x, cl_context context);
extern void buffer_sync_realize(buffer_t *buffer, cl_command_queue command_queue);
extern void buffer_sync_update(buffer_t *buffer, sync_e sync);

#define BUFFER_AT(buffer, a, z, y, x)                                                                                  \
    ((buffer).val[(buffer).str_a * (a) + (buffer).str_z * (z) + (buffer).str_y * (y) + (buffer).str_x * (x) +          \
                  (buffer).off])
#define BUFFER_AT_(buffer, a, z, y, x)                                                                                 \
    ((buffer)->val[(buffer)->str_a * (a) + (buffer)->str_z * (z) + (buffer)->str_y * (y) + (buffer)->str_x * (x) +     \
                   (buffer)->off])

typedef enum {
    op_unary,
    op_binary,
    op_reduce,
    op_move
} op_e;
typedef enum {
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
    unary_random,
    unary_tanh,
    unary_absolute,
    unary_sign
} unary_e;
typedef enum {
    binary_add,
    binary_subtract,
    binary_multiply,
    binary_divide,
    binary_max,
    binary_min,
    binary_copy,
    /* Use these as their respective unary ops, but the unary_value is not constant and instead provided by the
       in_buffer, that has to have a shape of
       `{1, 1, 1, 1}`*/
    binary_add_like,
    binary_subtract_like,
    binary_multiply_like,
    binary_divide_like,
    binary_max_like,
    binary_min_like,
    binary_copy_like
} binary_e;
typedef enum {
    reduce_sum,
    reduce_max,
    reduce_avg,
    reduce_min
} reduce_e;
typedef enum {
    move_reshape,
    move_resize,
    move_offset
} move_e;

#define MAX_DEPTH (0x100000)
/* TODO: Could maybe merge all the enums for a smaller op_t struct */
typedef struct op {
    op_e type_op;
    unary_e type_unary;
    binary_e type_binary;
    reduce_e type_reduce;
    move_e type_move;
    double var_unary;
    int64_t var_a;
    int64_t var_z;
    int64_t var_y;
    int64_t var_x;
    buffer_t buffer_out;
    buffer_t buffer_in;
} op_t;

extern void op_realize(const op_t *op);
extern void op_print(const op_t *op, const int padding, const int offset, const char *name);

#define OP_PRINT(op) op_print(&(op), 4, 0, (#op))
#define OP_PRINT_(op) op_print((op), 4, 0, (#op))

typedef struct {
    int64_t op_len;
    int64_t op_cap;
    op_t *op;
} linearized_t;

extern linearized_t linearized_alloc(void);
extern void linearized_free(linearized_t *linearized);
extern void linearized_clear(linearized_t *linearized);
extern void linearized_run(const linearized_t *linearized);
extern void linearized_add_op(linearized_t *linearized, const op_t *op);
extern void linearized_append(linearized_t *linearized1, linearized_t *linearized2);
extern void linearized_print(const linearized_t *linearized, const int padding, const int offset, const char *name);

#define LINEARIZED_PRINT(linearized) (linearized_print(&(linearized), 4, 0, (#linearized)))
#define LINEARIZED_PRINT_(linearized) (linearized_print((linearized), 4, 0, (#linearized)))

typedef struct {
    buffer_t *buffer;
    linearized_t *linearized;
} tensor_t;

extern tensor_t tensor_alloc(const int64_t a, const int64_t z, const int64_t y, const int64_t x, cl_context context);
extern void tensor_free(tensor_t *tensor);

extern void tensor_unary_add(tensor_t *tensor, const double value);
extern void tensor_unary_subtract(tensor_t *tensor, const double value);
extern void tensor_unary_multiply(tensor_t *tensor, const double value);
extern void tensor_unary_divide(tensor_t *tensor, const double value);
extern void tensor_unary_set(tensor_t *tensor, const double value);
extern void tensor_unary_exp(tensor_t *tensor);
extern void tensor_unary_log(tensor_t *tensor);
extern void tensor_unary_square(tensor_t *tensor);
extern void tensor_unary_sqrt(tensor_t *tensor);
extern void tensor_unary_reciprocal(tensor_t *tensor);
extern void tensor_unary_random(tensor_t *tensor);
extern void tensor_unary_tanh(tensor_t *tensor);
extern void tensor_unary_max(tensor_t *tensor, const double value);
extern void tensor_unary_min(tensor_t *tensor, const double value);
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

extern void tensor_move_reshape(tensor_t *tensor, const int64_t a, const int64_t z, const int64_t y, const int64_t x);
extern void tensor_move_resize(tensor_t *tensor, const int64_t a, const int64_t z, const int64_t y, const int64_t x);
extern void tensor_move_offset(tensor_t *tensor, const int64_t a, const int64_t z, const int64_t y, const int64_t x);

extern void tensor_realize(tensor_t *tensor);

extern void tensor_print(const tensor_t *tensor, const int padding, const int offset, const char *name);
extern void tensor_preview(const tensor_t *tensor, const int padding, const int offset, const char *name);

#define TENSOR_PRINT(tensor) tensor_print(&(tensor), 4, 0, (#tensor))
#define TENSOR_PRINT_(tensor) tensor_print((tensor), 4, 0, (#tensor))
#define TENSOR_PREVIEW(tensor) tensor_preview(&(tensor), 4, 0, (#tensor))
#define TENSOR_PREVIEW_(tensor) tensor_preview((tensor), 4, 0, (#tensor))

#endif
