#ifndef COMPILE_H_
#define COMPILE_H_

/*
    1. Function recognizes loops in linearized_t (could also be of size 1, meaning a singular op)
    2. Function assigns loops to work groups
    3. Each loop gets split up into multiple work items
 */

#include <stdint.h>

#include "linearize.h"
#include "tensor.h"

/* TODO: Add other compile languages like CUDA. */
/* TODO: Add compilation to fixed binaries and also just the "normal" compilation. */
enum compile_e { compile_none, compile_cl };

typedef struct {
    int64_t str_a_in;
    int64_t str_z_in;
    int64_t str_y_in;
    int64_t str_x_in;
    int64_t res_a_in;
    int64_t res_z_in;
    int64_t res_y_in;
    int64_t res_x_in;
    int64_t wai_a_in;
    int64_t wai_z_in;
    int64_t wai_y_in;
    int64_t wai_x_in;
    int64_t str_a_out;
    int64_t str_z_out;
    int64_t str_y_out;
    int64_t str_x_out;
    int64_t res_a_out;
    int64_t res_z_out;
    int64_t res_y_out;
    int64_t res_x_out;
    int64_t wai_a_out;
    int64_t wai_z_out;
    int64_t wai_y_out;
    int64_t wai_x_out;
} dim_info_t;

typedef struct {
    int64_t loop_num;
    int64_t loop_len;
    simple_op_t *op;
    dim_info_t *dim_info;
} simple_loop_t;
/* TODO: Maybe do this in an enum. */
#define OPTIMIZE_NONE (0UL)
#define OPTIMIZE_INLINE (1UL)
#define OPTIMIZE_FUSE (1UL << 1)
#define OPTIMIZE_ALL (OPTIMIZE_INLINE | OPTIMIZE_FUSE)
typedef struct {
    uint64_t optim;
    int64_t loop_num;
    int64_t loop_len;
    simple_op_t **op;
    dim_info_t **dim_info;
    int64_t *op_num;
    int64_t *op_cap;
} compile_loop_t;

/* Arguments names, number of arguments, kernel name and other stuff like that. These should exist for each compile option. */
typedef struct {
    const char *name;
    char **args;
    int64_t arg_num;
    int64_t global_size;
    int64_t local_size;
} cl_kernel_t;
typedef struct {
    cl_kernel_t *kernel;
    int64_t kernel_num;
    const char *filename;
} cl_program_t;

/* Could also be called `cl_program_alloc()`. */
extern int cl_program_compile(cl_program_t *program, const char *filename, linearized_t *linearized);
extern void cl_program_free(cl_program_t *program);

#endif /* COMPILE_H_ */
