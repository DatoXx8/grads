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
    uint64_t off_a_in;
    uint64_t off_z_in;
    uint64_t off_y_in;
    uint64_t off_x_in;
    uint64_t str_a_in;
    uint64_t str_z_in;
    uint64_t str_y_in;
    uint64_t str_x_in;
    uint64_t reset_a_in;
    uint64_t reset_z_in;
    uint64_t reset_y_in;
    uint64_t reset_x_in;
    uint64_t wait_a_in;
    uint64_t wait_z_in;
    uint64_t wait_y_in;
    uint64_t wait_x_in;
    uint64_t off_a_out;
    uint64_t off_z_out;
    uint64_t off_y_out;
    uint64_t off_x_out;
    uint64_t str_a_out;
    uint64_t str_z_out;
    uint64_t str_y_out;
    uint64_t str_x_out;
    uint64_t reset_a_out;
    uint64_t reset_z_out;
    uint64_t reset_y_out;
    uint64_t reset_x_out;
    uint64_t wait_a_out;
    uint64_t wait_z_out;
    uint64_t wait_y_out;
    uint64_t wait_x_out;
} dim_info_t;

typedef struct {
    uint64_t loop_num;
    uint64_t loop_len;
    simple_op_t *loop_instance;
    dim_info_t *dim_info;
} simple_loop_t;
/* TODO: Maybe do this in an enum. */
#define OPTIMIZE_INLINE (1UL)
#define OPTIMIZE_FUSE (1UL << 1)
#define OPTIMIZE_ALL (OPTIMIZE_INLINE | OPTIMIZE_FUSE)
typedef struct {
    uint64_t optimizations;
    uint64_t loop_num;
    uint64_t loop_len;
    simple_op_t **op;
    dim_info_t **dim_info;
    uint64_t *op_num;
    uint64_t *op_cap;
} compile_loop_t;

/* Arguments names, number of arguments, kernel name and other stuff like that. These should exist for each compile option. */
typedef struct {
    const char *name;
    char **args;
    uint64_t arg_num;
    uint64_t global_size;
    uint64_t local_size;
} cl_kernel_t;
typedef struct {
    cl_kernel_t *kernel;
    uint64_t kernel_num;
    const char *filename;
} cl_program_t;

/* Could also be called `cl_program_alloc()`. */
extern cl_program_t compile_linearized_to_cl(const char *filename, linearized_t *linearized);
extern void cl_program_free(cl_program_t *program);

#endif /* COMPILE_H_ */
