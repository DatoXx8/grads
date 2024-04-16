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
    uint64_t loop_num;
    uint64_t loop_len;
    simple_op_t *loop_instance;
    /* These following ones are essentialy x[number_of_tensors <= loop_length * 2][4]. In tensors are at x[even] and out tensors are at x[odd], and the 4 is cuz
     * of the 4 possible dimensions a tensor can have.
     * Doing it like this is extremely natural if you think about the way I do it just in the normal case on the CPU. */
    uint64_t *per_dim_off_a;
    uint64_t *per_dim_off_z;
    uint64_t *per_dim_off_y;
    uint64_t *per_dim_off_x;
    uint64_t *per_dim_str_a;
    uint64_t *per_dim_str_z;
    uint64_t *per_dim_str_y;
    uint64_t *per_dim_str_x;
    uint64_t *per_dim_reset_a;
    uint64_t *per_dim_reset_z;
    uint64_t *per_dim_reset_y;
    uint64_t *per_dim_reset_x;
    uint64_t *per_dim_wait_a;
    uint64_t *per_dim_wait_z;
    uint64_t *per_dim_wait_y;
    uint64_t *per_dim_wait_x;
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
    uint64_t **per_dim_off_a;
    uint64_t **per_dim_off_z;
    uint64_t **per_dim_off_y;
    uint64_t **per_dim_off_x;
    uint64_t **per_dim_str_a;
    uint64_t **per_dim_str_z;
    uint64_t **per_dim_str_y;
    uint64_t **per_dim_str_x;
    uint64_t **per_dim_reset_a;
    uint64_t **per_dim_reset_z;
    uint64_t **per_dim_reset_y;
    uint64_t **per_dim_reset_x;
    uint64_t **per_dim_wait_a;
    uint64_t **per_dim_wait_z;
    uint64_t **per_dim_wait_y;
    uint64_t **per_dim_wait_x;
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
