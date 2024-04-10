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

/* TODO: Different optimisation enums for debugging. No-Copy, fusing ops once, twice etc. */

/* These are instructions that repeat and only the offsets change, not the relative differences if ya catch my drift. */
typedef struct {
    uint64_t loop_number;
    uint64_t loop_length;
    simple_op_t **loop_instance;
    /* These following ones are essentialy x[number_of_tensors <= loop_length * 2][4]. In tensors are at x[even] and out tensors are at x[odd], and the 4 is cuz
     * of the 4 possible dimensions a tensor can have. */
    /* Initial offset per dimension. */
    uint64_t *per_dim_off_a;
    uint64_t *per_dim_off_z;
    uint64_t *per_dim_off_y;
    uint64_t *per_dim_off_x;
    /* Step size of the offset per dimension. */
    uint64_t *per_dim_str_a;
    uint64_t *per_dim_str_z;
    uint64_t *per_dim_str_y;
    uint64_t *per_dim_str_x;
    /* Total number of loops instances to go trough to get back to the same offset (equal to -1 or smth if it doesn't repeat). Kinda takes the role of a modulo
     * operation. */
    uint64_t *per_dim_reset_a;
    uint64_t *per_dim_reset_z;
    uint64_t *per_dim_reset_y;
    uint64_t *per_dim_reset_x;
    /* Number of loops to go through, before incrementing by the per dimension stride. */
    uint64_t *per_dim_wait_a;
    uint64_t *per_dim_wait_z;
    uint64_t *per_dim_wait_y;
    uint64_t *per_dim_wait_x;
} compile_loop_t;
/* Arguments names, number of arguments, kernel name and other stuff like that. These should exist for each compile option. */
typedef struct {
} cl_kernel_t;
typedef struct {
} cl_program_t;

extern void compile_linearized_to_cl(const char *filename, linearized_t *linearized);

#endif /* COMPILE_H_ */
