#ifndef COMPILE_H_
#define COMPILE_H_

/*
    1. Function recognizes loops in linearized_t (could also be of size 1, meaning a singular op)
    2. Function assigns loops to work groups
    3. Each loop gets split up into multiple work items
 */

#include <CL/cl.h>
#include <stdint.h>

#include "linearize.h"
#include "tensor.h"

typedef struct {
    int64_t *off_in;
    int64_t *off_out;
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

typedef struct {
    const char *name;
    char **arg_name;
    cl_mem *arg_mem;
    int64_t arg_num;
    int64_t arg_cap;
    int64_t size_global;
    int64_t size_local;
    char *source;
    int64_t source_len;
    int64_t source_cap;
    cl_kernel *cl_kernel;
} kernel_t;
typedef struct {
    kernel_t *kernel;
    int64_t kernel_num;
    char *source;
    int64_t source_len;
    cl_program *cl_program;
    cl_device_id *cl_device_id; /* Has to be done like this if we want these all the programs to have the same
                                   `device_id`s and all that other stuff. This is not technically necessary, but then
                                   `program_free` would ne very strange. */
    cl_context *cl_context;
    cl_command_queue *cl_command_queue;
} program_t;

/* Could also be called `program_alloc()`. */
extern void program_compile(program_t *program, linearized_t *linearized, cl_device_id *device_id, cl_context *context,
                            cl_command_queue *command_queue);
extern void program_free(program_t *program);

#endif /* COMPILE_H_ */
