#ifndef CGRAD_COMPILE_H_
#define CGRAD_COMPILE_H_

#include <CL/cl.h>
#include <stdint.h>

#include "tensor.h"

typedef struct {
    uint64_t *off_in;
    uint64_t *off_out;
} dim_info_t;

/* TODO: Get rid off this and just have rename `compile_loop_t` to `op_loop_t` */
typedef struct {
    uint64_t loop_num;
    uint64_t loop_len;
    op_t *op;
    dim_info_t *dim_info;
} simple_loop_t;

typedef enum {
    inline_op_none = 0,
    inline_op_in,
    inline_op_out,
} inline_op_e;

typedef struct {
    uint64_t loop_num;
    uint64_t op_num;
    op_t **op;
    dim_info_t **dim_info;
    inline_op_e **inline_type;
    uint64_t *inline_num;
    uint64_t *inline_cap;
} compile_loop_t;

#define KERNEL_NAME "k"
typedef struct {
    char **arg_name;
    cl_mem *arg_mem;
    uint64_t arg_num;
    uint64_t arg_cap;
    char *source;
    uint64_t source_len;
    uint64_t source_cap;
    cl_kernel cl_kernel;
    cl_program cl_program;
} kernel_t;
typedef struct {
    uint64_t kernel_num;
    uint64_t kernel_cap;
    kernel_t *kernel;
    uint64_t global_size;
    uint64_t local_size;
    cl_device_id *cl_device_id;
    cl_context *cl_context;
    cl_command_queue *cl_command_queue;
} program_t;

/* Could also be called `program_alloc()` */
extern void program_compile(program_t *program, const linearized_t *linearized, const cl_device_id *device_id,
                            const cl_context *context, const cl_command_queue *command_queue,
                            const uint64_t global_size, const uint64_t local_size);
extern void program_free(program_t *program);

#endif /* COMPILE_H_ */
