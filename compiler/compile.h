#ifndef COMPILER_COMPILE_H
#define COMPILER_COMPILE_H

/* I know it's more of a transpiler but who's counting */
/* I guess gathering all the information for the codegen / optimizations should happen in here */
/* Hmmm... Not sure though */

#include "../tensor.h"
#include "CL/cl.h"

typedef struct {
    uint64_t off_in;
    uint64_t off_out;
    uint64_t str_a_in;
    uint64_t str_z_in;
    uint64_t str_y_in;
    uint64_t str_x_in;
    uint64_t str_a_out;
    uint64_t str_z_out;
    uint64_t str_y_out;
    uint64_t str_x_out;
    uint64_t wai_a_in;
    uint64_t wai_z_in;
    uint64_t wai_y_in;
    uint64_t wai_x_in;
    uint64_t wai_a_out;
    uint64_t wai_z_out;
    uint64_t wai_y_out;
    uint64_t wai_x_out;
    uint64_t res_a_in;
    uint64_t res_z_in;
    uint64_t res_y_in;
    uint64_t res_x_in;
    uint64_t res_a_out;
    uint64_t res_z_out;
    uint64_t res_y_out;
    uint64_t res_x_out;
} dim_info_t;

typedef struct {
    uint64_t repeat_num;
    uint64_t group_len;
    op_t *op;
    dim_info_t *dim_info;
} op_group_t;

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
extern program_t program_compile(const linearized_t *linearized, const cl_device_id *device_id,
                                 const cl_context *context, const cl_command_queue *command_queue,
                                 const uint64_t global_size, const uint64_t local_size);
extern void program_free(program_t *program);

#endif
