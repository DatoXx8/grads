/*
OPTIONS TO RUN LINEARIZED OPS:
    1. C ON THE CPU (DONE)
    2. NAIVE BY LAYER
    3. COMPILE
        3.1 BY LAYER (TRICKIER THAN EXPECTED)
        3.2 BY SECTOR
        3.3 BY NN

There might be a *lot* of optimization potential when compiling, i.e. only copy values, if they change before getting used again, otherwise use the value directly
 */
#ifndef RUN_H_
#define RUN_H_

#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <stdint.h>

#include "tensor.h"
#include "nn.h"
#include "linearize.h"

/* NOTE: If runtime_compile_sector is chosen, then the program will automatically infer the size of the sectors. */
enum runtime_e {
    runtime_c, runtime_layer, runtime_compile_layer, runtime_compile_sector, runtime_compile_nn
};

typedef struct {
    uint64_t a_stride;
    uint64_t z_stride;
    uint64_t y_stride;
    uint64_t x_stride;
    uint64_t a_size;
    uint64_t z_size;
    uint64_t y_size;
    uint64_t x_size;
    uint64_t offset;
    char cl_name[CL_NAME_SIZE + 1];
} cl_buffer_t;

typedef struct {
    enum operation_e type;
    enum unary_e unary_type;
    enum binary_e binary_type;
    enum reduce_e reduce_type;
    double var_unary;
    cl_buffer_t out_buffer;
    cl_buffer_t in_buffer;
} cl_op_t;

typedef struct {
    uint64_t cl_op_length;
    uint64_t cl_op_capacity;
    cl_op_t *cl_op;
} cl_linearized_t;

extern cl_linearized_t cl_linearized_alloc(void);
extern void cl_linearized_free(cl_linearized_t *cl_linearized);
extern void cl_linearized_build(cl_linearized_t *cl_linearized, linearized_t *linearized);
extern void cl_linearized_print(cl_linearized_t *cl_linearized, int padding, int offset, const char *name);

typedef struct {
    enum runtime_e type;
    /* C */
    linearized_t *linearized;

    /* COMPILED */
    uint64_t number_of_subprograms;
    cl_kernel *kernel;
    /* TODO: Check if these are backed into the cl_program. May have to store them anyway to free them when calling runtime_free. */
    
    cl_program *program;
    cl_context *context;
    cl_device_id *device;
    cl_command_queue *queue;
} runtime_t;

extern runtime_t runtime_alloc(enum runtime_e type, linearized_t *linearized);
extern void runtime_execute(runtime_t *runtime);
extern void runtime_free(runtime_t *runtime);


#endif
