/*
OPTIONS TO RUN LINEARIZED OPS:
    1. C ON THE CPU
    2. NAIVE BY LAYER
    3. COMPILE
        3.1 BY LAYER
        3.2 BY SECTOR
        3.3 BY NN
 */
#ifndef RUN_H_
#define RUN_H_

#include <CL/cl.h>

#include "tensor.h"
#include "nn.h"
#include "linearize.h"

/* NOTE: If run_compile_sector is chosen, then the program will automatically infer the size of the sectors. */
enum runtime_e {
    runtime_c, runtime_layer, runtime_compile_layer, runtime_compile_sector, runtime_compile_nn
};

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

extern runtime_t runtime_allocate(enum runtime_e type, linearized_t *linearized);
extern void runtime_execute(runtime_t *runtime);
extern void runtime_free(runtime_t *runtime);


#endif
