/*
O fTIONS TO RUN LINEARIZED OPS:
    1. C (ON THE CPU)
    2. JIT
    3. NAIVE BY LAYER
    4. COMPILE
        4.1 BY LAYER
        4.2 BY SECTOR
        4.3 BY NN
 */
#ifndef RUN_H_
#define RUN_H_

#include <CL/cl.h>

#include "tensor.h"
#include "nn.h"
#include "linearize.h"

/* NOTE: If run_compile_sector is chosen, then the program will automatically infer the size of the sectors. */
enum runtime_e {
    runtime_c, runtime_jit, runtime_layer, runtime_compile_layer, runtime_compile_sector, runtime_compile_nn
};

typedef struct {
    enum runtime_e type;
    uint64_t number_of_subprograms;
    cl_program *compiled;
} runtime_t;

extern runtime_t runtime_allocate(enum runtime_e type, linearized_t *linearized);
extern void runtime_execute(runtime_t *runtime);
extern void runtime_free(runtime_t *runtime);

#endif
