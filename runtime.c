#include "tensor.h"
#include "nn.h"
#include "linearize.h"
#include "runtime.h"

/* TODO: How to get the layer starting points and sizes? */
static void runtime_compile_linearized_(runtime_t *runtime, linearized_t *linearized, uint64_t section_start_i, uint64_t section_length) {
}
static void runtime_run_c_(runtime_t *runtime) {
    for(uint64_t i = 0; i < runtime->linearized->op_count; i++) {
        simple_op_realize(&runtime->linearized->simple[i]);
    }
}
static void runtime_compile_layer_primitives_(runtime_t *runtime) {
}
runtime_t runtime_allocate(enum runtime_e type, linearized_t *linearized) {
    runtime_t runtime = {0};
    switch(type) {
        case(runtime_c): {
            runtime.type = runtime_c;
            runtime.linearized = linearized;
            break;
        }
        case(runtime_layer): {
            runtime.type = runtime_layer;
            break;
        }
        case(runtime_compile_layer): {
            runtime.type = runtime_compile_layer;
            break;
        }
        case(runtime_compile_sector): {
            /* TODO: Hmmm... How do I do this splitting into subsections for this? Max layers? Max operations? */
            runtime.type = runtime_compile_sector;
            break;
        }
        case(runtime_compile_nn): {
            runtime.type = runtime_compile_nn;
            break;
        }
    }
    return(runtime);
}
void runtime_free(runtime_t *runtime) {
}
void runtime_execute(runtime_t *runtime) {
    switch(runtime->type) {
        case(runtime_c): {
            runtime_run_c_(runtime);
            break;
        }
        case(runtime_layer): {
            break;
        }
        case(runtime_compile_layer): {
            break;
        }
        case(runtime_compile_sector): {
            /* TODO: Hmmm... How do I do this splitting into subsections for this? Max layers? Max operations? */
            break;
        }
        case(runtime_compile_nn): {
            break;
        }
    }
}
