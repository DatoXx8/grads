#include <CL/cl.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn.h"
#include "runtimes/cl.h"
#include "tensor.h"
#include "utils.h"

/*
 *  TODO: Rewrite compiler from the ground up. That thing horrible
 *  TODO: Support `local_size > 1` (Maybe do work-groups and work-items as parameters for `program_compile()` so that
 * `global_size` is guaranteed to be a multiple of `local_size`)
 *  TODO: Add multi-thread c runtime
 *  TODO: Fix inlining ops that already have stuff inlined. (Might not be necessary when you think about it.)
 *  TODO: Make reduce backprop real and not fake.
 *  TODO: Maybe remove explicit backprop and make autograd things.
 *  TODO: FLOP/S estimator.
 *  TODO: Update README with installation and usage guides.
 *  TODO: Investigate OpenCL apparent memory leaks. Valgrind does not find memory leaks in my code but still the memory
 * usage is *super* high and seems to be rising. Also investige the OpenCL compiler being stupidly slow
 *  TODO: Make OpenCL opt-out with a -U<macro> flag (Unsure about this one cuz it makes the code very ugly)
 *  TODO: Train solely on chess960 self play. Bunch of different heads with a core net. Piece placing chess is very
 * interesting.
 *  TODO: Make a go engine.
 */

void usage_print(const char *program_name) {
    assert(program_name);
    printf("USAGE: %s [runtime]\n"
           "    -cl   for using OpenCL\n"
           "    -c    for using C\n",
           program_name);
}

int main(int argc, const char **argv) {
    // const uint32_t RNG = time(NULL);
    const uint32_t RNG = 1716482642;
    printf("INFO: RNG Seed %u\n", RNG);
    srand(RNG);
    compile_e compile_type;
    if(argc != 2) {
        usage_print(argv[0]);
        ERROR("Program expects an argument\n");
    }
    if(!strncmp(argv[1], "-cl", 3)) {
        printf("INFO: Using OpenCL\n");
        compile_type = compile_cl;
    } else if(!strncmp(argv[1], "-c", 2)) {
        printf("INFO: Not using OpenCL\n");
        compile_type = compile_none;
    } else {
        usage_print(argv[0]);
        ERROR("Invaling argument\n");
    }
    cl_device_id device_id;
    cl_context context;
    if(compile_type == compile_cl) {
        int err;
        device_id = cl_device_get();
        context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    } else {
        context = NULL;
    }

    INIT_TIMER();
    START_TIME();

    const int64_t SAMPLES = 1;
    const double LEARNING = 1e-2;
    const int64_t LAYERS = 2;
    const int64_t INPUT_Z = 2;
    const int64_t INPUT_Y = 3;
    const int64_t INPUT_X = INPUT_Y;
    layerconfig_t *layerconfig = calloc(LAYERS, sizeof(layerconfig_t));
    assert(layerconfig);
    layerconfig_t l0 = {
        .layer_type = layer_input,
        .input_z = INPUT_Z,
        .input_y = INPUT_Y,
        .input_x = INPUT_X,
    };
    layerconfig_t l1 = {
        .layer_type = layer_convolution,
        .norm_type = norm_none,
        .convolution_filters = 2,
        .convolution_kernel_size = 2,
        .convolution_kernel_stride = 1,
        .convolution_kernel_padding = 0,
        .activation_function = activation_none,
    };
    layerconfig_t l2 = {
        .layer_type = layer_split,
        .norm_type = norm_none,
        .split_filters = 2,
        .activation_function = activation_none,
    };
    layerconfig_t l3 = {
        .layer_type = layer_reduce,
        .reduce_type = layer_reduce_max,
        .reduce_kernel_size = 2,
        .reduce_kernel_stride = 1,
    };
    layerconfig_t l4 = {
        .layer_type = layer_dense,
        .norm_type = norm_none,
        .dense_output_size = 3,
        .activation_function = activation_none,
    };
    layerconfig[0] = l0;
    layerconfig[1] = l4;
    // layerconfig[1] = l1;
    // layerconfig[2] = l2;
    // layerconfig[3] = l3;
    // layerconfig[4] = l4;

    neuralnet_t neuralnet = neuralnet_alloc(LAYERS, layerconfig, LEARNING, compile_type);
    tensor_t input = tensor_alloc(SAMPLES, NEURALNET_INPUT(neuralnet).activation->buffer->sze_z,
                                  NEURALNET_INPUT(neuralnet).activation->buffer->sze_y,
                                  NEURALNET_INPUT(neuralnet).activation->buffer->sze_x, context);
    tensor_t output = tensor_alloc(SAMPLES, NEURALNET_OUTPUT(neuralnet).activation->buffer->sze_z,
                                   NEURALNET_OUTPUT(neuralnet).activation->buffer->sze_y,
                                   NEURALNET_OUTPUT(neuralnet).activation->buffer->sze_x, context);
    tensor_unary_random(&input);
    tensor_unary_random(&output);
    tensor_realize(&input);
    tensor_realize(&output);
    neuralnet_random(&neuralnet);

    neuralnet_forward(&neuralnet, &input);
    TENSOR_PRINT_(NEURALNET_OUTPUT(neuralnet).activation);
    TENSOR_PRINT(input);

    neuralnet_free(&neuralnet);
    tensor_free(&input);
    tensor_free(&output);
    free(layerconfig);

    STOP_TIME();
    PRINT_TIME("main");

    return 0;
}
