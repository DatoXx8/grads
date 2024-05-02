#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "compile.h"
#include "nn.h"
#include "tensor.h"
#include "utils.h"

/*
 *  TODO: Make OpenCL work with the in memory programs.
 *  TODO: Refactor op-trees to deep copy the op tree to make sure flipping tensors would work (think of the linearizer
 * simulator debacle and why that broke).
 *  TODO: Write more tiger-beetle style tests.
 *      Compilation gives the same results (within error) in all compilation options.
 *      Test compiler edge cases.
 *  TODO: Fix inlining ops that already have stuff inlined. (Might not be necessary when you think about it.)
 *  TODO: Make reduce backprop real and not fake.
 *  TODO: Maybe remove explicit backprop and make autograd things.
 *  TODO: FLOP/S estimator.
 *  TODO: Support SYCL, that seems pretty neat.
 *  TODO: Update README with installation and usage guides.
 *
 *  Idea for chess engine: Train solely on fischer-random self play.
 *                         Bunch of different heads with a core net.
 */

int main(void) {
    const uint32_t RNG = time(NULL);
    printf("RNG Seed %u\n", RNG);
    // srand(rng);
    INIT_TIMER();

    START_TIME();

    const double LEARNING = 1e-2;
    const int64_t LAYERS = 2;
    const int64_t INPUT_CHANNELS = 2;
    const int64_t INPUT_Y = 4;
    const int64_t INPUT_X = INPUT_Y;
    layerconfig_t **layerconfig = calloc(LAYERS, sizeof(layerconfig_t *));
    assert(layerconfig);
    layerconfig_t l0 = {
        .layer_type = layer_input,
        .input_channels = INPUT_CHANNELS,
        .input_y = INPUT_Y,
        .input_x = INPUT_X,
    };
    layerconfig_t l1 = {
        .layer_type = layer_convolution,
        .norm_type = norm_none,
        .convolution_filters = 2,
        .convolution_kernel_size = 3,
        .convolution_kernel_stride = 1,
        .convolution_kernel_padding = 1,
        .activation_function = activation_identity,
    };
    // layerconfig_t l2 = {
    //     .layer_type = layer_split,
    //     .norm_type = norm_none,
    //     .split_filters = 2,
    //     .activation_function = activation_identity,
    // };
    // layerconfig_t l3 = {
    //     .layer_type = layer_reduce,
    //     .reduce_type = layer_reduce_max,
    //     .reduce_kernel_size = 2,
    //     .reduce_kernel_stride = 1,
    // };
    // layerconfig_t l4 = {
    //     .layer_type = layer_dense,
    //     .norm_type = norm_none,
    //     .dense_output_size = 3,
    //     .activation_function = activation_identity,
    // };
    layerconfig[0] = &l0;
    layerconfig[1] = &l1;
    // layerconfig[2] = &l2;
    // layerconfig[3] = &l3;
    // layerconfig[4] = &l4;

    neuralnet_t neuralnet = neuralnet_alloc(LAYERS, layerconfig, LEARNING);
    neuralnet_random(&neuralnet);

    const int64_t SAMPLES = 1;
    tensor_t input = tensor_alloc(SAMPLES, NEURALNET_INPUT(neuralnet).activation->buffer->sze_z,
                                  NEURALNET_INPUT(neuralnet).activation->buffer->sze_y,
                                  NEURALNET_INPUT(neuralnet).activation->buffer->sze_x);
    tensor_t output = tensor_alloc(SAMPLES, NEURALNET_OUTPUT(neuralnet).activation->buffer->sze_z,
                                   NEURALNET_OUTPUT(neuralnet).activation->buffer->sze_y,
                                   NEURALNET_OUTPUT(neuralnet).activation->buffer->sze_x);
    program_t program = {0};
    program_compile(&program, neuralnet.forward);
    for(int64_t i = 0; i < program.kernel_num; i++) { printf("%s\n", program.kernel[i].source); }

    program_free(&program);
    neuralnet_free(&neuralnet);
    tensor_free(&input);
    tensor_free(&output);
    free(layerconfig);

    STOP_TIME();
    PRINT_TIME();

    return 0;
}
