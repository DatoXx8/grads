#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "compile.h"
#include "nn.h"
#include "tensor.h"
#include "utils.h"

/*
    TODO: Neural net saving and loading to disk.
    TODO: Update README with installation and usage guides.
    TODO: Write SPEC with technical details of this stuff.
    TODO: Rename stuff to get the cl_ prefixes away.
    TODO: Think about how to parallelize from the linearized ops.
    TODO: Make a compiler from ops to parallelized OpenCL code.
    TODO: Make reduce backprop real and not fake.
    TODO: Maybe remove explicit backprop and make autograd things.
    TODO: Replace all strcpy and realloc with the safe n-byte max versions.
    TODO: Think about how to do handle things like stride 2 convolutions, since they might cause problems, because of the fact, that they might mess up get_global_id() cuz that might always increment by 1.

    FURTHER OUT
    TODO: FLOP/S estimator
*/

int main(void) {
    const uint64_t rng = time(NULL);
    printf("RNG Seed %lu\n", rng);
    // srand(rng);
    INIT_TIMER();

    START_TIME();

    const uint64_t layers = 2;
    const uint64_t input_channels = 2;
    const uint64_t input_y = 3;
    const uint64_t input_x = input_y;
    layerconfig_t **layerconfig = calloc(layers, sizeof(layerconfig_t *));
    layerconfig_t l0 = {
        .layer_type = layer_input,
        .input_channels = input_channels,
        .input_y = input_y,
        .input_x = input_x,
    };
    layerconfig_t l1 = {
        .layer_type = layer_convolution,
        .norm_type = norm_none,
        .convolution_filters = 4,
        .convolution_kernel_size = 3,
        .convolution_kernel_stride = 1,
        .convolution_kernel_padding = 1,
        .activation_function = activation_identity,
    };
    layerconfig_t l2 = {
        .layer_type = layer_split,
        .norm_type = norm_none,
        .split_filters = 2,
        .activation_function = activation_identity,
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
        .dense_output_size = 5,
        .activation_function = activation_identity,
    };
    layerconfig[0] = &l0;
    layerconfig[1] = &l4;

    neuralnet_t neuralnet = neuralnet_alloc(layers, layerconfig);

    const uint64_t samples = 1;
    tensor_t input = tensor_alloc(samples, input_channels, input_y, input_x);
    tensor_t output = tensor_alloc(samples, NEURALNET_OUTPUT(neuralnet).activation->buffer->z_size, NEURALNET_OUTPUT(neuralnet).activation->buffer->y_size,
                                   NEURALNET_OUTPUT(neuralnet).activation->buffer->x_size);
    tensor_random_unary(&output);
    tensor_cpu_realize(&output);

    tensor_random_unary(&input);
    neuralnet_random(&neuralnet);
    neuralnet_linearize(&neuralnet, 1e-2);
    LINEARIZED_PRINT_(neuralnet.forward);
    compile_linearized_to_cl("source.cl", neuralnet.forward);

    STOP_TIME();
    PRINT_TIME();
    return 0;
}
