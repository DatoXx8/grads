#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include "nn.h"
#include "tensor.h"
// #include "linearize.h"
#include "utils.h"

int main(void) {
    const uint64_t rng = time(NULL);
    // printf("RNG Seed %lu\n", rng);
    // srand(rng);
    INIT_TIMER;

    START_TIME;

    const uint64_t input_size = 3;
    const uint64_t input_channels = 2;
    const uint64_t filters = 1;
    const uint64_t kernel_size = 2;
    const uint64_t kernel_stride = 1;
    const uint64_t kernel_padding = 1;
    const uint64_t output_size = REDUCE_OUTPUT_SIZE(input_size, kernel_size, kernel_stride);
    tensor_t in = tensor_alloc(1, input_channels, input_size, input_size);
    tensor_t out = tensor_alloc(1, input_channels, output_size, output_size);

    reduce_t reduce = reduce_alloc(layer_reduce_min, input_channels, input_size, input_size, kernel_size, kernel_stride);

    tensor_random_unary(&in);

    reduce_forward(&in, &reduce, &out);

    tensor_cpu_realize(&out);
    tensor_cpu_realize(&in);
    TENSOR_PRINT(in);
    TENSOR_PRINT(out);

    STOP_TIME;
    PRINT_TIME;
    return(0);
}
