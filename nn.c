#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"
#include "nn.h"

dense_t dense_alloc(uint64_t input_size, uint64_t output_size) {
    dense_t dense = {
        .biases = calloc(1, sizeof(tensor_t)),
        .biases_g = calloc(1, sizeof(tensor_t)),
        .weights = calloc(1, sizeof(tensor_t)),
        .weights_g = calloc(1, sizeof(tensor_t)),

        .input_size = input_size,
        .input_multiply_temp = calloc(1, sizeof(tensor_t)),

        .output_size = output_size,
        .output_multiply_temp = calloc(1, sizeof(tensor_t)),
    };
    *dense.biases = tensor_alloc(1, 1, 1, output_size);
    *dense.biases_g = tensor_alloc(1, 1, 1, output_size);
    *dense.weights = tensor_alloc(1, 1, input_size, output_size);
    *dense.weights_g = tensor_alloc(1, 1, input_size, output_size);
    *dense.input_multiply_temp = tensor_alloc(1, 1, input_size, 1);
    *dense.output_multiply_temp = tensor_alloc(1, 1, 1, output_size);

    return(dense);
}
void dense_free(dense_t *dense) {
    tensor_free(dense->biases);
    tensor_free(dense->biases_g);
    tensor_free(dense->weights);
    tensor_free(dense->weights_g);
    tensor_free(dense->input_multiply_temp);
    tensor_free(dense->output_multiply_temp);
    free(dense->biases);
    free(dense->biases_g);
    free(dense->weights);
    free(dense->weights_g);
    free(dense->input_multiply_temp);
    free(dense->output_multiply_temp);
}
/* NOTE: Automagically "flattens" the input tensor to shape `{1, 1, 1, a * z * y * x}`. */
void dense_forward(tensor_t *input, dense_t *dense, tensor_t *output) {
    uint64_t input_flattened = input->buffer->a_size * input->buffer->z_size * input->buffer->y_size * input->buffer->x_size;
    uint64_t input_a = input->buffer->a_size;
    uint64_t input_z = input->buffer->z_size;
    uint64_t input_y = input->buffer->y_size;
    uint64_t input_x = input->buffer->x_size;
    uint64_t output_a = output->buffer->a_size;
    uint64_t output_z = output->buffer->z_size;
    uint64_t output_y = output->buffer->y_size;
    uint64_t output_x = output->buffer->x_size;

    tensor_reshape_move(input, 1, 1, input_flattened, 1);
    tensor_resize_move(dense->weights, 1, 1, dense->input_size, 1);
    tensor_reshape_move(output, 1, 1, 1, 1);

    for(uint64_t i = 0; i < dense->output_size; i++) {
        tensor_offset_move(dense->weights, 0, 0, 0, i);
        tensor_offset_move(output, 0, 0, 0, i);
        tensor_copy_binary(dense->input_multiply_temp, dense->weights);
        tensor_multiply_binary(dense->input_multiply_temp, input);
        tensor_sum_reduce(output, dense->input_multiply_temp);
    }

    tensor_reshape_move(input, input_a, input_z, input_y, input_x);
    tensor_offset_move(input, 0, 0, 0, 0);
    tensor_reshape_move(output, output_a, output_z, output_y, output_x);
    tensor_offset_move(output, 0, 0, 0, 0);
    tensor_reshape_move(dense->weights, 1, 1, dense->input_size, dense->output_size);
    tensor_offset_move(dense->weights, 0, 0, 0, 0);

    tensor_add_binary(output, dense->biases);
}
void dense_backward(tensor_t *output, dense_t *dense, tensor_t *input) {
}
void dense_print(dense_t *dense, int padding, int offset, const char *name) {
    if(strcmp(name, "") != 0) {
        printf("%*s%s dense\n", offset, "", name);
    } else {
        printf("%*sdense\n", offset, "");
    }
    tensor_print(dense->biases, padding, offset + padding, "biases");
    tensor_print(dense->biases_g, padding, offset + padding, "biases_g");
    tensor_print(dense->weights, padding, offset + padding, "weights");
    tensor_print(dense->weights_g, padding, offset + padding, "weights_g");
}
void dense_print_shape(dense_t *dense, int padding, int offset, const char *name) {
    if(strcmp(name, "") != 0) {
        printf("%*s%s dense shape\n", offset, "", name);
    } else {
        printf("%*sdense shape\n", offset, "");
    }
    printf("%*sBiases  {%lu, %lu, %lu, %lu}\n", offset + padding, "", dense->biases->buffer->a_size, dense->biases->buffer->z_size, dense->biases->buffer->y_size, dense->biases->buffer->x_size);
    printf("%*sWeights {%lu, %lu, %lu, %lu}\n", offset + padding, "", dense->weights->buffer->a_size, dense->weights->buffer->z_size, dense->weights->buffer->y_size, dense->weights->buffer->x_size);
}

convolution_t convolution_alloc(uint64_t input_channels, uint64_t input_y, uint64_t input_x, uint64_t filters, uint64_t kernel_size, uint64_t kernel_stride, uint64_t kernel_padding) {
    convolution_t convolution = {
        .input_channels = input_channels,
        .input_y = input_y,
        .input_x = input_x,
        .filters = filters,
        .kernel_size = kernel_size,
        .kernel_stride = kernel_stride,
        .kernel_padding = kernel_padding,

        .biases = calloc(1, sizeof(tensor_t)),
        .biases_g = calloc(1, sizeof(tensor_t)),
        .weights = calloc(1, sizeof(tensor_t)),
        .weights_g = calloc(1, sizeof(tensor_t)),

        .padded_input = calloc(1, sizeof(tensor_t)),
        .kernel_temp = calloc(1, sizeof(tensor_t)),
    };

    *convolution.biases = tensor_alloc(filters, 1, 1, 1);
    *convolution.biases_g = tensor_alloc(filters, 1, 1, 1);
    *convolution.weights = tensor_alloc(filters, input_channels, kernel_size, kernel_size);
    *convolution.weights_g = tensor_alloc(filters, input_channels, kernel_size, kernel_size);
    *convolution.padded_input = tensor_alloc(1, input_channels, input_y + 2 * kernel_padding, input_x + 2 * kernel_padding);
    *convolution.kernel_temp = tensor_alloc(1, input_channels, kernel_size, kernel_size);

    return(convolution);
}
void convolution_free(convolution_t *convolution) {
    tensor_free(convolution->biases);
    tensor_free(convolution->biases_g);
    tensor_free(convolution->weights);
    tensor_free(convolution->weights_g);
    tensor_free(convolution->padded_input);
    tensor_free(convolution->kernel_temp);
    free(convolution->biases);
    free(convolution->biases_g);
    free(convolution->weights);
    free(convolution->weights_g);
    free(convolution->padded_input);
    free(convolution->kernel_temp);
}
void convolution_forward(tensor_t *input, convolution_t *convolution, tensor_t *output) {
    uint64_t input_a = input->buffer->a_size;
    uint64_t input_z = input->buffer->z_size;
    uint64_t input_y = input->buffer->y_size;
    uint64_t input_x = input->buffer->x_size;
    uint64_t output_a = output->buffer->a_size;
    uint64_t output_z = output->buffer->z_size;
    uint64_t output_y = output->buffer->y_size;
    uint64_t output_x = output->buffer->x_size;

    uint64_t input_x_max = input_x + 2 * convolution->kernel_padding - 1;
    uint64_t input_y_max = input_y + 2 * convolution->kernel_padding - 1;

    uint64_t output_y_i;
    uint64_t output_x_i;
    tensor_resize_move(convolution->biases, 1, 1, 1, 1);
    tensor_resize_move(convolution->weights, 1, input_z, convolution->kernel_size, convolution->kernel_size);
    tensor_resize_move(output, 1, 1, 1, 1);
    tensor_resize_move(convolution->padded_input, input_a, input_z, input_y, input_x);
    tensor_offset_move(convolution->padded_input, 0, 0, convolution->kernel_padding, convolution->kernel_padding);
    tensor_copy_binary(convolution->padded_input, input);
    tensor_resize_move(convolution->padded_input, 1, input_z, convolution->kernel_size, convolution->kernel_size);

    for(uint64_t filter = 0; filter < convolution->filters; filter++) {
        tensor_offset_move(convolution->biases, filter, 0, 0, 0);
        tensor_offset_move(convolution->weights, filter, 0, 0, 0);
        output_y_i = 0;
        for(uint64_t input_y_i = 0; input_y_i < input_y_max; input_y_i += convolution->kernel_stride) {
            output_x_i = 0;
            for(uint64_t input_x_i = 0; input_x_i < input_x_max; input_x_i += convolution->kernel_stride) {
                tensor_offset_move(output, 0, filter, output_y_i, output_x_i);
                tensor_offset_move(convolution->padded_input, 0, 0, input_y_i, input_x_i);
                tensor_copy_binary(convolution->kernel_temp, convolution->padded_input);
                tensor_multiply_binary(convolution->kernel_temp, convolution->weights);
                tensor_sum_reduce(output, convolution->kernel_temp);
                tensor_add_binary(output, convolution->biases);
                output_x_i++;
            }
            output_y_i++;
        }
    }
    tensor_resize_move(convolution->biases, 1, convolution->filters, 1, 1);
    tensor_offset_move(convolution->biases, 0, 0, 0, 0);
    tensor_resize_move(convolution->weights, convolution->input_channels, convolution->filters, convolution->kernel_size, convolution->kernel_size);
    tensor_offset_move(convolution->weights, 0, 0, 0, 0);
    tensor_resize_move(output, output_a, output_z, output_y, output_x);
    tensor_offset_move(output, 0, 0, 0, 0);
    tensor_resize_move(convolution->padded_input, input_a, input_z, input_y + 2 * convolution->kernel_padding, input_x + 2 * convolution->kernel_padding); /* TODO: Remove this for optimal performance. */
    tensor_offset_move(convolution->padded_input, 0, 0, 0, 0); /* TODO: Remove this for optimal performance. */
}
void convolution_backward(tensor_t *output, convolution_t *convolution, tensor_t *input) {
}
void convolution_print(convolution_t *convolution, int padding, int offset, const char *name) {
    if(strcmp(name, "") != 0) {
        printf("%*s%s convolution\n", offset, "", name);
    } else {
        printf("%*sconvolution\n", offset, "");
    }
    tensor_print(convolution->biases, padding, offset + padding, "biases");
    tensor_print(convolution->biases_g, padding, offset + padding, "biases_g");
    tensor_print(convolution->weights, padding, offset + padding, "weights");
    tensor_print(convolution->weights_g, padding, offset + padding, "weights_g");
}
void convolution_print_shape(convolution_t *convolution, int padding, int offset, const char *name) {
    if(strcmp(name, "") != 0) {
        printf("%*s%s convolution shape\n", offset, "", name);
    } else {
        printf("%*sconvolution shape\n", offset, "");
    }
    printf("%*sBiases  {%lu, %lu, %lu, %lu}\n", offset + padding, "", convolution->biases->buffer->a_size, convolution->biases->buffer->z_size, convolution->biases->buffer->y_size, convolution->biases->buffer->x_size);
    printf("%*sWeights {%lu, %lu, %lu, %lu}\n", offset + padding, "", convolution->weights->buffer->a_size, convolution->weights->buffer->z_size, convolution->weights->buffer->y_size, convolution->weights->buffer->x_size);
}

reduce_t reduce_alloc(enum layer_reduce_e type, uint64_t input_channels, uint64_t input_y, uint64_t input_x, uint64_t kernel_size, uint64_t kernel_stride) {
    reduce_t reduce = {
        .type = type,
        .input_channels = input_channels,
        .input_y = input_y,
        .input_x = input_x,
        .kernel_size = kernel_size,
        .kernel_stride = kernel_stride,
    };

    return(reduce);
}
// void reduce_free(reduce_t *reduce) {
// }
void reduce_forward(tensor_t *input, reduce_t *reduce, tensor_t *output) {
    uint64_t input_z = input->buffer->z_size;
    uint64_t input_y = input->buffer->y_size;
    uint64_t input_x = input->buffer->x_size;
    uint64_t output_z = output->buffer->z_size;
    uint64_t output_y = output->buffer->y_size;
    uint64_t output_x = output->buffer->x_size;

    uint64_t output_y_i = 0;
    uint64_t output_x_i = 0;

    tensor_resize_move(input, 1, 1, reduce->kernel_size, reduce->kernel_size);
    tensor_resize_move(output, 1, 1, 1, 1);
    /* PERF: Switch statement is on the outside cuz it only needs to be done once then. */
    switch(reduce->type) {
        case(layer_reduce_max): {
            for(uint64_t channel = 0; channel < reduce->input_channels; channel++) {
                output_y_i = 0;
                for(uint64_t y = 0; y < reduce->input_y - reduce->kernel_size + 1; y += reduce->kernel_stride) {
                    output_x_i = 0;
                    for(uint64_t x = 0; x < reduce->input_x - reduce->kernel_size + 1; x += reduce->kernel_stride) {
                        tensor_offset_move(input, 0, channel, y, x);
                        tensor_offset_move(output, 0, channel, output_y_i, output_x_i);
                        tensor_max_reduce(output, input);
                        output_x_i++;
                    }
                    output_y_i++;
                }
            }
            break;
        }
        case(layer_reduce_min): {
            for(uint64_t channel = 0; channel < reduce->input_channels; channel++) {
                output_y_i = 0;
                for(uint64_t y = 0; y < reduce->input_y; y += reduce->kernel_stride) {
                    output_x_i = 0;
                    for(uint64_t x = 0; x < reduce->input_x; x += reduce->kernel_stride) {
                        tensor_offset_move(input, 0, channel, y, x);
                        tensor_offset_move(output, 0, channel, output_y_i, output_x_i);
                        tensor_min_reduce(output, input);
                        output_x_i++;
                    }
                    output_y_i++;
                }
            }
            break;
        }
        case(layer_reduce_avg): {
            for(uint64_t channel = 0; channel < reduce->input_channels; channel++) {
                output_y_i = 0;
                for(uint64_t y = 0; y < reduce->input_y; y += reduce->kernel_stride) {
                    output_x_i = 0;
                    for(uint64_t x = 0; x < reduce->input_x; x += reduce->kernel_stride) {
                        tensor_offset_move(input, 0, channel, y, x);
                        tensor_offset_move(output, 0, channel, output_y_i, output_x_i);
                        tensor_avg_reduce(output, input);
                        output_x_i++;
                    }
                    output_y_i++;
                }
            }
            break;
        }
    }
    /* NOTE: Could probably remove this for optimal performance. */
    tensor_resize_move(input, 1, input_z, input_y, input_x);
    tensor_offset_move(input, 0, 0, 0, 0);
    tensor_resize_move(output, 1, output_z, output_y, output_x);
    tensor_offset_move(output, 0, 0, 0, 0);
}
void reduce_backward(tensor_t *output, reduce_t *reduce, tensor_t *input) {
}
void reduce_print(reduce_t *reduce, int padding, int offset, const char *name) {
    if(strcmp(name, "") != 0) {
        printf("%*s%s convolution\n", offset, "", name);
    } else {
        printf("%*sconvolution\n", offset, "");
    }
    printf("%*sSize %lu, Stride %lu, Channels %lu, Input y %lu, Input x %lu\n", offset + padding, "", reduce->kernel_size, reduce->kernel_stride, reduce->input_channels, reduce->input_y, reduce->input_x);
}

split_t split_alloc(uint64_t filters, uint64_t input_channels, uint64_t input_y, uint64_t input_x) {
    split_t split = {
        .filters = filters,
        .input_channels = input_channels,
        .input_y = input_y,
        .input_x = input_x,

        .biases = calloc(1, sizeof(tensor_t)),
        .biases_g = calloc(1, sizeof(tensor_t)),
        .weights = calloc(1, sizeof(tensor_t)),
        .weights_g = calloc(1, sizeof(tensor_t)),
    };

    // *split.biases = tensor_alloc(filters, 1, 1, 1);
    // *split.biases_g = tensor_alloc(filters, 1, 1, 1);
    // *split.weights = tensor_alloc(filters, 1, input_y, input_x);
    // *split.weights_g = tensor_alloc(filters, 1, input_y, input_x);
    *split.biases = tensor_alloc(filters, input_channels, input_y, input_x);
    *split.biases_g = tensor_alloc(filters, input_channels, input_y, input_x);
    *split.weights = tensor_alloc(filters, input_channels, input_y, input_x);
    *split.weights_g = tensor_alloc(filters, input_channels, input_y, input_x);

    return(split);
}
void split_free(split_t *split) {
    tensor_free(split->biases);
    tensor_free(split->biases_g);
    tensor_free(split->weights);
    tensor_free(split->weights_g);
    free(split->biases);
    free(split->biases_g);
    free(split->weights);
    free(split->weights_g);
}
void split_forward(tensor_t *input, split_t *split, tensor_t *output) {
    uint64_t input_z = input->buffer->z_size;
    uint64_t output_z = output->buffer->z_size;
    uint64_t output_y = output->buffer->y_size;
    uint64_t output_x = output->buffer->x_size;

    tensor_resize_move(output, 1, input_z, output_y, output_x);
    tensor_resize_move(split->weights, 1, input_z, output_y, output_x);
    tensor_resize_move(split->biases, 1, input_z, output_y, output_x);
    for(uint64_t filter = 0; filter < split->filters; filter++) {
        tensor_offset_move(output, 0, filter * input_z, 0, 0);
        tensor_offset_move(split->weights, filter, 0, 0, 0);
        tensor_offset_move(split->biases, filter, 0, 0, 0);
        tensor_copy_binary(output, input);
        tensor_multiply_binary(output, split->weights);
        tensor_add_binary(output, split->biases);
    }

    tensor_resize_move(output, 1, output_z, output_y, output_x);
    tensor_offset_move(output, 0, 0, 0, 0);
    tensor_resize_move(split->weights, split->filters, input_z, output_y, output_x);
    tensor_offset_move(split->weights, 0, 0, 0, 0);
    tensor_resize_move(split->biases, split->filters, input_z, output_y, output_x);
    tensor_offset_move(split->biases, 0, 0, 0, 0);
}
/* NOTE: Names are flipped here due to it being *backward* propagation. */
void split_backward(tensor_t *output, split_t *split, tensor_t *input) {
}
void split_print(split_t *split, int padding, int offset, const char *name) {
    if(strcmp(name, "") != 0) {
        printf("%*s%s split\n", offset, "", name);
    } else {
        printf("%*ssplit\n", offset, "");
    }
    tensor_print(split->biases, padding, offset + padding, "biases");
    tensor_print(split->biases_g, padding, offset + padding, "biases_g");
    tensor_print(split->weights, padding, offset + padding, "weights");
    tensor_print(split->weights_g, padding, offset + padding, "weights_g");
}
void split_print_shape(split_t *split, int padding, int offset, const char *name) {
    if(strcmp(name, "") != 0) {
        printf("%*s%s split shape\n", offset, "", name);
    } else {
        printf("%*ssplit shape\n", offset, "");
    }
    printf("%*sBiases  {%lu, %lu, %lu, %lu}\n", offset + padding, "", split->biases->buffer->a_size, split->biases->buffer->z_size, split->biases->buffer->y_size, split->biases->buffer->x_size);
    printf("%*sWeights {%lu, %lu, %lu, %lu}\n", offset + padding, "", split->weights->buffer->a_size, split->weights->buffer->z_size, split->weights->buffer->y_size, split->weights->buffer->x_size);
}
