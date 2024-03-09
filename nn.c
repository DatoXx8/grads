#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "linearize.h"
#include "tensor.h"
#include "nn.h"

activation_t activation_alloc(enum activation_e activation_type, uint64_t a, uint64_t z, uint64_t y, uint64_t x) {
    activation_t activation = {0};
    switch(activation_type) {
        case(activation_identity): {
            activation.type = activation_identity;
            break;
        }
        case(activation_relu): {
            activation.type = activation_relu;
            break;
        }
        case(activation_sigmoid): {
            activation.type = activation_sigmoid;
            break;
        }
        case(activation_tanh): {
            activation.type = activation_tanh;
            break;
        }
        case(activation_silu): {
            activation.type = activation_silu;
            activation.intermediary = calloc(1, sizeof(tensor_t));
            *activation.intermediary = tensor_alloc(a, z, y, x);
            break;
        }
        case(activation_gelu): {
            activation.type = activation_gelu;
            activation.intermediary = calloc(1, sizeof(tensor_t));
            *activation.intermediary = tensor_alloc(a, z, y, x);
            break;
        }
        case(activation_leaky): {
            activation.type = activation_leaky;
            activation.intermediary = calloc(1, sizeof(tensor_t));
            *activation.intermediary = tensor_alloc(a, z, y, x);
            break;
        }
    }
    return(activation);
}
void activation_free(activation_t *activation) {
    switch(activation->type) {
        case(activation_identity): {
            break;
        }
        case(activation_relu): {
            break;
        }
        case(activation_sigmoid): {
            break;
        }
        case(activation_tanh): {
            break;
        }
        case(activation_silu): {
            tensor_free(activation->intermediary);
            free(activation->intermediary);
            break;
        }
        case(activation_gelu): {
            tensor_free(activation->intermediary);
            free(activation->intermediary);
            break;
        }
        case(activation_leaky): {
            tensor_free(activation->intermediary);
            free(activation->intermediary);
            break;
        }
    }
}
/* TODO: Implement the other activation functions */
void activation_activate(tensor_t *tensor, activation_t *activation_type) {
    switch(activation_type->type) {
        case(activation_identity): {
            break;
        }
        case(activation_relu): {
            tensor_max_unary(tensor, 0);
            break;
        }
        case(activation_sigmoid): {
            tensor_negate_unary(tensor);
            tensor_exp_unary(tensor);
            tensor_add_unary(tensor, 1);
            tensor_reciprocal_unary(tensor);
            break;
        }
        case(activation_tanh): {
            tensor_tanh_unary(tensor);
            break;
        }
        case(activation_silu): {
            tensor_copy_binary(activation_type->intermediary, tensor);
            tensor_max_unary(tensor, 0);
            tensor_negate_unary(activation_type->intermediary);
            tensor_exp_unary(activation_type->intermediary);
            tensor_add_unary(activation_type->intermediary, 1);
            tensor_reciprocal_unary(activation_type->intermediary);
            tensor_multiply_binary(tensor, activation_type->intermediary);
            break;
        }
        case(activation_gelu): {
            break;
        }
        case(activation_leaky): {
            break;
        }
    }
}
/* TODO: Implement the other activation functions */
void activation_derivative(tensor_t *tensor, activation_t *activation_type) {
    switch(activation_type->type) {
        case(activation_identity): {
            break;
        }
        case(activation_relu): {
            break;
        }
        case(activation_sigmoid): {
            break;
        }
        case(activation_tanh): {
            break;
        }
        case(activation_silu): {
            break;
        }
        case(activation_gelu): {
            break;
        }
        case(activation_leaky): {
            break;
        }
    }
}

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
    uint64_t input_flattened = input->buffer->a_inherent * input->buffer->z_inherent * input->buffer->y_inherent * input->buffer->x_inherent;
    uint64_t input_a = input->buffer->a_inherent;
    uint64_t input_z = input->buffer->z_inherent;
    uint64_t input_y = input->buffer->y_inherent;
    uint64_t input_x = input->buffer->x_inherent;
    uint64_t output_a = output->buffer->a_inherent;
    uint64_t output_z = output->buffer->z_inherent;
    uint64_t output_y = output->buffer->y_inherent;
    uint64_t output_x = output->buffer->x_inherent;

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
    uint64_t input_a = input->buffer->a_inherent;
    uint64_t input_z = input->buffer->z_inherent;
    uint64_t input_y = input->buffer->y_inherent;
    uint64_t input_x = input->buffer->x_inherent;
    uint64_t output_a = output->buffer->a_inherent;
    uint64_t output_z = output->buffer->z_inherent;
    uint64_t output_y = output->buffer->y_inherent;
    uint64_t output_x = output->buffer->x_inherent;

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

/* NOTE: Kind of a misnomer as this doesn't allocate any dynamic memory, which is also why there is no reduce_free(). However I like the name continuity. */
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
void reduce_forward(tensor_t *input, reduce_t *reduce, tensor_t *output) {
    uint64_t input_a = input->buffer->a_inherent;
    uint64_t input_z = input->buffer->z_inherent;
    uint64_t input_y = input->buffer->y_inherent;
    uint64_t input_x = input->buffer->x_inherent;
    uint64_t output_a = output->buffer->a_inherent;
    uint64_t output_z = output->buffer->z_inherent;
    uint64_t output_y = output->buffer->y_inherent;
    uint64_t output_x = output->buffer->x_inherent;

    uint64_t output_y_i = 0;
    uint64_t output_x_i = 0;

    tensor_resize_move(input, 1, 1, reduce->kernel_size, reduce->kernel_size);
    tensor_offset_move(input, 0, 0, 0, 0);
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
    /* NOTE: Could probably remove this for optimal performance. Although it would be unpleasant for debugging. */
    tensor_resize_move(input, input_a, input_z, input_y, input_x);
    tensor_offset_move(input, 0, 0, 0, 0);
    tensor_resize_move(output, output_a, output_z, output_y, output_x);
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
    uint64_t input_z = input->buffer->z_inherent;
    uint64_t output_z = output->buffer->z_inherent;
    uint64_t output_y = output->buffer->y_inherent;
    uint64_t output_x = output->buffer->x_inherent;

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

    tensor_resize_move(input, 1, input_z, output_y, output_x);
    tensor_offset_move(input, 0, 0, 0, 0);
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

/* TODO: Implement residual connections. */
layer_t layer_alloc(layerconfig_t *layerconfig) {
    layer_t layer = {0};
    switch(layerconfig->layer_type) {
        case(layer_input): {
            layer.activation = calloc(1, sizeof(tensor_t));
            *layer.activation = tensor_alloc(1, layerconfig->input_channels, layerconfig->input_y, layerconfig->input_x);
            layer.activation_g = calloc(1, sizeof(tensor_t));
            *layer.activation_g = tensor_alloc(1, layerconfig->input_channels, layerconfig->input_y, layerconfig->input_x);
            layer.layer_type = layer_input;
            break;
        }
        case(layer_dense): {
            layer.activation = calloc(1, sizeof(tensor_t));
            *layer.activation = tensor_alloc(1, 1, 1, layerconfig->dense_output_size);
            layer.activation_g = calloc(1, sizeof(tensor_t));
            *layer.activation_g = tensor_alloc(1, 1, 1, layerconfig->dense_output_size);
            layer.activation_type = calloc(1, sizeof(activation_t));
            *layer.activation_type = activation_alloc(layerconfig->activation_type, 1, 1, 1, layerconfig->dense_input_channels * layerconfig->dense_input_y * layerconfig->dense_input_x);
            layer.layer_type = layer_dense;
            layer.dense = calloc(1, sizeof(dense_t));
            *layer.dense = dense_alloc(layerconfig->dense_input_channels * layerconfig->dense_input_y * layerconfig->dense_input_x, layerconfig->dense_output_size);
            break;
        }
        case(layer_convolution): {
            uint64_t new_size_y = CONVOLUTION_OUTPUT_SIZE(layerconfig->convolution_input_y, layerconfig->convolution_kernel_size, layerconfig->convolution_kernel_stride, layerconfig->convolution_kernel_padding);
            uint64_t new_size_x = CONVOLUTION_OUTPUT_SIZE(layerconfig->convolution_input_x, layerconfig->convolution_kernel_size, layerconfig->convolution_kernel_stride, layerconfig->convolution_kernel_padding);
            layer.activation = calloc(1, sizeof(tensor_t));
            *layer.activation = tensor_alloc(1, layerconfig->convolution_filters, new_size_y, new_size_x);
            layer.activation_g = calloc(1, sizeof(tensor_t));
            *layer.activation_g = tensor_alloc(1, layerconfig->convolution_filters, new_size_y, new_size_x);
            layer.activation_type = calloc(1, sizeof(activation_t));
            *layer.activation_type = activation_alloc(layerconfig->activation_type, 1, layerconfig->dense_input_channels, layerconfig->dense_input_y, layerconfig->dense_input_x);
            layer.layer_type = layer_convolution;
            layer.convolution = calloc(1, sizeof(convolution_t));
            *layer.convolution = convolution_alloc(layerconfig->convolution_input_channels, layerconfig->convolution_input_y, layerconfig->convolution_input_x, layerconfig->convolution_filters, layerconfig->convolution_kernel_size, layerconfig->convolution_kernel_stride, layerconfig->convolution_kernel_padding);
            break;
        }
        case(layer_reduce): {
            uint64_t new_size_y = REDUCE_OUTPUT_SIZE(layerconfig->reduce_input_y, layerconfig->reduce_kernel_size, layerconfig->reduce_kernel_stride);
            uint64_t new_size_x = REDUCE_OUTPUT_SIZE(layerconfig->reduce_input_x, layerconfig->reduce_kernel_size, layerconfig->reduce_kernel_stride);
            layer.activation = calloc(1, sizeof(tensor_t));
            *layer.activation = tensor_alloc(1, layerconfig->reduce_input_channels, new_size_y, new_size_x);
            layer.activation_g = calloc(1, sizeof(tensor_t));
            *layer.activation_g = tensor_alloc(1, layerconfig->reduce_input_channels, new_size_y, new_size_x);
            layer.activation_type = calloc(1, sizeof(activation_t));
            *layer.activation_type = activation_alloc(layerconfig->activation_type, 1, layerconfig->dense_input_channels, layerconfig->dense_input_y, layerconfig->dense_input_x);
            layer.layer_type = layer_reduce;
            layer.reduce = calloc(1, sizeof(reduce_t));
            *layer.reduce = reduce_alloc(layerconfig->reduce_type, layerconfig->reduce_input_channels, layerconfig->reduce_input_y, layerconfig->reduce_input_x, layerconfig->reduce_kernel_size, layerconfig->reduce_kernel_stride);
            break;
        }
        case(layer_split): {
            layer.activation = calloc(1, sizeof(tensor_t));
            *layer.activation = tensor_alloc(1, layerconfig->split_filters * layerconfig->split_input_channels, layerconfig->split_input_y, layerconfig->split_input_x);
            layer.activation_g = calloc(1, sizeof(tensor_t));
            *layer.activation_g = tensor_alloc(1, layerconfig->split_filters * layerconfig->split_input_channels, layerconfig->split_input_y, layerconfig->split_input_x);
            layer.activation_type = calloc(1, sizeof(activation_t));
            *layer.activation_type = activation_alloc(layerconfig->activation_type, 1, layerconfig->dense_input_channels, layerconfig->dense_input_y, layerconfig->dense_input_x);
            layer.layer_type = layer_split;
            layer.split = calloc(1, sizeof(split_t));
            *layer.split = split_alloc(layerconfig->split_filters, layerconfig->split_input_channels, layerconfig->split_input_y, layerconfig->split_input_x);
            break;
        }
    }
    return(layer);
}
void layer_free(layer_t *layer) {
    switch(layer->layer_type) {
        case(layer_input): {
            tensor_free(layer->activation);
            free(layer->activation);
            tensor_free(layer->activation_g);
            free(layer->activation_g);
            break;
        }
        case(layer_dense): {
            tensor_free(layer->activation);
            free(layer->activation);
            tensor_free(layer->activation_g);
            free(layer->activation_g);
            dense_free(layer->dense);
            free(layer->dense);
            break;
        }
        case(layer_convolution): {
            tensor_free(layer->activation);
            free(layer->activation);
            tensor_free(layer->activation_g);
            free(layer->activation_g);
            convolution_free(layer->convolution);
            free(layer->convolution);
            break;
        }
        case(layer_reduce): {
            tensor_free(layer->activation);
            free(layer->activation);
            tensor_free(layer->activation_g);
            free(layer->activation_g);
            // reduce_free(layer->reduce);
            free(layer->reduce);
            break;
        }
        case(layer_split): {
            tensor_free(layer->activation);
            free(layer->activation);
            tensor_free(layer->activation_g);
            free(layer->activation_g);
            split_free(layer->split);
            free(layer->split);
            break;
        }
    }
}

neuralnet_t neuralnet_alloc(uint64_t layers, layerconfig_t **layerconfig) {
    neuralnet_t neuralnet = {
        .layers = layers,
        .layer = calloc(layers, sizeof(layer_t)),
        .linearized_forward = calloc(1, sizeof(linearized_t)),
    };
    *neuralnet.linearized_forward = linearized_alloc();

    uint64_t previous_z;
    uint64_t previous_y;
    uint64_t previous_x;
    assert(layerconfig[0]->layer_type == layer_input); /* Beginning layer has to be an input layer. */
    neuralnet.layer[0] = layer_alloc(layerconfig[0]);
    for(uint64_t layer = 1; layer < layers; layer++) {
        previous_z = neuralnet.layer[layer - 1].activation->buffer->z_size;
        previous_y = neuralnet.layer[layer - 1].activation->buffer->y_size;
        previous_x = neuralnet.layer[layer - 1].activation->buffer->x_size;
        switch(layerconfig[layer]->layer_type) {
            case(layer_dense): {
                layerconfig[layer]->dense_input_channels = previous_z;
                layerconfig[layer]->dense_input_y = previous_y;
                layerconfig[layer]->dense_input_x = previous_x;
                neuralnet.layer[layer] = layer_alloc(layerconfig[layer]);
                break;
            }
            case(layer_convolution): {
                layerconfig[layer]->convolution_input_channels = previous_z;
                layerconfig[layer]->convolution_input_y = previous_y;
                layerconfig[layer]->convolution_input_x = previous_x;
                neuralnet.layer[layer] = layer_alloc(layerconfig[layer]);
                break;
            }
            case(layer_reduce): {
                layerconfig[layer]->reduce_input_channels = previous_z;
                layerconfig[layer]->reduce_input_y = previous_y;
                layerconfig[layer]->reduce_input_x = previous_x;
                neuralnet.layer[layer] = layer_alloc(layerconfig[layer]);
                break;
            }
            case(layer_split): {
                layerconfig[layer]->split_input_channels = previous_z;
                layerconfig[layer]->split_input_y = previous_y;
                layerconfig[layer]->split_input_x = previous_x;
                neuralnet.layer[layer] = layer_alloc(layerconfig[layer]);
                break;
            }
            case(layer_input): {
                fprintf(stderr, "ERROR: Tried to allocate input layer at a layer with index %lu\n", layer);
                exit(1);
            }
        }
    }

    return(neuralnet);
}
void neuralnet_free(neuralnet_t *neuralnet) {
    for(uint64_t i = 0; i < neuralnet->layers; i++) {
        layer_free(&neuralnet->layer[i]);
    }
    free(neuralnet->layer);
    linearized_free(neuralnet->linearized_forward);
    free(neuralnet->linearized_forward);
}
void neuralnet_random(neuralnet_t *neuralnet) {
    for(uint64_t layer = 1; layer < neuralnet->layers; layer++) {
        switch(neuralnet->layer[layer].layer_type) {
            case(layer_dense): {
                tensor_random_unary(neuralnet->layer[layer].dense->biases);
                tensor_random_unary(neuralnet->layer[layer].dense->weights);
                tensor_cpu_realize(neuralnet->layer[layer].dense->biases);
                tensor_cpu_realize(neuralnet->layer[layer].dense->weights);
                break;
            }
            case(layer_convolution): {
                tensor_random_unary(neuralnet->layer[layer].convolution->biases);
                tensor_random_unary(neuralnet->layer[layer].convolution->weights);
                tensor_cpu_realize(neuralnet->layer[layer].convolution->biases);
                tensor_cpu_realize(neuralnet->layer[layer].convolution->weights);
                break;
            }
            case(layer_reduce): {
                /* Nothing to initialize. */
                break;
            }
            case(layer_split): {
                tensor_random_unary(neuralnet->layer[layer].split->biases);
                tensor_random_unary(neuralnet->layer[layer].split->weights);
                tensor_cpu_realize(neuralnet->layer[layer].split->biases);
                tensor_cpu_realize(neuralnet->layer[layer].split->weights);
                break;
            }
            case(layer_input): {
                fprintf(stderr, "ERROR: Input layer at layer %lu. I don't even know how this can possibly happen.\n", layer);
                exit(1);
            }
        }
    }
}
/* NOTE: Used for linearizing all needed ops from the input to the output. Only needs to be called once per neuralnet. */
/* TODO: Once backpropagation is implemented also have this linearize that into lienarized_backward. */
void neuralnet_linearize(neuralnet_t *neuralnet) {
    for(uint64_t layer = 1; layer < neuralnet->layers; layer++) {
        switch(neuralnet->layer[layer].layer_type) {
            case(layer_dense): {
                dense_forward(neuralnet->layer[layer - 1].activation, neuralnet->layer[layer].dense, neuralnet->layer[layer].activation);
                activation_activate(neuralnet->layer[layer].activation, neuralnet->layer[layer].activation_type);
                break;
            }
            case(layer_convolution): {
                convolution_forward(neuralnet->layer[layer - 1].activation, neuralnet->layer[layer].convolution, neuralnet->layer[layer].activation);
                activation_activate(neuralnet->layer[layer].activation, neuralnet->layer[layer].activation_type);
                break;
            }
            case(layer_reduce): {
                reduce_forward(neuralnet->layer[layer - 1].activation, neuralnet->layer[layer].reduce, neuralnet->layer[layer].activation);
                // activation_activate(neuralnet->layer[layer].activation, neuralnet->layer[layer].activation_type);
                break;
            }
            case(layer_split): {
                split_forward(neuralnet->layer[layer - 1].activation, neuralnet->layer[layer].split, neuralnet->layer[layer].activation);
                activation_activate(neuralnet->layer[layer].activation, neuralnet->layer[layer].activation_type);
                break;
            }
            case(layer_input): {
                fprintf(stderr, "ERROR: Input layer at layer %lu. I don't even know how this can possibly happen.\n", layer);
                exit(1);
            }
        }
    }
    linearized_from_op(neuralnet->linearized_forward, NEURALNET_OUTPUT_(neuralnet)->op);
}
/* WARN: neuralnet_linearize `has` to be called `once` before this one. */
void neuralnet_forward(neuralnet_t *neuralnet, tensor_t *input) {
    tensor_copy_binary(NEURALNET_INPUT_(neuralnet), input);
    /* TODO: Think about how to remove this and have it be a part of the neuralnet linearization. */
    tensor_cpu_realize(NEURALNET_INPUT_(neuralnet));
}
void neuralnet_backward(neuralnet_t *neuralnet, tensor_t *training_input, tensor_t *training_output) {
}
void neuralnet_learn(neuralnet_t *neuralnet, double learning) {
    for(uint64_t layer = 1; layer < neuralnet->layers; layer++) {
        switch(neuralnet->layer[layer].layer_type) {
            case(layer_dense): {
                tensor_multiply_unary(neuralnet->layer[layer].dense->weights_g, learning);
                tensor_subtract_binary(neuralnet->layer[layer].dense->weights, neuralnet->layer[layer].dense->weights_g);
                tensor_multiply_unary(neuralnet->layer[layer].dense->biases_g, learning);
                tensor_subtract_binary(neuralnet->layer[layer].dense->biases, neuralnet->layer[layer].dense->biases_g);
                break;
            }
            case(layer_convolution): {
                tensor_multiply_unary(neuralnet->layer[layer].convolution->weights_g, learning);
                tensor_subtract_binary(neuralnet->layer[layer].convolution->weights, neuralnet->layer[layer].convolution->weights_g);
                tensor_multiply_unary(neuralnet->layer[layer].convolution->biases_g, learning);
                tensor_subtract_binary(neuralnet->layer[layer].convolution->biases, neuralnet->layer[layer].convolution->biases_g);
                break;
            }
            case(layer_reduce): {
                /* NOTE: This has no parameters so there is nothing to update. */
                break;
            }
            case(layer_split): {
                tensor_multiply_unary(neuralnet->layer[layer].split->weights_g, learning);
                tensor_subtract_binary(neuralnet->layer[layer].split->weights, neuralnet->layer[layer].split->weights_g);
                tensor_multiply_unary(neuralnet->layer[layer].split->biases_g, learning);
                tensor_subtract_binary(neuralnet->layer[layer].split->biases, neuralnet->layer[layer].split->biases_g);
                break;
            }
            case(layer_input): {
                fprintf(stderr, "ERROR: Input layer at layer %lu. I don't even know how this can possibly happen.\n", layer);
                exit(1);
            }
        }
    }
}
void neuralnet_print(neuralnet_t *neuralnet, int padding, int offset, const char *name) {
    if(strcmp(name, "") != 0) {
        printf("%*s%s\n", offset, "", name);
    } else {
        printf("%*sneuralnet\n", offset, "");
    }
    for(uint64_t layer = 0; layer < neuralnet->layers; layer++) {
        switch(neuralnet->layer[layer].layer_type) {
            case(layer_dense): {
                printf("%*slayer[%lu] dense\n", offset + padding, "", layer);
                dense_print(neuralnet->layer[layer].dense, padding, offset + 2 * padding, "");
                break;
            }
            case(layer_convolution): {
                printf("%*slayer[%lu] convolution\n", offset + padding, "", layer);
                convolution_print(neuralnet->layer[layer].convolution, padding, offset + 2 * padding, "");
                break;
            }
            case(layer_reduce): {
                printf("%*slayer[%lu] reduce\n", offset + padding, "", layer);
                reduce_print(neuralnet->layer[layer].reduce, padding, offset + 2 * padding, "");
                break;
            }
            case(layer_split): {
                printf("%*slayer[%lu] split\n", offset + padding, "", layer);
                split_print(neuralnet->layer[layer].split, padding, offset + 2 * padding, "");
                break;
            }
            case(layer_input): {
                printf("%*slayer[%lu] input\n", offset + padding, "", layer);
                printf("%*sinput\n", offset + 2 * padding, "");
                tensor_print(neuralnet->layer[layer].activation, padding, offset + 3 * padding, "activation");
                tensor_print(neuralnet->layer[layer].activation, padding, offset + 3 * padding, "activation_g");
            }
        }
    }
}
void neuralnet_print_shape(neuralnet_t *neuralnet, int padding, int offset, const char *name) {
    if(strcmp(name, "") != 0) {
        printf("%*s%s shape\n", offset, "", name);
    } else {
        printf("%*sneuralnet shape\n", offset, "");
    }
    for(uint64_t layer = 0; layer < neuralnet->layers; layer++) {
        switch(neuralnet->layer[layer].layer_type) {
            case(layer_dense): {
                printf("%*slayer[%lu] dense\n", offset + padding, "", layer);
                dense_print_shape(neuralnet->layer[layer].dense, padding, offset + 2 * padding, "");
                printf("%*s{%lu, %lu, %lu, %lu} %lu\n", offset + 2 * padding, "", neuralnet->layer[layer].activation->buffer->a_size, neuralnet->layer[layer].activation->buffer->z_size, neuralnet->layer[layer].activation->buffer->y_size, neuralnet->layer[layer].activation->buffer->x_size, neuralnet->layer[layer].activation->buffer->offset);
                break;
            }
            case(layer_convolution): {
                printf("%*slayer[%lu] convolution\n", offset + padding, "", layer);
                convolution_print_shape(neuralnet->layer[layer].convolution, padding, offset + 2 * padding, "");
                printf("%*s{%lu, %lu, %lu, %lu} %lu\n", offset + 2 * padding, "", neuralnet->layer[layer].activation->buffer->a_size, neuralnet->layer[layer].activation->buffer->z_size, neuralnet->layer[layer].activation->buffer->y_size, neuralnet->layer[layer].activation->buffer->x_size, neuralnet->layer[layer].activation->buffer->offset);
                break;
            }
            case(layer_reduce): {
                printf("%*slayer[%lu] reduce\n", offset + padding, "", layer);
                // reduce_print_shape(neuralnet->layer[layer].reduce, padding, offset + 2 * padding, "");
                printf("%*s{%lu, %lu, %lu, %lu} %lu\n", offset + 2 * padding, "", neuralnet->layer[layer].activation->buffer->a_size, neuralnet->layer[layer].activation->buffer->z_size, neuralnet->layer[layer].activation->buffer->y_size, neuralnet->layer[layer].activation->buffer->x_size, neuralnet->layer[layer].activation->buffer->offset);
                break;
            }
            case(layer_split): {
                printf("%*slayer[%lu] split\n", offset + padding, "", layer);
                split_print_shape(neuralnet->layer[layer].split, padding, offset + 2 * padding, "");
                printf("%*s{%lu, %lu, %lu, %lu} %lu\n", offset + 2 * padding, "", neuralnet->layer[layer].activation->buffer->a_size, neuralnet->layer[layer].activation->buffer->z_size, neuralnet->layer[layer].activation->buffer->y_size, neuralnet->layer[layer].activation->buffer->x_size, neuralnet->layer[layer].activation->buffer->offset);
                break;
            }
            case(layer_input): {
                printf("%*slayer[%lu] input\n", offset + padding, "", layer);
                printf("%*s{%lu, %lu, %lu, %lu} %lu\n", offset + 2 * padding, "", neuralnet->layer[layer].activation->buffer->a_size, neuralnet->layer[layer].activation->buffer->z_size, neuralnet->layer[layer].activation->buffer->y_size, neuralnet->layer[layer].activation->buffer->x_size, neuralnet->layer[layer].activation->buffer->offset);
            }
        }
    }
}
// for(uint64_t layer = 0; layer < neuralnet->layers; layer++) {
//     switch(neuralnet->layer[layer].layer_type) {
//         case(layer_dense): {
//             printf("%*slayer[%lu] dense\n", offset + padding, "", layer);
//             dense_print_shape(neuralnet->layer[layer].dense, padding, offset + 2 * padding, "");
//             break;
//         }
//         case(layer_convolution): {
//             printf("%*slayer[%lu] convolution\n", offset + padding, "", layer);
//             convolution_print_shape(neuralnet->layer[layer].convolution, padding, offset + 2 * padding, "");
//             break;
//         }
//         case(layer_reduce): {
//             printf("%*slayer[%lu] reduce\n", offset + padding, "", layer);
//             // reduce_print_shape(neuralnet->layer[layer].reduce, padding, offset + 2 * padding, "");
//             break;
//         }
//         case(layer_split): {
//             printf("%*slayer[%lu] split\n", offset + padding, "", layer);
//             split_print_shape(neuralnet->layer[layer].split, padding, offset + 2 * padding, "");
//             break;
//         }
//         case(layer_input): {
//             printf("%*slayer[%lu] input\n", offset + padding, "", layer);
//             printf("%*sinput shape\n", offset + 2 * padding, "");
//             printf("%*s{%lu, %lu, %lu, %lu} %lu\n", offset + 3 * padding, "", neuralnet->layer[layer].activation->buffer->a_size, neuralnet->layer[layer].activation->buffer->z_size, neuralnet->layer[layer].activation->buffer->y_size, neuralnet->layer[layer].activation->buffer->x_size, neuralnet->layer[layer].activation->buffer->offset);
//         }
//     }
// }
