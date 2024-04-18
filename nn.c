#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linearize.h"
#include "nn.h"
#include "tensor.h"
#include "utils.h"

static activation_t activation_alloc_(enum activation_e activation_type, uint64_t a, uint64_t z, uint64_t y, uint64_t x) {
    activation_t activation = {0};
    switch(activation_type) {
        case activation_identity: {
            activation.type = activation_identity;
            break;
        }
        case activation_relu: {
            activation.type = activation_relu;
            break;
        }
        case activation_sigmoid: {
            activation.type = activation_sigmoid;
            break;
        }
        case activation_tanh: {
            activation.type = activation_tanh;
            break;
        }
        case activation_silu: {
            activation.type = activation_silu;
            activation.intermediary = calloc(1, sizeof(tensor_t));
            assert(activation.intermediary);
            *activation.intermediary = tensor_alloc(a, z, y, x);
            break;
        }
        case activation_gelu: {
            activation.type = activation_gelu;
            activation.intermediary = calloc(1, sizeof(tensor_t));
            assert(activation.intermediary);
            *activation.intermediary = tensor_alloc(a, z, y, x);
            break;
        }
        case activation_leaky: {
            activation.type = activation_leaky;
            activation.intermediary = calloc(1, sizeof(tensor_t));
            assert(activation.intermediary);
            *activation.intermediary = tensor_alloc(a, z, y, x);
            break;
        }
    }
    return activation;
}
static void activation_free_(activation_t *activation) {
    switch(activation->type) {
        case activation_identity: {
            break;
        }
        case activation_relu: {
            break;
        }
        case activation_sigmoid: {
            break;
        }
        case activation_tanh: {
            break;
        }
        case activation_silu: {
            tensor_free(activation->intermediary);
            free(activation->intermediary);
            break;
        }
        case activation_gelu: {
            tensor_free(activation->intermediary);
            free(activation->intermediary);
            break;
        }
        case activation_leaky: {
            tensor_free(activation->intermediary);
            free(activation->intermediary);
            break;
        }
    }
}
const double leaky_factor_ = 0.1;
/* TODO: Implement the other activation functions */
/* TODO: This should also calculate the derivatives if a new flag `forward_only` is not set for the neuralnet. */
static void activation_activate_(tensor_t *tensor, activation_t *activation_type) {
    switch(activation_type->type) {
        case activation_identity: {
            break;
        }
        case activation_relu: {
            tensor_max_unary(tensor, 0);
            break;
        }
        case activation_sigmoid: {
            tensor_negate_unary(tensor);
            tensor_exp_unary(tensor);
            tensor_add_unary(tensor, 1);
            tensor_reciprocal_unary(tensor);
            break;
        }
        case activation_tanh: {
            tensor_tanh_unary(tensor);
            break;
        }
        case activation_silu: {
            tensor_copy_binary(activation_type->intermediary, tensor);
            tensor_max_unary(tensor, 0);
            tensor_negate_unary(activation_type->intermediary);
            tensor_exp_unary(activation_type->intermediary);
            tensor_add_unary(activation_type->intermediary, 1);
            tensor_reciprocal_unary(activation_type->intermediary);
            tensor_multiply_binary(tensor, activation_type->intermediary);
            break;
        }
        case activation_gelu: {
            /* NOTE: This is an approximation and can be found here: https://paperswithcode.com/method/gelu */
            /* TODO: Write a brute forcer that minimizes the error by varying the constant. */
            tensor_copy_binary(activation_type->intermediary, tensor);
            tensor_max_unary(tensor, 0);
            tensor_multiply_unary(activation_type->intermediary, 1.703);
            tensor_negate_unary(activation_type->intermediary);
            tensor_exp_unary(activation_type->intermediary);
            tensor_add_unary(activation_type->intermediary, 1);
            tensor_reciprocal_unary(activation_type->intermediary);
            tensor_multiply_binary(tensor, activation_type->intermediary);
            break;
        }
        case activation_leaky: {
            tensor_copy_binary(activation_type->intermediary, tensor);
            tensor_multiply_unary(activation_type->intermediary, leaky_factor_);
            tensor_max_binary(tensor, activation_type->intermediary);
            break;
        }
    }
}
static norm_t norm_alloc_(enum norm_e type, tensor_t *tensor) {
    norm_t norm = {
        .type = type,
    };
    switch(norm.type) {
        case norm_none: {
            break;
        }
        case norm_batch: {
            norm.batch_variance = calloc(1, sizeof(tensor_t));
            norm.batch_expected = calloc(1, sizeof(tensor_t));
            assert(norm.batch_expected);
            assert(norm.batch_variance);
            *norm.batch_expected = tensor_alloc(tensor->buffer->a_inherent, tensor->buffer->z_inherent, tensor->buffer->y_inherent, tensor->buffer->x_inherent);
            *norm.batch_variance = tensor_alloc(tensor->buffer->a_inherent, tensor->buffer->z_inherent, tensor->buffer->y_inherent, tensor->buffer->x_inherent);
            assert(0);
            break;
        }
        case norm_layer: {
            norm.layer_expected = calloc(1, sizeof(tensor_t));
            norm.layer_variance = calloc(1, sizeof(tensor_t));
            norm.layer_intermediary = calloc(1, sizeof(tensor_t));
            assert(norm.layer_expected);
            assert(norm.layer_variance);
            assert(norm.layer_intermediary);
            *norm.layer_expected = tensor_alloc(1, 1, 1, 1);
            *norm.layer_variance = tensor_alloc(1, 1, 1, 1);
            *norm.layer_intermediary =
                tensor_alloc(tensor->buffer->a_inherent, tensor->buffer->z_inherent, tensor->buffer->y_inherent, tensor->buffer->x_inherent);
            break;
        }
        case norm_simple: {
            norm.simple_max = calloc(1, sizeof(tensor_t));
            norm.simple_intermediary = calloc(1, sizeof(tensor_t));
            assert(norm.simple_max);
            assert(norm.simple_intermediary);
            *norm.simple_max = tensor_alloc(1, 1, 1, 1);
            *norm.simple_intermediary =
                tensor_alloc(tensor->buffer->a_inherent, tensor->buffer->z_inherent, tensor->buffer->y_inherent, tensor->buffer->x_inherent);
            break;
        }
    }
    return norm;
}
static void norm_free_(norm_t *norm) {
    switch(norm->type) {
        case norm_none: {
            break;
        }
        case norm_batch: {
            tensor_free(norm->batch_expected);
            tensor_free(norm->batch_variance);
            free(norm->batch_expected);
            free(norm->batch_variance);
            break;
        }
        case norm_layer: {
            tensor_free(norm->layer_expected);
            tensor_free(norm->layer_variance);
            tensor_free(norm->layer_intermediary);
            free(norm->layer_expected);
            free(norm->layer_variance);
            free(norm->layer_intermediary);
            break;
        }
        case norm_simple: {
            tensor_free(norm->simple_intermediary);
            tensor_free(norm->simple_max);
            free(norm->simple_intermediary);
            free(norm->simple_max);
            break;
        }
    }
}
const double epsilon = 1e-6;
static void norm_calculate_layer_(norm_t *norm, tensor_t *tensor) {
    tensor_avg_reduce(norm->layer_expected, tensor);
    tensor_copy_binary(norm->layer_intermediary, tensor);
    /* NOTE: The reason this is commented out is quite ugly. Basically when realizing the tensor the expected value already gets subtracted before in the
     * calculation. I understand that isn't nice, but otherwise I would have to make a copy here and that would be worse. */
    // tensor_subtract_like_binary(norm->layer_intermediary, norm->layer_expected);
    tensor_square_unary(norm->layer_intermediary);
    tensor_avg_reduce(norm->layer_variance, norm->layer_intermediary);
    /* NOTE: Added to avoid dividing by 0 when normalizing the layer. */
    tensor_add_unary(norm->layer_variance, epsilon);
    tensor_sqrt_unary(norm->layer_variance);
}
/* This ones tricky. Even the function signature isn't obvious. */
static void norm_calculate_batch_(void) {}
static void norm_apply_(norm_t *norm, tensor_t *tensor) {
    switch(norm->type) {
        case norm_none: {
            break;
        }
        case norm_batch: {
            norm_calculate_batch_();
            assert(0);
            break;
        }
        case norm_layer: {
            norm_calculate_layer_(norm, tensor);
            tensor_subtract_like_binary(tensor, norm->layer_expected);
            tensor_divide_like_binary(tensor, norm->layer_variance);
            break;
        }
        case norm_simple: {
            tensor_copy_binary(norm->simple_intermediary, tensor);
            tensor_absolute_unary(norm->simple_intermediary);
            tensor_max_reduce(norm->simple_max, norm->simple_intermediary);
            tensor_divide_like_binary(tensor, norm->simple_max);
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

        .input_size_ = input_size,
        .output_size = output_size,

        .input_multiply_temp_ = calloc(1, sizeof(tensor_t)),
        .output_multiply_temp_ = calloc(1, sizeof(tensor_t)),
        .full_temp_ = calloc(1, sizeof(tensor_t)),
    };
    assert(dense.biases);
    assert(dense.biases_g);
    assert(dense.weights);
    assert(dense.weights_g);
    assert(dense.input_multiply_temp_);
    assert(dense.output_multiply_temp_);
    assert(dense.full_temp_);

    *dense.biases = tensor_alloc(1, 1, 1, output_size);
    *dense.biases_g = tensor_alloc(1, 1, 1, output_size);
    *dense.weights = tensor_alloc(1, 1, input_size, output_size);
    *dense.weights_g = tensor_alloc(1, 1, input_size, output_size);
    *dense.input_multiply_temp_ = tensor_alloc(1, 1, input_size, 1);
    *dense.output_multiply_temp_ = tensor_alloc(1, 1, 1, output_size);
    *dense.full_temp_ = tensor_alloc(1, 1, input_size, output_size);

    return dense;
}
void dense_free(dense_t *dense) {
    tensor_free(dense->biases);
    tensor_free(dense->biases_g);
    tensor_free(dense->weights);
    tensor_free(dense->weights_g);
    tensor_free(dense->input_multiply_temp_);
    tensor_free(dense->output_multiply_temp_);
    tensor_free(dense->full_temp_);
    free(dense->biases);
    free(dense->biases_g);
    free(dense->weights);
    free(dense->weights_g);
    free(dense->input_multiply_temp_);
    free(dense->output_multiply_temp_);
    free(dense->full_temp_);
}
/* NOTE: Automagically "flattens" the input tensor to shape `{1, 1, a * z * y * x, 1}`. */
void dense_forward(tensor_t *input, dense_t *dense, tensor_t *output) {
    uint64_t input_z = input->buffer->z_inherent;
    uint64_t input_y = input->buffer->y_inherent;
    uint64_t input_x = input->buffer->x_inherent;
    uint64_t output_z = output->buffer->z_inherent;
    uint64_t output_y = output->buffer->y_inherent;
    uint64_t output_x = output->buffer->x_inherent;

    tensor_reshape_move(input, 1, 1, dense->input_size_, 1);
    tensor_resize_move(dense->weights, 1, 1, dense->input_size_, 1);
    tensor_resize_move(output, 1, 1, 1, 1);

    for(uint64_t i = 0; i < dense->output_size; i++) {
        tensor_offset_move(dense->weights, 0, 0, 0, i);
        tensor_offset_move(output, 0, 0, 0, i);
        tensor_copy_binary(dense->input_multiply_temp_, dense->weights);
        tensor_multiply_binary(dense->input_multiply_temp_, input);
        tensor_sum_reduce(output, dense->input_multiply_temp_);
    }

    tensor_reshape_move(input, 1, input_z, input_y, input_x);
    tensor_offset_move(input, 0, 0, 0, 0);
    tensor_resize_move(output, 1, output_z, output_y, output_x);
    tensor_offset_move(output, 0, 0, 0, 0);
    tensor_resize_move(dense->weights, 1, 1, dense->input_size_, dense->output_size);
    tensor_offset_move(dense->weights, 0, 0, 0, 0);

    tensor_add_binary(output, dense->biases);
}
void dense_backward(tensor_t *input, tensor_t *input_gradient, dense_t *dense, tensor_t *output_gradient) {
    uint64_t input_z = input->buffer->z_inherent;
    uint64_t input_y = input->buffer->y_inherent;
    uint64_t input_x = input->buffer->x_inherent;
    /* Biases */
    tensor_add_binary(dense->biases_g, output_gradient);
    /* Weights */
    tensor_resize_move(dense->full_temp_, 1, 1, 1, dense->output_size);
    for(uint64_t i = 0; i < dense->input_size_; i++) {
        tensor_offset_move(dense->full_temp_, 0, 0, i, 0);
        tensor_copy_binary(dense->full_temp_, output_gradient);
    }
    tensor_resize_move(dense->full_temp_, 1, 1, dense->input_size_, 1);
    tensor_reshape_move(input, 1, 1, dense->input_size_, 1);
    for(uint64_t i = 0; i < dense->output_size; i++) {
        tensor_offset_move(dense->full_temp_, 0, 0, 0, i);
        tensor_multiply_binary(dense->full_temp_, input);
    }
    tensor_resize_move(dense->full_temp_, 1, 1, dense->input_size_, dense->output_size);
    tensor_offset_move(dense->full_temp_, 0, 0, 0, 0);
    tensor_reshape_move(input, 1, input_z, input_y, input_x);
    tensor_add_binary(dense->weights_g, dense->full_temp_);
    /* Previous activation grad */
    tensor_reshape_move(input_gradient, 1, 1, dense->input_size_, 1);
    tensor_resize_move(input_gradient, 1, 1, 1, 1);
    tensor_resize_move(dense->weights, 1, 1, 1, dense->output_size);
    for(uint64_t i = 0; i < dense->input_size_; i++) {
        tensor_offset_move(input_gradient, 0, 0, i, 0);
        tensor_offset_move(dense->weights, 0, 0, i, 0);
        tensor_copy_binary(dense->output_multiply_temp_, dense->weights);
        tensor_multiply_binary(dense->output_multiply_temp_, output_gradient);
        /* NOTE: Sum is technically the right one I think, but avg provides better scaling. */
        tensor_sum_reduce(input_gradient, dense->output_multiply_temp_);
        // tensor_avg_reduce(input_gradient, dense->output_multiply_temp_);
    }
    tensor_reshape_move(input_gradient, 1, input_z, input_y, input_x);
    tensor_offset_move(input_gradient, 0, 0, 0, 0);
    tensor_resize_move(dense->weights, 1, 1, dense->input_size_, dense->output_size);
    tensor_offset_move(dense->weights, 0, 0, 0, 0);
}
void dense_print(dense_t *dense, int padding, int offset, const char *name) {
    if(strncmp(name, "", 1)) {
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
    if(strncmp(name, "", 1)) {
        printf("%*s%s dense shape\n", offset, "", name);
    } else {
        printf("%*sdense shape\n", offset, "");
    }
    printf("%*sBiases  {%lu, %lu, %lu, %lu}\n", offset + padding, "", dense->biases->buffer->a_size, dense->biases->buffer->z_size,
           dense->biases->buffer->y_size, dense->biases->buffer->x_size);
    printf("%*sWeights {%lu, %lu, %lu, %lu}\n", offset + padding, "", dense->weights->buffer->a_size, dense->weights->buffer->z_size,
           dense->weights->buffer->y_size, dense->weights->buffer->x_size);
}

convolution_t convolution_alloc(uint64_t input_channels, uint64_t input_y, uint64_t input_x, uint64_t filters, uint64_t kernel_size, uint64_t kernel_stride,
                                uint64_t kernel_padding) {
    convolution_t convolution = {
        .input_channels_ = input_channels,
        .input_y_ = input_y,
        .input_x_ = input_x,
        .filters = filters,
        .kernel_size = kernel_size,
        .kernel_stride = kernel_stride,
        .kernel_padding = kernel_padding,

        .biases = calloc(1, sizeof(tensor_t)),
        .biases_g = calloc(1, sizeof(tensor_t)),
        .weights = calloc(1, sizeof(tensor_t)),
        .weights_g = calloc(1, sizeof(tensor_t)),

        .padded_input_ = calloc(1, sizeof(tensor_t)),
        .padded_grad_ = calloc(1, sizeof(tensor_t)),
        .kernel_temp_ = calloc(1, sizeof(tensor_t)),
        .single_temp_ = calloc(1, sizeof(tensor_t)),
    };
    assert(convolution.biases);
    assert(convolution.biases_g);
    assert(convolution.weights);
    assert(convolution.weights_g);
    assert(convolution.padded_input_);
    assert(convolution.padded_grad_);
    assert(convolution.kernel_temp_);
    assert(convolution.single_temp_);

    *convolution.biases = tensor_alloc(filters, 1, 1, 1);
    *convolution.biases_g = tensor_alloc(filters, 1, 1, 1);
    *convolution.weights = tensor_alloc(filters, input_channels, kernel_size, kernel_size);
    *convolution.weights_g = tensor_alloc(filters, input_channels, kernel_size, kernel_size);
    *convolution.padded_input_ = tensor_alloc(1, input_channels, input_y + 2 * kernel_padding, input_x + 2 * kernel_padding);
    *convolution.padded_grad_ = tensor_alloc(1, input_channels, input_y + 2 * kernel_padding, input_x + 2 * kernel_padding);
    *convolution.kernel_temp_ = tensor_alloc(1, input_channels, kernel_size, kernel_size);
    *convolution.single_temp_ = tensor_alloc(1, 1, 1, 1);

    return convolution;
}
void convolution_free(convolution_t *convolution) {
    tensor_free(convolution->biases);
    tensor_free(convolution->biases_g);
    tensor_free(convolution->weights);
    tensor_free(convolution->weights_g);
    tensor_free(convolution->padded_input_);
    tensor_free(convolution->padded_grad_);
    tensor_free(convolution->kernel_temp_);
    tensor_free(convolution->single_temp_);
    free(convolution->biases);
    free(convolution->biases_g);
    free(convolution->weights);
    free(convolution->weights_g);
    free(convolution->padded_input_);
    free(convolution->padded_grad_);
    free(convolution->kernel_temp_);
    free(convolution->single_temp_);
}
void convolution_forward(tensor_t *input, convolution_t *convolution, tensor_t *output) {
    uint64_t input_z = input->buffer->z_inherent;
    uint64_t input_y = input->buffer->y_inherent;
    uint64_t input_x = input->buffer->x_inherent;
    uint64_t output_z = output->buffer->z_inherent;
    uint64_t output_y = output->buffer->y_inherent;
    uint64_t output_x = output->buffer->x_inherent;

    uint64_t input_x_max = input_x + convolution->kernel_padding - 1;
    uint64_t input_y_max = input_y + convolution->kernel_padding - 1;

    uint64_t output_y_i;
    uint64_t output_x_i;
    tensor_offset_move(input, 0, 0, 0, 0);
    tensor_resize_move(convolution->biases, 1, 1, 1, 1);
    tensor_resize_move(convolution->weights, 1, input_z, convolution->kernel_size, convolution->kernel_size);
    tensor_resize_move(output, 1, 1, 1, 1);
    tensor_resize_move(convolution->padded_input_, 1, input_z, input_y, input_x);
    tensor_offset_move(convolution->padded_input_, 0, 0, convolution->kernel_padding, convolution->kernel_padding);
    tensor_copy_binary(convolution->padded_input_, input);
    tensor_resize_move(convolution->padded_input_, 1, input_z, convolution->kernel_size, convolution->kernel_size);

    for(uint64_t filter = 0; filter < convolution->filters; filter++) {
        tensor_offset_move(convolution->biases, filter, 0, 0, 0);
        tensor_offset_move(convolution->weights, filter, 0, 0, 0);
        output_y_i = 0;
        for(uint64_t input_y_i = 0; input_y_i < input_y_max; input_y_i += convolution->kernel_stride) {
            output_x_i = 0;
            for(uint64_t input_x_i = 0; input_x_i < input_x_max; input_x_i += convolution->kernel_stride) {
                tensor_offset_move(output, 0, filter, output_y_i, output_x_i);
                tensor_offset_move(convolution->padded_input_, 0, 0, input_y_i, input_x_i);
                tensor_copy_binary(convolution->kernel_temp_, convolution->padded_input_);
                tensor_multiply_binary(convolution->kernel_temp_, convolution->weights);
                tensor_sum_reduce(output, convolution->kernel_temp_);
                tensor_add_binary(output, convolution->biases);
                output_x_i++;
            }
            output_y_i++;
        }
    }
    tensor_resize_move(convolution->biases, convolution->filters, 1, 1, 1);
    tensor_offset_move(convolution->biases, 0, 0, 0, 0);
    tensor_resize_move(convolution->weights, convolution->filters, convolution->input_channels_, convolution->kernel_size, convolution->kernel_size);
    tensor_offset_move(convolution->weights, 0, 0, 0, 0);
    tensor_resize_move(output, 1, output_z, output_y, output_x);
    tensor_offset_move(output, 0, 0, 0, 0);
    tensor_resize_move(convolution->padded_input_, 1, input_z, input_y + 2 * convolution->kernel_padding, input_x + 2 * convolution->kernel_padding);
    tensor_offset_move(convolution->padded_input_, 0, 0, 0, 0);
}
void convolution_backward(tensor_t *input, tensor_t *input_gradient, convolution_t *convolution, tensor_t *output, tensor_t *output_gradient) {
    uint64_t input_z = input->buffer->z_inherent;
    uint64_t input_y = input->buffer->y_inherent;
    uint64_t input_x = input->buffer->x_inherent;
    uint64_t output_z = output->buffer->z_inherent;
    uint64_t output_y = output->buffer->y_inherent;
    uint64_t output_x = output->buffer->x_inherent;
    /* Biases */
    tensor_resize_move(convolution->biases_g, 1, 1, 1, 1);
    tensor_resize_move(output_gradient, 1, 1, output_y, output_x);
    for(uint64_t i = 0; i < convolution->filters; i++) {
        tensor_offset_move(convolution->biases_g, i, 0, 0, 0);
        tensor_offset_move(output_gradient, 0, i, 0, 0);
        tensor_avg_reduce(convolution->single_temp_, output_gradient);
        tensor_add_binary(convolution->biases_g, convolution->single_temp_);
    }
    tensor_resize_move(convolution->biases_g, convolution->filters, 1, 1, 1);
    tensor_offset_move(convolution->biases_g, 0, 0, 0, 0);
    tensor_resize_move(output_gradient, 1, output_z, output_y, output_x);
    tensor_offset_move(output_gradient, 0, 0, 0, 0);
    /* Weights */
    uint64_t input_y_i;
    uint64_t input_x_i;
    tensor_resize_move(output_gradient, 1, 1, 1, 1);
    tensor_offset_move(output_gradient, 0, 0, 0, 0);
    tensor_resize_move(convolution->weights_g, 1, input_z, convolution->kernel_size, convolution->kernel_size);
    tensor_offset_move(convolution->weights_g, 0, 0, 0, 0);
    tensor_resize_move(convolution->padded_input_, 1, input_z, convolution->kernel_size, convolution->kernel_size);
    tensor_offset_move(convolution->padded_input_, 0, 0, 0, 0);
    for(uint64_t filter = 0; filter < convolution->filters; filter++) {
        tensor_offset_move(convolution->weights_g, filter, 0, 0, 0);
        input_y_i = 0;
        for(uint64_t output_y_i = 0; output_y_i < output_y; output_y_i++) {
            input_x_i = 0;
            for(uint64_t output_x_i = 0; output_x_i < output_x; output_x_i++) {
                tensor_offset_move(output_gradient, 0, filter, output_y_i, output_x_i);
                tensor_offset_move(convolution->padded_input_, 0, 0, input_y_i, input_x_i);
                tensor_copy_binary(convolution->kernel_temp_, convolution->padded_input_);
                tensor_multiply_like_binary(convolution->kernel_temp_, output_gradient);
                tensor_add_binary(convolution->weights_g, convolution->kernel_temp_);
                input_x_i += convolution->kernel_padding;
            }
            input_y_i += convolution->kernel_padding;
        }
    }
    tensor_resize_move(output_gradient, 1, output_z, output_y, output_x);
    tensor_offset_move(output_gradient, 0, 0, 0, 0);
    tensor_resize_move(convolution->padded_input_, 1, input_z, input_y, input_x);
    tensor_offset_move(convolution->padded_input_, 0, 0, 0, 0);
    tensor_resize_move(convolution->weights_g, convolution->filters, input_z, convolution->kernel_size, convolution->kernel_size);
    tensor_offset_move(convolution->weights_g, 0, 0, 0, 0);
    /* Previous activation grad */
    tensor_resize_move(output_gradient, 1, 1, 1, 1);
    tensor_offset_move(output_gradient, 0, 0, 0, 0);
    tensor_resize_move(convolution->weights, 1, input_z, convolution->kernel_size, convolution->kernel_size);
    tensor_offset_move(convolution->weights, 0, 0, 0, 0);
    tensor_resize_move(convolution->padded_grad_, 1, input_z, convolution->kernel_size, convolution->kernel_size);
    tensor_offset_move(convolution->padded_grad_, 0, 0, 0, 0);
    for(uint64_t filter = 0; filter < convolution->filters; filter++) {
        tensor_offset_move(convolution->weights, filter, 0, 0, 0);
        input_y_i = 0;
        for(uint64_t output_y_i = 0; output_y_i < output_y; output_y_i++) {
            input_x_i = 0;
            for(uint64_t output_x_i = 0; output_x_i < output_x; output_x_i++) {
                tensor_offset_move(output_gradient, 0, filter, output_y_i, output_x_i);
                tensor_offset_move(convolution->padded_grad_, 0, 0, output_y_i, output_x_i);
                tensor_copy_binary(convolution->kernel_temp_, convolution->weights);
                tensor_multiply_like_binary(convolution->kernel_temp_, output_gradient);
                tensor_add_binary(convolution->padded_grad_, convolution->kernel_temp_);
                input_x_i += convolution->kernel_padding;
            }
            input_y_i += convolution->kernel_padding;
        }
    }
    tensor_resize_move(output_gradient, 1, output_z, output_y, output_x);
    tensor_offset_move(output_gradient, 0, 0, 0, 0);
    tensor_resize_move(convolution->weights, convolution->filters, input_z, convolution->kernel_size, convolution->kernel_size);
    tensor_offset_move(convolution->weights, 0, 0, 0, 0);
    tensor_resize_move(convolution->padded_grad_, 1, input_z, input_y, input_x);
    tensor_offset_move(convolution->padded_grad_, 0, 0, convolution->kernel_padding, convolution->kernel_padding);
    tensor_copy_binary(input_gradient, convolution->padded_grad_);

    tensor_resize_move(convolution->padded_grad_, 1, input_z, input_y + 2 * convolution->kernel_padding, input_x + 2 * convolution->kernel_padding);
    tensor_offset_move(convolution->padded_grad_, 0, 0, 0, 0);
}
void convolution_print(convolution_t *convolution, int padding, int offset, const char *name) {
    if(strncmp(name, "", 1)) {
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
    if(strncmp(name, "", 1)) {
        printf("%*s%s convolution shape\n", offset, "", name);
    } else {
        printf("%*sconvolution shape\n", offset, "");
    }
    printf("%*sBiases  {%lu, %lu, %lu, %lu}\n", offset + padding, "", convolution->biases->buffer->a_size, convolution->biases->buffer->z_size,
           convolution->biases->buffer->y_size, convolution->biases->buffer->x_size);
    printf("%*sWeights {%lu, %lu, %lu, %lu}\n", offset + padding, "", convolution->weights->buffer->a_size, convolution->weights->buffer->z_size,
           convolution->weights->buffer->y_size, convolution->weights->buffer->x_size);
}

/* NOTE: Kind of a misnomer as this doesn't allocate any dynamic memory, which is also why there is no reduce_free(). I like the name continuity tho. */
reduce_t reduce_alloc(enum layer_reduce_e type, uint64_t input_channels, uint64_t input_y, uint64_t input_x, uint64_t kernel_size, uint64_t kernel_stride) {
    reduce_t reduce = {
        .type = type,
        .input_channels_ = input_channels,
        .input_y_ = input_y,
        .input_x_ = input_x,
        .kernel_size = kernel_size,
        .kernel_stride = kernel_stride,
    };

    return reduce;
}
void reduce_forward(tensor_t *input, reduce_t *reduce, tensor_t *output) {
    uint64_t input_z = input->buffer->z_inherent;
    uint64_t input_y = input->buffer->y_inherent;
    uint64_t input_x = input->buffer->x_inherent;
    uint64_t output_z = output->buffer->z_inherent;
    uint64_t output_y = output->buffer->y_inherent;
    uint64_t output_x = output->buffer->x_inherent;

    uint64_t output_y_i = 0;
    uint64_t output_x_i = 0;

    tensor_reshape_move(input, 1, input_z, input_y, input_x);
    tensor_resize_move(input, 1, 1, reduce->kernel_size, reduce->kernel_size);
    tensor_resize_move(output, 1, 1, 1, 1);
    /* PERF: Switch statement is on the outside cuz it only needs to be done once then. */
    switch(reduce->type) {
        case layer_reduce_max: {
            for(uint64_t channel = 0; channel < reduce->input_channels_; channel++) {
                output_y_i = 0;
                for(uint64_t y = 0; y < reduce->input_y_ - reduce->kernel_size + 1; y += reduce->kernel_stride) {
                    output_x_i = 0;
                    for(uint64_t x = 0; x < reduce->input_x_ - reduce->kernel_size + 1; x += reduce->kernel_stride) {
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
        case layer_reduce_min: {
            for(uint64_t channel = 0; channel < reduce->input_channels_; channel++) {
                output_y_i = 0;
                for(uint64_t y = 0; y < reduce->input_y_ - reduce->kernel_size + 1; y += reduce->kernel_stride) {
                    output_x_i = 0;
                    for(uint64_t x = 0; x < reduce->input_x_ - reduce->kernel_size + 1; x += reduce->kernel_stride) {
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
        case layer_reduce_avg: {
            for(uint64_t channel = 0; channel < reduce->input_channels_; channel++) {
                output_y_i = 0;
                for(uint64_t y = 0; y < reduce->input_y_ - reduce->kernel_size + 1; y += reduce->kernel_stride) {
                    output_x_i = 0;
                    for(uint64_t x = 0; x < reduce->input_x_ - reduce->kernel_size + 1; x += reduce->kernel_stride) {
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
    tensor_resize_move(input, 1, input_z, input_y, input_x);
    tensor_offset_move(input, 0, 0, 0, 0);
    tensor_resize_move(output, 1, output_z, output_y, output_x);
    tensor_offset_move(output, 0, 0, 0, 0);
}
/* TODO: Replace this "solution", that is just flat out wrong (it approximates it somewhat), with the real solution. */
void reduce_backward(tensor_t *input_gradient, reduce_t *reduce, tensor_t *output_gradient) {
    uint64_t input_z = input_gradient->buffer->z_inherent;
    uint64_t input_y = input_gradient->buffer->y_inherent;
    uint64_t input_x = input_gradient->buffer->x_inherent;
    uint64_t output_z = output_gradient->buffer->z_inherent;
    uint64_t output_y = output_gradient->buffer->y_inherent;
    uint64_t output_x = output_gradient->buffer->x_inherent;
    uint64_t input_y_i;
    uint64_t input_x_i;
    tensor_resize_move(input_gradient, 1, 1, reduce->kernel_size, reduce->kernel_size);
    tensor_resize_move(output_gradient, 1, 1, 1, 1);
    switch(reduce->type) {
        case layer_reduce_max: {
            for(uint64_t channel = 0; channel < input_z; channel++) {
                input_y_i = 0;
                for(uint64_t output_y_i = 0; output_y_i < output_y; output_y_i++) {
                    input_x_i = 0;
                    for(uint64_t output_x_i = 0; output_x_i < output_x; output_x_i++) {
                        tensor_offset_move(input_gradient, 0, channel, input_y_i, input_x_i);
                        tensor_offset_move(output_gradient, 0, channel, output_y_i, output_x_i);
                        tensor_add_like_binary(input_gradient, output_gradient);
                        input_x_i += reduce->kernel_stride;
                    }
                    input_y_i += reduce->kernel_stride;
                }
            }
            break;
        }
        case layer_reduce_min: {
            for(uint64_t channel = 0; channel < input_z; channel++) {
                input_y_i = 0;
                for(uint64_t output_y_i = 0; output_y_i < output_y; output_y_i++) {
                    input_x_i = 0;
                    for(uint64_t output_x_i = 0; output_x_i < output_x; output_x_i++) {
                        tensor_offset_move(input_gradient, 0, channel, input_y_i, input_x_i);
                        tensor_offset_move(output_gradient, 0, channel, output_y_i, output_x_i);
                        tensor_add_like_binary(input_gradient, output_gradient);
                        input_x_i += reduce->kernel_stride;
                    }
                    input_y_i += reduce->kernel_stride;
                }
            }
            break;
        }
        case layer_reduce_avg: {
            for(uint64_t channel = 0; channel < input_z; channel++) {
                input_y_i = 0;
                for(uint64_t output_y_i = 0; output_y_i < output_y; output_y_i++) {
                    input_x_i = 0;
                    for(uint64_t output_x_i = 0; output_x_i < output_x; output_x_i++) {
                        tensor_offset_move(input_gradient, 0, channel, input_y_i, input_x_i);
                        tensor_offset_move(output_gradient, 0, channel, output_y_i, output_x_i);
                        tensor_add_like_binary(input_gradient, output_gradient);
                        input_x_i += reduce->kernel_stride;
                    }
                    input_y_i += reduce->kernel_stride;
                }
            }
            break;
        }
    }
    tensor_resize_move(input_gradient, 1, input_z, input_y, input_x);
    tensor_offset_move(input_gradient, 0, 0, 0, 0);
    tensor_resize_move(output_gradient, 1, output_z, output_y, output_x);
    tensor_offset_move(output_gradient, 0, 0, 0, 0);
}
void reduce_print(reduce_t *reduce, int padding, int offset, const char *name) {
    if(strncmp(name, "", 1)) {
        printf("%*s%s convolution\n", offset, "", name);
    } else {
        printf("%*sconvolution\n", offset, "");
    }
    printf("%*ssize %lu, stride %lu, channels %lu, input y %lu, input x %lu\n", offset + padding, "", reduce->kernel_size, reduce->kernel_stride,
           reduce->input_channels_, reduce->input_y_, reduce->input_x_);
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

        .input_temp_ = calloc(1, sizeof(tensor_t)),
    };
    assert(split.biases);
    assert(split.weights);
    assert(split.biases_g);
    assert(split.weights_g);
    assert(split.input_temp_);

    // *split.biases = tensor_alloc(filters, 1, 1, 1);
    // *split.biases_g = tensor_alloc(filters, 1, 1, 1);
    // *split.weights = tensor_alloc(filters, 1, input_y, input_x);
    // *split.weights_g = tensor_alloc(filters, 1, input_y, input_x);
    *split.biases = tensor_alloc(filters, input_channels, input_y, input_x);
    *split.biases_g = tensor_alloc(filters, input_channels, input_y, input_x);
    *split.weights = tensor_alloc(filters, input_channels, input_y, input_x);
    *split.weights_g = tensor_alloc(filters, input_channels, input_y, input_x);
    *split.input_temp_ = tensor_alloc(1, input_channels, input_y, input_x);

    return split;
}
void split_free(split_t *split) {
    tensor_free(split->biases);
    tensor_free(split->biases_g);
    tensor_free(split->weights);
    tensor_free(split->weights_g);
    tensor_free(split->input_temp_);
    free(split->biases);
    free(split->biases_g);
    free(split->weights);
    free(split->weights_g);
    free(split->input_temp_);
}
void split_forward(tensor_t *input, split_t *split, tensor_t *output) {
    uint64_t input_z = input->buffer->z_inherent;
    uint64_t input_y = input->buffer->y_inherent;
    uint64_t input_x = input->buffer->x_inherent;
    uint64_t output_z = output->buffer->z_inherent;
    uint64_t output_y = output->buffer->y_inherent;
    uint64_t output_x = output->buffer->x_inherent;

    tensor_offset_move(input, 0, 0, 0, 0);
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

    tensor_resize_move(input, 1, input_z, input_y, input_x);
    tensor_offset_move(input, 0, 0, 0, 0);
    tensor_resize_move(output, 1, output_z, output_y, output_x);
    tensor_offset_move(output, 0, 0, 0, 0);
    tensor_resize_move(split->weights, split->filters, input_z, output_y, output_x);
    tensor_offset_move(split->weights, 0, 0, 0, 0);
    tensor_resize_move(split->biases, split->filters, input_z, output_y, output_x);
    tensor_offset_move(split->biases, 0, 0, 0, 0);
}
void split_backward(tensor_t *input, tensor_t *input_gradient, split_t *split, tensor_t *output, tensor_t *output_gradient) {
    uint64_t input_z = input->buffer->z_inherent;
    uint64_t input_y = input->buffer->y_inherent;
    uint64_t input_x = input->buffer->x_inherent;
    uint64_t output_z = output->buffer->z_inherent;
    uint64_t output_y = output->buffer->y_inherent;
    uint64_t output_x = output->buffer->x_inherent;
    /* Biases */
    tensor_reshape_move(split->biases_g, 1, output_z, output_y, output_x);
    tensor_add_binary(split->biases_g, output_gradient);
    tensor_reshape_move(split->biases_g, split->filters, input_z, output_y, output_x);
    /* Weights */
    tensor_resize_move(split->weights_g, 1, input_z, output_y, output_x);
    tensor_resize_move(output_gradient, 1, input_z, output_y, output_x);
    for(uint64_t i = 0; i < split->filters; i++) {
        tensor_offset_move(split->weights_g, i, 0, 0, 0);
        tensor_offset_move(output_gradient, 0, i * input_z, 0, 0);
        tensor_copy_binary(split->input_temp_, output_gradient);
        tensor_multiply_binary(split->input_temp_, input);
        tensor_add_binary(split->weights_g, split->input_temp_);
    }
    tensor_resize_move(split->weights_g, split->filters, input_z, output_y, output_x);
    tensor_offset_move(split->weights_g, 0, 0, 0, 0);
    tensor_resize_move(output_gradient, 1, output_z, output_y, output_x);
    tensor_offset_move(output_gradient, 0, 0, 0, 0);
    /* Previous activation grad */
    tensor_resize_move(output_gradient, 1, input_z, input_y, input_x);
    tensor_resize_move(split->weights, 1, input_z, input_y, input_x);
    for(uint64_t i = 0; i < split->filters; i++) {
        tensor_offset_move(split->weights, i, 0, 0, 0);
        tensor_offset_move(output_gradient, 0, i * input_z, 0, 0);
        tensor_copy_binary(split->input_temp_, output_gradient);
        tensor_multiply_binary(split->input_temp_, split->weights);
        tensor_add_binary(input_gradient, split->input_temp_);
    }
    tensor_resize_move(split->weights, split->filters, input_z, input_y, input_x);
    tensor_offset_move(split->weights, 0, 0, 0, 0);
    tensor_resize_move(output_gradient, 1, output_z, output_y, output_x);
    tensor_offset_move(output_gradient, 0, 0, 0, 0);
}
void split_print(split_t *split, int padding, int offset, const char *name) {
    if(strncmp(name, "", 1)) {
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
    if(strncmp(name, "", 1)) {
        printf("%*s%s split shape\n", offset, "", name);
    } else {
        printf("%*ssplit shape\n", offset, "");
    }
    printf("%*sBiases  {%lu, %lu, %lu, %lu}\n", offset + padding, "", split->biases->buffer->a_size, split->biases->buffer->z_size,
           split->biases->buffer->y_size, split->biases->buffer->x_size);
    printf("%*sWeights {%lu, %lu, %lu, %lu}\n", offset + padding, "", split->weights->buffer->a_size, split->weights->buffer->z_size,
           split->weights->buffer->y_size, split->weights->buffer->x_size);
}

/* TODO: Implement residual connections. */
layer_t layer_alloc(layerconfig_t *layerconfig) {
    layer_t layer = {0};
    switch(layerconfig->layer_type) {
        case layer_input: {
            layer.activation = calloc(1, sizeof(tensor_t));
            *layer.activation = tensor_alloc(1, layerconfig->input_channels, layerconfig->input_y, layerconfig->input_x);
            layer.activation_g = calloc(1, sizeof(tensor_t));
            *layer.activation_g = tensor_alloc(1, layerconfig->input_channels, layerconfig->input_y, layerconfig->input_x);
            layer.layer_type = layer_input;
            assert(layer.activation);
            assert(layer.activation_g);
            break;
        }
        case layer_dense: {
            layer.activation = calloc(1, sizeof(tensor_t));
            *layer.activation = tensor_alloc(1, 1, 1, layerconfig->dense_output_size);
            layer.activation_g = calloc(1, sizeof(tensor_t));
            *layer.activation_g = tensor_alloc(1, 1, 1, layerconfig->dense_output_size);
            layer.activation_function = calloc(1, sizeof(activation_t));
            *layer.activation_function = activation_alloc_(layerconfig->activation_function, 1, 1, 1,
                                                           layerconfig->dense_input_channels_ * layerconfig->dense_input_y_ * layerconfig->dense_input_x_);
            layer.norm = calloc(1, sizeof(norm_t));
            *layer.norm = norm_alloc_(layerconfig->norm_type, layer.activation);
            layer.layer_type = layer_dense;
            layer.dense = calloc(1, sizeof(dense_t));
            *layer.dense =
                dense_alloc(layerconfig->dense_input_channels_ * layerconfig->dense_input_y_ * layerconfig->dense_input_x_, layerconfig->dense_output_size);
            assert(layer.activation);
            assert(layer.activation_g);
            assert(layer.activation_function);
            assert(layer.norm);
            assert(layer.dense);
            break;
        }
        case layer_convolution: {
            uint64_t new_size_y = CONVOLUTION_OUTPUT_SIZE(layerconfig->convolution_input_y_, layerconfig->convolution_kernel_size,
                                                          layerconfig->convolution_kernel_stride, layerconfig->convolution_kernel_padding);
            uint64_t new_size_x = CONVOLUTION_OUTPUT_SIZE(layerconfig->convolution_input_x_, layerconfig->convolution_kernel_size,
                                                          layerconfig->convolution_kernel_stride, layerconfig->convolution_kernel_padding);
            layer.activation = calloc(1, sizeof(tensor_t));
            *layer.activation = tensor_alloc(1, layerconfig->convolution_filters, new_size_y, new_size_x);
            layer.activation_g = calloc(1, sizeof(tensor_t));
            *layer.activation_g = tensor_alloc(1, layerconfig->convolution_filters, new_size_y, new_size_x);
            layer.activation_function = calloc(1, sizeof(activation_t));
            *layer.activation_function = activation_alloc_(layerconfig->activation_function, 1, layerconfig->dense_input_channels_, layerconfig->dense_input_y_,
                                                           layerconfig->dense_input_x_);
            layer.norm = calloc(1, sizeof(norm_t));
            *layer.norm = norm_alloc_(layerconfig->norm_type, layer.activation);
            layer.layer_type = layer_convolution;
            layer.convolution = calloc(1, sizeof(convolution_t));
            *layer.convolution = convolution_alloc(layerconfig->convolution_input_channels_, layerconfig->convolution_input_y_,
                                                   layerconfig->convolution_input_x_, layerconfig->convolution_filters, layerconfig->convolution_kernel_size,
                                                   layerconfig->convolution_kernel_stride, layerconfig->convolution_kernel_padding);
            assert(layer.activation);
            assert(layer.activation_g);
            assert(layer.activation_function);
            assert(layer.norm);
            assert(layer.convolution);
            break;
        }
        case layer_reduce: {
            uint64_t new_size_y = REDUCE_OUTPUT_SIZE(layerconfig->reduce_input_y_, layerconfig->reduce_kernel_size, layerconfig->reduce_kernel_stride);
            uint64_t new_size_x = REDUCE_OUTPUT_SIZE(layerconfig->reduce_input_x_, layerconfig->reduce_kernel_size, layerconfig->reduce_kernel_stride);
            layer.activation = calloc(1, sizeof(tensor_t));
            *layer.activation = tensor_alloc(1, layerconfig->reduce_input_channels_, new_size_y, new_size_x);
            layer.activation_g = calloc(1, sizeof(tensor_t));
            *layer.activation_g = tensor_alloc(1, layerconfig->reduce_input_channels_, new_size_y, new_size_x);
            layer.layer_type = layer_reduce;
            layer.reduce = calloc(1, sizeof(reduce_t));
            *layer.reduce = reduce_alloc(layerconfig->reduce_type, layerconfig->reduce_input_channels_, layerconfig->reduce_input_y_,
                                         layerconfig->reduce_input_x_, layerconfig->reduce_kernel_size, layerconfig->reduce_kernel_stride);
            assert(layer.activation);
            assert(layer.activation_g);
            assert(layerconfig->norm_type == norm_none);
            assert(layerconfig->activation_function == activation_identity);
            assert(layer.reduce);
            break;
        }
        case layer_split: {
            layer.activation = calloc(1, sizeof(tensor_t));
            *layer.activation =
                tensor_alloc(1, layerconfig->split_filters * layerconfig->split_input_channels_, layerconfig->split_input_y_, layerconfig->split_input_x_);
            layer.activation_g = calloc(1, sizeof(tensor_t));
            *layer.activation_g =
                tensor_alloc(1, layerconfig->split_filters * layerconfig->split_input_channels_, layerconfig->split_input_y_, layerconfig->split_input_x_);
            layer.activation_function = calloc(1, sizeof(activation_t));
            *layer.activation_function = activation_alloc_(layerconfig->activation_function, 1, layerconfig->dense_input_channels_, layerconfig->dense_input_y_,
                                                           layerconfig->dense_input_x_);
            layer.norm = calloc(1, sizeof(norm_t));
            *layer.norm = norm_alloc_(layerconfig->norm_type, layer.activation);
            layer.layer_type = layer_split;
            layer.split = calloc(1, sizeof(split_t));
            *layer.split =
                split_alloc(layerconfig->split_filters, layerconfig->split_input_channels_, layerconfig->split_input_y_, layerconfig->split_input_x_);
            assert(layer.activation);
            assert(layer.activation_g);
            assert(layer.activation_function);
            assert(layer.norm);
            assert(layer.split);
            break;
        }
    }
    return layer;
}
void layer_free(layer_t *layer) {
    switch(layer->layer_type) {
        case layer_input: {
            tensor_free(layer->activation);
            free(layer->activation);
            tensor_free(layer->activation_g);
            free(layer->activation_g);
            break;
        }
        case layer_dense: {
            tensor_free(layer->activation);
            free(layer->activation);
            tensor_free(layer->activation_g);
            free(layer->activation_g);
            dense_free(layer->dense);
            free(layer->dense);
            activation_free_(layer->activation_function);
            free(layer->activation_function);
            norm_free_(layer->norm);
            free(layer->norm);
            break;
        }
        case layer_convolution: {
            tensor_free(layer->activation);
            free(layer->activation);
            tensor_free(layer->activation_g);
            free(layer->activation_g);
            convolution_free(layer->convolution);
            free(layer->convolution);
            activation_free_(layer->activation_function);
            free(layer->activation_function);
            norm_free_(layer->norm);
            free(layer->norm);
            break;
        }
        case layer_reduce: {
            tensor_free(layer->activation);
            free(layer->activation);
            tensor_free(layer->activation_g);
            free(layer->activation_g);
            // reduce_free(layer->reduce);
            free(layer->reduce);
            break;
        }
        case layer_split: {
            tensor_free(layer->activation);
            free(layer->activation);
            tensor_free(layer->activation_g);
            free(layer->activation_g);
            split_free(layer->split);
            free(layer->split);
            activation_free_(layer->activation_function);
            free(layer->activation_function);
            norm_free_(layer->norm);
            free(layer->norm);
            break;
        }
    }
}

/* TODO: Make learning a parameter in `neuralnet_learn()` and not here. */
neuralnet_t neuralnet_alloc(uint64_t layers, layerconfig_t **layerconfig, double learning) {
    neuralnet_t neuralnet = {
        .layers = layers,
        .layer = calloc(layers, sizeof(layer_t)),
        .forward = calloc(1, sizeof(linearized_t)),
        .backward = calloc(1, sizeof(linearized_t)),
        .learn = calloc(1, sizeof(linearized_t)),
    };
    assert(neuralnet.layer);
    assert(neuralnet.forward);
    *neuralnet.forward = linearized_alloc();
    assert(neuralnet.backward);
    *neuralnet.backward = linearized_alloc();
    assert(neuralnet.learn);
    *neuralnet.learn = linearized_alloc();

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
            case layer_dense: {
                layerconfig[layer]->dense_input_channels_ = previous_z;
                layerconfig[layer]->dense_input_y_ = previous_y;
                layerconfig[layer]->dense_input_x_ = previous_x;
                neuralnet.layer[layer] = layer_alloc(layerconfig[layer]);
                break;
            }
            case layer_convolution: {
                layerconfig[layer]->convolution_input_channels_ = previous_z;
                layerconfig[layer]->convolution_input_y_ = previous_y;
                layerconfig[layer]->convolution_input_x_ = previous_x;
                neuralnet.layer[layer] = layer_alloc(layerconfig[layer]);
                break;
            }
            case layer_reduce: {
                layerconfig[layer]->reduce_input_channels_ = previous_z;
                layerconfig[layer]->reduce_input_y_ = previous_y;
                layerconfig[layer]->reduce_input_x_ = previous_x;
                neuralnet.layer[layer] = layer_alloc(layerconfig[layer]);
                break;
            }
            case layer_split: {
                layerconfig[layer]->split_input_channels_ = previous_z;
                layerconfig[layer]->split_input_y_ = previous_y;
                layerconfig[layer]->split_input_x_ = previous_x;
                neuralnet.layer[layer] = layer_alloc(layerconfig[layer]);
                break;
            }
            case layer_input: {
                ERROR("ERROR: Tried to allocate input layer at a layer with index %lu\n", layer);
            }
        }
    }

    for(uint64_t layer = 1; layer < neuralnet.layers; layer++) {
        switch(neuralnet.layer[layer].layer_type) {
            case layer_dense: {
                dense_forward(neuralnet.layer[layer - 1].activation, neuralnet.layer[layer].dense, neuralnet.layer[layer].activation);
                activation_activate_(neuralnet.layer[layer].activation, neuralnet.layer[layer].activation_function);
                norm_apply_(neuralnet.layer[layer].norm, neuralnet.layer[layer].activation);
                break;
            }
            case layer_convolution: {
                convolution_forward(neuralnet.layer[layer - 1].activation, neuralnet.layer[layer].convolution, neuralnet.layer[layer].activation);
                activation_activate_(neuralnet.layer[layer].activation, neuralnet.layer[layer].activation_function);
                norm_apply_(neuralnet.layer[layer].norm, neuralnet.layer[layer].activation);
                break;
            }
            case layer_reduce: {
                reduce_forward(neuralnet.layer[layer - 1].activation, neuralnet.layer[layer].reduce, neuralnet.layer[layer].activation);
                break;
            }
            case layer_split: {
                split_forward(neuralnet.layer[layer - 1].activation, neuralnet.layer[layer].split, neuralnet.layer[layer].activation);
                activation_activate_(neuralnet.layer[layer].activation, neuralnet.layer[layer].activation_function);
                norm_apply_(neuralnet.layer[layer].norm, neuralnet.layer[layer].activation);
                break;
            }
            case layer_input: {
                ERROR("ERROR: Input layer at layer %lu. I don't even know how this can possibly happen.\n", layer);
            }
        }
    }
    /* NOTE: Has to be done like this to ensure that each activation tensor gets resized back to it's needed shape. */
    for(int64_t layer = neuralnet.layers - 1; layer >= 0; layer--) {
        switch(neuralnet.layer[layer].layer_type) {
            case layer_dense: {
                linearized_from_op(neuralnet.forward, neuralnet.layer[layer].activation->op);
                break;
            }
            case layer_convolution: {
                linearized_from_op(neuralnet.forward, neuralnet.layer[layer].activation->op);
                break;
            }
            case layer_reduce: {
                linearized_from_op(neuralnet.forward, neuralnet.layer[layer].activation->op);
                break;
            }
            case layer_split: {
                linearized_from_op(neuralnet.forward, neuralnet.layer[layer].activation->op);
                break;
            }
            case layer_input: {
                linearized_from_op(neuralnet.forward, neuralnet.layer[layer].activation->op);
                break;
            }
        }
    }
    for(uint64_t layer = 1; layer < neuralnet.layers; layer++) {
        switch(neuralnet.layer[layer].layer_type) {
            case layer_dense: {
                tensor_set_unary(neuralnet.layer[layer - 1].activation_g, 0);
                dense_backward(neuralnet.layer[layer - 1].activation, neuralnet.layer[layer - 1].activation_g, neuralnet.layer[layer].dense,
                               neuralnet.layer[layer].activation_g);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer - 1].activation_g->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].dense->weights_g->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].dense->weights->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].dense->biases_g->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].dense->biases->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer - 1].activation->op);
                break;
            }
            case layer_convolution: {
                /* NOTE: This is `padded_grad_` here, because gradients get calculate in there and then copied into `activation_g`. */
                tensor_set_unary(neuralnet.layer[layer].convolution->padded_grad_, 0);
                convolution_backward(neuralnet.layer[layer - 1].activation, neuralnet.layer[layer - 1].activation_g, neuralnet.layer[layer].convolution,
                                     neuralnet.layer[layer].activation, neuralnet.layer[layer].activation_g);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].convolution->biases_g->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].convolution->biases->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].convolution->weights_g->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer - 1].activation_g->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer - 1].activation->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].convolution->weights->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].convolution->padded_grad_->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].convolution->padded_input_->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].activation_g->op);
                break;
            }
            case layer_reduce: {
                tensor_set_unary(neuralnet.layer[layer - 1].activation_g, 0);
                reduce_backward(neuralnet.layer[layer - 1].activation_g, neuralnet.layer[layer].reduce, neuralnet.layer[layer].activation_g);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer - 1].activation_g->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer - 1].activation->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].activation_g->op);
                break;
            }
            case layer_split: {
                tensor_set_unary(neuralnet.layer[layer - 1].activation_g, 0);
                split_backward(neuralnet.layer[layer - 1].activation, neuralnet.layer[layer - 1].activation_g, neuralnet.layer[layer].split,
                               neuralnet.layer[layer].activation, neuralnet.layer[layer].activation_g);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].split->biases_g->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].split->biases->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].split->weights_g->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].split->weights->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer - 1].activation_g->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer - 1].activation->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].activation_g->op);
                break;
            }
            case layer_input: {
                ERROR("ERROR: Input layer at layer %lu. I don't even know how this can possibly happen.\n", layer);
            }
        }
        for(uint64_t layer = 1; layer < neuralnet.layers; layer++) {
            switch(neuralnet.layer[layer].layer_type) {
                case layer_dense: {
                    tensor_multiply_unary(neuralnet.layer[layer].dense->weights_g, learning);
                    tensor_subtract_binary(neuralnet.layer[layer].dense->weights, neuralnet.layer[layer].dense->weights_g);
                    tensor_set_unary(neuralnet.layer[layer].dense->weights_g, 0);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].dense->weights->op);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].dense->weights_g->op);
                    tensor_multiply_unary(neuralnet.layer[layer].dense->biases_g, learning);
                    tensor_subtract_binary(neuralnet.layer[layer].dense->biases, neuralnet.layer[layer].dense->biases_g);
                    tensor_set_unary(neuralnet.layer[layer].dense->biases_g, 0);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].dense->biases->op);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].dense->biases_g->op);
                    break;
                }
                case layer_convolution: {
                    tensor_multiply_unary(neuralnet.layer[layer].convolution->weights_g, learning);
                    tensor_subtract_binary(neuralnet.layer[layer].convolution->weights, neuralnet.layer[layer].convolution->weights_g);
                    tensor_set_unary(neuralnet.layer[layer].convolution->weights_g, 0);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].convolution->weights->op);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].convolution->weights_g->op);
                    tensor_multiply_unary(neuralnet.layer[layer].convolution->biases_g, learning);
                    tensor_subtract_binary(neuralnet.layer[layer].convolution->biases, neuralnet.layer[layer].convolution->biases_g);
                    tensor_set_unary(neuralnet.layer[layer].convolution->biases_g, 0);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].convolution->biases->op);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].convolution->biases_g->op);
                    break;
                }
                case layer_reduce: {
                    /* Nothing to update. */
                    break;
                }
                case layer_split: {
                    tensor_multiply_unary(neuralnet.layer[layer].split->biases_g, learning);
                    tensor_subtract_binary(neuralnet.layer[layer].split->biases, neuralnet.layer[layer].split->biases_g);
                    tensor_set_unary(neuralnet.layer[layer].split->biases_g, 0);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].split->biases->op);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].split->biases_g->op);
                    tensor_multiply_unary(neuralnet.layer[layer].split->weights_g, learning);
                    tensor_subtract_binary(neuralnet.layer[layer].split->weights, neuralnet.layer[layer].split->weights_g);
                    tensor_set_unary(neuralnet.layer[layer].split->weights_g, 0);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].split->weights->op);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].split->weights_g->op);
                    break;
                }
                case layer_input: {
                    ERROR("ERROR: Input layer at layer %lu. I don't even know how this can possibly happen.\n", layer);
                }
            }
        }
    }

    return neuralnet;
}
void neuralnet_free(neuralnet_t *neuralnet) {
    for(uint64_t i = 0; i < neuralnet->layers; i++) { layer_free(&neuralnet->layer[i]); }
    free(neuralnet->layer);
    linearized_free(neuralnet->forward);
    free(neuralnet->forward);
    linearized_free(neuralnet->backward);
    free(neuralnet->backward);
    linearized_free(neuralnet->learn);
    free(neuralnet->learn);
}
void neuralnet_random(neuralnet_t *neuralnet) {
    for(uint64_t layer = 1; layer < neuralnet->layers; layer++) {
        switch(neuralnet->layer[layer].layer_type) {
            case layer_dense: {
                tensor_random_unary(neuralnet->layer[layer].dense->biases);
                tensor_random_unary(neuralnet->layer[layer].dense->weights);
                tensor_cpu_realize(neuralnet->layer[layer].dense->biases);
                tensor_cpu_realize(neuralnet->layer[layer].dense->weights);
                break;
            }
            case layer_convolution: {
                tensor_random_unary(neuralnet->layer[layer].convolution->biases);
                tensor_random_unary(neuralnet->layer[layer].convolution->weights);
                tensor_cpu_realize(neuralnet->layer[layer].convolution->biases);
                tensor_cpu_realize(neuralnet->layer[layer].convolution->weights);
                break;
            }
            case layer_reduce: {
                /* Nothing to initialize. */
                break;
            }
            case layer_split: {
                tensor_random_unary(neuralnet->layer[layer].split->biases);
                tensor_random_unary(neuralnet->layer[layer].split->weights);
                tensor_cpu_realize(neuralnet->layer[layer].split->biases);
                tensor_cpu_realize(neuralnet->layer[layer].split->weights);
                break;
            }
            case layer_input: {
                ERROR("ERROR: Input layer at layer %lu. I don't even know how this can possibly happen.\n", layer);
            }
        }
    }
}
void neuralnet_forward(neuralnet_t *neuralnet, tensor_t *input) {
    assert(neuralnet->forward);
    tensor_copy_binary(NEURALNET_INPUT_(neuralnet).activation, input);
    tensor_cpu_realize(NEURALNET_INPUT_(neuralnet).activation);
    linearized_run(neuralnet->forward);
}
void neuralnet_backward(neuralnet_t *neuralnet, tensor_t *training_input, tensor_t *training_output) {
    assert(training_input->buffer->a_size == training_output->buffer->a_size);
    assert(neuralnet->backward);
    uint64_t training_samples = training_input->buffer->a_size;
    uint64_t input_z = training_input->buffer->z_size;
    uint64_t input_y = training_input->buffer->y_size;
    uint64_t input_x = training_input->buffer->x_size;
    uint64_t output_z = training_output->buffer->z_size;
    uint64_t output_y = training_output->buffer->y_size;
    uint64_t output_x = training_output->buffer->x_size;
    tensor_resize_move(training_input, 1, input_z, input_y, input_x);
    tensor_resize_move(training_output, 1, output_z, output_y, output_x);
    for(uint64_t sample = 0; sample < training_samples; sample++) {
        tensor_offset_move(training_input, sample, 0, 0, 0);
        tensor_offset_move(training_output, sample, 0, 0, 0);
        neuralnet_forward(neuralnet, training_input);
        tensor_copy_binary(NEURALNET_OUTPUT_(neuralnet).activation_g, NEURALNET_OUTPUT_(neuralnet).activation);
        tensor_subtract_binary(NEURALNET_OUTPUT_(neuralnet).activation_g, training_output);
        // tensor_multiply_unary(NEURALNET_OUTPUT_(neuralnet).activation_g, 2);
        tensor_cpu_realize(NEURALNET_OUTPUT_(neuralnet).activation_g);
        linearized_run(neuralnet->backward);
    }
    tensor_resize_move(training_input, training_samples, input_z, input_y, input_x);
    tensor_offset_move(training_input, 0, 0, 0, 0);
    tensor_cpu_realize(training_input);
    tensor_resize_move(training_output, training_samples, output_z, output_y, output_x);
    tensor_offset_move(training_output, 0, 0, 0, 0);
    tensor_cpu_realize(training_output);
}
/* NOTE: Have to call `neuralnet_backward()` before this one. This also clears the gradients. */
void neuralnet_learn(neuralnet_t *neuralnet) {
    assert(neuralnet->learn);
    linearized_run(neuralnet->learn);
}
void neuralnet_print(neuralnet_t *neuralnet, int padding, int offset, const char *name) {
    if(strncmp(name, "", 1)) {
        printf("%*s%s\n", offset, "", name);
    } else {
        printf("%*sneuralnet\n", offset, "");
    }
    for(uint64_t layer = 0; layer < neuralnet->layers; layer++) {
        switch(neuralnet->layer[layer].layer_type) {
            case layer_dense: {
                printf("%*slayer[%lu] dense\n", offset + padding, "", layer);
                dense_print(neuralnet->layer[layer].dense, padding, offset + 2 * padding, "");
                break;
            }
            case layer_convolution: {
                printf("%*slayer[%lu] convolution\n", offset + padding, "", layer);
                convolution_print(neuralnet->layer[layer].convolution, padding, offset + 2 * padding, "");
                break;
            }
            case layer_reduce: {
                printf("%*slayer[%lu] reduce\n", offset + padding, "", layer);
                reduce_print(neuralnet->layer[layer].reduce, padding, offset + 2 * padding, "");
                break;
            }
            case layer_split: {
                printf("%*slayer[%lu] split\n", offset + padding, "", layer);
                split_print(neuralnet->layer[layer].split, padding, offset + 2 * padding, "");
                break;
            }
            case layer_input: {
                printf("%*slayer[%lu] input\n", offset + padding, "", layer);
                printf("%*sinput\n", offset + 2 * padding, "");
                tensor_print(neuralnet->layer[layer].activation, padding, offset + 3 * padding, "activation");
                tensor_print(neuralnet->layer[layer].activation_g, padding, offset + 3 * padding, "activation_g");
            }
        }
    }
}
void neuralnet_print_shape(neuralnet_t *neuralnet, int padding, int offset, const char *name) {
    if(strncmp(name, "", 1)) {
        printf("%*s%s shape\n", offset, "", name);
    } else {
        printf("%*sneuralnet shape\n", offset, "");
    }
    for(uint64_t layer = 0; layer < neuralnet->layers; layer++) {
        switch(neuralnet->layer[layer].layer_type) {
            case layer_dense: {
                printf("%*slayer[%lu] dense\n", offset + padding, "", layer);
                dense_print_shape(neuralnet->layer[layer].dense, padding, offset + 2 * padding, "");
                printf("%*s{%lu, %lu, %lu, %lu} %lu\n", offset + 2 * padding, "", neuralnet->layer[layer].activation->buffer->a_size,
                       neuralnet->layer[layer].activation->buffer->z_size, neuralnet->layer[layer].activation->buffer->y_size,
                       neuralnet->layer[layer].activation->buffer->x_size, neuralnet->layer[layer].activation->buffer->offset);
                break;
            }
            case layer_convolution: {
                printf("%*slayer[%lu] convolution\n", offset + padding, "", layer);
                convolution_print_shape(neuralnet->layer[layer].convolution, padding, offset + 2 * padding, "");
                printf("%*s{%lu, %lu, %lu, %lu} %lu\n", offset + 2 * padding, "", neuralnet->layer[layer].activation->buffer->a_size,
                       neuralnet->layer[layer].activation->buffer->z_size, neuralnet->layer[layer].activation->buffer->y_size,
                       neuralnet->layer[layer].activation->buffer->x_size, neuralnet->layer[layer].activation->buffer->offset);
                break;
            }
            case layer_reduce: {
                printf("%*slayer[%lu] reduce\n", offset + padding, "", layer);
                // reduce_print_shape(neuralnet->layer[layer].reduce, padding, offset + 2 * padding, "");
                printf("%*s{%lu, %lu, %lu, %lu} %lu\n", offset + 2 * padding, "", neuralnet->layer[layer].activation->buffer->a_size,
                       neuralnet->layer[layer].activation->buffer->z_size, neuralnet->layer[layer].activation->buffer->y_size,
                       neuralnet->layer[layer].activation->buffer->x_size, neuralnet->layer[layer].activation->buffer->offset);
                break;
            }
            case layer_split: {
                printf("%*slayer[%lu] split\n", offset + padding, "", layer);
                split_print_shape(neuralnet->layer[layer].split, padding, offset + 2 * padding, "");
                printf("%*s{%lu, %lu, %lu, %lu} %lu\n", offset + 2 * padding, "", neuralnet->layer[layer].activation->buffer->a_size,
                       neuralnet->layer[layer].activation->buffer->z_size, neuralnet->layer[layer].activation->buffer->y_size,
                       neuralnet->layer[layer].activation->buffer->x_size, neuralnet->layer[layer].activation->buffer->offset);
                break;
            }
            case layer_input: {
                printf("%*slayer[%lu] input\n", offset + padding, "", layer);
                printf("%*s{%lu, %lu, %lu, %lu} %lu\n", offset + 2 * padding, "", neuralnet->layer[layer].activation->buffer->a_size,
                       neuralnet->layer[layer].activation->buffer->z_size, neuralnet->layer[layer].activation->buffer->y_size,
                       neuralnet->layer[layer].activation->buffer->x_size, neuralnet->layer[layer].activation->buffer->offset);
            }
        }
    }
}
