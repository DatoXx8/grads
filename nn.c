#include <CL/cl.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compile.h"
#include "linearize.h"
#include "nn.h"
#include "runtimes/cl.h"
#include "tensor.h"
#include "utils.h"

static activation_t _activation_alloc(enum activation_e activation_type, int64_t a, int64_t z, int64_t y, int64_t x,
                                      cl_context context) {
    assert(a > 0);
    assert(z > 0);
    assert(y > 0);
    assert(x > 0);
    activation_t activation = {0};
    switch(activation_type) {
        case activation_none: {
            activation.type = activation_none;
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
            *activation.intermediary = tensor_alloc(a, z, y, x, context);
            break;
        }
        case activation_gelu: {
            activation.type = activation_gelu;
            activation.intermediary = calloc(1, sizeof(tensor_t));
            assert(activation.intermediary);
            *activation.intermediary = tensor_alloc(a, z, y, x, context);
            break;
        }
        case activation_leaky: {
            activation.type = activation_leaky;
            activation.intermediary = calloc(1, sizeof(tensor_t));
            assert(activation.intermediary);
            *activation.intermediary = tensor_alloc(a, z, y, x, context);
            break;
        }
    }
    return activation;
}
static void _activation_free(activation_t *activation) {
    assert(activation);
    switch(activation->type) {
        case activation_none: {
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
const double LEAKY_FACTOR = 0.1;
/* TODO: Implement the other activation functions */
/* TODO: This should also calculate the derivatives if a new flag `forward_only` is not set for the neuralnet. */
static void _activation_activate(tensor_t *tensor, activation_t *activation) {
    assert(tensor);
    assert(activation);
    switch(activation->type) {
        case activation_none: {
            break;
        }
        case activation_relu: {
            tensor_unary_max(tensor, 0);
            break;
        }
        case activation_sigmoid: {
            tensor_unary_multiply(tensor, -1);
            tensor_unary_exp(tensor);
            tensor_unary_add(tensor, 1);
            tensor_unary_reciprocal(tensor);
            break;
        }
        case activation_tanh: {
            tensor_unary_tanh(tensor);
            break;
        }
        case activation_silu: {
            tensor_binary_copy(activation->intermediary, tensor);
            tensor_unary_max(tensor, 0);
            tensor_unary_multiply(tensor, -1);
            tensor_unary_exp(activation->intermediary);
            tensor_unary_add(activation->intermediary, 1);
            tensor_unary_reciprocal(activation->intermediary);
            tensor_binary_multiply(tensor, activation->intermediary);
            break;
        }
        case activation_gelu: {
            /* This is an approximation and can be found here: https://paperswithcode.com/method/gelu */
            /* TODO: Write a brute forcer that minimizes the error by varying the constant. */
            tensor_binary_copy(activation->intermediary, tensor);
            tensor_unary_max(tensor, 0);
            tensor_unary_multiply(activation->intermediary, 1.703);
            tensor_unary_multiply(tensor, -1);
            tensor_unary_exp(activation->intermediary);
            tensor_unary_add(activation->intermediary, 1);
            tensor_unary_reciprocal(activation->intermediary);
            tensor_binary_multiply(tensor, activation->intermediary);
            break;
        }
        case activation_leaky: {
            tensor_binary_copy(activation->intermediary, tensor);
            tensor_unary_multiply(activation->intermediary, LEAKY_FACTOR);
            tensor_binary_max(tensor, activation->intermediary);
            break;
        }
    }
}
static norm_t _norm_alloc(enum norm_e type, tensor_t *tensor, cl_context context) {
    assert(tensor);
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
            *norm.batch_expected = tensor_alloc(tensor->buffer->inh_a, tensor->buffer->inh_z, tensor->buffer->inh_y,
                                                tensor->buffer->inh_x, context);
            *norm.batch_variance = tensor_alloc(tensor->buffer->inh_a, tensor->buffer->inh_z, tensor->buffer->inh_y,
                                                tensor->buffer->inh_x, context);
            TODO();
            break;
        }
        case norm_layer: {
            norm.layer_expected = calloc(1, sizeof(tensor_t));
            norm.layer_variance = calloc(1, sizeof(tensor_t));
            norm.layer_intermediary = calloc(1, sizeof(tensor_t));
            assert(norm.layer_expected);
            assert(norm.layer_variance);
            assert(norm.layer_intermediary);
            *norm.layer_expected = tensor_alloc(1, 1, 1, 1, context);
            *norm.layer_variance = tensor_alloc(1, 1, 1, 1, context);
            *norm.layer_intermediary = tensor_alloc(tensor->buffer->inh_a, tensor->buffer->inh_z, tensor->buffer->inh_y,
                                                    tensor->buffer->inh_x, context);
            break;
        }
        case norm_simple: {
            norm.simple_max = calloc(1, sizeof(tensor_t));
            norm.simple_intermediary = calloc(1, sizeof(tensor_t));
            assert(norm.simple_max);
            assert(norm.simple_intermediary);
            *norm.simple_max = tensor_alloc(1, 1, 1, 1, context);
            *norm.simple_intermediary = tensor_alloc(tensor->buffer->inh_a, tensor->buffer->inh_z,
                                                     tensor->buffer->inh_y, tensor->buffer->inh_x, context);
            break;
        }
    }
    return norm;
}
static void _norm_free(norm_t *norm) {
    assert(norm);
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
const double EPSILON = 1e-6;
static void _norm_calculate_layer(norm_t *norm, tensor_t *tensor) {
    assert(EPSILON > 0);
    assert(norm);
    assert(tensor);
    tensor_reduce_avg(norm->layer_expected, tensor);
    tensor_binary_copy(norm->layer_intermediary, tensor);
    /* The reason this is commented out is quite ugly. Basically when realizing the tensor the expected value
     * already gets subtracted before in the calculation. I understand that isn't nice, but otherwise I would have to
     * make a copy here and that would be worse. */
    // tensor_lbinary_subtract(norm->layer_intermediary, norm->layer_expected);
    tensor_unary_square(norm->layer_intermediary);
    tensor_reduce_avg(norm->layer_variance, norm->layer_intermediary);
    /* Added to avoid dividing by 0 when normalizing the layer. */
    tensor_unary_add(norm->layer_variance, EPSILON);
    tensor_unary_sqrt(norm->layer_variance);
}
/* This ones tricky. Even the function signature isn't obvious. */
static void _norm_calculate_batch(void) {}
static void _norm_apply(norm_t *norm, tensor_t *tensor) {
    assert(norm);
    assert(tensor);
    switch(norm->type) {
        case norm_none: {
            break;
        }
        case norm_batch: {
            _norm_calculate_batch();
            assert(0);
            break;
        }
        case norm_layer: {
            _norm_calculate_layer(norm, tensor);
            tensor_lbinary_subtract(tensor, norm->layer_expected);
            tensor_lbinary_divide(tensor, norm->layer_variance);
            break;
        }
        case norm_simple: {
            tensor_binary_copy(norm->simple_intermediary, tensor);
            tensor_unary_absolute(norm->simple_intermediary);
            tensor_reduce_max(norm->simple_max, norm->simple_intermediary);
            tensor_lbinary_divide(tensor, norm->simple_max);
            break;
        }
    }
}

dense_t dense_alloc(int64_t input_size, int64_t output_size, cl_context context) {
    assert(input_size > 0);
    assert(output_size > 0);
    dense_t dense = {
        .biases = calloc(1, sizeof(tensor_t)),
        .biases_g = calloc(1, sizeof(tensor_t)),
        .weights = calloc(1, sizeof(tensor_t)),
        .weights_g = calloc(1, sizeof(tensor_t)),

        ._input_size = input_size,
        .output_size = output_size,

        ._input_multiply_temp = calloc(1, sizeof(tensor_t)),
        ._output_multiply_temp = calloc(1, sizeof(tensor_t)),
        ._full_temp = calloc(1, sizeof(tensor_t)),
    };
    assert(dense.biases);
    assert(dense.biases_g);
    assert(dense.weights);
    assert(dense.weights_g);
    assert(dense._input_multiply_temp);
    assert(dense._output_multiply_temp);
    assert(dense._full_temp);

    *dense.biases = tensor_alloc(1, 1, 1, output_size, context);
    *dense.biases_g = tensor_alloc(1, 1, 1, output_size, context);
    *dense.weights = tensor_alloc(1, 1, input_size, output_size, context);
    *dense.weights_g = tensor_alloc(1, 1, input_size, output_size, context);
    *dense._input_multiply_temp = tensor_alloc(1, 1, input_size, 1, context);
    *dense._output_multiply_temp = tensor_alloc(1, 1, 1, output_size, context);
    *dense._full_temp = tensor_alloc(1, 1, input_size, output_size, context);

    return dense;
}
void dense_free(dense_t *dense) {
    assert(dense);
    tensor_free(dense->biases);
    tensor_free(dense->biases_g);
    tensor_free(dense->weights);
    tensor_free(dense->weights_g);
    tensor_free(dense->_input_multiply_temp);
    tensor_free(dense->_output_multiply_temp);
    tensor_free(dense->_full_temp);
    free(dense->biases);
    free(dense->biases_g);
    free(dense->weights);
    free(dense->weights_g);
    free(dense->_input_multiply_temp);
    free(dense->_output_multiply_temp);
    free(dense->_full_temp);
}
/* Automagically "flattens" the input tensor to shape `{1, 1, a * z * y * x, 1}`. */
void dense_forward(tensor_t *input, dense_t *dense, tensor_t *output) {
    assert(input);
    assert(dense);
    assert(output);
    int64_t input_z = input->buffer->inh_z;
    int64_t input_y = input->buffer->inh_y;
    int64_t input_x = input->buffer->inh_x;
    int64_t output_z = output->buffer->inh_z;
    int64_t output_y = output->buffer->inh_y;
    int64_t output_x = output->buffer->inh_x;

    tensor_move_reshape(input, 1, 1, dense->_input_size, 1);
    tensor_move_resize(dense->weights, 1, 1, dense->_input_size, 1);
    tensor_move_resize(output, 1, 1, 1, 1);

    for(int64_t i = 0; i < dense->output_size; i++) {
        tensor_move_offset(dense->weights, 0, 0, 0, i);
        tensor_move_offset(output, 0, 0, 0, i);
        tensor_binary_copy(dense->_input_multiply_temp, dense->weights);
        tensor_binary_multiply(dense->_input_multiply_temp, input);
        tensor_reduce_sum(output, dense->_input_multiply_temp);
    }

    tensor_move_reshape(input, 1, input_z, input_y, input_x);
    tensor_move_offset(input, 0, 0, 0, 0);
    tensor_move_resize(output, 1, output_z, output_y, output_x);
    tensor_move_offset(output, 0, 0, 0, 0);
    tensor_move_resize(dense->weights, 1, 1, dense->_input_size, dense->output_size);
    tensor_move_offset(dense->weights, 0, 0, 0, 0);

    tensor_binary_add(output, dense->biases);
}
void dense_backward(tensor_t *input, tensor_t *input_gradient, dense_t *dense, tensor_t *output_gradient) {
    assert(input);
    assert(input_gradient);
    assert(dense);
    assert(output_gradient);
    int64_t input_z = input->buffer->inh_z;
    int64_t input_y = input->buffer->inh_y;
    int64_t input_x = input->buffer->inh_x;
    /* Biases */
    tensor_binary_add(dense->biases_g, output_gradient);
    /* Weights */
    tensor_move_resize(dense->_full_temp, 1, 1, 1, dense->output_size);
    for(int64_t i = 0; i < dense->_input_size; i++) {
        tensor_move_offset(dense->_full_temp, 0, 0, i, 0);
        tensor_binary_copy(dense->_full_temp, output_gradient);
    }
    tensor_move_resize(dense->_full_temp, 1, 1, dense->_input_size, 1);
    tensor_move_reshape(input, 1, 1, dense->_input_size, 1);
    for(int64_t i = 0; i < dense->output_size; i++) {
        tensor_move_offset(dense->_full_temp, 0, 0, 0, i);
        tensor_binary_multiply(dense->_full_temp, input);
    }
    tensor_move_resize(dense->_full_temp, 1, 1, dense->_input_size, dense->output_size);
    tensor_move_offset(dense->_full_temp, 0, 0, 0, 0);
    tensor_move_reshape(input, 1, input_z, input_y, input_x);
    tensor_binary_add(dense->weights_g, dense->_full_temp);
    /* Previous activation grad */
    tensor_move_reshape(input_gradient, 1, 1, dense->_input_size, 1);
    tensor_move_resize(input_gradient, 1, 1, 1, 1);
    tensor_move_resize(dense->weights, 1, 1, 1, dense->output_size);
    for(int64_t i = 0; i < dense->_input_size; i++) {
        tensor_move_offset(input_gradient, 0, 0, i, 0);
        tensor_move_offset(dense->weights, 0, 0, i, 0);
        tensor_binary_copy(dense->_output_multiply_temp, dense->weights);
        tensor_binary_multiply(dense->_output_multiply_temp, output_gradient);
        /* Sum is technically the right one I think, but avg provides better scaling. */
        tensor_reduce_sum(input_gradient, dense->_output_multiply_temp);
        // tensor_reduce_avg(input_gradient, dense->_output_multiply_temp);
    }
    tensor_move_reshape(input_gradient, 1, input_z, input_y, input_x);
    tensor_move_offset(input_gradient, 0, 0, 0, 0);
    tensor_move_resize(dense->weights, 1, 1, dense->_input_size, dense->output_size);
    tensor_move_offset(dense->weights, 0, 0, 0, 0);
}
void dense_print(dense_t *dense, int padding, int offset, const char *name) {
    assert(dense);
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
    assert(dense);
    if(strncmp(name, "", 1)) {
        printf("%*s%s dense shape\n", offset, "", name);
    } else {
        printf("%*sdense shape\n", offset, "");
    }
    printf("%*sBiases  {%lu, %lu, %lu, %lu}\n", offset + padding, "", dense->biases->buffer->sze_a,
           dense->biases->buffer->sze_z, dense->biases->buffer->sze_y, dense->biases->buffer->sze_x);
    printf("%*sWeights {%lu, %lu, %lu, %lu}\n", offset + padding, "", dense->weights->buffer->sze_a,
           dense->weights->buffer->sze_z, dense->weights->buffer->sze_y, dense->weights->buffer->sze_x);
}

convolution_t convolution_alloc(int64_t input_z, int64_t input_y, int64_t input_x, int64_t filters, int64_t kernel_size,
                                int64_t kernel_stride, int64_t kernel_padding, cl_context context) {
    assert(filters > 0);
    assert(kernel_size > 0);
    assert(kernel_stride > 0);
    assert(kernel_padding >= 0);
    assert(input_z > 0);
    assert(input_y >= kernel_size);
    assert(input_x >= kernel_size);
    convolution_t convolution = {
        ._input_z = input_z,
        ._input_y = input_y,
        ._input_x = input_x,
        .filters = filters,
        .kernel_size = kernel_size,
        .kernel_stride = kernel_stride,
        .kernel_padding = kernel_padding,

        .biases = calloc(1, sizeof(tensor_t)),
        .biases_g = calloc(1, sizeof(tensor_t)),
        .weights = calloc(1, sizeof(tensor_t)),
        .weights_g = calloc(1, sizeof(tensor_t)),

        ._padded_input = calloc(1, sizeof(tensor_t)),
        ._padded_grad = calloc(1, sizeof(tensor_t)),
        ._kernel_temp = calloc(1, sizeof(tensor_t)),
        ._single_temp = calloc(1, sizeof(tensor_t)),
    };
    assert(convolution.biases);
    assert(convolution.biases_g);
    assert(convolution.weights);
    assert(convolution.weights_g);
    assert(convolution._padded_input);
    assert(convolution._padded_grad);
    assert(convolution._kernel_temp);
    assert(convolution._single_temp);

    *convolution.biases = tensor_alloc(filters, 1, 1, 1, context);
    *convolution.biases_g = tensor_alloc(filters, 1, 1, 1, context);
    *convolution.weights = tensor_alloc(filters, input_z, kernel_size, kernel_size, context);
    *convolution.weights_g = tensor_alloc(filters, input_z, kernel_size, kernel_size, context);
    *convolution._padded_input =
        tensor_alloc(1, input_z, input_y + 2 * kernel_padding, input_x + 2 * kernel_padding, context);
    *convolution._padded_grad =
        tensor_alloc(1, input_z, input_y + 2 * kernel_padding, input_x + 2 * kernel_padding, context);
    *convolution._kernel_temp = tensor_alloc(1, input_z, kernel_size, kernel_size, context);
    *convolution._single_temp = tensor_alloc(1, 1, 1, 1, context);

    return convolution;
}
void convolution_free(convolution_t *convolution) {
    assert(convolution);
    tensor_free(convolution->biases);
    tensor_free(convolution->biases_g);
    tensor_free(convolution->weights);
    tensor_free(convolution->weights_g);
    tensor_free(convolution->_padded_input);
    tensor_free(convolution->_padded_grad);
    tensor_free(convolution->_kernel_temp);
    tensor_free(convolution->_single_temp);
    free(convolution->biases);
    free(convolution->biases_g);
    free(convolution->weights);
    free(convolution->weights_g);
    free(convolution->_padded_input);
    free(convolution->_padded_grad);
    free(convolution->_kernel_temp);
    free(convolution->_single_temp);
}
void convolution_forward(tensor_t *input, convolution_t *convolution, tensor_t *output) {
    assert(input);
    assert(convolution);
    assert(output);
    int64_t input_z = input->buffer->inh_z;
    int64_t input_y = input->buffer->inh_y;
    int64_t input_x = input->buffer->inh_x;
    int64_t output_z = output->buffer->inh_z;
    int64_t output_y = output->buffer->inh_y;
    int64_t output_x = output->buffer->inh_x;

    int64_t input_x_max = input_x + convolution->kernel_padding - 1;
    int64_t input_y_max = input_y + convolution->kernel_padding - 1;

    int64_t output_y_i;
    int64_t output_x_i;
    tensor_move_offset(input, 0, 0, 0, 0);
    tensor_move_resize(convolution->biases, 1, 1, 1, 1);
    tensor_move_resize(convolution->weights, 1, input_z, convolution->kernel_size, convolution->kernel_size);
    tensor_move_resize(output, 1, 1, 1, 1);
    tensor_move_resize(convolution->_padded_input, 1, input_z, input_y, input_x);
    tensor_move_offset(convolution->_padded_input, 0, 0, convolution->kernel_padding, convolution->kernel_padding);
    tensor_binary_copy(convolution->_padded_input, input);
    tensor_move_resize(convolution->_padded_input, 1, input_z, convolution->kernel_size, convolution->kernel_size);

    for(int64_t filter = 0; filter < convolution->filters; filter++) {
        tensor_move_offset(convolution->biases, filter, 0, 0, 0);
        tensor_move_offset(convolution->weights, filter, 0, 0, 0);
        output_y_i = 0;
        for(int64_t input_y_i = 0; input_y_i < input_y_max; input_y_i += convolution->kernel_stride) {
            output_x_i = 0;
            for(int64_t input_x_i = 0; input_x_i < input_x_max; input_x_i += convolution->kernel_stride) {
                tensor_move_offset(output, 0, filter, output_y_i, output_x_i);
                tensor_move_offset(convolution->_padded_input, 0, 0, input_y_i, input_x_i);
                tensor_binary_copy(convolution->_kernel_temp, convolution->_padded_input);
                tensor_binary_multiply(convolution->_kernel_temp, convolution->weights);
                tensor_reduce_sum(output, convolution->_kernel_temp);
                tensor_binary_add(output, convolution->biases);
                output_x_i++;
            }
            output_y_i++;
        }
    }
    tensor_move_resize(convolution->biases, convolution->filters, 1, 1, 1);
    tensor_move_offset(convolution->biases, 0, 0, 0, 0);
    tensor_move_resize(convolution->weights, convolution->filters, convolution->_input_z, convolution->kernel_size,
                       convolution->kernel_size);
    tensor_move_offset(convolution->weights, 0, 0, 0, 0);
    tensor_move_resize(output, 1, output_z, output_y, output_x);
    tensor_move_offset(output, 0, 0, 0, 0);
    tensor_move_resize(convolution->_padded_input, 1, input_z, input_y + 2 * convolution->kernel_padding,
                       input_x + 2 * convolution->kernel_padding);
    tensor_move_offset(convolution->_padded_input, 0, 0, 0, 0);
}
void convolution_backward(tensor_t *input, tensor_t *input_gradient, convolution_t *convolution, tensor_t *output,
                          tensor_t *output_gradient) {
    assert(input);
    assert(input_gradient);
    assert(convolution);
    assert(output);
    assert(output_gradient);
    int64_t input_z = input->buffer->inh_z;
    int64_t input_y = input->buffer->inh_y;
    int64_t input_x = input->buffer->inh_x;
    int64_t output_z = output->buffer->inh_z;
    int64_t output_y = output->buffer->inh_y;
    int64_t output_x = output->buffer->inh_x;
    /* Biases */
    tensor_move_resize(convolution->biases_g, 1, 1, 1, 1);
    tensor_move_resize(output_gradient, 1, 1, output_y, output_x);
    for(int64_t i = 0; i < convolution->filters; i++) {
        tensor_move_offset(convolution->biases_g, i, 0, 0, 0);
        tensor_move_offset(output_gradient, 0, i, 0, 0);
        tensor_reduce_avg(convolution->_single_temp, output_gradient);
        tensor_binary_add(convolution->biases_g, convolution->_single_temp);
    }
    tensor_move_resize(convolution->biases_g, convolution->filters, 1, 1, 1);
    tensor_move_offset(convolution->biases_g, 0, 0, 0, 0);
    tensor_move_resize(output_gradient, 1, output_z, output_y, output_x);
    tensor_move_offset(output_gradient, 0, 0, 0, 0);
    /* Weights */
    int64_t input_y_i;
    int64_t input_x_i;
    tensor_move_resize(output_gradient, 1, 1, 1, 1);
    tensor_move_offset(output_gradient, 0, 0, 0, 0);
    tensor_move_resize(convolution->weights_g, 1, input_z, convolution->kernel_size, convolution->kernel_size);
    tensor_move_offset(convolution->weights_g, 0, 0, 0, 0);
    tensor_move_resize(convolution->_padded_input, 1, input_z, convolution->kernel_size, convolution->kernel_size);
    tensor_move_offset(convolution->_padded_input, 0, 0, 0, 0);
    for(int64_t filter = 0; filter < convolution->filters; filter++) {
        tensor_move_offset(convolution->weights_g, filter, 0, 0, 0);
        input_y_i = 0;
        for(int64_t output_y_i = 0; output_y_i < output_y; output_y_i++) {
            input_x_i = 0;
            for(int64_t output_x_i = 0; output_x_i < output_x; output_x_i++) {
                tensor_move_offset(output_gradient, 0, filter, output_y_i, output_x_i);
                tensor_move_offset(convolution->_padded_input, 0, 0, input_y_i, input_x_i);
                tensor_binary_copy(convolution->_kernel_temp, convolution->_padded_input);
                tensor_lbinary_multiply(convolution->_kernel_temp, output_gradient);
                tensor_binary_add(convolution->weights_g, convolution->_kernel_temp);
                input_x_i += convolution->kernel_padding;
            }
            input_y_i += convolution->kernel_padding;
        }
    }
    tensor_move_resize(output_gradient, 1, output_z, output_y, output_x);
    tensor_move_offset(output_gradient, 0, 0, 0, 0);
    tensor_move_resize(convolution->_padded_input, 1, input_z, input_y, input_x);
    tensor_move_offset(convolution->_padded_input, 0, 0, 0, 0);
    tensor_move_resize(convolution->weights_g, convolution->filters, input_z, convolution->kernel_size,
                       convolution->kernel_size);
    tensor_move_offset(convolution->weights_g, 0, 0, 0, 0);
    /* Previous activation grad */
    tensor_move_resize(output_gradient, 1, 1, 1, 1);
    tensor_move_offset(output_gradient, 0, 0, 0, 0);
    tensor_move_resize(convolution->weights, 1, input_z, convolution->kernel_size, convolution->kernel_size);
    tensor_move_offset(convolution->weights, 0, 0, 0, 0);
    tensor_move_resize(convolution->_padded_grad, 1, input_z, convolution->kernel_size, convolution->kernel_size);
    tensor_move_offset(convolution->_padded_grad, 0, 0, 0, 0);
    for(int64_t filter = 0; filter < convolution->filters; filter++) {
        tensor_move_offset(convolution->weights, filter, 0, 0, 0);
        input_y_i = 0;
        for(int64_t output_y_i = 0; output_y_i < output_y; output_y_i++) {
            input_x_i = 0;
            for(int64_t output_x_i = 0; output_x_i < output_x; output_x_i++) {
                tensor_move_offset(output_gradient, 0, filter, output_y_i, output_x_i);
                tensor_move_offset(convolution->_padded_grad, 0, 0, output_y_i, output_x_i);
                tensor_binary_copy(convolution->_kernel_temp, convolution->weights);
                tensor_lbinary_multiply(convolution->_kernel_temp, output_gradient);
                tensor_binary_add(convolution->_padded_grad, convolution->_kernel_temp);
                input_x_i += convolution->kernel_padding;
            }
            input_y_i += convolution->kernel_padding;
        }
    }
    tensor_move_resize(output_gradient, 1, output_z, output_y, output_x);
    tensor_move_offset(output_gradient, 0, 0, 0, 0);
    tensor_move_resize(convolution->weights, convolution->filters, input_z, convolution->kernel_size,
                       convolution->kernel_size);
    tensor_move_offset(convolution->weights, 0, 0, 0, 0);
    tensor_move_resize(convolution->_padded_grad, 1, input_z, input_y, input_x);
    tensor_move_offset(convolution->_padded_grad, 0, 0, convolution->kernel_padding, convolution->kernel_padding);
    tensor_binary_copy(input_gradient, convolution->_padded_grad);

    tensor_move_resize(convolution->_padded_grad, 1, input_z, input_y + 2 * convolution->kernel_padding,
                       input_x + 2 * convolution->kernel_padding);
    tensor_move_offset(convolution->_padded_grad, 0, 0, 0, 0);
}
void convolution_print(convolution_t *convolution, int padding, int offset, const char *name) {
    assert(convolution);
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
    assert(convolution);
    if(strncmp(name, "", 1)) {
        printf("%*s%s convolution shape\n", offset, "", name);
    } else {
        printf("%*sconvolution shape\n", offset, "");
    }
    printf("%*sBiases  {%lu, %lu, %lu, %lu}\n", offset + padding, "", convolution->biases->buffer->sze_a,
           convolution->biases->buffer->sze_z, convolution->biases->buffer->sze_y, convolution->biases->buffer->sze_x);
    printf("%*sWeights {%lu, %lu, %lu, %lu}\n", offset + padding, "", convolution->weights->buffer->sze_a,
           convolution->weights->buffer->sze_z, convolution->weights->buffer->sze_y,
           convolution->weights->buffer->sze_x);
}

/* Kind of a misnomer as this doesn't allocate any dynamic memory, which is also why there is no reduce_free(). I
 * like the name continuity tho. */
reduce_t reduce_alloc(enum layer_reduce_e type, int64_t input_z, int64_t input_y, int64_t input_x, int64_t kernel_size,
                      int64_t kernel_stride) {
    assert(kernel_size > 0);
    assert(kernel_stride > 0);
    assert(input_z > 0);
    assert(input_y >= kernel_size);
    assert(input_x >= kernel_size);
    reduce_t reduce = {
        .type = type,
        ._input_z = input_z,
        ._input_y = input_y,
        ._input_x = input_x,
        .kernel_size = kernel_size,
        .kernel_stride = kernel_stride,
    };

    return reduce;
}
void reduce_forward(tensor_t *input, reduce_t *reduce, tensor_t *output) {
    assert(input);
    assert(reduce);
    assert(output);
    int64_t input_z = input->buffer->inh_z;
    int64_t input_y = input->buffer->inh_y;
    int64_t input_x = input->buffer->inh_x;
    int64_t output_z = output->buffer->inh_z;
    int64_t output_y = output->buffer->inh_y;
    int64_t output_x = output->buffer->inh_x;

    int64_t output_y_i = 0;
    int64_t output_x_i = 0;

    tensor_move_reshape(input, 1, input_z, input_y, input_x);
    tensor_move_resize(input, 1, 1, reduce->kernel_size, reduce->kernel_size);
    tensor_move_resize(output, 1, 1, 1, 1);
    /* Switch statement is on the outside cuz it only needs to be done once then. */
    switch(reduce->type) {
        case layer_reduce_max: {
            for(int64_t channel = 0; channel < reduce->_input_z; channel++) {
                output_y_i = 0;
                for(int64_t y = 0; y < reduce->_input_y - reduce->kernel_size + 1; y += reduce->kernel_stride) {
                    output_x_i = 0;
                    for(int64_t x = 0; x < reduce->_input_x - reduce->kernel_size + 1; x += reduce->kernel_stride) {
                        tensor_move_offset(input, 0, channel, y, x);
                        tensor_move_offset(output, 0, channel, output_y_i, output_x_i);
                        tensor_reduce_max(output, input);
                        output_x_i++;
                    }
                    output_y_i++;
                }
            }
            break;
        }
        case layer_reduce_min: {
            for(int64_t channel = 0; channel < reduce->_input_z; channel++) {
                output_y_i = 0;
                for(int64_t y = 0; y < reduce->_input_y - reduce->kernel_size + 1; y += reduce->kernel_stride) {
                    output_x_i = 0;
                    for(int64_t x = 0; x < reduce->_input_x - reduce->kernel_size + 1; x += reduce->kernel_stride) {
                        tensor_move_offset(input, 0, channel, y, x);
                        tensor_move_offset(output, 0, channel, output_y_i, output_x_i);
                        tensor_reduce_min(output, input);
                        output_x_i++;
                    }
                    output_y_i++;
                }
            }
            break;
        }
        case layer_reduce_avg: {
            for(int64_t channel = 0; channel < reduce->_input_z; channel++) {
                output_y_i = 0;
                for(int64_t y = 0; y < reduce->_input_y - reduce->kernel_size + 1; y += reduce->kernel_stride) {
                    output_x_i = 0;
                    for(int64_t x = 0; x < reduce->_input_x - reduce->kernel_size + 1; x += reduce->kernel_stride) {
                        tensor_move_offset(input, 0, channel, y, x);
                        tensor_move_offset(output, 0, channel, output_y_i, output_x_i);
                        tensor_reduce_avg(output, input);
                        output_x_i++;
                    }
                    output_y_i++;
                }
            }
            break;
        }
    }
    tensor_move_resize(input, 1, input_z, input_y, input_x);
    tensor_move_offset(input, 0, 0, 0, 0);
    tensor_move_resize(output, 1, output_z, output_y, output_x);
    tensor_move_offset(output, 0, 0, 0, 0);
}
/* TODO: Replace this "solution", that is just flat out wrong (it approximates it somewhat), with the real solution. */
void reduce_backward(tensor_t *input_gradient, reduce_t *reduce, tensor_t *output_gradient) {
    assert(input_gradient);
    assert(reduce);
    assert(output_gradient);
    int64_t input_z = input_gradient->buffer->inh_z;
    int64_t input_y = input_gradient->buffer->inh_y;
    int64_t input_x = input_gradient->buffer->inh_x;
    int64_t output_z = output_gradient->buffer->inh_z;
    int64_t output_y = output_gradient->buffer->inh_y;
    int64_t output_x = output_gradient->buffer->inh_x;
    int64_t input_y_i;
    int64_t input_x_i;
    tensor_move_resize(input_gradient, 1, 1, reduce->kernel_size, reduce->kernel_size);
    tensor_move_resize(output_gradient, 1, 1, 1, 1);
    switch(reduce->type) {
        case layer_reduce_max: {
            for(int64_t channel = 0; channel < input_z; channel++) {
                input_y_i = 0;
                for(int64_t output_y_i = 0; output_y_i < output_y; output_y_i++) {
                    input_x_i = 0;
                    for(int64_t output_x_i = 0; output_x_i < output_x; output_x_i++) {
                        tensor_move_offset(input_gradient, 0, channel, input_y_i, input_x_i);
                        tensor_move_offset(output_gradient, 0, channel, output_y_i, output_x_i);
                        tensor_lbinary_add(input_gradient, output_gradient);
                        input_x_i += reduce->kernel_stride;
                    }
                    input_y_i += reduce->kernel_stride;
                }
            }
            break;
        }
        case layer_reduce_min: {
            for(int64_t channel = 0; channel < input_z; channel++) {
                input_y_i = 0;
                for(int64_t output_y_i = 0; output_y_i < output_y; output_y_i++) {
                    input_x_i = 0;
                    for(int64_t output_x_i = 0; output_x_i < output_x; output_x_i++) {
                        tensor_move_offset(input_gradient, 0, channel, input_y_i, input_x_i);
                        tensor_move_offset(output_gradient, 0, channel, output_y_i, output_x_i);
                        tensor_lbinary_add(input_gradient, output_gradient);
                        input_x_i += reduce->kernel_stride;
                    }
                    input_y_i += reduce->kernel_stride;
                }
            }
            break;
        }
        case layer_reduce_avg: {
            for(int64_t channel = 0; channel < input_z; channel++) {
                input_y_i = 0;
                for(int64_t output_y_i = 0; output_y_i < output_y; output_y_i++) {
                    input_x_i = 0;
                    for(int64_t output_x_i = 0; output_x_i < output_x; output_x_i++) {
                        tensor_move_offset(input_gradient, 0, channel, input_y_i, input_x_i);
                        tensor_move_offset(output_gradient, 0, channel, output_y_i, output_x_i);
                        tensor_lbinary_add(input_gradient, output_gradient);
                        input_x_i += reduce->kernel_stride;
                    }
                    input_y_i += reduce->kernel_stride;
                }
            }
            break;
        }
    }
    tensor_move_resize(input_gradient, 1, input_z, input_y, input_x);
    tensor_move_offset(input_gradient, 0, 0, 0, 0);
    tensor_move_resize(output_gradient, 1, output_z, output_y, output_x);
    tensor_move_offset(output_gradient, 0, 0, 0, 0);
}
void reduce_print(reduce_t *reduce, int padding, int offset, const char *name) {
    assert(reduce);
    if(strncmp(name, "", 1)) {
        printf("%*s%s convolution\n", offset, "", name);
    } else {
        printf("%*sconvolution\n", offset, "");
    }
    printf("%*ssize %lu, stride %lu, z %lu, y %lu, x %lu\n", offset + padding, "", reduce->kernel_size,
           reduce->kernel_stride, reduce->_input_z, reduce->_input_y, reduce->_input_x);
}

split_t split_alloc(int64_t filters, int64_t input_z, int64_t input_y, int64_t input_x, cl_context context) {
    assert(filters > 0);
    assert(input_z > 0);
    assert(input_y > 0);
    assert(input_x > 0);
    split_t split = {
        .filters = filters,
        .input_z = input_z,
        .input_y = input_y,
        .input_x = input_x,

        .biases = calloc(1, sizeof(tensor_t)),
        .biases_g = calloc(1, sizeof(tensor_t)),
        .weights = calloc(1, sizeof(tensor_t)),
        .weights_g = calloc(1, sizeof(tensor_t)),

        ._input_temp = calloc(1, sizeof(tensor_t)),
    };
    assert(split.biases);
    assert(split.weights);
    assert(split.biases_g);
    assert(split.weights_g);
    assert(split._input_temp);

    *split.biases = tensor_alloc(filters, input_z, input_y, input_x, context);
    *split.biases_g = tensor_alloc(filters, input_z, input_y, input_x, context);
    *split.weights = tensor_alloc(filters, input_z, input_y, input_x, context);
    *split.weights_g = tensor_alloc(filters, input_z, input_y, input_x, context);
    *split._input_temp = tensor_alloc(1, input_z, input_y, input_x, context);

    return split;
}
void split_free(split_t *split) {
    assert(split);
    tensor_free(split->biases);
    tensor_free(split->biases_g);
    tensor_free(split->weights);
    tensor_free(split->weights_g);
    tensor_free(split->_input_temp);
    free(split->biases);
    free(split->biases_g);
    free(split->weights);
    free(split->weights_g);
    free(split->_input_temp);
}
void split_forward(tensor_t *input, split_t *split, tensor_t *output) {
    assert(input);
    assert(split);
    assert(output);
    int64_t input_z = input->buffer->inh_z;
    int64_t input_y = input->buffer->inh_y;
    int64_t input_x = input->buffer->inh_x;
    int64_t output_z = output->buffer->inh_z;
    int64_t output_y = output->buffer->inh_y;
    int64_t output_x = output->buffer->inh_x;

    tensor_move_offset(input, 0, 0, 0, 0);
    tensor_move_resize(output, 1, input_z, output_y, output_x);
    tensor_move_resize(split->weights, 1, input_z, output_y, output_x);
    tensor_move_resize(split->biases, 1, input_z, output_y, output_x);
    for(int64_t filter = 0; filter < split->filters; filter++) {
        tensor_move_offset(output, 0, filter * input_z, 0, 0);
        tensor_move_offset(split->weights, filter, 0, 0, 0);
        tensor_move_offset(split->biases, filter, 0, 0, 0);
        tensor_binary_copy(output, input);
        tensor_binary_multiply(output, split->weights);
        tensor_binary_add(output, split->biases);
    }

    tensor_move_resize(input, 1, input_z, input_y, input_x);
    tensor_move_offset(input, 0, 0, 0, 0);
    tensor_move_resize(output, 1, output_z, output_y, output_x);
    tensor_move_offset(output, 0, 0, 0, 0);
    tensor_move_resize(split->weights, split->filters, input_z, output_y, output_x);
    tensor_move_offset(split->weights, 0, 0, 0, 0);
    tensor_move_resize(split->biases, split->filters, input_z, output_y, output_x);
    tensor_move_offset(split->biases, 0, 0, 0, 0);
}
void split_backward(tensor_t *input, tensor_t *input_gradient, split_t *split, tensor_t *output,
                    tensor_t *output_gradient) {
    assert(input);
    assert(input_gradient);
    assert(split);
    assert(output);
    assert(output_gradient);
    int64_t input_z = input->buffer->inh_z;
    int64_t input_y = input->buffer->inh_y;
    int64_t input_x = input->buffer->inh_x;
    int64_t output_z = output->buffer->inh_z;
    int64_t output_y = output->buffer->inh_y;
    int64_t output_x = output->buffer->inh_x;
    /* Biases */
    tensor_move_reshape(split->biases_g, 1, output_z, output_y, output_x);
    tensor_binary_add(split->biases_g, output_gradient);
    tensor_move_reshape(split->biases_g, split->filters, input_z, output_y, output_x);
    /* Weights */
    tensor_move_resize(split->weights_g, 1, input_z, output_y, output_x);
    tensor_move_resize(output_gradient, 1, input_z, output_y, output_x);
    for(int64_t i = 0; i < split->filters; i++) {
        tensor_move_offset(split->weights_g, i, 0, 0, 0);
        tensor_move_offset(output_gradient, 0, i * input_z, 0, 0);
        tensor_binary_copy(split->_input_temp, output_gradient);
        tensor_binary_multiply(split->_input_temp, input);
        tensor_binary_add(split->weights_g, split->_input_temp);
    }
    tensor_move_resize(split->weights_g, split->filters, input_z, output_y, output_x);
    tensor_move_offset(split->weights_g, 0, 0, 0, 0);
    tensor_move_resize(output_gradient, 1, output_z, output_y, output_x);
    tensor_move_offset(output_gradient, 0, 0, 0, 0);
    /* Previous activation grad */
    tensor_move_resize(output_gradient, 1, input_z, input_y, input_x);
    tensor_move_resize(split->weights, 1, input_z, input_y, input_x);
    for(int64_t i = 0; i < split->filters; i++) {
        tensor_move_offset(split->weights, i, 0, 0, 0);
        tensor_move_offset(output_gradient, 0, i * input_z, 0, 0);
        tensor_binary_copy(split->_input_temp, output_gradient);
        tensor_binary_multiply(split->_input_temp, split->weights);
        tensor_binary_add(input_gradient, split->_input_temp);
    }
    tensor_move_resize(split->weights, split->filters, input_z, input_y, input_x);
    tensor_move_offset(split->weights, 0, 0, 0, 0);
    tensor_move_resize(output_gradient, 1, output_z, output_y, output_x);
    tensor_move_offset(output_gradient, 0, 0, 0, 0);
}
void split_print(split_t *split, int padding, int offset, const char *name) {
    assert(split);
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
    assert(split);
    if(strncmp(name, "", 1)) {
        printf("%*s%s split shape\n", offset, "", name);
    } else {
        printf("%*ssplit shape\n", offset, "");
    }
    printf("%*sBiases  {%lu, %lu, %lu, %lu}\n", offset + padding, "", split->biases->buffer->sze_a,
           split->biases->buffer->sze_z, split->biases->buffer->sze_y, split->biases->buffer->sze_x);
    printf("%*sWeights {%lu, %lu, %lu, %lu}\n", offset + padding, "", split->weights->buffer->sze_a,
           split->weights->buffer->sze_z, split->weights->buffer->sze_y, split->weights->buffer->sze_x);
}

/* TODO: Implement residual connections. */
layer_t layer_alloc(layerconfig_t *layerconfig, cl_context context) {
    assert(layerconfig);
    layer_t layer = {0};
    switch(layerconfig->layer_type) {
        case layer_input: {
            layer.layer_type = layer_input;
            layer.activation = calloc(1, sizeof(tensor_t));
            layer.activation_g = calloc(1, sizeof(tensor_t));
            assert(layer.activation);
            assert(layer.activation_g);
            *layer.activation =
                tensor_alloc(1, layerconfig->input_z, layerconfig->input_y, layerconfig->input_x, context);
            *layer.activation_g =
                tensor_alloc(1, layerconfig->input_z, layerconfig->input_y, layerconfig->input_x, context);
            break;
        }
        case layer_dense: {
            layer.layer_type = layer_dense;
            layer.activation = calloc(1, sizeof(tensor_t));
            layer.activation_g = calloc(1, sizeof(tensor_t));
            layer.activation_function = calloc(1, sizeof(activation_t));
            layer.norm = calloc(1, sizeof(norm_t));
            layer.dense = calloc(1, sizeof(dense_t));
            assert(layer.activation);
            assert(layer.activation_g);
            assert(layer.activation_function);
            assert(layer.norm);
            assert(layer.dense);
            *layer.activation = tensor_alloc(1, 1, 1, layerconfig->dense_output_size, context);
            *layer.activation_g = tensor_alloc(1, 1, 1, layerconfig->dense_output_size, context);
            *layer.activation_function = _activation_alloc(
                layerconfig->activation_function, 1, 1, 1,
                layerconfig->_dense_input_z * layerconfig->_dense_input_y * layerconfig->_dense_input_x, context);
            *layer.norm = _norm_alloc(layerconfig->norm_type, layer.activation, context);
            *layer.dense =
                dense_alloc(layerconfig->_dense_input_z * layerconfig->_dense_input_y * layerconfig->_dense_input_x,
                            layerconfig->dense_output_size, context);
            break;
        }
        case layer_convolution: {
            layer.layer_type = layer_convolution;
            int64_t new_size_y = CONVOLUTION_OUTPUT_SIZE(
                layerconfig->_convolution_input_y, layerconfig->convolution_kernel_size,
                layerconfig->convolution_kernel_stride, layerconfig->convolution_kernel_padding);
            int64_t new_size_x = CONVOLUTION_OUTPUT_SIZE(
                layerconfig->_convolution_input_x, layerconfig->convolution_kernel_size,
                layerconfig->convolution_kernel_stride, layerconfig->convolution_kernel_padding);
            layer.activation = calloc(1, sizeof(tensor_t));
            layer.activation_g = calloc(1, sizeof(tensor_t));
            layer.activation_function = calloc(1, sizeof(activation_t));
            layer.norm = calloc(1, sizeof(norm_t));
            layer.convolution = calloc(1, sizeof(convolution_t));
            assert(layer.activation);
            assert(layer.activation_g);
            assert(layer.activation_function);
            assert(layer.norm);
            assert(layer.convolution);
            *layer.activation = tensor_alloc(1, layerconfig->convolution_filters, new_size_y, new_size_x, context);
            *layer.activation_g = tensor_alloc(1, layerconfig->convolution_filters, new_size_y, new_size_x, context);
            *layer.activation_function =
                _activation_alloc(layerconfig->activation_function, 1, layerconfig->_convolution_input_z,
                                  layerconfig->_convolution_input_y, layerconfig->_convolution_input_x, context);
            *layer.norm = _norm_alloc(layerconfig->norm_type, layer.activation, context);
            *layer.convolution = convolution_alloc(
                layerconfig->_convolution_input_z, layerconfig->_convolution_input_y, layerconfig->_convolution_input_x,
                layerconfig->convolution_filters, layerconfig->convolution_kernel_size,
                layerconfig->convolution_kernel_stride, layerconfig->convolution_kernel_padding, context);
            break;
        }
        case layer_reduce: {
            layer.layer_type = layer_reduce;
            int64_t new_size_y = REDUCE_OUTPUT_SIZE(layerconfig->_reduce_input_y, layerconfig->reduce_kernel_size,
                                                    layerconfig->reduce_kernel_stride);
            int64_t new_size_x = REDUCE_OUTPUT_SIZE(layerconfig->_reduce_input_x, layerconfig->reduce_kernel_size,
                                                    layerconfig->reduce_kernel_stride);
            layer.activation = calloc(1, sizeof(tensor_t));
            layer.activation_g = calloc(1, sizeof(tensor_t));
            layer.reduce = calloc(1, sizeof(reduce_t));
            assert(layer.activation);
            assert(layer.activation_g);
            assert(layerconfig->norm_type == norm_none);
            assert(layerconfig->activation_function == activation_none);
            assert(layer.reduce);
            *layer.activation = tensor_alloc(1, layerconfig->_reduce_input_z, new_size_y, new_size_x, context);
            *layer.activation_g = tensor_alloc(1, layerconfig->_reduce_input_z, new_size_y, new_size_x, context);
            *layer.reduce = reduce_alloc(layerconfig->reduce_type, layerconfig->_reduce_input_z,
                                         layerconfig->_reduce_input_y, layerconfig->_reduce_input_x,
                                         layerconfig->reduce_kernel_size, layerconfig->reduce_kernel_stride);
            break;
        }
        case layer_split: {
            layer.layer_type = layer_split;
            layer.activation = calloc(1, sizeof(tensor_t));
            layer.activation_g = calloc(1, sizeof(tensor_t));
            layer.activation_function = calloc(1, sizeof(activation_t));
            layer.norm = calloc(1, sizeof(norm_t));
            layer.split = calloc(1, sizeof(split_t));
            assert(layer.activation);
            assert(layer.activation_g);
            assert(layer.activation_function);
            assert(layer.norm);
            assert(layer.split);
            *layer.activation = tensor_alloc(1, layerconfig->split_filters * layerconfig->_split_input_z,
                                             layerconfig->_split_input_y, layerconfig->_split_input_x, context);
            *layer.activation_g = tensor_alloc(1, layerconfig->split_filters * layerconfig->_split_input_z,
                                               layerconfig->_split_input_y, layerconfig->_split_input_x, context);
            *layer.activation_function =
                _activation_alloc(layerconfig->activation_function, 1, layerconfig->_split_input_z,
                                  layerconfig->_split_input_y, layerconfig->_split_input_x, context);
            *layer.norm = _norm_alloc(layerconfig->norm_type, layer.activation, context);
            *layer.split = split_alloc(layerconfig->split_filters, layerconfig->_split_input_z,
                                       layerconfig->_split_input_y, layerconfig->_split_input_x, context);
            break;
        }
    }
    return layer;
}
void layer_free(layer_t *layer) {
    assert(layer);
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
            _activation_free(layer->activation_function);
            free(layer->activation_function);
            _norm_free(layer->norm);
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
            _activation_free(layer->activation_function);
            free(layer->activation_function);
            _norm_free(layer->norm);
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
            _activation_free(layer->activation_function);
            free(layer->activation_function);
            _norm_free(layer->norm);
            free(layer->norm);
            break;
        }
    }
}
void layer_sync(layer_t *layer, cl_command_queue command_queue) {
    switch(layer->layer_type) {
        case layer_dense: {
            buffer_sync_realize(layer->activation->buffer, command_queue);
            buffer_sync_realize(layer->dense->biases->buffer, command_queue);
            buffer_sync_realize(layer->dense->weights->buffer, command_queue);
            buffer_sync_realize(layer->activation_g->buffer, command_queue);
            buffer_sync_realize(layer->dense->biases_g->buffer, command_queue);
            buffer_sync_realize(layer->dense->weights_g->buffer, command_queue);
            break;
        }
        case layer_convolution: {
            buffer_sync_realize(layer->activation->buffer, command_queue);
            buffer_sync_realize(layer->convolution->biases->buffer, command_queue);
            buffer_sync_realize(layer->convolution->weights->buffer, command_queue);
            buffer_sync_realize(layer->activation_g->buffer, command_queue);
            buffer_sync_realize(layer->convolution->biases_g->buffer, command_queue);
            buffer_sync_realize(layer->convolution->weights_g->buffer, command_queue);
            break;
        }
        case layer_reduce: {
            buffer_sync_realize(layer->activation->buffer, command_queue);
            buffer_sync_realize(layer->activation_g->buffer, command_queue);
            break;
        }
        case layer_split: {
            buffer_sync_realize(layer->activation->buffer, command_queue);
            buffer_sync_realize(layer->split->biases->buffer, command_queue);
            buffer_sync_realize(layer->split->weights->buffer, command_queue);
            buffer_sync_realize(layer->activation_g->buffer, command_queue);
            buffer_sync_realize(layer->split->biases_g->buffer, command_queue);
            buffer_sync_realize(layer->split->weights_g->buffer, command_queue);
            break;
        }
        case layer_input: {
            buffer_sync_realize(layer->activation->buffer, command_queue);
            buffer_sync_realize(layer->activation_g->buffer, command_queue);
            break;
        }
    }
}

/* TODO: Make learning a parameter in `neuralnet_learn()` and not here. For this `learning` needs to be wrapped in a
 * tensor. */
neuralnet_t neuralnet_alloc(int64_t layers, layerconfig_t *layerconfig, double learning, enum compile_e compile_type) {
    assert(layers > 1);
    assert(learning > 0);
    neuralnet_t neuralnet = {
        .compile_type = compile_type,
        .layers = layers,
        .layer = calloc(layers, sizeof(layer_t)),
        .forward = calloc(1, sizeof(linearized_t)),
        .backward = calloc(1, sizeof(linearized_t)),
        .learn = calloc(1, sizeof(linearized_t)),
    };
    assert(neuralnet.layer);
    assert(neuralnet.forward);
    assert(neuralnet.backward);
    assert(neuralnet.learn);
    *neuralnet.forward = linearized_alloc();
    *neuralnet.backward = linearized_alloc();
    *neuralnet.learn = linearized_alloc();

    cl_device_id *device_id = calloc(1, sizeof(cl_device_id));
    assert(device_id);
    cl_context *context = calloc(1, sizeof(cl_context));
    assert(context);
    cl_command_queue *command_queue = calloc(1, sizeof(cl_command_queue));
    assert(command_queue);
    switch(compile_type) {
        case compile_cl: {
            int err;
            *device_id = cl_device_get();
            *context = clCreateContext(NULL, 1, device_id, NULL, NULL, &err);
            assert(err == 0);
            *command_queue = clCreateCommandQueueWithProperties(*context, *device_id, NULL, &err);
            assert(err == 0);
            break;
        }
        case compile_none: {
            *context = NULL;
            break;
        }
    }

    int64_t previous_z;
    int64_t previous_y;
    int64_t previous_x;
    assert(layerconfig[0].layer_type == layer_input);
    neuralnet.layer[0] = layer_alloc(&layerconfig[0], *context);
    for(int64_t layer = 1; layer < layers; layer++) {
        previous_z = neuralnet.layer[layer - 1].activation->buffer->sze_z;
        previous_y = neuralnet.layer[layer - 1].activation->buffer->sze_y;
        previous_x = neuralnet.layer[layer - 1].activation->buffer->sze_x;
        switch(layerconfig[layer].layer_type) {
            case layer_dense: {
                layerconfig[layer]._dense_input_z = previous_z;
                layerconfig[layer]._dense_input_y = previous_y;
                layerconfig[layer]._dense_input_x = previous_x;
                neuralnet.layer[layer] = layer_alloc(&layerconfig[layer], *context);
                break;
            }
            case layer_convolution: {
                layerconfig[layer]._convolution_input_z = previous_z;
                layerconfig[layer]._convolution_input_y = previous_y;
                layerconfig[layer]._convolution_input_x = previous_x;
                neuralnet.layer[layer] = layer_alloc(&layerconfig[layer], *context);
                break;
            }
            case layer_reduce: {
                layerconfig[layer]._reduce_input_z = previous_z;
                layerconfig[layer]._reduce_input_y = previous_y;
                layerconfig[layer]._reduce_input_x = previous_x;
                neuralnet.layer[layer] = layer_alloc(&layerconfig[layer], *context);
                break;
            }
            case layer_split: {
                layerconfig[layer]._split_input_z = previous_z;
                layerconfig[layer]._split_input_y = previous_y;
                layerconfig[layer]._split_input_x = previous_x;
                neuralnet.layer[layer] = layer_alloc(&layerconfig[layer], *context);
                break;
            }
            case layer_input: {
                ERROR("Tried to allocate input layer at a layer with index %lu\n", layer);
            }
        }
    }

    for(int64_t layer = 1; layer < neuralnet.layers; layer++) {
        switch(neuralnet.layer[layer].layer_type) {
            case layer_dense: {
                dense_forward(neuralnet.layer[layer - 1].activation, neuralnet.layer[layer].dense,
                              neuralnet.layer[layer].activation);
                _activation_activate(neuralnet.layer[layer].activation, neuralnet.layer[layer].activation_function);
                _norm_apply(neuralnet.layer[layer].norm, neuralnet.layer[layer].activation);
                break;
            }
            case layer_convolution: {
                convolution_forward(neuralnet.layer[layer - 1].activation, neuralnet.layer[layer].convolution,
                                    neuralnet.layer[layer].activation);
                _activation_activate(neuralnet.layer[layer].activation, neuralnet.layer[layer].activation_function);
                _norm_apply(neuralnet.layer[layer].norm, neuralnet.layer[layer].activation);
                break;
            }
            case layer_reduce: {
                reduce_forward(neuralnet.layer[layer - 1].activation, neuralnet.layer[layer].reduce,
                               neuralnet.layer[layer].activation);
                break;
            }
            case layer_split: {
                split_forward(neuralnet.layer[layer - 1].activation, neuralnet.layer[layer].split,
                              neuralnet.layer[layer].activation);
                _activation_activate(neuralnet.layer[layer].activation, neuralnet.layer[layer].activation_function);
                _norm_apply(neuralnet.layer[layer].norm, neuralnet.layer[layer].activation);
                break;
            }
            case layer_input: {
                ERROR("Input layer at layer %lu\n", layer);
            }
        }
    }
    /* Has to be done like this to ensure that each activation tensor gets resized back to it's needed shape. */
    for(int64_t layer = neuralnet.layers - 1; layer >= 0; layer--) {
        linearized_from_op(neuralnet.forward, neuralnet.layer[layer].activation->op);
    }
    for(int64_t layer = 1; layer < neuralnet.layers; layer++) {
        switch(neuralnet.layer[layer].layer_type) {
            case layer_dense: {
                tensor_unary_set(neuralnet.layer[layer - 1].activation_g, 0);
                dense_backward(neuralnet.layer[layer - 1].activation, neuralnet.layer[layer - 1].activation_g,
                               neuralnet.layer[layer].dense, neuralnet.layer[layer].activation_g);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer - 1].activation_g->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].dense->weights_g->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].dense->weights->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].dense->biases_g->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].dense->biases->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer - 1].activation->op);
                break;
            }
            case layer_convolution: {
                /* This is `_padded_grad` here, because gradients get calculate in there and then copied into
                 * `activation_g`. */
                tensor_unary_set(neuralnet.layer[layer].convolution->_padded_grad, 0);
                convolution_backward(neuralnet.layer[layer - 1].activation, neuralnet.layer[layer - 1].activation_g,
                                     neuralnet.layer[layer].convolution, neuralnet.layer[layer].activation,
                                     neuralnet.layer[layer].activation_g);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].convolution->biases_g->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].convolution->biases->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].convolution->weights_g->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer - 1].activation_g->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer - 1].activation->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].convolution->weights->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].convolution->_padded_grad->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].convolution->_padded_input->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].activation_g->op);
                break;
            }
            case layer_reduce: {
                tensor_unary_set(neuralnet.layer[layer - 1].activation_g, 0);
                reduce_backward(neuralnet.layer[layer - 1].activation_g, neuralnet.layer[layer].reduce,
                                neuralnet.layer[layer].activation_g);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer - 1].activation_g->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer - 1].activation->op);
                linearized_from_op(neuralnet.backward, neuralnet.layer[layer].activation_g->op);
                break;
            }
            case layer_split: {
                tensor_unary_set(neuralnet.layer[layer - 1].activation_g, 0);
                split_backward(neuralnet.layer[layer - 1].activation, neuralnet.layer[layer - 1].activation_g,
                               neuralnet.layer[layer].split, neuralnet.layer[layer].activation,
                               neuralnet.layer[layer].activation_g);
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
                ERROR("Input layer at layer %lu\n", layer);
            }
        }
        for(int64_t layer = 1; layer < neuralnet.layers; layer++) {
            switch(neuralnet.layer[layer].layer_type) {
                case layer_dense: {
                    tensor_unary_multiply(neuralnet.layer[layer].dense->weights_g, learning);
                    tensor_binary_subtract(neuralnet.layer[layer].dense->weights,
                                           neuralnet.layer[layer].dense->weights_g);
                    tensor_unary_set(neuralnet.layer[layer].dense->weights_g, 0);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].dense->weights->op);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].dense->weights_g->op);
                    tensor_unary_multiply(neuralnet.layer[layer].dense->biases_g, learning);
                    tensor_binary_subtract(neuralnet.layer[layer].dense->biases,
                                           neuralnet.layer[layer].dense->biases_g);
                    tensor_unary_set(neuralnet.layer[layer].dense->biases_g, 0);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].dense->biases->op);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].dense->biases_g->op);
                    break;
                }
                case layer_convolution: {
                    tensor_unary_multiply(neuralnet.layer[layer].convolution->weights_g, learning);
                    tensor_binary_subtract(neuralnet.layer[layer].convolution->weights,
                                           neuralnet.layer[layer].convolution->weights_g);
                    tensor_unary_set(neuralnet.layer[layer].convolution->weights_g, 0);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].convolution->weights->op);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].convolution->weights_g->op);
                    tensor_unary_multiply(neuralnet.layer[layer].convolution->biases_g, learning);
                    tensor_binary_subtract(neuralnet.layer[layer].convolution->biases,
                                           neuralnet.layer[layer].convolution->biases_g);
                    tensor_unary_set(neuralnet.layer[layer].convolution->biases_g, 0);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].convolution->biases->op);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].convolution->biases_g->op);
                    break;
                }
                case layer_reduce: {
                    /* Nothing to update. */
                    break;
                }
                case layer_split: {
                    tensor_unary_multiply(neuralnet.layer[layer].split->biases_g, learning);
                    tensor_binary_subtract(neuralnet.layer[layer].split->biases,
                                           neuralnet.layer[layer].split->biases_g);
                    tensor_unary_set(neuralnet.layer[layer].split->biases_g, 0);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].split->biases->op);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].split->biases_g->op);
                    tensor_unary_multiply(neuralnet.layer[layer].split->weights_g, learning);
                    tensor_binary_subtract(neuralnet.layer[layer].split->weights,
                                           neuralnet.layer[layer].split->weights_g);
                    tensor_unary_set(neuralnet.layer[layer].split->weights_g, 0);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].split->weights->op);
                    linearized_from_op(neuralnet.learn, neuralnet.layer[layer].split->weights_g->op);
                    break;
                }
                case layer_input: {
                    ERROR("Input layer at layer %lu\n", layer);
                }
            }
        }
    }
    if(neuralnet.compile_type == compile_cl) {
        program_compile(&neuralnet.forward_cl, neuralnet.forward, device_id, context, command_queue);
        program_compile(&neuralnet.backward_cl, neuralnet.backward, device_id, context, command_queue);
        program_compile(&neuralnet.learn_cl, neuralnet.learn, device_id, context, command_queue);
    } else {
        free(device_id);
        free(context);
        free(command_queue);
    }

    return neuralnet;
}
void neuralnet_free(neuralnet_t *neuralnet) {
    assert(neuralnet);
    assert(neuralnet->layer);
    assert(neuralnet->forward);
    assert(neuralnet->backward);
    assert(neuralnet->learn);
    for(int64_t i = 0; i < neuralnet->layers; i++) { layer_free(&neuralnet->layer[i]); }
    free(neuralnet->layer);
    linearized_free(neuralnet->forward);
    free(neuralnet->forward);
    linearized_free(neuralnet->backward);
    free(neuralnet->backward);
    linearized_free(neuralnet->learn);
    free(neuralnet->learn);
    if(neuralnet->compile_type == compile_cl) {
        cl_device_id *device_id_temp = neuralnet->forward_cl.cl_device_id;
        cl_context *context_temp = neuralnet->forward_cl.cl_context;
        cl_command_queue *command_queue_temp = neuralnet->forward_cl.cl_command_queue;
        program_free(&neuralnet->forward_cl);
        program_free(&neuralnet->backward_cl);
        program_free(&neuralnet->learn_cl);
        free(device_id_temp);
        free(context_temp);
        free(command_queue_temp);
    }
}
/* TODO: Make this save the neuralnet structure and not only the weights and biases. */
void neuralnet_save(neuralnet_t *neuralnet, const char *filename) {
    assert(neuralnet);
    assert(filename);
    int err;
    FILE *file = fopen(filename, "wb");
    assert(file);
    for(int64_t layer = 1; layer < neuralnet->layers; layer++) {
        err = 0;
        switch(neuralnet->layer[layer].layer_type) {
            case layer_dense: {
                int64_t bias_size = neuralnet->layer[layer].dense->output_size;
                int64_t weight_size =
                    neuralnet->layer[layer].dense->output_size * neuralnet->layer[layer].dense->_input_size;
                err |= fwrite(neuralnet->layer[layer].dense->biases->buffer->val, sizeof(double), bias_size, file);
                err |= fwrite(neuralnet->layer[layer].dense->weights->buffer->val, sizeof(double), weight_size, file);
                assert(err == 0);
                break;
            }
            case layer_convolution: {
                int64_t bias_size = neuralnet->layer[layer].convolution->filters;
                int64_t weight_size =
                    neuralnet->layer[layer].convolution->filters * neuralnet->layer[layer].convolution->_input_z *
                    neuralnet->layer[layer].convolution->kernel_size * neuralnet->layer[layer].convolution->kernel_size;
                err |=
                    fwrite(neuralnet->layer[layer].convolution->biases->buffer->val, sizeof(double), bias_size, file);
                err |= fwrite(neuralnet->layer[layer].convolution->weights->buffer->val, sizeof(double), weight_size,
                              file);
                assert(err == 0);
                break;
            }
            case layer_reduce: {
                /* Nothing to initialize. */
                break;
            }
            case layer_split: {
                int64_t bias_size = neuralnet->layer[layer].split->filters * neuralnet->layer[layer].split->input_z *
                                    neuralnet->layer[layer].split->input_y * neuralnet->layer[layer].split->input_x;
                int64_t weight_size = bias_size;
                err |= fwrite(neuralnet->layer[layer].dense->biases->buffer->val, sizeof(double), bias_size, file);
                err |= fwrite(neuralnet->layer[layer].dense->weights->buffer->val, sizeof(double), weight_size, file);
                assert(err == 0);
                break;
            }
            case layer_input: {
                ERROR("Input layer at layer %lu. I don't even know how this can possibly happen.\n", layer);
            }
        }
    }
    err = fclose(file);
    assert(err == 0);
}
void neuralnet_load(neuralnet_t *neuralnet, const char *filename) {
    assert(neuralnet);
    assert(filename);
    int err;
    FILE *file = fopen(filename, "rb");
    assert(file);
    for(int64_t layer = 1; layer < neuralnet->layers; layer++) {
        err = 0;
        switch(neuralnet->layer[layer].layer_type) {
            case layer_dense: {
                int64_t bias_size = neuralnet->layer[layer].dense->output_size;
                int64_t weight_size =
                    neuralnet->layer[layer].dense->output_size * neuralnet->layer[layer].dense->_input_size;
                err |= fread(neuralnet->layer[layer].dense->biases->buffer->val, sizeof(double), bias_size, file);
                err |= fread(neuralnet->layer[layer].dense->weights->buffer->val, sizeof(double), weight_size, file);
                assert(err == 0);
                break;
            }
            case layer_convolution: {
                int64_t bias_size = neuralnet->layer[layer].convolution->filters;
                int64_t weight_size =
                    neuralnet->layer[layer].convolution->filters * neuralnet->layer[layer].convolution->_input_z *
                    neuralnet->layer[layer].convolution->kernel_size * neuralnet->layer[layer].convolution->kernel_size;
                err |= fread(neuralnet->layer[layer].convolution->biases->buffer->val, sizeof(double), bias_size, file);
                err |=
                    fread(neuralnet->layer[layer].convolution->weights->buffer->val, sizeof(double), weight_size, file);
                assert(err == 0);
                break;
            }
            case layer_reduce: {
                /* Nothing to initialize. */
                break;
            }
            case layer_split: {
                int64_t bias_size = neuralnet->layer[layer].split->filters * neuralnet->layer[layer].split->input_z *
                                    neuralnet->layer[layer].split->input_y * neuralnet->layer[layer].split->input_x;
                int64_t weight_size = bias_size;
                err |= fread(neuralnet->layer[layer].dense->biases->buffer->val, sizeof(double), bias_size, file);
                assert(err == 0);
                err |= fread(neuralnet->layer[layer].dense->weights->buffer->val, sizeof(double), weight_size, file);
                assert(err == 0);
                break;
            }
            case layer_input: {
                ERROR("Input layer at layer %lu. I don't even know how this can possibly happen.\n", layer);
            }
        }
    }
    err = fclose(file);
    assert(err == 0);
}
void neuralnet_random(neuralnet_t *neuralnet) {
    assert(neuralnet);
    for(int64_t layer = 1; layer < neuralnet->layers; layer++) {
        switch(neuralnet->layer[layer].layer_type) {
            case layer_dense: {
                tensor_unary_random(neuralnet->layer[layer].dense->biases);
                tensor_unary_random(neuralnet->layer[layer].dense->weights);
                tensor_realize(neuralnet->layer[layer].dense->biases);
                tensor_realize(neuralnet->layer[layer].dense->weights);
                break;
            }
            case layer_convolution: {
                tensor_unary_random(neuralnet->layer[layer].convolution->biases);
                tensor_unary_random(neuralnet->layer[layer].convolution->weights);
                tensor_realize(neuralnet->layer[layer].convolution->biases);
                tensor_realize(neuralnet->layer[layer].convolution->weights);
                break;
            }
            case layer_reduce: {
                /* Nothing to initialize. */
                break;
            }
            case layer_split: {
                tensor_unary_random(neuralnet->layer[layer].split->biases);
                tensor_unary_random(neuralnet->layer[layer].split->weights);
                tensor_realize(neuralnet->layer[layer].split->biases);
                tensor_realize(neuralnet->layer[layer].split->weights);
                break;
            }
            case layer_input: {
                ERROR("Input layer at layer %lu. I don't even know how this can possibly happen.\n", layer);
            }
        }
    }
}
void neuralnet_forward(neuralnet_t *neuralnet, tensor_t *input) {
    assert(neuralnet);
    assert(input);
    assert(neuralnet->forward);
    tensor_binary_copy(NEURALNET_INPUT_(neuralnet).activation, input);
    tensor_realize(NEURALNET_INPUT_(neuralnet).activation);

    switch(neuralnet->compile_type) {
        case(compile_cl): {
            for(int64_t layer = 0; layer < neuralnet->layers; layer++) {
                layer_sync(&neuralnet->layer[layer], *neuralnet->forward_cl.cl_command_queue);
            }
            clFinish(*neuralnet->forward_cl.cl_command_queue);
            program_run(&neuralnet->forward_cl);
            clFinish(*neuralnet->forward_cl.cl_command_queue);
            buffer_sync_update(NEURALNET_OUTPUT_(neuralnet).activation->buffer, sync_to_host);
            buffer_sync_realize(NEURALNET_OUTPUT_(neuralnet).activation->buffer,
                                *neuralnet->forward_cl.cl_command_queue);
            clFinish(*neuralnet->forward_cl.cl_command_queue);
            break;
        }
        case(compile_none): {
            linearized_run(neuralnet->forward);
            break;
        }
    }
}
void neuralnet_backward(neuralnet_t *neuralnet, tensor_t *training_input, tensor_t *training_output) {
    assert(neuralnet);
    assert(training_input);
    assert(training_output);
    assert(training_input->buffer->sze_a == training_output->buffer->sze_a);
    assert(neuralnet->backward);
    int64_t training_samples = training_input->buffer->sze_a;
    int64_t input_z = training_input->buffer->sze_z;
    int64_t input_y = training_input->buffer->sze_y;
    int64_t input_x = training_input->buffer->sze_x;
    int64_t output_z = training_output->buffer->sze_z;
    int64_t output_y = training_output->buffer->sze_y;
    int64_t output_x = training_output->buffer->sze_x;
    tensor_move_resize(training_input, 1, input_z, input_y, input_x);
    tensor_move_resize(training_output, 1, output_z, output_y, output_x);
    for(int64_t sample = 0; sample < training_samples; sample++) {
        tensor_move_offset(training_input, sample, 0, 0, 0);
        tensor_move_offset(training_output, sample, 0, 0, 0);
        neuralnet_forward(neuralnet, training_input);
        tensor_binary_copy(NEURALNET_OUTPUT_(neuralnet).activation_g, NEURALNET_OUTPUT_(neuralnet).activation);
        tensor_binary_subtract(NEURALNET_OUTPUT_(neuralnet).activation_g, training_output);
        // tensor_unary_multiply(NEURALNET_OUTPUT_(neuralnet).activation_g, 2);
        tensor_realize(NEURALNET_OUTPUT_(neuralnet).activation_g);
        switch(neuralnet->compile_type) {
            case(compile_cl): {
                program_run(&neuralnet->backward_cl);
                break;
            }
            case(compile_none): {
                linearized_run(neuralnet->backward);
                break;
            }
        }
    }
    tensor_move_resize(training_input, training_samples, input_z, input_y, input_x);
    tensor_move_offset(training_input, 0, 0, 0, 0);
    tensor_realize(training_input);
    tensor_move_resize(training_output, training_samples, output_z, output_y, output_x);
    tensor_move_offset(training_output, 0, 0, 0, 0);
    tensor_realize(training_output);
}
/* Have to call `neuralnet_backward()` before this one. This also clears the gradients. */
void neuralnet_learn(neuralnet_t *neuralnet) {
    assert(neuralnet);
    assert(neuralnet->learn);
    switch(neuralnet->compile_type) {
        case(compile_cl): {
            program_run(&neuralnet->learn_cl);
            break;
        }
        case(compile_none): {
            linearized_run(neuralnet->learn);
            break;
        }
    }
}
void neuralnet_print(neuralnet_t *neuralnet, int padding, int offset, const char *name) {
    assert(neuralnet);
    if(strncmp(name, "", 1)) {
        printf("%*s%s\n", offset, "", name);
    } else {
        printf("%*sneuralnet\n", offset, "");
    }
    for(int64_t layer = 0; layer < neuralnet->layers; layer++) {
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
    assert(neuralnet);
    if(strncmp(name, "", 1)) {
        printf("%*s%s shape\n", offset, "", name);
    } else {
        printf("%*sneuralnet shape\n", offset, "");
    }
    for(int64_t layer = 0; layer < neuralnet->layers; layer++) {
        switch(neuralnet->layer[layer].layer_type) {
            case layer_dense: {
                printf("%*slayer[%lu] dense\n", offset + padding, "", layer);
                dense_print_shape(neuralnet->layer[layer].dense, padding, offset + 2 * padding, "");
                printf("%*s{%lu, %lu, %lu, %lu} %lu\n", offset + 2 * padding, "",
                       neuralnet->layer[layer].activation->buffer->sze_a,
                       neuralnet->layer[layer].activation->buffer->sze_z,
                       neuralnet->layer[layer].activation->buffer->sze_y,
                       neuralnet->layer[layer].activation->buffer->sze_x,
                       neuralnet->layer[layer].activation->buffer->off);
                break;
            }
            case layer_convolution: {
                printf("%*slayer[%lu] convolution\n", offset + padding, "", layer);
                convolution_print_shape(neuralnet->layer[layer].convolution, padding, offset + 2 * padding, "");
                printf("%*s{%lu, %lu, %lu, %lu} %lu\n", offset + 2 * padding, "",
                       neuralnet->layer[layer].activation->buffer->sze_a,
                       neuralnet->layer[layer].activation->buffer->sze_z,
                       neuralnet->layer[layer].activation->buffer->sze_y,
                       neuralnet->layer[layer].activation->buffer->sze_x,
                       neuralnet->layer[layer].activation->buffer->off);
                break;
            }
            case layer_reduce: {
                printf("%*slayer[%lu] reduce\n", offset + padding, "", layer);
                // reduce_print_shape(neuralnet->layer[layer].reduce, padding, offset + 2 * padding, "");
                printf("%*s{%lu, %lu, %lu, %lu} %lu\n", offset + 2 * padding, "",
                       neuralnet->layer[layer].activation->buffer->sze_a,
                       neuralnet->layer[layer].activation->buffer->sze_z,
                       neuralnet->layer[layer].activation->buffer->sze_y,
                       neuralnet->layer[layer].activation->buffer->sze_x,
                       neuralnet->layer[layer].activation->buffer->off);
                break;
            }
            case layer_split: {
                printf("%*slayer[%lu] split\n", offset + padding, "", layer);
                split_print_shape(neuralnet->layer[layer].split, padding, offset + 2 * padding, "");
                printf("%*s{%lu, %lu, %lu, %lu} %lu\n", offset + 2 * padding, "",
                       neuralnet->layer[layer].activation->buffer->sze_a,
                       neuralnet->layer[layer].activation->buffer->sze_z,
                       neuralnet->layer[layer].activation->buffer->sze_y,
                       neuralnet->layer[layer].activation->buffer->sze_x,
                       neuralnet->layer[layer].activation->buffer->off);
                break;
            }
            case layer_input: {
                printf("%*slayer[%lu] input\n", offset + padding, "", layer);
                printf("%*s{%lu, %lu, %lu, %lu} %lu\n", offset + 2 * padding, "",
                       neuralnet->layer[layer].activation->buffer->sze_a,
                       neuralnet->layer[layer].activation->buffer->sze_z,
                       neuralnet->layer[layer].activation->buffer->sze_y,
                       neuralnet->layer[layer].activation->buffer->sze_x,
                       neuralnet->layer[layer].activation->buffer->off);
            }
        }
    }
}
