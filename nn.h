#ifndef NN_H_
#define NN_H_

#include <stdint.h>

#include "tensor.h"
#include "linearize.h"

enum activation_e {
    activation_identity, activation_relu, activation_sigmoid, activation_tanh, activation_silu, activation_gelu, activation_leaky
};

typedef struct {
    enum activation_e type;
    tensor_t *intermediary;
} activation_t;

extern activation_t activation_alloc(enum activation_e activation_type, uint64_t a, uint64_t z, uint64_t y, uint64_t x);            
extern void activation_free(activation_t *activation_type);
extern void activation_activate(tensor_t *tensor, activation_t *activation_type);
extern void activation_derivative(tensor_t *tensor, activation_t *activation_type);

/* TODO: Store expected value and standard deviation in a norm_t struct */
enum norm_e {
    norm_none, norm_layer, norm_batch, norm_simple
};

typedef struct {
    uint64_t input_size; /* Not set directly. */
    uint64_t output_size;
    
    tensor_t *weights;
    tensor_t *weights_g;
    tensor_t *biases;
    tensor_t *biases_g;

    tensor_t *input_multiply_temp;
    tensor_t *output_multiply_temp;
} dense_t;

extern dense_t dense_alloc(uint64_t input_size, uint64_t output_size);
extern void dense_free(dense_t *dense);
extern void dense_forward(tensor_t *input, dense_t *dense, tensor_t *output);
/* NOTE: Names are flipped here due to it being *backward* propagation. */
extern void dense_backward(tensor_t *output, dense_t *dense, tensor_t *input);
extern void dense_print(dense_t *dense, int padding, int offset, const char *name);
extern void dense_print_shape(dense_t *dense, int padding, int offset, const char *name);

typedef struct {
    uint64_t input_channels; /* Equivalent to input_z. Also not set directly. */
    uint64_t input_y; /* Not set directly. */
    uint64_t input_x; /* Not set directly. */
    uint64_t filters;
    uint64_t kernel_size;
    uint64_t kernel_stride;
    uint64_t kernel_padding;
    
    tensor_t *weights;
    tensor_t *weights_g;
    tensor_t *biases;
    tensor_t *biases_g;

    tensor_t *padded_input;
    tensor_t *kernel_temp;
} convolution_t;

/* Calculates output size per dimension. */
#define CONVOLUTION_OUTPUT_SIZE(input_size, kernel_size, kernel_stride, kernel_padding) ((((input_size) + 2 * (kernel_padding) - (kernel_size)) / (kernel_stride)) + 1)
extern convolution_t convolution_alloc(uint64_t input_channels, uint64_t input_y, uint64_t input_x, uint64_t filters, uint64_t kernel_size, uint64_t kernel_stride, uint64_t kernel_padding);
extern void convolution_free(convolution_t *convolution);
extern void convolution_forward(tensor_t *input, convolution_t *convolution, tensor_t *output);
/* NOTE: Names are flipped here due to it being *backward* propagation. */
extern void convolution_backward(tensor_t *output, convolution_t *convolution, tensor_t *input);
extern void convolution_print(convolution_t *convolution, int padding, int offset, const char *name);
extern void convolution_print_shape(convolution_t *convolution, int padding, int offset, const char *name);

/* Can rename this to reduce_e after unifying all the op types. */
enum layer_reduce_e {
    layer_reduce_max, layer_reduce_avg, layer_reduce_min
};

typedef struct {
    enum layer_reduce_e type;
    uint64_t input_channels; /* Equivalent to input_z. */
    uint64_t input_y; /* Not set directly. */
    uint64_t input_x; /* Not set directly. */
    uint64_t kernel_size;
    uint64_t kernel_stride;
    //uint64_t kernel_padding;
} reduce_t;

/* Calculates output size per dimension. */
#define REDUCE_OUTPUT_SIZE(input_size, kernel_size, kernel_stride) ((((input_size) - (kernel_size)) / (kernel_stride)) + 1)
extern reduce_t reduce_alloc(enum layer_reduce_e type, uint64_t input_channels, uint64_t input_y, uint64_t input_x, uint64_t kernel_size, uint64_t kernel_stride);
extern void reduce_forward(tensor_t *input, reduce_t *reduce, tensor_t *output);
/* NOTE: Names are flipped here due to it being *backward* propagation. */
extern void reduce_backward(tensor_t *output, reduce_t *reduce, tensor_t *input);
extern void reduce_print(reduce_t *reduce, int padding, int offset, const char *name);

/* Trying some new types of residual connections beyond identity and conv. */
enum residual_e {
    residual_identity, residual_convolution, residual_dense, residual_reduce
};

/* NOTE: Specifies a residual connection and not a residual block per se. Also only identity and convolutions are supported right now. */
/* TODO: Think about specifying the amount of layers you need to go back for the residual connection. This way you can reuse the same layerconf and it is more similar to the standard residual block. */
typedef struct {
    enum residual_e type;
    uint64_t connection_from_layer;
    convolution_t *convolution; /* Optional. */
} residual_t;

typedef struct {
    uint64_t filters;
    uint64_t input_channels;
    uint64_t input_y;
    uint64_t input_x;

    tensor_t *weights;
    tensor_t *weights_g;
    tensor_t *biases;
    tensor_t *biases_g;
} split_t;

extern split_t split_alloc(uint64_t filters, uint64_t input_channels, uint64_t input_y, uint64_t input_x);
extern void split_free(split_t *split);
extern void split_forward(tensor_t *input, split_t *split, tensor_t *output);
/* NOTE: Names are flipped here due to it being *backward* propagation. */
extern void split_backward(tensor_t *output, split_t *split, tensor_t *input);
extern void split_print(split_t *split, int padding, int offset, const char *name);
extern void split_print_shape(split_t *split, int padding, int offset, const char *name);

enum layer_e {
    layer_input, layer_dense, layer_convolution, layer_reduce, layer_split
};

typedef struct {
    enum layer_e layer_type;

    uint64_t input_channels;
    uint64_t input_y;
    uint64_t input_x;

    uint64_t dense_input_channels; /* Not set directly */
    uint64_t dense_input_y; /* Not set directly */
    uint64_t dense_input_x; /* Not set directly */
    uint64_t dense_output_size;

    uint64_t convolution_input_channels; /* Not set directly */
    uint64_t convolution_input_y; /* Not set directly */
    uint64_t convolution_input_x; /* Not set directly */
    uint64_t convolution_filters;
    uint64_t convolution_kernel_size;
    uint64_t convolution_kernel_stride;
    uint64_t convolution_kernel_padding;

    enum layer_reduce_e reduce_type;
    uint64_t reduce_input_channels; /* Not set directly */
    uint64_t reduce_input_y; /* Not set directly */
    uint64_t reduce_input_x; /* Not set directly */
    uint64_t reduce_kernel_size;
    uint64_t reduce_kernel_stride;
    //uint64_t reduce_kernel_padding;

    uint64_t split_input_channels; /* Not set directly */
    uint64_t split_input_y; /* Not set directly */
    uint64_t split_input_x; /* Not set directly */
    uint64_t split_filters;

    enum residual_e residual_type;
    uint64_t residual_connection_from_layer;
    uint64_t residual_convolution_input_channels; /* Not set directly */
    uint64_t residual_convolution_input_y; /* Not set directly */
    uint64_t residual_convolution_input_x; /* Not set directly */
    uint64_t residual_convolution_filters;
    uint64_t residual_convolution_kernel_size;
    uint64_t residual_convolution_kernel_stride;
    uint64_t residual_convolution_kernel_padding;

    enum activation_e activation_type;
    enum norm_e norm_type;
} layerconfig_t;

typedef struct {
    activation_t *activation_type;
    enum norm_e norm_type;
    enum layer_e layer_type;

    dense_t *dense;
    convolution_t *convolution;
    reduce_t *reduce;
    split_t *split;

    residual_t *residual;
    
    tensor_t *activation;
    tensor_t *activation_g;
} layer_t;

extern layer_t layer_alloc(layerconfig_t *layerconfig);
extern void layer_free(layer_t *layer);

typedef struct {
    uint64_t layers;
    layer_t *layer;
    linearized_t *linearized_forward;
    // linearized_t *linearized_backward;
} neuralnet_t;

#define NEURALNET_INPUT(neuralnet) ((neuralnet).layer[0].activation)
#define NEURALNET_INPUT_(neuralnet) ((neuralnet)->layer[0].activation)
#define NEURALNET_OUTPUT(neuralnet) ((neuralnet).layer[(neuralnet).layers - 1].activation)
#define NEURALNET_OUTPUT_(neuralnet) ((neuralnet)->layer[(neuralnet)->layers - 1].activation)

extern neuralnet_t neuralnet_alloc(uint64_t layers, layerconfig_t **layerconfig);
extern void neuralnet_free(neuralnet_t *neuralnet);
/* NOTE: Used for linearizing all needed ops from the input to the output. Only need to be called once per neuralnet. */
extern void neuralnet_linearize(neuralnet_t *neuralnet);
extern void neuralnet_forward(neuralnet_t *neuralnet, tensor_t *input);
extern void neuralnet_backward(neuralnet_t *neuralnet, tensor_t *training_input, tensor_t *training_output);
extern void neuralnet_learn(neuralnet_t *neuralnet, double learning);
extern void neuralnet_print(neuralnet_t *neuralnet, int padding, int offset, const char *name);
extern void neuralnet_print_shape(neuralnet_t *neuralnet, int padding, int offset, const char *name);

#endif
