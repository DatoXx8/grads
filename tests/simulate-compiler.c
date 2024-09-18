#include <CL/cl.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../compile.h"
#include "../runtimes/cl.h"
#include "../tensor.h"
#include "../utils.h"

#define RANDOM_MAX_TRIES 100ul
const uint64_t DIM_SZE = 3;
const double EPSILON = 1e-3;
const double MARGIN_OF_ERROR = 1e-4; /* 0.01% max error */
#define TENSOR_NUM 16ul
#define MAX_LOOPS 4096ul
#define OP_NUM 6ul
#define SWITCH_ODS ((double) 1 / (double) 16)
static void simulate_compiler(tensor_t *tensor1, tensor_t *tensor2, cl_device_id *device_id, cl_context *context,
                              cl_command_queue *command_queue) {
    assert(tensor1);
    assert(tensor2);
    assert(device_id);
    assert(*device_id);
    assert(context);
    assert(*context);
    assert(command_queue);
    assert(*command_queue);

    uint64_t a_off, z_off, y_off, x_off;
    uint64_t a_sze, z_sze, y_sze, x_sze;
    op_e bp_type[OP_NUM];
    unary_e bp_unary[OP_NUM];
    double bp_val[OP_NUM];
    binary_e bp_binary[OP_NUM];
    reduce_e bp_reduce[OP_NUM];
    uint64_t bp_out_idx[OP_NUM];
    uint64_t bp_in_idx[OP_NUM];

    uint64_t bp_base_out = rand() % TENSOR_NUM;
    uint64_t bp_base_in;
    for(uint64_t rand_idx = 0; rand_idx < RANDOM_MAX_TRIES; rand_idx++) {
        bp_base_in = rand() % TENSOR_NUM;
        if(bp_base_in != bp_base_out) {
            break;
        }
    }
    assert(bp_base_in != bp_base_out);
    bp_out_idx[0] = bp_base_out;
    bp_in_idx[0] = bp_base_in;

    /* This is up here to make sure that I can reduce the number of ops and that is the only thing that changes */
    const uint64_t size_a = rand() % DIM_SZE + 1;
    const uint64_t size_z = rand() % DIM_SZE + 1;
    const uint64_t size_y = rand() % DIM_SZE + 1;
    const uint64_t size_x = rand() % DIM_SZE + 1;

    const uint64_t a_loop = size_a == DIM_SZE ? 1 : rand() % (DIM_SZE - size_a) + 1;
    const uint64_t z_loop = size_z == DIM_SZE ? 1 : rand() % (DIM_SZE - size_z) + 1;
    const uint64_t y_loop = size_y == DIM_SZE ? 1 : rand() % (DIM_SZE - size_y) + 1;
    const uint64_t x_loop = size_x == DIM_SZE ? 1 : rand() % (DIM_SZE - size_x) + 1;

    for(uint64_t op_idx = 0; op_idx < OP_NUM; op_idx++) {
        if(op_idx) {
            const double switch_tensor = ((double) rand()) / RAND_MAX;
            if(switch_tensor <= SWITCH_ODS) {
                bp_in_idx[op_idx] = bp_out_idx[op_idx - 1];
                for(uint64_t rand_idx = 0; rand_idx < RANDOM_MAX_TRIES; rand_idx++) {
                    bp_out_idx[op_idx] = rand() % TENSOR_NUM;
                    if(bp_out_idx[op_idx] != bp_in_idx[op_idx]) {
                        break;
                    }
                }
                assert(bp_out_idx[op_idx] != bp_in_idx[op_idx]);
            } else {
                bp_out_idx[op_idx] = bp_out_idx[op_idx - 1];
                bp_in_idx[op_idx] = bp_in_idx[op_idx - 1];
            }

            bp_type[op_idx] = rand() % 2;
            switch(bp_type[op_idx]) {
                case op_unary: {
                    bp_unary[op_idx] = rand() % 16;
                    if(bp_unary[op_idx] == unary_divide) {
                        bp_val[op_idx] = ((double) rand() / RAND_MAX) + 1;
                    } else {
                        bp_val[op_idx] = 2 * ((double) rand() / RAND_MAX) - 1;
                    }
                    break;
                }
                case op_binary: {
                    bp_binary[op_idx] = rand() % 14;
                    break;
                }
                case op_reduce: {
                    ERROR("Op move is invalid for ops other than the first in `simulate_compiler()`\n");
                }
                case op_move: {
                    ERROR("Op move is invalid in `simulate_compiler()`\n");
                }
            }
        } else {
            bp_type[0] = rand() % 3;
            switch(bp_type[0]) {
                case op_unary: {
                    bp_unary[0] = rand() % 16;
                    bp_val[0] = ((double) rand() / RAND_MAX) + 1;
                    break;
                }
                case op_binary: {
                    bp_binary[0] = rand() % 14;
                    break;
                }
                case op_reduce: {
                    bp_reduce[0] = rand() % 4;
                    break;
                }
                case op_move: {
                    ERROR("Op move is invalid in `simulate_compiler()`\n");
                }
            }
        }
    }
    for(uint64_t a_idx = 0; a_idx < a_loop; a_idx++) {
        for(uint64_t z_idx = 0; z_idx < z_loop; z_idx++) {
            for(uint64_t y_idx = 0; y_idx < y_loop; y_idx++) {
                for(uint64_t x_idx = 0; x_idx < x_loop; x_idx++) {
                    for(uint64_t op_idx = 0; op_idx < OP_NUM - 4; op_idx++) {
                        if(bp_type[op_idx] == op_binary && bp_binary[op_idx] < binary_add_like) {
                            tensor_move_resize(&tensor1[bp_in_idx[op_idx]], size_a, size_z, size_y, size_x);
                            tensor_move_resize(&tensor2[bp_in_idx[op_idx]], size_a, size_z, size_y, size_x);
                        } else {
                            tensor_move_resize(&tensor1[bp_in_idx[op_idx]], 1, 1, 1, 1);
                            tensor_move_resize(&tensor2[bp_in_idx[op_idx]], 1, 1, 1, 1);
                        }

                        if(bp_type[op_idx] == op_reduce) {
                            tensor_move_resize(&tensor1[bp_out_idx[op_idx]], 1, 1, 1, 1);
                            tensor_move_resize(&tensor2[bp_out_idx[op_idx]], 1, 1, 1, 1);
                        } else {
                            tensor_move_resize(&tensor1[bp_out_idx[op_idx]], size_a, size_z, size_y, size_x);
                            tensor_move_resize(&tensor2[bp_out_idx[op_idx]], size_a, size_z, size_y, size_x);
                        }

                        tensor_move_offset(&tensor1[bp_out_idx[op_idx]], a_idx, z_idx, y_idx, x_idx);
                        tensor_move_offset(&tensor1[bp_in_idx[op_idx]], a_idx, z_idx, y_idx, x_idx);
                        tensor_move_offset(&tensor2[bp_out_idx[op_idx]], a_idx, z_idx, y_idx, x_idx);
                        tensor_move_offset(&tensor2[bp_in_idx[op_idx]], a_idx, z_idx, y_idx, x_idx);

                        switch(bp_type[op_idx]) {
                            case op_unary: {
                                switch(bp_unary[op_idx]) {
                                    case unary_add: {
                                        tensor_unary_add(&tensor1[bp_out_idx[op_idx]], bp_val[op_idx]);
                                        tensor_unary_add(&tensor2[bp_out_idx[op_idx]], bp_val[op_idx]);
                                        break;
                                    }
                                    case unary_subtract: {
                                        tensor_unary_subtract(&tensor1[bp_out_idx[op_idx]], bp_val[op_idx]);
                                        tensor_unary_subtract(&tensor2[bp_out_idx[op_idx]], bp_val[op_idx]);
                                        break;
                                    }
                                    case unary_multiply: {
                                        tensor_unary_multiply(&tensor1[bp_out_idx[op_idx]], bp_val[op_idx]);
                                        tensor_unary_multiply(&tensor2[bp_out_idx[op_idx]], bp_val[op_idx]);
                                        break;
                                    }
                                    case unary_divide: {
                                        tensor_unary_divide(&tensor1[bp_out_idx[op_idx]], bp_val[op_idx]);
                                        tensor_unary_divide(&tensor2[bp_out_idx[op_idx]], bp_val[op_idx]);
                                        break;
                                    }
                                    case unary_exp: {
                                        tensor_unary_exp(&tensor1[bp_out_idx[op_idx]]);
                                        tensor_unary_exp(&tensor2[bp_out_idx[op_idx]]);
                                        break;
                                    }
                                    case unary_log: {
                                        tensor_unary_log(&tensor1[bp_out_idx[op_idx]]);
                                        tensor_unary_log(&tensor2[bp_out_idx[op_idx]]);
                                        break;
                                    }
                                    case unary_square: {
                                        tensor_unary_square(&tensor1[bp_out_idx[op_idx]]);
                                        tensor_unary_square(&tensor2[bp_out_idx[op_idx]]);
                                        break;
                                    }
                                    case unary_sqrt: {
                                        /* Nan prevention */
                                        tensor_unary_absolute(&tensor2[bp_out_idx[op_idx]]);
                                        tensor_unary_absolute(&tensor1[bp_out_idx[op_idx]]);

                                        tensor_unary_sqrt(&tensor1[bp_out_idx[op_idx]]);
                                        tensor_unary_sqrt(&tensor2[bp_out_idx[op_idx]]);
                                        break;
                                    }
                                    case unary_reciprocal: {
                                        /* Nan prevention */
                                        tensor_unary_absolute(&tensor1[bp_out_idx[op_idx]]);
                                        tensor_unary_absolute(&tensor2[bp_out_idx[op_idx]]);
                                        tensor_unary_add(&tensor1[bp_out_idx[op_idx]], 1);
                                        tensor_unary_add(&tensor2[bp_out_idx[op_idx]], 1);

                                        tensor_unary_reciprocal(&tensor1[bp_out_idx[op_idx]]);
                                        tensor_unary_reciprocal(&tensor2[bp_out_idx[op_idx]]);
                                        break;
                                    }
                                    case unary_max: {
                                        tensor_unary_max(&tensor1[bp_out_idx[op_idx]], bp_val[op_idx]);
                                        tensor_unary_max(&tensor2[bp_out_idx[op_idx]], bp_val[op_idx]);
                                        break;
                                    }
                                    case unary_min: {
                                        tensor_unary_min(&tensor1[bp_out_idx[op_idx]], bp_val[op_idx]);
                                        tensor_unary_min(&tensor2[bp_out_idx[op_idx]], bp_val[op_idx]);
                                        break;
                                    }
                                    case unary_set: {
                                        if(op_idx == 0) {
                                            tensor_unary_set(&tensor1[bp_out_idx[op_idx]], bp_val[op_idx]);
                                            tensor_unary_set(&tensor2[bp_out_idx[op_idx]], bp_val[op_idx]);
                                            break;
                                        } else {
                                            // Fallthrough to get to the next one
                                        }
                                    }
                                    case unary_random: {
                                        // Fallthrough to get to the next one

                                        // tensor_unary_random(&tensor1[bp_out_idx[idx_op]], bp_val[idx_op]);
                                        // tensor_unary_random(&tensor2[bp_out_idx[idx_op]], bp_val[idx_op]);
                                        // break;
                                    }
                                    case unary_tanh: {
                                        tensor_unary_tanh(&tensor1[bp_out_idx[op_idx]]);
                                        tensor_unary_tanh(&tensor2[bp_out_idx[op_idx]]);
                                        break;
                                    }
                                    case unary_sign: {
                                        // Fallthrough to get to the next one

                                        // tensor_unary_sign(&tensor1[bp_out_idx[idx_op]]);
                                        // tensor_unary_sign(&tensor2[bp_out_idx[idx_op]]);
                                        // break;
                                    }
                                    case unary_absolute: {
                                        tensor_unary_absolute(&tensor2[bp_out_idx[op_idx]]);
                                        tensor_unary_absolute(&tensor1[bp_out_idx[op_idx]]);
                                        break;
                                    }
                                }
                                break;
                            }
                            case op_binary: {
                                switch(bp_binary[op_idx]) {
                                    case binary_add: {
                                        tensor_binary_add(&tensor1[bp_out_idx[op_idx]], &tensor1[bp_in_idx[op_idx]]);
                                        tensor_binary_add(&tensor2[bp_out_idx[op_idx]], &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                    case binary_subtract: {
                                        tensor_binary_subtract(&tensor1[bp_out_idx[op_idx]],
                                                               &tensor1[bp_in_idx[op_idx]]);
                                        tensor_binary_subtract(&tensor2[bp_out_idx[op_idx]],
                                                               &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                    case binary_multiply: {
                                        tensor_binary_multiply(&tensor1[bp_out_idx[op_idx]],
                                                               &tensor1[bp_in_idx[op_idx]]);
                                        tensor_binary_multiply(&tensor2[bp_out_idx[op_idx]],
                                                               &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                    case binary_divide: {
                                        /* Nan prevention */
                                        tensor_unary_absolute(&tensor1[bp_in_idx[op_idx]]);
                                        tensor_unary_absolute(&tensor2[bp_in_idx[op_idx]]);
                                        tensor_unary_add(&tensor1[bp_in_idx[op_idx]], 1);
                                        tensor_unary_add(&tensor2[bp_in_idx[op_idx]], 1);

                                        tensor_binary_divide(&tensor1[bp_out_idx[op_idx]], &tensor1[bp_in_idx[op_idx]]);
                                        tensor_binary_divide(&tensor2[bp_out_idx[op_idx]], &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                    case binary_max: {
                                        tensor_binary_max(&tensor1[bp_out_idx[op_idx]], &tensor1[bp_in_idx[op_idx]]);
                                        tensor_binary_max(&tensor2[bp_out_idx[op_idx]], &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                    case binary_min: {
                                        tensor_binary_min(&tensor1[bp_out_idx[op_idx]], &tensor1[bp_in_idx[op_idx]]);
                                        tensor_binary_min(&tensor2[bp_out_idx[op_idx]], &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                    case binary_copy: {
                                        tensor_binary_copy(&tensor1[bp_out_idx[op_idx]], &tensor1[bp_in_idx[op_idx]]);
                                        tensor_binary_copy(&tensor2[bp_out_idx[op_idx]], &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                    case binary_add_like: {
                                        tensor_lbinary_add(&tensor1[bp_out_idx[op_idx]], &tensor1[bp_in_idx[op_idx]]);
                                        tensor_lbinary_add(&tensor2[bp_out_idx[op_idx]], &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                    case binary_subtract_like: {
                                        tensor_lbinary_subtract(&tensor1[bp_out_idx[op_idx]],
                                                                &tensor1[bp_in_idx[op_idx]]);
                                        tensor_lbinary_subtract(&tensor2[bp_out_idx[op_idx]],
                                                                &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                    case binary_multiply_like: {
                                        tensor_lbinary_multiply(&tensor1[bp_out_idx[op_idx]],
                                                                &tensor1[bp_in_idx[op_idx]]);
                                        tensor_lbinary_multiply(&tensor2[bp_out_idx[op_idx]],
                                                                &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                    case binary_divide_like: {
                                        /* Nan prevention */
                                        tensor_unary_absolute(&tensor1[bp_in_idx[op_idx]]);
                                        tensor_unary_absolute(&tensor2[bp_in_idx[op_idx]]);
                                        tensor_unary_add(&tensor1[bp_in_idx[op_idx]], 1);
                                        tensor_unary_add(&tensor2[bp_in_idx[op_idx]], 1);

                                        tensor_lbinary_divide(&tensor1[bp_out_idx[op_idx]],
                                                              &tensor1[bp_in_idx[op_idx]]);
                                        tensor_lbinary_divide(&tensor2[bp_out_idx[op_idx]],
                                                              &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                    case binary_max_like: {
                                        tensor_lbinary_max(&tensor1[bp_out_idx[op_idx]], &tensor1[bp_in_idx[op_idx]]);
                                        tensor_lbinary_max(&tensor2[bp_out_idx[op_idx]], &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                    case binary_min_like: {
                                        tensor_lbinary_min(&tensor1[bp_out_idx[op_idx]], &tensor1[bp_in_idx[op_idx]]);
                                        tensor_lbinary_min(&tensor2[bp_out_idx[op_idx]], &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                    case binary_copy_like: {
                                        tensor_lbinary_copy(&tensor1[bp_out_idx[op_idx]], &tensor1[bp_in_idx[op_idx]]);
                                        tensor_lbinary_copy(&tensor2[bp_out_idx[op_idx]], &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                }
                                break;
                            }
                            case op_reduce: {
                                switch(bp_reduce[op_idx]) {
                                    case reduce_sum: {
                                        tensor_reduce_sum(&tensor1[bp_out_idx[op_idx]], &tensor1[bp_in_idx[op_idx]]);
                                        tensor_reduce_sum(&tensor2[bp_out_idx[op_idx]], &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                    case reduce_max: {
                                        tensor_reduce_max(&tensor1[bp_out_idx[op_idx]], &tensor1[bp_in_idx[op_idx]]);
                                        tensor_reduce_max(&tensor2[bp_out_idx[op_idx]], &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                    case reduce_min: {
                                        tensor_reduce_min(&tensor1[bp_out_idx[op_idx]], &tensor1[bp_in_idx[op_idx]]);
                                        tensor_reduce_min(&tensor2[bp_out_idx[op_idx]], &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                    case reduce_avg: {
                                        tensor_reduce_avg(&tensor1[bp_out_idx[op_idx]], &tensor1[bp_in_idx[op_idx]]);
                                        tensor_reduce_avg(&tensor2[bp_out_idx[op_idx]], &tensor2[bp_in_idx[op_idx]]);
                                        break;
                                    }
                                }
                                break;
                            }
                            case op_move: {
                                UNREACHABLE();
                            }
                        }
                    }
                }
            }
        }
    }

    // LINEARIZED_PRINT_(tensor1[bp_out_idx[OP_NUM - 1]].linearized);
    // LINEARIZED_PRINT_(tensor2[bp_out_idx[OP_NUM - 1]].linearized);
    linearized_run(tensor1[bp_out_idx[OP_NUM - 1]].linearized);
    program_t program = {0};
    program_compile(&program, tensor2[bp_out_idx[OP_NUM - 1]].linearized, device_id, context, command_queue, 9, 9);
    // for(uint64_t kernel_idx = 0; kernel_idx < program.kernel_num; kernel_idx++) {
    //     printf("%s\n", program.kernel[kernel_idx].source);
    // }
    for(uint64_t tensor_idx = 0; tensor_idx < TENSOR_NUM; tensor_idx++) {
        buffer_sync_update(tensor2[tensor_idx].buffer, sync_to_device);
        buffer_sync_realize(tensor2[tensor_idx].buffer, *command_queue);
    }
    program_run(&program);
    for(uint64_t tensor_idx = 0; tensor_idx < TENSOR_NUM; tensor_idx++) {
        buffer_sync_update(tensor2[tensor_idx].buffer, sync_to_host);
        buffer_sync_realize(tensor2[tensor_idx].buffer, *command_queue);
    }
    clFinish(*command_queue);

    tensor_move_resize(&tensor1[bp_out_idx[OP_NUM - 1]], DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE);
    tensor_move_resize(&tensor2[bp_out_idx[OP_NUM - 1]], DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE);
    tensor_move_offset(&tensor1[bp_out_idx[OP_NUM - 1]], 0, 0, 0, 0);
    tensor_move_offset(&tensor2[bp_out_idx[OP_NUM - 1]], 0, 0, 0, 0);
    // TENSOR_PRINT(tensor1[bp_out_idx[OP_NUM - 1]]);
    // TENSOR_PRINT(tensor2[bp_out_idx[OP_NUM - 1]]);
    double margin_of_error = pow(1 + MARGIN_OF_ERROR, OP_NUM) - 1;
    for(uint64_t a = 0; a < DIM_SZE; a++) {
        for(uint64_t z = 0; z < DIM_SZE; z++) {
            for(uint64_t y = 0; y < DIM_SZE; y++) {
                for(uint64_t x = 0; x < DIM_SZE; x++) {
                    /* Both isnan and isinf should be xnor I guess */
                    assert(!isnan(BUFFER_AT_(tensor1[bp_out_idx[OP_NUM - 1]].buffer, a, z, y, x)));
                    assert(!isnan(BUFFER_AT_(tensor2[bp_out_idx[OP_NUM - 1]].buffer, a, z, y, x)));
                    assert(!isinf(BUFFER_AT_(tensor1[bp_out_idx[OP_NUM - 1]].buffer, a, z, y, x)));
                    assert(!isinf(BUFFER_AT_(tensor2[bp_out_idx[OP_NUM - 1]].buffer, a, z, y, x)));
                    if(fabs(BUFFER_AT_(tensor1[bp_out_idx[OP_NUM - 1]].buffer, a, z, y, x) -
                            BUFFER_AT_(tensor2[bp_out_idx[OP_NUM - 1]].buffer, a, z, y, x)) > margin_of_error) {
                        if(fabs(BUFFER_AT_(tensor1[bp_out_idx[OP_NUM - 1]].buffer, a, z, y, x) /
                                    BUFFER_AT_(tensor2[bp_out_idx[OP_NUM - 1]].buffer, a, z, y, x) -
                                1) > margin_of_error) {
                            ERROR("Invalid values %lf %lf in tensors %lu %s and %s\n",
                                  BUFFER_AT_(tensor1[bp_out_idx[OP_NUM - 1]].buffer, a, z, y, x),
                                  BUFFER_AT_(tensor2[bp_out_idx[OP_NUM - 1]].buffer, a, z, y, x),
                                  bp_out_idx[OP_NUM - 1], tensor1[bp_out_idx[OP_NUM - 1]].buffer->name,
                                  tensor2[bp_out_idx[OP_NUM - 1]].buffer->name);
                        }
                    }
                }
            }
        }
    }

    for(uint64_t kernel_idx = 0; kernel_idx < program.kernel_num; kernel_idx++) {
        for(uint64_t arg_idx = 0; arg_idx < program.kernel[kernel_idx].arg_num; arg_idx++) {
            free(program.kernel[kernel_idx].arg_name[arg_idx]);
        }
        free(program.kernel[kernel_idx].arg_name);
        free(program.kernel[kernel_idx].arg_mem);
        free(program.kernel[kernel_idx].source);
        clReleaseKernel(program.kernel[kernel_idx].cl_kernel);
        clReleaseProgram(program.kernel[kernel_idx].cl_program);
    }
    free(program.kernel);
    program.kernel = NULL;
}

int main(int argc, char **argv) {
    assert(argc == 1 || argc == 3); /* 0 or 2 args but since argv[0] is the program name this is 1 and 3 */
    uint32_t rng;
    if(argc == 1) {
        rng = time(NULL);
        printf("Compiler simulation with random %u...\n", rng);
    } else {
        if(strncmp(argv[1], "--rng", 5) != 0) {
            ERROR("Expected second argument to be `--rng` but got `%s`\n", argv[1]);
        }
        rng = (uint32_t) strtoul(argv[2], NULL, 10);
        printf("Compiler simulation with provided %u...\n", rng);
    }
    srand(rng);

    int32_t err;
    cl_device_id device_id = cl_device_get();
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    assert(!err);
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &err);
    assert(!err);

    tensor_t *tensor1 = calloc(TENSOR_NUM, sizeof(tensor_t));
    tensor_t *tensor2 = calloc(TENSOR_NUM, sizeof(tensor_t));
    assert(tensor1);
    assert(tensor2);
    for(uint64_t tensor_idx = 0; tensor_idx < TENSOR_NUM; tensor_idx++) {
        /* This variation is to test wether indexing still works correctly with offsets and different sizes */
        const uint64_t a_size = DIM_SZE + rand() % 3;
        const uint64_t z_size = DIM_SZE + rand() % 3;
        const uint64_t y_size = DIM_SZE + rand() % 3;
        const uint64_t x_size = DIM_SZE + rand() % 3;
        /* TODO: Make make random offsets */
        tensor1[tensor_idx] = tensor_alloc(a_size, z_size, y_size, x_size, context);
        tensor2[tensor_idx] = tensor_alloc(a_size, z_size, y_size, x_size, context);
        for(uint64_t val_idx = 0; val_idx < a_size * z_size * y_size * x_size; val_idx++) {
            tensor1[tensor_idx].buffer->val[val_idx] = 2 * (double) rand() / (double) RAND_MAX - 1;
            tensor2[tensor_idx].buffer->val[val_idx] = tensor1[tensor_idx].buffer->val[val_idx];
            tensor1[tensor_idx].buffer->val[val_idx] = 1;
            tensor2[tensor_idx].buffer->val[val_idx] = tensor1[tensor_idx].buffer->val[val_idx];
        }
        tensor_move_reshape(&tensor1[tensor_idx], DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE);
        tensor_move_reshape(&tensor2[tensor_idx], DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE);
    }

    simulate_compiler(tensor1, tensor2, &device_id, &context, &command_queue);

    for(uint64_t tensor_idx = 0; tensor_idx < TENSOR_NUM; tensor_idx++) {
        tensor_free(&tensor1[tensor_idx]);
        tensor_free(&tensor2[tensor_idx]);
    }
    clReleaseDevice(device_id);
    clReleaseContext(context);
    clReleaseCommandQueue(command_queue);
    free(tensor1);
    free(tensor2);
    printf("Passed\n");
    return 0;
}
