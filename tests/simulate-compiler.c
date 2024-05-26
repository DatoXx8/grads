#include <CL/cl.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compile.h"
#include "../runtimes/cl.h"
#include "../tensor.h"

static void program_free_non_cl(program_t *program) {
    for(int64_t arg_idx = 0; arg_idx < program->arg_num; arg_idx++) { free(program->arg_name[arg_idx]); }
    free(program->arg_name);
    program->arg_name = NULL;
    free(program->arg_mem);
    program->arg_mem = NULL;
    free(program->source);
    program->source = NULL;
    if(program->cl_kernel) {
        clReleaseKernel(program->cl_kernel);
        program->cl_kernel = NULL;
    }
    /* This is a very disgusting fix, but I suppose it works for now. TODO: Make this nicer */
    if(program->cl_program) {
        if(*program->cl_program) {
            clReleaseProgram(*program->cl_program);
            *program->cl_program = NULL;
            free(*program->cl_program);
        }
        free(program->cl_program);
        program->cl_program = NULL;
    }
}

const int64_t RANDOM_MAX_TRIES = 100;
const int64_t DIM_SZE = 3;
const double EPSILON = 1e-3;
const double MARGIN_OF_ERROR = 1e-4; /* 0.01% max error */
/* TODO: Increase perf by moving all the tensor allocs and frees to the outside */
static void simulate_compiler(tensor_t *tensor1, tensor_t *tensor2, int64_t op_num, int64_t tensor_num,
                              cl_device_id *device_id, cl_context *context, cl_command_queue *command_queue) {
    assert(tensor1);
    assert(tensor2);
    assert(op_num > 0);
    assert(tensor_num > 1);
    assert(device_id);
    assert(*device_id);
    assert(context);
    assert(*context);
    assert(command_queue);
    assert(*command_queue);

    int64_t tensor_out = rand() % tensor_num, tensor_in;
    int64_t a_off, z_off, y_off, x_off;
    int64_t a_sze, z_sze, y_sze, x_sze;
    op_e type;
    unary_e type_unary;
    binary_e type_binary;
    reduce_e type_reduce;
    for(int64_t op_idx = 0; op_idx < op_num; op_idx++) {
        type = rand() % 3;
        switch(type) {
            /* TODO: Make this randomize in and out tensor */
            /* TODO: Make sure exp and square don't get too big */
            case op_unary: {
                // tensor_out = tensor_in;
                type_unary = rand() % 16;
                a_sze = rand() % DIM_SZE + 1;
                z_sze = rand() % DIM_SZE + 1;
                y_sze = rand() % DIM_SZE + 1;
                x_sze = rand() % DIM_SZE + 1;
                a_off = DIM_SZE == a_sze ? 0 : rand() % (DIM_SZE - a_sze);
                z_off = DIM_SZE == z_sze ? 0 : rand() % (DIM_SZE - z_sze);
                y_off = DIM_SZE == y_sze ? 0 : rand() % (DIM_SZE - y_sze);
                x_off = DIM_SZE == x_sze ? 0 : rand() % (DIM_SZE - x_sze);
                tensor_move_resize(&tensor1[tensor_out], a_sze, z_sze, y_sze, x_sze);
                tensor_move_resize(&tensor2[tensor_out], a_sze, z_sze, y_sze, x_sze);
                tensor_move_offset(&tensor1[tensor_out], a_off, z_off, y_off, x_off);
                tensor_move_offset(&tensor2[tensor_out], a_off, z_off, y_off, x_off);
                double random_value = ((double) rand() / RAND_MAX) * 2 - 1;
                switch(type_unary) {
                    case unary_add: {
                        tensor_unary_add(&tensor1[tensor_out], random_value);
                        tensor_unary_add(&tensor2[tensor_out], random_value);
                        break;
                    }
                    case unary_subtract: {
                        tensor_unary_subtract(&tensor1[tensor_out], random_value);
                        tensor_unary_subtract(&tensor2[tensor_out], random_value);
                        break;
                    }
                    case unary_multiply: {
                        tensor_unary_multiply(&tensor1[tensor_out], random_value);
                        tensor_unary_multiply(&tensor2[tensor_out], random_value);
                        break;
                    }
                    case unary_divide: {
                        random_value += 2;
                        tensor_unary_divide(&tensor1[tensor_out], random_value);
                        tensor_unary_divide(&tensor2[tensor_out], random_value);
                        break;
                    }
                    case unary_exp: {
                        tensor_unary_exp(&tensor1[tensor_out]);
                        tensor_unary_exp(&tensor2[tensor_out]);
                        break;
                    }
                    case unary_log: {
                        tensor_unary_absolute(&tensor1[tensor_out]);
                        tensor_unary_absolute(&tensor2[tensor_out]);
                        tensor_unary_add(&tensor1[tensor_out], EPSILON);
                        tensor_unary_add(&tensor2[tensor_out], EPSILON);
                        tensor_unary_log(&tensor1[tensor_out]);
                        tensor_unary_log(&tensor2[tensor_out]);
                        break;
                    }
                    case unary_square: {
                        tensor_unary_square(&tensor1[tensor_out]);
                        tensor_unary_square(&tensor2[tensor_out]);
                        break;
                    }
                    case unary_sqrt: {
                        tensor_unary_absolute(&tensor1[tensor_out]);
                        tensor_unary_absolute(&tensor2[tensor_out]);
                        tensor_unary_sqrt(&tensor1[tensor_out]);
                        tensor_unary_sqrt(&tensor2[tensor_out]);
                        break;
                    }
                    case unary_reciprocal: {
                        tensor_unary_absolute(&tensor1[tensor_out]);
                        tensor_unary_absolute(&tensor2[tensor_out]);
                        tensor_unary_add(&tensor1[tensor_out], EPSILON);
                        tensor_unary_add(&tensor2[tensor_out], EPSILON);
                        tensor_unary_reciprocal(&tensor1[tensor_out]);
                        tensor_unary_reciprocal(&tensor2[tensor_out]);
                        break;
                    }
                    case unary_max: {
                        tensor_unary_max(&tensor1[tensor_out], random_value);
                        tensor_unary_max(&tensor2[tensor_out], random_value);
                        break;
                    }
                    case unary_min: {
                        tensor_unary_min(&tensor1[tensor_out], random_value);
                        tensor_unary_min(&tensor2[tensor_out], random_value);
                        break;
                    }
                    case unary_set: {
                        tensor_unary_set(&tensor1[tensor_out], random_value);
                        tensor_unary_set(&tensor2[tensor_out], random_value);
                        break;
                    }
                    case unary_sign: {
                    }
                    case unary_random: {
                    }
                    case unary_tanh: {
                        tensor_unary_tanh(&tensor1[tensor_out]);
                        tensor_unary_tanh(&tensor2[tensor_out]);
                        break;
                    }
                    case unary_absolute: {
                        tensor_unary_absolute(&tensor1[tensor_out]);
                        tensor_unary_absolute(&tensor2[tensor_out]);
                        break;
                    }
                }
                break;
            }
            case op_binary: {
                // tensor_out = tensor_in;
                for(int64_t ran_try = 0; ran_try < RANDOM_MAX_TRIES; ran_try++) {
                    tensor_in = rand() % tensor_num;
                    if(tensor_out != tensor_in) { break; }
                }
                assert(tensor_in != tensor_out);
                type_binary = rand() % 14;
                a_sze = rand() % DIM_SZE + 1;
                z_sze = rand() % DIM_SZE + 1;
                y_sze = rand() % DIM_SZE + 1;
                x_sze = rand() % DIM_SZE + 1;
                a_off = DIM_SZE == a_sze ? 0 : rand() % (DIM_SZE - a_sze);
                z_off = DIM_SZE == z_sze ? 0 : rand() % (DIM_SZE - z_sze);
                y_off = DIM_SZE == y_sze ? 0 : rand() % (DIM_SZE - y_sze);
                x_off = DIM_SZE == x_sze ? 0 : rand() % (DIM_SZE - x_sze);
                tensor_move_resize(&tensor1[tensor_out], a_sze, z_sze, y_sze, x_sze);
                tensor_move_resize(&tensor2[tensor_out], a_sze, z_sze, y_sze, x_sze);
                tensor_move_offset(&tensor1[tensor_out], a_off, z_off, y_off, x_off);
                tensor_move_offset(&tensor2[tensor_out], a_off, z_off, y_off, x_off);
                if(type_binary >= binary_add_like) {
                    a_sze = 1;
                    z_sze = 1;
                    y_sze = 1;
                    x_sze = 1;
                }
                a_off = DIM_SZE == a_sze ? 0 : rand() % (DIM_SZE - a_sze);
                z_off = DIM_SZE == z_sze ? 0 : rand() % (DIM_SZE - z_sze);
                y_off = DIM_SZE == y_sze ? 0 : rand() % (DIM_SZE - y_sze);
                x_off = DIM_SZE == x_sze ? 0 : rand() % (DIM_SZE - x_sze);
                tensor_move_resize(&tensor1[tensor_in], a_sze, z_sze, y_sze, x_sze);
                tensor_move_resize(&tensor2[tensor_in], a_sze, z_sze, y_sze, x_sze);
                tensor_move_offset(&tensor1[tensor_in], a_off, z_off, y_off, x_off);
                tensor_move_offset(&tensor2[tensor_in], a_off, z_off, y_off, x_off);
                switch(type_binary) {
                    case binary_add: {
                        tensor_binary_add(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_binary_add(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                    case binary_subtract: {
                        tensor_binary_subtract(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_binary_subtract(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                    case binary_multiply: {
                        tensor_binary_multiply(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_binary_multiply(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                    case binary_divide: {
                        tensor_unary_absolute(&tensor1[tensor_in]);
                        tensor_unary_absolute(&tensor2[tensor_in]);
                        tensor_unary_add(&tensor1[tensor_in], EPSILON);
                        tensor_unary_add(&tensor2[tensor_in], EPSILON);
                        tensor_binary_divide(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_binary_divide(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                    case binary_max: {
                        tensor_binary_max(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_binary_max(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                    case binary_min: {
                        tensor_binary_min(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_binary_min(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                    case binary_copy: {
                        tensor_binary_copy(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_binary_copy(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                    case binary_add_like: {
                        tensor_lbinary_add(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_lbinary_add(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                    case binary_subtract_like: {
                        tensor_lbinary_subtract(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_lbinary_subtract(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                    case binary_multiply_like: {
                        tensor_lbinary_multiply(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_lbinary_multiply(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                    case binary_divide_like: {
                        tensor_unary_absolute(&tensor1[tensor_in]);
                        tensor_unary_absolute(&tensor2[tensor_in]);
                        tensor_unary_add(&tensor1[tensor_in], EPSILON);
                        tensor_unary_add(&tensor2[tensor_in], EPSILON);
                        tensor_lbinary_divide(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_lbinary_divide(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                    case binary_max_like: {
                        tensor_lbinary_max(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_lbinary_max(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                    case binary_min_like: {
                        tensor_lbinary_min(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_lbinary_min(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                    case binary_copy_like: {
                        tensor_lbinary_copy(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_lbinary_copy(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case op_reduce: {
                // tensor_out = tensor_in;
                for(int64_t ran_try = 0; ran_try < RANDOM_MAX_TRIES; ran_try++) {
                    tensor_in = rand() % tensor_num;
                    if(tensor_out != tensor_in) { break; }
                }
                assert(tensor_in != tensor_out);
                type_reduce = rand() % 4;
                a_off = rand() % DIM_SZE;
                z_off = rand() % DIM_SZE;
                y_off = rand() % DIM_SZE;
                x_off = rand() % DIM_SZE;
                tensor_move_resize(&tensor1[tensor_out], 1, 1, 1, 1);
                tensor_move_resize(&tensor2[tensor_out], 1, 1, 1, 1);
                tensor_move_offset(&tensor1[tensor_out], a_off, z_off, y_off, x_off);
                tensor_move_offset(&tensor2[tensor_out], a_off, z_off, y_off, x_off);
                a_sze = rand() % DIM_SZE + 1;
                z_sze = rand() % DIM_SZE + 1;
                y_sze = rand() % DIM_SZE + 1;
                x_sze = rand() % DIM_SZE + 1;
                a_off = DIM_SZE == a_sze ? 0 : rand() % (DIM_SZE - a_sze);
                z_off = DIM_SZE == z_sze ? 0 : rand() % (DIM_SZE - z_sze);
                y_off = DIM_SZE == y_sze ? 0 : rand() % (DIM_SZE - y_sze);
                x_off = DIM_SZE == x_sze ? 0 : rand() % (DIM_SZE - x_sze);
                tensor_move_resize(&tensor1[tensor_in], a_sze, z_sze, y_sze, x_sze);
                tensor_move_resize(&tensor2[tensor_in], a_sze, z_sze, y_sze, x_sze);
                tensor_move_offset(&tensor1[tensor_in], a_off, z_off, y_off, x_off);
                tensor_move_offset(&tensor2[tensor_in], a_off, z_off, y_off, x_off);
                switch(type_reduce) {
                    case reduce_sum: {
                        tensor_reduce_sum(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_reduce_sum(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                    case reduce_avg: {
                        tensor_reduce_avg(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_reduce_avg(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                    case reduce_max: {
                        tensor_reduce_max(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_reduce_max(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                    case reduce_min: {
                        tensor_reduce_min(&tensor1[tensor_out], &tensor1[tensor_in]);
                        tensor_reduce_min(&tensor2[tensor_out], &tensor2[tensor_in]);
                        break;
                    }
                }
                break;
            }
            default: {
                ERROR("Invalind operation in `simulate_compiler()`\n");
            }
        }
    }

    linearized_run(tensor1[tensor_out].linearized);

    program_t program = {0};

    program_compile(&program, tensor2[tensor_out].linearized, device_id, context, command_queue, 9, 1);
    for(int64_t tensor_idx = 0; tensor_idx < tensor_num; tensor_idx++) {
        buffer_sync_update(tensor2[tensor_idx].buffer, sync_to_device);
        buffer_sync_realize(tensor2[tensor_idx].buffer, *command_queue);
    }
    program_run(&program);
    for(int64_t tensor_idx = 0; tensor_idx < tensor_num; tensor_idx++) {
        buffer_sync_update(tensor2[tensor_idx].buffer, sync_to_host);
        buffer_sync_realize(tensor2[tensor_idx].buffer, *command_queue);
    }
    clFinish(*command_queue);

    double margin_of_error = pow(1 + MARGIN_OF_ERROR, op_num) - 1;
    for(int64_t val_idx = 0; val_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; val_idx++) {
        assert(!isnan(tensor1[tensor_out].buffer->val[val_idx]));
        assert(!isnan(tensor2[tensor_out].buffer->val[val_idx]));
        assert(!isinf(tensor1[tensor_out].buffer->val[val_idx]));
        assert(!isinf(tensor2[tensor_out].buffer->val[val_idx]));
        printf("%lf %lf\n", tensor1[tensor_out].buffer->val[val_idx], tensor2[tensor_out].buffer->val[val_idx]);
        if((fabs(tensor1[tensor_out].buffer->val[val_idx] - tensor2[tensor_out].buffer->val[val_idx]) >
            margin_of_error)) {
            if((fabs(tensor1[tensor_out].buffer->val[val_idx] / tensor2[tensor_out].buffer->val[val_idx]) - 1 >
                margin_of_error)) {
                printf("%lf %lf in tensors %lu %s and %s\n", tensor1[tensor_out].buffer->val[val_idx],
                       tensor2[tensor_out].buffer->val[val_idx], tensor_out, tensor1[tensor_out].buffer->name,
                       tensor2[tensor_out].buffer->name);
                ERROR("Invalid values\n");
            }
            printf("%lf %lf in tensors %lu %s and %s. Not too big a diff tho.\n",
                   tensor1[tensor_out].buffer->val[val_idx], tensor2[tensor_out].buffer->val[val_idx], tensor_out,
                   tensor1[tensor_out].buffer->name, tensor2[tensor_out].buffer->name);
        }
    }

    program_free_non_cl(&program);
}

int main(int argc, char **argv) {
    if(argc != 4) {
        printf("USAGE: %s [ops] [tensors] [iterations]\n", argv[0]);
        return 1;
    }
    int err;
    const uint32_t seed = time(NULL);
    printf("RNG Seed %u\n", seed);
    srand(seed);
    const int64_t op_num = strtoll(argv[1], NULL, 10);
    const int64_t tensor_num = strtoll(argv[2], NULL, 10);
    const int64_t iter_num = strtoll(argv[3], NULL, 10);
    assert(op_num > 0);
    assert(tensor_num > 1);
    assert(iter_num > 0);

    cl_device_id device_id = cl_device_get();
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    assert(err == 0);
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &err);
    assert(err == 0);

    double *random_values = calloc(DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE, sizeof(double));
    assert(random_values);
    for(int64_t val_idx = 0; val_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; val_idx++) {
        // random_values[val_idx] = ((double) rand() / RAND_MAX) * 2 - 1;
        random_values[val_idx] = 1;
    }
    tensor_t *tensor1 = calloc(tensor_num, sizeof(tensor_t));
    tensor_t *tensor2 = calloc(tensor_num, sizeof(tensor_t));
    assert(tensor1);
    assert(tensor2);
    for(int64_t tensor_idx = 0; tensor_idx < tensor_num; tensor_idx++) {
        tensor1[tensor_idx] = tensor_alloc(DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE, context);
        tensor2[tensor_idx] = tensor_alloc(DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE, context);
    }

    for(int64_t iter_idx = 0; iter_idx < iter_num; iter_idx++) {
        for(int64_t tensor_idx = 0; tensor_idx < tensor_num; tensor_idx++) {
            memcpy(tensor1[tensor_idx].buffer->val, random_values,
                   DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
            memcpy(tensor2[tensor_idx].buffer->val, random_values,
                   DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
        }
        simulate_compiler(tensor1, tensor2, op_num, tensor_num, &device_id, &context, &command_queue);
    }

    for(int64_t tensor_idx = 0; tensor_idx < tensor_num; tensor_idx++) {
        tensor_free(&tensor1[tensor_idx]);
        tensor_free(&tensor2[tensor_idx]);
    }
    clReleaseDevice(device_id);
    clReleaseContext(context);
    clReleaseCommandQueue(command_queue);
    free(tensor1);
    free(tensor2);
    free(random_values);
    printf("Passed compiler simulation with %lu ops, %lu tensors, and %lu iteratations!\n", op_num, tensor_num,
           iter_num);
    return 0;
}
