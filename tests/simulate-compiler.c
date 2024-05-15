#include <CL/cl.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compile.h"
#include "../linearize.h"
#include "../runtimes/cl.h"
#include "../tensor.h"

void program_free_non_reusable(program_t *program) {
    for(int64_t kernel_idx = 0; kernel_idx < program->kernel_num; kernel_idx++) {
        for(int64_t i = 0; i < program->kernel[kernel_idx].arg_num; i++) {
            free(program->kernel[kernel_idx].arg_name[i]);
        }
        free(program->kernel[kernel_idx].arg_name);
        free((void *) program->kernel[kernel_idx].name);
        free(program->kernel[kernel_idx].source);
        free(program->kernel[kernel_idx].arg_mem);
        if(program->kernel[kernel_idx].cl_kernel) {
            clReleaseKernel(*program->kernel[kernel_idx].cl_kernel);
            free(program->kernel[kernel_idx].cl_kernel);
            program->kernel[kernel_idx].cl_kernel = NULL;
        }
    }
    free(program->kernel);
    free(program->source);
    clReleaseProgram(*program->cl_program);
    free(program->cl_program);
}

const int64_t RANDOM_MAX_TRIES = 100;
const int64_t DIM_SZE = 3;
const double EPSILON = 1e-3;
const double MARGIN_OF_ERROR = 1e-4; /* .1% max error */
/* TODO: Increase perf by moving all the tensor allocs and frees to the outside */
void simulate_compile(int64_t op_num, int64_t tensor_num, cl_command_queue *command_queue, cl_context *context,
                      cl_device_id *device_id) {
    /* No particular reason for 1e5. Just felt like that was a good amount */
    assert(op_num > 0 && op_num < 1e5);
    assert(tensor_num > 1);
    assert(command_queue && *command_queue);
    assert(context && *context);
    assert(device_id && *device_id);
    linearized_t linearized = linearized_alloc();
    linearized_t linearized_d = linearized_alloc();
    tensor_t *tensor = calloc(tensor_num, sizeof(tensor_t));
    tensor_t *tensor_d = calloc(tensor_num, sizeof(tensor_t));
    for(int64_t i = 0; i < tensor_num; i++) {
        tensor[i] = tensor_alloc(DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE, *context);
        tensor_unary_random(&tensor[i]);
        tensor_d[i] = tensor_alloc(DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE, *context);
        tensor_binary_copy(&tensor_d[i], &tensor[i]);
        tensor_realize(&tensor_d[i]);
    }

    /* TODO: I don't think there is a easy way to make sure that all these modulos are accurate, when adding new ops.
     * However I *really* should figure out a way to do that. */
    op_e op_type;
    unary_e type_unary;
    binary_e type_binary;
    reduce_e type_reduce;
    // enum move_e type_move;
    int64_t tensor_out = 0;
    int64_t tensor_in;
    double ran;
    int64_t sze_a = DIM_SZE;
    int64_t sze_z = DIM_SZE;
    int64_t sze_y = DIM_SZE;
    int64_t sze_x = DIM_SZE;
    int64_t off_a = 0;
    int64_t off_z = 0;
    int64_t off_y = 0;
    int64_t off_x = 0;
    int64_t a_temp = 0;
    int64_t z_temp = 0;
    int64_t y_temp = 0;
    int64_t x_temp = 0;
    for(int64_t i = 0; i < op_num; i++) {
        // op_type = rand() % 4;
        op_type = rand() % 3;
        switch(op_type) {
            case operation_unary: {
                type_unary = rand() % 16;
                ran = ((double) rand() / RAND_MAX) * 2 - 1;
                sze_a = rand() % DIM_SZE + 1;
                sze_z = rand() % DIM_SZE + 1;
                sze_y = rand() % DIM_SZE + 1;
                sze_x = rand() % DIM_SZE + 1;
                off_a = (DIM_SZE > sze_a) ? rand() % (1 + DIM_SZE - sze_a) : 0;
                off_z = (DIM_SZE > sze_z) ? rand() % (1 + DIM_SZE - sze_z) : 0;
                off_y = (DIM_SZE > sze_y) ? rand() % (1 + DIM_SZE - sze_y) : 0;
                off_x = (DIM_SZE > sze_x) ? rand() % (1 + DIM_SZE - sze_x) : 0;
                tensor_move_resize(tensor + tensor_out, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor + tensor_out, off_a, off_z, off_y, off_x);

                tensor_move_resize(tensor_d + tensor_out, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor_d + tensor_out, off_a, off_z, off_y, off_x);
                switch(type_unary) {
                    case unary_add: {
                        tensor_unary_add(&tensor[tensor_out], ran);

                        tensor_unary_add(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_subtract: {
                        tensor_unary_subtract(&tensor[tensor_out], ran);

                        tensor_unary_subtract(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_multiply: {
                        tensor_unary_multiply(&tensor[tensor_out], ran);

                        tensor_unary_multiply(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_divide: {
                        ran += 2; /* Avoid dividing by 0 */
                        tensor_unary_divide(&tensor[tensor_out], ran);

                        tensor_unary_divide(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_exp: {
                        tensor_unary_set(&tensor[tensor_out], ran);

                        tensor_unary_set(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_log: {
                        tensor_unary_absolute(&tensor[tensor_out]);
                        tensor_unary_add(&tensor[tensor_out], EPSILON); /* Avoid log of x <= 0 */
                        tensor_unary_log(&tensor[tensor_out]);

                        tensor_unary_absolute(&tensor_d[tensor_out]);
                        tensor_unary_add(&tensor_d[tensor_out], EPSILON); /* Avoid log of x <= 0 */
                        tensor_unary_log(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_square: {
                        tensor_unary_square(&tensor[tensor_out]);

                        tensor_unary_square(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_sqrt: {
                        tensor_unary_absolute(&tensor[tensor_out]); /* Avoid sqrt of x < 0 */
                        tensor_unary_sqrt(&tensor[tensor_out]);

                        tensor_unary_absolute(&tensor_d[tensor_out]); /* Avoid sqrt of x < 0 */
                        tensor_unary_sqrt(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_reciprocal: {
                        tensor_unary_absolute(&tensor[tensor_out]);
                        tensor_unary_add(&tensor[tensor_out],
                                         EPSILON); /* Avoid recip of x == 0. Has to be done by making all numbers > 0 */
                        tensor_unary_reciprocal(&tensor[tensor_out]);

                        tensor_unary_absolute(&tensor_d[tensor_out]);
                        tensor_unary_add(&tensor_d[tensor_out],
                                         EPSILON); /* Avoid recip of x == 0. Has to be done by making all numbers > 0 */
                        tensor_unary_reciprocal(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_max: {
                        tensor_unary_max(&tensor[tensor_out], ran);

                        tensor_unary_max(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_min: {
                        tensor_unary_min(&tensor[tensor_out], ran);

                        tensor_unary_min(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_set: {
                        tensor_unary_set(&tensor[tensor_out], ran);

                        tensor_unary_set(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_random: {
                        /* Just to `tanh` here cuz RNG isn't supported for OpenCL */
                    }
                    case unary_sign: {
                        /* This one will be suported but it isn't yet. TODO: Support it */
                    }
                    case unary_tanh: {
                        tensor_unary_tanh(&tensor[tensor_out]);

                        tensor_unary_tanh(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_absolute: {
                        tensor_unary_absolute(&tensor[tensor_out]);

                        tensor_unary_absolute(&tensor_d[tensor_out]);
                        break;
                    }
                }
                break;
            }
            case operation_binary: {
                type_binary = rand() % 14;
                for(int64_t rand_idx = 0; rand_idx < RANDOM_MAX_TRIES; rand_idx++) {
                    tensor_in = rand() % tensor_num;
                    if(tensor_in != tensor_out) { break; }
                }
                if(tensor_in == tensor_out) { ERROR("Got really unlucky here or there's a bug with `rand()`\n"); }
                sze_a = rand() % DIM_SZE + 1;
                sze_z = rand() % DIM_SZE + 1;
                sze_y = rand() % DIM_SZE + 1;
                sze_x = rand() % DIM_SZE + 1;
                off_a = (DIM_SZE > sze_a) ? rand() % (1 + DIM_SZE - sze_a) : 0;
                off_z = (DIM_SZE > sze_z) ? rand() % (1 + DIM_SZE - sze_z) : 0;
                off_y = (DIM_SZE > sze_y) ? rand() % (1 + DIM_SZE - sze_y) : 0;
                off_x = (DIM_SZE > sze_x) ? rand() % (1 + DIM_SZE - sze_x) : 0;
                tensor_move_resize(tensor + tensor_in, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor + tensor_in, off_a, off_z, off_y, off_x);

                tensor_move_resize(tensor_d + tensor_in, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor_d + tensor_in, off_a, off_z, off_y, off_x);
                off_a = (DIM_SZE > sze_a) ? rand() % (1 + DIM_SZE - sze_a) : 0;
                off_z = (DIM_SZE > sze_z) ? rand() % (1 + DIM_SZE - sze_z) : 0;
                off_y = (DIM_SZE > sze_y) ? rand() % (1 + DIM_SZE - sze_y) : 0;
                off_x = (DIM_SZE > sze_x) ? rand() % (1 + DIM_SZE - sze_x) : 0;
                tensor_move_resize(tensor + tensor_out, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor + tensor_out, off_a, off_z, off_y, off_x);

                tensor_move_resize(tensor_d + tensor_out, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor_d + tensor_out, off_a, off_z, off_y, off_x);
                switch(type_binary) {
                    case binary_add: {
                        tensor_binary_add(&tensor[tensor_out], &tensor[tensor_in]);

                        tensor_binary_add(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_subtract: {
                        tensor_binary_subtract(&tensor[tensor_out], &tensor[tensor_in]);

                        tensor_binary_subtract(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_multiply: {
                        tensor_binary_multiply(&tensor[tensor_out], &tensor[tensor_in]);

                        tensor_binary_multiply(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_divide: {
                        tensor_unary_absolute(&tensor[tensor_in]);
                        tensor_unary_add(
                            &tensor[tensor_in],
                            1); /* Prevent dividing by 0 and such small values that the next op could give an `inf`
                                 */
                        tensor_binary_divide(&tensor[tensor_out], &tensor[tensor_in]);

                        tensor_unary_absolute(&tensor_d[tensor_in]);
                        tensor_unary_add(
                            &tensor_d[tensor_in],
                            1); /* Prevent dividing by 0 and such small values that the next op could give an `inf`
                                 */
                        tensor_binary_divide(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_max: {
                        tensor_binary_max(&tensor[tensor_out], &tensor[tensor_in]);

                        tensor_binary_max(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_min: {
                        tensor_binary_min(&tensor[tensor_out], &tensor[tensor_in]);

                        tensor_binary_min(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_copy: {
                        tensor_binary_copy(&tensor[tensor_out], &tensor[tensor_in]);

                        tensor_binary_copy(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_add_like: {
                        tensor_binary_add(&tensor[tensor_out], &tensor[tensor_in]);

                        tensor_binary_add(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_subtract_like: {
                        tensor_binary_subtract(&tensor[tensor_out], &tensor[tensor_in]);

                        tensor_binary_subtract(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_multiply_like: {
                        tensor_binary_multiply(&tensor[tensor_out], &tensor[tensor_in]);

                        tensor_binary_multiply(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_divide_like: {
                        tensor_unary_absolute(&tensor[tensor_in]);
                        tensor_unary_add(
                            &tensor[tensor_in],
                            1); /* Prevent dividing by 0 and such small values that the next op could give an `inf`
                                 */
                        tensor_binary_divide(&tensor[tensor_out], &tensor[tensor_in]);

                        tensor_unary_absolute(&tensor_d[tensor_in]);
                        tensor_unary_add(
                            &tensor_d[tensor_in],
                            1); /* Prevent dividing by 0 and such small values that the next op could give an `inf`
                                 */
                        tensor_binary_divide(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_max_like: {
                        tensor_binary_max(&tensor[tensor_out], &tensor[tensor_in]);

                        tensor_binary_max(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_min_like: {
                        tensor_binary_min(&tensor[tensor_out], &tensor[tensor_in]);

                        tensor_binary_min(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_copy_like: {
                        tensor_binary_copy(&tensor[tensor_out], &tensor[tensor_in]);

                        tensor_binary_copy(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                }
                break;
            }
            case operation_reduce: {
                type_reduce = rand() % 4;
                for(int64_t rand_idx = 0; rand_idx < RANDOM_MAX_TRIES; rand_idx++) {
                    tensor_in = rand() % tensor_num;
                    if(tensor_in != tensor_out) { break; }
                }
                if(tensor_in == tensor_out) { ERROR("Got really unlucky here or there's a bug with `rand()`\n"); }
                off_a = (DIM_SZE > sze_a) ? rand() % (1 + DIM_SZE - sze_a) : 0;
                off_z = (DIM_SZE > sze_z) ? rand() % (1 + DIM_SZE - sze_z) : 0;
                off_y = (DIM_SZE > sze_y) ? rand() % (1 + DIM_SZE - sze_y) : 0;
                off_x = (DIM_SZE > sze_x) ? rand() % (1 + DIM_SZE - sze_x) : 0;
                a_temp = rand() % DIM_SZE;
                z_temp = rand() % DIM_SZE;
                y_temp = rand() % DIM_SZE;
                x_temp = rand() % DIM_SZE;
                tensor_move_resize(tensor + tensor_in, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor + tensor_in, off_a, off_z, off_y, off_x);
                tensor_move_resize(tensor + tensor_out, 1, 1, 1, 1);
                tensor_move_offset(tensor + tensor_out, a_temp, z_temp, y_temp, x_temp);

                tensor_move_resize(tensor_d + tensor_in, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor_d + tensor_in, off_a, off_z, off_y, off_x);
                tensor_move_resize(tensor_d + tensor_out, 1, 1, 1, 1);
                tensor_move_offset(tensor_d + tensor_out, a_temp, z_temp, y_temp, x_temp);
                switch(type_reduce) {
                    case reduce_sum: {
                        tensor_reduce_sum(tensor + tensor_out, tensor + tensor_in);

                        tensor_reduce_sum(tensor_d + tensor_out, tensor_d + tensor_in);
                        break;
                    }
                    case reduce_avg: {
                        tensor_reduce_avg(tensor + tensor_out, tensor + tensor_in);

                        tensor_reduce_avg(tensor_d + tensor_out, tensor_d + tensor_in);
                        break;
                    }
                    case reduce_min: {
                        tensor_reduce_min(tensor + tensor_out, tensor + tensor_in);

                        tensor_reduce_min(tensor_d + tensor_out, tensor_d + tensor_in);
                        break;
                    }
                    case reduce_max: {
                        tensor_reduce_max(tensor + tensor_out, tensor + tensor_in);

                        tensor_reduce_max(tensor_d + tensor_out, tensor_d + tensor_in);
                        break;
                    }
                }
                off_a = (DIM_SZE > sze_a) ? rand() % (1 + DIM_SZE - sze_a) : 0;
                off_z = (DIM_SZE > sze_z) ? rand() % (1 + DIM_SZE - sze_z) : 0;
                off_y = (DIM_SZE > sze_y) ? rand() % (1 + DIM_SZE - sze_y) : 0;
                off_x = (DIM_SZE > sze_x) ? rand() % (1 + DIM_SZE - sze_x) : 0;
                tensor_move_resize(tensor + tensor_out, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor + tensor_out, off_a, off_z, off_y, off_x);

                tensor_move_resize(tensor_d + tensor_out, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor_d + tensor_out, off_a, off_z, off_y, off_x);
                break;
            }
            case operation_move: {
                break;
            }
        }
    }
    linearized_from_op(&linearized, tensor[tensor_out].op);
    linearized_run(&linearized);
    linearized_from_op(&linearized_d, tensor_d[tensor_out].op);

    program_t program = {0};
    program_compile(&program, &linearized_d, device_id, context, command_queue);
    for(int64_t i = 0; i < tensor_num; i++) { buffer_sync_realize(tensor_d[i].buffer, *command_queue); }
    clFinish(*command_queue);
    program_run(&program);
    for(int64_t i = 0; i < tensor_num; i++) {
        buffer_sync_update(tensor_d[i].buffer, sync_to_host);
        buffer_sync_realize(tensor_d[i].buffer, *command_queue);
    }
    clFinish(*command_queue);

    for(int64_t i = 0; i < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; i++) {
        if(isnan(tensor[tensor_out].buffer->val[i])) {
            printf("out t %lf\n", tensor[tensor_out].buffer->val[i]);
            exit(1);
        }
        if(isnan(tensor_d[tensor_out].buffer->val[i])) {
            printf("out d %lf\n", tensor_d[tensor_out].buffer->val[i]);
            exit(1);
        }
        if(isinf(tensor[tensor_out].buffer->val[i])) {
            printf("out t %lf\n", tensor[tensor_out].buffer->val[i]);
            exit(1);
        }
        if(isinf(tensor_d[tensor_out].buffer->val[i])) {
            printf("out d %lf\n", tensor_d[tensor_out].buffer->val[i]);
            exit(1);
        }
        if(fabs((tensor[tensor_out].buffer->val[i] - tensor_d[tensor_out].buffer->val[i])) >= (MARGIN_OF_ERROR) &&
           fabs((tensor[tensor_out].buffer->val[i] / tensor_d[tensor_out].buffer->val[i]) - 1) >= (MARGIN_OF_ERROR)) {
            printf("%lf %lf\n", tensor[tensor_out].buffer->val[i], tensor_d[tensor_out].buffer->val[i]);
            printf("%lf >= %lf\n", fabs((tensor[tensor_out].buffer->val[i] / tensor_d[tensor_out].buffer->val[i]) - 1),
                   MARGIN_OF_ERROR);
            exit(1);
        }
    }

    for(int64_t i = 0; i < tensor_num; i++) {
        linearized_from_op(&linearized, tensor[i].op);
        tensor_free(&tensor[i]);
        linearized_from_op(&linearized_d, tensor_d[i].op);
        tensor_free(&tensor_d[i]);
    }
    free(tensor);
    free(tensor_d);
    linearized_free(&linearized);
    linearized_free(&linearized_d);
    program_free_non_reusable(&program);
}

int main(int argc, char **argv) {
    if(argc != 4) {
        printf("USAGE: %s [number of ops] [number of tensors] [number of iterations]\n", argv[0]);
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
    for(int64_t iter_idx = 0; iter_idx < iter_num; iter_idx++) {
        simulate_compile(op_num, tensor_num, &command_queue, &context, &device_id);
    }
    printf("Passed compiler simulation with %lu ops, %lu tensors, and %lu iteratations!\n", op_num, tensor_num,
           iter_num);
    return 0;
}
