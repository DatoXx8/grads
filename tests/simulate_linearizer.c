#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../linearize.h"
#include "../tensor.h"

/* TODO: Validate equivalence between linearized and non-linearized fresh from the tree. */
const int64_t RANDOM_MAX_TRIES = 100;
const int64_t DIM_SIZE = 3;
void simulate_linearize(int64_t op_num, int64_t tensor_num) {
    linearized_t linearized = linearized_alloc();
    tensor_t *tensor = calloc(tensor_num, sizeof(tensor_t));
    for(int64_t i = 0; i < tensor_num; i++) { tensor[i] = tensor_alloc(DIM_SIZE, DIM_SIZE, DIM_SIZE, DIM_SIZE); }

    /* TODO: I don't think there is a easy way to make sure that all these modulos are accurate, when adding new ops. However I *really* should figure out a way
     * to do that. */
    enum operation_e op_type;
    enum unary_e unary_type;
    enum binary_e binary_type;
    enum reduce_e reduce_type;
    // enum move_e move_type;
    int64_t tensor_out = 0;
    int64_t tensor_in;
    double ran;
    int64_t a_size = DIM_SIZE;
    int64_t z_size = DIM_SIZE;
    int64_t y_size = DIM_SIZE;
    int64_t x_size = DIM_SIZE;
    int64_t a_off = 0;
    int64_t z_off = 0;
    int64_t y_off = 0;
    int64_t x_off = 0;
    for(int64_t i = 0; i < op_num; i++) {
        op_type = rand() % 4;
        switch(op_type) {
            case operation_unary: {
                unary_type = rand() % 16;
                ran = ((double) rand() / RAND_MAX) * 2 - 1;
                a_size = rand() % DIM_SIZE + 1;
                z_size = rand() % DIM_SIZE + 1;
                y_size = rand() % DIM_SIZE + 1;
                x_size = rand() % DIM_SIZE + 1;
                a_off = (DIM_SIZE > a_size) ? rand() % (1 + DIM_SIZE - a_size) : 0;
                z_off = (DIM_SIZE > z_size) ? rand() % (1 + DIM_SIZE - z_size) : 0;
                y_off = (DIM_SIZE > y_size) ? rand() % (1 + DIM_SIZE - y_size) : 0;
                x_off = (DIM_SIZE > x_size) ? rand() % (1 + DIM_SIZE - x_size) : 0;
                tensor_resize_move(tensor + tensor_out, a_size, z_size, y_size, x_size);
                tensor_offset_move(tensor + tensor_out, a_off, z_off, y_off, x_off);
                switch(unary_type) {
                    case unary_add: {
                        tensor_add_unary(&tensor[tensor_out], ran);
                        break;
                    }
                    case unary_subtract: {
                        tensor_subtract_unary(&tensor[tensor_out], ran);
                        break;
                    }
                    case unary_multiply: {
                        tensor_multiply_unary(&tensor[tensor_out], ran);
                        break;
                    }
                    case unary_divide: {
                        tensor_divide_unary(&tensor[tensor_out], ran);
                        break;
                    }
                    case unary_exp: {
                        tensor_exp_unary(&tensor[tensor_out]);
                        break;
                    }
                    case unary_log: {
                        tensor_log_unary(&tensor[tensor_out]);
                        break;
                    }
                    case unary_square: {
                        tensor_square_unary(&tensor[tensor_out]);
                        break;
                    }
                    case unary_sqrt: {
                        tensor_sqrt_unary(&tensor[tensor_out]);
                        break;
                    }
                    case unary_reciprocal: {
                        tensor_reciprocal_unary(&tensor[tensor_out]);
                        break;
                    }
                    case unary_max: {
                        tensor_max_unary(&tensor[tensor_out], ran);
                        break;
                    }
                    case unary_min: {
                        tensor_min_unary(&tensor[tensor_out], ran);
                        break;
                    }
                    case unary_set: {
                        tensor_set_unary(&tensor[tensor_out], ran);
                        break;
                    }
                    case unary_random: {
                        tensor_random_unary(&tensor[tensor_out]);
                        break;
                    }
                    case unary_tanh: {
                        tensor_tanh_unary(&tensor[tensor_out]);
                        break;
                    }
                    case unary_absolute: {
                        tensor_absolute_unary(&tensor[tensor_out]);
                        break;
                    }
                    case unary_sign: {
                        tensor_sign_unary(&tensor[tensor_out]);
                        break;
                    }
                }
                break;
            }
            case operation_binary: {
                binary_type = rand() % 7;
                for(int64_t rand_idx = 0; rand_idx < RANDOM_MAX_TRIES; rand_idx++) {
                    tensor_in = rand() % tensor_num;
                    if(tensor_in != tensor_out) { break; }
                }
                if(tensor_in == tensor_out) { ERROR("Got really unlucky here or there's a bug with `rand()`\n"); }
                a_size = rand() % DIM_SIZE + 1;
                z_size = rand() % DIM_SIZE + 1;
                y_size = rand() % DIM_SIZE + 1;
                x_size = rand() % DIM_SIZE + 1;
                a_off = (DIM_SIZE > a_size) ? rand() % (1 + DIM_SIZE - a_size) : 0;
                z_off = (DIM_SIZE > z_size) ? rand() % (1 + DIM_SIZE - z_size) : 0;
                y_off = (DIM_SIZE > y_size) ? rand() % (1 + DIM_SIZE - y_size) : 0;
                x_off = (DIM_SIZE > x_size) ? rand() % (1 + DIM_SIZE - x_size) : 0;
                tensor_resize_move(tensor + tensor_in, a_size, z_size, y_size, x_size);
                tensor_offset_move(tensor + tensor_in, a_off, z_off, y_off, x_off);
                a_off = (DIM_SIZE > a_size) ? rand() % (1 + DIM_SIZE - a_size) : 0;
                z_off = (DIM_SIZE > z_size) ? rand() % (1 + DIM_SIZE - z_size) : 0;
                y_off = (DIM_SIZE > y_size) ? rand() % (1 + DIM_SIZE - y_size) : 0;
                x_off = (DIM_SIZE > x_size) ? rand() % (1 + DIM_SIZE - x_size) : 0;
                tensor_resize_move(tensor + tensor_out, a_size, z_size, y_size, x_size);
                tensor_offset_move(tensor + tensor_out, a_off, z_off, y_off, x_off);
                switch(binary_type) {
                    case binary_add: {
                        tensor_add_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        break;
                    }
                    case binary_subtract: {
                        tensor_subtract_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        break;
                    }
                    case binary_multiply: {
                        tensor_multiply_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        break;
                    }
                    case binary_divide: {
                        tensor_divide_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        break;
                    }
                    case binary_max: {
                        tensor_max_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        break;
                    }
                    case binary_min: {
                        tensor_min_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        break;
                    }
                    case binary_copy: {
                        tensor_copy_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        break;
                    }
                    case binary_add_like: {
                        tensor_add_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        break;
                    }
                    case binary_subtract_like: {
                        tensor_subtract_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        break;
                    }
                    case binary_multiply_like: {
                        tensor_multiply_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        break;
                    }
                    case binary_divide_like: {
                        tensor_divide_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        break;
                    }
                    case binary_max_like: {
                        tensor_max_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        break;
                    }
                    case binary_min_like: {
                        tensor_min_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        break;
                    }
                    case binary_copy_like: {
                        tensor_copy_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        break;
                    }
                }
                break;
            }
            case operation_reduce: {
                reduce_type = rand() % 4;
                for(int64_t rand_idx = 0; rand_idx < RANDOM_MAX_TRIES; rand_idx++) {
                    tensor_in = rand() % tensor_num;
                    if(tensor_in != tensor_out) { break; }
                }
                if(tensor_in == tensor_out) { ERROR("Got really unlucky here or there's a bug with `rand()`\n"); }
                a_off = (DIM_SIZE > a_size) ? rand() % (1 + DIM_SIZE - a_size) : 0;
                z_off = (DIM_SIZE > z_size) ? rand() % (1 + DIM_SIZE - z_size) : 0;
                y_off = (DIM_SIZE > y_size) ? rand() % (1 + DIM_SIZE - y_size) : 0;
                x_off = (DIM_SIZE > x_size) ? rand() % (1 + DIM_SIZE - x_size) : 0;
                tensor_resize_move(tensor + tensor_in, a_size, z_size, y_size, x_size);
                tensor_offset_move(tensor + tensor_in, a_off, z_off, y_off, x_off);
                tensor_resize_move(tensor + tensor_out, 1, 1, 1, 1);
                tensor_offset_move(tensor + tensor_out, rand() % DIM_SIZE, rand() % DIM_SIZE, rand() % DIM_SIZE, rand() % DIM_SIZE);
                switch(reduce_type) {
                    case reduce_sum: {
                        tensor_sum_reduce(tensor + tensor_out, tensor + tensor_in);
                        break;
                    }
                    case reduce_avg: {
                        tensor_avg_reduce(tensor + tensor_out, tensor + tensor_in);
                        break;
                    }
                    case reduce_min: {
                        tensor_min_reduce(tensor + tensor_out, tensor + tensor_in);
                        break;
                    }
                    case reduce_max: {
                        tensor_max_reduce(tensor + tensor_out, tensor + tensor_in);
                        break;
                    }
                }
                a_off = (DIM_SIZE > a_size) ? rand() % (1 + DIM_SIZE - a_size) : 0;
                z_off = (DIM_SIZE > z_size) ? rand() % (1 + DIM_SIZE - z_size) : 0;
                y_off = (DIM_SIZE > y_size) ? rand() % (1 + DIM_SIZE - y_size) : 0;
                x_off = (DIM_SIZE > x_size) ? rand() % (1 + DIM_SIZE - x_size) : 0;
                tensor_resize_move(tensor + tensor_out, a_size, z_size, y_size, x_size);
                tensor_offset_move(tensor + tensor_out, a_off, z_off, y_off, x_off);
                break;
            }
            case operation_move: {
                break;
            }
        }
    }
    linearized_from_op(&linearized, tensor[tensor_out].op);
    linearized_run(&linearized);

    for(int64_t i = 0; i < tensor_num; i++) {
        linearized_from_op(&linearized, tensor[i].op);
        tensor_free(&tensor[i]);
    }
    free(tensor);
    linearized_free(&linearized);
}

int main(int argc, char **argv) {
    if(argc == 4) {
        const uint32_t seed = time(NULL);
        printf("RNG Seed %u\n", seed);
        srand(seed);
        const int64_t op_num = strtoll(argv[1], NULL, 10);
        const int64_t tensor_num = strtoll(argv[2], NULL, 10);
        const int64_t iter_num = strtoll(argv[3], NULL, 10);
        assert(op_num > 0);
        assert(tensor_num > 1);
        assert(iter_num > 0);
        for(int64_t iter_idx = 0; iter_idx < iter_num; iter_idx++) { simulate_linearize(op_num, tensor_num); }
        printf("Passed linearizer simulation with %lu ops, %lu tensors, and %lu iteratations!\n", op_num, tensor_num, iter_num);
        return 0;
    }
    printf("USAGE: %s [number of ops] [number of tensors] [number of iterations]\n", argv[0]);
    return 0;
}
