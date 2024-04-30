#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tensor.h"

const int64_t DIM_SIZE = 10;
const double MARGIN_OF_ERROR = 1e-5;
int main(void) {
    const uint32_t seed = time(NULL);
    printf("RNG Seed %u\n", seed);
    srand(seed);

    double *data_in = calloc(DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE, sizeof(double));
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        data_in[element_idx] = ((double) rand() / RAND_MAX) * 2 - 1;
    }
    double *data_out = calloc(DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE, sizeof(double));
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        data_out[element_idx] = ((double) rand() / RAND_MAX) * 2 - 1;
    }

    tensor_t in = tensor_alloc(DIM_SIZE, DIM_SIZE, DIM_SIZE, DIM_SIZE);
    tensor_t out = tensor_alloc(DIM_SIZE, DIM_SIZE, DIM_SIZE, DIM_SIZE);

    /* Unary ops. */
    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_add_unary(&in, 1);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] + 1);
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_subtract_unary(&in, 1);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] - 1);
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_multiply_unary(&in, 2);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] * 2);
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_divide_unary(&in, 2);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] / 2);
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_exp_unary(&in);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == exp(data_in[element_idx]));
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_absolute_unary(&in); /* NOTE: Needed to deal with negative values. */
    tensor_log_unary(&in);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == log(fabs(data_in[element_idx])));
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_square_unary(&in);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] * data_in[element_idx]);
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_absolute_unary(&in); /* NOTE: Needed to deal with negative values. */
    tensor_sqrt_unary(&in);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == sqrt(fabs(data_in[element_idx])));
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_reciprocal_unary(&in);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == 1 / data_in[element_idx]);
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_random_unary(&in);
    tensor_cpu_realize(&in);
    double temp = 0;
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) { temp *= in.buffer->val[element_idx]; }
    assert(fabs(temp) <= MARGIN_OF_ERROR);

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_tanh_unary(&in);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == tanh(data_in[element_idx]));
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_max_unary(&in, 0.5);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) { assert(in.buffer->val[element_idx] >= 0.5); }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_min_unary(&in, 0.5);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) { assert(in.buffer->val[element_idx] <= 0.5); }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_absolute_unary(&in);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == fabs(data_in[element_idx]));
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_sign_unary(&in);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        if(data_in[element_idx] < 0) {
            assert(in.buffer->val[element_idx] == -1);
        } else if(data_in[element_idx] > 0) {
            assert(in.buffer->val[element_idx] == 1);
        } else {
            assert(in.buffer->val[element_idx] == 0);
        }
    }

    /* Binary ops. */
    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_add_binary(&in, &out);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] + data_out[element_idx]);
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_subtract_binary(&in, &out);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] - data_out[element_idx]);
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_multiply_binary(&in, &out);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] * data_out[element_idx]);
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_divide_binary(&in, &out);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] / data_out[element_idx]);
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_max_binary(&in, &out);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        if(data_in[element_idx] > data_out[element_idx]) {
            assert(in.buffer->val[element_idx] == data_in[element_idx]);
        } else {
            assert(in.buffer->val[element_idx] == data_out[element_idx]);
        }
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_min_binary(&in, &out);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        if(data_in[element_idx] < data_out[element_idx]) {
            assert(in.buffer->val[element_idx] == data_in[element_idx]);
        } else {
            assert(in.buffer->val[element_idx] == data_out[element_idx]);
        }
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_copy_binary(&in, &out);
    tensor_cpu_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_out[element_idx]);
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_resize_move(&out, 1, 1, 1, 1);
    tensor_add_like_binary(&in, &out);
    tensor_resize_move(&out, DIM_SIZE, DIM_SIZE, DIM_SIZE, DIM_SIZE);
    tensor_cpu_realize(&in);
    tensor_cpu_realize(&out);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] + data_out[0]);
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_resize_move(&out, 1, 1, 1, 1);
    tensor_subtract_like_binary(&in, &out);
    tensor_resize_move(&out, DIM_SIZE, DIM_SIZE, DIM_SIZE, DIM_SIZE);
    tensor_cpu_realize(&in);
    tensor_cpu_realize(&out);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] - data_out[0]);
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_resize_move(&out, 1, 1, 1, 1);
    tensor_multiply_like_binary(&in, &out);
    tensor_resize_move(&out, DIM_SIZE, DIM_SIZE, DIM_SIZE, DIM_SIZE);
    tensor_cpu_realize(&in);
    tensor_cpu_realize(&out);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] * data_out[0]);
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_resize_move(&out, 1, 1, 1, 1);
    tensor_divide_like_binary(&in, &out);
    tensor_resize_move(&out, DIM_SIZE, DIM_SIZE, DIM_SIZE, DIM_SIZE);
    tensor_cpu_realize(&in);
    tensor_cpu_realize(&out);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] / data_out[0]);
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_resize_move(&out, 1, 1, 1, 1);
    tensor_max_like_binary(&in, &out);
    tensor_resize_move(&out, DIM_SIZE, DIM_SIZE, DIM_SIZE, DIM_SIZE);
    tensor_cpu_realize(&in);
    tensor_cpu_realize(&out);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        if(data_in[element_idx] > data_out[0]) {
            assert(in.buffer->val[element_idx] == data_in[element_idx]);
        } else {
            assert(in.buffer->val[element_idx] == data_out[0]);
        }
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_resize_move(&out, 1, 1, 1, 1);
    tensor_min_like_binary(&in, &out);
    tensor_resize_move(&out, DIM_SIZE, DIM_SIZE, DIM_SIZE, DIM_SIZE);
    tensor_cpu_realize(&in);
    tensor_cpu_realize(&out);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        if(data_in[element_idx] < data_out[0]) {
            assert(in.buffer->val[element_idx] == data_in[element_idx]);
        } else {
            assert(in.buffer->val[element_idx] == data_out[0]);
        }
    }

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_resize_move(&out, 1, 1, 1, 1);
    tensor_copy_like_binary(&in, &out);
    tensor_resize_move(&out, DIM_SIZE, DIM_SIZE, DIM_SIZE, DIM_SIZE);
    tensor_cpu_realize(&in);
    tensor_cpu_realize(&out);
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) { assert(in.buffer->val[element_idx] == data_out[0]); }

    /* Reduce ops. */
    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_resize_move(&in, 1, 1, 1, 1);
    tensor_sum_reduce(&in, &out);
    tensor_resize_move(&in, DIM_SIZE, DIM_SIZE, DIM_SIZE, DIM_SIZE);
    tensor_cpu_realize(&in);
    temp = 0;
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) { temp += data_out[element_idx]; }
    assert(in.buffer->val[0] == temp);

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_resize_move(&in, 1, 1, 1, 1);
    tensor_avg_reduce(&in, &out);
    tensor_resize_move(&in, DIM_SIZE, DIM_SIZE, DIM_SIZE, DIM_SIZE);
    tensor_cpu_realize(&in);
    temp = 0;
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) { temp += data_out[element_idx]; }
    assert(in.buffer->val[0] == temp / ((double) DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE));

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_resize_move(&in, 1, 1, 1, 1);
    tensor_max_reduce(&in, &out);
    tensor_resize_move(&in, DIM_SIZE, DIM_SIZE, DIM_SIZE, DIM_SIZE);
    tensor_cpu_realize(&in);
    temp = -INFINITY;
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        temp = temp < data_out[element_idx] ? data_out[element_idx] : temp;
    }
    assert(in.buffer->val[0] == temp);

    memcpy(in.buffer->val, data_in, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE * sizeof(double));
    tensor_resize_move(&in, 1, 1, 1, 1);
    tensor_min_reduce(&in, &out);
    tensor_resize_move(&in, DIM_SIZE, DIM_SIZE, DIM_SIZE, DIM_SIZE);
    tensor_cpu_realize(&in);
    temp = INFINITY;
    for(int64_t element_idx = 0; element_idx < DIM_SIZE * DIM_SIZE * DIM_SIZE * DIM_SIZE; element_idx++) {
        temp = temp > data_out[element_idx] ? data_out[element_idx] : temp;
    }
    assert(in.buffer->val[0] == temp);

    printf("Passed all unit tests for the elementary operations!\n");
    free(data_in);
    free(data_out);
    tensor_free(&in);
    tensor_free(&out);
    return 0;
}
