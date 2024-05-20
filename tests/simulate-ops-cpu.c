#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tensor.h"

const int64_t DIM_SZE = 10;
const double MARGIN_OF_ERROR = 1e-5;
int main(void) {
    const uint32_t seed = time(NULL);
    printf("RNG Seed %u\n", seed);
    srand(seed);

    double *data_in = calloc(DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE, sizeof(double));
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        data_in[element_idx] = ((double) rand() / RAND_MAX) * 2 - 1;
    }
    double *data_out = calloc(DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE, sizeof(double));
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        data_out[element_idx] = ((double) rand() / RAND_MAX) * 2 - 1;
    }

    tensor_t in = tensor_alloc(DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE, NULL);
    tensor_t out = tensor_alloc(DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE, NULL);

    /* Unary ops. */
    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_unary_add(&in, 1);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] + 1);
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_unary_subtract(&in, 1);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] - 1);
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_unary_multiply(&in, 2);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] * 2);
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_unary_divide(&in, 2);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] / 2);
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_unary_exp(&in);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == exp(data_in[element_idx]));
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_unary_absolute(&in); /* Needed to deal with negative values. */
    tensor_unary_log(&in);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == log(fabs(data_in[element_idx])));
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_unary_square(&in);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] * data_in[element_idx]);
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_unary_absolute(&in); /* Needed to deal with negative values. */
    tensor_unary_sqrt(&in);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == sqrt(fabs(data_in[element_idx])));
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_unary_reciprocal(&in);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == 1 / data_in[element_idx]);
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_unary_random(&in);
    tensor_realize(&in);
    double temp = 0;
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        temp *= in.buffer->val[element_idx];
    }
    assert(fabs(temp) <= MARGIN_OF_ERROR);

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_unary_tanh(&in);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == tanh(data_in[element_idx]));
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_unary_max(&in, 0.5);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] >= 0.5);
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_unary_min(&in, 0.5);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] <= 0.5);
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_unary_absolute(&in);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == fabs(data_in[element_idx]));
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_unary_sign(&in);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        if(data_in[element_idx] < 0) {
            assert(in.buffer->val[element_idx] == -1);
        } else if(data_in[element_idx] > 0) {
            assert(in.buffer->val[element_idx] == 1);
        } else {
            assert(in.buffer->val[element_idx] == 0);
        }
    }

    /* Binary ops. */
    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_binary_add(&in, &out);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] + data_out[element_idx]);
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_binary_subtract(&in, &out);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] - data_out[element_idx]);
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_binary_multiply(&in, &out);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] * data_out[element_idx]);
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_binary_divide(&in, &out);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] / data_out[element_idx]);
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_binary_max(&in, &out);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        if(data_in[element_idx] > data_out[element_idx]) {
            assert(in.buffer->val[element_idx] == data_in[element_idx]);
        } else {
            assert(in.buffer->val[element_idx] == data_out[element_idx]);
        }
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_binary_min(&in, &out);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        if(data_in[element_idx] < data_out[element_idx]) {
            assert(in.buffer->val[element_idx] == data_in[element_idx]);
        } else {
            assert(in.buffer->val[element_idx] == data_out[element_idx]);
        }
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_binary_copy(&in, &out);
    tensor_realize(&in);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_out[element_idx]);
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_move_resize(&out, 1, 1, 1, 1);
    tensor_lbinary_add(&in, &out);
    tensor_move_resize(&out, DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE);
    tensor_realize(&in);
    tensor_realize(&out);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] + data_out[0]);
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_move_resize(&out, 1, 1, 1, 1);
    tensor_lbinary_subtract(&in, &out);
    tensor_move_resize(&out, DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE);
    tensor_realize(&in);
    tensor_realize(&out);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] - data_out[0]);
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_move_resize(&out, 1, 1, 1, 1);
    tensor_lbinary_multiply(&in, &out);
    tensor_move_resize(&out, DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE);
    tensor_realize(&in);
    tensor_realize(&out);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] * data_out[0]);
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_move_resize(&out, 1, 1, 1, 1);
    tensor_lbinary_divide(&in, &out);
    tensor_move_resize(&out, DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE);
    tensor_realize(&in);
    tensor_realize(&out);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_in[element_idx] / data_out[0]);
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_move_resize(&out, 1, 1, 1, 1);
    tensor_lbinary_max(&in, &out);
    tensor_move_resize(&out, DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE);
    tensor_realize(&in);
    tensor_realize(&out);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        if(data_in[element_idx] > data_out[0]) {
            assert(in.buffer->val[element_idx] == data_in[element_idx]);
        } else {
            assert(in.buffer->val[element_idx] == data_out[0]);
        }
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_move_resize(&out, 1, 1, 1, 1);
    tensor_lbinary_min(&in, &out);
    tensor_move_resize(&out, DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE);
    tensor_realize(&in);
    tensor_realize(&out);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        if(data_in[element_idx] < data_out[0]) {
            assert(in.buffer->val[element_idx] == data_in[element_idx]);
        } else {
            assert(in.buffer->val[element_idx] == data_out[0]);
        }
    }

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_move_resize(&out, 1, 1, 1, 1);
    tensor_lbinary_copy(&in, &out);
    tensor_move_resize(&out, DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE);
    tensor_realize(&in);
    tensor_realize(&out);
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        assert(in.buffer->val[element_idx] == data_out[0]);
    }

    /* Reduce ops. */
    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_move_resize(&in, 1, 1, 1, 1);
    tensor_reduce_sum(&in, &out);
    tensor_move_resize(&in, DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE);
    tensor_realize(&in);
    temp = 0;
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        temp += data_out[element_idx];
    }
    assert(in.buffer->val[0] == temp);

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_move_resize(&in, 1, 1, 1, 1);
    tensor_reduce_avg(&in, &out);
    tensor_move_resize(&in, DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE);
    tensor_realize(&in);
    temp = 0;
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        temp += data_out[element_idx];
    }
    assert(in.buffer->val[0] == temp / ((double) DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE));

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_move_resize(&in, 1, 1, 1, 1);
    tensor_reduce_max(&in, &out);
    tensor_move_resize(&in, DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE);
    tensor_realize(&in);
    temp = -INFINITY;
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
        temp = temp < data_out[element_idx] ? data_out[element_idx] : temp;
    }
    assert(in.buffer->val[0] == temp);

    memcpy(in.buffer->val, data_in, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    memcpy(out.buffer->val, data_out, DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE * sizeof(double));
    tensor_move_resize(&in, 1, 1, 1, 1);
    tensor_reduce_min(&in, &out);
    tensor_move_resize(&in, DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE);
    tensor_realize(&in);
    temp = INFINITY;
    for(int64_t element_idx = 0; element_idx < DIM_SZE * DIM_SZE * DIM_SZE * DIM_SZE; element_idx++) {
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
