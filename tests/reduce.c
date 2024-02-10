#include <stdio.h>
#include <assert.h>

#include "../tensor.h"

int main(void) {

    const uint64_t a_size = 2;
    const uint64_t z_size = 3;
    const uint64_t y_size = 4;
    const uint64_t x_size = 5;

    tensor_t in = tensor_alloc(a_size, z_size, y_size, x_size);
    tensor_t out = tensor_alloc(1, 1, 1, 1);

    printf("Reduce sum ");
    tensor_random_unary(&in);
    tensor_cpu_realize(&in);
    tensor_sum_reduce(&out, &in);
    tensor_cpu_realize(&out);
    double sum = 0;
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        sum += in.buffer->values[i];
    }
    assert(out.buffer->values[0] == sum);
    printf("passed!\n");

    printf("Reduce max ");
    tensor_random_unary(&in);
    tensor_cpu_realize(&in);
    tensor_max_reduce(&out, &in);
    tensor_cpu_realize(&out);
    double max = - INFINITY;
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        if(in.buffer->values[i] > max) {
            max = in.buffer->values[i];
        }
    }
    assert(out.buffer->values[0] == max);
    printf("passed!\n");

    printf("Reduce min ");
    tensor_random_unary(&in);
    tensor_cpu_realize(&in);
    tensor_min_reduce(&out, &in);
    tensor_cpu_realize(&out);
    double min = INFINITY;
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        if(in.buffer->values[i] < min) {
            min = in.buffer->values[i];
        }
    }
    assert(out.buffer->values[0] == min);
    printf("passed!\n");

    printf("Reduce avg ");
    tensor_random_unary(&in);
    tensor_cpu_realize(&in);
    tensor_avg_reduce(&out, &in);
    tensor_cpu_realize(&out);
    double avg = 0;
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        avg += in.buffer->values[i];
    }
    avg /= a_size * z_size * y_size * x_size;
    assert(out.buffer->values[0] == avg);
    printf("passed!\n");

    return(0);
}
