#include <stdio.h>
#include <assert.h>

#include "../tensor.h"
// #include "../utils.h"

int main(void) {

    const uint64_t a_size = 2;
    const uint64_t z_size = 3;
    const uint64_t y_size = 4;
    const uint64_t x_size = 5;

    tensor_t in = tensor_alloc(a_size, z_size, y_size, x_size);
    tensor_t out = tensor_alloc(a_size, z_size, y_size, x_size);

    printf("Unary set ");
    tensor_set_unary(&in, 1);
    tensor_cpu_realize(&in);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == 1);
    }
    printf("passed!\n");

    printf("Unary add ");
    tensor_set_unary(&in, 1);
    tensor_add_unary(&in, 2);
    tensor_cpu_realize(&in);
    tensor_set_unary(&out, 3);
    tensor_cpu_realize(&out);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == out.buffer->values[i]);
    }
    printf("passed!\n");

    printf("Unary subtract ");
    tensor_set_unary(&in, 1);
    tensor_subtract_unary(&in, 2);
    tensor_cpu_realize(&in);
    tensor_set_unary(&out, -1);
    tensor_cpu_realize(&out);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == out.buffer->values[i]);
    }
    printf("passed!\n");

    printf("Unary multiply ");
    tensor_set_unary(&in, 2);
    tensor_multiply_unary(&in, 3);
    tensor_cpu_realize(&in);
    tensor_set_unary(&out, 6);
    tensor_cpu_realize(&out);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == out.buffer->values[i]);
    }
    printf("passed!\n");

    printf("Unary divide ");
    tensor_set_unary(&in, 1);
    tensor_divide_unary(&in, 2);
    tensor_cpu_realize(&in);
    tensor_set_unary(&out, 0.5);
    tensor_cpu_realize(&out);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == out.buffer->values[i]);
    }
    printf("passed!\n");

    printf("Unary exp ");
    tensor_set_unary(&in, 0);
    tensor_exp_unary(&in);
    tensor_cpu_realize(&in);
    tensor_set_unary(&out, 1);
    tensor_cpu_realize(&out);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == out.buffer->values[i]);
    }
    printf("passed!\n");

    printf("Unary log ");
    tensor_set_unary(&in, exp(1));
    tensor_log_unary(&in);
    tensor_cpu_realize(&in);
    tensor_set_unary(&out, 1);
    tensor_cpu_realize(&out);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == out.buffer->values[i]);
    }
    printf("passed!\n");

    printf("Unary square ");
    tensor_set_unary(&in, 3);
    tensor_square_unary(&in);
    tensor_cpu_realize(&in);
    tensor_set_unary(&out, 9);
    tensor_cpu_realize(&out);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == out.buffer->values[i]);
    }
    printf("passed!\n");

    printf("Unary sqrt ");
    tensor_set_unary(&in, 4);
    tensor_sqrt_unary(&in);
    tensor_cpu_realize(&in);
    tensor_set_unary(&out, 2);
    tensor_cpu_realize(&out);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == out.buffer->values[i]);
    }
    printf("passed!\n");

    printf("Unary negate ");
    tensor_set_unary(&in, 1);
    tensor_negate_unary(&in);
    tensor_cpu_realize(&in);
    tensor_set_unary(&out, -1);
    tensor_cpu_realize(&out);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == out.buffer->values[i]);
    }
    printf("passed!\n");

    printf("Unary reciprocal ");
    tensor_set_unary(&in, 2);
    tensor_reciprocal_unary(&in);
    tensor_cpu_realize(&in);
    tensor_set_unary(&out, 0.5);
    tensor_cpu_realize(&out);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == out.buffer->values[i]);
    }
    printf("passed!\n");

    printf("Unary random ");
    double product = 1;
    tensor_random_unary(&in);
    tensor_cpu_realize(&in);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        product *= in.buffer->values[i];
    }
    assert((product <= 0.001 && product >= -0.001) && "FIXME: Technicaly this one is random, meaning it could fail even if the code works, but the odds of having that happen are almost impossible. However if it still failed here try running it again.");
    printf("passed!\n");

    printf("Unary max ");
    tensor_random_unary(&in);
    tensor_max_unary(&in, 0);
    tensor_cpu_realize(&in);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] >= 0);
    }
    printf("passed!\n");

    printf("Unary min ");
    tensor_random_unary(&in);
    tensor_min_unary(&in, 0);
    tensor_cpu_realize(&in);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] <= 0);
    }
    printf("passed!\n");



    return(0);
}
