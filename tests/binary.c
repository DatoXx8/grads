#include <assert.h>
#include <stdio.h>

#include "../tensor.h"

int main(void) {

    const uint64_t a_size = 2;
    const uint64_t z_size = 3;
    const uint64_t y_size = 4;
    const uint64_t x_size = 5;

    tensor_t in = tensor_alloc(a_size, z_size, y_size, x_size);
    tensor_t out = tensor_alloc(a_size, z_size, y_size, x_size);

    printf("Binary add ");
    tensor_set_unary(&in, 1);
    tensor_set_unary(&out, 2);
    tensor_add_binary(&in, &out);
    tensor_cpu_realize(&in);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == 3);
    }
    printf("passed!\n");

    printf("Binary subtract ");
    tensor_set_unary(&in, 1);
    tensor_set_unary(&out, 2);
    tensor_subtract_binary(&in, &out);
    tensor_cpu_realize(&in);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == -1);
    }
    printf("passed!\n");

    printf("Binary multiply ");
    tensor_set_unary(&in, 2);
    tensor_set_unary(&out, 3);
    tensor_multiply_binary(&in, &out);
    tensor_cpu_realize(&in);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == 6);
    }
    printf("passed!\n");

    printf("Binary divide ");
    tensor_set_unary(&in, 1);
    tensor_set_unary(&out, 2);
    tensor_divide_binary(&in, &out);
    tensor_cpu_realize(&in);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == 0.5);
    }
    printf("passed!\n");

    printf("Binary max ");
    tensor_set_unary(&in, 1);
    tensor_set_unary(&out, 2);
    tensor_max_binary(&in, &out);
    tensor_cpu_realize(&in);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == 2);
    }
    printf("passed!\n");

    printf("Binary min ");
    tensor_set_unary(&in, 1);
    tensor_set_unary(&out, 2);
    tensor_min_binary(&in, &out);
    tensor_cpu_realize(&in);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == 1);
    }
    printf("passed!\n");

    printf("Binary copy ");
    tensor_set_unary(&in, 1);
    tensor_set_unary(&out, 2);
    tensor_copy_binary(&in, &out);
    tensor_cpu_realize(&in);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == 2);
    }
    printf("passed!\n");

    return(0);
}
