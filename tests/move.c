#include <stdio.h>
#include <assert.h>

#include "../tensor.h"

int main(void) {

    const uint64_t a_size = 2;
    const uint64_t z_size = 3;
    const uint64_t y_size = 4;
    const uint64_t x_size = 5;

    tensor_t in = tensor_alloc(a_size, z_size, y_size, x_size);
    tensor_t out = tensor_alloc(a_size, z_size, y_size, x_size);

    printf("Move reshape ");
    tensor_random_unary(&in);
    tensor_cpu_realize(&in);
    tensor_copy_binary(&out, &in);
    tensor_cpu_realize(&out);
    tensor_reshape_move(&out, x_size, y_size, z_size, a_size);
    tensor_cpu_realize(&out);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == out.buffer->values[i]);
    }
    assert(out.buffer->a_size == x_size);
    assert(out.buffer->z_size == y_size);
    assert(out.buffer->y_size == z_size);
    assert(out.buffer->x_size == a_size);
    printf("passed!\n");
    tensor_reshape_move(&out, a_size, z_size, y_size, x_size);
    tensor_cpu_realize(&out);

    printf("Move resize ");
    tensor_resize_move(&out, a_size - 1, z_size - 1, y_size - 1, x_size - 1);
    tensor_cpu_realize(&out);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == out.buffer->values[i]);
    }
    assert(out.buffer->a_size == a_size - 1);
    assert(out.buffer->z_size == z_size - 1);
    assert(out.buffer->y_size == y_size - 1);
    assert(out.buffer->x_size == x_size - 1);
    for(uint64_t a = 0; a < a_size - 1; a++) {
        for(uint64_t z = 0; z < z_size - 1; z++) {
            for(uint64_t y = 0; y < y_size - 1; y++) {
                for(uint64_t x = 0; x < x_size - 1; x++) {
                    assert(BUFFER_AT_(in.buffer, a, z, y, x) == BUFFER_AT_(out.buffer, a, z, y, x));
                }
            }
        }
    }
    printf("passed!\n");

    printf("Move offset ");
    tensor_offset_move(&out, 1, 1, 1, 1);
    tensor_cpu_realize(&out);
    for(uint64_t i = 0; i < a_size * z_size * y_size * x_size; i++) {
        assert(in.buffer->values[i] == out.buffer->values[i]);
    }
    assert(out.buffer->a_size == a_size - 1);
    assert(out.buffer->z_size == z_size - 1);
    assert(out.buffer->y_size == y_size - 1);
    assert(out.buffer->x_size == x_size - 1);
    for(uint64_t a = 0; a < a_size - 1; a++) {
        for(uint64_t z = 0; z < z_size - 1; z++) {
            for(uint64_t y = 0; y < y_size - 1; y++) {
                for(uint64_t x = 0; x < x_size - 1; x++) {
                    assert(BUFFER_AT_(in.buffer, a + 1, z + 1, y + 1, x + 1) == BUFFER_AT_(out.buffer, a, z, y, x));
                }
            }
        }
    }
    printf("passed!\n");

    return(0);
}
