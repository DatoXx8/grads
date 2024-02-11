#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include "tensor.h"
#include "linearize.h"
#include "utils.h"

int main(void) {
    const uint64_t rng = time(NULL);
    printf("RNG Seed %lu\n", rng);
    srand(rng);
    INIT_TIMER;

    START_TIME;

    tensor_t in = tensor_alloc(1, 1, 1, 16);
    tensor_t out = tensor_alloc(1, 1, 1, 1);

    tensor_random_unary(&in);
    tensor_avg_reduce(&out, &in);
    linearized_t linearized = linearized_alloc();
    linearized_from_op(&linearized, out.op);
    // linearized_print(&linearized, 4, 0, "gamer");

    STOP_TIME;

    PRINT_TIME;
    return(0);
}
