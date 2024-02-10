#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include "tensor.h"
#include "utils.h"

int main(void) {
    const uint64_t rng = time(NULL);
    printf("RNG Seed %lu\n", rng);
    srand(rng);
    INIT_TIMER;

    START_TIME;

    tensor_t tensor = tensor_alloc(1, 1, 1, 16);
    tensor_random_unary(&tensor);
    op_print(tensor.op, 4, 0, "tensor.op");
    tensor_cpu_realize(&tensor);
    TENSOR_PRINT(tensor);

    STOP_TIME;

    PRINT_TIME;
    return(0);
}
