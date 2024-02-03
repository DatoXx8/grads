#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include "tensor.h"

int main(void) {
    const uint64_t rng = time(NULL);
    printf("RNG Seed %lu\n", rng);
    srand(rng);

    return(0);
}
