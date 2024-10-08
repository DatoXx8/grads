#include "pcg.h"
#include <stdint.h>

uint64_t state = 0;
const uint64_t mult = 6364136223846793005;
const uint64_t incr = 1442695040888963407;

static uint32_t rotate32(uint32_t x, uint32_t pivot) {
    return x >> pivot | x << (-pivot & 31);
}

uint32_t pcg_rand(void) {
    uint64_t x = state;
    const uint32_t pivot = x >> 59;

    state = state * mult + incr;
    x ^= x >> 18;
    return rotate32((uint32_t)(x >> 27), pivot);
}
