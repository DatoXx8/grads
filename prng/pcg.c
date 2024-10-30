#include "pcg.h"
#include <assert.h>
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
    return rotate32((uint32_t) (x >> 27), pivot);
}

void pcg_init(const uint64_t init) {
    state = init;
}

/* Just doing modulo gives slightly biases results which is why this function exists. It is taken from
 * https://www.pcg-random.org/posts/bounded-rands.html */
/* This gives results in [0, top) */
uint32_t pcg_rand_below(const uint32_t top) {
    assert(top != 0);
    if(top == 1) {
        return 0;
    }
    uint32_t x = pcg_rand();
    uint64_t m = (uint64_t) x * (uint64_t) top;
    uint32_t l = (uint32_t) m;
    if(l < top) {
        uint32_t t = -top;
        if(t >= top) {
            t -= top;
            if(t >= top) {
                t %= top;
            }
        }
        while(l < t) {
            x = pcg_rand();
            m = (uint64_t) x * (uint64_t) top;
            l = (uint32_t) m;
        }
    }
    return m >> 32;
}
