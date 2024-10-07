#include "utils.h"
#include <assert.h>
#include <stdint.h>
#include <time.h>

/* Arbitrary limits */
#define MAX_TIMERS 100
struct timespec times[MAX_TIMERS] = {0};
void time_ns_store(const uint64_t id) {
    assert(id < MAX_TIMERS);
    timespec_get(&times[id], CLOCK_MONOTONIC);
}
uint64_t time_ns_load(const uint64_t id) {
    assert(id < MAX_TIMERS);
    /* Implies reading from a timer which hasn't started */
    assert(times[id].tv_sec);
    return 1e9 * times[id].tv_sec + times[id].tv_nsec;
}
