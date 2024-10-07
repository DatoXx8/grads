/* Bunch of small macros and the like */
#ifndef CGRAD_UTILS_H_
#define CGRAD_UTILS_H_

#include "math.h"
#include <stdint.h>

#define ALWAYS_INLINE inline __attribute__((always_inline))

extern void time_ns_store(const uint64_t id);
extern uint64_t time_ns_load(const uint64_t id);

#define PRINT_TIME(id, time_ns)                                                                                        \
    printf("TIME: %.9lus for " #id "\n", (time_ns));                                                                   \
    if((time_ns) && (int) log10(time_ns)) {                                                                            \
        printf("TIME:   %*smmmµµµnnn\n", (int) log10(time_ns), "");                                                    \
    } else {                                                                                                           \
        printf("TIME:   mmmµµµnnn\n");                                                                                 \
    }

#define UNREACHABLE()                                                                                                  \
    fprintf(stderr, "ERROR: Reached `UNREACHABLE()` in %s at line %d at %s %s\n", __FILE__, __LINE__, __DATE__,        \
            __TIME__);                                                                                                 \
    exit(1);
#define TODO()                                                                                                         \
    fprintf(stderr, "ERROR: Tried to execute not implemented feature at line %d in file %s\n", __LINE__, __FILE__);    \
    exit(1);
#define ERROR(...)                                                                                                     \
    fprintf(stderr, "ERROR: " __VA_ARGS__);                                                                            \
    exit(1);
#define WARN(...)                                                                                                      \
    if(getenv("LOGGER") && getenv("LOGGER")[0] >= '1') {                                                               \
        fprintf(stderr, "WARN: " __VA_ARGS__);                                                                         \
    }
#define INFO(...)                                                                                                      \
    if(getenv("LOGGER") && getenv("LOGGER")[0] >= '2') {                                                               \
        fprintf(stderr, "INFO: " __VA_ARGS__);                                                                         \
    }

#endif
