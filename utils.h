/* Bunch of small macros and the like */
#ifndef UTILS_H_
#define UTILS_H_

#include <math.h>
#include <time.h>
#define ALWAYS_INLINE inline __attribute__((always_inline))

/* TODO: Have different timers with `INIT_TIMER(name)` */
#define INIT_TIMER()                                                                                                   \
    struct timespec start = {0};                                                                                       \
    struct timespec stop = {0};

#define START_TIME() clock_gettime(CLOCK_REALTIME, &start);
#define STOP_TIME() clock_gettime(CLOCK_REALTIME, &stop);

#define PRINT_TIME(name)                                                                                               \
    printf("TIME: %.9lfs for " name "\n",                                                                             \
           ((double) stop.tv_sec + 1.0e-9 * stop.tv_nsec) - ((double) start.tv_sec + 1.0e-9 * start.tv_nsec));         \
    if((int) (((double) stop.tv_sec + 1.0e-9 * stop.tv_nsec) - ((double) start.tv_sec + 1.0e-9 * start.tv_nsec)) &&    \
       ((int) log10(((double) stop.tv_sec + 1.0e-9 * stop.tv_nsec) -                                                   \
                    ((double) start.tv_sec + 1.0e-9 * start.tv_nsec)))) {                                              \
        printf("TIME:   %*smmmµµµnnn\n",                                                                               \
               (int) log10(((double) stop.tv_sec + 1.0e-9 * stop.tv_nsec) -                                            \
                           ((double) start.tv_sec + 1.0e-9 * start.tv_nsec)),                                          \
               "");                                                                                                    \
    } else {                                                                                                           \
        printf("TIME:   mmmµµµnnn\n");                                                                                 \
    }

#define TODO()                                                                                                         \
    fprintf(stderr, "ERROR: Tried to execute not implemented feature at line %d in file %s\n", __LINE__, __FILE__);    \
    exit(1);

#define ERROR(...)                                                                                                     \
    fprintf(stderr, "ERROR: " __VA_ARGS__);                                                                            \
    exit(1);
#define WARN(...)                                                                                                      \
    if(getenv("LOGGER") && getenv("LOGGER")[0] >= '1') { fprintf(stderr, "WARN: " __VA_ARGS__); }
#define INFO(...)                                                                                                      \
    if(getenv("LOGGER") && getenv("LOGGER")[0] >= '2') { fprintf(stderr, "INFO: " __VA_ARGS__); }

#endif
