/* Bunch of small macros and the like. */
#ifndef UTILS_H_
#define UTILS_H_

#include <time.h>
#include <math.h>
#define ALWAYS_INLINE inline __attribute__((always_inline))

#define INIT_TIMER \
struct timespec start = {0}; \
struct timespec stop = {0}; \

#define START_TIME clock_gettime(0, &start);
#define STOP_TIME clock_gettime(0, &stop);

#define PRINT_TIME \
printf("TIME: %.9lfs\n", (stop.tv_sec - start.tv_sec) + (double) (stop.tv_nsec - start.tv_nsec) / (double) (1e9)); \
printf("TIME:   %*smmmµµµnnn\n", (int) (floor(log10(stop.tv_sec - start.tv_sec))), ""); \

#endif
