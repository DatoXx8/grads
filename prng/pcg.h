#ifndef PRNG_PCG_H_
#define PRNG_PCG_H_

#include <stdint.h>

/* TODO: Make a 128 -> 64 bit version that is opt-in at compile time with some compiler flag like -DPCG128*/

/*
For details see http://www.pcg-random.org/

This is an implementation of PCG that uses 64 bits of state and outputs 32 bit values.
The values for the multiplier and increment are from the Wikipedia article at
https://en.wikipedia.org/wiki/Permuted_congruential_generator. I tested these Values using PractRand for about 8 hours
(4 TB generated) and there were no statistical anomalies found. Keep in mind that the stdlib rand() fails almost
instantly. WARNING: Obviously this is not for suitable for cryptography. Never ever use this for cryptographic purposes.
*/

extern void pcg_init(const uint64_t init);
extern uint32_t pcg_rand(void);

#endif
