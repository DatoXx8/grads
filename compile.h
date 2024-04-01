#ifndef COMPILE_H_
#define COMPILE_H_

/* 
    1. Function recognizes loops in linearized_t (could also be of size 1, meaning a singular op)
    2. Function assigns loops to work groups
    3. Each loop gets split up into multiple work items
 */

#include <stdint.h>

#include "linearize.h"
#include "tensor.h"

/* NOTE: These each get compiled and are the most basic descriptors of compute in this framework. */
/* TODO: Indices shouldn't be stored I think. They should be computed via get_global_id() and stuff like that, as each loop has to be computed by the same kernel, which would be impossible if it was done with constant indices. */
typedef struct {
    char o_name[BUFFER_NAME_SIZE + 1];
    uint64_t o_index;
    char i_name[BUFFER_NAME_SIZE + 1];
    uint64_t i_index;
    enum operation_e type;
    enum unary_e unary_type;
    double unary_value;
    enum binary_e binary_type;
    enum reduce_e reduce_type;
} compile_op_t;

/* NOTE: This is solely a test function to see how to approach this. */
extern void compile_linearized_to_c(const char *filename, linearized_t *linearized);

#endif /* COMPILE_H_ */
