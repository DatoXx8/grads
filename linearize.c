#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "tensor.h"
#include "linearize.h"
#include "utils.h"

ALWAYS_INLINE void simple_op_convert(simple_op_t *simple_op, op_t *op) {
    simple_op->type = op->type;
    simple_op->unary_type = op->unary_type;
    simple_op->binary_type = op->binary_type;
    simple_op->reduce_type = op->reduce_type;
    simple_op->move_type = op->move_type;
    simple_op->var_unary = op->var_unary;
    simple_op->var_a = op->var_a;
    simple_op->var_z = op->var_z;
    simple_op->var_y = op->var_y;
    simple_op->var_x = op->var_x;
    simple_op->out_buffer = op->out_buffer;
    simple_op->in_buffer = op->in_buffer;
}
void simple_op_print(simple_op_t *simple_op, int padding, int offset, const char *name) {
    if(strcmp(name, "") != 0) {
        printf("%*s%s\n", offset, "", name);
    }
    printf("%*s<%p> ", offset + padding, "", (void *) simple_op);
    switch(simple_op->type) {
        case(operation_unary): {
            switch(simple_op->unary_type) {
                case(unary_add): {
                    printf("U add [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf [%p] %s\n", simple_op->out_buffer->a_inherent, simple_op->out_buffer->z_inherent, simple_op->out_buffer->y_inherent, simple_op->out_buffer->x_inherent, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, simple_op->var_unary, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(unary_subtract): {
                    printf("U sub [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf [%p] %s\n", simple_op->out_buffer->a_inherent, simple_op->out_buffer->z_inherent, simple_op->out_buffer->y_inherent, simple_op->out_buffer->x_inherent, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, simple_op->var_unary, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(unary_multiply): {
                    printf("U mul [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf [%p] %s\n", simple_op->out_buffer->a_inherent, simple_op->out_buffer->z_inherent, simple_op->out_buffer->y_inherent, simple_op->out_buffer->x_inherent, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, simple_op->var_unary, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(unary_divide): {
                    printf("U div [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf [%p] %s\n", simple_op->out_buffer->a_inherent, simple_op->out_buffer->z_inherent, simple_op->out_buffer->y_inherent, simple_op->out_buffer->x_inherent, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, simple_op->var_unary, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(unary_exp): {
                    printf("U exp [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p] %s\n", simple_op->out_buffer->a_inherent, simple_op->out_buffer->z_inherent, simple_op->out_buffer->y_inherent, simple_op->out_buffer->x_inherent, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(unary_log): {
                    printf("U log [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p] %s\n", simple_op->out_buffer->a_inherent, simple_op->out_buffer->z_inherent, simple_op->out_buffer->y_inherent, simple_op->out_buffer->x_inherent, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(unary_square): {
                    printf("U sqr [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p] %s\n", simple_op->out_buffer->a_inherent, simple_op->out_buffer->z_inherent, simple_op->out_buffer->y_inherent, simple_op->out_buffer->x_inherent, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(unary_sqrt): {
                    printf("U sqt [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p] %s\n", simple_op->out_buffer->a_inherent, simple_op->out_buffer->z_inherent, simple_op->out_buffer->y_inherent, simple_op->out_buffer->x_inherent, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(unary_negate): {
                    printf("U ngt [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p] %s\n", simple_op->out_buffer->a_inherent, simple_op->out_buffer->z_inherent, simple_op->out_buffer->y_inherent, simple_op->out_buffer->x_inherent, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(unary_reciprocal): {
                    printf("U rcp [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p] %s\n", simple_op->out_buffer->a_inherent, simple_op->out_buffer->z_inherent, simple_op->out_buffer->y_inherent, simple_op->out_buffer->x_inherent, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(unary_max): {
                    printf("U max [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf [%p] %s\n", simple_op->out_buffer->a_inherent, simple_op->out_buffer->z_inherent, simple_op->out_buffer->y_inherent, simple_op->out_buffer->x_inherent, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, simple_op->var_unary, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(unary_min): {
                    printf("U min [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf [%p] %s\n", simple_op->out_buffer->a_inherent, simple_op->out_buffer->z_inherent, simple_op->out_buffer->y_inherent, simple_op->out_buffer->x_inherent, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, simple_op->var_unary, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(unary_set): {
                    printf("U set [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu %lf [%p] %s\n", simple_op->out_buffer->a_inherent, simple_op->out_buffer->z_inherent, simple_op->out_buffer->y_inherent, simple_op->out_buffer->x_inherent, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, simple_op->var_unary, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(unary_zero): {
                    printf("U zer [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p] %s\n", simple_op->out_buffer->a_inherent, simple_op->out_buffer->z_inherent, simple_op->out_buffer->y_inherent, simple_op->out_buffer->x_inherent, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(unary_random): {
                    printf("U ran [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p] %s\n", simple_op->out_buffer->a_inherent, simple_op->out_buffer->z_inherent, simple_op->out_buffer->y_inherent, simple_op->out_buffer->x_inherent, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(unary_tanh): {
                    printf("U tnh [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p] %s\n", simple_op->out_buffer->a_inherent, simple_op->out_buffer->z_inherent, simple_op->out_buffer->y_inherent, simple_op->out_buffer->x_inherent, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(unary_absolute): {
                    printf("U abs [%lu, %lu, %lu, %lu] > {%lu, %lu, %lu, %lu} %lu [%p] %s\n", simple_op->out_buffer->a_inherent, simple_op->out_buffer->z_inherent, simple_op->out_buffer->y_inherent, simple_op->out_buffer->x_inherent, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
            }
            break;
        }
        case(operation_binary): {
            switch(simple_op->binary_type) {
                case(binary_add): {
                    printf("B add {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
                case(binary_subtract): {
                    printf("B sub {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
                case(binary_multiply): {
                    printf("B mul {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
                case(binary_divide): {
                    printf("B div {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
                case(binary_max): {
                    printf("B max {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
                case(binary_min): {
                    printf("B min {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
                case(binary_copy): {
                    printf("B cpy {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
                case(binary_add_like): {
                    printf("B ldd {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
                case(binary_subtract_like): {
                    printf("B lub {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
                case(binary_multiply_like): {
                    printf("B lul {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
                case(binary_divide_like): {
                    printf("B liv {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
                case(binary_max_like): {
                    printf("B lax {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
                case(binary_min_like): {
                    printf("B lin {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
                case(binary_copy_like): {
                    printf("B lpy {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
            }
            break;
        }
        case(operation_reduce): {
            switch(simple_op->reduce_type) {
                case(reduce_sum): {
                    printf("R sum {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
                case(reduce_avg): {
                    printf("R avg {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
                case(reduce_max): {
                    printf("R max {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
                case(reduce_min): {
                    printf("R min {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu [%p] [%p] %s %s\n", simple_op->in_buffer->a_size, simple_op->in_buffer->z_size, simple_op->in_buffer->y_size, simple_op->in_buffer->x_size, simple_op->in_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, (void *) simple_op->in_buffer, (void *) simple_op->out_buffer, simple_op->in_buffer->cl_name, simple_op->out_buffer->cl_name);
                    break;
                }
            }
            break;
        }
        case(operation_move): {
            switch(simple_op->move_type) {
                case(move_reshape): {
                    printf("M rsp {%lu, %lu, %lu, %lu} %lu - {%lu, %lu, %lu, %lu} %lu [%p] %s\n", simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, simple_op->var_a, simple_op->var_z, simple_op->var_y, simple_op->var_x, simple_op->out_buffer->offset, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(move_resize): {
                    printf("M rsz {%lu, %lu, %lu, %lu} %lu - {%lu, %lu, %lu, %lu} %lu [%p] %s\n", simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, simple_op->var_a, simple_op->var_z, simple_op->var_y, simple_op->var_x, simple_op->out_buffer->offset, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
                case(move_offset): {
                    printf("M off {%lu, %lu, %lu, %lu} %lu - {%lu, %lu, %lu, %lu} %lu [%p] %s\n", simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->offset, simple_op->out_buffer->a_size, simple_op->out_buffer->z_size, simple_op->out_buffer->y_size, simple_op->out_buffer->x_size, simple_op->out_buffer->a_stride * simple_op->var_a + simple_op->out_buffer->z_stride * simple_op->var_z + simple_op->out_buffer->y_stride * simple_op->var_y + simple_op->out_buffer->x_stride * simple_op->var_x, (void *) simple_op->out_buffer, simple_op->out_buffer->cl_name);
                    break;
                }
            }
            break;
        }
    }
}
void simple_op_realize(simple_op_t *simple_op) {
    switch(simple_op->type) {
        case(operation_unary): {
            switch(simple_op->unary_type) {
                case(unary_add): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) += simple_op->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_subtract): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) -= simple_op->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_multiply): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) *= simple_op->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_divide): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) /= simple_op->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_exp): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) = exp(BUFFER_AT_(simple_op->out_buffer, a, z, y ,x));
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_log): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) = log(BUFFER_AT_(simple_op->out_buffer, a, z, y ,x));
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_square): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) *= BUFFER_AT_(simple_op->out_buffer, a, z, y ,x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_sqrt): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) = sqrt(BUFFER_AT_(simple_op->out_buffer, a, z, y ,x));
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_negate): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) = - BUFFER_AT_(simple_op->out_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_reciprocal): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) = 1 / BUFFER_AT_(simple_op->out_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_max): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    if(BUFFER_AT_(simple_op->out_buffer, a, z, y, x) < simple_op->var_unary) {
                                        BUFFER_AT_(simple_op->out_buffer, a, z, y, x) = simple_op->var_unary;
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_min): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    if(BUFFER_AT_(simple_op->out_buffer, a, z, y, x) > simple_op->var_unary) {
                                        BUFFER_AT_(simple_op->out_buffer, a, z, y, x) = simple_op->var_unary;
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_set): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) = simple_op->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_zero): {
                    explicit_bzero(simple_op->out_buffer->values, simple_op->out_buffer->a_size * simple_op->out_buffer->z_size * simple_op->out_buffer->y_size * simple_op->out_buffer->x_size * sizeof(double));
                    break;
                }
                case(unary_random): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) = ((double) rand() / RAND_MAX) * 2 - 1;
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_tanh): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) = tanh(BUFFER_AT_(simple_op->out_buffer, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case(unary_absolute): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    if(BUFFER_AT_(simple_op->out_buffer, a, z, y, x) < 0) {
                                        BUFFER_AT_(simple_op->out_buffer, a, z, y, x) *= -1;
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
            }
            break;
        }
        case(operation_binary): {
            switch(simple_op->binary_type) {
                case(binary_add): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) += BUFFER_AT_(simple_op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_subtract): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) -= BUFFER_AT_(simple_op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_multiply): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) *= BUFFER_AT_(simple_op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_divide): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) /= BUFFER_AT_(simple_op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_max): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    if(BUFFER_AT_(simple_op->out_buffer, a, z, y, x) < BUFFER_AT_(simple_op->in_buffer, a, z, y, x)) {
                                        BUFFER_AT_(simple_op->out_buffer, a, z, y, x) = BUFFER_AT_(simple_op->in_buffer, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_min): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    if(BUFFER_AT_(simple_op->out_buffer, a, z, y, x) > BUFFER_AT_(simple_op->in_buffer, a, z, y, x)) {
                                        BUFFER_AT_(simple_op->out_buffer, a, z, y, x) = BUFFER_AT_(simple_op->in_buffer, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_copy): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) = BUFFER_AT_(simple_op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_add_like): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) += BUFFER_AT_(simple_op->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_subtract_like): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) -= BUFFER_AT_(simple_op->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_multiply_like): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) *= BUFFER_AT_(simple_op->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_divide_like): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) /= BUFFER_AT_(simple_op->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_max_like): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    if(BUFFER_AT_(simple_op->out_buffer, a, z, y, x) < BUFFER_AT_(simple_op->in_buffer, 0, 0, 0, 0)) {
                                        BUFFER_AT_(simple_op->out_buffer, a, z, y, x) = BUFFER_AT_(simple_op->in_buffer, 0, 0, 0, 0);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_min_like): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    if(BUFFER_AT_(simple_op->out_buffer, a, z, y, x) > BUFFER_AT_(simple_op->in_buffer, 0, 0, 0, 0)) {
                                        BUFFER_AT_(simple_op->out_buffer, a, z, y, x) = BUFFER_AT_(simple_op->in_buffer, 0, 0, 0, 0);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case(binary_copy_like): {
                    for(uint64_t a = 0; a < simple_op->out_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->out_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->out_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->out_buffer->x_size; x++) {
                                    BUFFER_AT_(simple_op->out_buffer, a, z, y, x) = BUFFER_AT_(simple_op->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
            }
            break;
        }
        case(operation_reduce): {
            switch(simple_op->reduce_type) {
                case(reduce_sum): {
                    double temp = 0;
                    for(uint64_t a = 0; a < simple_op->in_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->in_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->in_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->in_buffer->x_size; x++) {
                                    temp += BUFFER_AT_(simple_op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    BUFFER_AT_(simple_op->out_buffer, 0, 0, 0, 0) = temp;
                    break;
                }
                case(reduce_max): {
                    double temp = - INFINITY;
                    for(uint64_t a = 0; a < simple_op->in_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->in_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->in_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->in_buffer->x_size; x++) {
                                    if(temp < BUFFER_AT_(simple_op->in_buffer, a, z, y, x)) {
                                        temp = BUFFER_AT_(simple_op->in_buffer, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    BUFFER_AT_(simple_op->out_buffer, 0, 0, 0, 0) = temp;
                    break;
                }
                case(reduce_avg): {
                    double temp = 0;
                    for(uint64_t a = 0; a < simple_op->in_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->in_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->in_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->in_buffer->x_size; x++) {
                                    temp += BUFFER_AT_(simple_op->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    BUFFER_AT_(simple_op->out_buffer, 0, 0, 0, 0) = temp / (simple_op->in_buffer->x_size * simple_op->in_buffer->y_size * simple_op->in_buffer->z_size * simple_op->in_buffer->a_size);
                    break;
                }
                case(reduce_min): {
                    double temp = INFINITY;
                    for(uint64_t a = 0; a < simple_op->in_buffer->a_size; a++) {
                        for(uint64_t z = 0; z < simple_op->in_buffer->z_size; z++) {
                            for(uint64_t y = 0; y < simple_op->in_buffer->y_size; y++) {
                                for(uint64_t x = 0; x < simple_op->in_buffer->x_size; x++) {
                                    if(temp > BUFFER_AT_(simple_op->in_buffer, a, z, y, x)) {
                                        temp = BUFFER_AT_(simple_op->in_buffer, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    BUFFER_AT_(simple_op->out_buffer, 0, 0, 0, 0) = temp;
                    break;
                }
            }
            break;
        }
        case(operation_move): {
            switch(simple_op->move_type) {
                case(move_reshape): {
                    simple_op->out_buffer->a_size = simple_op->var_a;
                    simple_op->out_buffer->z_size = simple_op->var_z;
                    simple_op->out_buffer->y_size = simple_op->var_y;
                    simple_op->out_buffer->x_size = simple_op->var_x;
                    simple_op->out_buffer->a_stride = simple_op->var_z * simple_op->var_y * simple_op->var_x;
                    simple_op->out_buffer->z_stride = simple_op->var_y * simple_op->var_x;
                    simple_op->out_buffer->y_stride = simple_op->var_x;
                    simple_op->out_buffer->x_stride = 1;
                    break;
                }
                case(move_resize): {
                    simple_op->out_buffer->a_size = simple_op->var_a;
                    simple_op->out_buffer->z_size = simple_op->var_z;
                    simple_op->out_buffer->y_size = simple_op->var_y;
                    simple_op->out_buffer->x_size = simple_op->var_x;
                    break;
                }
                case(move_offset): {
                    simple_op->out_buffer->offset = simple_op->out_buffer->a_stride * simple_op->var_a + simple_op->out_buffer->z_stride * simple_op->var_z + simple_op->out_buffer->y_stride * simple_op->var_y + simple_op->out_buffer->x_stride * simple_op->var_x;
                    break;
                }
            }
            break;
        }
    }
}

/* NOTE: Completely made up value. Not tested at all. */
const uint64_t initial_simple_op_capactity = 100;
linearized_t linearized_alloc(void) {
    linearized_t linearized = {
        .op_count = 0,
        .op_capacity = initial_simple_op_capactity,
        .simple = calloc(initial_simple_op_capactity, sizeof(simple_op_t)),
    };
    assert(linearized.simple);

    return(linearized);
}
/* NOTE: Does `not` override the linearized ops instead appends ops. */
void linearized_from_op(linearized_t *linearized, op_t *op) {
    while(op->parent_count > 0) {
        linearized_from_op(linearized, op->parent[0]);
    }
    if(linearized->op_capacity == linearized->op_count) {
        linearized->op_capacity *= 2;
        linearized->simple = realloc(linearized->simple, linearized->op_capacity * sizeof(simple_op_t));
    }
    simple_op_convert(&linearized->simple[linearized->op_count++], op);
    op_cleanup(op);
    op_free(op);
    free(op);
}
void linearized_free(linearized_t *linearized) {
    free(linearized->simple);
}
void linearized_clear(linearized_t *linearized) {
    linearized->op_count = 0;
}

void linearized_print(linearized_t *linearized, int padding, int offset, const char *name) {
    if(strcmp(name, "") != 0) {
        printf("%*slen %lu, cap %lu %s\n", offset, "", linearized->op_count, linearized->op_capacity, name);
    } else {
        printf("%*slen %lu, cap %lu\n", offset, "", linearized->op_count, linearized->op_capacity);
    }
    /* NOTE: Kind of a nice allignment for printing */

    // uint64_t max = log10(linearized->op_count);
    // for(uint64_t i = 0; i < linearized->op_count; i++) {
    //     printf("%*s[%*s%lu] ", padding + offset, "", (int) max - (uint64_t) log10(i), "", i);
    //     simple_op_print(linearized->simple + i, 0, 0, "");
    // }
    
    /* This one is not alligned. */

    for(uint64_t i = 0; i < linearized->op_count; i++) {
        printf("%*s[%lu] ", padding + offset, "", i);
        simple_op_print(linearized->simple + i, 0, 0, "");
    }
}
