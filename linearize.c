#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linearize.h"
#include "tensor.h"
#include "utils.h"

void ALWAYS_INLINE simple_op_simulate_move(op_t *op) {
    switch(op->move_type) {
        case move_reshape: {
            op->out_buffer->sim_a_sze = op->var_a;
            op->out_buffer->sim_z_sze = op->var_z;
            op->out_buffer->sim_y_sze = op->var_y;
            op->out_buffer->sim_x_sze = op->var_x;
            op->out_buffer->sim_a_str = op->var_z * op->var_y * op->var_x;
            op->out_buffer->sim_z_str = op->var_y * op->var_x;
            op->out_buffer->sim_y_str = op->var_x;
            op->out_buffer->sim_x_str = 1;
            break;
        }
        case move_resize: {
            op->out_buffer->sim_a_sze = op->var_a;
            op->out_buffer->sim_z_sze = op->var_z;
            op->out_buffer->sim_y_sze = op->var_y;
            op->out_buffer->sim_x_sze = op->var_x;
            break;
        }
        case move_offset: {
            op->out_buffer->sim_off = op->out_buffer->sim_a_str * op->var_a + op->out_buffer->sim_z_str * op->var_z +
                                        op->out_buffer->sim_y_str * op->var_y + op->out_buffer->sim_x_str * op->var_x;
            op->out_buffer->sim_a_off = op->var_a;
            op->out_buffer->sim_z_off = op->var_z;
            op->out_buffer->sim_y_off = op->var_y;
            op->out_buffer->sim_x_off = op->var_x;
            break;
        }
    }
}
void ALWAYS_INLINE simple_op_convert(simple_op_t *simple_op, op_t *op) {
    simple_op->type = op->type;
    simple_op->unary_type = op->unary_type;
    simple_op->binary_type = op->binary_type;
    simple_op->reduce_type = op->reduce_type;
    simple_op->var_unary = op->var_unary;
    simple_op->out_buffer.a_sze = op->out_buffer->sim_a_sze;
    simple_op->out_buffer.z_sze = op->out_buffer->sim_z_sze;
    simple_op->out_buffer.y_sze = op->out_buffer->sim_y_sze;
    simple_op->out_buffer.x_sze = op->out_buffer->sim_x_sze;
    simple_op->out_buffer.a_str = op->out_buffer->sim_a_str;
    simple_op->out_buffer.z_str = op->out_buffer->sim_z_str;
    simple_op->out_buffer.y_str = op->out_buffer->sim_y_str;
    simple_op->out_buffer.x_str = op->out_buffer->sim_x_str;
    simple_op->out_buffer.off = op->out_buffer->sim_off;
    simple_op->out_buffer.a_off = op->out_buffer->sim_a_off;
    simple_op->out_buffer.z_off = op->out_buffer->sim_z_off;
    simple_op->out_buffer.y_off = op->out_buffer->sim_y_off;
    simple_op->out_buffer.x_off = op->out_buffer->sim_x_off;
    simple_op->out_buffer.val = op->out_buffer->val;
    strncpy(simple_op->out_buffer.name, op->out_buffer->name, BUFFER_NAME_SIZE + 1);
    if((op->type == operation_binary) || (op->type == operation_reduce)) {
        simple_op->in_buffer.a_sze = op->in_buffer->sim_a_sze;
        simple_op->in_buffer.z_sze = op->in_buffer->sim_z_sze;
        simple_op->in_buffer.y_sze = op->in_buffer->sim_y_sze;
        simple_op->in_buffer.x_sze = op->in_buffer->sim_x_sze;
        simple_op->in_buffer.a_str = op->in_buffer->sim_a_str;
        simple_op->in_buffer.z_str = op->in_buffer->sim_z_str;
        simple_op->in_buffer.y_str = op->in_buffer->sim_y_str;
        simple_op->in_buffer.x_str = op->in_buffer->sim_x_str;
        simple_op->in_buffer.off = op->in_buffer->sim_off;
        simple_op->in_buffer.a_off = op->in_buffer->sim_a_off;
        simple_op->in_buffer.z_off = op->in_buffer->sim_z_off;
        simple_op->in_buffer.y_off = op->in_buffer->sim_y_off;
        simple_op->in_buffer.x_off = op->in_buffer->sim_x_off;
        simple_op->in_buffer.val = op->in_buffer->val;
        strncpy(simple_op->in_buffer.name, op->in_buffer->name, BUFFER_NAME_SIZE + 1);
    }
}
void simple_op_print(simple_op_t *simple, int padding, int offset, const char *name) {
    if(strncmp(name, "", 1)) { printf("%*s%s\n", offset, "", name); }
    printf("%*s<%p> ", offset + padding, "", (void *) simple);
    switch(simple->type) {
        case operation_unary: {
            switch(simple->unary_type) {
                case unary_add: {
                    printf("U add {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %lf %s\n", simple->out_buffer.a_sze, simple->out_buffer.z_sze,
                           simple->out_buffer.y_sze, simple->out_buffer.x_sze, simple->out_buffer.off, simple->out_buffer.a_off,
                           simple->out_buffer.z_off, simple->out_buffer.y_off, simple->out_buffer.x_off, simple->var_unary,
                           simple->out_buffer.name);
                    break;
                }
                case unary_subtract: {
                    printf("U sub {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %lf %s\n", simple->out_buffer.a_sze, simple->out_buffer.z_sze,
                           simple->out_buffer.y_sze, simple->out_buffer.x_sze, simple->out_buffer.off, simple->out_buffer.a_off,
                           simple->out_buffer.z_off, simple->out_buffer.y_off, simple->out_buffer.x_off, simple->var_unary,
                           simple->out_buffer.name);
                    break;
                }
                case unary_multiply: {
                    printf("U mul {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %lf %s\n", simple->out_buffer.a_sze, simple->out_buffer.z_sze,
                           simple->out_buffer.y_sze, simple->out_buffer.x_sze, simple->out_buffer.off, simple->out_buffer.a_off,
                           simple->out_buffer.z_off, simple->out_buffer.y_off, simple->out_buffer.x_off, simple->var_unary,
                           simple->out_buffer.name);
                    break;
                }
                case unary_divide: {
                    printf("U div {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %lf %s\n", simple->out_buffer.a_sze, simple->out_buffer.z_sze,
                           simple->out_buffer.y_sze, simple->out_buffer.x_sze, simple->out_buffer.off, simple->out_buffer.a_off,
                           simple->out_buffer.z_off, simple->out_buffer.y_off, simple->out_buffer.x_off, simple->var_unary,
                           simple->out_buffer.name);
                    break;
                }
                case unary_exp: {
                    printf("U exp {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->out_buffer.a_sze, simple->out_buffer.z_sze,
                           simple->out_buffer.y_sze, simple->out_buffer.x_sze, simple->out_buffer.off, simple->out_buffer.a_off,
                           simple->out_buffer.z_off, simple->out_buffer.y_off, simple->out_buffer.x_off, simple->out_buffer.name);
                    break;
                }
                case unary_log: {
                    printf("U log {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->out_buffer.a_sze, simple->out_buffer.z_sze,
                           simple->out_buffer.y_sze, simple->out_buffer.x_sze, simple->out_buffer.off, simple->out_buffer.a_off,
                           simple->out_buffer.z_off, simple->out_buffer.y_off, simple->out_buffer.x_off, simple->out_buffer.name);
                    break;
                }
                case unary_square: {
                    printf("U sqr {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->out_buffer.a_sze, simple->out_buffer.z_sze,
                           simple->out_buffer.y_sze, simple->out_buffer.x_sze, simple->out_buffer.off, simple->out_buffer.a_off,
                           simple->out_buffer.z_off, simple->out_buffer.y_off, simple->out_buffer.x_off, simple->out_buffer.name);
                    break;
                }
                case unary_sqrt: {
                    printf("U sqt {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->out_buffer.a_sze, simple->out_buffer.z_sze,
                           simple->out_buffer.y_sze, simple->out_buffer.x_sze, simple->out_buffer.off, simple->out_buffer.a_off,
                           simple->out_buffer.z_off, simple->out_buffer.y_off, simple->out_buffer.x_off, simple->out_buffer.name);
                    break;
                }
                case unary_reciprocal: {
                    printf("U rcp {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->out_buffer.a_sze, simple->out_buffer.z_sze,
                           simple->out_buffer.y_sze, simple->out_buffer.x_sze, simple->out_buffer.off, simple->out_buffer.a_off,
                           simple->out_buffer.z_off, simple->out_buffer.y_off, simple->out_buffer.x_off, simple->out_buffer.name);
                    break;
                }
                case unary_max: {
                    printf("U max {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %lf %s\n", simple->out_buffer.a_sze, simple->out_buffer.z_sze,
                           simple->out_buffer.y_sze, simple->out_buffer.x_sze, simple->out_buffer.off, simple->out_buffer.a_off,
                           simple->out_buffer.z_off, simple->out_buffer.y_off, simple->out_buffer.x_off, simple->var_unary,
                           simple->out_buffer.name);
                    break;
                }
                case unary_min: {
                    printf("U min {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %lf %s\n", simple->out_buffer.a_sze, simple->out_buffer.z_sze,
                           simple->out_buffer.y_sze, simple->out_buffer.x_sze, simple->out_buffer.off, simple->out_buffer.a_off,
                           simple->out_buffer.z_off, simple->out_buffer.y_off, simple->out_buffer.x_off, simple->var_unary,
                           simple->out_buffer.name);
                    break;
                }
                case unary_set: {
                    printf("U set {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %lf %s\n", simple->out_buffer.a_sze, simple->out_buffer.z_sze,
                           simple->out_buffer.y_sze, simple->out_buffer.x_sze, simple->out_buffer.off, simple->out_buffer.a_off,
                           simple->out_buffer.z_off, simple->out_buffer.y_off, simple->out_buffer.x_off, simple->var_unary,
                           simple->out_buffer.name);
                    break;
                }
                /* Never *ever* use this for things like encryption, where the randomnes of the numbers is important! */
                case unary_random: {
                    printf("U ran {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->out_buffer.a_sze, simple->out_buffer.z_sze,
                           simple->out_buffer.y_sze, simple->out_buffer.x_sze, simple->out_buffer.off, simple->out_buffer.a_off,
                           simple->out_buffer.z_off, simple->out_buffer.y_off, simple->out_buffer.x_off, simple->out_buffer.name);
                    break;
                }
                case unary_tanh: {
                    printf("U tnh {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->out_buffer.a_sze, simple->out_buffer.z_sze,
                           simple->out_buffer.y_sze, simple->out_buffer.x_sze, simple->out_buffer.off, simple->out_buffer.a_off,
                           simple->out_buffer.z_off, simple->out_buffer.y_off, simple->out_buffer.x_off, simple->out_buffer.name);
                    break;
                }
                case unary_absolute: {
                    printf("U abs {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->out_buffer.a_sze, simple->out_buffer.z_sze,
                           simple->out_buffer.y_sze, simple->out_buffer.x_sze, simple->out_buffer.off, simple->out_buffer.a_off,
                           simple->out_buffer.z_off, simple->out_buffer.y_off, simple->out_buffer.x_off, simple->out_buffer.name);
                    break;
                }
                case unary_sign: {
                    printf("U sng {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->out_buffer.a_sze, simple->out_buffer.z_sze,
                           simple->out_buffer.y_sze, simple->out_buffer.x_sze, simple->out_buffer.off, simple->out_buffer.a_off,
                           simple->out_buffer.z_off, simple->out_buffer.y_off, simple->out_buffer.x_off, simple->out_buffer.name);
                    break;
                }
            }
            break;
        }
        case operation_binary: {
            switch(simple->binary_type) {
                case binary_add: {
                    printf("B add {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
                case binary_subtract: {
                    printf("B sub {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
                case binary_multiply: {
                    printf("B mul {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
                case binary_divide: {
                    printf("B div {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
                case binary_max: {
                    printf("B max {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
                case binary_min: {
                    printf("B min {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
                case binary_copy: {
                    printf("B cpy {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
                case binary_add_like: {
                    printf("L add {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
                case binary_subtract_like: {
                    printf("L sub {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
                case binary_multiply_like: {
                    printf("L mul{%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
                case binary_divide_like: {
                    printf("L div {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
                case binary_max_like: {
                    printf("L max {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
                case binary_min_like: {
                    printf("L min {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
                case binary_copy_like: {
                    printf("L cpy {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
            }
            break;
        }
        case operation_reduce: {
            switch(simple->reduce_type) {
                case reduce_sum: {
                    printf("R sum {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
                case reduce_avg: {
                    printf("R avg {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
                case reduce_max: {
                    printf("R max {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
                case reduce_min: {
                    printf("R min {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                           simple->out_buffer.a_sze, simple->out_buffer.z_sze, simple->out_buffer.y_sze, simple->out_buffer.x_sze,
                           simple->out_buffer.off, simple->out_buffer.a_off, simple->out_buffer.z_off, simple->out_buffer.y_off,
                           simple->out_buffer.x_off, simple->in_buffer.a_sze, simple->in_buffer.z_sze, simple->in_buffer.y_sze,
                           simple->in_buffer.x_sze, simple->in_buffer.off, simple->in_buffer.a_off, simple->in_buffer.z_off,
                           simple->in_buffer.y_off, simple->in_buffer.x_off, simple->out_buffer.name, simple->in_buffer.name);
                    break;
                }
            }
            break;
        }
        case operation_move: {
            ERROR("ERROR: simple_op should not be a move operation!\n");
        }
    }
}
ALWAYS_INLINE void simple_op_realize(simple_op_t *simple) {
    switch(simple->type) {
        case operation_unary: {
            switch(simple->unary_type) {
                case unary_add: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) += simple->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_subtract: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) -= simple->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_multiply: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) *= simple->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_divide: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) /= simple->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_exp: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) = exp(SIMPLE_AT(simple->out_buffer, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_log: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) = log(SIMPLE_AT(simple->out_buffer, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_square: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) *= SIMPLE_AT(simple->out_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_sqrt: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) = sqrt(SIMPLE_AT(simple->out_buffer, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_reciprocal: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) = 1 / SIMPLE_AT(simple->out_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_max: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    if(SIMPLE_AT(simple->out_buffer, a, z, y, x) < simple->var_unary) {
                                        SIMPLE_AT(simple->out_buffer, a, z, y, x) = simple->var_unary;
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_min: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    if(SIMPLE_AT(simple->out_buffer, a, z, y, x) > simple->var_unary) {
                                        SIMPLE_AT(simple->out_buffer, a, z, y, x) = simple->var_unary;
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_set: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) = simple->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_random: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) = ((double) rand() / RAND_MAX) * 2 - 1;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_tanh: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) = tanh(SIMPLE_AT(simple->out_buffer, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_absolute: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) = fabs(SIMPLE_AT(simple->out_buffer, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_sign: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    if(SIMPLE_AT(simple->out_buffer, a, z, y, x) < 0) {
                                        SIMPLE_AT(simple->out_buffer, a, z, y, x) = -1;
                                        /* Better perf but kinda ugly */
                                        // } else if(!SIMPLE_AT(simple_op->out_buffer, a, z, y, x)) {
                                    } else if(SIMPLE_AT(simple->out_buffer, a, z, y, x) == 0) {
                                        SIMPLE_AT(simple->out_buffer, a, z, y, x) = 0;
                                    } else {
                                        SIMPLE_AT(simple->out_buffer, a, z, y, x) = 1;
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
        case operation_binary: {
            switch(simple->binary_type) {
                case binary_add: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) += SIMPLE_AT(simple->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_subtract: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) -= SIMPLE_AT(simple->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_multiply: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) *= SIMPLE_AT(simple->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_divide: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) /= SIMPLE_AT(simple->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_max: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    if(SIMPLE_AT(simple->out_buffer, a, z, y, x) < SIMPLE_AT(simple->in_buffer, a, z, y, x)) {
                                        SIMPLE_AT(simple->out_buffer, a, z, y, x) = SIMPLE_AT(simple->in_buffer, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_min: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    if(SIMPLE_AT(simple->out_buffer, a, z, y, x) > SIMPLE_AT(simple->in_buffer, a, z, y, x)) {
                                        SIMPLE_AT(simple->out_buffer, a, z, y, x) = SIMPLE_AT(simple->in_buffer, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_copy: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) = SIMPLE_AT(simple->in_buffer, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_add_like: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) += SIMPLE_AT(simple->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_subtract_like: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) -= SIMPLE_AT(simple->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_multiply_like: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) *= SIMPLE_AT(simple->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_divide_like: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) /= SIMPLE_AT(simple->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_max_like: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    if(SIMPLE_AT(simple->out_buffer, a, z, y, x) < SIMPLE_AT(simple->in_buffer, 0, 0, 0, 0)) {
                                        SIMPLE_AT(simple->out_buffer, a, z, y, x) = SIMPLE_AT(simple->in_buffer, 0, 0, 0, 0);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_min_like: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    if(SIMPLE_AT(simple->out_buffer, a, z, y, x) > SIMPLE_AT(simple->in_buffer, 0, 0, 0, 0)) {
                                        SIMPLE_AT(simple->out_buffer, a, z, y, x) = SIMPLE_AT(simple->in_buffer, 0, 0, 0, 0);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_copy_like: {
                    for(int64_t a = 0; a < simple->out_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->out_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->out_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->out_buffer.x_sze; x++) {
                                    SIMPLE_AT(simple->out_buffer, a, z, y, x) = SIMPLE_AT(simple->in_buffer, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
            }
            break;
        }
        case operation_reduce: {
            switch(simple->reduce_type) {
                case reduce_sum: {
                    double temp = 0;
                    for(int64_t a = 0; a < simple->in_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->in_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->in_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->in_buffer.x_sze; x++) { temp += SIMPLE_AT(simple->in_buffer, a, z, y, x); }
                            }
                        }
                    }
                    SIMPLE_AT(simple->out_buffer, 0, 0, 0, 0) = temp;
                    break;
                }
                case reduce_max: {
                    double temp = -INFINITY;
                    for(int64_t a = 0; a < simple->in_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->in_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->in_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->in_buffer.x_sze; x++) {
                                    if(temp < SIMPLE_AT(simple->in_buffer, a, z, y, x)) { temp = SIMPLE_AT(simple->in_buffer, a, z, y, x); }
                                }
                            }
                        }
                    }
                    SIMPLE_AT(simple->out_buffer, 0, 0, 0, 0) = temp;
                    break;
                }
                case reduce_avg: {
                    double temp = 0;
                    for(int64_t a = 0; a < simple->in_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->in_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->in_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->in_buffer.x_sze; x++) { temp += SIMPLE_AT(simple->in_buffer, a, z, y, x); }
                            }
                        }
                    }
                    SIMPLE_AT(simple->out_buffer, 0, 0, 0, 0) =
                        temp / (simple->in_buffer.x_sze * simple->in_buffer.y_sze * simple->in_buffer.z_sze * simple->in_buffer.a_sze);
                    break;
                }
                case reduce_min: {
                    double temp = INFINITY;
                    for(int64_t a = 0; a < simple->in_buffer.a_sze; a++) {
                        for(int64_t z = 0; z < simple->in_buffer.z_sze; z++) {
                            for(int64_t y = 0; y < simple->in_buffer.y_sze; y++) {
                                for(int64_t x = 0; x < simple->in_buffer.x_sze; x++) {
                                    if(temp > SIMPLE_AT(simple->in_buffer, a, z, y, x)) { temp = SIMPLE_AT(simple->in_buffer, a, z, y, x); }
                                }
                            }
                        }
                    }
                    SIMPLE_AT(simple->out_buffer, 0, 0, 0, 0) = temp;
                    break;
                }
            }
            break;
        }
        case operation_move: {
            ERROR("ERROR: simple_op should not be a move operation!\n");
        }
    }
}

/* NOTE: Completely made up value. No reasoning behind it at all. */
const int64_t INITIAL_SIMPLE_OP_CAPACTITY = 25;
linearized_t linearized_alloc(void) {
    linearized_t linearized = {
        .op_count = 0,
        .op_capacity = INITIAL_SIMPLE_OP_CAPACTITY,
        .simple = calloc(INITIAL_SIMPLE_OP_CAPACTITY, sizeof(simple_op_t)),
    };
    assert(linearized.simple);

    return linearized;
}
/* NOTE: Does `not` override the linearized ops instead appends ops. */
void linearized_from_op(linearized_t *linearized, op_t *op) {
    if(!op) { return; }
    while(op->parent_count > 0) { linearized_from_op(linearized, op->parent[0]); }
    if(linearized->op_capacity == linearized->op_count) {
        linearized->op_capacity *= 2;
        linearized->simple = realloc(linearized->simple, linearized->op_capacity * sizeof(simple_op_t));
        assert(linearized->simple);
    }
    if(op->type == operation_move) {
        simple_op_simulate_move(op);
    } else {
        simple_op_convert(&linearized->simple[linearized->op_count++], op);
    }
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
void linearized_run(linearized_t *linearized) {
    for(int64_t i = 0; i < linearized->op_count; i++) { simple_op_realize(&linearized->simple[i]); }
}
void linearized_print(linearized_t *linearized, int padding, int offset, const char *name) {
    if(!linearized) { return; }
    if(strncmp(name, "", 1)) {
        printf("%*slen %lu, cap %lu %s\n", offset, "", linearized->op_count, linearized->op_capacity, name);
    } else {
        printf("%*slen %lu, cap %lu\n", offset, "", linearized->op_count, linearized->op_capacity);
    }
    /* NOTE: Kind of a nice allignment for printing */
    // int64_t max = log10(linearized->op_count);
    // for(int64_t i = 0; i < linearized->op_count; i++) {
    //     printf("%*s[%*s%lu] ", padding + offset, "", (int) (max - (int64_t) log10(i)), "", i);
    //     simple_op_print(linearized->simple + i, 0, 0, "");
    // }
    /* This one is not alligned. */
    for(int64_t i = 0; i < linearized->op_count; i++) {
        printf("%*s[%lu] ", padding + offset, "", i);
        simple_op_print(linearized->simple + i, 0, 0, "");
    }
}
