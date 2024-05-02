#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linearize.h"
#include "tensor.h"

void simple_op_simulate_move(op_t *op) {
    assert(op);
    switch(op->type_move) {
        case move_reshape: {
            op->buffer_out->sze_a_sim = op->var_a;
            op->buffer_out->sze_z_sim = op->var_z;
            op->buffer_out->sze_y_sim = op->var_y;
            op->buffer_out->sze_x_sim = op->var_x;
            op->buffer_out->str_a_sim = op->var_z * op->var_y * op->var_x;
            op->buffer_out->str_z_sim = op->var_y * op->var_x;
            op->buffer_out->str_y_sim = op->var_x;
            op->buffer_out->str_x_sim = 1;
            break;
        }
        case move_resize: {
            op->buffer_out->sze_a_sim = op->var_a;
            op->buffer_out->sze_z_sim = op->var_z;
            op->buffer_out->sze_y_sim = op->var_y;
            op->buffer_out->sze_x_sim = op->var_x;
            break;
        }
        case move_offset: {
            op->buffer_out->off_sim = op->buffer_out->str_a_sim * op->var_a + op->buffer_out->str_z_sim * op->var_z +
                                      op->buffer_out->str_y_sim * op->var_y + op->buffer_out->str_x_sim * op->var_x;
            op->buffer_out->off_a_sim = op->var_a;
            op->buffer_out->off_z_sim = op->var_z;
            op->buffer_out->off_y_sim = op->var_y;
            op->buffer_out->off_x_sim = op->var_x;
            break;
        }
    }
}
void simple_op_convert(simple_op_t *simple_op, op_t *op) {
    assert(simple_op);
    assert(op);
    simple_op->type = op->type;
    simple_op->type_unary = op->type_unary;
    simple_op->type_binary = op->type_binary;
    simple_op->type_reduce = op->type_reduce;
    simple_op->var_unary = op->var_unary;
    simple_op->buffer_out.sze_a = op->buffer_out->sze_a_sim;
    simple_op->buffer_out.sze_z = op->buffer_out->sze_z_sim;
    simple_op->buffer_out.sze_y = op->buffer_out->sze_y_sim;
    simple_op->buffer_out.sze_x = op->buffer_out->sze_x_sim;
    simple_op->buffer_out.str_a = op->buffer_out->str_a_sim;
    simple_op->buffer_out.str_z = op->buffer_out->str_z_sim;
    simple_op->buffer_out.str_y = op->buffer_out->str_y_sim;
    simple_op->buffer_out.str_x = op->buffer_out->str_x_sim;
    simple_op->buffer_out.off = op->buffer_out->off_sim;
    simple_op->buffer_out.off_a = op->buffer_out->off_a_sim;
    simple_op->buffer_out.off_z = op->buffer_out->off_z_sim;
    simple_op->buffer_out.off_y = op->buffer_out->off_y_sim;
    simple_op->buffer_out.off_x = op->buffer_out->off_x_sim;
    simple_op->buffer_out.val = op->buffer_out->val;
    strncpy(simple_op->buffer_out.name, op->buffer_out->name, BUFFER_NAME_SIZE + 1);
    if((op->type == operation_binary) || (op->type == operation_reduce)) {
        simple_op->buffer_in.sze_a = op->buffer_in->sze_a_sim;
        simple_op->buffer_in.sze_z = op->buffer_in->sze_z_sim;
        simple_op->buffer_in.sze_y = op->buffer_in->sze_y_sim;
        simple_op->buffer_in.sze_x = op->buffer_in->sze_x_sim;
        simple_op->buffer_in.str_a = op->buffer_in->str_a_sim;
        simple_op->buffer_in.str_z = op->buffer_in->str_z_sim;
        simple_op->buffer_in.str_y = op->buffer_in->str_y_sim;
        simple_op->buffer_in.str_x = op->buffer_in->str_x_sim;
        simple_op->buffer_in.off = op->buffer_in->off_sim;
        simple_op->buffer_in.off_a = op->buffer_in->off_a_sim;
        simple_op->buffer_in.off_z = op->buffer_in->off_z_sim;
        simple_op->buffer_in.off_y = op->buffer_in->off_y_sim;
        simple_op->buffer_in.off_x = op->buffer_in->off_x_sim;
        simple_op->buffer_in.val = op->buffer_in->val;
        strncpy(simple_op->buffer_in.name, op->buffer_in->name, BUFFER_NAME_SIZE + 1);
    }
}
void simple_op_print(simple_op_t *simple, int padding, int offset, const char *name) {
    assert(simple);
    if(strncmp(name, "", 1)) { printf("%*s%s\n", offset, "", name); }
    printf("%*s<%p> ", offset + padding, "", (void *) simple);
    switch(simple->type) {
        case operation_unary: {
            switch(simple->type_unary) {
                case unary_add: {
                    printf("U add {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %lf %s\n", simple->buffer_out.sze_a,
                           simple->buffer_out.sze_z, simple->buffer_out.sze_y, simple->buffer_out.sze_x,
                           simple->buffer_out.off, simple->buffer_out.off_a, simple->buffer_out.off_z,
                           simple->buffer_out.off_y, simple->buffer_out.off_x, simple->var_unary,
                           simple->buffer_out.name);
                    break;
                }
                case unary_subtract: {
                    printf("U sub {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %lf %s\n", simple->buffer_out.sze_a,
                           simple->buffer_out.sze_z, simple->buffer_out.sze_y, simple->buffer_out.sze_x,
                           simple->buffer_out.off, simple->buffer_out.off_a, simple->buffer_out.off_z,
                           simple->buffer_out.off_y, simple->buffer_out.off_x, simple->var_unary,
                           simple->buffer_out.name);
                    break;
                }
                case unary_multiply: {
                    printf("U mul {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %lf %s\n", simple->buffer_out.sze_a,
                           simple->buffer_out.sze_z, simple->buffer_out.sze_y, simple->buffer_out.sze_x,
                           simple->buffer_out.off, simple->buffer_out.off_a, simple->buffer_out.off_z,
                           simple->buffer_out.off_y, simple->buffer_out.off_x, simple->var_unary,
                           simple->buffer_out.name);
                    break;
                }
                case unary_divide: {
                    printf("U div {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %lf %s\n", simple->buffer_out.sze_a,
                           simple->buffer_out.sze_z, simple->buffer_out.sze_y, simple->buffer_out.sze_x,
                           simple->buffer_out.off, simple->buffer_out.off_a, simple->buffer_out.off_z,
                           simple->buffer_out.off_y, simple->buffer_out.off_x, simple->var_unary,
                           simple->buffer_out.name);
                    break;
                }
                case unary_exp: {
                    printf("U exp {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->buffer_out.sze_a,
                           simple->buffer_out.sze_z, simple->buffer_out.sze_y, simple->buffer_out.sze_x,
                           simple->buffer_out.off, simple->buffer_out.off_a, simple->buffer_out.off_z,
                           simple->buffer_out.off_y, simple->buffer_out.off_x, simple->buffer_out.name);
                    break;
                }
                case unary_log: {
                    printf("U log {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->buffer_out.sze_a,
                           simple->buffer_out.sze_z, simple->buffer_out.sze_y, simple->buffer_out.sze_x,
                           simple->buffer_out.off, simple->buffer_out.off_a, simple->buffer_out.off_z,
                           simple->buffer_out.off_y, simple->buffer_out.off_x, simple->buffer_out.name);
                    break;
                }
                case unary_square: {
                    printf("U sqr {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->buffer_out.sze_a,
                           simple->buffer_out.sze_z, simple->buffer_out.sze_y, simple->buffer_out.sze_x,
                           simple->buffer_out.off, simple->buffer_out.off_a, simple->buffer_out.off_z,
                           simple->buffer_out.off_y, simple->buffer_out.off_x, simple->buffer_out.name);
                    break;
                }
                case unary_sqrt: {
                    printf("U sqt {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->buffer_out.sze_a,
                           simple->buffer_out.sze_z, simple->buffer_out.sze_y, simple->buffer_out.sze_x,
                           simple->buffer_out.off, simple->buffer_out.off_a, simple->buffer_out.off_z,
                           simple->buffer_out.off_y, simple->buffer_out.off_x, simple->buffer_out.name);
                    break;
                }
                case unary_reciprocal: {
                    printf("U rcp {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->buffer_out.sze_a,
                           simple->buffer_out.sze_z, simple->buffer_out.sze_y, simple->buffer_out.sze_x,
                           simple->buffer_out.off, simple->buffer_out.off_a, simple->buffer_out.off_z,
                           simple->buffer_out.off_y, simple->buffer_out.off_x, simple->buffer_out.name);
                    break;
                }
                case unary_max: {
                    printf("U max {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %lf %s\n", simple->buffer_out.sze_a,
                           simple->buffer_out.sze_z, simple->buffer_out.sze_y, simple->buffer_out.sze_x,
                           simple->buffer_out.off, simple->buffer_out.off_a, simple->buffer_out.off_z,
                           simple->buffer_out.off_y, simple->buffer_out.off_x, simple->var_unary,
                           simple->buffer_out.name);
                    break;
                }
                case unary_min: {
                    printf("U min {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %lf %s\n", simple->buffer_out.sze_a,
                           simple->buffer_out.sze_z, simple->buffer_out.sze_y, simple->buffer_out.sze_x,
                           simple->buffer_out.off, simple->buffer_out.off_a, simple->buffer_out.off_z,
                           simple->buffer_out.off_y, simple->buffer_out.off_x, simple->var_unary,
                           simple->buffer_out.name);
                    break;
                }
                case unary_set: {
                    printf("U set {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %lf %s\n", simple->buffer_out.sze_a,
                           simple->buffer_out.sze_z, simple->buffer_out.sze_y, simple->buffer_out.sze_x,
                           simple->buffer_out.off, simple->buffer_out.off_a, simple->buffer_out.off_z,
                           simple->buffer_out.off_y, simple->buffer_out.off_x, simple->var_unary,
                           simple->buffer_out.name);
                    break;
                }
                /* Never *ever* use this for things like encryption, where the randomnes of the numbers is important! */
                case unary_random: {
                    printf("U ran {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->buffer_out.sze_a,
                           simple->buffer_out.sze_z, simple->buffer_out.sze_y, simple->buffer_out.sze_x,
                           simple->buffer_out.off, simple->buffer_out.off_a, simple->buffer_out.off_z,
                           simple->buffer_out.off_y, simple->buffer_out.off_x, simple->buffer_out.name);
                    break;
                }
                case unary_tanh: {
                    printf("U tnh {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->buffer_out.sze_a,
                           simple->buffer_out.sze_z, simple->buffer_out.sze_y, simple->buffer_out.sze_x,
                           simple->buffer_out.off, simple->buffer_out.off_a, simple->buffer_out.off_z,
                           simple->buffer_out.off_y, simple->buffer_out.off_x, simple->buffer_out.name);
                    break;
                }
                case unary_absolute: {
                    printf("U abs {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->buffer_out.sze_a,
                           simple->buffer_out.sze_z, simple->buffer_out.sze_y, simple->buffer_out.sze_x,
                           simple->buffer_out.off, simple->buffer_out.off_a, simple->buffer_out.off_z,
                           simple->buffer_out.off_y, simple->buffer_out.off_x, simple->buffer_out.name);
                    break;
                }
                case unary_sign: {
                    printf("U sng {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s\n", simple->buffer_out.sze_a,
                           simple->buffer_out.sze_z, simple->buffer_out.sze_y, simple->buffer_out.sze_x,
                           simple->buffer_out.off, simple->buffer_out.off_a, simple->buffer_out.off_z,
                           simple->buffer_out.off_y, simple->buffer_out.off_x, simple->buffer_out.name);
                    break;
                }
            }
            break;
        }
        case operation_binary: {
            switch(simple->type_binary) {
                case binary_add: {
                    printf(
                        "B add {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
                case binary_subtract: {
                    printf(
                        "B sub {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
                case binary_multiply: {
                    printf(
                        "B mul {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
                case binary_divide: {
                    printf(
                        "B div {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
                case binary_max: {
                    printf(
                        "B max {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
                case binary_min: {
                    printf(
                        "B min {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
                case binary_copy: {
                    printf(
                        "B cpy {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
                case binary_add_like: {
                    printf(
                        "L add {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
                case binary_subtract_like: {
                    printf(
                        "L sub {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
                case binary_multiply_like: {
                    printf(
                        "L mul {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
                case binary_divide_like: {
                    printf(
                        "L div {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
                case binary_max_like: {
                    printf(
                        "L max {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
                case binary_min_like: {
                    printf(
                        "L min {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
                case binary_copy_like: {
                    printf(
                        "L cpy {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
            }
            break;
        }
        case operation_reduce: {
            switch(simple->type_reduce) {
                case reduce_sum: {
                    printf(
                        "R sum {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
                case reduce_avg: {
                    printf(
                        "R avg {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
                case reduce_max: {
                    printf(
                        "R max {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
                case reduce_min: {
                    printf(
                        "R min {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] < {%lu, %lu, %lu, %lu} %lu [%lu, %lu, %lu, %lu] %s %s\n",
                        simple->buffer_out.sze_a, simple->buffer_out.sze_z, simple->buffer_out.sze_y,
                        simple->buffer_out.sze_x, simple->buffer_out.off, simple->buffer_out.off_a,
                        simple->buffer_out.off_z, simple->buffer_out.off_y, simple->buffer_out.off_x,
                        simple->buffer_in.sze_a, simple->buffer_in.sze_z, simple->buffer_in.sze_y,
                        simple->buffer_in.sze_x, simple->buffer_in.off, simple->buffer_in.off_a,
                        simple->buffer_in.off_z, simple->buffer_in.off_y, simple->buffer_in.off_x,
                        simple->buffer_out.name, simple->buffer_in.name);
                    break;
                }
            }
            break;
        }
        case operation_move: {
            ERROR("simple_op should not be a move operation!\n");
        }
    }
}
void simple_op_realize(simple_op_t *simple) {
    assert(simple);
    switch(simple->type) {
        case operation_unary: {
            switch(simple->type_unary) {
                case unary_add: {
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) += simple->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_subtract: {
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) -= simple->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_multiply: {
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) *= simple->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_divide: {
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) /= simple->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_exp: {
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) =
                                        exp(SIMPLE_AT(simple->buffer_out, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_log: {
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) =
                                        log(SIMPLE_AT(simple->buffer_out, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_square: {
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) *=
                                        SIMPLE_AT(simple->buffer_out, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_sqrt: {
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) =
                                        sqrt(SIMPLE_AT(simple->buffer_out, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_reciprocal: {
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) =
                                        1 / SIMPLE_AT(simple->buffer_out, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_max: {
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    if(SIMPLE_AT(simple->buffer_out, a, z, y, x) < simple->var_unary) {
                                        SIMPLE_AT(simple->buffer_out, a, z, y, x) = simple->var_unary;
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_min: {
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    if(SIMPLE_AT(simple->buffer_out, a, z, y, x) > simple->var_unary) {
                                        SIMPLE_AT(simple->buffer_out, a, z, y, x) = simple->var_unary;
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_set: {
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) = simple->var_unary;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_random: {
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) = ((double) rand() / RAND_MAX) * 2 - 1;
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_tanh: {
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) =
                                        tanh(SIMPLE_AT(simple->buffer_out, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_absolute: {
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) =
                                        fabs(SIMPLE_AT(simple->buffer_out, a, z, y, x));
                                }
                            }
                        }
                    }
                    break;
                }
                case unary_sign: {
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    if(SIMPLE_AT(simple->buffer_out, a, z, y, x) < 0) {
                                        SIMPLE_AT(simple->buffer_out, a, z, y, x) = -1;
                                        /* Better perf but kinda ugly */
                                        // } else if(!SIMPLE_AT(simple_op->buffer_out, a, z, y, x)) {
                                    } else if(SIMPLE_AT(simple->buffer_out, a, z, y, x) == 0) {
                                        SIMPLE_AT(simple->buffer_out, a, z, y, x) = 0;
                                    } else {
                                        SIMPLE_AT(simple->buffer_out, a, z, y, x) = 1;
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
            switch(simple->type_binary) {
                case binary_add: {
                    assert(simple->buffer_out.sze_a == simple->buffer_in.sze_a);
                    assert(simple->buffer_out.sze_z == simple->buffer_in.sze_z);
                    assert(simple->buffer_out.sze_y == simple->buffer_in.sze_y);
                    assert(simple->buffer_out.sze_x == simple->buffer_in.sze_x);
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) +=
                                        SIMPLE_AT(simple->buffer_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_subtract: {
                    assert(simple->buffer_out.sze_a == simple->buffer_in.sze_a);
                    assert(simple->buffer_out.sze_z == simple->buffer_in.sze_z);
                    assert(simple->buffer_out.sze_y == simple->buffer_in.sze_y);
                    assert(simple->buffer_out.sze_x == simple->buffer_in.sze_x);
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) -=
                                        SIMPLE_AT(simple->buffer_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_multiply: {
                    assert(simple->buffer_out.sze_a == simple->buffer_in.sze_a);
                    assert(simple->buffer_out.sze_z == simple->buffer_in.sze_z);
                    assert(simple->buffer_out.sze_y == simple->buffer_in.sze_y);
                    assert(simple->buffer_out.sze_x == simple->buffer_in.sze_x);
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) *=
                                        SIMPLE_AT(simple->buffer_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_divide: {
                    assert(simple->buffer_out.sze_a == simple->buffer_in.sze_a);
                    assert(simple->buffer_out.sze_z == simple->buffer_in.sze_z);
                    assert(simple->buffer_out.sze_y == simple->buffer_in.sze_y);
                    assert(simple->buffer_out.sze_x == simple->buffer_in.sze_x);
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) /=
                                        SIMPLE_AT(simple->buffer_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_max: {
                    assert(simple->buffer_out.sze_a == simple->buffer_in.sze_a);
                    assert(simple->buffer_out.sze_z == simple->buffer_in.sze_z);
                    assert(simple->buffer_out.sze_y == simple->buffer_in.sze_y);
                    assert(simple->buffer_out.sze_x == simple->buffer_in.sze_x);
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    if(SIMPLE_AT(simple->buffer_out, a, z, y, x) <
                                       SIMPLE_AT(simple->buffer_in, a, z, y, x)) {
                                        SIMPLE_AT(simple->buffer_out, a, z, y, x) =
                                            SIMPLE_AT(simple->buffer_in, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_min: {
                    assert(simple->buffer_out.sze_a == simple->buffer_in.sze_a);
                    assert(simple->buffer_out.sze_z == simple->buffer_in.sze_z);
                    assert(simple->buffer_out.sze_y == simple->buffer_in.sze_y);
                    assert(simple->buffer_out.sze_x == simple->buffer_in.sze_x);
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    if(SIMPLE_AT(simple->buffer_out, a, z, y, x) >
                                       SIMPLE_AT(simple->buffer_in, a, z, y, x)) {
                                        SIMPLE_AT(simple->buffer_out, a, z, y, x) =
                                            SIMPLE_AT(simple->buffer_in, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_copy: {
                    assert(simple->buffer_out.sze_a == simple->buffer_in.sze_a);
                    assert(simple->buffer_out.sze_z == simple->buffer_in.sze_z);
                    assert(simple->buffer_out.sze_y == simple->buffer_in.sze_y);
                    assert(simple->buffer_out.sze_x == simple->buffer_in.sze_x);
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) =
                                        SIMPLE_AT(simple->buffer_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_add_like: {
                    assert(simple->buffer_in.sze_a == 1);
                    assert(simple->buffer_in.sze_z == 1);
                    assert(simple->buffer_in.sze_y == 1);
                    assert(simple->buffer_in.sze_x == 1);
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) +=
                                        SIMPLE_AT(simple->buffer_in, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_subtract_like: {
                    assert(simple->buffer_in.sze_a == 1);
                    assert(simple->buffer_in.sze_z == 1);
                    assert(simple->buffer_in.sze_y == 1);
                    assert(simple->buffer_in.sze_x == 1);
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) -=
                                        SIMPLE_AT(simple->buffer_in, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_multiply_like: {
                    assert(simple->buffer_in.sze_a == 1);
                    assert(simple->buffer_in.sze_z == 1);
                    assert(simple->buffer_in.sze_y == 1);
                    assert(simple->buffer_in.sze_x == 1);
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) *=
                                        SIMPLE_AT(simple->buffer_in, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_divide_like: {
                    assert(simple->buffer_in.sze_a == 1);
                    assert(simple->buffer_in.sze_z == 1);
                    assert(simple->buffer_in.sze_y == 1);
                    assert(simple->buffer_in.sze_x == 1);
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) /=
                                        SIMPLE_AT(simple->buffer_in, 0, 0, 0, 0);
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_max_like: {
                    assert(simple->buffer_in.sze_a == 1);
                    assert(simple->buffer_in.sze_z == 1);
                    assert(simple->buffer_in.sze_y == 1);
                    assert(simple->buffer_in.sze_x == 1);
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    if(SIMPLE_AT(simple->buffer_out, a, z, y, x) <
                                       SIMPLE_AT(simple->buffer_in, 0, 0, 0, 0)) {
                                        SIMPLE_AT(simple->buffer_out, a, z, y, x) =
                                            SIMPLE_AT(simple->buffer_in, 0, 0, 0, 0);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_min_like: {
                    assert(simple->buffer_in.sze_a == 1);
                    assert(simple->buffer_in.sze_z == 1);
                    assert(simple->buffer_in.sze_y == 1);
                    assert(simple->buffer_in.sze_x == 1);
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    if(SIMPLE_AT(simple->buffer_out, a, z, y, x) >
                                       SIMPLE_AT(simple->buffer_in, 0, 0, 0, 0)) {
                                        SIMPLE_AT(simple->buffer_out, a, z, y, x) =
                                            SIMPLE_AT(simple->buffer_in, 0, 0, 0, 0);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case binary_copy_like: {
                    assert(simple->buffer_in.sze_a == 1);
                    assert(simple->buffer_in.sze_z == 1);
                    assert(simple->buffer_in.sze_y == 1);
                    assert(simple->buffer_in.sze_x == 1);
                    for(int64_t a = 0; a < simple->buffer_out.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_out.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_out.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_out.sze_x; x++) {
                                    SIMPLE_AT(simple->buffer_out, a, z, y, x) =
                                        SIMPLE_AT(simple->buffer_in, 0, 0, 0, 0);
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
            assert(simple->buffer_out.sze_a == 1);
            assert(simple->buffer_out.sze_z == 1);
            assert(simple->buffer_out.sze_y == 1);
            assert(simple->buffer_out.sze_x == 1);
            switch(simple->type_reduce) {
                case reduce_sum: {
                    double temp = 0;
                    for(int64_t a = 0; a < simple->buffer_in.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_in.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_in.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_in.sze_x; x++) {
                                    temp += SIMPLE_AT(simple->buffer_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    SIMPLE_AT(simple->buffer_out, 0, 0, 0, 0) = temp;
                    break;
                }
                case reduce_max: {
                    double temp = -INFINITY;
                    for(int64_t a = 0; a < simple->buffer_in.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_in.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_in.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_in.sze_x; x++) {
                                    if(temp < SIMPLE_AT(simple->buffer_in, a, z, y, x)) {
                                        temp = SIMPLE_AT(simple->buffer_in, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    SIMPLE_AT(simple->buffer_out, 0, 0, 0, 0) = temp;
                    break;
                }
                case reduce_avg: {
                    double temp = 0;
                    for(int64_t a = 0; a < simple->buffer_in.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_in.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_in.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_in.sze_x; x++) {
                                    temp += SIMPLE_AT(simple->buffer_in, a, z, y, x);
                                }
                            }
                        }
                    }
                    SIMPLE_AT(simple->buffer_out, 0, 0, 0, 0) =
                        temp / (simple->buffer_in.sze_x * simple->buffer_in.sze_y * simple->buffer_in.sze_z *
                                simple->buffer_in.sze_a);
                    break;
                }
                case reduce_min: {
                    double temp = INFINITY;
                    for(int64_t a = 0; a < simple->buffer_in.sze_a; a++) {
                        for(int64_t z = 0; z < simple->buffer_in.sze_z; z++) {
                            for(int64_t y = 0; y < simple->buffer_in.sze_y; y++) {
                                for(int64_t x = 0; x < simple->buffer_in.sze_x; x++) {
                                    if(temp > SIMPLE_AT(simple->buffer_in, a, z, y, x)) {
                                        temp = SIMPLE_AT(simple->buffer_in, a, z, y, x);
                                    }
                                }
                            }
                        }
                    }
                    SIMPLE_AT(simple->buffer_out, 0, 0, 0, 0) = temp;
                    break;
                }
            }
            break;
        }
        case operation_move: {
            ERROR("simple_op should not be a move operation!\n");
        }
    }
}

/* NOTE: Completely made up value. No reasoning behind it at all. */
const int64_t INITIAL_SIMPLE_OP_CAPACTITY = 25;
linearized_t linearized_alloc(void) {
    linearized_t linearized = {
        .op_len = 0,
        .op_cap = INITIAL_SIMPLE_OP_CAPACTITY,
        .simple = calloc(INITIAL_SIMPLE_OP_CAPACTITY, sizeof(simple_op_t)),
    };
    assert(linearized.simple);

    return linearized;
}
/* NOTE: Does `not` override the linearized ops instead appends ops. */
void linearized_from_op(linearized_t *linearized, op_t *op) {
    assert(linearized);
    if(!op) { return; }
    op_t *temp;
    op_t *next = op;
    while(op->parent_count > 0) {
        temp = next;
        for(int64_t i = 0; i < MAX_DEPTH; i++) {
            if(temp->parent_count > 0) {
                temp = temp->parent[0];
            } else {
                break;
            }
        }
        assert(temp);
        assert(temp->parent_count == 0);
        if(linearized->op_cap == linearized->op_len) {
            linearized->op_cap *= 2;
            linearized->simple = reallocarray(linearized->simple, linearized->op_cap, sizeof(simple_op_t));
            assert(linearized->simple);
        }
        if(temp->type == operation_move) {
            simple_op_simulate_move(temp);
        } else {
            simple_op_convert(&linearized->simple[linearized->op_len++], temp);
        }
        next = temp->child_count > 0 ? temp->child[0] : op;
        op_cleanup(temp);
        op_free(temp);
        free(temp);
    }
    if(linearized->op_cap == linearized->op_len) {
        linearized->op_cap *= 2;
        linearized->simple = reallocarray(linearized->simple, linearized->op_cap, sizeof(simple_op_t));
        assert(linearized->simple);
    }
    if(op->type == operation_move) {
        simple_op_simulate_move(op);
    } else {
        simple_op_convert(&linearized->simple[linearized->op_len++], op);
    }
    op_cleanup(op);
    op_free(op);
    free(op);
}
void linearized_free(linearized_t *linearized) {
    assert(linearized);
    free(linearized->simple);
}
void linearized_clear(linearized_t *linearized) {
    assert(linearized);
    linearized->op_len = 0;
}
void linearized_run(linearized_t *linearized) {
    assert(linearized);
    for(int64_t i = 0; i < linearized->op_len; i++) { simple_op_realize(&linearized->simple[i]); }
}
void linearized_print(linearized_t *linearized, int padding, int offset, const char *name) {
    assert(linearized);
    if(!linearized) { return; }
    if(strncmp(name, "", 1)) {
        printf("%*slen %lu, cap %lu %s\n", offset, "", linearized->op_len, linearized->op_cap, name);
    } else {
        printf("%*slen %lu, cap %lu\n", offset, "", linearized->op_len, linearized->op_cap);
    }
    /* NOTE: Kind of a nice allignment for printing */
    // int64_t max = log10(linearized->op_count);
    // for(int64_t i = 0; i < linearized->op_count; i++) {
    //     printf("%*s[%*s%lu] ", padding + offset, "", (int) (max - (int64_t) log10(i)), "", i);
    //     simple_op_print(linearized->simple + i, 0, 0, "");
    // }
    /* This one is not alligned. */
    for(int64_t i = 0; i < linearized->op_len; i++) {
        printf("%*s[%lu] ", padding + offset, "", i);
        simple_op_print(linearized->simple + i, 0, 0, "");
    }
}
