#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"
#include "linearize.h"
#include "runtime.h"

const uint64_t cl_linearized_initial_capacity = 16;
static cl_linearized_t cl_linearized_alloc(void) {
    cl_linearized_t cl_linearized = {
        .cl_op_length = 0,
        .cl_op_capacity = cl_linearized_initial_capacity,
        .cl_op = calloc(cl_linearized_initial_capacity, sizeof(cl_op_t))
    };
    assert(cl_linearized.cl_op);

    return(cl_linearized);
}
static void cl_linearized_free(cl_linearized_t *cl_linearized) {
    free(cl_linearized->cl_op);
}
static void cl_linearized_build(cl_linearized_t *cl_linearized, linearized_t *linearized) {
    cl_linearized->cl_op_length = 0;
    for(uint64_t i = 0; i < linearized->op_count; i++) {
        if(cl_linearized->cl_op_length == cl_linearized->cl_op_capacity) {
            cl_linearized->cl_op_capacity *= 2;
            cl_linearized->cl_op = realloc(cl_linearized->cl_op, sizeof(cl_op_t) * cl_linearized->cl_op_capacity);
        }
        switch(linearized->simple[i].type) {
            case(operation_move): {
                switch(linearized->simple[i].move_type) {
                    case(move_reshape): {
                        linearized->simple[i].out_buffer->cl_a_size = linearized->simple[i].var_a;
                        linearized->simple[i].out_buffer->cl_z_size = linearized->simple[i].var_z;
                        linearized->simple[i].out_buffer->cl_y_size = linearized->simple[i].var_y;
                        linearized->simple[i].out_buffer->cl_x_size = linearized->simple[i].var_x;
                        linearized->simple[i].out_buffer->cl_a_stride = linearized->simple[i].var_z * linearized->simple[i].var_y * linearized->simple[i].var_x;
                        linearized->simple[i].out_buffer->cl_z_stride = linearized->simple[i].var_y * linearized->simple[i].var_x;
                        linearized->simple[i].out_buffer->cl_y_stride = linearized->simple[i].var_x;
                        linearized->simple[i].out_buffer->cl_x_stride = 1;
                        break;
                    }
                    case(move_resize): {
                        linearized->simple[i].out_buffer->cl_a_size = linearized->simple[i].var_a;
                        linearized->simple[i].out_buffer->cl_z_size = linearized->simple[i].var_z;
                        linearized->simple[i].out_buffer->cl_y_size = linearized->simple[i].var_y;
                        linearized->simple[i].out_buffer->cl_x_size = linearized->simple[i].var_x;
                        // linearized->simple[i].out_buffer->cl_a_stride = linearized->simple[i].var_z * linearized->simple[i].var_y * linearized->simple[i].var_x;
                        // linearized->simple[i].out_buffer->cl_z_stride = linearized->simple[i].var_y * linearized->simple[i].var_x;
                        // linearized->simple[i].out_buffer->cl_y_stride = linearized->simple[i].var_x;
                        // linearized->simple[i].out_buffer->cl_x_stride = 1;
                        break;
                    }
                    case(move_offset): {
                        linearized->simple[i].out_buffer->cl_offset = linearized->simple[i].out_buffer->cl_a_stride * linearized->simple[i].var_a + linearized->simple[i].out_buffer->cl_z_stride * linearized->simple[i].var_z + linearized->simple[i].out_buffer->cl_y_stride * linearized->simple[i].var_y + linearized->simple[i].out_buffer->cl_x_stride * linearized->simple[i].var_x;
                        break;
                    }
                }
                break;
            }
            case(operation_unary): {
                cl_linearized->cl_op[cl_linearized->cl_op_length].type = operation_unary;
                cl_linearized->cl_op[cl_linearized->cl_op_length].unary_type = linearized->simple[i].unary_type;
                cl_linearized->cl_op[cl_linearized->cl_op_length].var_unary = linearized->simple[i].var_unary;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.offset = linearized->simple[i].out_buffer->cl_offset;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.a_size = linearized->simple[i].out_buffer->cl_a_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.z_size = linearized->simple[i].out_buffer->cl_z_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.y_size = linearized->simple[i].out_buffer->cl_y_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.x_size = linearized->simple[i].out_buffer->cl_x_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.a_stride = linearized->simple[i].out_buffer->cl_a_stride;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.z_stride = linearized->simple[i].out_buffer->cl_z_stride;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.y_stride = linearized->simple[i].out_buffer->cl_y_stride;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.x_stride = linearized->simple[i].out_buffer->cl_x_stride;
                /* TODO: memcpy here. */
                for(uint64_t j = 0; j < CL_NAME_SIZE; j++) {
                    cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.cl_name[j] = linearized->simple[i].out_buffer->cl_name[j];
                }
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.cl_name[CL_NAME_SIZE] = '\0';
                cl_linearized->cl_op_length++;
                break;
            }
            case(operation_binary): {
                cl_linearized->cl_op[cl_linearized->cl_op_length].type = operation_binary;
                cl_linearized->cl_op[cl_linearized->cl_op_length].binary_type = linearized->simple[i].binary_type;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.offset = linearized->simple[i].out_buffer->cl_offset;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.a_size = linearized->simple[i].out_buffer->cl_a_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.z_size = linearized->simple[i].out_buffer->cl_z_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.y_size = linearized->simple[i].out_buffer->cl_y_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.x_size = linearized->simple[i].out_buffer->cl_x_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.a_stride = linearized->simple[i].out_buffer->cl_a_stride;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.z_stride = linearized->simple[i].out_buffer->cl_z_stride;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.y_stride = linearized->simple[i].out_buffer->cl_y_stride;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.x_stride = linearized->simple[i].out_buffer->cl_x_stride;
                for(uint64_t j = 0; j < CL_NAME_SIZE; j++) {
                    cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.cl_name[j] = linearized->simple[i].out_buffer->cl_name[j];
                }
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.cl_name[CL_NAME_SIZE] = '\0';
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.offset = linearized->simple[i].in_buffer->cl_offset;
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.a_size = linearized->simple[i].in_buffer->cl_a_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.z_size = linearized->simple[i].in_buffer->cl_z_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.y_size = linearized->simple[i].in_buffer->cl_y_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.x_size = linearized->simple[i].in_buffer->cl_x_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.a_stride = linearized->simple[i].in_buffer->cl_a_stride;
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.z_stride = linearized->simple[i].in_buffer->cl_z_stride;
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.y_stride = linearized->simple[i].in_buffer->cl_y_stride;
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.x_stride = linearized->simple[i].in_buffer->cl_x_stride;
                for(uint64_t j = 0; j < CL_NAME_SIZE; j++) {
                    cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.cl_name[j] = linearized->simple[i].in_buffer->cl_name[j];
                }
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.cl_name[CL_NAME_SIZE] = '\0';
                cl_linearized->cl_op_length++;
                break;
            }
            case(operation_reduce): {
                /* TODO: Think about how to do these. They are BY FAR the hardest to do, as they might require sharing memory globally and other strange stuff. */
                cl_linearized->cl_op[cl_linearized->cl_op_length].type = operation_reduce;
                cl_linearized->cl_op[cl_linearized->cl_op_length].reduce_type = linearized->simple[i].reduce_type;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.offset = linearized->simple[i].out_buffer->cl_offset;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.a_size = linearized->simple[i].out_buffer->cl_a_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.z_size = linearized->simple[i].out_buffer->cl_z_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.y_size = linearized->simple[i].out_buffer->cl_y_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.x_size = linearized->simple[i].out_buffer->cl_x_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.a_stride = linearized->simple[i].out_buffer->cl_a_stride;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.z_stride = linearized->simple[i].out_buffer->cl_z_stride;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.y_stride = linearized->simple[i].out_buffer->cl_y_stride;
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.x_stride = linearized->simple[i].out_buffer->cl_x_stride;
                for(uint64_t j = 0; j < CL_NAME_SIZE; j++) {
                    cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.cl_name[j] = linearized->simple[i].out_buffer->cl_name[j];
                }
                cl_linearized->cl_op[cl_linearized->cl_op_length].out_buffer.cl_name[CL_NAME_SIZE] = '\0';
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.offset = linearized->simple[i].in_buffer->cl_offset;
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.a_size = linearized->simple[i].in_buffer->cl_a_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.z_size = linearized->simple[i].in_buffer->cl_z_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.y_size = linearized->simple[i].in_buffer->cl_y_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.x_size = linearized->simple[i].in_buffer->cl_x_size;
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.a_stride = linearized->simple[i].in_buffer->cl_a_stride;
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.z_stride = linearized->simple[i].in_buffer->cl_z_stride;
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.y_stride = linearized->simple[i].in_buffer->cl_y_stride;
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.x_stride = linearized->simple[i].in_buffer->cl_x_stride;
                for(uint64_t j = 0; j < CL_NAME_SIZE; j++) {
                    cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.cl_name[j] = linearized->simple[i].in_buffer->cl_name[j];
                }
                cl_linearized->cl_op[cl_linearized->cl_op_length].in_buffer.cl_name[CL_NAME_SIZE] = '\0';
                cl_linearized->cl_op_length++;
                break;
            }
        }
    }
}
// static void cl_linearized_print(cl_linearized_t *cl_linearized, int padding, int offset, const char *name) {
//     for(uint64_t i = 0; i < cl_linearized->cl_op_length; i++) {
//         printf("%*s[%lu] ", offset + padding, "", i);
//         switch(cl_linearized->cl_op[i].type) {
//             case(operation_unary): {
//                 switch(cl_linearized->cl_op[i].unary_type) {
//                     case(unary_add): {
//                         printf("U add {%lu, %lu, %lu, %lu} %lu %lf %s\n", cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].var_unary, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(unary_subtract): {
//                         printf("U sub {%lu, %lu, %lu, %lu} %lu %lf %s\n", cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].var_unary, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(unary_multiply): {
//                         printf("U mul {%lu, %lu, %lu, %lu} %lu %lf %s\n", cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].var_unary, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(unary_divide): {
//                         printf("U div {%lu, %lu, %lu, %lu} %lu %lf %s\n", cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].var_unary, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(unary_exp): {
//                         printf("U exp {%lu, %lu, %lu, %lu} %lu %s\n", cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(unary_log): {
//                         printf("U log {%lu, %lu, %lu, %lu} %lu %s\n", cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(unary_square): {
//                         printf("U sqr {%lu, %lu, %lu, %lu} %lu %s\n", cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(unary_sqrt): {
//                         printf("U sqt {%lu, %lu, %lu, %lu} %lu %s\n", cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(unary_negate): {
//                         printf("U ngt {%lu, %lu, %lu, %lu} %lu %s\n", cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(unary_reciprocal): {
//                         printf("U rcp {%lu, %lu, %lu, %lu} %lu %s\n", cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(unary_max): {
//                         printf("U max {%lu, %lu, %lu, %lu} %lu %lf %s\n", cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].var_unary, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(unary_min): {
//                         printf("U min {%lu, %lu, %lu, %lu} %lu %lf %s\n", cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].var_unary, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(unary_set): {
//                         printf("U set {%lu, %lu, %lu, %lu} %lu %lf %s\n", cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].var_unary, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(unary_zero): {
//                         printf("U zer {%lu, %lu, %lu, %lu} %lu %lf %s\n", cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].var_unary, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(unary_random): {
//                         printf("U ran {%lu, %lu, %lu, %lu} %lu %s\n", cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(unary_tanh): {
//                         printf("U tnh {%lu, %lu, %lu, %lu} %lu %s\n", cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(unary_absolute): {
//                         printf("U abs {%lu, %lu, %lu, %lu} %lu %s\n", cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                 }
//                 break;
//             }
//             case(operation_binary): {
//                 switch(cl_linearized->cl_op[i].binary_type) {
//                     case(binary_add): {
//                         printf("B add {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(binary_subtract): {
//                         printf("B sub {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(binary_multiply): {
//                         printf("B mul {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(binary_divide): {
//                         printf("B div {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(binary_max): {
//                         printf("B max {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(binary_min): {
//                         printf("B min {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(binary_copy): {
//                         printf("B cpy {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(binary_add_like): {
//                         printf("B ldd {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(binary_subtract_like): {
//                         printf("B lub {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(binary_multiply_like): {
//                         printf("B lul {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(binary_divide_like): {
//                         printf("B liv {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(binary_max_like): {
//                         printf("B lax {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(binary_min_like): {
//                         printf("B lin {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(binary_copy_like): {
//                         printf("B lpy {%lu, %lu, %lu, %lu} %lu & {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                 }
//                 break;
//             }
//             case(operation_reduce): {
//                 switch(cl_linearized->cl_op[i].reduce_type) {
//                     case(reduce_sum): {
//                         printf("R sum {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(reduce_avg): {
//                         printf("R avg {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(reduce_max): {
//                         printf("R max {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                     case(reduce_min): {
//                         printf("R min {%lu, %lu, %lu, %lu} %lu > {%lu, %lu, %lu, %lu} %lu %s %s\n", cl_linearized->cl_op[i].in_buffer.a_size, cl_linearized->cl_op[i].in_buffer.z_size, cl_linearized->cl_op[i].in_buffer.y_size, cl_linearized->cl_op[i].in_buffer.x_size, cl_linearized->cl_op[i].in_buffer.offset, cl_linearized->cl_op[i].out_buffer.a_size, cl_linearized->cl_op[i].out_buffer.z_size, cl_linearized->cl_op[i].out_buffer.y_size, cl_linearized->cl_op[i].out_buffer.x_size, cl_linearized->cl_op[i].out_buffer.offset, cl_linearized->cl_op[i].in_buffer.cl_name, cl_linearized->cl_op[i].out_buffer.cl_name);
//                         break;
//                     }
//                 }
//                 break;
//             }
//             case(operation_move): {
//                 fprintf(stderr, "ERROR: Should never have move operation in cl_linearized, but got one at index %lu\n", i);
//                 exit(1);
//             }
//         }
//     }
// }

/* NOTE: Value completely made up. */
const uint64_t initial_source_size = 16;
/* NOTE: Return pointer to source code that computes `cl_linearized`. */
static void runtime_compile_cl_linearized_(cl_linearized_t *cl_linearized, uint64_t work_groups, uint64_t work_items, uint64_t max_source_size) {
    for(uint64_t i = 0; i < cl_linearized->cl_op_length; i++) {
    }
}
/* TODO: How to get the layer starting points and sizes? */
static void runtime_compile_linearized_(runtime_t *runtime, linearized_t *linearized, uint64_t section_start_i, uint64_t section_length) {
    cl_linearized_t cl_linearized = cl_linearized_alloc();
    cl_linearized_build(&cl_linearized, linearized);
    switch(runtime->type) {
        case(runtime_c): {
            fprintf(stderr, "ERROR: Called runtime_compile_linearized with runtime_c\n");
            exit(1);
        }
        case(runtime_compile): {
            break;
        }
    }
    cl_linearized_free(&cl_linearized);
}
static void runtime_run_c_(runtime_t *runtime) {
    for(uint64_t i = 0; i < runtime->linearized->op_count; i++) {
        simple_op_realize(&runtime->linearized->simple[i]);
    }
}
runtime_t runtime_alloc(enum runtime_e type) {
    runtime_t runtime = {0};
    switch(type) {
        case(runtime_c): {
            runtime.type = runtime_c;
            runtime.linearized = calloc(1, sizeof(linearized_t));
            break;
        }
        case(runtime_compile): {
            runtime.type = runtime_compile;
            /* TODO: This needs to be done in a seperate functions for forward, backward and learning, such that runtime_t can remain very generic. */
            // runtime_compile_layer_primitives_(&runtime, neuralnet);
            break;
        }
    }
    return(runtime);
}
void runtime_free(runtime_t *runtime) {
    switch(runtime->type) {
        case(runtime_c): {
            /* TODO: Think about if I want this to free the linearized thing. */
            // linearized_free(runtime->linearized);
            // free(runtime->linearized);
            break;
        }
        case(runtime_compile): {
            break;
        }
    }
}
void runtime_execute(runtime_t *runtime) {
    switch(runtime->type) {
        case(runtime_c): {
            runtime_run_c_(runtime);
            break;
        }
        case(runtime_compile): {
            break;
        }
    }
}
