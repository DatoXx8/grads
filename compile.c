#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linearize.h"
#include "tensor.h"
#include "compile.h"

const int64_t initial_source_size = 100;
const int64_t initial_arg_num = 10;
const int64_t max_arg_size = 24;
const int64_t max_index_digits = 9;
/* NOTE: Biggest I found was 131 for `max` or `min` binary ops*/
const int64_t max_op_size = 160;
#define SIMPLE_INDEX(simple, a, z, y, x) ((simple).a_stride * (a) + (simple).z_stride * (z) + (simple).y_stride * (y) + (simple).x_stride * (x) + (simple).offset)
#define SIMPLE_INDEX_(simple, a, z, y, x) ((simple)->a_stride * (a) + (simple)->z_stride * (z) + (simple)->y_stride * (y) + (simple)->x_stride * (x) + (simple)->offset)
/* TODO: BIG refactor needed. */
void compile_linearized_to_c(const char *filename, linearized_t *linearized) {
    int64_t arg_capacity = initial_arg_num;
    int64_t source_size = initial_source_size;
    int64_t offset;
    int64_t arg_num = 0;
    int64_t found_o;
    int64_t found_i;
    FILE *source_file = fopen(filename, "w");
    char *source = calloc(initial_source_size, sizeof(char));
    char *curr = source;
    char *func_name = "my_net";
    char **args = calloc(initial_arg_num, sizeof(char *));
    for(uint64_t i = 0; i < initial_arg_num; i++) {
        args[i] = calloc(BUFFER_NAME_SIZE + 1, sizeof(char));
    }

    for(uint64_t i = 0; i < linearized->op_count; i++) {
        switch(linearized->simple[i].type) {
            case(operation_unary): {
                found_o = 0;
                for(int64_t j = 0; j < arg_num; j++) {
                    if(!strcmp(args[j], linearized->simple[i].out_buffer.name)) {
                        found_o = 1;
                        break;
                    }
                }
                if(!found_o) {
                    if(arg_num == arg_capacity) {
                        arg_capacity *= 2;
                        args = realloc(args, arg_capacity);
                    }
                    args[arg_num] = strncpy(args[arg_num], linearized->simple[i].out_buffer.name, BUFFER_NAME_SIZE + 1);
                    arg_num++;
                }
                switch(linearized->simple[i].unary_type) {
                    case(unary_add): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] += %.16lf;\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].var_unary);
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_subtract): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] -= %.16lf;\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].var_unary);
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_multiply): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] *= %.16lf;\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].var_unary);
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_divide): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] /= %.16lf;\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].var_unary);
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_exp): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] = exp(%s[%lu]);\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_log): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] = log(%s[%lu]);\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_square): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] *= %s[%lu];\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_sqrt): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] = sqrt(%s[%lu]);\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_negate): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] *= -1;\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_reciprocal): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] = 1 / %s[%lu];\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x));
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_max): {
                        exit(2);
                        break;
                    }
                    case(unary_min): {
                        exit(2);
                        break;
                    }
                    case(unary_set): {
                        /* TODO: Figure out real max size value. */
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] = %.16lf;\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].var_unary);
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_random): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] = ((double) rand() / RAND_MAX) * 2 - 1;\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x));
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_tanh): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] = tanh(%s[%lu]);\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_absolute): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] = fabs(%s[%lu]);\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_sign): {
                        exit(2);
                        break;
                    }
                }
                break;
            }
            case(operation_binary): {
                found_o = 0;
                for(int64_t j = 0; j < arg_num; j++) {
                    if(!strcmp(args[j], linearized->simple[i].out_buffer.name)) {
                        found_o = 1;
                        break;
                    }
                }
                if(!found_o) {
                    if(arg_num == arg_capacity) {
                        arg_capacity *= 2;
                        args = realloc(args, arg_capacity);
                    }
                    args[arg_num] = strncpy(args[arg_num], linearized->simple[i].out_buffer.name, BUFFER_NAME_SIZE + 1);
                    arg_num++;
                }
                found_i = 0;
                for(int64_t j = 0; j < arg_num; j++) {
                    if(!strcmp(args[j], linearized->simple[i].in_buffer.name)) {
                        found_i = 1;
                        break;
                    }
                }
                if(!found_i) {
                    if(arg_num == arg_capacity) {
                        arg_capacity *= 2;
                        args = realloc(args, arg_capacity);
                    }
                    args[arg_num] = strncpy(args[arg_num], linearized->simple[i].in_buffer.name, BUFFER_NAME_SIZE + 1);
                    arg_num++;
                }
                switch(linearized->simple[i].binary_type) {
                    case(binary_add): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] += %s[%lu];\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].in_buffer.name, SIMPLE_INDEX(linearized->simple[i].in_buffer, a, z, y, x));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_subtract): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] -= %s[%lu];\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].in_buffer.name, SIMPLE_INDEX(linearized->simple[i].in_buffer, a, z, y, x));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_multiply): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] *= %s[%lu];\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].in_buffer.name, SIMPLE_INDEX(linearized->simple[i].in_buffer, a, z, y, x));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_divide): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] /= %s[%lu];\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].in_buffer.name, SIMPLE_INDEX(linearized->simple[i].in_buffer, a, z, y, x));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_max): {
                        exit(2);
                        break;
                    }
                    case(binary_min): {
                        exit(2);
                        break;
                    }
                    case(binary_copy): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] = %s[%lu];\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].in_buffer.name, SIMPLE_INDEX(linearized->simple[i].in_buffer, a, z, y, x));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_add_like): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] += %s[%lu];\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].in_buffer.name, SIMPLE_INDEX(linearized->simple[i].in_buffer, 0, 0, 0, 0));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_subtract_like): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] -= %s[%lu];\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].in_buffer.name, SIMPLE_INDEX(linearized->simple[i].in_buffer, 0, 0, 0, 0));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_multiply_like): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] *= %s[%lu];\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].in_buffer.name, SIMPLE_INDEX(linearized->simple[i].in_buffer, 0, 0, 0, 0));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_divide_like): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] /= %s[%lu];\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].in_buffer.name, SIMPLE_INDEX(linearized->simple[i].in_buffer, 0, 0, 0, 0));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_max_like): {
                        exit(2);
                        break;
                    }
                    case(binary_min_like): {
                        exit(2);
                        break;
                    }
                    case(binary_copy_like): {
                        for(uint64_t a = 0; a < linearized->simple[i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] = %s[%lu];\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, a, z, y, x), linearized->simple[i].in_buffer.name, SIMPLE_INDEX(linearized->simple[i].in_buffer, 0, 0, 0, 0));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
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
            case(operation_reduce): {
                found_o = 0;
                for(int64_t j = 0; j < arg_num; j++) {
                    if(!strcmp(args[j], linearized->simple[i].out_buffer.name)) {
                        found_o = 1;
                        break;
                    }
                }
                if(!found_o) {
                    if(arg_num == arg_capacity) {
                        arg_capacity *= 2;
                        args = realloc(args, arg_capacity);
                    }
                    args[arg_num] = strncpy(args[arg_num], linearized->simple[i].out_buffer.name, BUFFER_NAME_SIZE + 1);
                    arg_num++;
                }
                found_i = 0;
                for(int64_t j = 0; j < arg_num; j++) {
                    if(!strcmp(args[j], linearized->simple[i].in_buffer.name)) {
                        found_i = 1;
                        break;
                    }
                }
                if(!found_i) {
                    if(arg_num == arg_capacity) {
                        arg_capacity *= 2;
                        args = realloc(args, arg_capacity);
                    }
                    args[arg_num] = strncpy(args[arg_num], linearized->simple[i].in_buffer.name, BUFFER_NAME_SIZE + 1);
                    arg_num++;
                }
                switch(linearized->simple[i].reduce_type) {
                    case(reduce_sum): {
                        for(uint64_t a = 0; a < linearized->simple[i].in_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].in_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].in_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].in_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] += %s[%lu];\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, 0, 0, 0, 0), linearized->simple[i].in_buffer.name, SIMPLE_INDEX(linearized->simple[i].in_buffer, a, z, y, x));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(reduce_avg): {
                        for(uint64_t a = 0; a < linearized->simple[i].in_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < linearized->simple[i].in_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < linearized->simple[i].in_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < linearized->simple[i].in_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%lu] += %s[%lu];\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, 0, 0, 0, 0), linearized->simple[i].in_buffer.name, SIMPLE_INDEX(linearized->simple[i].in_buffer, a, z, y, x));
                                        if(source_size - (curr - source) < max_op_size) {
                                            source_size *= 2;
                                            offset = curr - source;
                                            source = realloc(source, source_size);
                                            curr = source + offset;
                                        }
                                    }
                                }
                            }
                        }
                        curr += snprintf(curr, max_op_size, "%s[%lu] /= %lu;\n", linearized->simple[i].out_buffer.name, SIMPLE_INDEX(linearized->simple[i].out_buffer, 0, 0, 0, 0), linearized->simple[i].in_buffer.a_size * linearized->simple[i].in_buffer.z_size * linearized->simple[i].in_buffer.y_size * linearized->simple[i].in_buffer.x_size);
                        if(source_size - (curr - source) < max_op_size) {
                            source_size *= 2;
                            offset = curr - source;
                            source = realloc(source, source_size);
                            curr = source + offset;
                        }
                        break;
                    }
                    case(reduce_max): {
                        exit(2);
                        break;
                    }
                    case(reduce_min): {
                        exit(2);
                        break;
                    }
                }
                break;
            }
            case(operation_move): {
                fprintf(stderr, "ERROR: Tried to compile move operation for file %s at index %lu.\n", filename, i);
            }
        }
    }
    fwrite("void ", 1, 5, source_file);
    fwrite(func_name, 1, strlen(func_name), source_file);
    fwrite("(", 1, 1, source_file);
    for(int64_t i = 0; i < arg_num; i++) {
        fwrite("double *", 1, 8, source_file);
        fwrite(args[i], 1, strlen(args[i]), source_file);
        if(i != arg_num - 1) {
            fwrite(", ", 1, 2, source_file);
        }
        free(args[i]);
    }
    fwrite(") {\n", 1, 4, source_file);
    free(args);
    fwrite(source, 1, strlen(source), source_file);
    fwrite("}\n", 1, 2, source_file);
    fclose(source_file);
    free(source);
}
