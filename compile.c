#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compile.h"
#include "linearize.h"
#include "tensor.h"
#include "utils.h"

#define SIMPLE_INDEX(simple, a, z, y, x)                                                                                                                       \
    ((simple).a_stride * (a) + (simple).z_stride * (z) + (simple).y_stride * (y) + (simple).x_stride * (x) + (simple).offset)
#define SIMPLE_INDEX_(simple, a, z, y, x)                                                                                                                      \
    ((simple)->a_stride * (a) + (simple)->z_stride * (z) + (simple)->y_stride * (y) + (simple)->x_stride * (x) + (simple)->offset)
static void compile_loop_free(compile_loop_t *compile_loop) {
    for(uint64_t i = 0; i < compile_loop->loop_number; i++) { free(compile_loop->loop_instance[i]); }
    free(compile_loop->loop_instance);
}
/* TODO: Check if `compile_loop` has already been configured, by checking pointers for NULL. */
static void compile_loop_configure(compile_loop_t *compile_loop, simple_op_t **simple_op, uint64_t loop_length, uint64_t loop_number) {
    compile_loop->loop_length = loop_length;
    compile_loop->loop_number = loop_number;
    compile_loop->loop_instance = calloc(loop_number, sizeof(simple_op_t *));
    for(uint64_t i = 0; i < loop_number; i++) {
        compile_loop->loop_instance[i] = calloc(loop_length, sizeof(simple_op_t));
        for(uint64_t j = 0; j < loop_length; j++) { compile_loop->loop_instance[i][j] = simple_op[i][j]; }
    }
}
static void compile_loop_print(compile_loop_t *compile_loop, int padding, int offset, const char *name) {
    if(!strcmp(name, "")) {
        printf("%*scompile loop\n", offset, "");
    } else {
        printf("%*s %s\n", offset, "", name);
    }
    for(uint64_t i = 0; i < compile_loop->loop_number; i++) {
        if(i) { printf("\n"); }
        for(uint64_t j = 0; j < compile_loop->loop_length; j++) {
            printf("%*s[%lu, %lu] ", offset + padding, "", i, j);
            simple_op_print(&compile_loop->loop_instance[i][j], 0, 0, "");
        }
    }
}
/* Has to have the same input and output tensors, with the same shape and be the same op type. Offsets however should be irrelevant. */
static ALWAYS_INLINE bool compile_loop_simple_op_equal(simple_op_t *starting_op, simple_op_t *compared_op) {
    /* NOTE: This comparison is probably not needed technically. */
    if(starting_op->type != compared_op->type) { return false; }
    /* NOTE: Always checking every single one cuz it probably takes longer to go to the different cases. */
    if(starting_op->unary_type != compared_op->unary_type) { return false; }
    if(starting_op->binary_type != compared_op->binary_type) { return false; }
    if(starting_op->reduce_type != compared_op->reduce_type) { return false; }

    if(strncmp(starting_op->out_buffer.name, compared_op->out_buffer.name, BUFFER_NAME_SIZE)) { return false; }
    if(starting_op->out_buffer.a_size != compared_op->out_buffer.a_size) { return false; }
    if(starting_op->out_buffer.z_size != compared_op->out_buffer.z_size) { return false; }
    if(starting_op->out_buffer.y_size != compared_op->out_buffer.y_size) { return false; }
    if(starting_op->out_buffer.x_size != compared_op->out_buffer.x_size) { return false; }
    /* NOTE: Not just doing `if(starting_op->type)` here, because I might add another `operation_e` member which would break it if `operation_unary` is no longer 0. */
    if(starting_op->type != operation_unary) {
        if(strncmp(starting_op->in_buffer.name, compared_op->in_buffer.name, BUFFER_NAME_SIZE)) { return false; }
        if(starting_op->in_buffer.a_size != compared_op->in_buffer.a_size) { return false; }
        if(starting_op->in_buffer.z_size != compared_op->in_buffer.z_size) { return false; }
        if(starting_op->in_buffer.y_size != compared_op->in_buffer.y_size) { return false; }
        if(starting_op->in_buffer.x_size != compared_op->in_buffer.x_size) { return false; }
    }
    return true;
}
/* Returns the amount of ops in all the iterations of the loop combined, which makes it possible to use like `snprintf` for format-string appending. */
static uint64_t compile_loop_from_linearized_index(compile_loop_t *compile_loop, linearized_t *linearized, uint64_t start_index) {
    uint64_t loop_length = 0;
    uint64_t loop_number = 0;
    simple_op_t starting_op = linearized->simple[start_index];
    for(uint64_t i = start_index + 1; i < linearized->op_count; i++) {
        /* This is kind of a heuristic. I am 99% sure it works tho. */
        if(compile_loop_simple_op_equal(&starting_op, &linearized->simple[i])) {
            loop_length = i - start_index;
            break;
        }
    }
    if(!loop_length) { /* Could not find loop. */
        simple_op_t **loop_instances = calloc(1, sizeof(simple_op_t *));
        loop_instances[0] = calloc(1, sizeof(simple_op_t));
        loop_instances[0][0] = linearized->simple[start_index];
        compile_loop_configure(compile_loop, loop_instances, 1, 1);
        free(loop_instances[0]);
        free(loop_instances);
        return 1;
    }
    for(uint64_t i = start_index; i < linearized->op_count; i += loop_length) {
        if(compile_loop_simple_op_equal(&starting_op, &linearized->simple[i])) {
            loop_number++;
        } else {
            break;
        }
    }

    simple_op_t **loop_instances = calloc(loop_number, sizeof(simple_op_t *));
    for(uint64_t i = 0; i < loop_number; i++) { loop_instances[i] = calloc(loop_length, sizeof(simple_op_t)); }

    for(uint64_t i = 0; i < loop_number; i++) {
        for(uint64_t j = 0; j < loop_length; j++) { loop_instances[i][j] = linearized->simple[start_index + (loop_length * i) + j]; }
    }
    compile_loop_configure(compile_loop, loop_instances, loop_length, loop_number);

    for(uint64_t i = 0; i < loop_number; i++) { free(loop_instances[i]); }
    free(loop_instances);
    return loop_length * loop_number;
}
const uint64_t initial_source_size = 1000;
const uint64_t max_arg_size = 24;
const uint64_t max_index_digits = 9;
/* NOTE: Biggest I found was 131 for `max` or `min` binary ops. */
const uint64_t max_op_size = 320;
#define EXPAND_SOURCE_IF_NEEDED()                                                                                                                              \
    if(source_size - (curr - source) < max_op_size) {                                                                                                          \
        source_size *= 2;                                                                                                                                      \
        offset = curr - source;                                                                                                                                \
        source = realloc(source, source_size);                                                                                                                 \
        curr = source + offset;                                                                                                                                \
    }
/* Appends code for kernel that computes `compile_loop` utilizing `work_groups` work groups with `work_items` work items a piece. */
static void compile_loop_to_cl(const char *filename, compile_loop_t *compile_loop, uint64_t work_groups, uint64_t work_items) {
    uint64_t loops_per_workgroup = compile_loop->loop_number / work_groups;
    uint64_t leftover_loops = compile_loop->loop_number % work_groups;
    uint64_t source_size = initial_source_size;
    char *source = malloc(initial_source_size);
    char *curr = source;
    uint64_t global_id_counter = 0;
    uint64_t offset;

    /* TODO: Think about how to handle cases where `global_id` skips numbers, like with stride 2 convolutions. */
    /* TODO: Gather all args. */
    uint64_t arg_num = 1;
    char **args = malloc(arg_num * sizeof(char *));
    args[0] = malloc(max_arg_size * sizeof(char));

    FILE *f = fopen(filename, "a");

    for(uint64_t i = 0; i < compile_loop->loop_length; i++) {
        switch(compile_loop->loop_instance[0][i].type) {
            case(operation_unary): {
                curr +=
                    snprintf(curr, max_op_size,
                             "int %s_off_%lu = get_global_id(%lu) * %lu + get_global_id(%lu) * %lu + get_global_id(%lu) * %lu + get_global_id(%lu) * %lu;\n",
                             compile_loop->loop_instance[0][i].out_buffer.name, i, global_id_counter, compile_loop->loop_instance[0][i].out_buffer.a_stride,
                             global_id_counter + 1, compile_loop->loop_instance[0][i].out_buffer.z_stride, global_id_counter + 2,
                             compile_loop->loop_instance[0][i].out_buffer.y_stride, global_id_counter + 3, compile_loop->loop_instance[0][i].out_buffer.x_stride);
                EXPAND_SOURCE_IF_NEEDED();
                global_id_counter += 4;
                switch(compile_loop->loop_instance[0][i].unary_type) {
                    case(unary_add): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] += %.16lf;\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].var_unary);
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_subtract): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] -= %.16lf;\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].var_unary);
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_multiply): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] *= %.16lf;\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].var_unary);
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_divide): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] /= %.16lf;\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].var_unary);
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_exp): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] = exp(%s[%s_off_%lu + %lu]);\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_log): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] = log(%s[%s_off_%lu + %lu]);\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_square): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] *= %s[%s_off_%lu + %lu];\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_sqrt): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] = sqrt(%s[%s_off_%lu + %lu]);\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_negate): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] *= -1;\n", compile_loop->loop_instance[0][i].out_buffer.name,
                                                         compile_loop->loop_instance[0][i].out_buffer.name, i,
                                                         SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_reciprocal): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] = 1 / %s[%s_off_%lu + %lu];\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_max): {
                        TODO();
                        break;
                    }
                    case(unary_min): {
                        TODO();
                        break;
                    }
                    case(unary_set): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] = %.16lf;\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].var_unary);
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_random): {
                        TODO();
                        break;
                    }
                    case(unary_tanh): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] = tanh(%s[%s_off_%lu + %lu]);\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_absolute): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] = fabs(%s[%s_off_%lu + %lu]);\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(unary_sign): {
                        TODO();
                        break;
                    }
                }
                break;
            }
            case(operation_binary): {
                curr +=
                    snprintf(curr, max_op_size,
                             "int %s_off_%lu = get_global_id(%lu) * %lu + get_global_id(%lu) * %lu + get_global_id(%lu) * %lu + get_global_id(%lu) * %lu;\n",
                             compile_loop->loop_instance[0][i].out_buffer.name, i, global_id_counter, compile_loop->loop_instance[0][i].out_buffer.a_stride,
                             global_id_counter + 1, compile_loop->loop_instance[0][i].out_buffer.z_stride, global_id_counter + 2,
                             compile_loop->loop_instance[0][i].out_buffer.y_stride, global_id_counter + 3, compile_loop->loop_instance[0][i].out_buffer.x_stride);
                EXPAND_SOURCE_IF_NEEDED();
                global_id_counter += 4;
                curr +=
                    snprintf(curr, max_op_size,
                             "int %s_off_%lu = get_global_id(%lu) * %lu + get_global_id(%lu) * %lu + get_global_id(%lu) * %lu + get_global_id(%lu) * %lu;\n",
                             compile_loop->loop_instance[0][i].out_buffer.name, i, global_id_counter, compile_loop->loop_instance[0][i].out_buffer.a_stride,
                             global_id_counter + 1, compile_loop->loop_instance[0][i].out_buffer.z_stride, global_id_counter + 2,
                             compile_loop->loop_instance[0][i].out_buffer.y_stride, global_id_counter + 3, compile_loop->loop_instance[0][i].out_buffer.x_stride);
                EXPAND_SOURCE_IF_NEEDED();
                global_id_counter += 4;
                switch(compile_loop->loop_instance[0][i].binary_type) {
                    case(binary_add): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] += %s[%s_off_%lu + %lu];\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].in_buffer.name, compile_loop->loop_instance[0][i].in_buffer.name, i,
                                                         SIMPLE_INDEX(compile_loop->loop_instance[0][i].in_buffer, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_subtract): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] -= %s[%s_off_%lu + %lu];\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].in_buffer.name, compile_loop->loop_instance[0][i].in_buffer.name, i,
                                                         SIMPLE_INDEX(compile_loop->loop_instance[0][i].in_buffer, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_multiply): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] *= %s[%s_off_%lu + %lu];\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].in_buffer.name, compile_loop->loop_instance[0][i].in_buffer.name, i,
                                                         SIMPLE_INDEX(compile_loop->loop_instance[0][i].in_buffer, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_divide): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] /= %s[%s_off_%lu + %lu];\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].in_buffer.name, compile_loop->loop_instance[0][i].in_buffer.name, i,
                                                         SIMPLE_INDEX(compile_loop->loop_instance[0][i].in_buffer, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_max): {
                        TODO();
                        break;
                    }
                    case(binary_min): {
                        TODO();
                        break;
                    }
                    case(binary_copy): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] = %s[%s_off_%lu + %lu];\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].in_buffer.name, compile_loop->loop_instance[0][i].in_buffer.name, i,
                                                         SIMPLE_INDEX(compile_loop->loop_instance[0][i].in_buffer, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_add_like): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] += %s[%s_off_%lu];\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].in_buffer.name, compile_loop->loop_instance[0][i].in_buffer.name, i);
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_subtract_like): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] -= %s[%s_off_%lu];\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].in_buffer.name, compile_loop->loop_instance[0][i].in_buffer.name, i);
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_multiply_like): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] *= %s[%s_off_%lu];\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].in_buffer.name, compile_loop->loop_instance[0][i].in_buffer.name, i);
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_divide_like): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] /= %s[%s_off_%lu];\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].in_buffer.name, compile_loop->loop_instance[0][i].in_buffer.name, i);
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(binary_max_like): {
                        TODO();
                        break;
                    }
                    case(binary_min_like): {
                        TODO();
                        break;
                    }
                    case(binary_copy_like): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].out_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].out_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].out_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].out_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu + %lu] = %s[%s_off_%lu];\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].out_buffer, a, z, y, x),
                                                         compile_loop->loop_instance[0][i].in_buffer.name, compile_loop->loop_instance[0][i].in_buffer.name, i);
                                        EXPAND_SOURCE_IF_NEEDED();
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
                curr +=
                    snprintf(curr, max_op_size,
                             "int %s_off_%lu = get_global_id(%lu) * %lu + get_global_id(%lu) * %lu + get_global_id(%lu) * %lu + get_global_id(%lu) * %lu;\n",
                             compile_loop->loop_instance[0][i].out_buffer.name, i, global_id_counter, compile_loop->loop_instance[0][i].out_buffer.a_stride,
                             global_id_counter + 1, compile_loop->loop_instance[0][i].out_buffer.z_stride, global_id_counter + 2,
                             compile_loop->loop_instance[0][i].out_buffer.y_stride, global_id_counter + 3, compile_loop->loop_instance[0][i].out_buffer.x_stride);
                EXPAND_SOURCE_IF_NEEDED();
                global_id_counter += 4;
                curr +=
                    snprintf(curr, max_op_size,
                             "int %s_off_%lu = get_global_id(%lu) * %lu + get_global_id(%lu) * %lu + get_global_id(%lu) * %lu + get_global_id(%lu) * %lu;\n",
                             compile_loop->loop_instance[0][i].out_buffer.name, i, global_id_counter, compile_loop->loop_instance[0][i].out_buffer.a_stride,
                             global_id_counter + 1, compile_loop->loop_instance[0][i].out_buffer.z_stride, global_id_counter + 2,
                             compile_loop->loop_instance[0][i].out_buffer.y_stride, global_id_counter + 3, compile_loop->loop_instance[0][i].out_buffer.x_stride);
                EXPAND_SOURCE_IF_NEEDED();
                global_id_counter += 4;
                switch(compile_loop->loop_instance[0][i].reduce_type) {
                    case(reduce_sum): {
                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu] = %lf;\n", compile_loop->loop_instance[0][i].out_buffer.name,
                                         compile_loop->loop_instance[0][i].out_buffer.name, i, 0.0);
                        EXPAND_SOURCE_IF_NEEDED();
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].in_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].in_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].in_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].in_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu] += %s[%s_off_%lu + %lu];\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, compile_loop->loop_instance[0][i].in_buffer.name, compile_loop->loop_instance[0][i].in_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].in_buffer, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case(reduce_avg): {
                        for(uint64_t a = 0; a < compile_loop->loop_instance[0][i].in_buffer.a_size; a++) {
                            for(uint64_t z = 0; z < compile_loop->loop_instance[0][i].in_buffer.z_size; z++) {
                                for(uint64_t y = 0; y < compile_loop->loop_instance[0][i].in_buffer.y_size; y++) {
                                    for(uint64_t x = 0; x < compile_loop->loop_instance[0][i].in_buffer.x_size; x++) {
                                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu] += %s[%s_off_%lu + %lu];\n",
                                                         compile_loop->loop_instance[0][i].out_buffer.name, compile_loop->loop_instance[0][i].out_buffer.name,
                                                         i, compile_loop->loop_instance[0][i].in_buffer.name, compile_loop->loop_instance[0][i].in_buffer.name,
                                                         i, SIMPLE_INDEX(compile_loop->loop_instance[0][i].in_buffer, a, z, y, x));
                                        EXPAND_SOURCE_IF_NEEDED();
                                    }
                                }
                            }
                        }
                        curr += snprintf(curr, max_op_size, "%s[%s_off_%lu] /= %lf;\n", compile_loop->loop_instance[0][i].out_buffer.name,
                                         compile_loop->loop_instance[0][i].out_buffer.name, i,
                                         (double)compile_loop->loop_instance[0][i].in_buffer.a_size * compile_loop->loop_instance[0][i].in_buffer.z_size *
                                             compile_loop->loop_instance[0][i].in_buffer.y_size * compile_loop->loop_instance[0][i].in_buffer.x_size);
                        EXPAND_SOURCE_IF_NEEDED();
                        break;
                    }
                    case(reduce_max): {
                        TODO();
                        break;
                    }
                    case(reduce_min): {
                        TODO();
                        break;
                    }
                }
                break;
            }
            case(operation_move): {
                ERROR("ERROR: Tried to compile move operation to OpenCL at index %lu\n", i);
            }
        }
    }

    fwrite(source, sizeof(char), curr - source, f);
    fclose(f);

    free(source);
    for(uint64_t i = 0; i < arg_num; i++) {
        free(args[i]);
    }
    free(args);
}
void compile_linearized_to_cl(const char *filename, linearized_t *linearized) {
    compile_loop_t compile_loop;
    /* Clears file. */
    FILE *f = fopen(filename, "w");
    fclose(f);
    uint64_t i = compile_loop_from_linearized_index(&compile_loop, linearized, 0);
    printf("len %lu\n", i);
    compile_loop_print(&compile_loop, 4, 0, "");
    compile_loop_to_cl(filename, &compile_loop, 2, 3);
}
