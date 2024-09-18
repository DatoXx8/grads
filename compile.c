#include <CL/cl.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compile.h"
#include "tensor.h"
#include "utils.h"

#define INDEX(buffer, a, z, y, x)                                                                                      \
    ((buffer).str_a * (a) + (buffer).str_z * (z) + (buffer).str_y * (y) + (buffer).str_x * (x))
#define INDEX_(buffer, a, z, y, x)                                                                                     \
    ((buffer)->str_a * (a) + (buffer)->str_z * (z) + (buffer)->str_y * (y) + (buffer)->str_x * (x))
static void simple_loop_free(simple_loop_t *simple) {
    assert(simple);
    assert(simple->op);
    assert(simple->dim_info);
    free(simple->dim_info);
    free(simple->op);
    simple->dim_info = NULL;
    simple->op = NULL;
}
/* Has to have the same input and output tensors, with the same shape and be the same op type. Offsets however should be
 * irrelevant */
static uint64_t op_equal(const op_t *starting, const op_t *compared) {
    assert(starting);
    assert(compared);
    if(starting->type_op != compared->type_op) {
        return 0;
    }
    if(starting->type_unary != compared->type_unary) {
        return 0;
    }
    if(starting->type_binary != compared->type_binary) {
        return 0;
    }
    if(starting->type_reduce != compared->type_reduce) {
        return 0;
    }
    if(starting->buffer_out.name_off != compared->buffer_out.name_off) {
        return 0;
    }
    if(starting->buffer_out.sze_a != compared->buffer_out.sze_a) {
        return 0;
    }
    if(starting->buffer_out.sze_z != compared->buffer_out.sze_z) {
        return 0;
    }
    if(starting->buffer_out.sze_y != compared->buffer_out.sze_y) {
        return 0;
    }
    if(starting->buffer_out.sze_x != compared->buffer_out.sze_x) {
        return 0;
    }
    if(starting->type_op != op_unary) {
        if(starting->buffer_in.name_off != compared->buffer_in.name_off) {
            return 0;
        }
        if(starting->buffer_in.sze_a != compared->buffer_in.sze_a) {
            return 0;
        }
        if(starting->buffer_in.sze_z != compared->buffer_in.sze_z) {
            return 0;
        }
        if(starting->buffer_in.sze_y != compared->buffer_in.sze_y) {
            return 0;
        }
        if(starting->buffer_in.sze_x != compared->buffer_in.sze_x) {
            return 0;
        }
    }
    if(starting->var_unary != compared->var_unary) {
        return 0;
    }
    if(starting->var_a != compared->var_a) {
        return 0;
    }
    if(starting->var_z != compared->var_z) {
        return 0;
    }
    if(starting->var_y != compared->var_y) {
        return 0;
    }
    if(starting->var_x != compared->var_x) {
        return 0;
    }
    return 1;
}
static void simple_loop_configure(simple_loop_t *loop, const op_t **op, const uint64_t loop_len,
                                  const uint64_t loop_num) {
    assert(loop);
    assert(op);
    assert(loop_len > 0);
    assert(loop_num > 0);
    for(uint64_t i = 0; i < loop_num; i++) {
        assert(op[i]);
    }
    if(loop->op) {
        simple_loop_free(loop);
    }
    loop->loop_num = loop_num;
    loop->loop_len = loop_len;
    loop->op = calloc(loop_len, sizeof(op_t));
    assert(loop->op);
    loop->dim_info = calloc(loop_len, sizeof(dim_info_t));
    assert(loop->dim_info);
    /* TODO: Sort this sucker and error if the offsets can't be modelled by our 4 var model */
    /* Right now I am just assuming that the first op has the lowest offset */
    for(uint64_t op_idx = 0; op_idx < loop_len; op_idx++) {
        loop->op[op_idx] = op[0][op_idx];
        loop->dim_info[op_idx].off_out = loop->op[op_idx].buffer_out.off;
        if(loop->op[op_idx].type_op != op_unary) {
            loop->dim_info[op_idx].off_in = loop->op[op_idx].buffer_in.off;
        }
    }

    for(uint64_t op_idx = 0; op_idx < loop_len; op_idx++) {
        loop->dim_info[op_idx].str_a_out = 0;
        loop->dim_info[op_idx].str_z_out = 0;
        loop->dim_info[op_idx].str_y_out = 0;
        loop->dim_info[op_idx].str_x_out = 0;
        loop->dim_info[op_idx].wai_a_out = loop_len;
        loop->dim_info[op_idx].wai_z_out = loop_len;
        loop->dim_info[op_idx].wai_y_out = loop_len;
        loop->dim_info[op_idx].wai_x_out = loop_len;
        uint64_t a_offset_base = op[0][op_idx].buffer_out.off_a;
        uint64_t z_offset_base = op[0][op_idx].buffer_out.off_z;
        uint64_t y_offset_base = op[0][op_idx].buffer_out.off_y;
        uint64_t x_offset_base = op[0][op_idx].buffer_out.off_x;
        uint64_t a_offset_found = 0;
        uint64_t z_offset_found = 0;
        uint64_t y_offset_found = 0;
        uint64_t x_offset_found = 0;
        /* Currently assuming all strides are positive */
        for(uint64_t loop_idx = 0; loop_idx < loop_num; loop_idx++) {
            if(!a_offset_found && op[loop_idx][op_idx].buffer_out.off_a > a_offset_base) {
                a_offset_found = 1;
                loop->dim_info[op_idx].str_a_out = op[loop_idx][op_idx].buffer_out.off_a - a_offset_base;
                loop->dim_info[op_idx].wai_a_out = loop_idx;
            }
            if(!z_offset_found && op[loop_idx][op_idx].buffer_out.off_z > z_offset_base) {
                z_offset_found = 1;
                loop->dim_info[op_idx].str_z_out = op[loop_idx][op_idx].buffer_out.off_z - z_offset_base;
                loop->dim_info[op_idx].wai_z_out = loop_idx;
            }
            if(!y_offset_found && op[loop_idx][op_idx].buffer_out.off_y > y_offset_base) {
                y_offset_found = 1;
                loop->dim_info[op_idx].str_y_out = op[loop_idx][op_idx].buffer_out.off_y - y_offset_base;
                loop->dim_info[op_idx].wai_y_out = loop_idx;
            }
            if(!x_offset_found && op[loop_idx][op_idx].buffer_out.off_x > x_offset_base) {
                x_offset_found = 1;
                loop->dim_info[op_idx].str_x_out = op[loop_idx][op_idx].buffer_out.off_x - x_offset_base;
                loop->dim_info[op_idx].wai_x_out = loop_idx;
            }
        }

        loop->dim_info[op_idx].str_a_in = 0;
        loop->dim_info[op_idx].str_z_in = 0;
        loop->dim_info[op_idx].str_y_in = 0;
        loop->dim_info[op_idx].str_x_in = 0;
        loop->dim_info[op_idx].wai_a_in = loop_len;
        loop->dim_info[op_idx].wai_z_in = loop_len;
        loop->dim_info[op_idx].wai_y_in = loop_len;
        loop->dim_info[op_idx].wai_x_in = loop_len;
        if(op[0][op_idx].type_op != op_unary) {
            a_offset_base = op[0][op_idx].buffer_in.off_a;
            z_offset_base = op[0][op_idx].buffer_in.off_z;
            y_offset_base = op[0][op_idx].buffer_in.off_y;
            x_offset_base = op[0][op_idx].buffer_in.off_x;
            a_offset_found = 0;
            z_offset_found = 0;
            y_offset_found = 0;
            x_offset_found = 0;
            for(uint64_t loop_idx = 0; loop_idx < loop_num; loop_idx++) {
                if(!a_offset_found && op[loop_idx][op_idx].buffer_in.off_a > a_offset_base) {
                    a_offset_found = 1;
                    loop->dim_info[op_idx].str_a_in = op[loop_idx][op_idx].buffer_in.off_a - a_offset_base;
                    loop->dim_info[op_idx].wai_a_in = loop_idx;
                }
                if(!z_offset_found && op[loop_idx][op_idx].buffer_in.off_z > z_offset_base) {
                    z_offset_found = 1;
                    loop->dim_info[op_idx].str_z_in = op[loop_idx][op_idx].buffer_in.off_z - z_offset_base;
                    loop->dim_info[op_idx].wai_z_in = loop_idx;
                }
                if(!y_offset_found && op[loop_idx][op_idx].buffer_in.off_y > y_offset_base) {
                    y_offset_found = 1;
                    loop->dim_info[op_idx].str_y_in = op[loop_idx][op_idx].buffer_in.off_y - y_offset_base;
                    loop->dim_info[op_idx].wai_y_in = loop_idx;
                }
                if(!x_offset_found && op[loop_idx][op_idx].buffer_in.off_x > x_offset_base) {
                    x_offset_found = 1;
                    loop->dim_info[op_idx].str_x_in = op[loop_idx][op_idx].buffer_in.off_x - x_offset_base;
                    loop->dim_info[op_idx].wai_x_in = loop_idx;
                }
            }
        }
    }
    for(uint64_t op_idx = 0; op_idx < loop_len; op_idx++) {
        /* TODO: Compute the actual size (product of dim_sizes) * loop_num */
        // loop->dim_info[op_idx].res_a_out = loop_len;
        // loop->dim_info[op_idx].res_z_out = loop_len;
        // loop->dim_info[op_idx].res_y_out = loop_len;
        // loop->dim_info[op_idx].res_x_out = loop_len;
        loop->dim_info[op_idx].res_a_out = 1 << 24;
        loop->dim_info[op_idx].res_z_out = 1 << 24;
        loop->dim_info[op_idx].res_y_out = 1 << 24;
        loop->dim_info[op_idx].res_x_out = 1 << 24;
        uint64_t a_offset_base = op[0][op_idx].buffer_out.off_a;
        uint64_t z_offset_base = op[0][op_idx].buffer_out.off_z;
        uint64_t y_offset_base = op[0][op_idx].buffer_out.off_y;
        uint64_t x_offset_base = op[0][op_idx].buffer_out.off_x;
        uint64_t a_offset_left = 0;
        uint64_t z_offset_left = 0;
        uint64_t y_offset_left = 0;
        uint64_t x_offset_left = 0;
        uint64_t a_offset_found = 0;
        uint64_t z_offset_found = 0;
        uint64_t y_offset_found = 0;
        uint64_t x_offset_found = 0;
        for(uint64_t loop_idx = 0; loop_idx < loop_num; loop_idx++) {
            /* Don't know if the +1 is right here */
            if(!a_offset_found && a_offset_left && op[loop_idx][op_idx].buffer_out.off_a == a_offset_base) {
                a_offset_found = 1;
                loop->dim_info[op_idx].res_a_out = loop_idx;
            }
            if(!a_offset_left && op[loop_idx][op_idx].buffer_out.off_a > a_offset_base) {
                a_offset_left = 1;
            }
            if(!z_offset_found && z_offset_left && op[loop_idx][op_idx].buffer_out.off_z == z_offset_base) {
                z_offset_found = 1;
                loop->dim_info[op_idx].res_z_out = loop_idx;
            }
            if(!z_offset_left && op[loop_idx][op_idx].buffer_out.off_z > z_offset_base) {
                z_offset_left = 1;
            }
            if(!y_offset_found && y_offset_left && op[loop_idx][op_idx].buffer_out.off_y == y_offset_base) {
                y_offset_found = 1;
                loop->dim_info[op_idx].res_y_out = loop_idx;
            }
            if(!y_offset_left && op[loop_idx][op_idx].buffer_out.off_y > y_offset_base) {
                y_offset_left = 1;
            }
            if(!x_offset_found && x_offset_left && op[loop_idx][op_idx].buffer_out.off_x == x_offset_base) {
                x_offset_found = 1;
                loop->dim_info[op_idx].res_x_out = loop_idx;
            }
            if(!x_offset_left && op[loop_idx][op_idx].buffer_out.off_x > x_offset_base) {
                x_offset_left = 1;
            }
        }

        if(op[0][op_idx].type_op != op_unary) {
            a_offset_base = op[0][op_idx].buffer_in.off_a;
            z_offset_base = op[0][op_idx].buffer_in.off_z;
            y_offset_base = op[0][op_idx].buffer_in.off_y;
            x_offset_base = op[0][op_idx].buffer_in.off_x;
            // loop->dim_info[op_idx].res_a_in = loop_len;
            // loop->dim_info[op_idx].res_z_in = loop_len;
            // loop->dim_info[op_idx].res_y_in = loop_len;
            // loop->dim_info[op_idx].res_x_in = loop_len;
            loop->dim_info[op_idx].res_a_in = 1 << 24;
            loop->dim_info[op_idx].res_z_in = 1 << 24;
            loop->dim_info[op_idx].res_y_in = 1 << 24;
            loop->dim_info[op_idx].res_x_in = 1 << 24;
            a_offset_left = 0;
            z_offset_left = 0;
            y_offset_left = 0;
            x_offset_left = 0;
            a_offset_found = 0;
            z_offset_found = 0;
            y_offset_found = 0;
            x_offset_found = 0;
            for(uint64_t loop_idx = 0; loop_idx < loop_num; loop_idx++) {
                if(!a_offset_found && a_offset_left && op[loop_idx][op_idx].buffer_in.off_a == a_offset_base) {
                    a_offset_found = 1;
                    loop->dim_info[op_idx].res_a_in = loop_idx;
                }
                if(!a_offset_left && op[loop_idx][op_idx].buffer_in.off_a > a_offset_base) {
                    a_offset_left = 1;
                }
                if(!z_offset_found && z_offset_left && op[loop_idx][op_idx].buffer_in.off_z == z_offset_base) {
                    z_offset_found = 1;
                    loop->dim_info[op_idx].res_z_in = loop_idx;
                }
                if(!z_offset_left && op[loop_idx][op_idx].buffer_in.off_z > z_offset_base) {
                    z_offset_left = 1;
                }
                if(!y_offset_found && y_offset_left && op[loop_idx][op_idx].buffer_in.off_y == y_offset_base) {
                    y_offset_found = 1;
                    loop->dim_info[op_idx].res_y_in = loop_idx;
                }
                if(!y_offset_left && op[loop_idx][op_idx].buffer_in.off_y > y_offset_base) {
                    y_offset_left = 1;
                }
                if(!x_offset_found && x_offset_left && op[loop_idx][op_idx].buffer_in.off_x == x_offset_base) {
                    x_offset_found = 1;
                    loop->dim_info[op_idx].res_x_in = loop_idx;
                }
                if(!x_offset_left && op[loop_idx][op_idx].buffer_in.off_x > x_offset_base) {
                    x_offset_left = 1;
                }
            }
        }
    }
    /* TODO: Assert that the actual indices get accurately modeled by the model */
}
/* Returns the amount of ops in all the iterations of the loop combined, which makes it possible to use like `snprintf`
 * for format string appending */
static uint64_t simple_loop_from_linearized_index(simple_loop_t *simple, const linearized_t *linearized,
                                                  const uint64_t start_idx) {
    assert(simple);
    assert(linearized);
    assert(start_idx >= 0);
    assert(start_idx < linearized->op_len);
    uint64_t loop_length = 0;
    uint64_t loop_number = 0;
    uint64_t diff;
    op_t starting_op = linearized->op[start_idx];
    for(uint64_t i = start_idx + 1; i < linearized->op_len; i++) {
        if(op_equal(&starting_op, &linearized->op[i])) {
            /* TODO: This could probably just be done in the `for` statement */
            if(2 * i - start_idx < linearized->op_len) {
                diff = 0;
                for(uint64_t j = 0; j < i - start_idx; j++) {
                    if(!op_equal(&linearized->op[start_idx + j], &linearized->op[i + j])) {
                        diff = 1;
                        break;
                    }
                }
                if(!diff) {
                    loop_length = i - start_idx;
                    break;
                }
            } else {
                break;
            }
        }
    }
    if(!loop_length) { /* Could not find loop */
        op_t **loop_instances = calloc(1, sizeof(op_t *));
        assert(loop_instances);
        loop_instances[0] = calloc(1, sizeof(op_t));
        assert(loop_instances[0]);
        loop_instances[0][0] = linearized->op[start_idx];
        simple_loop_configure(simple, (const op_t **) loop_instances, 1, 1);
        free(loop_instances[0]);
        free(loop_instances);
        return 1;
    }
    for(uint64_t i = start_idx; i < linearized->op_len; i += loop_length) {
        if(op_equal(&starting_op, &linearized->op[i])) {
            loop_number++;
        } else {
            break;
        }
    }
    assert(loop_number > 0);

    op_t **loop_instances = calloc(loop_number, sizeof(op_t *));
    assert(loop_instances);
    for(uint64_t i = 0; i < loop_number; i++) {
        loop_instances[i] = calloc(loop_length, sizeof(op_t));
        assert(loop_instances[i]);
    }

    for(uint64_t i = 0; i < loop_number; i++) {
        for(uint64_t j = 0; j < loop_length; j++) {
            loop_instances[i][j] = linearized->op[start_idx + (loop_length * i) + j];
        }
    }
    simple_loop_configure(simple, (const op_t **) loop_instances, loop_length, loop_number);

    for(uint64_t i = 0; i < loop_number; i++) {
        free(loop_instances[i]);
    }
    free(loop_instances);

    return loop_length * loop_number;
}
const uint64_t INITIAL_CAP = 4;
#define INLINABLE_OVERRIDE(op)                                                                                         \
    ((op).type_op == op_unary && ((op).type_unary == unary_set)) ||                                                    \
        ((op).type_op == op_binary && ((op).type_binary == binary_copy || (op).type_binary == binary_copy_like))
#define INLINABLE_OVERRIDE_(op)                                                                                        \
    ((op)->type_op == op_unary && ((op)->type_unary == unary_set)) ||                                                  \
        ((op)->type_op == op_binary && ((op)->type_binary == binary_copy || (op)->type_binary == binary_copy_like))

#define OVERRIDES_OUTPUT(op)                                                                                           \
    (((op).type_op == op_unary && ((op).type_unary == unary_set)) ||                                                   \
     ((op).type_op == op_binary && ((op).type_binary == binary_copy || (op).type_binary == binary_copy_like)) ||       \
     ((op).type_op == op_reduce))
#define OVERRIDES_OUTPUT_(op)                                                                                          \
    (((op)->op_type == op_unary && ((op)->type_unary == unary_set)) ||                                                 \
     ((op)->op_type == op_binary && ((op)->type_binary == binary_copy || (op)->type_binary == binary_copy_like)) ||    \
     ((op)->op_type == op_reduce))

#define SPLITTABLE(op) ((op).type_op != op_reduce)
#define SPLITTABLE_(op) ((op)->type_op != op_reduce)

static void compile_loop_optimize(compile_loop_t *compile) {
    assert(compile);
    /* Inline */
    uint64_t *delete = calloc(compile->op_num, sizeof(uint64_t));
    assert(delete);
    for(uint64_t op_idx = 0; op_idx < compile->op_num; op_idx++) {
        assert(compile->inline_num[op_idx] == 1);
        if(delete[op_idx]) {
            continue;
        }
        if(INLINABLE_OVERRIDE(compile->op[op_idx][0])) {
            for(uint64_t search_off = 1; search_off < compile->op_num - op_idx; search_off++) {
                assert(compile->inline_num[op_idx + search_off] == 1);
                if(compile->op[op_idx][0].buffer_out.name_off ==
                   compile->op[op_idx + search_off][0].buffer_out.name_off) {
                    if(OVERRIDES_OUTPUT(compile->op[op_idx + search_off][0])) {
                        break;
                    } else {
                        uint64_t inline_cap = compile->inline_cap[op_idx];
                        uint64_t inline_num = compile->inline_num[op_idx];
                        compile->op[op_idx][inline_num] = compile->op[op_idx + search_off][0];
                        compile->dim_info[op_idx][inline_num] = compile->dim_info[op_idx + search_off][0];
                        compile->inline_type[op_idx][inline_num] = inline_op_out;
                        delete[op_idx + search_off] = 1;
                        inline_num++;
                        if(inline_num == inline_cap) {
                            inline_cap *= 2;
                            compile->op[op_idx] = reallocarray(compile->op[op_idx], inline_cap, sizeof(op_t));
                            compile->dim_info[op_idx] =
                                reallocarray(compile->dim_info[op_idx], inline_cap, sizeof(dim_info_t));
                            compile->inline_type[op_idx] =
                                reallocarray(compile->inline_type[op_idx], inline_cap, sizeof(inline_op_e));
                            assert(compile->op[op_idx]);
                            assert(compile->dim_info[op_idx]);
                            assert(compile->inline_type[op_idx]);
                        }
                        compile->inline_cap[op_idx] = inline_cap;
                        compile->inline_num[op_idx] = inline_num;
                    }
                } else if(compile->op[op_idx][0].buffer_out.name_off ==
                          compile->op[op_idx + search_off][0].buffer_in.name_off) {
                    uint64_t inline_cap = compile->inline_cap[op_idx];
                    uint64_t inline_num = compile->inline_num[op_idx];
                    compile->op[op_idx][inline_num] = compile->op[op_idx + search_off][0];
                    compile->dim_info[op_idx][inline_num] = compile->dim_info[op_idx + search_off][0];
                    compile->inline_type[op_idx][inline_num] = inline_op_in;
                    delete[op_idx + search_off] = 1;
                    inline_num++;
                    if(inline_num == inline_cap) {
                        inline_cap *= 2;
                        compile->op[op_idx] = reallocarray(compile->op[op_idx], inline_cap, sizeof(op_t));
                        compile->dim_info[op_idx] =
                            reallocarray(compile->dim_info[op_idx], inline_cap, sizeof(dim_info_t));
                        compile->inline_type[op_idx] =
                            reallocarray(compile->inline_type[op_idx], inline_cap, sizeof(inline_op_e));
                        assert(compile->op[op_idx]);
                        assert(compile->dim_info[op_idx]);
                        assert(compile->inline_type[op_idx]);
                    }
                    compile->inline_cap[op_idx] = inline_cap;
                    compile->inline_num[op_idx] = inline_num;
                }
            }
        }
    }
    uint64_t new_idx = 0;
    uint64_t new_num = compile->op_num;
    for(uint64_t op_idx = 0; op_idx < compile->op_num; op_idx++) {
        if(delete[op_idx]) {
            free(compile->op[op_idx]);
            free(compile->dim_info[op_idx]);
            free(compile->inline_type[op_idx]);
            new_num--;
        } else {
            compile->op[new_idx] = compile->op[op_idx];
            compile->dim_info[new_idx] = compile->dim_info[op_idx];
            compile->inline_type[new_idx] = compile->inline_type[op_idx];
            compile->inline_num[new_idx] = compile->inline_num[op_idx];
            compile->inline_cap[new_idx] = compile->inline_cap[op_idx];
            op_t *op = calloc(compile->inline_num[new_idx], sizeof(op_t));
            dim_info_t *dim_info = calloc(compile->inline_num[new_idx], sizeof(dim_info_t));
            inline_op_e *inline_type = calloc(compile->inline_num[new_idx], sizeof(dim_info_t));
            for(uint64_t swap_idx = 0; swap_idx < compile->inline_num[new_idx]; swap_idx++) {
                op[swap_idx] = compile->op[new_idx][swap_idx];
                dim_info[swap_idx] = compile->dim_info[new_idx][swap_idx];
                inline_type[swap_idx] = compile->inline_type[new_idx][swap_idx];
            }
            for(uint64_t swap_idx = 0; swap_idx < compile->inline_num[new_idx]; swap_idx++) {
                compile->op[new_idx][swap_idx] = op[compile->inline_num[new_idx] - swap_idx - 1];
                compile->dim_info[new_idx][swap_idx] = dim_info[compile->inline_num[new_idx] - swap_idx - 1];
                compile->inline_type[new_idx][swap_idx] = inline_type[compile->inline_num[new_idx] - swap_idx - 1];
            }
            new_idx++;
            free(op);
            free(dim_info);
            free(inline_type);
        }
    }
    compile->op_num = new_num;
    free(delete);
    /* Fuse */
}
static void compile_loop_print(compile_loop_t *compile, int padding, int offset, const char *name) {
    assert(compile);
    if(strncmp(name, "", 1) != 0) {
        printf("%*s%s\n", offset, "", name);
    } else {
        printf("%*scompile loop with %lu iterations\n", offset, "", compile->loop_num);
    }
    for(uint64_t op_idx = 0; op_idx < compile->op_num; op_idx++) {
        printf("%*s%d ", offset + padding, "", compile->inline_type[op_idx][0]);
        op_print(&compile->op[op_idx][0], 0, 0, "");
        printf("%*so off %lu str {%lu, %lu, %lu, %lu} wai {%lu, %lu, %lu, %lu} res {%lu, %lu, %lu, %lu}\n",
               offset + padding, "", compile->dim_info[op_idx][0].off_out, compile->dim_info[op_idx][0].str_a_out,
               compile->dim_info[op_idx][0].str_z_out, compile->dim_info[op_idx][0].str_y_out,
               compile->dim_info[op_idx][0].str_x_out, compile->dim_info[op_idx][0].wai_a_out,
               compile->dim_info[op_idx][0].wai_z_out, compile->dim_info[op_idx][0].wai_y_out,
               compile->dim_info[op_idx][0].wai_x_out, compile->dim_info[op_idx][0].res_a_out,
               compile->dim_info[op_idx][0].res_z_out, compile->dim_info[op_idx][0].res_y_out,
               compile->dim_info[op_idx][0].res_x_out);
        if(compile->op[op_idx][0].type_op != op_unary) {
            printf("%*si off %lu str {%lu, %lu, %lu, %lu} wai {%lu, %lu, %lu, %lu} res {%lu, %lu, %lu, %lu}\n",
                   offset + padding, "", compile->dim_info[op_idx][0].off_in, compile->dim_info[op_idx][0].str_a_in,
                   compile->dim_info[op_idx][0].str_z_in, compile->dim_info[op_idx][0].str_y_in,
                   compile->dim_info[op_idx][0].str_x_in, compile->dim_info[op_idx][0].wai_a_in,
                   compile->dim_info[op_idx][0].wai_z_in, compile->dim_info[op_idx][0].wai_y_in,
                   compile->dim_info[op_idx][0].wai_x_in, compile->dim_info[op_idx][0].res_a_in,
                   compile->dim_info[op_idx][0].res_z_in, compile->dim_info[op_idx][0].res_y_in,
                   compile->dim_info[op_idx][0].res_x_in);
        }
        for(uint64_t inline_idx = 1; inline_idx < compile->inline_num[op_idx]; inline_idx++) {
            printf("%*s%d ", offset + 2 * padding, "", compile->inline_type[op_idx][inline_idx]);
            op_print(&compile->op[op_idx][inline_idx], 0, 0, "");
            printf("%*so off %lu str {%lu, %lu, %lu, %lu} wai {%lu, %lu, %lu, %lu} res {%lu, %lu, %lu, %lu}\n",
                   offset + 2 * padding, "", compile->dim_info[op_idx][inline_idx].off_out,
                   compile->dim_info[op_idx][inline_idx].str_a_out, compile->dim_info[op_idx][inline_idx].str_z_out,
                   compile->dim_info[op_idx][inline_idx].str_y_out, compile->dim_info[op_idx][inline_idx].str_x_out,
                   compile->dim_info[op_idx][inline_idx].wai_a_out, compile->dim_info[op_idx][inline_idx].wai_z_out,
                   compile->dim_info[op_idx][inline_idx].wai_y_out, compile->dim_info[op_idx][inline_idx].wai_x_out,
                   compile->dim_info[op_idx][inline_idx].res_a_out, compile->dim_info[op_idx][inline_idx].res_z_out,
                   compile->dim_info[op_idx][inline_idx].res_y_out, compile->dim_info[op_idx][inline_idx].res_x_out);
            if(compile->op[op_idx][0].type_op != op_unary) {
                printf("%*si off %lu str {%lu, %lu, %lu, %lu} wai {%lu, %lu, %lu, %lu} res {%lu, %lu, %lu, %lu}\n",
                       offset + 2 * padding, "", compile->dim_info[op_idx][inline_idx].off_in,
                       compile->dim_info[op_idx][inline_idx].str_a_in, compile->dim_info[op_idx][inline_idx].str_z_in,
                       compile->dim_info[op_idx][inline_idx].str_y_in, compile->dim_info[op_idx][inline_idx].str_x_in,
                       compile->dim_info[op_idx][inline_idx].wai_a_in, compile->dim_info[op_idx][inline_idx].wai_z_in,
                       compile->dim_info[op_idx][inline_idx].wai_y_in, compile->dim_info[op_idx][inline_idx].wai_x_in,
                       compile->dim_info[op_idx][inline_idx].res_a_in, compile->dim_info[op_idx][inline_idx].res_z_in,
                       compile->dim_info[op_idx][inline_idx].res_y_in, compile->dim_info[op_idx][inline_idx].res_x_in);
            }
        }
    }
}
static void compile_loop_free(compile_loop_t *compile) {
    assert(compile);
    assert(compile->op);
    assert(compile->dim_info);
    assert(compile->inline_type);
    assert(compile->inline_num);
    assert(compile->inline_cap);
    for(uint64_t op_idx = 0; op_idx < compile->op_num; op_idx++) {
        assert(compile->op[op_idx]);
        assert(compile->inline_type[op_idx]);
        assert(compile->dim_info[op_idx]);
        free(compile->op[op_idx]);
        free(compile->dim_info[op_idx]);
        free(compile->inline_type[op_idx]);
    }
    free(compile->op);
    free(compile->inline_num);
    free(compile->inline_cap);
    free(compile->dim_info);
    free(compile->inline_type);
    compile->op = NULL;
    compile->inline_num = NULL;
    compile->inline_cap = NULL;
    compile->dim_info = NULL;
    compile->inline_type = NULL;
}
static compile_loop_t compile_loop_alloc(const simple_loop_t *simple) {
    assert(simple);
    assert(simple->loop_len > 0);
    assert(simple->loop_num > 0);
    compile_loop_t compile = {
        .op_num = simple->loop_len,
        .loop_num = simple->loop_num,
        .op = NULL,
        .inline_num = NULL,
    };
    compile.inline_num = calloc(compile.op_num, sizeof(uint64_t));
    compile.inline_cap = calloc(compile.op_num, sizeof(uint64_t));
    compile.op = calloc(compile.op_num, sizeof(op_t *));
    compile.dim_info = calloc(compile.op_num, sizeof(dim_info_t *));
    compile.inline_type = calloc(compile.op_num, sizeof(inline_op_e *));
    assert(compile.inline_num);
    assert(compile.inline_cap);
    assert(compile.op);
    assert(compile.dim_info);
    assert(compile.inline_type);
    for(uint64_t op_idx = 0; op_idx < compile.op_num; op_idx++) {
        compile.inline_num[op_idx] = 1;
        compile.inline_cap[op_idx] = INITIAL_CAP;

        compile.op[op_idx] = calloc(INITIAL_CAP, sizeof(op_t));
        assert(compile.op[op_idx]);
        compile.op[op_idx][0] = simple->op[op_idx];

        compile.dim_info[op_idx] = calloc(INITIAL_CAP, sizeof(dim_info_t));
        assert(compile.dim_info[op_idx]);
        compile.dim_info[op_idx][0] = simple->dim_info[op_idx];
        compile.dim_info[op_idx][0] = simple->dim_info[op_idx];

        compile.inline_type[op_idx] = calloc(INITIAL_CAP, sizeof(inline_op_e));
        assert(compile.inline_type[op_idx]);
        compile.inline_type[op_idx][0] = inline_op_none;
    }
    // compile_loop_print(&compile, 4, 0, "");
    /* TODO: Optimizer breaks it sometimes. THIS ONE IS REALLY IMPORTANT */
    // compile_loop_optimize(&compile);
    return compile;
}
static const uint64_t INITIAL_SOURCE_SIZE = 12500;
static const uint64_t MAX_OP_SIZE = 1000;
static inline void compile_expand_source(char **source, char **source_curr, uint64_t *source_cap,
                                         const uint64_t padding) {
    uint64_t source_off = *source_curr - *source;
    while(*source_cap - (*source_curr - *source) <= padding) {
        *source_cap *= 2;
    }
    *source = reallocarray(*source, *source_cap, sizeof(char *));
    assert(*source);
    *source_curr = *source + source_off;
}
static void compile_loop_gather_args(kernel_t *kernel, const compile_loop_t *compile) {
    assert(kernel);
    assert(compile);
    uint64_t arg_num = 0;
    uint64_t arg_cap = INITIAL_CAP;
    char **arg_name = calloc(INITIAL_CAP, sizeof(char *));
    uint64_t *arg_name_off = calloc(INITIAL_CAP, sizeof(uint64_t));
    cl_mem *arg_mem = calloc(INITIAL_CAP, sizeof(cl_mem));
    assert(arg_name);
    assert(arg_name_off);
    assert(arg_mem);
    for(uint64_t op_idx = 0; op_idx < compile->op_num; op_idx++) {
        uint64_t found;
        /* Out */
        found = 0;
        for(uint64_t arg_idx = 0; arg_idx < arg_num; arg_idx++) {
            if(arg_name_off[arg_idx] == compile->op[op_idx][0].buffer_out.name_off) {
                found = 1;
                break;
            }
        }
        if(!found) {
            if(arg_num == arg_cap) {
                arg_cap *= 2;
                arg_name = reallocarray(arg_name, arg_cap, sizeof(char *));
                arg_name_off = reallocarray(arg_name_off, arg_cap, sizeof(uint64_t));
                arg_mem = reallocarray(arg_mem, arg_cap, sizeof(cl_mem));
                assert(arg_name);
                assert(arg_name_off);
                assert(arg_mem);
            }
            arg_name[arg_num] = strndup(compile->op[op_idx][0].buffer_out.name, BUFFER_NAME_SIZE + 1);
            arg_name_off[arg_num] = compile->op[op_idx][0].buffer_out.name_off;
            assert(arg_name[arg_num]);
            arg_mem[arg_num] = compile->op[op_idx][0].buffer_out.val_cl;
            arg_num++;
        }
        if(compile->op[op_idx][0].type_op != op_unary) {
            /* In */
            found = 0;
            for(uint64_t arg_idx = 0; arg_idx < arg_num; arg_idx++) {
                if(arg_name_off[arg_idx] == compile->op[op_idx][0].buffer_in.name_off) {
                    found = 1;
                    break;
                }
            }
            if(!found) {
                if(arg_num == arg_cap) {
                    arg_cap *= 2;
                    arg_name = reallocarray(arg_name, arg_cap, sizeof(char *));
                    arg_name_off = reallocarray(arg_name_off, arg_cap, sizeof(uint64_t));
                    arg_mem = reallocarray(arg_mem, arg_cap, sizeof(cl_mem));
                    assert(arg_name);
                    assert(arg_name_off);
                    assert(arg_mem);
                }
                arg_name[arg_num] = strndup(compile->op[op_idx][0].buffer_in.name, BUFFER_NAME_SIZE + 1);
                arg_name_off[arg_num] = compile->op[op_idx][0].buffer_in.name_off;
                assert(arg_name[arg_num]);
                arg_mem[arg_num] = compile->op[op_idx][0].buffer_in.val_cl;
                arg_num++;
            }
        }
        for(uint64_t inline_idx = 1; inline_idx < compile->inline_num[op_idx]; inline_idx++) {
            if(compile->op[op_idx][inline_idx].type_op == op_unary) {
                /* Out */
                found = 0;
                for(uint64_t arg_idx = 0; arg_idx < arg_num; arg_idx++) {
                    if(arg_name_off[arg_idx] != compile->op[op_idx][inline_idx].buffer_out.name_off) {
                        found = 1;
                        break;
                    }
                }
                if(!found) {
                    if(arg_num == arg_cap) {
                        arg_cap *= 2;
                        arg_name = reallocarray(arg_name, arg_cap, sizeof(char *));
                        arg_name_off = reallocarray(arg_name_off, arg_cap, sizeof(uint64_t));
                        arg_mem = reallocarray(arg_mem, arg_cap, sizeof(cl_mem));
                        assert(arg_name);
                        assert(arg_name_off);
                        assert(arg_mem);
                    }
                    arg_name[arg_num] = strndup(compile->op[op_idx][inline_idx].buffer_out.name, BUFFER_NAME_SIZE + 1);
                    arg_name_off[arg_num] = compile->op[op_idx][inline_idx].buffer_out.name_off;
                    assert(arg_name[arg_num]);
                    arg_mem[arg_num] = compile->op[op_idx][inline_idx].buffer_out.val_cl;
                    arg_num++;
                }
            } else {
                /* In */
                found = 0;
                for(uint64_t arg_idx = 0; arg_idx < arg_num; arg_idx++) {
                    if(arg_name_off[arg_idx] == compile->op[op_idx][inline_idx].buffer_in.name_off) {
                        found = 1;
                        break;
                    }
                }
                if(!found) {
                    if(arg_num == arg_cap) {
                        arg_cap *= 2;
                        arg_name = reallocarray(arg_name, arg_cap, sizeof(char *));
                        arg_name_off = reallocarray(arg_name_off, arg_cap, sizeof(uint64_t));
                        arg_mem = reallocarray(arg_mem, arg_cap, sizeof(cl_mem));
                        assert(arg_name);
                        assert(arg_name_off);
                        assert(arg_mem);
                    }
                    arg_name[arg_num] = strndup(compile->op[op_idx][inline_idx].buffer_in.name, BUFFER_NAME_SIZE + 1);
                    arg_name_off[arg_num] = compile->op[op_idx][inline_idx].buffer_in.name_off;
                    assert(arg_name[arg_num]);
                    arg_mem[arg_num] = compile->op[op_idx][inline_idx].buffer_in.val_cl;
                    arg_num++;
                }
            }
        }
    }
    kernel->arg_name = arg_name;
    kernel->arg_mem = arg_mem;
    kernel->arg_num = arg_num;
    kernel->arg_cap = arg_cap;
    free(arg_name_off);
    assert(kernel->arg_num > 0);
}
/* TODO: Maybe inline checks aren't the fixes and they are more complicated?? rework this completely probably */
static void compile_append_op_index(char **source, char **source_curr, uint64_t *source_cap, const op_t *op,
                                    const dim_info_t *dim_info, const inline_op_e *inline_type,
                                    const uint64_t inline_num, const uint64_t loop_num, const uint64_t compile_loop_idx,
                                    const uint64_t op_idx, const uint64_t loop_idx, const uint64_t splittable,
                                    const uint64_t global_size) {
    assert(source);
    assert(*source);
    assert(source_curr);
    assert(*source_curr);
    assert(source_cap);
    assert(*source_cap >= INITIAL_SOURCE_SIZE);
    if(splittable) {
        uint64_t a_max = op[0].buffer_out.sze_a;
        uint64_t z_max = op[0].buffer_out.sze_z;
        uint64_t y_max = op[0].buffer_out.sze_y;
        uint64_t x_max = op[0].buffer_out.sze_x;
        uint64_t op_size = a_max * z_max * y_max * x_max;
        uint64_t op_iters = op_size * loop_num;
        uint64_t op_unique_offsets = op_iters % global_size ? op_iters / global_size + 1 : op_iters / global_size;
        for(uint64_t op_unique_idx = 0; op_unique_idx < op_unique_offsets; op_unique_idx++) {
            /* My god this was horrible */
            /* I now know what Andrew Kelley was talking about with making systems accessible. TODO: This needs a major
             * rewrite to make it more see-through so to speak */
            if(op_unique_idx) {
                *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "id+=%lu;\n", global_size);
            } else {
                *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "id=gid;\n");
            }
            // out 0
            *source_curr += snprintf(
                *source_curr, MAX_OP_SIZE,
                "__const int "
                "%s_%lu_%lu_%d_%lu_%lu=%lu+((id/%lu)%%%lu)/%lu*%lu+((id/%lu)%%%lu)/%lu*%lu+((id/%lu)%%%lu)/"
                "%lu*%lu+((id/%lu)%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/"
                "%lu*%lu;/*1*/\n",
                op[0].buffer_out.name, compile_loop_idx, op_idx, 0, loop_idx, op_unique_idx, dim_info[0].off_out,
                op_size, dim_info[0].res_a_out, dim_info[0].wai_a_out, dim_info[0].str_a_out * op[0].buffer_out.str_a,
                op_size, dim_info[0].res_z_out, dim_info[0].wai_z_out, dim_info[0].str_z_out * op[0].buffer_out.str_z,
                op_size, dim_info[0].res_y_out, dim_info[0].wai_y_out, dim_info[0].str_y_out * op[0].buffer_out.str_y,
                op_size, dim_info[0].res_x_out, dim_info[0].wai_x_out, dim_info[0].str_x_out * op[0].buffer_out.str_x,
                op[0].buffer_out.sze_a * op[0].buffer_out.sze_z * op[0].buffer_out.sze_y * op[0].buffer_out.sze_x,
                op[0].buffer_out.sze_z * op[0].buffer_out.sze_y * op[0].buffer_out.sze_x,
                op[0].buffer_out.sze_a == 1 ? 0 : op[0].buffer_out.str_a,
                op[0].buffer_out.sze_z * op[0].buffer_out.sze_y * op[0].buffer_out.sze_x,
                op[0].buffer_out.sze_y * op[0].buffer_out.sze_x,
                op[0].buffer_out.sze_z == 1 ? 0 : op[0].buffer_out.str_z,
                op[0].buffer_out.sze_y * op[0].buffer_out.sze_x, op[0].buffer_out.sze_x,
                op[0].buffer_out.sze_y == 1 ? 0 : op[0].buffer_out.str_y, op[0].buffer_out.sze_x, 1LU,
                op[0].buffer_out.sze_x == 1 ? 0 : op[0].buffer_out.str_x);
            compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
            if(op[0].type_op != op_unary) {
                // in 0
                *source_curr += snprintf(
                    *source_curr, MAX_OP_SIZE,
                    "__const int "
                    "%s_%lu_%lu_%d_%lu_%lu=%lu+((id/%lu)%%%lu)/%lu*%lu+((id/%lu)%%%lu)/%lu*%lu+((id/%lu)%%%lu)/"
                    "%lu*%lu+((id/%lu)%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/"
                    "%lu*%lu;/*2*/\n",
                    op[0].buffer_in.name, compile_loop_idx, op_idx, 0, loop_idx, op_unique_idx, dim_info[0].off_in,
                    op_size, dim_info[0].res_a_in, dim_info[0].wai_a_in, dim_info[0].str_a_in * op[0].buffer_in.str_a,
                    op_size, dim_info[0].res_z_in, dim_info[0].wai_z_in, dim_info[0].str_z_in * op[0].buffer_in.str_z,
                    op_size, dim_info[0].res_y_in, dim_info[0].wai_y_in, dim_info[0].str_y_in * op[0].buffer_in.str_y,
                    op_size, dim_info[0].res_x_in, dim_info[0].wai_x_in, dim_info[0].str_x_in * op[0].buffer_in.str_x,
                    op[0].buffer_in.sze_a * op[0].buffer_in.sze_z * op[0].buffer_in.sze_y * op[0].buffer_in.sze_x,
                    op[0].buffer_in.sze_z * op[0].buffer_in.sze_y * op[0].buffer_in.sze_x,
                    op[0].buffer_in.sze_a == 1 ? 0 : op[0].buffer_in.str_a,
                    op[0].buffer_in.sze_z * op[0].buffer_in.sze_y * op[0].buffer_in.sze_x,
                    op[0].buffer_in.sze_y * op[0].buffer_in.sze_x,
                    op[0].buffer_in.sze_z == 1 ? 0 : op[0].buffer_in.str_z,
                    op[0].buffer_in.sze_y * op[0].buffer_in.sze_x, op[0].buffer_in.sze_x,
                    op[0].buffer_in.sze_y == 1 ? 0 : op[0].buffer_in.str_y, op[0].buffer_in.sze_x, 1LU,
                    op[0].buffer_in.sze_x == 1 ? 0 : op[0].buffer_in.str_x);
                compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
            }
            for(uint64_t inline_idx = 1; inline_idx < inline_num; inline_idx++) {
                if(op[inline_idx].type_op == op_unary) {
                    // out
                    *source_curr +=
                        snprintf(*source_curr, MAX_OP_SIZE,
                                 "__const int "
                                 "%s_%lu_%lu_%lu_%lu_%lu=%lu+((id/%lu)%%%lu)/%lu*%lu+((id/%lu)%%%lu)/%lu*%lu+((id/"
                                 "%lu)%%%lu)/%lu*%lu+((id/%lu)%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/"
                                 "%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu;/*3*/\n",
                                 op[inline_idx].buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx,
                                 op_unique_idx, dim_info[inline_idx].off_out, op_size, dim_info[inline_idx].res_a_out,
                                 dim_info[inline_idx].wai_a_out,
                                 dim_info[inline_idx].str_a_out * op[inline_idx].buffer_out.str_a, op_size,
                                 dim_info[inline_idx].res_z_out, dim_info[inline_idx].wai_z_out,
                                 dim_info[inline_idx].str_z_out * op[inline_idx].buffer_out.str_z, op_size,
                                 dim_info[inline_idx].res_y_out, dim_info[inline_idx].wai_y_out,
                                 dim_info[inline_idx].str_y_out * op[inline_idx].buffer_out.str_y, op_size,
                                 dim_info[inline_idx].res_x_out, dim_info[inline_idx].wai_x_out,
                                 dim_info[inline_idx].str_x_out * op[inline_idx].buffer_out.str_x,
                                 op[inline_idx].buffer_out.sze_a * op[inline_idx].buffer_out.sze_z *
                                     op[inline_idx].buffer_out.sze_y * op[inline_idx].buffer_out.sze_x,
                                 op[inline_idx].buffer_out.sze_z * op[inline_idx].buffer_out.sze_y *
                                     op[inline_idx].buffer_out.sze_x,
                                 op[inline_idx].buffer_out.sze_a == 1 ? inline_idx : op[inline_idx].buffer_out.str_a,
                                 op[inline_idx].buffer_out.sze_z * op[inline_idx].buffer_out.sze_y *
                                     op[inline_idx].buffer_out.sze_x,
                                 op[inline_idx].buffer_out.sze_y * op[inline_idx].buffer_out.sze_x,
                                 op[inline_idx].buffer_out.sze_z == 1 ? inline_idx : op[inline_idx].buffer_out.str_z,
                                 op[inline_idx].buffer_out.sze_y * op[inline_idx].buffer_out.sze_x,
                                 op[inline_idx].buffer_out.sze_x,
                                 op[inline_idx].buffer_out.sze_y == 1 ? inline_idx : op[inline_idx].buffer_out.str_y,
                                 op[inline_idx].buffer_out.sze_x, 1LU,
                                 op[inline_idx].buffer_out.sze_x == 1 ? inline_idx : op[inline_idx].buffer_out.str_x);
                    compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
                } else {
                    if(inline_type[inline_idx] == inline_op_out) {
                        // in
                        *source_curr +=
                            snprintf(*source_curr, MAX_OP_SIZE,
                                     "__const int "
                                     "%s_%lu_%lu_%lu_%lu_%lu=%lu+((id/%lu)%%%lu)/%lu*%lu+((id/%lu)%%%lu)/%lu*%lu+((id/"
                                     "%lu)%%%lu)/%lu*%lu+((id/%lu)%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/"
                                     "%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu;/*4.1*/\n",
                                     op[inline_idx].buffer_in.name, compile_loop_idx, op_idx, inline_idx, loop_idx,
                                     op_unique_idx, dim_info[inline_idx].off_in, op_size, dim_info[inline_idx].res_a_in,
                                     dim_info[inline_idx].wai_a_in,
                                     dim_info[inline_idx].str_a_in * op[inline_idx].buffer_in.str_a, op_size,
                                     dim_info[inline_idx].res_z_in, dim_info[inline_idx].wai_z_in,
                                     dim_info[inline_idx].str_z_in * op[inline_idx].buffer_in.str_z, op_size,
                                     dim_info[inline_idx].res_y_in, dim_info[inline_idx].wai_y_in,
                                     dim_info[inline_idx].str_y_in * op[inline_idx].buffer_in.str_y, op_size,
                                     dim_info[inline_idx].res_x_in, dim_info[inline_idx].wai_x_in,
                                     dim_info[inline_idx].str_x_in * op[inline_idx].buffer_in.str_x,
                                     op[inline_idx].buffer_in.sze_a * op[inline_idx].buffer_in.sze_z *
                                         op[inline_idx].buffer_in.sze_y * op[inline_idx].buffer_in.sze_x,
                                     op[inline_idx].buffer_in.sze_z * op[inline_idx].buffer_in.sze_y *
                                         op[inline_idx].buffer_in.sze_x,
                                     op[inline_idx].buffer_in.sze_a == 1 ? inline_idx : op[inline_idx].buffer_in.str_a,
                                     op[inline_idx].buffer_in.sze_z * op[inline_idx].buffer_in.sze_y *
                                         op[inline_idx].buffer_in.sze_x,
                                     op[inline_idx].buffer_in.sze_y * op[inline_idx].buffer_in.sze_x,
                                     op[inline_idx].buffer_in.sze_z == 1 ? inline_idx : op[inline_idx].buffer_in.str_z,
                                     op[inline_idx].buffer_in.sze_y * op[inline_idx].buffer_in.sze_x,
                                     op[inline_idx].buffer_in.sze_x,
                                     op[inline_idx].buffer_in.sze_y == 1 ? inline_idx : op[inline_idx].buffer_in.str_y,
                                     op[inline_idx].buffer_in.sze_x, 1LU,
                                     op[inline_idx].buffer_in.sze_x == 1 ? inline_idx : op[inline_idx].buffer_in.str_x);
                    } else {
                        /* Doing both indices here is a hack fix. TODO: Figure out when to use which one!!! */
                        // out & in
                        *source_curr += snprintf(
                            *source_curr, MAX_OP_SIZE,
                            "__const int "
                            "%s_%lu_%lu_%lu_%lu_%lu=%lu+((id/%lu)%%%lu)/%lu*%lu+((id/%lu)%%%lu)/%lu*%lu+((id/"
                            "%lu)%%%lu)/%lu*%lu+((id/%lu)%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/"
                            "%lu*%lu+(id%%%lu)/%lu*%lu;/*4.2.1*/\n",
                            op[inline_idx].buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx,
                            op_unique_idx, dim_info[inline_idx].off_out, op_size, dim_info[inline_idx].res_a_out,
                            dim_info[inline_idx].wai_a_out,
                            dim_info[inline_idx].str_a_out * op[inline_idx].buffer_out.str_a, op_size,
                            dim_info[inline_idx].res_z_out, dim_info[inline_idx].wai_z_out,
                            dim_info[inline_idx].str_z_out * op[inline_idx].buffer_out.str_z, op_size,
                            dim_info[inline_idx].res_y_out, dim_info[inline_idx].wai_y_out,
                            dim_info[inline_idx].str_y_out * op[inline_idx].buffer_out.str_y, op_size,
                            dim_info[inline_idx].res_x_out, dim_info[inline_idx].wai_x_out,
                            dim_info[inline_idx].str_x_out * op[inline_idx].buffer_out.str_x,
                            op[inline_idx].buffer_out.sze_a * op[inline_idx].buffer_out.sze_z *
                                op[inline_idx].buffer_out.sze_y * op[inline_idx].buffer_out.sze_x,
                            op[inline_idx].buffer_out.sze_z * op[inline_idx].buffer_out.sze_y *
                                op[inline_idx].buffer_out.sze_x,
                            op[inline_idx].buffer_out.sze_a == 1 ? inline_idx : op[inline_idx].buffer_out.str_a,
                            op[inline_idx].buffer_out.sze_z * op[inline_idx].buffer_out.sze_y *
                                op[inline_idx].buffer_out.sze_x,
                            op[inline_idx].buffer_out.sze_y * op[inline_idx].buffer_out.sze_x,
                            op[inline_idx].buffer_out.sze_z == 1 ? inline_idx : op[inline_idx].buffer_out.str_z,
                            op[inline_idx].buffer_out.sze_y * op[inline_idx].buffer_out.sze_x,
                            op[inline_idx].buffer_out.sze_x,
                            op[inline_idx].buffer_out.sze_y == 1 ? inline_idx : op[inline_idx].buffer_out.str_y,
                            op[inline_idx].buffer_out.sze_x, 1LU,
                            op[inline_idx].buffer_out.sze_x == 1 ? inline_idx : op[inline_idx].buffer_out.str_x);
                        *source_curr +=
                            snprintf(*source_curr, MAX_OP_SIZE,
                                     "__const int "
                                     "%s_%lu_%lu_%lu_%lu_%lu=%lu+((id/%lu)%%%lu)/%lu*%lu+((id/%lu)%%%lu)/%lu*%lu+((id/"
                                     "%lu)%%%lu)/%lu*%lu+((id/%lu)%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/"
                                     "%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu;/*4.2.2*/\n",
                                     op[inline_idx].buffer_in.name, compile_loop_idx, op_idx, inline_idx, loop_idx,
                                     op_unique_idx, dim_info[inline_idx].off_in, op_size, dim_info[inline_idx].res_a_in,
                                     dim_info[inline_idx].wai_a_in,
                                     dim_info[inline_idx].str_a_in * op[inline_idx].buffer_in.str_a, op_size,
                                     dim_info[inline_idx].res_z_in, dim_info[inline_idx].wai_z_in,
                                     dim_info[inline_idx].str_z_in * op[inline_idx].buffer_in.str_z, op_size,
                                     dim_info[inline_idx].res_y_in, dim_info[inline_idx].wai_y_in,
                                     dim_info[inline_idx].str_y_in * op[inline_idx].buffer_in.str_y, op_size,
                                     dim_info[inline_idx].res_x_in, dim_info[inline_idx].wai_x_in,
                                     dim_info[inline_idx].str_x_in * op[inline_idx].buffer_in.str_x,
                                     op[inline_idx].buffer_in.sze_a * op[inline_idx].buffer_in.sze_z *
                                         op[inline_idx].buffer_in.sze_y * op[inline_idx].buffer_in.sze_x,
                                     op[inline_idx].buffer_in.sze_z * op[inline_idx].buffer_in.sze_y *
                                         op[inline_idx].buffer_in.sze_x,
                                     op[inline_idx].buffer_in.sze_a == 1 ? inline_idx : op[inline_idx].buffer_in.str_a,
                                     op[inline_idx].buffer_in.sze_z * op[inline_idx].buffer_in.sze_y *
                                         op[inline_idx].buffer_in.sze_x,
                                     op[inline_idx].buffer_in.sze_y * op[inline_idx].buffer_in.sze_x,
                                     op[inline_idx].buffer_in.sze_z == 1 ? inline_idx : op[inline_idx].buffer_in.str_z,
                                     op[inline_idx].buffer_in.sze_y * op[inline_idx].buffer_in.sze_x,
                                     op[inline_idx].buffer_in.sze_x,
                                     op[inline_idx].buffer_in.sze_y == 1 ? inline_idx : op[inline_idx].buffer_in.str_y,
                                     op[inline_idx].buffer_in.sze_x, 1LU,
                                     op[inline_idx].buffer_in.sze_x == 1 ? inline_idx : op[inline_idx].buffer_in.str_x);
                    }
                    compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
                }
            }
        }
    } else {
        *source_curr += snprintf(
            *source_curr, MAX_OP_SIZE,
            "__const int "
            "%s_%lu_%lu_%d_%lu_%d=%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu;/*5*/\n",
            op[0].buffer_out.name, compile_loop_idx, op_idx, 0, loop_idx, 0, dim_info[0].off_out, dim_info[0].res_a_out,
            dim_info[0].wai_a_out, dim_info[0].str_a_out * op[0].buffer_out.str_a, dim_info[0].res_z_out,
            dim_info[0].wai_z_out, dim_info[0].str_z_out * op[0].buffer_out.str_z, dim_info[0].res_y_out,
            dim_info[0].wai_y_out, dim_info[0].str_y_out * op[0].buffer_out.str_y, dim_info[0].res_x_out,
            dim_info[0].wai_x_out, dim_info[0].str_x_out * op[0].buffer_out.str_x);
        compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
        if(op[0].type_op != op_unary) {
            *source_curr +=
                snprintf(*source_curr, MAX_OP_SIZE,
                         "__const int "
                         "%s_%lu_%lu_%d_%lu_%d=%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/"
                         "%lu*%lu;/*6*/\n",
                         op[0].buffer_in.name, compile_loop_idx, op_idx, 0, loop_idx, 0, dim_info[0].off_in,
                         dim_info[0].res_a_in, dim_info[0].wai_a_in, dim_info[0].str_a_in * op[0].buffer_in.str_a,
                         dim_info[0].res_z_in, dim_info[0].wai_z_in, dim_info[0].str_z_in * op[0].buffer_in.str_z,
                         dim_info[0].res_y_in, dim_info[0].wai_y_in, dim_info[0].str_y_in * op[0].buffer_in.str_y,
                         dim_info[0].res_x_in, dim_info[0].wai_x_in, dim_info[0].str_x_in * op[0].buffer_in.str_x);
            compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
        }
        for(uint64_t inline_idx = 1; inline_idx < inline_num; inline_idx++) {
            if(op[inline_idx].type_op == op_unary) {
                *source_curr += snprintf(
                    *source_curr, MAX_OP_SIZE,
                    "__const int "
                    "%s_%lu_%lu_%lu_%lu_%d=%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/"
                    "%lu*%lu;/*7*/\n",
                    op[inline_idx].buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0,
                    dim_info[inline_idx].off_out, dim_info[inline_idx].res_a_out, dim_info[inline_idx].wai_a_out,
                    dim_info[inline_idx].str_a_out * op[inline_idx].buffer_out.str_a, dim_info[inline_idx].res_z_out,
                    dim_info[inline_idx].wai_z_out, dim_info[inline_idx].str_z_out * op[inline_idx].buffer_out.str_z,
                    dim_info[inline_idx].res_y_out, dim_info[inline_idx].wai_y_out,
                    dim_info[inline_idx].str_y_out * op[inline_idx].buffer_out.str_y, dim_info[inline_idx].res_x_out,
                    dim_info[inline_idx].wai_x_out, dim_info[inline_idx].str_x_out * op[inline_idx].buffer_out.str_x);
            } else {
                /* Doing both indices here is a hack fix. TODO: Figure out when to use which one!!! */
                *source_curr += snprintf(
                    *source_curr, MAX_OP_SIZE,
                    "__const int "
                    "%s_%lu_%lu_%lu_%lu_%d=%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/"
                    "%lu*%lu;/*8.1*/\n",
                    op[inline_idx].buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0,
                    dim_info[inline_idx].off_out, dim_info[inline_idx].res_a_out, dim_info[inline_idx].wai_a_out,
                    dim_info[inline_idx].str_a_out * op[inline_idx].buffer_out.str_a, dim_info[inline_idx].res_z_out,
                    dim_info[inline_idx].wai_z_out, dim_info[inline_idx].str_z_out * op[inline_idx].buffer_out.str_z,
                    dim_info[inline_idx].res_y_out, dim_info[inline_idx].wai_y_out,
                    dim_info[inline_idx].str_y_out * op[inline_idx].buffer_out.str_y, dim_info[inline_idx].res_x_out,
                    dim_info[inline_idx].wai_x_out, dim_info[inline_idx].str_x_out * op[inline_idx].buffer_out.str_x);
                *source_curr += snprintf(
                    *source_curr, MAX_OP_SIZE,
                    "__const int "
                    "%s_%lu_%lu_%lu_%lu_%d=%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/"
                    "%lu*%lu;/*8.2*/\n",
                    op[inline_idx].buffer_in.name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0,
                    dim_info[inline_idx].off_in, dim_info[inline_idx].res_a_in, dim_info[inline_idx].wai_a_in,
                    dim_info[inline_idx].str_a_in * op[inline_idx].buffer_in.str_a, dim_info[inline_idx].res_z_in,
                    dim_info[inline_idx].wai_z_in, dim_info[inline_idx].str_z_in * op[inline_idx].buffer_in.str_z,
                    dim_info[inline_idx].res_y_in, dim_info[inline_idx].wai_y_in,
                    dim_info[inline_idx].str_y_in * op[inline_idx].buffer_in.str_y, dim_info[inline_idx].res_x_in,
                    dim_info[inline_idx].wai_x_in, dim_info[inline_idx].str_x_in * op[inline_idx].buffer_in.str_x);
            }
            compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
        }
    }
}
static void compile_append_header(char **source, char **source_curr, uint64_t *source_cap, const op_t *op,
                                  const uint64_t compile_loop_idx, const uint64_t op_idx, const uint64_t loop_idx) {
    assert(source);
    assert(*source);
    assert(source_curr);
    assert(*source_curr);
    assert(source_cap);
    assert(*source_cap >= INITIAL_SOURCE_SIZE);
    if(op->type_op == op_reduce) {
        char *name = (char *) op->buffer_out.name;
        switch(op->type_reduce) {
            case reduce_sum: {
                *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%d_%lu_%d]=0;\n", name, name,
                                         compile_loop_idx, op_idx, 0, loop_idx, 0);
                break;
            }
            case reduce_avg: {
                *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%d_%lu_%d]=0;\n", name, name,
                                         compile_loop_idx, op_idx, 0, loop_idx, 0);
                break;
            }
            case reduce_max: {
                *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%d_%lu_%d]=-INFINITY;\n", name, name,
                                         compile_loop_idx, op_idx, 0, loop_idx, 0);
                break;
            }
            case reduce_min: {
                *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%d_%lu_%d]=INFINITY;\n", name, name,
                                         compile_loop_idx, op_idx, 0, loop_idx, 0);
                break;
            }
        }
        compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
    }
}
static void compile_append_footer(char **source, char **source_curr, uint64_t *source_cap, const op_t *op,
                                  const uint64_t compile_loop_idx, const uint64_t op_idx, const uint64_t loop_idx,
                                  double avg_divisor) {
    assert(source);
    assert(*source);
    assert(source_curr);
    assert(*source_curr);
    assert(source_cap);
    assert(*source_cap >= INITIAL_SOURCE_SIZE);
    if(op->type_op == op_reduce) {
        switch(op->type_reduce) {
            case reduce_sum: {
                break;
            }
            case reduce_avg: {
                char *name = (char *) op->buffer_out.name;
                *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%d_%lu_%d]/=%lf;\n", name, name,
                                         compile_loop_idx, op_idx, 0, loop_idx, 0, avg_divisor);
                break;
            }
            case reduce_max: {
                break;
            }
            case reduce_min: {
                break;
            }
        }
        compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
    }
}
static void compile_append_assign(char **temp, char **temp_curr, uint64_t *temp_cap, const op_t *op,
                                  const uint64_t op_num, const uint64_t compile_loop_idx, const uint64_t op_idx,
                                  const uint64_t inline_idx, const uint64_t loop_idx, const uint64_t op_unique_idx,
                                  const uint64_t offset, const uint64_t splittable) {
    assert(temp);
    assert(*temp);
    assert(temp_curr);
    assert(*temp_curr);
    assert(temp_cap);
    assert(*temp_cap);
    assert(op);
    assert(compile_loop_idx >= 0);
    assert(op_idx >= 0);
    assert(loop_idx >= 0);
    assert(!inline_idx);
    char *name = (char *) op->buffer_out.name;
    if(splittable) {
        switch(op->type_op) {
            case op_unary: {
                switch(op->type_unary) {
                    case unary_add: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]+=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case unary_subtract: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]-=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case unary_multiply: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]*=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case unary_divide: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]/=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case unary_exp: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case unary_log: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case unary_square: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case unary_sqrt: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case unary_reciprocal: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case unary_max: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case unary_min: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case unary_set: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case unary_random: {
                        ERROR("Tried to compile unary_random");
                        break;
                    }
                    case unary_tanh: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case unary_sign: {
                        TODO();
                        break;
                    }
                    case unary_absolute: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                }
                break;
            }
            case op_binary: {
                switch(op->type_binary) {
                    case binary_add: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]+=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case binary_subtract: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]-=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case binary_multiply: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]*=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case binary_divide: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]/=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case binary_max: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_min: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_copy: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_add_like: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]+=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case binary_subtract_like: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]-=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case binary_multiply_like: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]*=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case binary_divide_like: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]/=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case binary_max_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_min_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_copy_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                }
                break;
            }
            case op_reduce: {
                switch(op->type_reduce) {
                    case reduce_sum: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]+=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case reduce_avg: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]+=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case reduce_max: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case reduce_min: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                }
                break;
            }
            case op_move: {
                ERROR("Tried to append an assign for a move op");
            }
        }
    } else {
        switch(op->type_op) {
            case op_unary: {
                switch(op->type_unary) {
                    case unary_add: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]+=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case unary_subtract: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]-=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case unary_multiply: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]*=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case unary_divide: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]/=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case unary_exp: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case unary_log: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case unary_square: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case unary_sqrt: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case unary_reciprocal: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case unary_max: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case unary_min: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case unary_set: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case unary_random: {
                        ERROR("Tried to compile unary_random");
                        break;
                    }
                    case unary_tanh: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case unary_sign: {
                        TODO();
                        break;
                    }
                    case unary_absolute: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                }
                break;
            }
            case op_binary: {
                switch(op->type_binary) {
                    case binary_add: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]+=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case binary_subtract: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]-=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case binary_multiply: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]*=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case binary_divide: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]/=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case binary_max: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case binary_min: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case binary_copy: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case binary_add_like: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]+=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case binary_subtract_like: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]-=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case binary_multiply_like: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]*=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case binary_divide_like: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]/=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case binary_max_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case binary_min_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case binary_copy_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                }
                break;
            }
            case op_reduce: {
                switch(op->type_reduce) {
                    case reduce_sum: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d]+=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0);
                        break;
                    }
                    case reduce_avg: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d]+=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0);
                        break;
                    }
                    case reduce_max: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0);
                        break;
                    }
                    case reduce_min: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d]=", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0);
                        break;
                    }
                }
                break;
            }
            case op_move: {
                ERROR("Tried to append an assign for a move op");
            }
        }
    }
    compile_expand_source(temp, temp_curr, temp_cap, MAX_OP_SIZE);
}
static void compile_append_prefix(char **temp, char **temp_curr, uint64_t *temp_cap, const op_t *op,
                                  const uint64_t op_num, const uint64_t compile_loop_idx, const uint64_t op_idx,
                                  const uint64_t inline_idx, const uint64_t loop_idx, const uint64_t op_unique_idx,
                                  const inline_op_e inline_type, const uint64_t offset, const uint64_t splittable) {
    assert(temp);
    assert(*temp);
    assert(temp_curr);
    assert(*temp_curr);
    assert(temp_cap);
    assert(*temp_cap);
    assert(op);
    assert(compile_loop_idx >= 0);
    assert(op_idx >= 0);
    assert(inline_idx >= 0);
    assert(loop_idx >= 0);
    if(splittable) {
        switch(op->type_op) {
            case op_unary: {
                switch(op->type_unary) {
                    case unary_add: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                    case unary_subtract: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                    case unary_multiply: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                    case unary_divide: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                    case unary_exp: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "exp(");
                        break;
                    }
                    case unary_log: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "log(");
                        break;
                    }
                    case unary_square: {
                        /* TODO: Do this with `x*x` for less compilated x instead of `pow(x,2)` cuz that's prolly
                         * faster than a universal `pow` algorithm */
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "pow(");
                        break;
                    }
                    case unary_sqrt: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "sqrt(");
                        break;
                    }
                    case unary_reciprocal: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "1/(");
                        break;
                    }
                    case unary_max: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmax(%lf,", op->var_unary);
                        break;
                    }
                    case unary_min: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmin(%lf,", op->var_unary);
                        break;
                    }
                    case unary_set: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                    case unary_random: {
                        ERROR("Tried to compile unary_random");
                        break;
                    }
                    case unary_tanh: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "tanh(");
                        break;
                    }
                    case unary_sign: {
                        TODO();
                        break;
                    }
                    case unary_absolute: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fabs(");
                        break;
                    }
                }
                break;
            }
            case op_binary: {
                char *name;
                if(inline_type == inline_op_none) {
                    name = (char *) op->buffer_out.name;
                } else if(inline_type == inline_op_out) {
                    name = (char *) op->buffer_in.name;
                } else {
                    name = (char *) op->buffer_out.name;
                }
                switch(op->type_binary) {
                    case binary_add: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu_%lu]+", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case binary_subtract: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu_%lu]-", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case binary_multiply: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu_%lu]*", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case binary_divide: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu_%lu]/", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case binary_max: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmax(%s[%s_%lu_%lu_%lu_%lu_%lu],", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_min: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmin(%s[%s_%lu_%lu_%lu_%lu_%lu],", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_copy: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                    case binary_add_like: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu_%lu]+", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case binary_subtract_like: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu_%lu]-", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case binary_multiply_like: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu_%lu]*", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case binary_divide_like: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu_%lu]/", name, name,
                                                   compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case binary_max_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmax(%s[%s_%lu_%lu_%lu_%lu_%lu],", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_min_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmin(%s[%s_%lu_%lu_%lu_%lu_%lu],", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_copy_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                }
                break;
            }
            case op_reduce: {
                assert(!inline_idx);
                char *name = (char *) op->buffer_out.name;
                switch(op->type_reduce) {
                    case reduce_sum: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                    case reduce_avg: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                    case reduce_max: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmax(%s[%s_%lu_%lu_%lu_%lu_%lu],", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case reduce_min: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmin(%s[%s_%lu_%lu_%lu_%lu_%lu],", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                }
                break;
            }
            case op_move: {
                ERROR("Tried to append prefix for a move op");
            }
        }
    } else {
        switch(op->type_op) {
            case op_unary: {
                switch(op->type_unary) {
                    case unary_add: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                    case unary_subtract: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                    case unary_multiply: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                    case unary_divide: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                    case unary_exp: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "exp(");
                        break;
                    }
                    case unary_log: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "log(");
                        break;
                    }
                    case unary_square: {
                        /* TODO: Do this with `x*x` for less compilated x instead of `pow(x,2)` cuz that's prolly
                         * faster than a universal `pow` algorithm */
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "pow(");
                        break;
                    }
                    case unary_sqrt: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "sqrt(");
                        break;
                    }
                    case unary_reciprocal: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "1/(");
                        break;
                    }
                    case unary_max: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmax(%lf,", op->var_unary);
                        break;
                    }
                    case unary_min: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmin(%lf,", op->var_unary);
                        break;
                    }
                    case unary_set: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                    case unary_random: {
                        ERROR("Tried to compile unary_random");
                        break;
                    }
                    case unary_tanh: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "tanh(");
                        break;
                    }
                    case unary_sign: {
                        TODO();
                        break;
                    }
                    case unary_absolute: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fabs(");
                        break;
                    }
                }
                break;
            }
            case op_binary: {
                char *name;
                if(inline_type == inline_op_none) {
                    name = (char *) op->buffer_out.name;
                } else if(inline_type == inline_op_out) {
                    name = (char *) op->buffer_in.name;
                } else {
                    name = (char *) op->buffer_out.name;
                }
                switch(op->type_binary) {
                    case binary_add: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu_%d+%lu]+", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case binary_subtract: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu_%d+%lu]-", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case binary_multiply: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu_%d+%lu]*", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case binary_divide: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu_%d+%lu]/", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case binary_max: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmax(%s[%s_%lu_%lu_%lu_%lu_%d+%lu],", name,
                                               name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case binary_min: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmin(%s[%s_%lu_%lu_%lu_%lu_%d+%lu],", name,
                                               name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case binary_copy: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                    case binary_add_like: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu_%d+%lu]+", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case binary_subtract_like: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu_%d+%lu]-", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case binary_multiply_like: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu_%d+%lu]*", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case binary_divide_like: {
                        if(op_num == 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        } else {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(%s[%s_%lu_%lu_%lu_%lu_%d+%lu]/", name,
                                                   name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case binary_max_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmax(%s[%s_%lu_%lu_%lu_%lu_%d+%lu],", name,
                                               name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case binary_min_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmin(%s[%s_%lu_%lu_%lu_%lu_%d+%lu],", name,
                                               name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case binary_copy_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                }
                break;
            }
            case op_reduce: {
                assert(!inline_idx);
                char *name = (char *) op->buffer_out.name;
                switch(op->type_reduce) {
                    case reduce_sum: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                    case reduce_avg: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "(");
                        break;
                    }
                    case reduce_max: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmax(%s[%s_%lu_%lu_%lu_%lu_%d],", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0);
                        break;
                    }
                    case reduce_min: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "fmin(%s[%s_%lu_%lu_%lu_%lu_%d],", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0);
                        break;
                    }
                }
                break;
            }
            case op_move: {
                ERROR("Tried to append prefix for a move op");
            }
        }
    }
    compile_expand_source(temp, temp_curr, temp_cap, MAX_OP_SIZE);
}
static void compile_append_inner(char **temp, char **temp_curr, uint64_t *temp_cap, const op_t *op,
                                 const uint64_t op_num, const uint64_t compile_loop_idx, const uint64_t op_idx,
                                 const uint64_t inline_idx, const uint64_t loop_idx, const uint64_t op_unique_idx,
                                 const uint64_t offset, const uint64_t splittable) {
    assert(temp);
    assert(*temp);
    assert(temp_curr);
    assert(*temp_curr);
    assert(temp_cap);
    assert(*temp_cap);
    assert(op);
    assert(compile_loop_idx >= 0);
    assert(op_idx >= 0);
    assert(loop_idx >= 0);
    assert(inline_idx >= 0);
    if(splittable) {
        switch(op->type_op) {
            case op_unary: {
                switch(op->type_unary) {
                    case unary_add: {
                        if(op_num != 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]",
                                                   op->buffer_out.name, op->buffer_out.name, compile_loop_idx, op_idx,
                                                   inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case unary_subtract: {
                        if(op_num != 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]",
                                                   op->buffer_out.name, op->buffer_out.name, compile_loop_idx, op_idx,
                                                   inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case unary_multiply: {
                        if(op_num != 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]",
                                                   op->buffer_out.name, op->buffer_out.name, compile_loop_idx, op_idx,
                                                   inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case unary_divide: {
                        if(op_num != 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]",
                                                   op->buffer_out.name, op->buffer_out.name, compile_loop_idx, op_idx,
                                                   inline_idx, loop_idx, op_unique_idx);
                        }
                        break;
                    }
                    case unary_exp: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]",
                                               op->buffer_out.name, op->buffer_out.name, compile_loop_idx, op_idx,
                                               inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case unary_log: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]",
                                               op->buffer_out.name, op->buffer_out.name, compile_loop_idx, op_idx,
                                               inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case unary_square: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]",
                                               op->buffer_out.name, op->buffer_out.name, compile_loop_idx, op_idx,
                                               inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case unary_sqrt: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]",
                                               op->buffer_out.name, op->buffer_out.name, compile_loop_idx, op_idx,
                                               inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case unary_reciprocal: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]",
                                               op->buffer_out.name, op->buffer_out.name, compile_loop_idx, op_idx,
                                               inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case unary_max: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]",
                                               op->buffer_out.name, op->buffer_out.name, compile_loop_idx, op_idx,
                                               inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case unary_min: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]",
                                               op->buffer_out.name, op->buffer_out.name, compile_loop_idx, op_idx,
                                               inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case unary_set: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%lf", op->var_unary);
                        break;
                    }
                    case unary_random: {
                        ERROR("Tried to compile unary_random");
                        break;
                    }
                    case unary_tanh: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]",
                                               op->buffer_out.name, op->buffer_out.name, compile_loop_idx, op_idx,
                                               inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case unary_sign: {
                        TODO();
                        break;
                    }
                    case unary_absolute: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]",
                                               op->buffer_out.name, op->buffer_out.name, compile_loop_idx, op_idx,
                                               inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                }
                break;
            }
            case op_binary: {
                char *name = (char *) op->buffer_in.name;
                switch(op->type_binary) {
                    case binary_add: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_subtract: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_multiply: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_divide: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_max: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_min: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_copy: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_add_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_subtract_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_multiply_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_divide_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_max_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_min_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                    case binary_copy_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                        break;
                    }
                }
                break;
            }
            case op_reduce: {
                assert(!inline_idx);
                *temp_curr +=
                    snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%lu]", op->buffer_in.name,
                             op->buffer_in.name, compile_loop_idx, op_idx, inline_idx, loop_idx, op_unique_idx);
                break;
            }
            case op_move: {
                ERROR("Tried to append innermost for a move op");
            }
        }
    } else {
        switch(op->type_op) {
            case op_unary: {
                switch(op->type_unary) {
                    case unary_add: {
                        if(op_num != 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]",
                                                   op->buffer_out.name, op->buffer_out.name, compile_loop_idx, op_idx,
                                                   inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case unary_subtract: {
                        if(op_num != 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]",
                                                   op->buffer_out.name, op->buffer_out.name, compile_loop_idx, op_idx,
                                                   inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case unary_multiply: {
                        if(op_num != 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]",
                                                   op->buffer_out.name, op->buffer_out.name, compile_loop_idx, op_idx,
                                                   inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case unary_divide: {
                        if(op_num != 1) {
                            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]",
                                                   op->buffer_out.name, op->buffer_out.name, compile_loop_idx, op_idx,
                                                   inline_idx, loop_idx, 0, offset);
                        }
                        break;
                    }
                    case unary_exp: {
                        *temp_curr +=
                            snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]", op->buffer_out.name,
                                     op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case unary_log: {
                        *temp_curr +=
                            snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]", op->buffer_out.name,
                                     op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case unary_square: {
                        *temp_curr +=
                            snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]", op->buffer_out.name,
                                     op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case unary_sqrt: {
                        *temp_curr +=
                            snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]", op->buffer_out.name,
                                     op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case unary_reciprocal: {
                        *temp_curr +=
                            snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]", op->buffer_out.name,
                                     op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case unary_max: {
                        *temp_curr +=
                            snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]", op->buffer_out.name,
                                     op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case unary_min: {
                        *temp_curr +=
                            snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]", op->buffer_out.name,
                                     op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case unary_set: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%lf", op->var_unary);
                        break;
                    }
                    case unary_random: {
                        ERROR("Tried to compile unary_random");
                        break;
                    }
                    case unary_tanh: {
                        *temp_curr +=
                            snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]", op->buffer_out.name,
                                     op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case unary_sign: {
                        TODO();
                        break;
                    }
                    case unary_absolute: {
                        *temp_curr +=
                            snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]", op->buffer_out.name,
                                     op->buffer_out.name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                }
                break;
            }
            case op_binary: {
                char *name = (char *) op->buffer_in.name;
                switch(op->type_binary) {
                    case binary_add: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case binary_subtract: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case binary_multiply: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case binary_divide: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case binary_max: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case binary_min: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case binary_copy: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                        break;
                    }
                    case binary_add_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0);
                        break;
                    }
                    case binary_subtract_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0);
                        break;
                    }
                    case binary_multiply_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0);
                        break;
                    }
                    case binary_divide_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0);
                        break;
                    }
                    case binary_max_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0);
                        break;
                    }
                    case binary_min_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0);
                        break;
                    }
                    case binary_copy_like: {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d]", name, name,
                                               compile_loop_idx, op_idx, inline_idx, loop_idx, 0);
                        break;
                    }
                }
                break;
            }
            case op_reduce: {
                assert(!inline_idx);
                *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%s[%s_%lu_%lu_%lu_%lu_%d+%lu]", op->buffer_in.name,
                                       op->buffer_in.name, compile_loop_idx, op_idx, inline_idx, loop_idx, 0, offset);
                break;
            }
            case op_move: {
                ERROR("Tried to append innermost for a move op");
            }
        }
    }
    compile_expand_source(temp, temp_curr, temp_cap, MAX_OP_SIZE);
}
static void compile_append_postfix(char **temp, char **temp_curr, uint64_t *temp_cap, const op_t *op,
                                   const uint64_t op_num) {
    assert(temp);
    assert(*temp);
    assert(temp_curr);
    assert(*temp_curr);
    assert(temp_cap);
    assert(*temp_cap);
    assert(op);
    switch(op->type_op) {
        case op_unary: {
            switch(op->type_unary) {
                case unary_add: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%lf)", op->var_unary);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "+(%lf))", op->var_unary);
                    }
                    break;
                }
                case unary_subtract: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%lf)", op->var_unary);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "-(%lf))", op->var_unary);
                    }
                    break;
                }
                case unary_multiply: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%lf)", op->var_unary);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "*(%lf))", op->var_unary);
                    }
                    break;
                }
                case unary_divide: {
                    if(op_num == 1) {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "%lf)", op->var_unary);
                    } else {
                        *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, "/(%lf))", op->var_unary);
                    }
                    break;
                }
                case unary_exp: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
                case unary_log: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
                case unary_square: {
                    /* TODO: Do this with `x*x` for less compilated x instead of `pow(x,2)` cuz that's prolly
                     * faster than a universal `pow` algorithm */
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ",2)");
                    break;
                }
                case unary_sqrt: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
                case unary_reciprocal: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
                case unary_max: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
                case unary_min: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
                case unary_set: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
                case unary_random: {
                    ERROR("Tried to compile unary_random");
                    break;
                }
                case unary_tanh: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
                case unary_sign: {
                    TODO();
                    break;
                }
                case unary_absolute: {
                    *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
                    break;
                }
            }
            break;
        }
        case op_binary: {
            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
            break;
        }
        case op_reduce: {
            *temp_curr += snprintf(*temp_curr, MAX_OP_SIZE, ")");
            break;
        }
        case op_move: {
            ERROR("Tried to append prefix for a move op");
        }
    }
    compile_expand_source(temp, temp_curr, temp_cap, MAX_OP_SIZE);
}
static void compile_append_op(char **source, char **source_curr, uint64_t *source_cap, const op_t *op,
                              const dim_info_t *dim_info, const inline_op_e *inline_info, const uint64_t op_num,
                              const uint64_t compile_loop_num, const uint64_t compile_loop_idx, const uint64_t op_idx,
                              const uint64_t loop_idx, const uint64_t global_size, const uint64_t splittable) {
    assert(source);
    assert(*source);
    assert(source_curr);
    assert(*source_curr);
    assert(source_cap);
    assert(*source_cap >= INITIAL_SOURCE_SIZE);
    assert(op);
    assert(dim_info);
    assert(op_num > 0);
    assert(compile_loop_idx >= 0);
    assert(op_idx >= 0);
    assert(loop_idx >= 0);
    uint64_t temp_cap = INITIAL_SOURCE_SIZE;
    char *temp = calloc(INITIAL_SOURCE_SIZE, sizeof(char));
    assert(temp);
    char *temp_curr = temp;

    uint64_t a_max = op->type_op == op_reduce ? op->buffer_in.sze_a : op->buffer_out.sze_a;
    uint64_t z_max = op->type_op == op_reduce ? op->buffer_in.sze_z : op->buffer_out.sze_z;
    uint64_t y_max = op->type_op == op_reduce ? op->buffer_in.sze_y : op->buffer_out.sze_y;
    uint64_t x_max = op->type_op == op_reduce ? op->buffer_in.sze_x : op->buffer_out.sze_x;
    uint64_t total_op_num = a_max * z_max * y_max * x_max * compile_loop_num;
    uint64_t leftover_op = total_op_num % global_size;
    uint64_t kernel_op_num = leftover_op ? total_op_num / global_size + 1 : total_op_num / global_size;
    if(splittable) {
        const uint64_t no_offset = 0;
        for(uint64_t kernel_op_idx = 0; kernel_op_idx < kernel_op_num; kernel_op_idx++) {
            if(leftover_op && kernel_op_idx == kernel_op_num - 1) {
                *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "if(gid < %lu) {\n", leftover_op);
                compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
            }

            /* TODO: Make better idx names than op_unique_idx */

            compile_append_assign(&temp, &temp_curr, &temp_cap, op, op_num, compile_loop_idx, op_idx, 0, loop_idx,
                                  kernel_op_idx, no_offset, splittable);
            for(uint64_t inline_idx = 0; inline_idx < op_num; inline_idx++) {
                compile_append_prefix(&temp, &temp_curr, &temp_cap, &op[inline_idx], op_num, compile_loop_idx, op_idx,
                                      inline_idx, loop_idx, kernel_op_idx, inline_info[inline_idx], no_offset,
                                      splittable);
            }

            uint64_t inner_idx = op_num - 1;
            compile_append_inner(&temp, &temp_curr, &temp_cap, &op[inner_idx], op_num, compile_loop_idx, op_idx,
                                 inner_idx, loop_idx, kernel_op_idx, no_offset, splittable);

            for(int64_t inline_idx = op_num - 1; inline_idx >= 0; inline_idx--) {
                compile_append_postfix(&temp, &temp_curr, &temp_cap, &op[inline_idx], op_num);
            }

            *source_curr += snprintf(*source_curr, temp_cap, "%s;\n", temp);
            compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
            if(leftover_op && kernel_op_idx == kernel_op_num - 1) {
                *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "}\n");
                compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);
            }

            temp_curr = temp;
            memset(temp, '\0', temp_cap);
        }
    } else {
        compile_append_header(source, source_curr, source_cap, op, compile_loop_idx, op_idx, loop_idx);
        for(uint64_t a_idx = 0; a_idx < a_max; a_idx++) {
            for(uint64_t z_idx = 0; z_idx < z_max; z_idx++) {
                for(uint64_t y_idx = 0; y_idx < y_max; y_idx++) {
                    for(uint64_t x_idx = 0; x_idx < x_max; x_idx++) {
                        int64_t offset;
                        offset = INDEX(op[0].buffer_out, a_idx, z_idx, y_idx, x_idx);
                        compile_append_assign(&temp, &temp_curr, &temp_cap, op, op_num, compile_loop_idx, op_idx, 0,
                                              loop_idx, 0, offset, splittable);
                        for(uint64_t inline_idx = 0; inline_idx < op_num; inline_idx++) {
                            offset = op[inline_idx].type_op == op_unary
                                         ? INDEX(op[inline_idx].buffer_out, a_idx, z_idx, y_idx, x_idx)
                                         : INDEX(op[inline_idx].buffer_in, a_idx, z_idx, y_idx, x_idx);
                            compile_append_prefix(&temp, &temp_curr, &temp_cap, &op[inline_idx], op_num,
                                                  compile_loop_idx, op_idx, inline_idx, loop_idx, 0,
                                                  inline_info[inline_idx], offset, splittable);
                        }

                        uint64_t inner_idx = op_num - 1;
                        offset = op[inner_idx].type_op == op_unary
                                     ? INDEX(op[inner_idx].buffer_out, a_idx, z_idx, y_idx, x_idx)
                                     : INDEX(op[inner_idx].buffer_in, a_idx, z_idx, y_idx, x_idx);
                        compile_append_inner(&temp, &temp_curr, &temp_cap, &op[inner_idx], op_num, compile_loop_idx,
                                             op_idx, inner_idx, loop_idx, 0, offset, splittable);

                        for(int64_t inline_idx = op_num - 1; inline_idx >= 0; inline_idx--) {
                            compile_append_postfix(&temp, &temp_curr, &temp_cap, &op[inline_idx], op_num);
                        }

                        *source_curr += snprintf(*source_curr, temp_cap, "%s;\n", temp);
                        compile_expand_source(source, source_curr, source_cap, MAX_OP_SIZE);

                        temp_curr = temp;
                        memset(temp, '\0', temp_cap);
                    }
                }
            }
        }
    }
    compile_append_footer(source, source_curr, source_cap, op, compile_loop_idx, op_idx, loop_idx,
                          a_max * z_max * y_max * x_max);
    free(temp);
}
static void compile_loop_to_cl(kernel_t *kernel, const compile_loop_t *compile_loop, const uint64_t global_size,
                               const uint64_t local_size) {
    assert(kernel);
    assert(compile_loop);
    assert(global_size > 0);
    assert(local_size > 0);
    assert(global_size % local_size == 0);
    compile_loop_gather_args(kernel, compile_loop);

    char *source = calloc(INITIAL_SOURCE_SIZE, sizeof(char));
    assert(source);
    char *source_curr = source;
    uint64_t source_cap = INITIAL_SOURCE_SIZE;
    uint64_t splittable = SPLITTABLE(compile_loop->op[0][0]);
    source_curr += snprintf(source_curr, MAX_OP_SIZE, "__kernel void " KERNEL_NAME "(");
    compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
    for(uint64_t arg_idx = 0; arg_idx < kernel->arg_num; arg_idx++) {
        if(!arg_idx) {
            source_curr += snprintf(source_curr, MAX_OP_SIZE, "__global double *%s", kernel->arg_name[arg_idx]);
        } else {
            source_curr += snprintf(source_curr, MAX_OP_SIZE, ",__global double *%s", kernel->arg_name[arg_idx]);
        }
        compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
    }
    source_curr += snprintf(source_curr, MAX_OP_SIZE, ") {\n");
    compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
    source_curr += snprintf(source_curr, MAX_OP_SIZE, "__const int gid = get_global_id(0);\n");
    compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
    source_curr += snprintf(source_curr, MAX_OP_SIZE, "int id;\n");
    compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
    uint64_t loops_left = compile_loop->loop_num % global_size;
    uint64_t loops_per_kernel =
        loops_left ? compile_loop->loop_num / global_size + 1 : compile_loop->loop_num / global_size;
    source_curr += snprintf(source_curr, MAX_OP_SIZE, "id = gid;\n");
    compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
    for(uint64_t loop_idx = 0; loop_idx < loops_per_kernel; loop_idx++) {
        if(loop_idx) {
            source_curr += snprintf(source_curr, MAX_OP_SIZE, "id += %lu;\n", global_size);
            compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
        }
        if(!splittable && loop_idx == loops_per_kernel - 1 && loops_left) {
            source_curr += snprintf(source_curr, MAX_OP_SIZE, "if(gid < %lu) {\n", loops_left);
            compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
        }
        for(uint64_t op_idx = 0; op_idx < compile_loop->op_num; op_idx++) {
            uint64_t a_max = compile_loop->op[op_idx][0].type_op == op_reduce
                                 ? compile_loop->op[op_idx][0].buffer_in.sze_a
                                 : compile_loop->op[op_idx][0].buffer_out.sze_a;
            uint64_t z_max = compile_loop->op[op_idx][0].type_op == op_reduce
                                 ? compile_loop->op[op_idx][0].buffer_in.sze_z
                                 : compile_loop->op[op_idx][0].buffer_out.sze_z;
            uint64_t y_max = compile_loop->op[op_idx][0].type_op == op_reduce
                                 ? compile_loop->op[op_idx][0].buffer_in.sze_y
                                 : compile_loop->op[op_idx][0].buffer_out.sze_y;
            uint64_t x_max = compile_loop->op[op_idx][0].type_op == op_reduce
                                 ? compile_loop->op[op_idx][0].buffer_in.sze_x
                                 : compile_loop->op[op_idx][0].buffer_out.sze_x;
            uint64_t total_op_num = a_max * z_max * y_max * x_max;
            compile_append_op_index(&source, &source_curr, &source_cap, compile_loop->op[op_idx],
                                    compile_loop->dim_info[op_idx], compile_loop->inline_type[op_idx],
                                    compile_loop->inline_num[op_idx], compile_loop->loop_num, 0, op_idx, loop_idx,
                                    splittable, global_size);
            compile_append_op(&source, &source_curr, &source_cap, compile_loop->op[op_idx],
                              compile_loop->dim_info[op_idx], compile_loop->inline_type[op_idx],
                              compile_loop->inline_num[op_idx], compile_loop->loop_num, 0, op_idx, loop_idx,
                              global_size, splittable);
        }
        if(!splittable && loop_idx == loops_per_kernel - 1 && loops_left) {
            source_curr += snprintf(source_curr, MAX_OP_SIZE, "}\n");
            compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
        }
    }
    source_curr += snprintf(source_curr, MAX_OP_SIZE, "}\n");
    compile_expand_source(&source, &source_curr, &source_cap, MAX_OP_SIZE);
    kernel->source = source;
    kernel->source_cap = source_cap;
}
void program_compile(program_t *program, const linearized_t *linearized, const cl_device_id *device_id,
                     const cl_context *context, const cl_command_queue *command_queue, const uint64_t global_size,
                     const uint64_t local_size) {
    assert(program);
    assert(linearized);
    assert(device_id);
    assert(context);
    assert(command_queue);
    assert(*device_id);
    assert(*context);
    assert(*command_queue);
    /* Having a global or local size of 1 is really stupid but it should be supported. */
    assert(global_size > 0);
    assert(local_size > 0);
    assert(!(global_size % local_size));
    if(!linearized->op_len) {
        return;
    }
    simple_loop_t simple = {0};
    compile_loop_t compile = {0};
    uint64_t kernel_num = 0;
    uint64_t kernel_cap = INITIAL_CAP;
    program->kernel = calloc(INITIAL_CAP, sizeof(kernel_t));
    uint64_t op_idx = 0;
    while(op_idx < linearized->op_len) {
        op_idx += simple_loop_from_linearized_index(&simple, linearized, op_idx);
        compile = compile_loop_alloc(&simple);
        compile_loop_to_cl(&program->kernel[kernel_num], &compile, global_size, local_size);
        program->kernel[kernel_num].cl_program = NULL;
        program->kernel[kernel_num].cl_kernel = NULL;
        kernel_num++;
        if(kernel_num == kernel_cap) {
            kernel_cap *= 2;
            program->kernel = reallocarray(program->kernel, kernel_cap, sizeof(kernel_t));
            assert(program->kernel);
        }
        compile_loop_free(&compile);
    }
    simple_loop_free(&simple);
    program->kernel_num = kernel_num;
    program->kernel_cap = kernel_cap;
    program->local_size = local_size;
    program->global_size = global_size;
    program->cl_device_id = (cl_device_id *) device_id;
    program->cl_context = (cl_context *) context;
    program->cl_command_queue = (cl_command_queue *) command_queue;
}
void program_free(program_t *program) {
    for(uint64_t kernel_idx = 0; kernel_idx < program->kernel_num; kernel_idx++) {
        for(uint64_t arg_idx = 0; arg_idx < program->kernel[kernel_idx].arg_num; arg_idx++) {
            free(program->kernel[kernel_idx].arg_name[arg_idx]);
        }
        free(program->kernel[kernel_idx].arg_name);
        free(program->kernel[kernel_idx].arg_mem);
        free(program->kernel[kernel_idx].source);
        if(program->kernel[kernel_idx].cl_kernel) {
            clReleaseKernel(program->kernel[kernel_idx].cl_kernel);
        }
        if(program->kernel[kernel_idx].cl_program) {
            clReleaseProgram(program->kernel[kernel_idx].cl_program);
        }
    }
    free(program->kernel);
    program->kernel = NULL;
    program->kernel_num = 0;
    if(program->cl_device_id) {
        if(*program->cl_device_id) {
            clReleaseDevice(*program->cl_device_id);
            *program->cl_device_id = NULL;
        }
        program->cl_device_id = NULL;
    }
    if(program->cl_context) {
        if(*program->cl_context) {
            clReleaseContext(*program->cl_context);
            *program->cl_context = NULL;
        }
        program->cl_context = NULL;
    }
    if(program->cl_command_queue) {
        if(*program->cl_command_queue) {
            clReleaseCommandQueue(*program->cl_command_queue);
            *program->cl_command_queue = NULL;
        }
        program->cl_command_queue = NULL;
    }
}
// if(compile->dim_info[i][j].off_a_in) {
//     free(compile->dim_info[i][j].off_a_in);
// }
// if(compile->dim_info[i][j].off_z_in) {
//     free(compile->dim_info[i][j].off_z_in);
// }
// if(compile->dim_info[i][j].off_y_in) {
//     free(compile->dim_info[i][j].off_y_in);
// }
// if(compile->dim_info[i][j].off_x_in) {
//     free(compile->dim_info[i][j].off_x_in);
// }
// if(compile->dim_info[i][j].off_a_out) {
//     free(compile->dim_info[i][j].off_a_out);
// }
// if(compile->dim_info[i][j].off_z_out) {
//     free(compile->dim_info[i][j].off_z_out);
// }
// if(compile->dim_info[i][j].off_y_out) {
//     free(compile->dim_info[i][j].off_y_out);
// }
// if(compile->dim_info[i][j].off_x_out) {
//     free(compile->dim_info[i][j].off_x_out);
// }
// if(compile->dim_info[i][j].str_a_in) {
//     free(compile->dim_info[i][j].str_a_in);
// }
// if(compile->dim_info[i][j].str_z_in) {
//     free(compile->dim_info[i][j].str_z_in);
// }
// if(compile->dim_info[i][j].str_y_in) {
//     free(compile->dim_info[i][j].str_y_in);
// }
// if(compile->dim_info[i][j].str_x_in) {
//     free(compile->dim_info[i][j].str_x_in);
// }
// if(compile->dim_info[i][j].str_a_out) {
//     free(compile->dim_info[i][j].str_a_out);
// }
// if(compile->dim_info[i][j].str_z_out) {
//     free(compile->dim_info[i][j].str_z_out);
// }
// if(compile->dim_info[i][j].str_y_out) {
//     free(compile->dim_info[i][j].str_y_out);
// }
// if(compile->dim_info[i][j].str_x_out) {
//     free(compile->dim_info[i][j].str_x_out);
// }
// if(compile->dim_info[i][j].wait_a_in) {
//     free(compile->dim_info[i][j].wait_a_in);
// }
// if(compile->dim_info[i][j].wait_z_in) {
//     free(compile->dim_info[i][j].wait_z_in);
// }
// if(compile->dim_info[i][j].wait_y_in) {
//     free(compile->dim_info[i][j].wait_y_in);
// }
// if(compile->dim_info[i][j].wait_x_in) {
//     free(compile->dim_info[i][j].wait_x_in);
// }
// if(compile->dim_info[i][j].wai_a_out) {
//     free(compile->dim_info[i][j].wai_a_out);
// }
// if(compile->dim_info[i][j].wai_z_out) {
//     free(compile->dim_info[i][j].wai_z_out);
// }
// if(compile->dim_info[i][j].wai_y_out) {
//     free(compile->dim_info[i][j].wai_y_out);
// }
// if(compile->dim_info[i][j].wai_x_out) {
//     free(compile->dim_info[i][j].wai_x_out);
// }
// if(compile->dim_info[i][j].res_a_in) {
//     free(compile->dim_info[i][j].res_a_in);
// }
// if(compile->dim_info[i][j].res_z_in) {
//     free(compile->dim_info[i][j].res_z_in);
// }
// if(compile->dim_info[i][j].res_y_in) {
//     free(compile->dim_info[i][j].res_y_in);
// }
// if(compile->dim_info[i][j].res_x_in) {
//     free(compile->dim_info[i][j].res_x_in);
// }
// if(compile->dim_info[i][j].res_a_out) {
//     free(compile->dim_info[i][j].res_a_out);
// }
// if(compile->dim_info[i][j].res_z_out) {
//     free(compile->dim_info[i][j].res_z_out);
// }
// if(compile->dim_info[i][j].res_y_out) {
//     free(compile->dim_info[i][j].res_y_out);
// }
// if(compile->dim_info[i][j].res_x_out) {
//     free(compile->dim_info[i][j].res_x_out);
// }
//
//
//
// return starting->type_op == compared->type_op && starting->type_unary == compared->type_unary &&
//        starting->type_binary == compared->type_binary && starting->type_reduce == compared->type_reduce &&
//        starting->buffer_out.name_off == compared->buffer_out.name_off &&
//        starting->buffer_out.sze_a == compared->buffer_out.sze_a &&
//        starting->buffer_out.sze_z == compared->buffer_out.sze_z &&
//        starting->buffer_out.sze_y == compared->buffer_out.sze_y &&
//        starting->buffer_out.sze_x == compared->buffer_out.sze_x &&
//        (starting->type_op == op_unary || (starting->buffer_in.name_off == compared->buffer_in.name_off &&
//                                           starting->buffer_in.sze_a == compared->buffer_in.sze_a &&
//                                           starting->buffer_in.sze_z == compared->buffer_in.sze_z &&
//                                           starting->buffer_in.sze_y == compared->buffer_in.sze_y &&
//                                           starting->buffer_in.sze_x == compared->buffer_in.sze_x));
//
//
// *source_curr += snprintf(*source_curr, MAX_OP_SIZE, "%ld",
//                          loop->dim_info[op_idx][0].off_out +
//                              (loop_idx % loop->dim_info[op_idx][0].res_a_out /
//                               loop->dim_info[op_idx][0].wai_a_out * loop->dim_info[op_idx][0].str_a_out) +
//                              (loop_idx % loop->dim_info[op_idx][0].res_z_out /
//                               loop->dim_info[op_idx][0].wai_z_out * loop->dim_info[op_idx][0].str_z_out) +
//                              (loop_idx % loop->dim_info[op_idx][0].res_y_out /
//                               loop->dim_info[op_idx][0].wai_y_out * loop->dim_info[op_idx][0].str_y_out) +
//                              (loop_idx % loop->dim_info[op_idx][0].res_x_out /
//                               loop->dim_info[op_idx][0].wai_x_out * loop->dim_info[op_idx][0].str_x_out));
//
//
//
// *source_curr += snprintf(
//     *source_curr, MAX_OP_SIZE,
//     "__const int "
//     "%s_%lu_%lu_%d_%lu_%lu=%lu+((id/%lu)%%%lu)/%lu*%lu+((id/%lu)%%%lu)/%lu*%lu+((id/%lu)%%%lu)/%lu*%lu+((id/%lu)%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu+(id%%%lu)/%lu*%lu;/*1*/\n",
//     op[0].buffer_out.name, compile_loop_idx, op_idx, 0, loop_idx, op_unique_idx, dim_info[0].off_out,
//     op_size, dim_info[0].res_a_out, dim_info[0].wai_a_out, dim_info[0].str_a_out * op[0].buffer_out.str_a,
//     op_size, dim_info[0].res_z_out, dim_info[0].wai_z_out, dim_info[0].str_z_out * op[0].buffer_out.str_z,
//     op_size, dim_info[0].res_y_out, dim_info[0].wai_y_out, dim_info[0].str_y_out * op[0].buffer_out.str_y,
//     op_size, dim_info[0].res_x_out, dim_info[0].wai_x_out, dim_info[0].str_x_out * op[0].buffer_out.str_x,
//     op[0].buffer_out.str_a * op[0].buffer_out.sze_a, op[0].buffer_out.sze_z * op[0].buffer_out.sze_y *
//     op[0].buffer_out.sze_x, op[0].buffer_out.str_a, op[0].buffer_out.str_a                         ,
//     op[0].buffer_out.sze_y * op[0].buffer_out.sze_x                         , op[0].buffer_out.str_z,
//     op[0].buffer_out.str_z                         , op[0].buffer_out.sze_x , op[0].buffer_out.str_y,
//     op[0].buffer_out.str_y                         , 1LU , op[0].buffer_out.str_x
// );
