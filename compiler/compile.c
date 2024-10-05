#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tensor.h"
#include "CL/cl.h"
#include "codegen.h"
#include "compile.h"

/* TODO: Clear up this group_len vs op_num thing. The same thing should only be refered to by the same name */
/* TODO: str_a -> a_str */

static inline _Bool op_equal(const op_t *op1, const op_t *op2) {
    /* I don't think memcmp works here because I think the offsets being irrelevant would mess that up */
    /* Strictly speaking I could modify the struct layout so that I could cast the pointer to uint8_t[something] and
     * just compare the stuff we want to compare but that is straight up horrible and wayyyy to bug prone */
    return op1->type_op == op2->type_op && op1->type_unary == op2->type_unary && op1->type_binary == op2->type_binary &&
           op1->type_reduce == op2->type_reduce && op1->type_move == op2->type_move && op1->var_a == op2->var_a &&
           op1->var_z == op2->var_z && op1->var_y == op2->var_y && op1->var_x == op2->var_x &&
           op1->var_unary == op2->var_unary && op1->buffer_out.name_off == op2->buffer_out.name_off &&
           op1->buffer_out.sze_a == op2->buffer_out.sze_a && op1->buffer_out.sze_z == op2->buffer_out.sze_z &&
           op1->buffer_out.sze_y == op2->buffer_out.sze_y && op1->buffer_out.sze_x == op2->buffer_out.sze_x &&
           op1->buffer_in.name_off == op2->buffer_in.name_off && op1->buffer_in.sze_a == op2->buffer_in.sze_a &&
           op1->buffer_in.sze_z == op2->buffer_in.sze_z && op1->buffer_in.sze_y == op2->buffer_in.sze_y &&
           op1->buffer_in.sze_x == op2->buffer_in.sze_x;
}

op_group_t op_group_alloc(const linearized_t *linearized, const uint64_t start_idx, uint64_t *op_used) {
    assert(linearized);
    assert(start_idx < linearized->op_len);
    assert(op_used);

    op_group_t group = {0};

    uint64_t op_num = 0;
    for(uint64_t op_off = 1; op_off + start_idx < linearized->op_len; op_off++) {
        if(op_equal(&linearized->op[start_idx], &linearized->op[start_idx + op_off])) {
            uint64_t all_same = 1;
            /* No point in the checking inner_off = 0 since that is guaranteed to be true by the if statement above */
            for(uint64_t inner_off = 1; inner_off < op_off; inner_off++) {
                if(!op_equal(&linearized->op[start_idx + inner_off], &linearized->op[start_idx + op_off + inner_off])) {
                    all_same = 0;
                    break;
                }
            }
            if(all_same) {
                op_num = op_off;
                break;
            } else {
                continue;
            }
        }
    }
    if(op_num) {
        uint64_t group_num = 1;
        for(uint64_t op_off = op_num; op_off < linearized->op_len - start_idx; op_off += op_num) {
            if(op_equal(&linearized->op[start_idx], &linearized->op[start_idx + op_off])) {
                group_num++;
            } else {
                break;
            }
        }
        group.repeat_num = group_num;
        group.group_len = op_num;
        group.op = calloc(group.group_len, sizeof(op_t));
        for(uint64_t op_off = 0; op_off < op_num; op_off++) {
            group.op[op_off] = linearized->op[start_idx + op_off];
        }
        *op_used += op_num * group_num;
    } else {
        /* Chop 1 op because no repeat was found. Could maybe match more ops but that is bug prone if done naively */
        group.repeat_num = 1;
        group.group_len = 1;
        group.op = calloc(group.group_len, sizeof(op_t));
        group.op[0] = linearized->op[start_idx];
        *op_used += 1;
    }

    /* TODO: Sort this and error or split up til it is possible to map all indices accurately */
    /* TODO: Merge these loops */
    group.dim_info = calloc(group.group_len, sizeof(dim_info_t));
    for(uint64_t op_idx = 0; op_idx < group.group_len; op_idx++) {
        group.dim_info[op_idx].off_out = linearized->op[start_idx + op_idx].buffer_out.off;
        const uint64_t a_initial = linearized->op[start_idx + op_idx].buffer_out.off_a;
        const uint64_t z_initial = linearized->op[start_idx + op_idx].buffer_out.off_z;
        const uint64_t y_initial = linearized->op[start_idx + op_idx].buffer_out.off_y;
        const uint64_t x_initial = linearized->op[start_idx + op_idx].buffer_out.off_x;
        uint64_t a_left = 0;
        uint64_t z_left = 0;
        uint64_t y_left = 0;
        uint64_t x_left = 0;
        uint64_t a_reenter = 0;
        uint64_t z_reenter = 0;
        uint64_t y_reenter = 0;
        uint64_t x_reenter = 0;
        for(uint64_t repeat_idx = 0; repeat_idx < group.repeat_num; repeat_idx++) {
            if(a_left) {
                if(!a_reenter) {
                    if(linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_out.off_a ==
                       a_initial) {
                        group.dim_info[op_idx].res_a_out = repeat_idx;
                        a_reenter = 1;
                    }
                }
            } else {
                if(linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_out.off_a != a_initial) {
                    group.dim_info[op_idx].wai_a_out = repeat_idx;
                    group.dim_info[op_idx].str_a_out =
                        linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_out.off_a - a_initial;
                    a_left = 1;
                }
            }
            if(z_left) {
                if(!z_reenter) {
                    if(linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_out.off_z ==
                       z_initial) {
                        group.dim_info[op_idx].res_z_out = repeat_idx;
                        z_reenter = 1;
                    }
                }
            } else {
                if(linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_out.off_z != z_initial) {
                    group.dim_info[op_idx].wai_z_out = repeat_idx;
                    group.dim_info[op_idx].str_z_out =
                        linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_out.off_z - z_initial;
                    z_left = 1;
                }
            }
            if(y_left) {
                if(!y_reenter) {
                    if(linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_out.off_y ==
                       y_initial) {
                        group.dim_info[op_idx].res_y_out = repeat_idx;
                        y_reenter = 1;
                    }
                }
            } else {
                if(linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_out.off_y != y_initial) {
                    group.dim_info[op_idx].wai_y_out = repeat_idx;
                    group.dim_info[op_idx].str_y_out =
                        linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_out.off_y - y_initial;
                    y_left = 1;
                }
            }
            if(x_left) {
                if(!x_reenter) {
                    if(linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_out.off_x ==
                       x_initial) {
                        group.dim_info[op_idx].res_x_out = repeat_idx;
                        x_reenter = 1;
                    }
                }
            } else {
                if(linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_out.off_x != x_initial) {
                    group.dim_info[op_idx].wai_x_out = repeat_idx;
                    group.dim_info[op_idx].str_x_out =
                        linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_out.off_x - x_initial;
                    x_left = 1;
                }
            }
        }
    }
    for(uint64_t op_idx = 1; op_idx < group.group_len; op_idx++) {
        group.dim_info[op_idx].off_in = linearized->op[start_idx + op_idx].buffer_in.off;
        const uint64_t a_initial = linearized->op[start_idx + op_idx].buffer_in.off_a;
        const uint64_t z_initial = linearized->op[start_idx + op_idx].buffer_in.off_z;
        const uint64_t y_initial = linearized->op[start_idx + op_idx].buffer_in.off_y;
        const uint64_t x_initial = linearized->op[start_idx + op_idx].buffer_in.off_x;
        uint64_t a_left = 0;
        uint64_t z_left = 0;
        uint64_t y_left = 0;
        uint64_t x_left = 0;
        uint64_t a_reenter = 0;
        uint64_t z_reenter = 0;
        uint64_t y_reenter = 0;
        uint64_t x_reenter = 0;
        for(uint64_t repeat_idx = 0; repeat_idx < group.repeat_num; repeat_idx++) {
            if(a_left) {
                if(!a_reenter) {
                    if(linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_in.off_a == a_initial) {
                        group.dim_info[op_idx].res_a_in = repeat_idx;
                        a_reenter = 1;
                    }
                }
            } else {
                if(linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_in.off_a != a_initial) {
                    group.dim_info[op_idx].wai_a_in = repeat_idx;
                    group.dim_info[op_idx].str_a_in =
                        linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_in.off_a - a_initial;
                    a_left = 1;
                }
            }
            if(z_left) {
                if(!z_reenter) {
                    if(linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_in.off_z == z_initial) {
                        group.dim_info[op_idx].res_z_in = repeat_idx;
                        z_reenter = 1;
                    }
                }
            } else {
                if(linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_in.off_z != z_initial) {
                    group.dim_info[op_idx].wai_z_in = repeat_idx;
                    group.dim_info[op_idx].str_z_in =
                        linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_in.off_z - z_initial;
                    z_left = 1;
                }
            }
            if(y_left) {
                if(!y_reenter) {
                    if(linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_in.off_y == y_initial) {
                        group.dim_info[op_idx].res_y_in = repeat_idx;
                        y_reenter = 1;
                    }
                }
            } else {
                if(linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_in.off_y != y_initial) {
                    group.dim_info[op_idx].wai_y_in = repeat_idx;
                    group.dim_info[op_idx].str_y_in =
                        linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_in.off_y - y_initial;
                    y_left = 1;
                }
            }
            if(x_left) {
                if(!x_reenter) {
                    if(linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_in.off_x == x_initial) {
                        group.dim_info[op_idx].res_x_in = repeat_idx;
                        x_reenter = 1;
                    }
                }
            } else {
                if(linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_in.off_x != x_initial) {
                    group.dim_info[op_idx].wai_x_in = repeat_idx;
                    group.dim_info[op_idx].str_x_in =
                        linearized->op[start_idx + op_idx + repeat_idx * group.group_len].buffer_in.off_x - x_initial;
                    x_left = 1;
                }
            }
        }
    }

    return group;
}
void op_group_free(op_group_t *group) {
    assert(group);
    if(group->op) {
        free(group->op);
        group->dim_info = NULL;
    }
    if(group->dim_info) {
        free(group->dim_info);
        group->dim_info = NULL;
    }
    group->group_len = 0;
    group->repeat_num = 0;
}
void op_group_print(op_group_t *group, int padding, int offset, const char *name) {
    if(!strncmp(name, "", 1)) {
        printf("%*s%s len: %lu, repeats: %lu, ops:\n", offset, "", name, group->group_len, group->repeat_num);
    } else {
        printf("%*sop group len: %lu, repeats: %lu, ops:\n", offset, "", group->group_len, group->repeat_num);
    }
    for(uint64_t op_idx = 0; op_idx < group->group_len; op_idx++) {
        printf("%*s[%lu] - ", offset + padding, "", op_idx);
        op_print(&group->op[op_idx], 0, 0, "");
        if(group->op[op_idx].type_op == op_unary) {
            printf("%*sa {%lu, %lu, %lu, %lu}\n", offset + 2 * padding, "", group->dim_info[op_idx].off_out,
                   group->dim_info[op_idx].str_a_out, group->dim_info[op_idx].wai_a_out,
                   group->dim_info[op_idx].res_a_out);
            printf("%*sz {%lu, %lu, %lu, %lu}\n", offset + 2 * padding, "", group->dim_info[op_idx].off_out,
                   group->dim_info[op_idx].str_z_out, group->dim_info[op_idx].wai_z_out,
                   group->dim_info[op_idx].res_z_out);
            printf("%*sy {%lu, %lu, %lu, %lu}\n", offset + 2 * padding, "", group->dim_info[op_idx].off_out,
                   group->dim_info[op_idx].str_y_out, group->dim_info[op_idx].wai_y_out,
                   group->dim_info[op_idx].res_y_out);
            printf("%*sx {%lu, %lu, %lu, %lu}\n", offset + 2 * padding, "", group->dim_info[op_idx].off_out,
                   group->dim_info[op_idx].str_x_out, group->dim_info[op_idx].wai_x_out,
                   group->dim_info[op_idx].res_x_out);
        } else {
            printf("%*sa {%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", offset + 2 * padding, "",
                   group->dim_info[op_idx].off_out, group->dim_info[op_idx].str_a_out,
                   group->dim_info[op_idx].wai_a_out, group->dim_info[op_idx].res_a_out, group->dim_info[op_idx].off_in,
                   group->dim_info[op_idx].str_a_in, group->dim_info[op_idx].wai_a_in,
                   group->dim_info[op_idx].res_a_in);
            printf("%*sz {%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", offset + 2 * padding, "",
                   group->dim_info[op_idx].off_out, group->dim_info[op_idx].str_z_out,
                   group->dim_info[op_idx].wai_z_out, group->dim_info[op_idx].res_z_out, group->dim_info[op_idx].off_in,
                   group->dim_info[op_idx].str_z_in, group->dim_info[op_idx].wai_z_in,
                   group->dim_info[op_idx].res_z_in);
            printf("%*sy {%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", offset + 2 * padding, "",
                   group->dim_info[op_idx].off_out, group->dim_info[op_idx].str_y_out,
                   group->dim_info[op_idx].wai_y_out, group->dim_info[op_idx].res_y_out, group->dim_info[op_idx].off_in,
                   group->dim_info[op_idx].str_y_in, group->dim_info[op_idx].wai_y_in,
                   group->dim_info[op_idx].res_y_in);
            printf("%*sx {%lu, %lu, %lu, %lu} {%lu, %lu, %lu, %lu}\n", offset + 2 * padding, "",
                   group->dim_info[op_idx].off_out, group->dim_info[op_idx].str_x_out,
                   group->dim_info[op_idx].wai_x_out, group->dim_info[op_idx].res_x_out, group->dim_info[op_idx].off_in,
                   group->dim_info[op_idx].str_x_in, group->dim_info[op_idx].wai_x_in,
                   group->dim_info[op_idx].res_x_in);
        }
    }
}

/* TODO: Also pass in optimization options? */
kernel_t kernel_alloc(const op_group_t *group, const uint64_t optimizations) {
    kernel_t kernel = {0};

    /* TODO: Gather args for kernel */

    kernel.source = compile_op_group(group, optimizations);
    /* TODO: Maybe pass this as a reference to the compile function? I don't really want to do that because it is not
     * beatiful and I want to get rid of source_len anyways when I write my own complete compiler. */
    kernel.source_len = strlen(kernel.source) + 1; /* ' + 1' for '\0' */

    /* TODO: Compile kernel and create program from generated source */

    return kernel;
}
void kernel_free(kernel_t *kernel) {
    assert(kernel);
    if(kernel->source) {
        free(kernel->source);
        kernel->source = NULL;
    }
    if(kernel->arg_name) {
        for(uint64_t arg_idx = 0; arg_idx < kernel->arg_num; arg_idx++) {
            if(kernel->arg_name[arg_idx]) {
                free(kernel->arg_name[arg_idx]);
                kernel->arg_name[arg_idx] = NULL;
            }
        }
        free(kernel->arg_name);
        kernel->arg_name = NULL;
    }
    if(kernel->arg_mem) {
        free(kernel->arg_mem);
        kernel->arg_mem = NULL;
    }
    kernel->arg_num = 0;
    kernel->arg_cap = 0;
    if(kernel->cl_kernel) {
        clReleaseKernel(kernel->cl_kernel);
        kernel->cl_kernel = NULL;
    }
    if(kernel->cl_program) {
        clReleaseProgram(kernel->cl_program);
        kernel->cl_program = NULL;
    }
}

/* TODO: Also pass allowed optimization options and then figure out which ones are good based on the kernel */
program_t program_compile(const linearized_t *linearized, const cl_device_id *device_id, const cl_context *context,
                          const cl_command_queue *command_queue, const uint64_t global_size,
                          const uint64_t local_size) {
    assert(linearized);
    assert(device_id);
    assert(*device_id);
    assert(context);
    assert(*context);
    assert(command_queue);
    assert(*command_queue);
    assert(global_size);
    assert(local_size);
    assert(local_size <= global_size);
    program_t program = {
        .local_size = local_size,
        .global_size = global_size,
        .kernel = NULL,
        .kernel_cap = 0,
        .kernel_num = 0,
        .cl_context = (cl_context *) context,
        .cl_device_id = (cl_device_id *) device_id,
        .cl_command_queue = (cl_command_queue *) command_queue,
    };

    if(!linearized->op_len) {
        // printf("Empty linearized\n");
        return program;
    }

    uint64_t op_used = 0;
    while(op_used < linearized->op_len) {
        op_group_t group = op_group_alloc(linearized, op_used, &op_used);
        op_group_print(&group, 4, 0, "");
        op_group_free(&group);
        printf("Used: %lu\n", op_used);
    }
    return program;
}
void program_free(program_t *program) {
    if(program->cl_context) {
        if(*program->cl_context) {
            clReleaseContext(*program->cl_context);
            *program->cl_context = NULL;
        }
        program->cl_context = NULL;
    }
    if(program->cl_device_id) {
        if(*program->cl_device_id) {
            clReleaseDevice(*program->cl_device_id);
            *program->cl_device_id = NULL;
        }
        program->cl_device_id = NULL;
    }
    if(program->cl_command_queue) {
        if(*program->cl_command_queue) {
            clReleaseCommandQueue(*program->cl_command_queue);
            *program->cl_command_queue = NULL;
        }
        program->cl_command_queue = NULL;
    }
    for(uint64_t kernel_idx = 0; kernel_idx < program->kernel_num; kernel_idx++) {
        kernel_free(&program->kernel[kernel_idx]);
    }
    free(program->kernel);
    program->kernel = NULL;
    program->kernel_num = 0;
    program->kernel_cap = 0;
    program->local_size = 0;
    program->global_size = 0;
}
