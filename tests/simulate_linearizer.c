#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../linearize.h"
#include "../tensor.h"

void linearized_op_tree_equal(linearized_t *linearized, op_t *op) {
    if(!op) { return; }
    int64_t index = 0;
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
        if(temp->type == operation_move) {
            switch(temp->type_move) {
                case move_reshape: {
                    temp->buffer_out->sze_a_sim = temp->var_a;
                    temp->buffer_out->sze_z_sim = temp->var_z;
                    temp->buffer_out->sze_y_sim = temp->var_y;
                    temp->buffer_out->sze_x_sim = temp->var_x;
                    temp->buffer_out->str_a_sim = temp->var_z * temp->var_y * temp->var_x;
                    temp->buffer_out->str_z_sim = temp->var_y * temp->var_x;
                    temp->buffer_out->str_y_sim = temp->var_x;
                    temp->buffer_out->str_x_sim = 1;
                    break;
                }
                case move_resize: {
                    temp->buffer_out->sze_a_sim = temp->var_a;
                    temp->buffer_out->sze_z_sim = temp->var_z;
                    temp->buffer_out->sze_y_sim = temp->var_y;
                    temp->buffer_out->sze_x_sim = temp->var_x;
                    break;
                }
                case move_offset: {
                    temp->buffer_out->off_sim = temp->buffer_out->str_a_sim * temp->var_a + temp->buffer_out->str_z_sim * temp->var_z +
                                              temp->buffer_out->str_y_sim * temp->var_y + temp->buffer_out->str_x_sim * temp->var_x;
                    temp->buffer_out->off_a_sim = temp->var_a;
                    temp->buffer_out->off_z_sim = temp->var_z;
                    temp->buffer_out->off_y_sim = temp->var_y;
                    temp->buffer_out->off_x_sim = temp->var_x;
                    break;
                }
            }
        } else {
            assert(linearized->simple[index].type == temp->type);
            assert(linearized->simple[index].type_unary == temp->type_unary);
            assert(linearized->simple[index].type_binary == temp->type_binary);
            assert(linearized->simple[index].type_reduce == temp->type_reduce);
            assert(linearized->simple[index].var_unary == temp->var_unary);
            assert(linearized->simple[index].buffer_out.sze_a == temp->buffer_out->sze_a_sim);
            assert(linearized->simple[index].buffer_out.sze_z == temp->buffer_out->sze_z_sim);
            assert(linearized->simple[index].buffer_out.sze_y == temp->buffer_out->sze_y_sim);
            assert(linearized->simple[index].buffer_out.sze_x == temp->buffer_out->sze_x_sim);
            assert(linearized->simple[index].buffer_out.str_a == temp->buffer_out->str_a_sim);
            assert(linearized->simple[index].buffer_out.str_z == temp->buffer_out->str_z_sim);
            assert(linearized->simple[index].buffer_out.str_y == temp->buffer_out->str_y_sim);
            assert(linearized->simple[index].buffer_out.str_x == temp->buffer_out->str_x_sim);
            assert(linearized->simple[index].buffer_out.off_a == temp->buffer_out->off_a_sim);
            assert(linearized->simple[index].buffer_out.off_z == temp->buffer_out->off_z_sim);
            assert(linearized->simple[index].buffer_out.off_y == temp->buffer_out->off_y_sim);
            assert(linearized->simple[index].buffer_out.off_x == temp->buffer_out->off_x_sim);
            if(linearized->simple[index].type != operation_unary) {
                assert(linearized->simple[index].buffer_in.sze_a == temp->buffer_in->sze_a_sim);
                assert(linearized->simple[index].buffer_in.sze_z == temp->buffer_in->sze_z_sim);
                assert(linearized->simple[index].buffer_in.sze_y == temp->buffer_in->sze_y_sim);
                assert(linearized->simple[index].buffer_in.sze_x == temp->buffer_in->sze_x_sim);
                assert(linearized->simple[index].buffer_in.str_a == temp->buffer_in->str_a_sim);
                assert(linearized->simple[index].buffer_in.str_z == temp->buffer_in->str_z_sim);
                assert(linearized->simple[index].buffer_in.str_y == temp->buffer_in->str_y_sim);
                assert(linearized->simple[index].buffer_in.str_x == temp->buffer_in->str_x_sim);
                assert(linearized->simple[index].buffer_in.off_a == temp->buffer_in->off_a_sim);
                assert(linearized->simple[index].buffer_in.off_z == temp->buffer_in->off_z_sim);
                assert(linearized->simple[index].buffer_in.off_y == temp->buffer_in->off_y_sim);
                assert(linearized->simple[index].buffer_in.off_x == temp->buffer_in->off_x_sim);
            }
            index++;
        }
        next = temp->child_count > 0 ? temp->child[0] : op;
        op_cleanup(temp);
        op_free(temp);
        free(temp);
    }
    if(op->type == operation_move) {
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
    } else {
        assert(linearized->simple[index].type == op->type);
        assert(linearized->simple[index].type_unary == op->type_unary);
        assert(linearized->simple[index].type_binary == op->type_binary);
        assert(linearized->simple[index].type_reduce == op->type_reduce);
        assert(linearized->simple[index].var_unary == op->var_unary);
        assert(linearized->simple[index].buffer_out.sze_a == op->buffer_out->sze_a_sim);
        assert(linearized->simple[index].buffer_out.sze_z == op->buffer_out->sze_z_sim);
        assert(linearized->simple[index].buffer_out.sze_y == op->buffer_out->sze_y_sim);
        assert(linearized->simple[index].buffer_out.sze_x == op->buffer_out->sze_x_sim);
        assert(linearized->simple[index].buffer_out.str_a == op->buffer_out->str_a_sim);
        assert(linearized->simple[index].buffer_out.str_z == op->buffer_out->str_z_sim);
        assert(linearized->simple[index].buffer_out.str_y == op->buffer_out->str_y_sim);
        assert(linearized->simple[index].buffer_out.str_x == op->buffer_out->str_x_sim);
        assert(linearized->simple[index].buffer_out.off_a == op->buffer_out->off_a_sim);
        assert(linearized->simple[index].buffer_out.off_z == op->buffer_out->off_z_sim);
        assert(linearized->simple[index].buffer_out.off_y == op->buffer_out->off_y_sim);
        assert(linearized->simple[index].buffer_out.off_x == op->buffer_out->off_x_sim);
        if(linearized->simple[index].type != operation_unary) {
            assert(linearized->simple[index].buffer_in.sze_a == op->buffer_in->sze_a_sim);
            assert(linearized->simple[index].buffer_in.sze_z == op->buffer_in->sze_z_sim);
            assert(linearized->simple[index].buffer_in.sze_y == op->buffer_in->sze_y_sim);
            assert(linearized->simple[index].buffer_in.sze_x == op->buffer_in->sze_x_sim);
            assert(linearized->simple[index].buffer_in.str_a == op->buffer_in->str_a_sim);
            assert(linearized->simple[index].buffer_in.str_z == op->buffer_in->str_z_sim);
            assert(linearized->simple[index].buffer_in.str_y == op->buffer_in->str_y_sim);
            assert(linearized->simple[index].buffer_in.str_x == op->buffer_in->str_x_sim);
            assert(linearized->simple[index].buffer_in.off_a == op->buffer_in->off_a_sim);
            assert(linearized->simple[index].buffer_in.off_z == op->buffer_in->off_z_sim);
            assert(linearized->simple[index].buffer_in.off_y == op->buffer_in->off_y_sim);
            assert(linearized->simple[index].buffer_in.off_x == op->buffer_in->off_x_sim);
        }
        index++;
    }
    op_cleanup(op);
    op_free(op);
    free(op);
    assert(linearized->op_len == index);
}

/* TODO: Validate equivalence between linearized and non-linearized fresh from the tree. */
const int64_t RANDOM_MAX_TRIES = 100;
const int64_t DIM_SZE = 3;
void simulate_linearize(int64_t op_num, int64_t tensor_num) {
    linearized_t linearized = linearized_alloc();
    tensor_t *tensor = calloc(tensor_num, sizeof(tensor_t));
    tensor_t *tensor_d = calloc(tensor_num, sizeof(tensor_t));
    for(int64_t i = 0; i < tensor_num; i++) {
        tensor[i] = tensor_alloc(DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE);
        tensor_d[i] = tensor_alloc(DIM_SZE, DIM_SZE, DIM_SZE, DIM_SZE);
    }

    /* TODO: I don't think there is a easy way to make sure that all these modulos are accurate, when adding new ops. However I *really* should figure out a way
     * to do that. */
    enum operation_e op_type;
    enum unary_e type_unary;
    enum binary_e type_binary;
    enum reduce_e type_reduce;
    // enum move_e type_move;
    int64_t tensor_out = 0;
    int64_t tensor_in;
    double ran;
    int64_t sze_a = DIM_SZE;
    int64_t sze_z = DIM_SZE;
    int64_t sze_y = DIM_SZE;
    int64_t sze_x = DIM_SZE;
    int64_t off_a = 0;
    int64_t off_z = 0;
    int64_t off_y = 0;
    int64_t off_x = 0;
    int64_t a_temp = 0;
    int64_t z_temp = 0;
    int64_t y_temp = 0;
    int64_t x_temp = 0;
    for(int64_t i = 0; i < op_num; i++) {
        // op_type = rand() % 4;
        op_type = rand() % 3;
        switch(op_type) {
            case operation_unary: {
                type_unary = rand() % 16;
                ran = ((double) rand() / RAND_MAX) * 2 - 1;
                sze_a = rand() % DIM_SZE + 1;
                sze_z = rand() % DIM_SZE + 1;
                sze_y = rand() % DIM_SZE + 1;
                sze_x = rand() % DIM_SZE + 1;
                off_a = (DIM_SZE > sze_a) ? rand() % (1 + DIM_SZE - sze_a) : 0;
                off_z = (DIM_SZE > sze_z) ? rand() % (1 + DIM_SZE - sze_z) : 0;
                off_y = (DIM_SZE > sze_y) ? rand() % (1 + DIM_SZE - sze_y) : 0;
                off_x = (DIM_SZE > sze_x) ? rand() % (1 + DIM_SZE - sze_x) : 0;
                tensor_move_resize(tensor + tensor_out, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor + tensor_out, off_a, off_z, off_y, off_x);
                tensor_move_resize(tensor_d + tensor_out, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor_d + tensor_out, off_a, off_z, off_y, off_x);
                switch(type_unary) {
                    case unary_add: {
                        tensor_unary_add(&tensor[tensor_out], ran);
                        tensor_unary_add(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_subtract: {
                        tensor_unary_subtract(&tensor[tensor_out], ran);
                        tensor_unary_subtract(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_multiply: {
                        tensor_unary_multiply(&tensor[tensor_out], ran);
                        tensor_unary_multiply(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_divide: {
                        tensor_unary_divide(&tensor[tensor_out], ran);
                        tensor_unary_divide(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_exp: {
                        tensor_unary_exp(&tensor[tensor_out]);
                        tensor_unary_exp(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_log: {
                        tensor_unary_log(&tensor[tensor_out]);
                        tensor_unary_log(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_square: {
                        tensor_unary_square(&tensor[tensor_out]);
                        tensor_unary_square(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_sqrt: {
                        tensor_unary_sqrt(&tensor[tensor_out]);
                        tensor_unary_sqrt(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_reciprocal: {
                        tensor_unary_reciprocal(&tensor[tensor_out]);
                        tensor_unary_reciprocal(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_max: {
                        tensor_unary_max(&tensor[tensor_out], ran);
                        tensor_unary_max(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_min: {
                        tensor_unary_min(&tensor[tensor_out], ran);
                        tensor_unary_min(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_set: {
                        tensor_unary_set(&tensor[tensor_out], ran);
                        tensor_unary_set(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_random: {
                        tensor_unary_random(&tensor[tensor_out]);
                        tensor_unary_random(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_tanh: {
                        tensor_unary_tanh(&tensor[tensor_out]);
                        tensor_unary_tanh(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_absolute: {
                        tensor_unary_absolute(&tensor[tensor_out]);
                        tensor_unary_absolute(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_sign: {
                        tensor_unary_sign(&tensor[tensor_out]);
                        tensor_unary_sign(&tensor_d[tensor_out]);
                        break;
                    }
                }
                break;
            }
            case operation_binary: {
                type_binary = rand() % 14;
                for(int64_t rand_idx = 0; rand_idx < RANDOM_MAX_TRIES; rand_idx++) {
                    tensor_in = rand() % tensor_num;
                    if(tensor_in != tensor_out) { break; }
                }
                if(tensor_in == tensor_out) { ERROR("Got really unlucky here or there's a bug with `rand()`\n"); }
                sze_a = rand() % DIM_SZE + 1;
                sze_z = rand() % DIM_SZE + 1;
                sze_y = rand() % DIM_SZE + 1;
                sze_x = rand() % DIM_SZE + 1;
                off_a = (DIM_SZE > sze_a) ? rand() % (1 + DIM_SZE - sze_a) : 0;
                off_z = (DIM_SZE > sze_z) ? rand() % (1 + DIM_SZE - sze_z) : 0;
                off_y = (DIM_SZE > sze_y) ? rand() % (1 + DIM_SZE - sze_y) : 0;
                off_x = (DIM_SZE > sze_x) ? rand() % (1 + DIM_SZE - sze_x) : 0;
                tensor_move_resize(tensor + tensor_in, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor + tensor_in, off_a, off_z, off_y, off_x);
                tensor_move_resize(tensor_d + tensor_in, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor_d + tensor_in, off_a, off_z, off_y, off_x);
                off_a = (DIM_SZE > sze_a) ? rand() % (1 + DIM_SZE - sze_a) : 0;
                off_z = (DIM_SZE > sze_z) ? rand() % (1 + DIM_SZE - sze_z) : 0;
                off_y = (DIM_SZE > sze_y) ? rand() % (1 + DIM_SZE - sze_y) : 0;
                off_x = (DIM_SZE > sze_x) ? rand() % (1 + DIM_SZE - sze_x) : 0;
                tensor_move_resize(tensor + tensor_out, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor + tensor_out, off_a, off_z, off_y, off_x);
                tensor_move_resize(tensor_d + tensor_out, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor_d + tensor_out, off_a, off_z, off_y, off_x);
                switch(type_binary) {
                    case binary_add: {
                        tensor_binary_add(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_binary_add(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_subtract: {
                        tensor_binary_subtract(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_binary_subtract(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_multiply: {
                        tensor_binary_multiply(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_binary_multiply(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_divide: {
                        tensor_binary_divide(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_binary_divide(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_max: {
                        tensor_binary_max(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_binary_max(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_min: {
                        tensor_binary_min(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_binary_min(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_copy: {
                        tensor_binary_copy(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_binary_copy(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_add_like: {
                        tensor_binary_add(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_binary_add(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_subtract_like: {
                        tensor_binary_subtract(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_binary_subtract(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_multiply_like: {
                        tensor_binary_multiply(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_binary_multiply(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_divide_like: {
                        tensor_binary_divide(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_binary_divide(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_max_like: {
                        tensor_binary_max(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_binary_max(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_min_like: {
                        tensor_binary_min(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_binary_min(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_copy_like: {
                        tensor_binary_copy(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_binary_copy(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                }
                break;
            }
            case operation_reduce: {
                type_reduce = rand() % 4;
                for(int64_t rand_idx = 0; rand_idx < RANDOM_MAX_TRIES; rand_idx++) {
                    tensor_in = rand() % tensor_num;
                    if(tensor_in != tensor_out) { break; }
                }
                if(tensor_in == tensor_out) { ERROR("Got really unlucky here or there's a bug with `rand()`\n"); }
                off_a = (DIM_SZE > sze_a) ? rand() % (1 + DIM_SZE - sze_a) : 0;
                off_z = (DIM_SZE > sze_z) ? rand() % (1 + DIM_SZE - sze_z) : 0;
                off_y = (DIM_SZE > sze_y) ? rand() % (1 + DIM_SZE - sze_y) : 0;
                off_x = (DIM_SZE > sze_x) ? rand() % (1 + DIM_SZE - sze_x) : 0;
                a_temp = rand() % DIM_SZE;
                z_temp = rand() % DIM_SZE;
                y_temp = rand() % DIM_SZE;
                x_temp = rand() % DIM_SZE;
                tensor_move_resize(tensor + tensor_in, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor + tensor_in, off_a, off_z, off_y, off_x);
                tensor_move_resize(tensor + tensor_out, 1, 1, 1, 1);
                tensor_move_offset(tensor + tensor_out, a_temp, z_temp, y_temp, x_temp);
                tensor_move_resize(tensor_d + tensor_in, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor_d + tensor_in, off_a, off_z, off_y, off_x);
                tensor_move_resize(tensor_d + tensor_out, 1, 1, 1, 1);
                tensor_move_offset(tensor_d + tensor_out, a_temp, z_temp, y_temp, x_temp);
                switch(type_reduce) {
                    case reduce_sum: {
                        tensor_reduce_sum(tensor + tensor_out, tensor + tensor_in);
                        tensor_reduce_sum(tensor_d + tensor_out, tensor_d + tensor_in);
                        break;
                    }
                    case reduce_avg: {
                        tensor_reduce_avg(tensor + tensor_out, tensor + tensor_in);
                        tensor_reduce_avg(tensor_d + tensor_out, tensor_d + tensor_in);
                        break;
                    }
                    case reduce_min: {
                        tensor_reduce_min(tensor + tensor_out, tensor + tensor_in);
                        tensor_reduce_min(tensor_d + tensor_out, tensor_d + tensor_in);
                        break;
                    }
                    case reduce_max: {
                        tensor_reduce_max(tensor + tensor_out, tensor + tensor_in);
                        tensor_reduce_max(tensor_d + tensor_out, tensor_d + tensor_in);
                        break;
                    }
                }
                off_a = (DIM_SZE > sze_a) ? rand() % (1 + DIM_SZE - sze_a) : 0;
                off_z = (DIM_SZE > sze_z) ? rand() % (1 + DIM_SZE - sze_z) : 0;
                off_y = (DIM_SZE > sze_y) ? rand() % (1 + DIM_SZE - sze_y) : 0;
                off_x = (DIM_SZE > sze_x) ? rand() % (1 + DIM_SZE - sze_x) : 0;
                tensor_move_resize(tensor + tensor_out, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor + tensor_out, off_a, off_z, off_y, off_x);
                tensor_move_resize(tensor_d + tensor_out, sze_a, sze_z, sze_y, sze_x);
                tensor_move_offset(tensor_d + tensor_out, off_a, off_z, off_y, off_x);
                break;
            }
            case operation_move: {
                break;
            }
        }
    }
    linearized_from_op(&linearized, tensor[tensor_out].op);
    linearized_op_tree_equal(&linearized, tensor_d[tensor_out].op);
    linearized_run(&linearized);

    for(int64_t i = 0; i < tensor_num; i++) {
        linearized_from_op(&linearized, tensor[i].op);
        linearized_from_op(&linearized, tensor_d[i].op);
        tensor_free(&tensor[i]);
        tensor_free(&tensor_d[i]);
    }
    free(tensor);
    free(tensor_d);
    linearized_free(&linearized);
}

int main(int argc, char **argv) {
    if(argc == 4) {
        const uint32_t seed = time(NULL);
        printf("RNG Seed %u\n", seed);
        srand(seed);
        const int64_t op_num = strtoll(argv[1], NULL, 10);
        const int64_t tensor_num = strtoll(argv[2], NULL, 10);
        const int64_t iter_num = strtoll(argv[3], NULL, 10);
        assert(op_num > 0);
        assert(tensor_num > 1);
        assert(iter_num > 0);
        for(int64_t iter_idx = 0; iter_idx < iter_num; iter_idx++) { simulate_linearize(op_num, tensor_num); }
        printf("Passed linearizer simulation with %lu ops, %lu tensors, and %lu iteratations!\n", op_num, tensor_num, iter_num);
        return 0;
    }
    printf("USAGE: %s [number of ops] [number of tensors] [number of iterations]\n", argv[0]);
    return 0;
}
