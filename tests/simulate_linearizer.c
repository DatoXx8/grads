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
            switch(temp->move_type) {
                case move_reshape: {
                    temp->out_buffer->sim_a_sze = temp->var_a;
                    temp->out_buffer->sim_z_sze = temp->var_z;
                    temp->out_buffer->sim_y_sze = temp->var_y;
                    temp->out_buffer->sim_x_sze = temp->var_x;
                    temp->out_buffer->sim_a_str = temp->var_z * temp->var_y * temp->var_x;
                    temp->out_buffer->sim_z_str = temp->var_y * temp->var_x;
                    temp->out_buffer->sim_y_str = temp->var_x;
                    temp->out_buffer->sim_x_str = 1;
                    break;
                }
                case move_resize: {
                    temp->out_buffer->sim_a_sze = temp->var_a;
                    temp->out_buffer->sim_z_sze = temp->var_z;
                    temp->out_buffer->sim_y_sze = temp->var_y;
                    temp->out_buffer->sim_x_sze = temp->var_x;
                    break;
                }
                case move_offset: {
                    temp->out_buffer->sim_off = temp->out_buffer->sim_a_str * temp->var_a + temp->out_buffer->sim_z_str * temp->var_z +
                                              temp->out_buffer->sim_y_str * temp->var_y + temp->out_buffer->sim_x_str * temp->var_x;
                    temp->out_buffer->sim_a_off = temp->var_a;
                    temp->out_buffer->sim_z_off = temp->var_z;
                    temp->out_buffer->sim_y_off = temp->var_y;
                    temp->out_buffer->sim_x_off = temp->var_x;
                    break;
                }
            }
        } else {
            assert(linearized->simple[index].type == temp->type);
            assert(linearized->simple[index].unary_type == temp->unary_type);
            assert(linearized->simple[index].binary_type == temp->binary_type);
            assert(linearized->simple[index].reduce_type == temp->reduce_type);
            assert(linearized->simple[index].var_unary == temp->var_unary);
            assert(linearized->simple[index].out_buffer.a_sze == temp->out_buffer->sim_a_sze);
            assert(linearized->simple[index].out_buffer.z_sze == temp->out_buffer->sim_z_sze);
            assert(linearized->simple[index].out_buffer.y_sze == temp->out_buffer->sim_y_sze);
            assert(linearized->simple[index].out_buffer.x_sze == temp->out_buffer->sim_x_sze);
            assert(linearized->simple[index].out_buffer.a_str == temp->out_buffer->sim_a_str);
            assert(linearized->simple[index].out_buffer.z_str == temp->out_buffer->sim_z_str);
            assert(linearized->simple[index].out_buffer.y_str == temp->out_buffer->sim_y_str);
            assert(linearized->simple[index].out_buffer.x_str == temp->out_buffer->sim_x_str);
            assert(linearized->simple[index].out_buffer.a_off == temp->out_buffer->sim_a_off);
            assert(linearized->simple[index].out_buffer.z_off == temp->out_buffer->sim_z_off);
            assert(linearized->simple[index].out_buffer.y_off == temp->out_buffer->sim_y_off);
            assert(linearized->simple[index].out_buffer.x_off == temp->out_buffer->sim_x_off);
            if(linearized->simple[index].type != operation_unary) {
                assert(linearized->simple[index].in_buffer.a_sze == temp->in_buffer->sim_a_sze);
                assert(linearized->simple[index].in_buffer.z_sze == temp->in_buffer->sim_z_sze);
                assert(linearized->simple[index].in_buffer.y_sze == temp->in_buffer->sim_y_sze);
                assert(linearized->simple[index].in_buffer.x_sze == temp->in_buffer->sim_x_sze);
                assert(linearized->simple[index].in_buffer.a_str == temp->in_buffer->sim_a_str);
                assert(linearized->simple[index].in_buffer.z_str == temp->in_buffer->sim_z_str);
                assert(linearized->simple[index].in_buffer.y_str == temp->in_buffer->sim_y_str);
                assert(linearized->simple[index].in_buffer.x_str == temp->in_buffer->sim_x_str);
                assert(linearized->simple[index].in_buffer.a_off == temp->in_buffer->sim_a_off);
                assert(linearized->simple[index].in_buffer.z_off == temp->in_buffer->sim_z_off);
                assert(linearized->simple[index].in_buffer.y_off == temp->in_buffer->sim_y_off);
                assert(linearized->simple[index].in_buffer.x_off == temp->in_buffer->sim_x_off);
            }
            index++;
        }
        next = temp->child_count > 0 ? temp->child[0] : op;
        op_cleanup(temp);
        op_free(temp);
        free(temp);
    }
    if(op->type == operation_move) {
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
    } else {
        assert(linearized->simple[index].type == op->type);
        assert(linearized->simple[index].unary_type == op->unary_type);
        assert(linearized->simple[index].binary_type == op->binary_type);
        assert(linearized->simple[index].reduce_type == op->reduce_type);
        assert(linearized->simple[index].var_unary == op->var_unary);
        assert(linearized->simple[index].out_buffer.a_sze == op->out_buffer->sim_a_sze);
        assert(linearized->simple[index].out_buffer.z_sze == op->out_buffer->sim_z_sze);
        assert(linearized->simple[index].out_buffer.y_sze == op->out_buffer->sim_y_sze);
        assert(linearized->simple[index].out_buffer.x_sze == op->out_buffer->sim_x_sze);
        assert(linearized->simple[index].out_buffer.a_str == op->out_buffer->sim_a_str);
        assert(linearized->simple[index].out_buffer.z_str == op->out_buffer->sim_z_str);
        assert(linearized->simple[index].out_buffer.y_str == op->out_buffer->sim_y_str);
        assert(linearized->simple[index].out_buffer.x_str == op->out_buffer->sim_x_str);
        assert(linearized->simple[index].out_buffer.a_off == op->out_buffer->sim_a_off);
        assert(linearized->simple[index].out_buffer.z_off == op->out_buffer->sim_z_off);
        assert(linearized->simple[index].out_buffer.y_off == op->out_buffer->sim_y_off);
        assert(linearized->simple[index].out_buffer.x_off == op->out_buffer->sim_x_off);
        if(linearized->simple[index].type != operation_unary) {
            assert(linearized->simple[index].in_buffer.a_sze == op->in_buffer->sim_a_sze);
            assert(linearized->simple[index].in_buffer.z_sze == op->in_buffer->sim_z_sze);
            assert(linearized->simple[index].in_buffer.y_sze == op->in_buffer->sim_y_sze);
            assert(linearized->simple[index].in_buffer.x_sze == op->in_buffer->sim_x_sze);
            assert(linearized->simple[index].in_buffer.a_str == op->in_buffer->sim_a_str);
            assert(linearized->simple[index].in_buffer.z_str == op->in_buffer->sim_z_str);
            assert(linearized->simple[index].in_buffer.y_str == op->in_buffer->sim_y_str);
            assert(linearized->simple[index].in_buffer.x_str == op->in_buffer->sim_x_str);
            assert(linearized->simple[index].in_buffer.a_off == op->in_buffer->sim_a_off);
            assert(linearized->simple[index].in_buffer.z_off == op->in_buffer->sim_z_off);
            assert(linearized->simple[index].in_buffer.y_off == op->in_buffer->sim_y_off);
            assert(linearized->simple[index].in_buffer.x_off == op->in_buffer->sim_x_off);
        }
        index++;
    }
    op_cleanup(op);
    op_free(op);
    free(op);
    assert(linearized->op_count == index);
    
    // int64_t index = 0;
    // assert(linearized);
    // assert(op);
    // while(op->parent_count) {
    //     op_t *temp = op;
    //     for(int64_t i = 0; i < MAX_DEPTH; i++) {
    //         if(temp->parent_count) {
    //             temp = temp->parent[0];
    //         } else {
    //             break;
    //         }
    //     }
    //     assert(temp);
    //     assert(temp->parent_count == 0);
    //     if(temp->type == operation_move) {
    //         switch(temp->move_type) {
    //             case move_reshape: {
    //                 temp->out_buffer->sim_a_sze = temp->var_a;
    //                 temp->out_buffer->sim_z_sze = temp->var_z;
    //                 temp->out_buffer->sim_y_sze = temp->var_y;
    //                 temp->out_buffer->sim_x_sze = temp->var_x;
    //                 temp->out_buffer->sim_a_str = temp->var_z * temp->var_y * temp->var_x;
    //                 temp->out_buffer->sim_z_str = temp->var_y * temp->var_x;
    //                 temp->out_buffer->sim_y_str = temp->var_x;
    //                 temp->out_buffer->sim_x_str = 1;
    //                 break;
    //             }
    //             case move_resize: {
    //                 temp->out_buffer->sim_a_sze = temp->var_a;
    //                 temp->out_buffer->sim_z_sze = temp->var_z;
    //                 temp->out_buffer->sim_y_sze = temp->var_y;
    //                 temp->out_buffer->sim_x_sze = temp->var_x;
    //                 break;
    //             }
    //             case move_offset: {
    //                 temp->out_buffer->sim_off = temp->out_buffer->sim_a_str * temp->var_a + temp->out_buffer->sim_z_str * temp->var_z +
    //                                           temp->out_buffer->sim_y_str * temp->var_y + temp->out_buffer->sim_x_str * temp->var_x;
    //                 temp->out_buffer->sim_a_off = temp->var_a;
    //                 temp->out_buffer->sim_z_off = temp->var_z;
    //                 temp->out_buffer->sim_y_off = temp->var_y;
    //                 temp->out_buffer->sim_x_off = temp->var_x;
    //                 break;
    //             }
    //         }
    //     } else {
    //         assert(linearized->simple[index].type == temp->type);
    //         assert(linearized->simple[index].unary_type == temp->unary_type);
    //         assert(linearized->simple[index].binary_type == temp->binary_type);
    //         assert(linearized->simple[index].reduce_type == temp->reduce_type);
    //         assert(linearized->simple[index].var_unary == temp->var_unary);
    //         assert(linearized->simple[index].out_buffer.a_sze == temp->out_buffer->sim_a_sze);
    //         assert(linearized->simple[index].out_buffer.z_sze == temp->out_buffer->sim_z_sze);
    //         assert(linearized->simple[index].out_buffer.y_sze == temp->out_buffer->sim_y_sze);
    //         assert(linearized->simple[index].out_buffer.x_sze == temp->out_buffer->sim_x_sze);
    //         assert(linearized->simple[index].out_buffer.a_str == temp->out_buffer->sim_a_str);
    //         assert(linearized->simple[index].out_buffer.z_str == temp->out_buffer->sim_z_str);
    //         assert(linearized->simple[index].out_buffer.y_str == temp->out_buffer->sim_y_str);
    //         assert(linearized->simple[index].out_buffer.x_str == temp->out_buffer->sim_x_str);
    //         assert(linearized->simple[index].out_buffer.a_off == temp->out_buffer->sim_a_off);
    //         assert(linearized->simple[index].out_buffer.z_off == temp->out_buffer->sim_z_off);
    //         assert(linearized->simple[index].out_buffer.y_off == temp->out_buffer->sim_y_off);
    //         assert(linearized->simple[index].out_buffer.x_off == temp->out_buffer->sim_x_off);
    //         if(linearized->simple[index].type != operation_unary) {
    //             assert(linearized->simple[index].in_buffer.a_sze == temp->in_buffer->sim_a_sze);
    //             assert(linearized->simple[index].in_buffer.z_sze == temp->in_buffer->sim_z_sze);
    //             assert(linearized->simple[index].in_buffer.y_sze == temp->in_buffer->sim_y_sze);
    //             assert(linearized->simple[index].in_buffer.x_sze == temp->in_buffer->sim_x_sze);
    //             assert(linearized->simple[index].in_buffer.a_str == temp->in_buffer->sim_a_str);
    //             assert(linearized->simple[index].in_buffer.z_str == temp->in_buffer->sim_z_str);
    //             assert(linearized->simple[index].in_buffer.y_str == temp->in_buffer->sim_y_str);
    //             assert(linearized->simple[index].in_buffer.x_str == temp->in_buffer->sim_x_str);
    //             assert(linearized->simple[index].in_buffer.a_off == temp->in_buffer->sim_a_off);
    //             assert(linearized->simple[index].in_buffer.z_off == temp->in_buffer->sim_z_off);
    //             assert(linearized->simple[index].in_buffer.y_off == temp->in_buffer->sim_y_off);
    //             assert(linearized->simple[index].in_buffer.x_off == temp->in_buffer->sim_x_off);
    //         }
    //         index++;
    //     }
    //     op_cleanup(temp);
    //     op_free(temp);
    //     free(temp);
    // }
    // if(op->type == operation_move) {
    //     assert(op);
    //     switch(op->move_type) {
    //         case move_reshape: {
    //             op->out_buffer->sim_a_sze = op->var_a;
    //             op->out_buffer->sim_z_sze = op->var_z;
    //             op->out_buffer->sim_y_sze = op->var_y;
    //             op->out_buffer->sim_x_sze = op->var_x;
    //             op->out_buffer->sim_a_str = op->var_z * op->var_y * op->var_x;
    //             op->out_buffer->sim_z_str = op->var_y * op->var_x;
    //             op->out_buffer->sim_y_str = op->var_x;
    //             op->out_buffer->sim_x_str = 1;
    //             break;
    //         }
    //         case move_resize: {
    //             op->out_buffer->sim_a_sze = op->var_a;
    //             op->out_buffer->sim_z_sze = op->var_z;
    //             op->out_buffer->sim_y_sze = op->var_y;
    //             op->out_buffer->sim_x_sze = op->var_x;
    //             break;
    //         }
    //         case move_offset: {
    //             op->out_buffer->sim_off = op->out_buffer->sim_a_str * op->var_a + op->out_buffer->sim_z_str * op->var_z +
    //                                         op->out_buffer->sim_y_str * op->var_y + op->out_buffer->sim_x_str * op->var_x;
    //             op->out_buffer->sim_a_off = op->var_a;
    //             op->out_buffer->sim_z_off = op->var_z;
    //             op->out_buffer->sim_y_off = op->var_y;
    //             op->out_buffer->sim_x_off = op->var_x;
    //             break;
    //         }
    //     }
    // } else {
    //     assert(linearized->simple[index].type == op->type);
    //     assert(linearized->simple[index].unary_type == op->unary_type);
    //     assert(linearized->simple[index].binary_type == op->binary_type);
    //     assert(linearized->simple[index].reduce_type == op->reduce_type);
    //     assert(linearized->simple[index].var_unary == op->var_unary);
    //     assert(linearized->simple[index].out_buffer.a_sze == op->out_buffer->sim_a_sze);
    //     assert(linearized->simple[index].out_buffer.z_sze == op->out_buffer->sim_z_sze);
    //     assert(linearized->simple[index].out_buffer.y_sze == op->out_buffer->sim_y_sze);
    //     assert(linearized->simple[index].out_buffer.x_sze == op->out_buffer->sim_x_sze);
    //     assert(linearized->simple[index].out_buffer.a_str == op->out_buffer->sim_a_str);
    //     assert(linearized->simple[index].out_buffer.z_str == op->out_buffer->sim_z_str);
    //     assert(linearized->simple[index].out_buffer.y_str == op->out_buffer->sim_y_str);
    //     assert(linearized->simple[index].out_buffer.x_str == op->out_buffer->sim_x_str);
    //     assert(linearized->simple[index].out_buffer.a_off == op->out_buffer->sim_a_off);
    //     assert(linearized->simple[index].out_buffer.z_off == op->out_buffer->sim_z_off);
    //     assert(linearized->simple[index].out_buffer.y_off == op->out_buffer->sim_y_off);
    //     assert(linearized->simple[index].out_buffer.x_off == op->out_buffer->sim_x_off);
    //     if(linearized->simple[index].type != operation_unary) {
    //         assert(linearized->simple[index].in_buffer.a_sze == op->in_buffer->sim_a_sze);
    //         assert(linearized->simple[index].in_buffer.z_sze == op->in_buffer->sim_z_sze);
    //         assert(linearized->simple[index].in_buffer.y_sze == op->in_buffer->sim_y_sze);
    //         assert(linearized->simple[index].in_buffer.x_sze == op->in_buffer->sim_x_sze);
    //         assert(linearized->simple[index].in_buffer.a_str == op->in_buffer->sim_a_str);
    //         assert(linearized->simple[index].in_buffer.z_str == op->in_buffer->sim_z_str);
    //         assert(linearized->simple[index].in_buffer.y_str == op->in_buffer->sim_y_str);
    //         assert(linearized->simple[index].in_buffer.x_str == op->in_buffer->sim_x_str);
    //         assert(linearized->simple[index].in_buffer.a_off == op->in_buffer->sim_a_off);
    //         assert(linearized->simple[index].in_buffer.z_off == op->in_buffer->sim_z_off);
    //         assert(linearized->simple[index].in_buffer.y_off == op->in_buffer->sim_y_off);
    //         assert(linearized->simple[index].in_buffer.x_off == op->in_buffer->sim_x_off);
    //     }
    //     index++;
    // }
    // op_cleanup(op);
    // op_free(op);
    // free(op);
    // assert(linearized->op_count == index);
}

/* TODO: Validate equivalence between linearized and non-linearized fresh from the tree. */
const int64_t RANDOM_MAX_TRIES = 100;
const int64_t DIM_SIZE = 3;
void simulate_linearize(int64_t op_num, int64_t tensor_num) {
    linearized_t linearized = linearized_alloc();
    tensor_t *tensor = calloc(tensor_num, sizeof(tensor_t));
    tensor_t *tensor_d = calloc(tensor_num, sizeof(tensor_t));
    for(int64_t i = 0; i < tensor_num; i++) {
        tensor[i] = tensor_alloc(DIM_SIZE, DIM_SIZE, DIM_SIZE, DIM_SIZE);
        tensor_d[i] = tensor_alloc(DIM_SIZE, DIM_SIZE, DIM_SIZE, DIM_SIZE);
    }

    /* TODO: I don't think there is a easy way to make sure that all these modulos are accurate, when adding new ops. However I *really* should figure out a way
     * to do that. */
    enum operation_e op_type;
    enum unary_e unary_type;
    enum binary_e binary_type;
    enum reduce_e reduce_type;
    // enum move_e move_type;
    int64_t tensor_out = 0;
    int64_t tensor_in;
    double ran;
    int64_t a_size = DIM_SIZE;
    int64_t z_size = DIM_SIZE;
    int64_t y_size = DIM_SIZE;
    int64_t x_size = DIM_SIZE;
    int64_t a_off = 0;
    int64_t z_off = 0;
    int64_t y_off = 0;
    int64_t x_off = 0;
    int64_t a_temp = 0;
    int64_t z_temp = 0;
    int64_t y_temp = 0;
    int64_t x_temp = 0;
    for(int64_t i = 0; i < op_num; i++) {
        // op_type = rand() % 4;
        op_type = rand() % 3;
        switch(op_type) {
            case operation_unary: {
                unary_type = rand() % 16;
                ran = ((double) rand() / RAND_MAX) * 2 - 1;
                a_size = rand() % DIM_SIZE + 1;
                z_size = rand() % DIM_SIZE + 1;
                y_size = rand() % DIM_SIZE + 1;
                x_size = rand() % DIM_SIZE + 1;
                a_off = (DIM_SIZE > a_size) ? rand() % (1 + DIM_SIZE - a_size) : 0;
                z_off = (DIM_SIZE > z_size) ? rand() % (1 + DIM_SIZE - z_size) : 0;
                y_off = (DIM_SIZE > y_size) ? rand() % (1 + DIM_SIZE - y_size) : 0;
                x_off = (DIM_SIZE > x_size) ? rand() % (1 + DIM_SIZE - x_size) : 0;
                tensor_resize_move(tensor + tensor_out, a_size, z_size, y_size, x_size);
                tensor_offset_move(tensor + tensor_out, a_off, z_off, y_off, x_off);
                tensor_resize_move(tensor_d + tensor_out, a_size, z_size, y_size, x_size);
                tensor_offset_move(tensor_d + tensor_out, a_off, z_off, y_off, x_off);
                switch(unary_type) {
                    case unary_add: {
                        tensor_add_unary(&tensor[tensor_out], ran);
                        tensor_add_unary(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_subtract: {
                        tensor_subtract_unary(&tensor[tensor_out], ran);
                        tensor_subtract_unary(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_multiply: {
                        tensor_multiply_unary(&tensor[tensor_out], ran);
                        tensor_multiply_unary(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_divide: {
                        tensor_divide_unary(&tensor[tensor_out], ran);
                        tensor_divide_unary(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_exp: {
                        tensor_exp_unary(&tensor[tensor_out]);
                        tensor_exp_unary(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_log: {
                        tensor_log_unary(&tensor[tensor_out]);
                        tensor_log_unary(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_square: {
                        tensor_square_unary(&tensor[tensor_out]);
                        tensor_square_unary(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_sqrt: {
                        tensor_sqrt_unary(&tensor[tensor_out]);
                        tensor_sqrt_unary(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_reciprocal: {
                        tensor_reciprocal_unary(&tensor[tensor_out]);
                        tensor_reciprocal_unary(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_max: {
                        tensor_max_unary(&tensor[tensor_out], ran);
                        tensor_max_unary(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_min: {
                        tensor_min_unary(&tensor[tensor_out], ran);
                        tensor_min_unary(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_set: {
                        tensor_set_unary(&tensor[tensor_out], ran);
                        tensor_set_unary(&tensor_d[tensor_out], ran);
                        break;
                    }
                    case unary_random: {
                        tensor_random_unary(&tensor[tensor_out]);
                        tensor_random_unary(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_tanh: {
                        tensor_tanh_unary(&tensor[tensor_out]);
                        tensor_tanh_unary(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_absolute: {
                        tensor_absolute_unary(&tensor[tensor_out]);
                        tensor_absolute_unary(&tensor_d[tensor_out]);
                        break;
                    }
                    case unary_sign: {
                        tensor_sign_unary(&tensor[tensor_out]);
                        tensor_sign_unary(&tensor_d[tensor_out]);
                        break;
                    }
                }
                break;
            }
            case operation_binary: {
                binary_type = rand() % 14;
                for(int64_t rand_idx = 0; rand_idx < RANDOM_MAX_TRIES; rand_idx++) {
                    tensor_in = rand() % tensor_num;
                    if(tensor_in != tensor_out) { break; }
                }
                if(tensor_in == tensor_out) { ERROR("Got really unlucky here or there's a bug with `rand()`\n"); }
                a_size = rand() % DIM_SIZE + 1;
                z_size = rand() % DIM_SIZE + 1;
                y_size = rand() % DIM_SIZE + 1;
                x_size = rand() % DIM_SIZE + 1;
                a_off = (DIM_SIZE > a_size) ? rand() % (1 + DIM_SIZE - a_size) : 0;
                z_off = (DIM_SIZE > z_size) ? rand() % (1 + DIM_SIZE - z_size) : 0;
                y_off = (DIM_SIZE > y_size) ? rand() % (1 + DIM_SIZE - y_size) : 0;
                x_off = (DIM_SIZE > x_size) ? rand() % (1 + DIM_SIZE - x_size) : 0;
                tensor_resize_move(tensor + tensor_in, a_size, z_size, y_size, x_size);
                tensor_offset_move(tensor + tensor_in, a_off, z_off, y_off, x_off);
                tensor_resize_move(tensor_d + tensor_in, a_size, z_size, y_size, x_size);
                tensor_offset_move(tensor_d + tensor_in, a_off, z_off, y_off, x_off);
                a_off = (DIM_SIZE > a_size) ? rand() % (1 + DIM_SIZE - a_size) : 0;
                z_off = (DIM_SIZE > z_size) ? rand() % (1 + DIM_SIZE - z_size) : 0;
                y_off = (DIM_SIZE > y_size) ? rand() % (1 + DIM_SIZE - y_size) : 0;
                x_off = (DIM_SIZE > x_size) ? rand() % (1 + DIM_SIZE - x_size) : 0;
                tensor_resize_move(tensor + tensor_out, a_size, z_size, y_size, x_size);
                tensor_offset_move(tensor + tensor_out, a_off, z_off, y_off, x_off);
                tensor_resize_move(tensor_d + tensor_out, a_size, z_size, y_size, x_size);
                tensor_offset_move(tensor_d + tensor_out, a_off, z_off, y_off, x_off);
                switch(binary_type) {
                    case binary_add: {
                        tensor_add_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_add_binary(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_subtract: {
                        tensor_subtract_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_subtract_binary(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_multiply: {
                        tensor_multiply_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_multiply_binary(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_divide: {
                        tensor_divide_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_divide_binary(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_max: {
                        tensor_max_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_max_binary(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_min: {
                        tensor_min_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_min_binary(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_copy: {
                        tensor_copy_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_copy_binary(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_add_like: {
                        tensor_add_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_add_binary(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_subtract_like: {
                        tensor_subtract_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_subtract_binary(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_multiply_like: {
                        tensor_multiply_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_multiply_binary(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_divide_like: {
                        tensor_divide_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_divide_binary(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_max_like: {
                        tensor_max_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_max_binary(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_min_like: {
                        tensor_min_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_min_binary(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                    case binary_copy_like: {
                        tensor_copy_binary(&tensor[tensor_out], &tensor[tensor_in]);
                        tensor_copy_binary(&tensor_d[tensor_out], &tensor_d[tensor_in]);
                        break;
                    }
                }
                break;
            }
            case operation_reduce: {
                reduce_type = rand() % 4;
                for(int64_t rand_idx = 0; rand_idx < RANDOM_MAX_TRIES; rand_idx++) {
                    tensor_in = rand() % tensor_num;
                    if(tensor_in != tensor_out) { break; }
                }
                if(tensor_in == tensor_out) { ERROR("Got really unlucky here or there's a bug with `rand()`\n"); }
                a_off = (DIM_SIZE > a_size) ? rand() % (1 + DIM_SIZE - a_size) : 0;
                z_off = (DIM_SIZE > z_size) ? rand() % (1 + DIM_SIZE - z_size) : 0;
                y_off = (DIM_SIZE > y_size) ? rand() % (1 + DIM_SIZE - y_size) : 0;
                x_off = (DIM_SIZE > x_size) ? rand() % (1 + DIM_SIZE - x_size) : 0;
                a_temp = rand() % DIM_SIZE;
                z_temp = rand() % DIM_SIZE;
                y_temp = rand() % DIM_SIZE;
                x_temp = rand() % DIM_SIZE;
                tensor_resize_move(tensor + tensor_in, a_size, z_size, y_size, x_size);
                tensor_offset_move(tensor + tensor_in, a_off, z_off, y_off, x_off);
                tensor_resize_move(tensor + tensor_out, 1, 1, 1, 1);
                tensor_offset_move(tensor + tensor_out, a_temp, z_temp, y_temp, x_temp);
                tensor_resize_move(tensor_d + tensor_in, a_size, z_size, y_size, x_size);
                tensor_offset_move(tensor_d + tensor_in, a_off, z_off, y_off, x_off);
                tensor_resize_move(tensor_d + tensor_out, 1, 1, 1, 1);
                tensor_offset_move(tensor_d + tensor_out, a_temp, z_temp, y_temp, x_temp);
                switch(reduce_type) {
                    case reduce_sum: {
                        tensor_sum_reduce(tensor + tensor_out, tensor + tensor_in);
                        tensor_sum_reduce(tensor_d + tensor_out, tensor_d + tensor_in);
                        break;
                    }
                    case reduce_avg: {
                        tensor_avg_reduce(tensor + tensor_out, tensor + tensor_in);
                        tensor_avg_reduce(tensor_d + tensor_out, tensor_d + tensor_in);
                        break;
                    }
                    case reduce_min: {
                        tensor_min_reduce(tensor + tensor_out, tensor + tensor_in);
                        tensor_min_reduce(tensor_d + tensor_out, tensor_d + tensor_in);
                        break;
                    }
                    case reduce_max: {
                        tensor_max_reduce(tensor + tensor_out, tensor + tensor_in);
                        tensor_max_reduce(tensor_d + tensor_out, tensor_d + tensor_in);
                        break;
                    }
                }
                a_off = (DIM_SIZE > a_size) ? rand() % (1 + DIM_SIZE - a_size) : 0;
                z_off = (DIM_SIZE > z_size) ? rand() % (1 + DIM_SIZE - z_size) : 0;
                y_off = (DIM_SIZE > y_size) ? rand() % (1 + DIM_SIZE - y_size) : 0;
                x_off = (DIM_SIZE > x_size) ? rand() % (1 + DIM_SIZE - x_size) : 0;
                tensor_resize_move(tensor + tensor_out, a_size, z_size, y_size, x_size);
                tensor_offset_move(tensor + tensor_out, a_off, z_off, y_off, x_off);
                tensor_resize_move(tensor_d + tensor_out, a_size, z_size, y_size, x_size);
                tensor_offset_move(tensor_d + tensor_out, a_off, z_off, y_off, x_off);
                break;
            }
            case operation_move: {
                break;
            }
        }
    }
    int64_t index = 0;
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
