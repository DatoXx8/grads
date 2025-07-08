// Helper file that implements a function that generates randomized linearized ops.
// This is extracted out because it makes it easier to keep consistent prng state for the same seed in every test
const std = @import("std");
const Pcg = std.Random.Pcg;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const grads = @import("grads");
const Tensor = grads.Tensor;
const Linearized = grads.Linearized;
const OpType = grads.Op.Type;
const Runtime = grads.Runtime;

pub const a_size_max: u32 = 7;
pub const z_size_max: u32 = 6;
pub const y_size_max: u32 = 5;
pub const x_size_max: u32 = 4;
pub const tensor_num: u32 = 10;
pub const op_num: u32 = 10;
comptime {
    assert(tensor_num > 1);
    assert(op_num > 0);
}
// $TODO Have chances of changing offsets, sizes, u_var and everything else

// $FIXME If this fails a bunch of memory leaks
pub fn randomLinearized(runtime: Runtime, allocator: Allocator, op_included: [op_num]bool, rng: u64) !struct {
    tensor: [tensor_num]Tensor,
    out_idx: u32,
} {
    var tensor: [tensor_num]Tensor = undefined;

    var pcg = Pcg.init(rng);

    var op_type: [op_num]OpType = undefined;
    var op_out: [op_num]u32 = undefined;
    var op_in: [op_num]u32 = undefined;

    for (0..op_num) |op_idx| {
        op_type[op_idx] = pcg.random().enumValueWithIndex(OpType, u32);

        if (op_idx == 0) {
            op_out[0] = pcg.random().uintLessThan(u32, tensor_num);

            op_in[0] = pcg.random().uintLessThan(u32, tensor_num - 1);
            op_in[0] = if (op_in[0] < op_out[0]) op_in[0] else op_in[0] + 1;
            assert(op_out[0] != op_in[0]);
        } else {
            const switch_likelyhood: u32 = 10;
            if (pcg.random().uintLessThan(u32, switch_likelyhood) == 0) {
                op_in[op_idx] = op_out[op_idx - 1];
                op_out[op_idx] = pcg.random().uintLessThan(u32, tensor_num - 1);
                // I think this should get a guaranteed random number different than tensor_in without biasing the result
                if (op_out[op_idx] >= op_in[op_idx]) {
                    op_out[op_idx] += 1;
                }
                assert(op_out[op_idx] != op_in[op_idx]);
            } else {
                op_out[op_idx] = op_out[op_idx - 1];
                op_in[op_idx] = op_in[op_idx - 1];
            }
        }
    }

    // $TODO I should just precompute the op amounts so there are way less allocations
    // $TODO Also make it randomize the out buffer and if it is random then assert that no actual ops are computed
    for (0..tensor_num) |tensor_idx| {
        if (tensor_idx == op_out[op_num - 1]) {
            tensor[tensor_idx] = try Tensor.alloc(runtime, allocator, a_size_max, z_size_max, //
                y_size_max, x_size_max, 1);
        } else {
            if (pcg.random().boolean()) {
                tensor[tensor_idx] = try Tensor.alloc(runtime, allocator, a_size_max, z_size_max, //
                    y_size_max, x_size_max, 1);
            } else {
                tensor[tensor_idx] = try Tensor.allocIntermediary(runtime, allocator, a_size_max, //
                    z_size_max, y_size_max, x_size_max, 1);
            }
        }
    }

    for (0..tensor_num) |tensor_idx| {
        for (0..a_size_max * z_size_max * y_size_max * x_size_max) |arg_idx| {
            tensor[tensor_idx].buffer.values[arg_idx] = pcg.random().floatNorm(f32);
        }
    }

    var op_idx_used: u32 = 0;
    var op_idx: u32 = 0;
    while (op_idx < op_num) : (op_idx += 1) {
        if (op_idx < op_idx_used) {
            continue;
        }

        const a_size: u32 = pcg.random().uintLessThan(u32, a_size_max) + 1;
        const z_size: u32 = pcg.random().uintLessThan(u32, z_size_max) + 1;
        const y_size: u32 = pcg.random().uintLessThan(u32, y_size_max) + 1;
        const x_size: u32 = pcg.random().uintLessThan(u32, x_size_max) + 1;
        const a_off: u32 = if (a_size_max > a_size) pcg.random().uintLessThan(u32, a_size_max - a_size) else 0;
        const z_off: u32 = if (z_size_max > z_size) pcg.random().uintLessThan(u32, z_size_max - z_size) else 0;
        const y_off: u32 = if (y_size_max > y_size) pcg.random().uintLessThan(u32, y_size_max - y_size) else 0;
        const x_off: u32 = if (x_size_max > x_size) pcg.random().uintLessThan(u32, x_size_max - x_size) else 0;
        const a_loop: u32 = (if (a_size + a_off == a_size_max) 0 else pcg.random().uintLessThan(u32, a_size_max - (a_size + a_off))) + 1;
        const z_loop: u32 = (if (z_size + z_off == z_size_max) 0 else pcg.random().uintLessThan(u32, z_size_max - (z_size + z_off))) + 1;
        const y_loop: u32 = (if (y_size + y_off == y_size_max) 0 else pcg.random().uintLessThan(u32, y_size_max - (y_size + y_off))) + 1;
        const x_loop: u32 = (if (x_size + x_off == x_size_max) 0 else pcg.random().uintLessThan(u32, x_size_max - (x_size + x_off))) + 1;

        // Putting this out here to make snycing the prng state trivial
        const u_var: f32 = pcg.random().floatNorm(f32);

        // $FIXME This should never be 0
        const loop_len: u32 = pcg.random().uintLessThan(u32, @truncate(op_num - op_idx));
        op_idx_used = op_idx + loop_len;

        var a_idx: u32 = 0;
        while (a_idx < a_loop) : (a_idx += 1) {
            var z_idx: u32 = 0;
            while (z_idx < z_loop) : (z_idx += 1) {
                var y_idx: u32 = 0;
                while (y_idx < y_loop) : (y_idx += 1) {
                    var x_idx: u32 = 0;
                    while (x_idx < x_loop) : (x_idx += 1) {
                        var loop_idx: u32 = 0;
                        while (loop_idx < loop_len) : (loop_idx += 1) {
                            if (!op_included[op_idx + loop_idx]) {
                                continue;
                            }

                            const tensor_out: u32 = op_out[op_idx + loop_idx];
                            const tensor_in: u32 = op_in[op_idx + loop_idx];

                            try tensor[tensor_out].linearized.capacityEnsure(allocator, 4 * (loop_len * a_loop * z_loop * y_loop * x_loop) +
                                tensor[tensor_in].linearized.op_num);
                            try tensor[tensor_in].linearized.capacityEnsure(allocator, 2);

                            if (op_type[op_idx + loop_idx].isReduce()) {
                                tensor[tensor_out].moveResize(1, 1, 1, 1);
                            } else {
                                tensor[tensor_out].moveResize(a_size, z_size, y_size, x_size);
                            }
                            if (op_type[op_idx + loop_idx].isExpand()) {
                                tensor[tensor_in].moveResize(1, 1, 1, 1);
                            } else {
                                tensor[tensor_in].moveResize(a_size, z_size, y_size, x_size);
                            }

                            tensor[tensor_out].moveOffset(a_off + a_idx, z_off + z_idx, y_off + y_idx, x_off + x_idx);
                            tensor[tensor_in].moveOffset(a_off + a_idx, z_off + z_idx, y_off + y_idx, x_off + x_idx);

                            switch (op_type[op_idx + loop_idx]) {
                                .unary_add => {
                                    tensor[tensor_out].unaryAdd(u_var);
                                },
                                .unary_subtract => {
                                    tensor[tensor_out].unarySubtract(u_var);
                                },
                                .unary_multiply => {
                                    tensor[tensor_out].unaryMultiply(u_var);
                                },
                                .unary_divide => {
                                    tensor[tensor_out].unaryDivide(@abs(u_var) + 1);
                                },
                                .unary_exp => {
                                    // NaN prevention
                                    tensor[tensor_out].unaryMax(10);
                                    tensor[tensor_out].unaryMin(-10);

                                    tensor[tensor_out].unaryExp();
                                },
                                .unary_log => {
                                    // NaN prevention
                                    tensor[tensor_out].unaryAbsolute();
                                    tensor[tensor_out].unaryAdd(1);

                                    tensor[tensor_out].unaryLog();
                                },
                                .unary_square => {
                                    // Inf prevention
                                    tensor[tensor_out].unaryMax(100);
                                    tensor[tensor_out].unaryMin(-100);

                                    tensor[tensor_out].unarySquare();
                                },
                                .unary_sqrt => {
                                    // NaN prevention
                                    tensor[tensor_out].unaryAbsolute();

                                    tensor[tensor_out].unarySqrt();
                                },
                                .unary_reciprocal => {
                                    // NaN prevention
                                    tensor[tensor_out].unaryAbsolute();
                                    tensor[tensor_out].unaryAdd(1);

                                    tensor[tensor_out].unaryReciprocal();
                                },
                                .unary_max => {
                                    tensor[tensor_out].unaryMax(u_var);
                                },
                                .unary_min => {
                                    tensor[tensor_out].unaryMin(u_var);
                                },
                                .unary_set => {
                                    tensor[tensor_out].unarySet(u_var);
                                },
                                .unary_random => {
                                    // $TODO This
                                    tensor[tensor_out].unarySet(u_var);
                                },
                                .unary_tanh => {
                                    tensor[tensor_out].unaryTanh();
                                },
                                .unary_absolute => {
                                    tensor[tensor_out].unaryAbsolute();
                                },
                                .unary_sign => {
                                    tensor[tensor_out].unaryAbsolute();
                                    // $TODO Reenable this when this is implemented
                                    // tensor1[tensor_out].unarySign();
                                },
                                .binary_add => {
                                    tensor[tensor_out].binaryAdd(&tensor[tensor_in]);
                                },
                                .binary_subtract => {
                                    tensor[tensor_out].binarySubtract(&tensor[tensor_in]);
                                },
                                .binary_multiply => {
                                    tensor[tensor_out].binaryMultiply(&tensor[tensor_in]);
                                },
                                .binary_divide => {
                                    // NaN prevention
                                    tensor[tensor_in].unaryAbsolute();
                                    tensor[tensor_in].unaryAdd(1);

                                    tensor[tensor_out].binaryDivide(&tensor[tensor_in]);
                                },
                                .binary_max => {
                                    tensor[tensor_out].binaryMax(&tensor[tensor_in]);
                                },
                                .binary_min => {
                                    tensor[tensor_out].binaryMin(&tensor[tensor_in]);
                                },
                                .binary_set => {
                                    tensor[tensor_out].binarySet(&tensor[tensor_in]);
                                },
                                .expand_add => {
                                    tensor[tensor_out].expandAdd(&tensor[tensor_in]);
                                },
                                .expand_subtract => {
                                    tensor[tensor_out].expandSubtract(&tensor[tensor_in]);
                                },
                                .expand_multiply => {
                                    tensor[tensor_out].expandMultiply(&tensor[tensor_in]);
                                },
                                .expand_divide => {
                                    // NaN prevention
                                    tensor[tensor_in].unaryAbsolute();
                                    tensor[tensor_in].unaryAdd(1);

                                    tensor[tensor_out].expandDivide(&tensor[tensor_in]);
                                },
                                .expand_max => {
                                    tensor[tensor_out].expandMax(&tensor[tensor_in]);
                                },
                                .expand_min => {
                                    tensor[tensor_out].expandMin(&tensor[tensor_in]);
                                },
                                .expand_set => {
                                    tensor[tensor_out].expandSet(&tensor[tensor_in]);
                                },
                                .reduce_sum => {
                                    tensor[tensor_out].reduceSum(&tensor[tensor_in]);
                                },
                                .reduce_max => {
                                    tensor[tensor_out].reduceMax(&tensor[tensor_in]);
                                },
                                .reduce_avg => {
                                    tensor[tensor_out].reduceAvg(&tensor[tensor_in]);
                                },
                                .reduce_min => {
                                    tensor[tensor_out].reduceMin(&tensor[tensor_in]);
                                },
                            }
                        }
                    }
                }
            }
        }
    }

    return .{ .tensor = tensor, .out_idx = op_out[op_idx_used] };
}
