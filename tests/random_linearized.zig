// Helper file that implements a function that generates randomized linearized ops.
// This is extracted out because it makes it easier to keep consistent prng state for the same seed in every test
const std = @import("std");
const Pcg = std.Random.Pcg;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const grads = @import("grads");
const Buffer = grads.Buffer;
const Linearized = grads.Linearized;
const OpKind = grads.Op.Kind;
const Runtime = grads.Runtime;

pub const a_size_max: u32 = 7;
pub const z_size_max: u32 = 6;
pub const y_size_max: u32 = 5;
pub const x_size_max: u32 = 4;
pub const buffer_num: u32 = 10;
pub const op_num: u32 = 40;
comptime {
    assert(buffer_num > 1);
    assert(op_num > 0);
}
// $TODO Have chances of changing offsets, sizes, u_var and everything else

pub fn randomLinearized(runtime: Runtime, arena: Allocator, op_included: [op_num]bool, rng: u64) !struct {
    buffer: [buffer_num]Buffer,
    linearized: Linearized,
    out_idx: u32,
} {
    var buffer: [buffer_num]Buffer = undefined;
    var linearized: Linearized = try .alloc(arena, 3 * op_num * a_size_max * z_size_max * y_size_max * x_size_max);

    var pcg = Pcg.init(rng);

    var op_kind: [op_num]OpKind = undefined;
    var op_out: [op_num]u32 = undefined;
    var op_in: [op_num]u32 = undefined;

    for (0..op_num) |op_idx| {
        op_kind[op_idx] = pcg.random().enumValueWithIndex(OpKind, u32);

        if (op_idx == 0) {
            op_out[0] = pcg.random().uintLessThan(u32, buffer_num);

            op_in[0] = pcg.random().uintLessThan(u32, buffer_num - 1);
            op_in[0] = if (op_in[0] < op_out[0]) op_in[0] else op_in[0] + 1;
            assert(op_out[0] != op_in[0]);
        } else {
            const switch_likelyhood: u32 = 10;
            if (pcg.random().uintLessThan(u32, switch_likelyhood) == 0) {
                op_in[op_idx] = op_out[op_idx - 1];
                op_out[op_idx] = pcg.random().uintLessThan(u32, buffer_num - 1);
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
    // $FIXME If there is an error here this is memory leak city
    for (0..buffer_num) |buffer_idx| {
        if (buffer_idx == op_out[op_num - 1]) {
            buffer[buffer_idx] = try Buffer.alloc(runtime, arena, a_size_max, z_size_max, //
                y_size_max, x_size_max, .normal);
        } else {
            if (pcg.random().boolean()) {
                buffer[buffer_idx] = try Buffer.alloc(runtime, arena, a_size_max, z_size_max, //
                    y_size_max, x_size_max, .normal);
            } else {
                buffer[buffer_idx] = try Buffer.alloc(runtime, arena, a_size_max, //
                    z_size_max, y_size_max, x_size_max, .intermediary);
            }
        }
    }

    for (0..buffer_num) |buffer_idx| {
        for (0..a_size_max * z_size_max * y_size_max * x_size_max) |arg_idx| {
            buffer[buffer_idx].values[arg_idx] = pcg.random().floatNorm(f32);
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

        const loop_len: u32 = if (op_num == op_idx + 1) 1 else pcg.random().uintLessThan(u32, @truncate(op_num - op_idx - 1)) + 1;
        assert(loop_len != 0);
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

                            const buffer_out: u32 = op_out[op_idx + loop_idx];
                            const buffer_in: u32 = op_in[op_idx + loop_idx];

                            if (op_kind[op_idx + loop_idx].isReduce()) {
                                buffer[buffer_out].moveResize(1, 1, 1, 1);
                            } else {
                                buffer[buffer_out].moveResize(a_size, z_size, y_size, x_size);
                            }
                            if (op_kind[op_idx + loop_idx].isExpand()) {
                                buffer[buffer_in].moveResize(1, 1, 1, 1);
                            } else {
                                buffer[buffer_in].moveResize(a_size, z_size, y_size, x_size);
                            }

                            buffer[buffer_out].moveOffset(a_off + a_idx, z_off + z_idx, y_off + y_idx, x_off + x_idx);
                            buffer[buffer_in].moveOffset(a_off + a_idx, z_off + z_idx, y_off + y_idx, x_off + x_idx);

                            switch (op_kind[op_idx + loop_idx]) {
                                .unary_add => {
                                    linearized.unaryAdd(buffer[buffer_out], u_var);
                                },
                                .unary_subtract => {
                                    linearized.unarySubtract(buffer[buffer_out], u_var);
                                },
                                .unary_multiply => {
                                    linearized.unaryMultiply(buffer[buffer_out], u_var);
                                },
                                .unary_divide => {
                                    linearized.unaryDivide(buffer[buffer_out], @abs(u_var) + 1);
                                },
                                .unary_exp => {
                                    // NaN prevention
                                    linearized.unaryMax(buffer[buffer_out], 10);
                                    linearized.unaryMin(buffer[buffer_out], -10);

                                    linearized.unaryExp(buffer[buffer_out]);
                                },
                                .unary_log => {
                                    // NaN prevention
                                    linearized.unaryAbsolute(buffer[buffer_out]);
                                    linearized.unaryAdd(buffer[buffer_out], 1);

                                    linearized.unaryLog(buffer[buffer_out]);
                                },
                                .unary_square => {
                                    // Inf prevention
                                    linearized.unaryMax(buffer[buffer_out], 100);
                                    linearized.unaryMin(buffer[buffer_out], -100);

                                    linearized.unarySquare(buffer[buffer_out]);
                                },
                                .unary_sqrt => {
                                    // NaN prevention
                                    linearized.unaryAbsolute(buffer[buffer_out]);

                                    linearized.unarySqrt(buffer[buffer_out]);
                                },
                                .unary_reciprocal => {
                                    // NaN prevention
                                    linearized.unaryAbsolute(buffer[buffer_out]);
                                    linearized.unaryAdd(buffer[buffer_out], 1);

                                    linearized.unaryReciprocal(buffer[buffer_out]);
                                },
                                .unary_max => {
                                    linearized.unaryMax(buffer[buffer_out], u_var);
                                },
                                .unary_min => {
                                    linearized.unaryMin(buffer[buffer_out], u_var);
                                },
                                .unary_set => {
                                    linearized.unarySet(buffer[buffer_out], u_var);
                                },
                                .unary_random => {
                                    // $TODO This
                                    linearized.unarySet(buffer[buffer_out], u_var);
                                },
                                .unary_tanh => {
                                    linearized.unaryTanh(buffer[buffer_out]);
                                },
                                .unary_absolute => {
                                    linearized.unaryAbsolute(buffer[buffer_out]);
                                },
                                .unary_sign => {
                                    linearized.unaryAbsolute(buffer[buffer_out]);
                                    // $TODO Reenable this when this is implemented
                                    // linearized.unarySign(buffer[buffer_out]);
                                },
                                .binary_add => {
                                    linearized.binaryAdd(buffer[buffer_out], buffer[buffer_in]);
                                },
                                .binary_subtract => {
                                    linearized.binarySubtract(buffer[buffer_out], buffer[buffer_in]);
                                },
                                .binary_multiply => {
                                    linearized.binaryMultiply(buffer[buffer_out], buffer[buffer_in]);
                                },
                                .binary_divide => {
                                    // NaN prevention
                                    linearized.unaryAbsolute(buffer[buffer_in]);
                                    linearized.unaryAdd(buffer[buffer_in], 1);

                                    linearized.binaryDivide(buffer[buffer_out], buffer[buffer_in]);
                                },
                                .binary_max => {
                                    linearized.binaryMax(buffer[buffer_out], buffer[buffer_in]);
                                },
                                .binary_min => {
                                    linearized.binaryMin(buffer[buffer_out], buffer[buffer_in]);
                                },
                                .binary_set => {
                                    linearized.binarySet(buffer[buffer_out], buffer[buffer_in]);
                                },
                                .expand_add => {
                                    linearized.expandAdd(buffer[buffer_out], buffer[buffer_in]);
                                },
                                .expand_subtract => {
                                    linearized.expandSubtract(buffer[buffer_out], buffer[buffer_in]);
                                },
                                .expand_multiply => {
                                    linearized.expandMultiply(buffer[buffer_out], buffer[buffer_in]);
                                },
                                .expand_divide => {
                                    // NaN prevention
                                    linearized.unaryAbsolute(buffer[buffer_in]);
                                    linearized.unaryAdd(buffer[buffer_in], 1);

                                    linearized.expandDivide(buffer[buffer_out], buffer[buffer_in]);
                                },
                                .expand_max => {
                                    linearized.expandMax(buffer[buffer_out], buffer[buffer_in]);
                                },
                                .expand_min => {
                                    linearized.expandMin(buffer[buffer_out], buffer[buffer_in]);
                                },
                                .expand_set => {
                                    linearized.expandSet(buffer[buffer_out], buffer[buffer_in]);
                                },
                                .reduce_sum => {
                                    linearized.reduceSum(buffer[buffer_out], buffer[buffer_in]);
                                },
                                .reduce_max => {
                                    linearized.reduceMax(buffer[buffer_out], buffer[buffer_in]);
                                },
                                .reduce_avg => {
                                    linearized.reduceAvg(buffer[buffer_out], buffer[buffer_in]);
                                },
                                .reduce_min => {
                                    linearized.reduceMin(buffer[buffer_out], buffer[buffer_in]);
                                },
                            }
                        }
                    }
                }
            }
        }
    }

    var out_idx: u32 = 0; // Should always be a safe return value even if no op is included
    var search_idx: i32 = @intCast(op_num - 1);
    while (search_idx >= 0) : (search_idx -= 1) {
        if (op_included[@intCast(search_idx)]) {
            out_idx = op_out[@intCast(search_idx)];
            break;
        }
    }
    assert(out_idx < op_num);

    return .{
        .buffer = buffer,
        .linearized = linearized,
        .out_idx = out_idx,
    };
}
