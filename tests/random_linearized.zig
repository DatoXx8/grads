// Helper file that implements a function that generates randomized linearized ops.
// This is extracted out because it makes it easier to keep consistent prng state for the same seed in every test
const std = @import("std");
const Pcg = std.Random.Pcg;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const grads = @import("grads");
const Buffer = grads.Buffer;
const Vec4 = Buffer.Vec4;
const Linearized = grads.Linearized;
const OpKind = grads.Op.Kind;
const Runtime = grads.Runtime;

pub const size: Vec4 = .{
    .a = 7,
    .z = 6,
    .y = 5,
    .x = 4,
};
pub const buffer_num: u32 = 10;
pub const op_num: u32 = 40;
comptime {
    assert(buffer_num > 1);
    assert(op_num > 0);
}
// $TODO Have chances of changing offsets, sizes, u_var and everything else

pub fn randomLinearized(runtime: Runtime, gpa: Allocator, arena: Allocator, op_included: [op_num]bool, rng: u64) !struct {
    buffer: [buffer_num]Buffer,
    linearized: Linearized,
    out_idx: u32,
} {
    var buffer: [buffer_num]Buffer = undefined;
    var linearized: Linearized = try .alloc(arena, 3 * op_num * size.productOfElements());

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
            buffer[buffer_idx] = try Buffer.alloc(runtime, gpa, arena, size, .normal);
        } else {
            if (pcg.random().boolean()) {
                buffer[buffer_idx] = try Buffer.alloc(runtime, gpa, arena, size, .normal);
            } else {
                buffer[buffer_idx] = try Buffer.alloc(runtime, gpa, arena, size, .intermediary);
            }
        }
    }

    for (0..buffer_num) |buffer_idx| {
        for (0..size.productOfElements()) |arg_idx| {
            buffer[buffer_idx].data().*.values[arg_idx] = pcg.random().floatNorm(f32);
        }
    }

    var op_idx_used: u32 = 0;
    var op_idx: u32 = 0;
    while (op_idx < op_num) : (op_idx += 1) {
        if (op_idx < op_idx_used) {
            continue;
        }

        // $TODO This should all be Vec4 as well
        const a_size: u32 = pcg.random().uintLessThan(u32, size.a) + 1;
        const z_size: u32 = pcg.random().uintLessThan(u32, size.z) + 1;
        const y_size: u32 = pcg.random().uintLessThan(u32, size.y) + 1;
        const x_size: u32 = pcg.random().uintLessThan(u32, size.x) + 1;
        const a_off: u32 = if (size.a > a_size) pcg.random().uintLessThan(u32, size.a - a_size) else 0;
        const z_off: u32 = if (size.z > z_size) pcg.random().uintLessThan(u32, size.z - z_size) else 0;
        const y_off: u32 = if (size.y > y_size) pcg.random().uintLessThan(u32, size.y - y_size) else 0;
        const x_off: u32 = if (size.x > x_size) pcg.random().uintLessThan(u32, size.x - x_size) else 0;
        const a_loop: u32 = (if (a_size + a_off == size.a) 0 else pcg.random().uintLessThan(u32, size.a - (a_size + a_off))) + 1;
        const z_loop: u32 = (if (z_size + z_off == size.z) 0 else pcg.random().uintLessThan(u32, size.z - (z_size + z_off))) + 1;
        const y_loop: u32 = (if (y_size + y_off == size.y) 0 else pcg.random().uintLessThan(u32, size.y - (y_size + y_off))) + 1;
        const x_loop: u32 = (if (x_size + x_off == size.x) 0 else pcg.random().uintLessThan(u32, size.x - (x_size + x_off))) + 1;

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

                            const buffer_out_idx: u32 = op_out[op_idx + loop_idx];
                            const buffer_in_idx: u32 = op_in[op_idx + loop_idx];

                            const buffer_out: Buffer = buffer[buffer_out_idx];
                            const buffer_in: Buffer = buffer[buffer_in_idx];

                            if (op_kind[op_idx + loop_idx].isReduce()) {
                                buffer_out.moveResize(.{ .a = 1, .z = 1, .y = 1, .x = 1 });
                            } else {
                                buffer_out.moveResize(.{ .a = a_size, .z = z_size, .y = y_size, .x = x_size });
                            }
                            if (op_kind[op_idx + loop_idx].isExpand()) {
                                buffer_in.moveResize(.{ .a = 1, .z = 1, .y = 1, .x = 1 });
                            } else {
                                buffer_in.moveResize(.{ .a = a_size, .z = z_size, .y = y_size, .x = x_size });
                            }

                            buffer_out.moveOffset(.{ .a = a_off + a_idx, .z = z_off + z_idx, .y = y_off + y_idx, .x = x_off + x_idx });
                            buffer_in.moveOffset(.{ .a = a_off + a_idx, .z = z_off + z_idx, .y = y_off + y_idx, .x = x_off + x_idx });

                            switch (op_kind[op_idx + loop_idx]) {
                                .unary_add => {
                                    linearized.unaryAdd(buffer_out, u_var);
                                },
                                .unary_subtract => {
                                    linearized.unarySubtract(buffer_out, u_var);
                                },
                                .unary_multiply => {
                                    linearized.unaryMultiply(buffer_out, u_var);
                                },
                                .unary_divide => {
                                    linearized.unaryDivide(buffer_out, @abs(u_var) + 1);
                                },
                                .unary_exp => {
                                    // NaN prevention
                                    linearized.unaryMax(buffer_out, 10);
                                    linearized.unaryMin(buffer_out, -10);

                                    linearized.unaryExp(buffer_out);
                                },
                                .unary_log => {
                                    // NaN prevention
                                    linearized.unaryAbsolute(buffer_out);
                                    linearized.unaryAdd(buffer_out, 1);

                                    linearized.unaryLog(buffer_out);
                                },
                                .unary_square => {
                                    // Inf prevention
                                    linearized.unaryMax(buffer_out, 100);
                                    linearized.unaryMin(buffer_out, -100);

                                    linearized.unarySquare(buffer_out);
                                },
                                .unary_sqrt => {
                                    // NaN prevention
                                    linearized.unaryAbsolute(buffer_out);

                                    linearized.unarySqrt(buffer_out);
                                },
                                .unary_reciprocal => {
                                    // NaN prevention
                                    linearized.unaryAbsolute(buffer_out);
                                    linearized.unaryAdd(buffer_out, 1);

                                    linearized.unaryReciprocal(buffer_out);
                                },
                                .unary_max => {
                                    linearized.unaryMax(buffer_out, u_var);
                                },
                                .unary_min => {
                                    linearized.unaryMin(buffer_out, u_var);
                                },
                                .unary_set => {
                                    linearized.unarySet(buffer_out, u_var);
                                },
                                .unary_random => {
                                    // $TODO This
                                    linearized.unarySet(buffer_out, u_var);
                                },
                                .unary_tanh => {
                                    linearized.unaryTanh(buffer_out);
                                },
                                .unary_absolute => {
                                    linearized.unaryAbsolute(buffer_out);
                                },
                                .unary_sign => {
                                    linearized.unaryAbsolute(buffer_out);
                                    // $TODO Reenable this when this is implemented
                                    // linearized.unarySign(buffer_out);
                                },
                                .binary_add => {
                                    linearized.binaryAdd(buffer_out, buffer_in);
                                },
                                .binary_subtract => {
                                    linearized.binarySubtract(buffer_out, buffer_in);
                                },
                                .binary_multiply => {
                                    linearized.binaryMultiply(buffer_out, buffer_in);
                                },
                                .binary_divide => {
                                    // NaN prevention
                                    linearized.unaryAbsolute(buffer_in);
                                    linearized.unaryAdd(buffer_in, 1);

                                    linearized.binaryDivide(buffer_out, buffer_in);
                                },
                                .binary_max => {
                                    linearized.binaryMax(buffer_out, buffer_in);
                                },
                                .binary_min => {
                                    linearized.binaryMin(buffer_out, buffer_in);
                                },
                                .binary_set => {
                                    linearized.binarySet(buffer_out, buffer_in);
                                },
                                .expand_add => {
                                    linearized.expandAdd(buffer_out, buffer_in);
                                },
                                .expand_subtract => {
                                    linearized.expandSubtract(buffer_out, buffer_in);
                                },
                                .expand_multiply => {
                                    linearized.expandMultiply(buffer_out, buffer_in);
                                },
                                .expand_divide => {
                                    // NaN prevention
                                    linearized.unaryAbsolute(buffer_in);
                                    linearized.unaryAdd(buffer_in, 1);

                                    linearized.expandDivide(buffer_out, buffer_in);
                                },
                                .expand_max => {
                                    linearized.expandMax(buffer_out, buffer_in);
                                },
                                .expand_min => {
                                    linearized.expandMin(buffer_out, buffer_in);
                                },
                                .expand_set => {
                                    linearized.expandSet(buffer_out, buffer_in);
                                },
                                .reduce_sum => {
                                    linearized.reduceSum(buffer_out, buffer_in);
                                },
                                .reduce_max => {
                                    linearized.reduceMax(buffer_out, buffer_in);
                                },
                                .reduce_avg => {
                                    linearized.reduceAvg(buffer_out, buffer_in);
                                },
                                .reduce_min => {
                                    linearized.reduceMin(buffer_out, buffer_in);
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
