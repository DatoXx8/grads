const std = @import("std");
const grads = @import("grads");

const Tensor = grads.Tensor;
const OpType = grads.Op.Type;
const pcg = grads.pcg;
const Program = grads.Program;
const ClContext = grads.ClContext;
const ClDevice = grads.ClDevice;
const ClCommandQueue = grads.ClCommandQueue;

const assert = std.debug.assert;

const AssertError = error{
    nan,
    inf,
    difference,
};
/// Margin of error
const epsilon: f32 = 1e-6;
const epsilon_relative: f32 = 1e-4;
/// Check for equality between the two floats within the margin of error of `epsilon`
fn assertEq(val1: f32, val2: f32) !void {
    if (std.math.isNan(val1) or std.math.isNan(val2)) {
        // For nicer output formatting
        std.debug.print("\n", .{});
        std.log.err("Found NaN in equality comparison.\n", .{});
        return AssertError.nan;
    } else if (std.math.isInf(val1) or std.math.isInf(val2)) {
        std.debug.print("\n", .{});
        std.log.err("Found Inf in equality comparison.\n", .{});
        return AssertError.nan;
    } else if (std.math.approxEqAbs(f32, val1, val2, epsilon) or std.math.approxEqRel(f32, val1, val2, epsilon_relative)) {
        return;
    } else {
        // For nicer output formatting
        std.debug.print("\n", .{});
        std.log.err("Difference between {d} and {d} is too large.\n", .{ val1, val2 });
        return AssertError.difference;
    }
}
const tensor_num: u32 = 10;
const op_num: u32 = 10;
const iterations: u32 = 1000;
comptime {
    assert(tensor_num > 1);
    assert(op_num > 0);
    assert(iterations > 0);
}

/// Computes and prints mean and variance
fn analyseTimes(ns_times: [iterations]i128, name: []const u8) void {
    var ns_sum: i128 = 0;
    for (0..iterations) |iteration_idx| {
        ns_sum += ns_times[iteration_idx];
    }
    const ns_mean: i128 = @divFloor(ns_sum, iterations);
    var ns_square_sum: i128 = 0;
    for (0..iterations) |iteration_idx| {
        const ns_difference: i128 = ns_times[iteration_idx] - ns_mean;
        ns_square_sum += ns_difference * ns_difference;
    }
    const ns_variance: u128 = std.math.sqrt(@as(u128, @intCast(@divFloor(ns_square_sum, iterations))));

    if (ns_mean < 1_000) {
        std.debug.print("Time: {d:8.4}ns +- {d:8.4}ns", .{ ns_mean, ns_variance });
    } else if (ns_mean < 1_000_000) {
        const us_mean: f64 = @as(f64, @floatFromInt(ns_mean)) / 1_000;
        const us_variance: f64 = @as(f64, @floatFromInt(ns_variance)) / 1_000;
        std.debug.print("Time: {d:8.4}us +- {d:8.4}us", .{ us_mean, us_variance });
    } else if (ns_mean < 1_000_000_000) {
        const ms_mean: f64 = @as(f64, @floatFromInt(ns_mean)) / 1_000_000;
        const ms_variance: f64 = @as(f64, @floatFromInt(ns_variance)) / 1_000_000;
        std.debug.print("Time: {d:8.4}ms +- {d:8.4}ms", .{ ms_mean, ms_variance });
    } else {
        const s_mean: f64 = @as(f64, @floatFromInt(ns_mean)) / 1_000_000_000;
        const s_variance: f64 = @as(f64, @floatFromInt(ns_variance)) / 1_000_000_000;
        std.debug.print("Time: {d:8.4} s +- {d:8.4} s", .{ s_mean, s_variance });
    }
    std.debug.print(" {s}\n", .{name});
}

// If this fails then you can use the rng seed from here in the simulator and get then same ops (At least if I don't break it in the future)
fn profileCompiler(allocator: anytype, rng: u64, device: ClDevice, context: ClContext, queue: ClCommandQueue) !void {
    assert(tensor_num > 1);
    assert(op_num > 0);

    var tensor1: [tensor_num]Tensor = undefined;
    var tensor2: [tensor_num]Tensor = undefined;

    const a_size_max: u32 = 7;
    const z_size_max: u32 = 6;
    const y_size_max: u32 = 5;
    const x_size_max: u32 = 4;

    for (0..tensor_num) |tensor_idx| {
        tensor1[tensor_idx] = try Tensor.alloc(allocator, a_size_max, z_size_max, y_size_max, x_size_max, context);
        tensor2[tensor_idx] = try Tensor.alloc(allocator, a_size_max, z_size_max, y_size_max, x_size_max, context);
    }
    defer {
        for (0..tensor_num) |tensor_idx| {
            tensor1[tensor_idx].free(allocator);
            tensor2[tensor_idx].free(allocator);
        }
    }

    pcg.init(rng);
    std.debug.print("profile-compiler: rng={}...\n", .{rng});

    for (0..tensor_num) |tensor_idx| {
        for (0..a_size_max * z_size_max * y_size_max * x_size_max) |arg_idx| {
            tensor1[tensor_idx].buffer.values[arg_idx] = pcg.randF32();
            tensor2[tensor_idx].buffer.values[arg_idx] = tensor1[tensor_idx].buffer.values[arg_idx];
        }
    }

    const op_type_max: u32 = @typeInfo(OpType).Enum.fields.len;
    var op_type: [op_num]OpType = undefined;
    var op_out: [op_num]u32 = undefined;
    var op_in: [op_num]u32 = undefined;

    for (0..op_num) |op_idx| {
        op_type[op_idx] = @enumFromInt(pcg.randBelow(op_type_max));

        if (op_idx == 0) {
            op_out[0] = pcg.randBelow(tensor_num);

            op_in[0] = pcg.randBelow(tensor_num - 1);
            op_in[0] = if (op_in[0] < op_out[0]) op_in[0] else op_in[0] + 1;
            assert(op_out[0] != op_in[0]);
        } else {
            const switch_likelyhood: u32 = 10;
            if (pcg.randBelow(switch_likelyhood) == 0) {
                op_in[op_idx] = op_out[op_idx - 1];
                op_out[op_idx] = pcg.randBelow(tensor_num - 1);
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

    // TODO: Come up with a better name. This is basically the first op that isn't in the last loop
    var op_idx_free: u32 = 0;
    for (0..op_num) |op_idx| {
        if (op_idx < op_idx_free) {
            continue;
        }

        const a_size: u32 = pcg.randBelow(a_size_max) + 1;
        const z_size: u32 = pcg.randBelow(z_size_max) + 1;
        const y_size: u32 = pcg.randBelow(y_size_max) + 1;
        const x_size: u32 = pcg.randBelow(x_size_max) + 1;
        const a_off: u32 = pcg.randBelow(a_size_max - a_size);
        const z_off: u32 = pcg.randBelow(z_size_max - z_size);
        const y_off: u32 = pcg.randBelow(y_size_max - y_size);
        const x_off: u32 = pcg.randBelow(x_size_max - x_size);
        const a_loop: u32 = pcg.randBelow(a_size_max - (a_size + a_off)) + 1;
        const z_loop: u32 = pcg.randBelow(z_size_max - (z_size + z_off)) + 1;
        const y_loop: u32 = pcg.randBelow(y_size_max - (y_size + y_off)) + 1;
        const x_loop: u32 = pcg.randBelow(x_size_max - (x_size + x_off)) + 1;

        const u_var: f32 = pcg.randF32();

        const loop_len: u32 = pcg.randBelow(@truncate(op_num - op_idx));
        op_idx_free = @truncate(op_idx + loop_len);

        for (0..a_loop) |a_idx_usize| {
            for (0..z_loop) |z_idx_usize| {
                for (0..y_loop) |y_idx_usize| {
                    for (0..x_loop) |x_idx_usize| {
                        for (0..loop_len) |loop_idx| {
                            const a_idx: u32 = @truncate(a_idx_usize);
                            const z_idx: u32 = @truncate(z_idx_usize);
                            const y_idx: u32 = @truncate(y_idx_usize);
                            const x_idx: u32 = @truncate(x_idx_usize);

                            const tensor_out: u32 = op_out[op_idx + loop_idx];
                            const tensor_in: u32 = op_in[op_idx + loop_idx];

                            if (op_type[op_idx + loop_idx].isReduce()) {
                                tensor1[tensor_out].moveResize(1, 1, 1, 1);
                                tensor2[tensor_out].moveResize(1, 1, 1, 1);
                            } else {
                                tensor1[tensor_out].moveResize(a_size, z_size, y_size, x_size);
                                tensor2[tensor_out].moveResize(a_size, z_size, y_size, x_size);
                            }
                            if (op_type[op_idx + loop_idx].isLinary()) {
                                tensor1[tensor_in].moveResize(1, 1, 1, 1);
                                tensor2[tensor_in].moveResize(1, 1, 1, 1);
                            } else {
                                tensor1[tensor_in].moveResize(a_size, z_size, y_size, x_size);
                                tensor2[tensor_in].moveResize(a_size, z_size, y_size, x_size);
                            }

                            tensor1[tensor_out].moveOffset(a_off + a_idx, z_off + z_idx, y_off + y_idx, x_off + x_idx);
                            tensor2[tensor_out].moveOffset(a_off + a_idx, z_off + z_idx, y_off + y_idx, x_off + x_idx);
                            tensor1[tensor_in].moveOffset(a_off + a_idx, z_off + z_idx, y_off + y_idx, x_off + x_idx);
                            tensor2[tensor_in].moveOffset(a_off + a_idx, z_off + z_idx, y_off + y_idx, x_off + x_idx);

                            switch (op_type[op_idx + loop_idx]) {
                                .unary_add => {
                                    try tensor1[tensor_out].unaryAdd(allocator, u_var);
                                    try tensor2[tensor_out].unaryAdd(allocator, u_var);
                                },
                                .unary_subtract => {
                                    try tensor1[tensor_out].unarySubtract(allocator, u_var);
                                    try tensor2[tensor_out].unarySubtract(allocator, u_var);
                                },
                                .unary_multiply => {
                                    try tensor1[tensor_out].unaryMultiply(allocator, u_var);
                                    try tensor2[tensor_out].unaryMultiply(allocator, u_var);
                                },
                                .unary_divide => {
                                    try tensor1[tensor_out].unaryDivide(allocator, @abs(u_var) + 1);
                                    try tensor2[tensor_out].unaryDivide(allocator, @abs(u_var) + 1);
                                },
                                .unary_exp => {
                                    // NaN prevention
                                    try tensor1[tensor_out].unaryMax(allocator, 10);
                                    try tensor2[tensor_out].unaryMax(allocator, 10);
                                    try tensor1[tensor_out].unaryMin(allocator, -10);
                                    try tensor2[tensor_out].unaryMin(allocator, -10);

                                    try tensor1[tensor_out].unaryExp(allocator);
                                    try tensor2[tensor_out].unaryExp(allocator);
                                },
                                .unary_log => {
                                    // NaN prevention
                                    try tensor1[tensor_out].unaryAbsolute(allocator);
                                    try tensor2[tensor_out].unaryAbsolute(allocator);
                                    try tensor1[tensor_out].unaryAdd(allocator, 1);
                                    try tensor2[tensor_out].unaryAdd(allocator, 1);

                                    try tensor1[tensor_out].unaryLog(allocator);
                                    try tensor2[tensor_out].unaryLog(allocator);
                                },
                                .unary_square => {
                                    // Inf prevention
                                    try tensor1[tensor_out].unaryMax(allocator, 100);
                                    try tensor2[tensor_out].unaryMax(allocator, 100);
                                    try tensor1[tensor_out].unaryMin(allocator, -100);
                                    try tensor2[tensor_out].unaryMin(allocator, -100);

                                    try tensor1[tensor_out].unarySquare(allocator);
                                    try tensor2[tensor_out].unarySquare(allocator);
                                },
                                .unary_sqrt => {
                                    // NaN prevention
                                    try tensor1[tensor_out].unaryAbsolute(allocator);
                                    try tensor2[tensor_out].unaryAbsolute(allocator);

                                    try tensor1[tensor_out].unarySqrt(allocator);
                                    try tensor2[tensor_out].unarySqrt(allocator);
                                },
                                .unary_reciprocal => {
                                    // NaN prevention
                                    try tensor1[tensor_out].unaryAbsolute(allocator);
                                    try tensor2[tensor_out].unaryAbsolute(allocator);
                                    try tensor1[tensor_out].unaryAdd(allocator, 1);
                                    try tensor2[tensor_out].unaryAdd(allocator, 1);

                                    try tensor1[tensor_out].unaryReciprocal(allocator);
                                    try tensor2[tensor_out].unaryReciprocal(allocator);
                                },
                                .unary_max => {
                                    try tensor1[tensor_out].unaryMax(allocator, u_var);
                                    try tensor2[tensor_out].unaryMax(allocator, u_var);
                                },
                                .unary_min => {
                                    try tensor1[tensor_out].unaryMin(allocator, u_var);
                                    try tensor2[tensor_out].unaryMin(allocator, u_var);
                                },
                                .unary_set => {
                                    try tensor1[tensor_out].unarySet(allocator, u_var);
                                    try tensor2[tensor_out].unarySet(allocator, u_var);
                                },
                                .unary_random => {
                                    // TODO: This
                                    try tensor1[tensor_out].unarySet(allocator, u_var);
                                    try tensor2[tensor_out].unarySet(allocator, u_var);
                                },
                                .unary_tanh => {
                                    try tensor1[tensor_out].unaryTanh(allocator);
                                    try tensor2[tensor_out].unaryTanh(allocator);
                                },
                                .unary_absolute => {
                                    try tensor1[tensor_out].unaryAbsolute(allocator);
                                    try tensor2[tensor_out].unaryAbsolute(allocator);
                                },
                                .unary_sign => {
                                    try tensor1[tensor_out].unarySign(allocator);
                                    try tensor2[tensor_out].unarySign(allocator);
                                },
                                .binary_add => {
                                    try tensor1[tensor_out].binaryAdd(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].binaryAdd(allocator, &tensor2[tensor_in]);
                                },
                                .binary_subtract => {
                                    try tensor1[tensor_out].binarySubtract(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].binarySubtract(allocator, &tensor2[tensor_in]);
                                },
                                .binary_multiply => {
                                    try tensor1[tensor_out].binaryMultiply(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].binaryMultiply(allocator, &tensor2[tensor_in]);
                                },
                                .binary_divide => {
                                    // NaN prevention
                                    try tensor1[tensor_in].unaryAbsolute(allocator);
                                    try tensor2[tensor_in].unaryAbsolute(allocator);
                                    try tensor1[tensor_in].unaryAdd(allocator, 1);
                                    try tensor2[tensor_in].unaryAdd(allocator, 1);

                                    try tensor1[tensor_out].binaryDivide(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].binaryDivide(allocator, &tensor2[tensor_in]);
                                },
                                .binary_max => {
                                    try tensor1[tensor_out].binaryMax(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].binaryMax(allocator, &tensor2[tensor_in]);
                                },
                                .binary_min => {
                                    try tensor1[tensor_out].binaryMin(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].binaryMin(allocator, &tensor2[tensor_in]);
                                },
                                .binary_set => {
                                    try tensor1[tensor_out].binarySet(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].binarySet(allocator, &tensor2[tensor_in]);
                                },
                                .linary_add => {
                                    try tensor1[tensor_out].linaryAdd(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].linaryAdd(allocator, &tensor2[tensor_in]);
                                },
                                .linary_subtract => {
                                    try tensor1[tensor_out].linarySubtract(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].linarySubtract(allocator, &tensor2[tensor_in]);
                                },
                                .linary_multiply => {
                                    try tensor1[tensor_out].linaryMultiply(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].linaryMultiply(allocator, &tensor2[tensor_in]);
                                },
                                .linary_divide => {
                                    // NaN prevention
                                    try tensor1[tensor_in].unaryAbsolute(allocator);
                                    try tensor2[tensor_in].unaryAbsolute(allocator);
                                    try tensor1[tensor_in].unaryAdd(allocator, 1);
                                    try tensor2[tensor_in].unaryAdd(allocator, 1);

                                    try tensor1[tensor_out].linaryDivide(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].linaryDivide(allocator, &tensor2[tensor_in]);
                                },
                                .linary_max => {
                                    try tensor1[tensor_out].linaryMax(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].linaryMax(allocator, &tensor2[tensor_in]);
                                },
                                .linary_min => {
                                    try tensor1[tensor_out].linaryMin(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].linaryMin(allocator, &tensor2[tensor_in]);
                                },
                                .linary_set => {
                                    try tensor1[tensor_out].linarySet(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].linarySet(allocator, &tensor2[tensor_in]);
                                },
                                .reduce_sum => {
                                    try tensor1[tensor_out].reduceSum(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].reduceSum(allocator, &tensor2[tensor_in]);
                                },
                                .reduce_max => {
                                    try tensor1[tensor_out].reduceMax(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].reduceMax(allocator, &tensor2[tensor_in]);
                                },
                                .reduce_avg => {
                                    try tensor1[tensor_out].reduceAvg(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].reduceAvg(allocator, &tensor2[tensor_in]);
                                },
                                .reduce_min => {
                                    try tensor1[tensor_out].reduceMin(allocator, &tensor1[tensor_in]);
                                    try tensor2[tensor_out].reduceMin(allocator, &tensor2[tensor_in]);
                                },
                            }
                        }
                    }
                }
            }
        }
    }

    var time_linearized: [iterations]i128 = undefined;
    for (0..iterations) |interation_idx| {
        // Not using realize here because that clears the linearized
        const time_start: i128 = std.time.nanoTimestamp();
        tensor2[op_out[op_num - 1]].linearized.run();
        time_linearized[interation_idx] = std.time.nanoTimestamp() - time_start;
    }
    analyseTimes(time_linearized, "linearized");

    const size_local: u32 = pcg.randBelow(10) + 1;
    const size_global: u32 = size_local * (pcg.randBelow(10) + 1);

    for (0..tensor_num) |tensor_idx| {
        tensor1[tensor_idx].buffer.syncUpdate(.sync_to_device);
        try tensor1[tensor_idx].buffer.syncToDevice(queue);
    }

    const program: Program = try Program.alloc(allocator, tensor1[op_out[op_num - 1]].linearized, //
        size_global, size_local, device, context, queue);
    defer program.free(allocator) catch {};

    var time_program: [iterations]i128 = undefined;
    for (0..iterations) |interation_idx| {
        const time_start: i128 = std.time.nanoTimestamp();
        try program.run();
        time_program[interation_idx] = std.time.nanoTimestamp() - time_start;
    }
    analyseTimes(time_program, "O0");

    for (0..tensor_num) |tensor_idx| {
        tensor1[tensor_idx].buffer.syncUpdate(.sync_to_host);
        try tensor1[tensor_idx].buffer.syncToHost(queue);
    }

    for (0..tensor_num) |tensor_idx| {
        for (0..a_size_max * z_size_max * y_size_max * x_size_max) |arg_idx| {
            try assertEq(tensor1[tensor_idx].buffer.values[arg_idx], tensor2[tensor_idx].buffer.values[arg_idx]);
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.detectLeaks();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    var rng_saved: ?u64 = null;
    // Skip the executable call
    _ = args.next();
    if (args.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "rng=")) {
            const offset = "rng="[0..].len;
            rng_saved = try std.fmt.parseInt(u64, arg[offset..], 10);
        } else {
            std.log.err("Found unrecognised option `{s}`, expected `rng=<number>`.\n", .{arg});
            unreachable;
        }
    }
    const rng: u64 = switch (rng_saved == null) {
        true => @bitCast(std.time.microTimestamp()),
        false => rng_saved.?,
    };

    const device: ClDevice = try ClDevice.alloc(.gpu);
    const context: ClContext = try ClContext.alloc(device);
    const queue: ClCommandQueue = try ClCommandQueue.alloc(device, context);

    try profileCompiler(allocator, rng, device, context, queue);
}
