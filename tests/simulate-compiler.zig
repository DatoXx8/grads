const std = @import("std");
const grads = @import("grads");

const Tensor = grads.Tensor;
const OpType = grads.Op.Type;
const Program = grads.Program;
const ClContext = grads.ClContext;
const ClDevice = grads.ClDevice;
const ClCommandQueue = grads.ClCommandQueue;

const Pcg = std.Random.Pcg;

const Optimization = grads.Optimization;

const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

// TODO: Randomize intermediary buffers, but make sure the final out tensor is not intermediary

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

const tensor_num = 10;
const op_num = 10;
comptime {
    assert(tensor_num > 1);
    assert(op_num > 0);
}

fn simulateCompiler(
    allocator: Allocator,
    op_off_low: usize,
    op_off_top: usize,
    rng: u64,
    optimization: Optimization,
    device: ClDevice,
    context: ClContext,
    queue: ClCommandQueue,
) !void {
    assert(op_num > op_off_low);
    assert(op_num > op_off_top);
    assert(op_num > op_off_top + op_off_low);

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

    var pcg = Pcg.init(rng);

    for (0..tensor_num) |tensor_idx| {
        for (0..a_size_max * z_size_max * y_size_max * x_size_max) |arg_idx| {
            tensor1[tensor_idx].buffer.values[arg_idx] = pcg.random().floatNorm(f32);
            tensor2[tensor_idx].buffer.values[arg_idx] = tensor1[tensor_idx].buffer.values[arg_idx];
        }
    }

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

    var op_idx_used: usize = 0;
    for (0..op_num) |op_idx| {
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

        const loop_len: u32 = pcg.random().uintLessThan(u32, @truncate(op_num - op_idx));
        op_idx_used = op_idx + loop_len;

        for (0..a_loop) |a_idx| {
            for (0..z_loop) |z_idx| {
                for (0..y_loop) |y_idx| {
                    for (0..x_loop) |x_idx| {
                        for (0..loop_len) |loop_idx| {
                            // Putting this in here hurts performance slightly but
                            // it allows partial loops for nicer debugging
                            if (op_idx + loop_idx < op_off_low or op_idx + loop_idx >= op_num - op_off_top) {
                                continue;
                            }

                            const tensor_out: u32 = op_out[op_idx + loop_idx];
                            const tensor_in: u32 = op_in[op_idx + loop_idx];

                            // Essentially free in case no alocattions are necessary
                            try tensor1[tensor_out].linearized.capacityEnsure(allocator, (4 + tensor1[tensor_in].linearized.op_num) * loop_len);
                            try tensor2[tensor_out].linearized.capacityEnsure(allocator, (4 + tensor2[tensor_in].linearized.op_num) * loop_len);

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
                                    tensor1[tensor_out].unaryAdd(u_var);
                                    tensor2[tensor_out].unaryAdd(u_var);
                                },
                                .unary_subtract => {
                                    tensor1[tensor_out].unarySubtract(u_var);
                                    tensor2[tensor_out].unarySubtract(u_var);
                                },
                                .unary_multiply => {
                                    tensor1[tensor_out].unaryMultiply(u_var);
                                    tensor2[tensor_out].unaryMultiply(u_var);
                                },
                                .unary_divide => {
                                    tensor1[tensor_out].unaryDivide(@abs(u_var) + 1);
                                    tensor2[tensor_out].unaryDivide(@abs(u_var) + 1);
                                },
                                .unary_exp => {
                                    // NaN prevention
                                    tensor1[tensor_out].unaryMax(10);
                                    tensor2[tensor_out].unaryMax(10);
                                    tensor1[tensor_out].unaryMin(-10);
                                    tensor2[tensor_out].unaryMin(-10);

                                    tensor1[tensor_out].unaryExp();
                                    tensor2[tensor_out].unaryExp();
                                },
                                .unary_log => {
                                    // NaN prevention
                                    tensor1[tensor_out].unaryAbsolute();
                                    tensor2[tensor_out].unaryAbsolute();
                                    tensor1[tensor_out].unaryAdd(1);
                                    tensor2[tensor_out].unaryAdd(1);

                                    tensor1[tensor_out].unaryLog();
                                    tensor2[tensor_out].unaryLog();
                                },
                                .unary_square => {
                                    // Inf prevention
                                    tensor1[tensor_out].unaryMax(100);
                                    tensor2[tensor_out].unaryMax(100);
                                    tensor1[tensor_out].unaryMin(-100);
                                    tensor2[tensor_out].unaryMin(-100);

                                    tensor1[tensor_out].unarySquare();
                                    tensor2[tensor_out].unarySquare();
                                },
                                .unary_sqrt => {
                                    // NaN prevention
                                    tensor1[tensor_out].unaryAbsolute();
                                    tensor2[tensor_out].unaryAbsolute();

                                    tensor1[tensor_out].unarySqrt();
                                    tensor2[tensor_out].unarySqrt();
                                },
                                .unary_reciprocal => {
                                    // NaN prevention
                                    tensor1[tensor_out].unaryAbsolute();
                                    tensor2[tensor_out].unaryAbsolute();
                                    tensor1[tensor_out].unaryAdd(1);
                                    tensor2[tensor_out].unaryAdd(1);

                                    tensor1[tensor_out].unaryReciprocal();
                                    tensor2[tensor_out].unaryReciprocal();
                                },
                                .unary_max => {
                                    tensor1[tensor_out].unaryMax(u_var);
                                    tensor2[tensor_out].unaryMax(u_var);
                                },
                                .unary_min => {
                                    tensor1[tensor_out].unaryMin(u_var);
                                    tensor2[tensor_out].unaryMin(u_var);
                                },
                                .unary_set => {
                                    tensor1[tensor_out].unarySet(u_var);
                                    tensor2[tensor_out].unarySet(u_var);
                                },
                                .unary_random => {
                                    // TODO: This
                                    tensor1[tensor_out].unarySet(u_var);
                                    tensor2[tensor_out].unarySet(u_var);
                                },
                                .unary_tanh => {
                                    tensor1[tensor_out].unaryTanh();
                                    tensor2[tensor_out].unaryTanh();
                                },
                                .unary_absolute => {
                                    tensor1[tensor_out].unaryAbsolute();
                                    tensor2[tensor_out].unaryAbsolute();
                                },
                                .unary_sign => {
                                    tensor1[tensor_out].unaryAbsolute();
                                    tensor2[tensor_out].unaryAbsolute();
                                    // TODO: Reenable this when this is implemented
                                    // tensor1[tensor_out].unarySign();
                                    // tensor2[tensor_out].unarySign();
                                },
                                .binary_add => {
                                    tensor1[tensor_out].binaryAdd(&tensor1[tensor_in]);
                                    tensor2[tensor_out].binaryAdd(&tensor2[tensor_in]);
                                },
                                .binary_subtract => {
                                    tensor1[tensor_out].binarySubtract(&tensor1[tensor_in]);
                                    tensor2[tensor_out].binarySubtract(&tensor2[tensor_in]);
                                },
                                .binary_multiply => {
                                    tensor1[tensor_out].binaryMultiply(&tensor1[tensor_in]);
                                    tensor2[tensor_out].binaryMultiply(&tensor2[tensor_in]);
                                },
                                .binary_divide => {
                                    // NaN prevention
                                    tensor1[tensor_in].unaryAbsolute();
                                    tensor2[tensor_in].unaryAbsolute();
                                    tensor1[tensor_in].unaryAdd(1);
                                    tensor2[tensor_in].unaryAdd(1);

                                    tensor1[tensor_out].binaryDivide(&tensor1[tensor_in]);
                                    tensor2[tensor_out].binaryDivide(&tensor2[tensor_in]);
                                },
                                .binary_max => {
                                    tensor1[tensor_out].binaryMax(&tensor1[tensor_in]);
                                    tensor2[tensor_out].binaryMax(&tensor2[tensor_in]);
                                },
                                .binary_min => {
                                    tensor1[tensor_out].binaryMin(&tensor1[tensor_in]);
                                    tensor2[tensor_out].binaryMin(&tensor2[tensor_in]);
                                },
                                .binary_set => {
                                    tensor1[tensor_out].binarySet(&tensor1[tensor_in]);
                                    tensor2[tensor_out].binarySet(&tensor2[tensor_in]);
                                },
                                .linary_add => {
                                    tensor1[tensor_out].linaryAdd(&tensor1[tensor_in]);
                                    tensor2[tensor_out].linaryAdd(&tensor2[tensor_in]);
                                },
                                .linary_subtract => {
                                    tensor1[tensor_out].linarySubtract(&tensor1[tensor_in]);
                                    tensor2[tensor_out].linarySubtract(&tensor2[tensor_in]);
                                },
                                .linary_multiply => {
                                    tensor1[tensor_out].linaryMultiply(&tensor1[tensor_in]);
                                    tensor2[tensor_out].linaryMultiply(&tensor2[tensor_in]);
                                },
                                .linary_divide => {
                                    // NaN prevention
                                    tensor1[tensor_in].unaryAbsolute();
                                    tensor2[tensor_in].unaryAbsolute();
                                    tensor1[tensor_in].unaryAdd(1);
                                    tensor2[tensor_in].unaryAdd(1);

                                    tensor1[tensor_out].linaryDivide(&tensor1[tensor_in]);
                                    tensor2[tensor_out].linaryDivide(&tensor2[tensor_in]);
                                },
                                .linary_max => {
                                    tensor1[tensor_out].linaryMax(&tensor1[tensor_in]);
                                    tensor2[tensor_out].linaryMax(&tensor2[tensor_in]);
                                },
                                .linary_min => {
                                    tensor1[tensor_out].linaryMin(&tensor1[tensor_in]);
                                    tensor2[tensor_out].linaryMin(&tensor2[tensor_in]);
                                },
                                .linary_set => {
                                    tensor1[tensor_out].linarySet(&tensor1[tensor_in]);
                                    tensor2[tensor_out].linarySet(&tensor2[tensor_in]);
                                },
                                .reduce_sum => {
                                    tensor1[tensor_out].reduceSum(&tensor1[tensor_in]);
                                    tensor2[tensor_out].reduceSum(&tensor2[tensor_in]);
                                },
                                .reduce_max => {
                                    tensor1[tensor_out].reduceMax(&tensor1[tensor_in]);
                                    tensor2[tensor_out].reduceMax(&tensor2[tensor_in]);
                                },
                                .reduce_avg => {
                                    tensor1[tensor_out].reduceAvg(&tensor1[tensor_in]);
                                    tensor2[tensor_out].reduceAvg(&tensor2[tensor_in]);
                                },
                                .reduce_min => {
                                    tensor1[tensor_out].reduceMin(&tensor1[tensor_in]);
                                    tensor2[tensor_out].reduceMin(&tensor2[tensor_in]);
                                },
                            }
                        }
                    }
                }
            }
        }
    }

    const tensor_out: usize = op_out[op_num - (op_off_top + 1)];
    tensor2[tensor_out].realize();

    const size_local: u32 = pcg.random().uintLessThan(u32, 10) + 1;
    const size_global: u32 = size_local * (pcg.random().uintLessThan(u32, 10) + 1);

    for (0..tensor_num) |tensor_idx| {
        tensor1[tensor_idx].buffer.syncUpdate(.sync_to_device);
        try tensor1[tensor_idx].buffer.syncToDevice(queue);
    }

    const program: Program = try Program.alloc(allocator, tensor1[tensor_out].linearized, //
        size_global, size_local, optimization, device, context, queue);
    defer program.free(allocator);

    try program.run();

    tensor1[tensor_out].buffer.syncUpdate(.sync_to_host);
    try tensor1[tensor_out].buffer.syncToHost(queue);

    for (0..a_size_max * z_size_max * y_size_max * x_size_max) |arg_idx| {
        try assertEq(tensor1[tensor_out].buffer.values[arg_idx], tensor2[tensor_out].buffer.values[arg_idx]);
    }
}

fn minifyCompiler(
    allocator: Allocator,
    rng: u64,
    optimization: Optimization,
    err: anytype,
    device: ClDevice,
    context: ClContext,
    queue: ClCommandQueue,
) !void {
    // TODO: Assert that the thing actually fails
    assert(tensor_num > 1);
    assert(op_num > 0);
    var op_top: usize = 1;
    for (1..op_num) |op_removed| {
        var failed: bool = false;
        simulateCompiler(allocator, 0, op_removed, rng, optimization, device, context, queue) catch {
            failed = true;
        };
        if (failed) {
            op_top = op_removed;
            std.debug.print("failed!\n", .{});
            continue;
        } else {
            std.debug.print("passed!\n", .{});
            break;
        }
    }
    // If it fails with no ops there's a serious issue
    assert(op_top > 0);
    var op_low: usize = 0;
    for (1..op_num - op_top) |op_removed| {
        var failed: bool = false;
        simulateCompiler(allocator, op_removed, op_top, rng, optimization, device, context, queue) catch {
            failed = true;
        };
        if (failed) {
            op_low = op_removed;
            std.debug.print("failed!\n", .{});
            continue;
        } else {
            std.debug.print("passed!\n", .{});
            break;
        }
    }
    std.debug.print("Passes below {} and not after {}\n", .{ op_low, op_num - op_top });
    return err;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.detectLeaks();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    var rng_saved: ?u64 = null;
    var loop_infinite: bool = false;
    var loop_count: u64 = 1;
    var opt_saved: ?Optimization = null;
    // Skip the executable call
    _ = args.next();
    while (args.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "rng=")) {
            const offset = "rng="[0..].len;
            rng_saved = try std.fmt.parseInt(u64, arg[offset..], 10);
        } else if (std.mem.startsWith(u8, arg, "loop=")) {
            const offset = "loop="[0..].len;
            loop_count = std.fmt.parseInt(u64, arg[offset..], 10) catch 0;
            if (loop_count == 0) {
                loop_infinite = true;
            }
            // Iff the loop is infinite then the loop count has to be 0
            assert(loop_infinite == (loop_count == 0));
        } else if (std.mem.startsWith(u8, arg, "opt=")) {
            const offset = "opt="[0..].len;
            const parse: []const u8 = arg[offset..];
            opt_saved = std.meta.stringToEnum(Optimization, parse);

            if (opt_saved == null) {
                std.debug.print("Found unrecognized optimization {s}, expected opt=[", .{parse});
                inline for (@typeInfo(Optimization).Enum.fields, 0..) |optimization, optimization_idx| {
                    if (optimization_idx == 0) {
                        std.debug.print("{s} ", .{optimization.name});
                    } else {
                        std.debug.print("| {s} ", .{optimization.name});
                    }
                }
                std.debug.print("]\n", .{});
                unreachable;
            }
        } else {
            std.debug.print("error: Found unrecognised option `{s}`, expected `rng=<number>`, `loop=[number] or opt=[", .{arg});
            inline for (@typeInfo(Optimization).Enum.fields, 0..) |optimization, optimization_idx| {
                if (optimization_idx == 0) {
                    std.debug.print("{s} ", .{optimization.name});
                } else {
                    std.debug.print("| {s} ", .{optimization.name});
                }
            }
            std.log.err("]\n", .{});
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

    if (loop_infinite) {
        var loop_idx: u64 = 0;
        // TODO: Decide how to reseed the random number generator here...
        // rng + loop_idx "wastes" the least seeds but it could cause issues
        // when running multiple threads with this because you then run the same tests over and over again
        while (true) {
            std.debug.print("{} => simulate-compiler: rng={}... ", .{ loop_idx, rng + loop_idx });
            if (opt_saved) |opt| {
                simulateCompiler(allocator, 0, 0, rng + loop_idx, opt, device, context, queue) catch |err| {
                    try minifyCompiler(allocator, rng + loop_idx, opt, err, device, context, queue);
                };
                std.debug.print("{s} ", .{
                    switch (opt) {
                        .O0 => "O0",
                        .O1 => "O1",
                        .O2 => "O2",
                        .O3 => "O3",
                    },
                });
            } else {
                inline for (@typeInfo(Optimization).Enum.fields) |optimization| {
                    const name: []const u8 = optimization.name;
                    const value: Optimization = @enumFromInt(optimization.value);
                    simulateCompiler(allocator, 0, 0, rng + loop_idx, value, device, context, queue) catch |err| {
                        try minifyCompiler(allocator, rng + loop_idx, value, err, device, context, queue);
                    };
                    std.debug.print("{s} ", .{name});
                }
            }
            loop_idx += 1;
            std.debug.print("passed!\n", .{});
        }
    } else {
        for (0..loop_count) |loop_idx| {
            std.debug.print("{} => simulate-compiler: rng={}... ", .{ loop_idx, rng + loop_idx });
            if (opt_saved) |opt| {
                simulateCompiler(allocator, 0, 0, rng + loop_idx, opt, device, context, queue) catch |err| {
                    try minifyCompiler(allocator, rng + loop_idx, opt, err, device, context, queue);
                };
                std.debug.print("{s} ", .{
                    switch (opt) {
                        .O0 => "O0",
                        .O1 => "O1",
                        .O2 => "O2",
                        .O3 => "O3",
                    },
                });
            } else {
                inline for (@typeInfo(Optimization).Enum.fields) |optimization| {
                    const name: []const u8 = optimization.name;
                    const value: Optimization = @enumFromInt(optimization.value);
                    simulateCompiler(allocator, 0, 0, rng + loop_idx, value, device, context, queue) catch |err| {
                        try minifyCompiler(allocator, rng + loop_idx, value, err, device, context, queue);
                    };
                    std.debug.print("{s} ", .{name});
                }
            }
            std.debug.print("passed!\n", .{});
        }
    }
}
