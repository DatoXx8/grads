const std = @import("std");
const Pcg = std.Random.Pcg;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const grads = @import("grads");
const Tensor = grads.Tensor;
const Linearized = grads.Linearized;
const OpType = grads.Op.Type;
const Program = grads.Program;
const ClContext = grads.ClContext;
const ClDevice = grads.ClDevice;
const ClCommandQueue = grads.ClCommandQueue;
const Optimization = grads.Optimization;

const randomLinearized = @import("random-linearized.zig").randomLinearized;
const a_size_max = @import("random-linearized.zig").a_size_max;
const z_size_max = @import("random-linearized.zig").z_size_max;
const y_size_max = @import("random-linearized.zig").y_size_max;
const x_size_max = @import("random-linearized.zig").x_size_max;
const op_num = @import("random-linearized.zig").op_num;
const tensor_num = @import("random-linearized.zig").tensor_num;

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

fn simulateCompiler(
    allocator: Allocator,
    op_included: [op_num]bool,
    rng: u64,
    optimization: Optimization,
    device: ClDevice,
    context: ClContext,
    queue: ClCommandQueue,
) !void {
    var pcg = Pcg.init(rng);

    var tensor1 = try randomLinearized(allocator, @splat(true), rng, context);
    defer {
        for (&tensor1.tensor) |*tensor| {
            tensor.free(allocator);
        }
    }
    var tensor2 = try randomLinearized(allocator, @splat(true), rng, context);
    defer {
        for (&tensor2.tensor) |*tensor| {
            tensor.free(allocator);
        }
    }
    assert(tensor1.out_idx == tensor2.out_idx);

    tensor2.tensor[tensor2.out_idx].realize();

    const size_local: u32 = pcg.random().uintLessThan(u32, 10) + 1;
    const size_global: u32 = size_local * (pcg.random().uintLessThan(u32, 10) + 1);

    for (0..tensor_num) |tensor_idx| {
        tensor1.tensor[tensor_idx].buffer.syncUpdate(.sync_to_device);
        try tensor1.tensor[tensor_idx].buffer.syncToDevice(queue);
    }

    const program: Program = try Program.alloc(allocator, tensor1.tensor[tensor1.out_idx].linearized, //
        size_global, size_local, optimization, device, context, queue);
    defer program.free(allocator);

    try program.run();

    tensor1.tensor[tensor1.out_idx].buffer.syncUpdate(.sync_to_host);
    try tensor1.tensor[tensor1.out_idx].buffer.syncToHost(queue);

    for (0..tensor1.tensor[tensor1.out_idx].buffer.values.len) |arg_idx| {
        assertEq(tensor1.tensor[tensor1.out_idx].buffer.values[arg_idx], tensor2.tensor[tensor2.out_idx].buffer.values[arg_idx]) catch |err| {
            std.log.err("Difference at index {} = [{}, {}, {}, {}] with {any} between compiled {d} \"{s}\" and linearized {d} \"{s}\"\n", .{
                arg_idx,
                arg_idx / (z_size_max * y_size_max * x_size_max),
                arg_idx / (y_size_max * x_size_max) % z_size_max,
                arg_idx / x_size_max % y_size_max,
                arg_idx % x_size_max,
                op_included,
                tensor1.tensor[tensor1.out_idx].buffer.values[arg_idx],
                tensor1.tensor[tensor1.out_idx].buffer.name(),
                tensor2.tensor[tensor2.out_idx].buffer.values[arg_idx],
                tensor2.tensor[tensor2.out_idx].buffer.name(),
            });
            return err;
        };
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
    assert(tensor_num > 1);
    assert(op_num > 0);
    var op_included: [op_num]bool = @splat(true);
    for (0..op_num) |op_idx| {
        var failed: bool = false;
        op_included[op_idx] = false;
        simulateCompiler(allocator, op_included, rng, optimization, device, context, queue) catch {
            failed = true;
        };
        if (!failed) {
            op_included[op_idx] = true;
        }
    }
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
                inline for (@typeInfo(Optimization).@"enum".fields, 0..) |optimization, optimization_idx| {
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
            inline for (@typeInfo(Optimization).@"enum".fields, 0..) |optimization, optimization_idx| {
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
        true => std.crypto.random.int(u64),
        false => rng_saved.?,
    };

    const device: ClDevice = try ClDevice.alloc(.gpu);
    const context: ClContext = try ClContext.alloc(device);
    const queue: ClCommandQueue = try ClCommandQueue.alloc(device, context);

    if (loop_infinite) {
        var loop_idx: u64 = 0;
        while (true) {
            std.debug.print("{} => simulate-compiler: rng={}... ", .{ loop_idx, rng +% loop_idx });
            if (opt_saved) |opt| {
                simulateCompiler(allocator, @splat(true), rng +% loop_idx, opt, device, context, queue) catch |err| {
                    try minifyCompiler(allocator, rng +% loop_idx, opt, err, device, context, queue);
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
                inline for (@typeInfo(Optimization).@"enum".fields) |optimization| {
                    const name: []const u8 = optimization.name;
                    const value: Optimization = @enumFromInt(optimization.value);
                    simulateCompiler(allocator, @splat(true), rng +% loop_idx, value, device, context, queue) catch |err| {
                        try minifyCompiler(allocator, rng +% loop_idx, value, err, device, context, queue);
                    };
                    std.debug.print("{s} ", .{name});
                }
            }
            loop_idx += 1;
            std.debug.print("passed!\n", .{});
        }
    } else {
        for (0..loop_count) |loop_idx| {
            std.debug.print("{} => simulate-compiler: rng={}... ", .{ loop_idx, rng +% loop_idx });
            if (opt_saved) |opt| {
                simulateCompiler(allocator, @splat(true), rng +% loop_idx, opt, device, context, queue) catch |err| {
                    try minifyCompiler(allocator, rng +% loop_idx, opt, err, device, context, queue);
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
                inline for (@typeInfo(Optimization).@"enum".fields) |optimization| {
                    const name: []const u8 = optimization.name;
                    const value: Optimization = @enumFromInt(optimization.value);
                    simulateCompiler(allocator, @splat(true), rng +% loop_idx, value, device, context, queue) catch |err| {
                        try minifyCompiler(allocator, rng +% loop_idx, value, err, device, context, queue);
                    };
                    std.debug.print("{s} ", .{name});
                }
            }
            std.debug.print("passed!\n", .{});
        }
    }
}
