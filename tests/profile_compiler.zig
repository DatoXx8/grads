const std = @import("std");
const assert = std.debug.assert;
const Pcg = std.Random.Pcg;
const Allocator = std.mem.Allocator;

const grads = @import("grads");
const Tensor = grads.Tensor;
const OpKind = grads.Op.Kind;
const Program = grads.Program;
const Runtime = grads.Runtime;
const RuntimeCl = grads.RuntimeCl;
const Optimization = grads.Optimization;

const randomLinearized = @import("random_linearized.zig").randomLinearized;
const a_size_max = @import("random_linearized.zig").a_size_max;
const z_size_max = @import("random_linearized.zig").z_size_max;
const y_size_max = @import("random_linearized.zig").y_size_max;
const x_size_max = @import("random_linearized.zig").x_size_max;
const op_num = @import("random_linearized.zig").op_num;
const tensor_num = @import("random_linearized.zig").tensor_num;

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
const iterations = 10000;
comptime {
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

// $WARN This does **not** check for correctness, for that use `zig build test-compiler`. I know that sucks, and I plan to change that, but for now that is how it is.
fn profileCompiler(runtime: Runtime, allocator: Allocator, rng: u64) !void {
    std.debug.print("profile-compiler: rng={}...\n", .{rng});

    var pcg = Pcg.init(rng);

    var tensor1 = try randomLinearized(runtime, allocator, @splat(true), rng);
    defer {
        for (&tensor1.tensor) |*tensor| {
            tensor.free(runtime, allocator);
        }
    }
    var tensor2 = try randomLinearized(runtime, allocator, @splat(true), rng);
    defer {
        for (&tensor2.tensor) |*tensor| {
            tensor.free(runtime, allocator);
        }
    }
    assert(tensor1.out_idx == tensor2.out_idx);

    // tensor1[tensor_out].linearized.print(4, 0, null);
    var time_linearized: [iterations]i128 = undefined;
    for (0..iterations) |interation_idx| {
        // Not using realize here because that clears the linearized
        const time_start: i128 = std.time.nanoTimestamp();
        tensor2.tensor[tensor2.out_idx].linearized.run();
        time_linearized[interation_idx] = std.time.nanoTimestamp() - time_start;
    }
    analyseTimes(time_linearized, "linearized");

    const size_local: u32 = pcg.random().uintLessThan(u32, 10) + 1;
    const size_global: u32 = size_local * (pcg.random().uintLessThan(u32, 10) + 1);

    for (0..tensor_num) |tensor_idx| {
        tensor1.tensor[tensor_idx].buffer.syncUpdate(.sync_to_device);
        try tensor1.tensor[tensor_idx].buffer.syncToDevice(runtime);
    }

    inline for (@typeInfo(Optimization).@"enum".fields) |optimization| {
        const name: []const u8 = optimization.name;
        const value: Optimization = @enumFromInt(optimization.value);

        var program: Program = try Program.alloc(runtime, allocator, tensor1.tensor[tensor1.out_idx].linearized, //
            value, size_global, size_local);
        defer program.free(runtime, allocator);

        var time_program: [iterations]i128 = undefined;
        for (0..iterations) |interation_idx| {
            const time_start: i128 = std.time.nanoTimestamp();
            try program.run(runtime);
            time_program[interation_idx] = std.time.nanoTimestamp() - time_start;
        }
        analyseTimes(time_program, name);
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
    while (args.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "rng=")) {
            const offset = "rng="[0..].len;
            rng_saved = try std.fmt.parseInt(u64, arg[offset..], 10);
        } else {
            std.log.err("Found unrecognised option `{s}`, expected `rng=<number>`.\n", .{arg});
            @panic("See above error message");
        }
    }
    const rng: u64 = switch (rng_saved == null) {
        true => std.crypto.random.int(u64),
        false => rng_saved.?,
    };

    var runtime_cl: RuntimeCl = undefined;
    var runtime: Runtime = runtime_cl.runtime();
    try runtime.init();
    defer runtime.deinit();

    try profileCompiler(runtime, allocator, rng);
}
