const std = @import("std");
const assert = std.debug.assert;
const Pcg = std.Random.Pcg;
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const grads = @import("grads");
const OpKind = grads.Op.Kind;
const Program = grads.Program;
const Runtime = grads.Runtime;
const RuntimeCl = grads.RuntimeCl;
const Optimization = grads.Optimization;
const util = grads.util;

const randomLinearized = @import("random_linearized.zig").randomLinearized;
const a_size_max = @import("random_linearized.zig").a_size_max;
const z_size_max = @import("random_linearized.zig").z_size_max;
const y_size_max = @import("random_linearized.zig").y_size_max;
const x_size_max = @import("random_linearized.zig").x_size_max;
const op_num = @import("random_linearized.zig").op_num;
const buffer_num = @import("random_linearized.zig").buffer_num;

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
        util.log.print("\n", .{});
        std.log.err("Found NaN in equality comparison.\n", .{});
        return AssertError.nan;
    } else if (std.math.isInf(val1) or std.math.isInf(val2)) {
        util.log.print("\n", .{});
        std.log.err("Found Inf in equality comparison.\n", .{});
        return AssertError.nan;
    } else if (std.math.approxEqAbs(f32, val1, val2, epsilon) or std.math.approxEqRel(f32, val1, val2, epsilon_relative)) {
        return;
    } else {
        // For nicer output formatting
        util.log.print("\n", .{});
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
        util.log.print("Time: {d:8.4}ns +- {d:8.4}ns", .{ ns_mean, ns_variance });
    } else if (ns_mean < 1_000_000) {
        const us_mean: f64 = @as(f64, @floatFromInt(ns_mean)) / 1_000;
        const us_variance: f64 = @as(f64, @floatFromInt(ns_variance)) / 1_000;
        util.log.print("Time: {d:8.4}us +- {d:8.4}us", .{ us_mean, us_variance });
    } else if (ns_mean < 1_000_000_000) {
        const ms_mean: f64 = @as(f64, @floatFromInt(ns_mean)) / 1_000_000;
        const ms_variance: f64 = @as(f64, @floatFromInt(ns_variance)) / 1_000_000;
        util.log.print("Time: {d:8.4}ms +- {d:8.4}ms", .{ ms_mean, ms_variance });
    } else {
        const s_mean: f64 = @as(f64, @floatFromInt(ns_mean)) / 1_000_000_000;
        const s_variance: f64 = @as(f64, @floatFromInt(ns_variance)) / 1_000_000_000;
        util.log.print("Time: {d:8.4} s +- {d:8.4} s", .{ s_mean, s_variance });
    }
    util.log.print(" {s}\n", .{name});
}

// $WARN This does **not** check for correctness, for that use `zig build test-compiler`. I know that sucks, and I plan to change that, but for now that is how it is.
fn profileCompiler(runtime: Runtime, gpa: Allocator, rng: u64) !void {
    util.log.print("profile_compiler: rng={}...\n", .{rng});

    var pcg = Pcg.init(rng);

    var arena_allocator: ArenaAllocator = .init(gpa);
    defer arena_allocator.deinit();
    const arena: Allocator = arena_allocator.allocator();

    var arena_temp_allocator: ArenaAllocator = .init(gpa);
    defer arena_temp_allocator.deinit();
    const arena_temp: Allocator = arena_temp_allocator.allocator();

    var buffer1 = try randomLinearized(runtime, arena, @splat(true), rng);
    defer {
        for (&buffer1.buffer) |buffer| {
            buffer.free(runtime);
        }
    }
    var buffer2 = try randomLinearized(runtime, arena, @splat(true), rng);
    defer {
        for (&buffer2.buffer) |buffer| {
            buffer.free(runtime);
        }
    }
    assert(buffer1.out_idx == buffer2.out_idx);

    var time_linearized: [iterations]i128 = undefined;
    for (0..iterations) |interation_idx| {
        // Not using realize here because that clears the linearized
        const time_start: i128 = std.time.nanoTimestamp();
        buffer2.linearized.run();
        time_linearized[interation_idx] = std.time.nanoTimestamp() - time_start;
    }
    analyseTimes(time_linearized, "linearized");

    const size_local: u32 = pcg.random().uintLessThan(u32, 10) + 1;
    const size_global: u32 = size_local * (pcg.random().uintLessThan(u32, 10) + 1);

    for (0..buffer_num) |buffer_idx| {
        buffer1.buffer[buffer_idx].syncUpdate(.sync_to_device);
        try buffer1.buffer[buffer_idx].syncToDevice(runtime);
    }

    const depth_max: []const u32 = &.{ 0, 1, 10, 100, 1000 };
    for (depth_max) |depth| {
        defer _ = arena_temp_allocator.reset(.retain_capacity);

        var program: Program = try Program.alloc(runtime, gpa, arena_temp, arena_temp, buffer1.linearized, //
            depth, size_global, size_local);

        var time_program: [iterations]i128 = undefined;
        for (0..iterations) |interation_idx| {
            const time_start: i128 = std.time.nanoTimestamp();
            try program.run(runtime);
            time_program[interation_idx] = std.time.nanoTimestamp() - time_start;
        }
        var buf: [512]u8 = @splat(0);
        const written: []const u8 = try std.fmt.bufPrint(&buf, "d={d}", .{depth});
        analyseTimes(time_program, written);
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
            std.debug.panic("Found unrecognised option `{s}`, expected `rng=<number>`.\n", .{arg});
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
