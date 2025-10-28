const std = @import("std");
const assert = std.debug.assert;
const Pcg = std.Random.Pcg;

const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const grads = @import("grads");
const Buffer = grads.Buffer;
const Linearized = grads.Linearized;
const Runtime = grads.Runtime;
const RuntimeNoop = grads.RuntimeNoop;

const AssertError = error{
    nan,
    difference,
};
/// Margin of error
const epsilon: f32 = 1e-9;
/// Check for equality between the two floats within the margin of error of `epsilon`
fn assertEq(val1: f32, val2: f32) !void {
    if (std.math.approxEqAbs(f32, val1, val2, epsilon)) {
        return;
    } else {
        if (std.math.isNan(val1) or std.math.isNan(val2)) {
            // For nicer output formatting
            std.debug.print("\n", .{});
            std.log.err("Found NaN in equality comparison.\n", .{});
            return AssertError.nan;
        } else {
            // For nicer output formatting
            std.debug.print("\n", .{});
            std.log.err("Difference between {d} and {d} is too large.\n", .{ val1, val2 });
            return AssertError.difference;
        }
    }
}

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = general_purpose_allocator.allocator();
    defer _ = general_purpose_allocator.detectLeaks();

    var arena_allocator: ArenaAllocator = .init(gpa);
    defer arena_allocator.deinit();
    const arena: Allocator = arena_allocator.allocator();

    var args = try std.process.argsWithAllocator(gpa);
    defer args.deinit();

    var rng_saved: ?u64 = null;
    // Skip the executable call
    _ = args.next();
    while (args.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "rng=")) {
            const offset = "rng="[0..].len;
            rng_saved = try std.fmt.parseInt(u64, arg[offset..], 10);
        } else {
            std.log.err("Found unrecognised option `{s}`, expected `rng=<number>` or nothing.\n", .{arg});
            unreachable;
        }
    }

    var runtime_noop: RuntimeNoop = undefined;
    const runtime: Runtime = runtime_noop.runtime();

    const a_size: u32 = 6;
    const z_size: u32 = 5;
    const y_size: u32 = 4;
    const x_size: u32 = 3;

    var linearized1: Linearized = try .alloc(arena, 3);
    var buffer1 = try Buffer.alloc(runtime, arena, a_size, z_size, y_size, x_size, .normal);
    defer buffer1.free(runtime);
    var buffer2 = try Buffer.alloc(runtime, arena, a_size, z_size, y_size, x_size, .normal);
    defer buffer2.free(runtime);
    const val1 = try arena.alloc(f32, a_size * z_size * y_size * x_size);
    const val2 = try arena.alloc(f32, a_size * z_size * y_size * x_size);

    const rng: u64 = switch (rng_saved == null) {
        true => std.crypto.random.int(u64),
        false => rng_saved.?,
    };
    std.debug.print("unit-ops: rng={}...", .{rng});
    defer std.debug.print(" passed!\n", .{});
    var pcg = Pcg.init(rng);

    for (0..a_size * z_size * y_size * x_size) |val_idx| {
        val1[val_idx] = pcg.random().floatNorm(f32);
        val2[val_idx] = pcg.random().floatNorm(f32);
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    std.mem.copyForwards(f32, buffer2.values, val2);
    linearized1.unaryAdd(buffer1, 2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], val1[arg_idx] + 2);
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.unarySubtract(buffer1, 2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], val1[arg_idx] - 2);
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.unaryMultiply(buffer1, 2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], val1[arg_idx] * 2);
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.unaryDivide(buffer1, 2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], val1[arg_idx] / 2);
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.unaryAbsolute(buffer1);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], @abs(val1[arg_idx]));
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.unaryExp(buffer1);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], std.math.exp(val1[arg_idx]));
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    // This is to avoid NaNs
    linearized1.unaryAbsolute(buffer1);
    linearized1.unaryAdd(buffer1, 1);
    linearized1.unaryLog(buffer1);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], std.math.log(f32, std.math.e, @abs(val1[arg_idx]) + 1));
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.unarySquare(buffer1);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], val1[arg_idx] * val1[arg_idx]);
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    // This it to avoid NaNs
    linearized1.unaryAbsolute(buffer1);
    linearized1.unarySqrt(buffer1);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], std.math.sqrt(@abs(val1[arg_idx])));
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    // This is to avoid NaNs
    linearized1.unaryAbsolute(buffer1);
    linearized1.unaryAdd(buffer1, 1);
    linearized1.unaryReciprocal(buffer1);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], 1 / (@abs(val1[arg_idx]) + 1));
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.unaryMax(buffer1, 1);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], @max(val1[arg_idx], 1));
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.unaryMin(buffer1, 1);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], @min(val1[arg_idx], 1));
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.unarySet(buffer1, 2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], 2);
    }

    // Theoretically speaking because this *is* random this could
    // error despite everything working as it should, but doing some
    // back of the envelope calculations that should be rare enough to
    // not really worry about it.
    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.unaryRandom(buffer1, @truncate(rng));
    linearized1.realize();
    var product: f32 = 1;
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        product *= buffer1.values[arg_idx];
    }
    if (@abs(product) < 1 / @as(f32, a_size * z_size * y_size * x_size)) {
        // Happy case
    } else {
        // For nicer output formatting
        std.debug.print("\n", .{});
        std.log.err(
            \\ Difference in unary_random too large. 
            \\ Because of the randomnes of unary_random this error could happen despite everything working correctly and should only be investigated if it happens when running this test again.
            \\ The values where {d} and {d}.
        , .{ @abs(product), 1 / @as(f32, a_size * z_size * y_size * x_size) });
        return AssertError.difference;
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.unaryTanh(buffer1);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], std.math.tanh(val1[arg_idx]));
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.unarySign(buffer1);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        if (val1[arg_idx] > 0) {
            try assertEq(buffer1.values[arg_idx], 1);
        } else if (val1[arg_idx] < 0) {
            try assertEq(buffer1.values[arg_idx], -1);
        } else {
            try assertEq(buffer1.values[arg_idx], 0);
        }
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.binaryAdd(buffer1, buffer2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], val1[arg_idx] + val2[arg_idx]);
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.binarySubtract(buffer1, buffer2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], val1[arg_idx] - val2[arg_idx]);
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.binaryMultiply(buffer1, buffer2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], val1[arg_idx] * val2[arg_idx]);
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.binaryDivide(buffer1, buffer2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], val1[arg_idx] / val2[arg_idx]);
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.binaryMax(buffer1, buffer2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], @max(val1[arg_idx], val2[arg_idx]));
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.binaryMin(buffer1, buffer2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], @min(val1[arg_idx], val2[arg_idx]));
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.binarySet(buffer1, buffer2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], val2[arg_idx]);
    }

    buffer2.moveResize(1, 1, 1, 1);
    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.expandAdd(buffer1, buffer2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], val1[arg_idx] + val2[0]);
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.expandSubtract(buffer1, buffer2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], val1[arg_idx] - val2[0]);
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.expandMultiply(buffer1, buffer2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], val1[arg_idx] * val2[0]);
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.expandDivide(buffer1, buffer2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], val1[arg_idx] / val2[0]);
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.expandMax(buffer1, buffer2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], @max(val1[arg_idx], val2[0]));
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.expandMin(buffer1, buffer2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], @min(val1[arg_idx], val2[0]));
    }

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.expandSet(buffer1, buffer2);
    linearized1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(buffer1.values[arg_idx], val2[0]);
    }

    buffer1.moveResize(1, 1, 1, 1);
    buffer2.moveResize(a_size, z_size, y_size, x_size);
    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.reduceSum(buffer1, buffer2);
    linearized1.realize();
    var sum: f32 = 0;
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        sum += val2[arg_idx];
    }
    try assertEq(buffer1.values[0], sum);

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.reduceAvg(buffer1, buffer2);
    linearized1.realize();
    sum = 0;
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        sum += val2[arg_idx];
    }
    try assertEq(buffer1.values[0], sum / @as(f32, a_size * z_size * y_size * x_size));

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.reduceMax(buffer1, buffer2);
    linearized1.realize();
    var max: f32 = -std.math.inf(f32);
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        max = @max(max, val2[arg_idx]);
    }
    try assertEq(buffer1.values[0], max);

    std.mem.copyForwards(f32, buffer1.values, val1);
    linearized1.reduceMin(buffer1, buffer2);
    linearized1.realize();
    var min: f32 = std.math.inf(f32);
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        min = @min(min, val2[arg_idx]);
    }
    try assertEq(buffer1.values[0], min);
}
