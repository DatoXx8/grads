const std = @import("std");
const assert = std.debug.assert;
const Pcg = std.Random.Pcg;

const grads = @import("grads");
const Tensor = grads.Tensor;
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
    var tensor1 = try Tensor.alloc(runtime, allocator, a_size, z_size, y_size, x_size, 3);
    var tensor2 = try Tensor.alloc(runtime, allocator, a_size, z_size, y_size, x_size, 3);
    const val1 = try allocator.alloc(f32, a_size * z_size * y_size * x_size);
    const val2 = try allocator.alloc(f32, a_size * z_size * y_size * x_size);
    defer tensor1.free(runtime, allocator);
    defer tensor2.free(runtime, allocator);
    defer allocator.free(val1);
    defer allocator.free(val2);

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

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    std.mem.copyForwards(f32, tensor2.buffer.values, val2);
    tensor1.unaryAdd(2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], val1[arg_idx] + 2);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.unarySubtract(2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], val1[arg_idx] - 2);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.unaryMultiply(2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], val1[arg_idx] * 2);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.unaryDivide(2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], val1[arg_idx] / 2);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.unaryAbsolute();
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], @abs(val1[arg_idx]));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.unaryExp();
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], std.math.exp(val1[arg_idx]));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    // This is to avoid NaNs
    tensor1.unaryAbsolute();
    tensor1.unaryAdd(1);
    tensor1.unaryLog();
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], std.math.log(f32, std.math.e, @abs(val1[arg_idx]) + 1));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.unarySquare();
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], val1[arg_idx] * val1[arg_idx]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    // This it to avoid NaNs
    tensor1.unaryAbsolute();
    tensor1.unarySqrt();
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], std.math.sqrt(@abs(val1[arg_idx])));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    // This is to avoid NaNs
    tensor1.unaryAbsolute();
    tensor1.unaryAdd(1);
    tensor1.unaryReciprocal();
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], 1 / (@abs(val1[arg_idx]) + 1));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.unaryMax(1);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], @max(val1[arg_idx], 1));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.unaryMin(1);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], @min(val1[arg_idx], 1));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.unarySet(2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], 2);
    }

    // Theoretically speaking because this *is* random this could
    // error despite everything working as it should, but doing some
    // back of the envelope calculations that should be rare enough to
    // not really worry about it.
    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.unaryRandom(@truncate(rng));
    tensor1.realize();
    var product: f32 = 1;
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        product *= tensor1.buffer.values[arg_idx];
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

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.unaryTanh();
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], std.math.tanh(val1[arg_idx]));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.unarySign();
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        if (val1[arg_idx] > 0) {
            try assertEq(tensor1.buffer.values[arg_idx], 1);
        } else if (val1[arg_idx] < 0) {
            try assertEq(tensor1.buffer.values[arg_idx], -1);
        } else {
            try assertEq(tensor1.buffer.values[arg_idx], 0);
        }
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.binaryAdd(&tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], val1[arg_idx] + val2[arg_idx]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.binarySubtract(&tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], val1[arg_idx] - val2[arg_idx]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.binaryMultiply(&tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], val1[arg_idx] * val2[arg_idx]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.binaryDivide(&tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], val1[arg_idx] / val2[arg_idx]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.binaryMax(&tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], @max(val1[arg_idx], val2[arg_idx]));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.binaryMin(&tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], @min(val1[arg_idx], val2[arg_idx]));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.binarySet(&tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], val2[arg_idx]);
    }

    tensor2.moveResize(1, 1, 1, 1);
    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.expandAdd(&tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], val1[arg_idx] + val2[0]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.expandSubtract(&tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], val1[arg_idx] - val2[0]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.expandMultiply(&tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], val1[arg_idx] * val2[0]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.expandDivide(&tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], val1[arg_idx] / val2[0]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.expandMax(&tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], @max(val1[arg_idx], val2[0]));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.expandMin(&tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], @min(val1[arg_idx], val2[0]));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.expandSet(&tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assertEq(tensor1.buffer.values[arg_idx], val2[0]);
    }

    tensor1.moveResize(1, 1, 1, 1);
    tensor2.moveResize(a_size, z_size, y_size, x_size);
    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.reduceSum(&tensor2);
    tensor1.realize();
    var sum: f32 = 0;
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        sum += val2[arg_idx];
    }
    try assertEq(tensor1.buffer.values[0], sum);

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.reduceAvg(&tensor2);
    tensor1.realize();
    sum = 0;
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        sum += val2[arg_idx];
    }
    try assertEq(tensor1.buffer.values[0], sum / @as(f32, a_size * z_size * y_size * x_size));

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.reduceMax(&tensor2);
    tensor1.realize();
    var max: f32 = -std.math.inf(f32);
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        max = @max(max, val2[arg_idx]);
    }
    try assertEq(tensor1.buffer.values[0], max);

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    tensor1.reduceMin(&tensor2);
    tensor1.realize();
    var min: f32 = std.math.inf(f32);
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        min = @min(min, val2[arg_idx]);
    }
    try assertEq(tensor1.buffer.values[0], min);
}
