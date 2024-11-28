const std = @import("std");

const Tensor = @import("./tensor.zig").Tensor;

const Pcg = @import("./prng.zig").Pcg;

const assert = @import("./util.zig").assert;

// TODO: Move tests to seperate directory
// TODO: Make options to run with provided seed

const AssertError = error{
    nan,
    difference,
};
/// Margin of error
const epsilon: f32 = 1e-9;
/// Check for equality between the two floats within the margin of error of `epsilon`
fn assert_eq(val1: f32, val2: f32) !void {
    if (std.math.approxEqAbs(f32, val1, val2, epsilon)) {
        return;
    } else {
        if (std.math.isNan(val1) or std.math.isNan(val2)) {
            std.log.err("Found NaN in equality comparison.\n", .{});
            return AssertError.nan;
        } else {
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
    if (args.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "rng=")) {
            const offset = "rng="[0..].len;
            rng_saved = try std.fmt.parseInt(u64, arg[offset..], 10);
        } else {
            std.log.err("Found unrecognised option `{s}`, expected `rng=<number>` or nothing.\n", .{arg});
            assert(false);
        }
    }

    const a_size: u32 = 4;
    const z_size: u32 = 3;
    const y_size: u32 = 2;
    const x_size: u32 = 1;
    var tensor1 = try Tensor.alloc(allocator, a_size, z_size, y_size, x_size, null);
    defer tensor1.free(allocator);
    var tensor2 = try Tensor.alloc(allocator, a_size, z_size, y_size, x_size, null);
    defer tensor2.free(allocator);
    const val1 = try allocator.alloc(f32, a_size * z_size * y_size * x_size);
    defer allocator.free(val1);
    const val2 = try allocator.alloc(f32, a_size * z_size * y_size * x_size);
    defer allocator.free(val2);

    const rng: u64 = switch (rng_saved == null) {
        true => @bitCast(std.time.microTimestamp()),
        false => rng_saved.?,
    };
    std.debug.print("rng: {}\n", .{rng});
    Pcg.init(rng);

    for (0..a_size * z_size * y_size * x_size) |val_idx| {
        val1[val_idx] = Pcg.rand_f32();
        val2[val_idx] = Pcg.rand_f32();
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    std.mem.copyForwards(f32, tensor2.buffer.values, val2);
    try tensor1.unary_add(allocator, 2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], val1[arg_idx] + 2);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.unary_subtract(allocator, 2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], val1[arg_idx] - 2);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.unary_multiply(allocator, 2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], val1[arg_idx] * 2);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.unary_divide(allocator, 2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], val1[arg_idx] / 2);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.unary_absolute(allocator);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], @abs(val1[arg_idx]));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.unary_exp(allocator);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], std.math.exp(val1[arg_idx]));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    // This is to avoid NaNs
    try tensor1.unary_absolute(allocator);
    try tensor1.unary_add(allocator, 1);
    try tensor1.unary_log(allocator);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], std.math.log(f32, std.math.e, @abs(val1[arg_idx]) + 1));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.unary_square(allocator);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], val1[arg_idx] * val1[arg_idx]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    // This it to avoid NaNs
    try tensor1.unary_absolute(allocator);
    try tensor1.unary_sqrt(allocator);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], std.math.sqrt(@abs(val1[arg_idx])));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    // This is to avoid NaNs
    try tensor1.unary_absolute(allocator);
    try tensor1.unary_add(allocator, 1);
    try tensor1.unary_reciprocal(allocator);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], 1 / (@abs(val1[arg_idx]) + 1));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.unary_max(allocator, 1);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], @max(val1[arg_idx], 1));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.unary_min(allocator, 1);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], @min(val1[arg_idx], 1));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.unary_set(allocator, 2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], 2);
    }

    // Theoretically speaking because this *is* random this could
    // error despite everything working as it should, but doing some
    // back of the envelope calculations that should be rare enough to
    // not really worry about it.
    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.unary_random(allocator);
    tensor1.realize();
    var product: f32 = 1;
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        product *= tensor1.buffer.values[arg_idx];
    }
    if (@abs(product) < 1.0 / @as(f32, a_size * z_size * y_size * x_size)) {
        // Happy case
    } else {
        std.log.err(
            \\ Difference in unary_random too large. 
            \\Because of the randomnes of unary_random this error could happen despite everything working correctly and should only be investigated if it happens when running this test again.
            \\ The values where {d} and {d}.
        , .{ @abs(product), 1 / @as(f32, a_size * z_size * y_size * x_size) });
        return AssertError.difference;
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.unary_tanh(allocator);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], std.math.tanh(val1[arg_idx]));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.unary_sign(allocator);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        if (val1[arg_idx] > 0) {
            try assert_eq(tensor1.buffer.values[arg_idx], 1);
        } else if (val1[arg_idx] < 0) {
            try assert_eq(tensor1.buffer.values[arg_idx], -1);
        } else {
            try assert_eq(tensor1.buffer.values[arg_idx], 0);
        }
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.binary_add(allocator, &tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], val1[arg_idx] + val2[arg_idx]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.binary_subtract(allocator, &tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], val1[arg_idx] - val2[arg_idx]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.binary_multiply(allocator, &tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], val1[arg_idx] * val2[arg_idx]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.binary_divide(allocator, &tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], val1[arg_idx] / val2[arg_idx]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.binary_max(allocator, &tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], @max(val1[arg_idx], val2[arg_idx]));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.binary_min(allocator, &tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], @min(val1[arg_idx], val2[arg_idx]));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.binary_set(allocator, &tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], val2[arg_idx]);
    }

    tensor2.move_resize(1, 1, 1, 1);
    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.linary_add(allocator, &tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], val1[arg_idx] + val2[0]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.linary_subtract(allocator, &tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], val1[arg_idx] - val2[0]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.linary_multiply(allocator, &tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], val1[arg_idx] * val2[0]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.linary_divide(allocator, &tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], val1[arg_idx] / val2[0]);
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.linary_max(allocator, &tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], @max(val1[arg_idx], val2[0]));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.linary_min(allocator, &tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], @min(val1[arg_idx], val2[0]));
    }

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.linary_set(allocator, &tensor2);
    tensor1.realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1.buffer.values[arg_idx], val2[0]);
    }

    tensor1.move_resize(1, 1, 1, 1);
    tensor2.move_resize(a_size, z_size, y_size, x_size);
    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.reduce_sum(allocator, &tensor2);
    tensor1.realize();
    var sum: f32 = 0;
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        sum += val2[arg_idx];
    }
    try assert_eq(tensor1.buffer.values[0], sum);

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.reduce_avg(allocator, &tensor2);
    tensor1.realize();
    sum = 0;
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        sum += val2[arg_idx];
    }
    try assert_eq(tensor1.buffer.values[0], sum / @as(f32, a_size * z_size * y_size * x_size));

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.reduce_max(allocator, &tensor2);
    tensor1.realize();
    var max: f32 = -std.math.inf(f32);
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        max = @max(max, val2[arg_idx]);
    }
    try assert_eq(tensor1.buffer.values[0], max);

    std.mem.copyForwards(f32, tensor1.buffer.values, val1);
    try tensor1.reduce_min(allocator, &tensor2);
    tensor1.realize();
    var min: f32 = std.math.inf(f32);
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        min = @min(min, val2[arg_idx]);
    }
    try assert_eq(tensor1.buffer.values[0], min);
}
