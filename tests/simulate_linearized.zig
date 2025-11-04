const std = @import("std");
const assert = std.debug.assert;
const Pcg = std.Random.Pcg;
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const grads = @import("grads");
const Linearized = grads.Linearized;
const Buffer = grads.Buffer;
const OpKind = grads.Op.Kind;
const Runtime = grads.Runtime;
const RuntimeNoop = grads.RuntimeNoop;
const util = grads.util;

const AssertError = error{
    nan,
    inf,
    difference,
};
/// Margin of error
const epsilon: f32 = 1e-9;
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
    } else if (std.math.approxEqAbs(f32, val1, val2, epsilon)) {
        return;
    } else {
        // For nicer output formatting
        util.log.print("\n", .{});
        std.log.err("Difference between {d} and {d} is too large.\n", .{ val1, val2 });
        return AssertError.difference;
    }
}

const buffer_num = 10;
const op_num = 20;
comptime {
    assert(buffer_num > 1);
    assert(op_num > 0);
}

fn simulateLinearized(gpa: Allocator, op_included: [op_num]bool, rng: u64) !void {
    assert(buffer_num > 1);
    assert(op_num > 0);

    var arena_allocator: ArenaAllocator = .init(gpa);
    defer arena_allocator.deinit();
    const arena: Allocator = arena_allocator.allocator();

    var runtime_noop: RuntimeNoop = undefined;
    const runtime: Runtime = runtime_noop.runtime();

    var buffer1: [buffer_num]Buffer = undefined;
    var buffer2: [buffer_num]Buffer = undefined;
    var linearized1: Linearized = try .alloc(arena, 4 * op_num);
    var linearized2: Linearized = try .alloc(arena, 1);

    const a_size_max: u32 = 7;
    const z_size_max: u32 = 6;
    const y_size_max: u32 = 5;
    const x_size_max: u32 = 4;

    // $TODO I should just precompute the op amounts so there are way less allocations
    for (0..buffer_num) |buffer_idx| {
        buffer1[buffer_idx] = try Buffer.alloc(runtime, arena, a_size_max, z_size_max, y_size_max, x_size_max, .normal);
        buffer2[buffer_idx] = try Buffer.alloc(runtime, arena, a_size_max, z_size_max, y_size_max, x_size_max, .normal);
    }
    defer {
        for (0..buffer_num) |buffer_idx| {
            buffer1[buffer_idx].free(runtime);
            buffer2[buffer_idx].free(runtime);
        }
    }

    var pcg = Pcg.init(rng);
    util.log.print("simulate_compiler: rng={}...", .{rng});

    for (0..buffer_num) |buffer_idx| {
        for (0..a_size_max * z_size_max * y_size_max * x_size_max) |arg_idx| {
            buffer1[buffer_idx].values[arg_idx] = pcg.random().floatNorm(f32);
            buffer2[buffer_idx].values[arg_idx] = buffer1[buffer_idx].values[arg_idx];
        }
    }

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

    for (0..op_num) |op_idx| {
        const a_size: u32 = pcg.random().uintLessThan(u32, a_size_max) + 1;
        const z_size: u32 = pcg.random().uintLessThan(u32, z_size_max) + 1;
        const y_size: u32 = pcg.random().uintLessThan(u32, y_size_max) + 1;
        const x_size: u32 = pcg.random().uintLessThan(u32, x_size_max) + 1;
        const a_off: u32 = if (a_size_max > a_size) pcg.random().uintLessThan(u32, a_size_max - a_size) else 0;
        const z_off: u32 = if (z_size_max > z_size) pcg.random().uintLessThan(u32, z_size_max - z_size) else 0;
        const y_off: u32 = if (y_size_max > y_size) pcg.random().uintLessThan(u32, y_size_max - y_size) else 0;
        const x_off: u32 = if (x_size_max > x_size) pcg.random().uintLessThan(u32, x_size_max - x_size) else 0;

        // Putting this here to make snycing the prng state trivial
        const u_var: f32 = pcg.random().floatNorm(f32);

        const buffer_out: u32 = op_out[op_idx];
        const buffer_in: u32 = op_in[op_idx];

        if (!op_included[op_idx]) {
            continue;
        }

        if (op_kind[op_idx].isReduce()) {
            buffer1[buffer_out].moveResize(1, 1, 1, 1);
            buffer2[buffer_out].moveResize(1, 1, 1, 1);
        } else {
            buffer1[buffer_out].moveResize(a_size, z_size, y_size, x_size);
            buffer2[buffer_out].moveResize(a_size, z_size, y_size, x_size);
        }
        if (op_kind[op_idx].isExpand()) {
            buffer1[buffer_in].moveResize(1, 1, 1, 1);
            buffer2[buffer_in].moveResize(1, 1, 1, 1);
        } else {
            buffer1[buffer_in].moveResize(a_size, z_size, y_size, x_size);
            buffer2[buffer_in].moveResize(a_size, z_size, y_size, x_size);
        }

        buffer1[buffer_out].moveOffset(a_off, z_off, y_off, x_off);
        buffer2[buffer_out].moveOffset(a_off, z_off, y_off, x_off);
        buffer1[buffer_in].moveOffset(a_off, z_off, y_off, x_off);
        buffer2[buffer_in].moveOffset(a_off, z_off, y_off, x_off);

        switch (op_kind[op_idx]) {
            .unary_add => {
                linearized1.unaryAdd(buffer1[buffer_out], u_var);
                linearized2.unaryAdd(buffer2[buffer_out], u_var);
                linearized2.realize();
            },
            .unary_subtract => {
                linearized1.unarySubtract(buffer1[buffer_out], u_var);
                linearized2.unarySubtract(buffer2[buffer_out], u_var);
                linearized2.realize();
            },
            .unary_multiply => {
                linearized1.unaryMultiply(buffer1[buffer_out], u_var);
                linearized2.unaryMultiply(buffer2[buffer_out], u_var);
                linearized2.realize();
            },
            .unary_divide => {
                linearized1.unaryDivide(buffer1[buffer_out], @abs(u_var) + 1);
                linearized2.unaryDivide(buffer2[buffer_out], @abs(u_var) + 1);
                linearized2.realize();
            },
            .unary_exp => {
                // NaN prevention
                linearized1.unaryMax(buffer1[buffer_out], 10);
                linearized2.unaryMax(buffer2[buffer_out], 10);
                linearized2.realize();
                linearized1.unaryMin(buffer1[buffer_out], -10);
                linearized2.unaryMin(buffer2[buffer_out], -10);
                linearized2.realize();

                linearized1.unaryExp(buffer1[buffer_out]);
                linearized2.unaryExp(buffer2[buffer_out]);
                linearized2.realize();
            },
            .unary_log => {
                // NaN prevention
                linearized1.unaryAbsolute(buffer1[buffer_out]);
                linearized2.unaryAbsolute(buffer2[buffer_out]);
                linearized2.realize();
                linearized1.unaryAdd(buffer1[buffer_out], 1);
                linearized2.unaryAdd(buffer2[buffer_out], 1);
                linearized2.realize();

                linearized1.unaryLog(buffer1[buffer_out]);
                linearized2.unaryLog(buffer2[buffer_out]);
                linearized2.realize();
            },
            .unary_square => {
                // Inf prevention
                linearized1.unaryMax(buffer1[buffer_out], 100);
                linearized2.unaryMax(buffer2[buffer_out], 100);
                linearized2.realize();
                linearized1.unaryMin(buffer1[buffer_out], -100);
                linearized2.unaryMin(buffer2[buffer_out], -100);
                linearized2.realize();

                linearized1.unarySquare(buffer1[buffer_out]);
                linearized2.unarySquare(buffer2[buffer_out]);
                linearized2.realize();
            },
            .unary_sqrt => {
                // NaN prevention
                linearized1.unaryAbsolute(buffer1[buffer_out]);
                linearized2.unaryAbsolute(buffer2[buffer_out]);
                linearized2.realize();

                linearized1.unarySqrt(buffer1[buffer_out]);
                linearized2.unarySqrt(buffer2[buffer_out]);
                linearized2.realize();
            },
            .unary_reciprocal => {
                // NaN prevention
                linearized1.unaryAbsolute(buffer1[buffer_out]);
                linearized2.unaryAbsolute(buffer2[buffer_out]);
                linearized2.realize();
                linearized1.unaryAdd(buffer1[buffer_out], 1);
                linearized2.unaryAdd(buffer2[buffer_out], 1);
                linearized2.realize();

                linearized1.unaryReciprocal(buffer1[buffer_out]);
                linearized2.unaryReciprocal(buffer2[buffer_out]);
                linearized2.realize();
            },
            .unary_max => {
                linearized1.unaryMax(buffer1[buffer_out], u_var);
                linearized2.unaryMax(buffer2[buffer_out], u_var);
                linearized2.realize();
            },
            .unary_min => {
                linearized1.unaryMin(buffer1[buffer_out], u_var);
                linearized2.unaryMin(buffer2[buffer_out], u_var);
                linearized2.realize();
            },
            .unary_set => {
                linearized1.unarySet(buffer1[buffer_out], u_var);
                linearized2.unarySet(buffer2[buffer_out], u_var);
                linearized2.realize();
            },
            .unary_random => {
                // $TODO This
                linearized1.unarySet(buffer1[buffer_out], u_var);
                linearized2.unarySet(buffer2[buffer_out], u_var);
                linearized2.realize();
            },
            .unary_tanh => {
                linearized1.unaryTanh(buffer1[buffer_out]);
                linearized2.unaryTanh(buffer2[buffer_out]);
                linearized2.realize();
            },
            .unary_absolute => {
                linearized1.unaryAbsolute(buffer1[buffer_out]);
                linearized2.unaryAbsolute(buffer2[buffer_out]);
                linearized2.realize();
            },
            .unary_sign => {
                linearized1.unarySign(buffer1[buffer_out]);
                linearized2.unarySign(buffer2[buffer_out]);
                linearized2.realize();
            },
            .binary_add => {
                linearized1.binaryAdd(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.binaryAdd(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
            .binary_subtract => {
                linearized1.binarySubtract(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.binarySubtract(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
            .binary_multiply => {
                linearized1.binaryMultiply(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.binaryMultiply(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
            .binary_divide => {
                // NaN prevention
                linearized1.unaryAbsolute(buffer1[buffer_in]);
                linearized2.unaryAbsolute(buffer2[buffer_in]);
                linearized2.realize();
                linearized1.unaryAdd(buffer1[buffer_in], 1);
                linearized2.unaryAdd(buffer2[buffer_in], 1);
                linearized2.realize();

                linearized1.binaryDivide(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.binaryDivide(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
            .binary_max => {
                linearized1.binaryMax(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.binaryMax(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
            .binary_min => {
                linearized1.binaryMin(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.binaryMin(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
            .binary_set => {
                linearized1.binarySet(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.binarySet(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
            .expand_add => {
                linearized1.expandAdd(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.expandAdd(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
            .expand_subtract => {
                linearized1.expandSubtract(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.expandSubtract(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
            .expand_multiply => {
                linearized1.expandMultiply(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.expandMultiply(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
            .expand_divide => {
                // NaN prevention
                linearized1.unaryAbsolute(buffer1[buffer_in]);
                linearized2.unaryAbsolute(buffer2[buffer_in]);
                linearized2.realize();
                linearized1.unaryAdd(buffer1[buffer_in], 1);
                linearized2.unaryAdd(buffer2[buffer_in], 1);
                linearized2.realize();

                linearized1.expandDivide(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.expandDivide(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
            .expand_max => {
                linearized1.expandMax(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.expandMax(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
            .expand_min => {
                linearized1.expandMin(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.expandMin(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
            .expand_set => {
                linearized1.expandSet(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.expandSet(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
            .reduce_sum => {
                linearized1.reduceSum(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.reduceSum(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
            .reduce_max => {
                linearized1.reduceMax(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.reduceMax(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
            .reduce_avg => {
                linearized1.reduceAvg(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.reduceAvg(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
            .reduce_min => {
                linearized1.reduceMin(buffer1[buffer_out], buffer1[buffer_in]);
                linearized2.reduceMin(buffer2[buffer_out], buffer2[buffer_in]);
                linearized2.realize();
            },
        }
    }

    linearized1.realize();

    for (0..buffer_num) |buffer_idx| {
        for (0..a_size_max * z_size_max * y_size_max * x_size_max) |arg_idx| {
            try assertEq(buffer1[buffer_idx].values[arg_idx], buffer2[buffer_idx].values[arg_idx]);
        }
    }
    util.log.print(" passed!\n", .{});
}

fn minifyLinearized(gpa: Allocator, rng: u64, err: anyerror) !void {
    assert(buffer_num > 1);
    assert(op_num > 0);
    var op_included: [op_num]bool = @splat(true);
    for (0..op_num) |op_idx| {
        var failed: bool = false;
        op_included[op_idx] = false;
        simulateLinearized(gpa, op_included, rng) catch {
            failed = true;
        };
        if (failed) {
            op_included[op_idx] = true;
        }
    }
    return err;
}

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = general_purpose_allocator.allocator();
    defer _ = general_purpose_allocator.detectLeaks();

    var args = try std.process.argsWithAllocator(gpa);
    defer args.deinit();

    var rng_saved: ?u64 = null;
    var loop_infinite: bool = false;
    var loop_count: u64 = 1;
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
        } else {
            std.log.err("Found unrecognised option `{s}`, expected `rng=<number>` or `loop=[number].\n", .{arg});
            @panic("");
        }
    }

    const rng: u64 = switch (rng_saved == null) {
        true => std.crypto.random.int(u64),
        false => rng_saved.?,
    };

    if (loop_infinite) {
        var loop_idx: u64 = 0;
        while (true) {
            util.log.print("{} => ", .{loop_idx});
            simulateLinearized(gpa, @splat(true), rng +% loop_idx) catch |err| {
                try minifyLinearized(gpa, rng +% loop_idx, err);
            };
            loop_idx += 1;
        }
    } else {
        for (0..loop_count) |loop_idx| {
            util.log.print("{} => ", .{loop_idx});
            simulateLinearized(gpa, @splat(true), rng +% loop_idx) catch |err| {
                try minifyLinearized(gpa, rng +% loop_idx, err);
            };
        }
    }
}
