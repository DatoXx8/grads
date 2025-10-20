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
        std.debug.print("\n", .{});
        std.log.err("Found NaN in equality comparison.\n", .{});
        return AssertError.nan;
    } else if (std.math.isInf(val1) or std.math.isInf(val2)) {
        std.debug.print("\n", .{});
        std.log.err("Found Inf in equality comparison.\n", .{});
        return AssertError.nan;
    } else if (std.math.approxEqAbs(f32, val1, val2, epsilon)) {
        return;
    } else {
        // For nicer output formatting
        std.debug.print("\n", .{});
        std.log.err("Difference between {d} and {d} is too large.\n", .{ val1, val2 });
        return AssertError.difference;
    }
}

const tensor_num = 10;
const op_num = 20;
comptime {
    assert(tensor_num > 1);
    assert(op_num > 0);
}

fn simulateLinearized(gpa: Allocator, op_included: [op_num]bool, rng: u64) !void {
    assert(tensor_num > 1);
    assert(op_num > 0);

    var arena_allocator: ArenaAllocator = .init(gpa);
    defer arena_allocator.deinit();
    const arena: Allocator = arena_allocator.allocator();

    var runtime_noop: RuntimeNoop = undefined;
    const runtime: Runtime = runtime_noop.runtime();

    var tensor1: [tensor_num]Buffer = undefined;
    var tensor2: [tensor_num]Buffer = undefined;
    var linearized1: Linearized = try .alloc(arena, 4 * op_num);
    var linearized2: Linearized = try .alloc(arena, 1);

    const a_size_max: u32 = 7;
    const z_size_max: u32 = 6;
    const y_size_max: u32 = 5;
    const x_size_max: u32 = 4;

    // $TODO I should just precompute the op amounts so there are way less allocations
    for (0..tensor_num) |tensor_idx| {
        tensor1[tensor_idx] = try Buffer.alloc(runtime, arena, a_size_max, z_size_max, y_size_max, x_size_max, .normal);
        tensor2[tensor_idx] = try Buffer.alloc(runtime, arena, a_size_max, z_size_max, y_size_max, x_size_max, .normal);
    }
    defer {
        for (0..tensor_num) |tensor_idx| {
            tensor1[tensor_idx].free(runtime);
            tensor2[tensor_idx].free(runtime);
        }
    }

    var pcg = Pcg.init(rng);
    std.debug.print("simulate_compiler: rng={}...", .{rng});

    for (0..tensor_num) |tensor_idx| {
        for (0..a_size_max * z_size_max * y_size_max * x_size_max) |arg_idx| {
            tensor1[tensor_idx].values[arg_idx] = pcg.random().floatNorm(f32);
            tensor2[tensor_idx].values[arg_idx] = tensor1[tensor_idx].values[arg_idx];
        }
    }

    var op_kind: [op_num]OpKind = undefined;
    var op_out: [op_num]u32 = undefined;
    var op_in: [op_num]u32 = undefined;

    for (0..op_num) |op_idx| {
        op_kind[op_idx] = pcg.random().enumValueWithIndex(OpKind, u32);

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

        const tensor_out: u32 = op_out[op_idx];
        const tensor_in: u32 = op_in[op_idx];

        if (!op_included[op_idx]) {
            continue;
        }

        if (op_kind[op_idx].isReduce()) {
            tensor1[tensor_out].moveResize(1, 1, 1, 1);
            tensor2[tensor_out].moveResize(1, 1, 1, 1);
        } else {
            tensor1[tensor_out].moveResize(a_size, z_size, y_size, x_size);
            tensor2[tensor_out].moveResize(a_size, z_size, y_size, x_size);
        }
        if (op_kind[op_idx].isExpand()) {
            tensor1[tensor_in].moveResize(1, 1, 1, 1);
            tensor2[tensor_in].moveResize(1, 1, 1, 1);
        } else {
            tensor1[tensor_in].moveResize(a_size, z_size, y_size, x_size);
            tensor2[tensor_in].moveResize(a_size, z_size, y_size, x_size);
        }

        tensor1[tensor_out].moveOffset(a_off, z_off, y_off, x_off);
        tensor2[tensor_out].moveOffset(a_off, z_off, y_off, x_off);
        tensor1[tensor_in].moveOffset(a_off, z_off, y_off, x_off);
        tensor2[tensor_in].moveOffset(a_off, z_off, y_off, x_off);

        switch (op_kind[op_idx]) {
            .unary_add => {
                linearized1.unaryAdd(tensor1[tensor_out], u_var);
                linearized2.unaryAdd(tensor2[tensor_out], u_var);
                linearized2.realize();
            },
            .unary_subtract => {
                linearized1.unarySubtract(tensor1[tensor_out], u_var);
                linearized2.unarySubtract(tensor2[tensor_out], u_var);
                linearized2.realize();
            },
            .unary_multiply => {
                linearized1.unaryMultiply(tensor1[tensor_out], u_var);
                linearized2.unaryMultiply(tensor2[tensor_out], u_var);
                linearized2.realize();
            },
            .unary_divide => {
                linearized1.unaryDivide(tensor1[tensor_out], @abs(u_var) + 1);
                linearized2.unaryDivide(tensor2[tensor_out], @abs(u_var) + 1);
                linearized2.realize();
            },
            .unary_exp => {
                // NaN prevention
                linearized1.unaryMax(tensor1[tensor_out], 10);
                linearized2.unaryMax(tensor2[tensor_out], 10);
                linearized2.realize();
                linearized1.unaryMin(tensor1[tensor_out], -10);
                linearized2.unaryMin(tensor2[tensor_out], -10);
                linearized2.realize();

                linearized1.unaryExp(tensor1[tensor_out]);
                linearized2.unaryExp(tensor2[tensor_out]);
                linearized2.realize();
            },
            .unary_log => {
                // NaN prevention
                linearized1.unaryAbsolute(tensor1[tensor_out]);
                linearized2.unaryAbsolute(tensor2[tensor_out]);
                linearized2.realize();
                linearized1.unaryAdd(tensor1[tensor_out], 1);
                linearized2.unaryAdd(tensor2[tensor_out], 1);
                linearized2.realize();

                linearized1.unaryLog(tensor1[tensor_out]);
                linearized2.unaryLog(tensor2[tensor_out]);
                linearized2.realize();
            },
            .unary_square => {
                // Inf prevention
                linearized1.unaryMax(tensor1[tensor_out], 100);
                linearized2.unaryMax(tensor2[tensor_out], 100);
                linearized2.realize();
                linearized1.unaryMin(tensor1[tensor_out], -100);
                linearized2.unaryMin(tensor2[tensor_out], -100);
                linearized2.realize();

                linearized1.unarySquare(tensor1[tensor_out]);
                linearized2.unarySquare(tensor2[tensor_out]);
                linearized2.realize();
            },
            .unary_sqrt => {
                // NaN prevention
                linearized1.unaryAbsolute(tensor1[tensor_out]);
                linearized2.unaryAbsolute(tensor2[tensor_out]);
                linearized2.realize();

                linearized1.unarySqrt(tensor1[tensor_out]);
                linearized2.unarySqrt(tensor2[tensor_out]);
                linearized2.realize();
            },
            .unary_reciprocal => {
                // NaN prevention
                linearized1.unaryAbsolute(tensor1[tensor_out]);
                linearized2.unaryAbsolute(tensor2[tensor_out]);
                linearized2.realize();
                linearized1.unaryAdd(tensor1[tensor_out], 1);
                linearized2.unaryAdd(tensor2[tensor_out], 1);
                linearized2.realize();

                linearized1.unaryReciprocal(tensor1[tensor_out]);
                linearized2.unaryReciprocal(tensor2[tensor_out]);
                linearized2.realize();
            },
            .unary_max => {
                linearized1.unaryMax(tensor1[tensor_out], u_var);
                linearized2.unaryMax(tensor2[tensor_out], u_var);
                linearized2.realize();
            },
            .unary_min => {
                linearized1.unaryMin(tensor1[tensor_out], u_var);
                linearized2.unaryMin(tensor2[tensor_out], u_var);
                linearized2.realize();
            },
            .unary_set => {
                linearized1.unarySet(tensor1[tensor_out], u_var);
                linearized2.unarySet(tensor2[tensor_out], u_var);
                linearized2.realize();
            },
            .unary_random => {
                // $TODO This
                linearized1.unarySet(tensor1[tensor_out], u_var);
                linearized2.unarySet(tensor2[tensor_out], u_var);
                linearized2.realize();
            },
            .unary_tanh => {
                linearized1.unaryTanh(tensor1[tensor_out]);
                linearized2.unaryTanh(tensor2[tensor_out]);
                linearized2.realize();
            },
            .unary_absolute => {
                linearized1.unaryAbsolute(tensor1[tensor_out]);
                linearized2.unaryAbsolute(tensor2[tensor_out]);
                linearized2.realize();
            },
            .unary_sign => {
                linearized1.unarySign(tensor1[tensor_out]);
                linearized2.unarySign(tensor2[tensor_out]);
                linearized2.realize();
            },
            .binary_add => {
                linearized1.binaryAdd(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.binaryAdd(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
            .binary_subtract => {
                linearized1.binarySubtract(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.binarySubtract(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
            .binary_multiply => {
                linearized1.binaryMultiply(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.binaryMultiply(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
            .binary_divide => {
                // NaN prevention
                linearized1.unaryAbsolute(tensor1[tensor_in]);
                linearized2.unaryAbsolute(tensor2[tensor_in]);
                linearized2.realize();
                linearized1.unaryAdd(tensor1[tensor_in], 1);
                linearized2.unaryAdd(tensor2[tensor_in], 1);
                linearized2.realize();

                linearized1.binaryDivide(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.binaryDivide(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
            .binary_max => {
                linearized1.binaryMax(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.binaryMax(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
            .binary_min => {
                linearized1.binaryMin(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.binaryMin(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
            .binary_set => {
                linearized1.binarySet(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.binarySet(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
            .expand_add => {
                linearized1.expandAdd(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.expandAdd(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
            .expand_subtract => {
                linearized1.expandSubtract(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.expandSubtract(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
            .expand_multiply => {
                linearized1.expandMultiply(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.expandMultiply(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
            .expand_divide => {
                // NaN prevention
                linearized1.unaryAbsolute(tensor1[tensor_in]);
                linearized2.unaryAbsolute(tensor2[tensor_in]);
                linearized2.realize();
                linearized1.unaryAdd(tensor1[tensor_in], 1);
                linearized2.unaryAdd(tensor2[tensor_in], 1);
                linearized2.realize();

                linearized1.expandDivide(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.expandDivide(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
            .expand_max => {
                linearized1.expandMax(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.expandMax(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
            .expand_min => {
                linearized1.expandMin(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.expandMin(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
            .expand_set => {
                linearized1.expandSet(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.expandSet(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
            .reduce_sum => {
                linearized1.reduceSum(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.reduceSum(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
            .reduce_max => {
                linearized1.reduceMax(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.reduceMax(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
            .reduce_avg => {
                linearized1.reduceAvg(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.reduceAvg(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
            .reduce_min => {
                linearized1.reduceMin(tensor1[tensor_out], tensor1[tensor_in]);
                linearized2.reduceMin(tensor2[tensor_out], tensor2[tensor_in]);
                linearized2.realize();
            },
        }
    }

    linearized1.realize();

    for (0..tensor_num) |tensor_idx| {
        for (0..a_size_max * z_size_max * y_size_max * x_size_max) |arg_idx| {
            try assertEq(tensor1[tensor_idx].values[arg_idx], tensor2[tensor_idx].values[arg_idx]);
        }
    }
    std.debug.print(" passed!\n", .{});
}

fn minifyLinearized(allocator: Allocator, rng: u64, err: anytype) !void {
    assert(tensor_num > 1);
    assert(op_num > 0);
    var op_included: [op_num]bool = @splat(true);
    for (0..op_num) |op_idx| {
        var failed: bool = false;
        op_included[op_idx] = false;
        simulateLinearized(allocator, op_included, rng) catch {
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
            unreachable;
        }
    }

    const rng: u64 = switch (rng_saved == null) {
        true => std.crypto.random.int(u64),
        false => rng_saved.?,
    };

    if (loop_infinite) {
        var loop_idx: u64 = 0;
        while (true) {
            std.debug.print("{} => ", .{loop_idx});
            simulateLinearized(gpa, @splat(true), rng +% loop_idx) catch |err| {
                try minifyLinearized(gpa, rng +% loop_idx, err);
            };
            loop_idx += 1;
        }
    } else {
        for (0..loop_count) |loop_idx| {
            std.debug.print("{} => ", .{loop_idx});
            simulateLinearized(gpa, @splat(true), rng +% loop_idx) catch |err| {
                try minifyLinearized(gpa, rng +% loop_idx, err);
            };
        }
    }
}
