const std = @import("std");
const assert = std.debug.assert;
const Pcg = std.Random.Pcg;
const Allocator = std.mem.Allocator;

const grads = @import("grads");
const Tensor = grads.Tensor;
const OpType = grads.Op.Type;

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

fn simulateLinearized(allocator: Allocator, op_included: [op_num]bool, rng: u64) !void {
    assert(tensor_num > 1);
    assert(op_num > 0);

    var tensor1: [tensor_num]Tensor = undefined;
    var tensor2: [tensor_num]Tensor = undefined;

    const a_size_max: u32 = 7;
    const z_size_max: u32 = 6;
    const y_size_max: u32 = 5;
    const x_size_max: u32 = 4;

    // $TODO I should just precompute the op amounts so there are way less allocations
    for (0..tensor_num) |tensor_idx| {
        tensor1[tensor_idx] = try Tensor.alloc(allocator, a_size_max, z_size_max, y_size_max, x_size_max, null, 1);
        tensor2[tensor_idx] = try Tensor.alloc(allocator, a_size_max, z_size_max, y_size_max, x_size_max, null, 1);
    }
    defer {
        for (0..tensor_num) |tensor_idx| {
            tensor1[tensor_idx].free(allocator);
            tensor2[tensor_idx].free(allocator);
        }
    }

    var pcg = Pcg.init(rng);
    std.debug.print("simulate-compiler: rng={}...", .{rng});

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

        // Essentially free in case no alocattions are necessary
        try tensor1[tensor_out].linearized.capacityEnsure(allocator, 4 + tensor1[tensor_in].linearized.op_num);
        try tensor2[tensor_out].linearized.capacityEnsure(allocator, 4 + tensor2[tensor_in].linearized.op_num);
        try tensor1[tensor_in].linearized.capacityEnsure(allocator, 2);
        try tensor2[tensor_in].linearized.capacityEnsure(allocator, 2);

        if (tensor1[tensor_in].linearized.op_num != 0) {
            tensor1[tensor_out].linearized.concat(&tensor1[tensor_in].linearized);
        }

        if (!op_included[op_idx]) {
            continue;
        }

        if (op_type[op_idx].isReduce()) {
            tensor1[tensor_out].moveResize(1, 1, 1, 1);
            tensor2[tensor_out].moveResize(1, 1, 1, 1);
        } else {
            tensor1[tensor_out].moveResize(a_size, z_size, y_size, x_size);
            tensor2[tensor_out].moveResize(a_size, z_size, y_size, x_size);
        }
        if (op_type[op_idx].isExpand()) {
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

        switch (op_type[op_idx]) {
            .unary_add => {
                tensor1[tensor_out].unaryAdd(u_var);
                tensor2[tensor_out].unaryAdd(u_var);
                tensor2[tensor_out].realize();
            },
            .unary_subtract => {
                tensor1[tensor_out].unarySubtract(u_var);
                tensor2[tensor_out].unarySubtract(u_var);
                tensor2[tensor_out].realize();
            },
            .unary_multiply => {
                tensor1[tensor_out].unaryMultiply(u_var);
                tensor2[tensor_out].unaryMultiply(u_var);
                tensor2[tensor_out].realize();
            },
            .unary_divide => {
                tensor1[tensor_out].unaryDivide(@abs(u_var) + 1);
                tensor2[tensor_out].unaryDivide(@abs(u_var) + 1);
                tensor2[tensor_out].realize();
            },
            .unary_exp => {
                // NaN prevention
                tensor1[tensor_out].unaryMax(10);
                tensor2[tensor_out].unaryMax(10);
                tensor2[tensor_out].realize();
                tensor1[tensor_out].unaryMin(-10);
                tensor2[tensor_out].unaryMin(-10);
                tensor2[tensor_out].realize();

                tensor1[tensor_out].unaryExp();
                tensor2[tensor_out].unaryExp();
                tensor2[tensor_out].realize();
            },
            .unary_log => {
                // NaN prevention
                tensor1[tensor_out].unaryAbsolute();
                tensor2[tensor_out].unaryAbsolute();
                tensor2[tensor_out].realize();
                tensor1[tensor_out].unaryAdd(1);
                tensor2[tensor_out].unaryAdd(1);
                tensor2[tensor_out].realize();

                tensor1[tensor_out].unaryLog();
                tensor2[tensor_out].unaryLog();
                tensor2[tensor_out].realize();
            },
            .unary_square => {
                // Inf prevention
                tensor1[tensor_out].unaryMax(100);
                tensor2[tensor_out].unaryMax(100);
                tensor2[tensor_out].realize();
                tensor1[tensor_out].unaryMin(-100);
                tensor2[tensor_out].unaryMin(-100);
                tensor2[tensor_out].realize();

                tensor1[tensor_out].unarySquare();
                tensor2[tensor_out].unarySquare();
                tensor2[tensor_out].realize();
            },
            .unary_sqrt => {
                // NaN prevention
                tensor1[tensor_out].unaryAbsolute();
                tensor2[tensor_out].unaryAbsolute();
                tensor2[tensor_out].realize();

                tensor1[tensor_out].unarySqrt();
                tensor2[tensor_out].unarySqrt();
                tensor2[tensor_out].realize();
            },
            .unary_reciprocal => {
                // NaN prevention
                tensor1[tensor_out].unaryAbsolute();
                tensor2[tensor_out].unaryAbsolute();
                tensor2[tensor_out].realize();
                tensor1[tensor_out].unaryAdd(1);
                tensor2[tensor_out].unaryAdd(1);
                tensor2[tensor_out].realize();

                tensor1[tensor_out].unaryReciprocal();
                tensor2[tensor_out].unaryReciprocal();
                tensor2[tensor_out].realize();
            },
            .unary_max => {
                tensor1[tensor_out].unaryMax(u_var);
                tensor2[tensor_out].unaryMax(u_var);
                tensor2[tensor_out].realize();
            },
            .unary_min => {
                tensor1[tensor_out].unaryMin(u_var);
                tensor2[tensor_out].unaryMin(u_var);
                tensor2[tensor_out].realize();
            },
            .unary_set => {
                tensor1[tensor_out].unarySet(u_var);
                tensor2[tensor_out].unarySet(u_var);
                tensor2[tensor_out].realize();
            },
            .unary_random => {
                // $TODO This
                tensor1[tensor_out].unarySet(u_var);
                tensor2[tensor_out].unarySet(u_var);
                tensor2[tensor_out].realize();
            },
            .unary_tanh => {
                tensor1[tensor_out].unaryTanh();
                tensor2[tensor_out].unaryTanh();
                tensor2[tensor_out].realize();
            },
            .unary_absolute => {
                tensor1[tensor_out].unaryAbsolute();
                tensor2[tensor_out].unaryAbsolute();
                tensor2[tensor_out].realize();
            },
            .unary_sign => {
                tensor1[tensor_out].unarySign();
                tensor2[tensor_out].unarySign();
                tensor2[tensor_out].realize();
            },
            .binary_add => {
                tensor1[tensor_out].binaryAdd(&tensor1[tensor_in]);
                tensor2[tensor_out].binaryAdd(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_subtract => {
                tensor1[tensor_out].binarySubtract(&tensor1[tensor_in]);
                tensor2[tensor_out].binarySubtract(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_multiply => {
                tensor1[tensor_out].binaryMultiply(&tensor1[tensor_in]);
                tensor2[tensor_out].binaryMultiply(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_divide => {
                // NaN prevention
                tensor1[tensor_in].unaryAbsolute();
                tensor2[tensor_in].unaryAbsolute();
                tensor2[tensor_out].realize();
                tensor1[tensor_in].unaryAdd(1);
                tensor2[tensor_in].unaryAdd(1);
                tensor2[tensor_out].realize();

                tensor1[tensor_out].binaryDivide(&tensor1[tensor_in]);
                tensor2[tensor_out].binaryDivide(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_max => {
                tensor1[tensor_out].binaryMax(&tensor1[tensor_in]);
                tensor2[tensor_out].binaryMax(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_min => {
                tensor1[tensor_out].binaryMin(&tensor1[tensor_in]);
                tensor2[tensor_out].binaryMin(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_set => {
                tensor1[tensor_out].binarySet(&tensor1[tensor_in]);
                tensor2[tensor_out].binarySet(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .expand_add => {
                tensor1[tensor_out].expandAdd(&tensor1[tensor_in]);
                tensor2[tensor_out].expandAdd(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .expand_subtract => {
                tensor1[tensor_out].expandSubtract(&tensor1[tensor_in]);
                tensor2[tensor_out].expandSubtract(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .expand_multiply => {
                tensor1[tensor_out].expandMultiply(&tensor1[tensor_in]);
                tensor2[tensor_out].expandMultiply(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .expand_divide => {
                // NaN prevention
                tensor1[tensor_in].unaryAbsolute();
                tensor2[tensor_in].unaryAbsolute();
                tensor2[tensor_out].realize();
                tensor1[tensor_in].unaryAdd(1);
                tensor2[tensor_in].unaryAdd(1);
                tensor2[tensor_out].realize();

                tensor1[tensor_out].expandDivide(&tensor1[tensor_in]);
                tensor2[tensor_out].expandDivide(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .expand_max => {
                tensor1[tensor_out].expandMax(&tensor1[tensor_in]);
                tensor2[tensor_out].expandMax(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .expand_min => {
                tensor1[tensor_out].expandMin(&tensor1[tensor_in]);
                tensor2[tensor_out].expandMin(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .expand_set => {
                tensor1[tensor_out].expandSet(&tensor1[tensor_in]);
                tensor2[tensor_out].expandSet(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .reduce_sum => {
                tensor1[tensor_out].reduceSum(&tensor1[tensor_in]);
                tensor2[tensor_out].reduceSum(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .reduce_max => {
                tensor1[tensor_out].reduceMax(&tensor1[tensor_in]);
                tensor2[tensor_out].reduceMax(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .reduce_avg => {
                tensor1[tensor_out].reduceAvg(&tensor1[tensor_in]);
                tensor2[tensor_out].reduceAvg(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .reduce_min => {
                tensor1[tensor_out].reduceMin(&tensor1[tensor_in]);
                tensor2[tensor_out].reduceMin(&tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
        }
    }

    tensor1[op_out[op_num - 1]].realize();

    for (0..tensor_num) |tensor_idx| {
        for (0..a_size_max * z_size_max * y_size_max * x_size_max) |arg_idx| {
            try assertEq(tensor1[tensor_idx].buffer.values[arg_idx], tensor2[tensor_idx].buffer.values[arg_idx]);
        }
    }
    std.debug.print(" passed!\n", .{});
}

fn minifyLinearized(allocator: Allocator, rng: u64, err: anytype) !void {
    // $TODO Assert that the thing actually fails
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
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.detectLeaks();

    var args = try std.process.argsWithAllocator(allocator);
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
            simulateLinearized(allocator, @splat(true), rng +% loop_idx) catch |err| {
                try minifyLinearized(allocator, rng +% loop_idx, err);
            };
            loop_idx += 1;
        }
    } else {
        for (0..loop_count) |loop_idx| {
            std.debug.print("{} => ", .{loop_idx});
            simulateLinearized(allocator, @splat(true), rng +% loop_idx) catch |err| {
                try minifyLinearized(allocator, rng +% loop_idx, err);
            };
        }
    }
}
