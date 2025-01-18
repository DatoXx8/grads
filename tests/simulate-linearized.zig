const std = @import("std");
const Grads = @import("grads");

const Tensor = Grads.Tensor;
const OpType = Grads.Op.Type;
const Pcg = Grads.Pcg;

const assert = std.debug.assert;

// TODO: Move tests to seperate directory

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

const tensor_num: u32 = 10;
const op_num: u32 = 20;
comptime {
    assert(tensor_num > 1);
    assert(op_num > 0);
}

fn simulateLinearized(allocator: anytype, op_off_low: u32, op_off_top: u32, rng: u64) !void {
    assert(tensor_num > 1);
    assert(op_num > 0);
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
        tensor1[tensor_idx] = try Tensor.alloc(allocator, a_size_max, z_size_max, y_size_max, x_size_max, null);
        tensor2[tensor_idx] = try Tensor.alloc(allocator, a_size_max, z_size_max, y_size_max, x_size_max, null);
    }
    defer {
        for (0..tensor_num) |tensor_idx| {
            tensor1[tensor_idx].free(allocator);
            tensor2[tensor_idx].free(allocator);
        }
    }

    Pcg.init(rng);
    std.debug.print("simulate-compiler: rng={}...", .{rng});

    for (0..tensor_num) |tensor_idx| {
        for (0..a_size_max * z_size_max * y_size_max * x_size_max) |arg_idx| {
            tensor1[tensor_idx].buffer.values[arg_idx] = Pcg.randF32();
            tensor2[tensor_idx].buffer.values[arg_idx] = tensor1[tensor_idx].buffer.values[arg_idx];
        }
    }

    const op_type_max: u32 = @typeInfo(OpType).Enum.fields.len;
    var op_type: [op_num]OpType = undefined;
    var op_out: [op_num]u32 = undefined;
    var op_in: [op_num]u32 = undefined;

    for (0..op_num) |op_idx| {
        op_type[op_idx] = @enumFromInt(Pcg.randBelow(op_type_max));

        if (op_idx == 0) {
            op_out[0] = Pcg.randBelow(tensor_num);

            op_in[0] = Pcg.randBelow(tensor_num - 1);
            op_in[0] = if (op_in[0] < op_out[0]) op_in[0] else op_in[0] + 1;
            assert(op_out[0] != op_in[0]);
        } else {
            const switch_likelyhood: u32 = 10;
            if (Pcg.randBelow(switch_likelyhood) == 0) {
                op_in[op_idx] = op_out[op_idx - 1];
                op_out[op_idx] = Pcg.randBelow(tensor_num - 1);
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

    // TODO: Come up with a better name. This is basically the last op that isn't in the last loop
    for (0..op_num) |op_idx| {
        const a_size: u32 = Pcg.randBelow(a_size_max) + 1;
        const z_size: u32 = Pcg.randBelow(z_size_max) + 1;
        const y_size: u32 = Pcg.randBelow(y_size_max) + 1;
        const x_size: u32 = Pcg.randBelow(x_size_max) + 1;
        const a_off: u32 = Pcg.randBelow(a_size_max - a_size);
        const z_off: u32 = Pcg.randBelow(z_size_max - z_size);
        const y_off: u32 = Pcg.randBelow(y_size_max - y_size);
        const x_off: u32 = Pcg.randBelow(x_size_max - x_size);

        // Putting this here to make snycing the prng state trivial
        const u_var: f32 = Pcg.randF32();

        const tensor_out: u32 = op_out[op_idx];
        const tensor_in: u32 = op_in[op_idx];

        if (tensor1[tensor_in].linearized.op_num != 0) {
            try tensor1[tensor_out].linearized.concat(allocator, &tensor1[tensor_in].linearized);
        }

        if (op_idx < op_off_low or op_idx >= op_num - op_off_top) {
            continue;
        }

        if (op_type[op_idx].isReduce()) {
            tensor1[tensor_out].moveResize(1, 1, 1, 1);
            tensor2[tensor_out].moveResize(1, 1, 1, 1);
        } else {
            tensor1[tensor_out].moveResize(a_size, z_size, y_size, x_size);
            tensor2[tensor_out].moveResize(a_size, z_size, y_size, x_size);
        }
        if (op_type[op_idx].isLinary()) {
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
                try tensor1[tensor_out].unaryAdd(allocator, u_var);
                try tensor2[tensor_out].unaryAdd(allocator, u_var);
                tensor2[tensor_out].realize();
            },
            .unary_subtract => {
                try tensor1[tensor_out].unarySubtract(allocator, u_var);
                try tensor2[tensor_out].unarySubtract(allocator, u_var);
                tensor2[tensor_out].realize();
            },
            .unary_multiply => {
                try tensor1[tensor_out].unaryMultiply(allocator, u_var);
                try tensor2[tensor_out].unaryMultiply(allocator, u_var);
                tensor2[tensor_out].realize();
            },
            .unary_divide => {
                try tensor1[tensor_out].unaryDivide(allocator, @abs(u_var) + 1);
                try tensor2[tensor_out].unaryDivide(allocator, @abs(u_var) + 1);
                tensor2[tensor_out].realize();
            },
            .unary_exp => {
                // NaN prevention
                try tensor1[tensor_out].unaryMax(allocator, 10);
                try tensor2[tensor_out].unaryMax(allocator, 10);
                tensor2[tensor_out].realize();
                try tensor1[tensor_out].unaryMin(allocator, -10);
                try tensor2[tensor_out].unaryMin(allocator, -10);
                tensor2[tensor_out].realize();

                try tensor1[tensor_out].unaryExp(allocator);
                try tensor2[tensor_out].unaryExp(allocator);
                tensor2[tensor_out].realize();
            },
            .unary_log => {
                // NaN prevention
                try tensor1[tensor_out].unaryAbsolute(allocator);
                try tensor2[tensor_out].unaryAbsolute(allocator);
                tensor2[tensor_out].realize();
                try tensor1[tensor_out].unaryAdd(allocator, 1);
                try tensor2[tensor_out].unaryAdd(allocator, 1);
                tensor2[tensor_out].realize();

                try tensor1[tensor_out].unaryLog(allocator);
                try tensor2[tensor_out].unaryLog(allocator);
                tensor2[tensor_out].realize();
            },
            .unary_square => {
                // Inf prevention
                try tensor1[tensor_out].unaryMax(allocator, 100);
                try tensor2[tensor_out].unaryMax(allocator, 100);
                tensor2[tensor_out].realize();
                try tensor1[tensor_out].unaryMin(allocator, -100);
                try tensor2[tensor_out].unaryMin(allocator, -100);
                tensor2[tensor_out].realize();

                try tensor1[tensor_out].unarySquare(allocator);
                try tensor2[tensor_out].unarySquare(allocator);
                tensor2[tensor_out].realize();
            },
            .unary_sqrt => {
                // NaN prevention
                try tensor1[tensor_out].unaryAbsolute(allocator);
                try tensor2[tensor_out].unaryAbsolute(allocator);
                tensor2[tensor_out].realize();

                try tensor1[tensor_out].unarySqrt(allocator);
                try tensor2[tensor_out].unarySqrt(allocator);
                tensor2[tensor_out].realize();
            },
            .unary_reciprocal => {
                // NaN prevention
                try tensor1[tensor_out].unaryAbsolute(allocator);
                try tensor2[tensor_out].unaryAbsolute(allocator);
                tensor2[tensor_out].realize();
                try tensor1[tensor_out].unaryAdd(allocator, 1);
                try tensor2[tensor_out].unaryAdd(allocator, 1);
                tensor2[tensor_out].realize();

                try tensor1[tensor_out].unaryReciprocal(allocator);
                try tensor2[tensor_out].unaryReciprocal(allocator);
                tensor2[tensor_out].realize();
            },
            .unary_max => {
                try tensor1[tensor_out].unaryMax(allocator, u_var);
                try tensor2[tensor_out].unaryMax(allocator, u_var);
                tensor2[tensor_out].realize();
            },
            .unary_min => {
                try tensor1[tensor_out].unaryMin(allocator, u_var);
                try tensor2[tensor_out].unaryMin(allocator, u_var);
                tensor2[tensor_out].realize();
            },
            .unary_set => {
                try tensor1[tensor_out].unarySet(allocator, u_var);
                try tensor2[tensor_out].unarySet(allocator, u_var);
                tensor2[tensor_out].realize();
            },
            .unary_random => {
                // TODO: This
                try tensor1[tensor_out].unarySet(allocator, u_var);
                try tensor2[tensor_out].unarySet(allocator, u_var);
                tensor2[tensor_out].realize();
            },
            .unary_tanh => {
                try tensor1[tensor_out].unaryTanh(allocator);
                try tensor2[tensor_out].unaryTanh(allocator);
                tensor2[tensor_out].realize();
            },
            .unary_absolute => {
                try tensor1[tensor_out].unaryAbsolute(allocator);
                try tensor2[tensor_out].unaryAbsolute(allocator);
                tensor2[tensor_out].realize();
            },
            .unary_sign => {
                try tensor1[tensor_out].unarySign(allocator);
                try tensor2[tensor_out].unarySign(allocator);
                tensor2[tensor_out].realize();
            },
            .binary_add => {
                try tensor1[tensor_out].binaryAdd(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].binaryAdd(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_subtract => {
                try tensor1[tensor_out].binarySubtract(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].binarySubtract(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_multiply => {
                try tensor1[tensor_out].binaryMultiply(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].binaryMultiply(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_divide => {
                // NaN prevention
                try tensor1[tensor_in].unaryAbsolute(allocator);
                try tensor2[tensor_in].unaryAbsolute(allocator);
                tensor2[tensor_out].realize();
                try tensor1[tensor_in].unaryAdd(allocator, 1);
                try tensor2[tensor_in].unaryAdd(allocator, 1);
                tensor2[tensor_out].realize();

                try tensor1[tensor_out].binaryDivide(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].binaryDivide(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_max => {
                try tensor1[tensor_out].binaryMax(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].binaryMax(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_min => {
                try tensor1[tensor_out].binaryMin(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].binaryMin(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_set => {
                try tensor1[tensor_out].binarySet(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].binarySet(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .linary_add => {
                try tensor1[tensor_out].linaryAdd(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].linaryAdd(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .linary_subtract => {
                try tensor1[tensor_out].linarySubtract(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].linarySubtract(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .linary_multiply => {
                try tensor1[tensor_out].linaryMultiply(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].linaryMultiply(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .linary_divide => {
                // NaN prevention
                try tensor1[tensor_in].unaryAbsolute(allocator);
                try tensor2[tensor_in].unaryAbsolute(allocator);
                tensor2[tensor_out].realize();
                try tensor1[tensor_in].unaryAdd(allocator, 1);
                try tensor2[tensor_in].unaryAdd(allocator, 1);
                tensor2[tensor_out].realize();

                try tensor1[tensor_out].linaryDivide(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].linaryDivide(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .linary_max => {
                try tensor1[tensor_out].linaryMax(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].linaryMax(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .linary_min => {
                try tensor1[tensor_out].linaryMin(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].linaryMin(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .linary_set => {
                try tensor1[tensor_out].linarySet(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].linarySet(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .reduce_sum => {
                try tensor1[tensor_out].reduceSum(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].reduceSum(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .reduce_max => {
                try tensor1[tensor_out].reduceMax(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].reduceMax(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .reduce_avg => {
                try tensor1[tensor_out].reduceAvg(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].reduceAvg(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .reduce_min => {
                try tensor1[tensor_out].reduceMin(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].reduceMin(allocator, &tensor2[tensor_in]);
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

fn minifyLinearized(allocator: anytype, rng: u64, err: anytype) !void {
    // TODO: Assert that the thing actually fails
    assert(tensor_num > 1);
    assert(op_num > 0);
    var op_top: u32 = 0;
    for (1..op_num) |op_removed| {
        var failed: bool = false;
        simulateLinearized(allocator, 0, @truncate(op_removed), rng) catch {
            failed = true;
        };
        if (failed) {
            op_top = @truncate(op_removed);
        } else {
            break;
        }
    }
    var op_low: u32 = 0;
    for (1..op_num - op_top) |op_removed| {
        var failed: bool = false;
        simulateLinearized(allocator, @truncate(op_removed), op_top, rng) catch {
            failed = true;
        };
        if (failed) {
            op_low = @truncate(op_removed);
        } else {
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
    // Skip the executable call
    _ = args.next();
    if (args.next()) |arg| {
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
            assert(false);
        }
    }
    const rng: u64 = switch (rng_saved == null) {
        true => @bitCast(std.time.microTimestamp()),
        false => rng_saved.?,
    };

    if (loop_infinite) {
        var loop_idx: u64 = 0;
        // TODO: Decide how to reseed the random number generator here...
        // rng + loop_idx "wastes" the least seeds but it could cause issues
        // when running multiple threads with this because you then run the same tests over and over again
        while (true) {
            std.debug.print("{} => ", .{loop_idx});
            simulateLinearized(allocator, 0, 0, rng + loop_idx) catch |err| {
                try minifyLinearized(allocator, rng + loop_idx, err);
            };
            loop_idx += 1;
        }
    } else {
        for (0..loop_count) |loop_idx| {
            std.debug.print("{} => ", .{loop_idx});
            simulateLinearized(allocator, 0, 0, rng + loop_idx) catch |err| {
                try minifyLinearized(allocator, rng + loop_idx, err);
            };
        }
    }
}
