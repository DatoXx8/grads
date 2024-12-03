const std = @import("std");

const Tensor = @import("./tensor.zig").Tensor;
const OpType = @import("./tensor.zig").Op.Type;

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

// TODO: --loop option
// TODO: Minify automatically
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

    const a_size: u32 = 6;
    const z_size: u32 = 5;
    const y_size: u32 = 4;
    const x_size: u32 = 3;

    const tensor_num: u32 = 10;
    const op_num: u32 = 50;
    comptime {
        assert(tensor_num > 1);
        assert(tensor_num > 0);
    }

    // Arbitrary start points
    var tensor_out: u32 = 0;
    var tensor_in: u32 = 1;
    tensor_out = tensor_out;
    tensor_in = tensor_in;

    var tensor1: []Tensor = try allocator.alloc(Tensor, tensor_num);
    var tensor2: []Tensor = try allocator.alloc(Tensor, tensor_num);
    defer allocator.free(tensor1);
    defer allocator.free(tensor2);

    for (0..tensor_num) |tensor_idx| {
        tensor1[tensor_idx] = try Tensor.alloc(allocator, a_size, z_size, y_size, x_size, null);
        tensor2[tensor_idx] = try Tensor.alloc(allocator, a_size, z_size, y_size, x_size, null);
    }
    defer {
        for (0..tensor_num) |tensor_idx| {
            tensor1[tensor_idx].free(allocator);
            tensor2[tensor_idx].free(allocator);
        }
    }
    const rng: u64 = switch (rng_saved == null) {
        true => @bitCast(std.time.microTimestamp()),
        false => rng_saved.?,
    };
    std.debug.print("simulate-linearized: rng={}...", .{rng});
    Pcg.init(rng);

    for (0..tensor_num) |tensor_idx| {
        for (0..a_size * z_size * y_size * x_size) |arg_idx| {
            tensor1[tensor_idx].buffer.values[arg_idx] = Pcg.rand_f32();
            tensor2[tensor_idx].buffer.values[arg_idx] = tensor1[tensor_idx].buffer.values[arg_idx];
        }
    }

    const op_type_max: u32 = @typeInfo(OpType).Enum.fields.len;
    for (0..op_num) |_| {
        const op_type: OpType = @enumFromInt(Pcg.rand_below(op_type_max));

        // The likelyhood is strictly speaking 1 / switch_likelyhood
        const switch_likelyhood: u32 = 10;
        if (Pcg.rand_below(switch_likelyhood) == 0) {
            tensor_in = tensor_out;
            // At the very worst case this fails one out of 2^100 ~= 10^30 times
            const rand_tries_max: u32 = 100;
            for (0..rand_tries_max) |_| {
                tensor_out = Pcg.rand_below(tensor_num);
                if (tensor_out != tensor_in) {
                    break;
                }
            }
            assert(tensor_out != tensor_in);
        }

        // It is a bit difficult to explain why this is necessary, but essentially it prevents
        // the loss of ops that have already been executed on tensor2 (only the ones for NaN prevention
        // because those have effects on the in tensor) that could happen when switching ops and
        // not reattaching to the in tensor by getting a unary op.
        if (tensor1[tensor_in].linearized.op_num != 0) {
            try tensor1[tensor_out].linearized.concat(allocator, &tensor1[tensor_in].linearized);
        }

        // TODO: Random offsets
        switch (op_type) {
            .unary_add => {
                const u_var: f32 = Pcg.rand_f32();
                try tensor1[tensor_out].unary_add(allocator, u_var);
                try tensor2[tensor_out].unary_add(allocator, u_var);
                tensor2[tensor_out].realize();
            },
            .unary_subtract => {
                const u_var: f32 = Pcg.rand_f32();
                try tensor1[tensor_out].unary_subtract(allocator, u_var);
                try tensor2[tensor_out].unary_subtract(allocator, u_var);
                tensor2[tensor_out].realize();
            },
            .unary_multiply => {
                const u_var: f32 = Pcg.rand_f32();
                try tensor1[tensor_out].unary_multiply(allocator, u_var);
                try tensor2[tensor_out].unary_multiply(allocator, u_var);
                tensor2[tensor_out].realize();
            },
            .unary_divide => {
                const u_var: f32 = @abs(Pcg.rand_f32()) + 1;
                try tensor1[tensor_out].unary_divide(allocator, u_var);
                try tensor2[tensor_out].unary_divide(allocator, u_var);
                tensor2[tensor_out].realize();
            },
            .unary_exp => {
                try tensor1[tensor_out].unary_exp(allocator);
                try tensor2[tensor_out].unary_exp(allocator);
                tensor2[tensor_out].realize();
            },
            .unary_log => {
                try tensor1[tensor_out].unary_absolute(allocator);
                try tensor2[tensor_out].unary_absolute(allocator);
                tensor2[tensor_out].realize();
                try tensor1[tensor_out].unary_add(allocator, 1);
                try tensor2[tensor_out].unary_add(allocator, 1);
                tensor2[tensor_out].realize();
                try tensor1[tensor_out].unary_log(allocator);
                try tensor2[tensor_out].unary_log(allocator);
                tensor2[tensor_out].realize();
            },
            .unary_square => {
                try tensor1[tensor_out].unary_square(allocator);
                try tensor2[tensor_out].unary_square(allocator);
                tensor2[tensor_out].realize();
            },
            .unary_sqrt => {
                try tensor1[tensor_out].unary_absolute(allocator);
                try tensor2[tensor_out].unary_absolute(allocator);
                tensor2[tensor_out].realize();
                try tensor1[tensor_out].unary_sqrt(allocator);
                try tensor2[tensor_out].unary_sqrt(allocator);
                tensor2[tensor_out].realize();
            },
            .unary_reciprocal => {
                try tensor1[tensor_out].unary_absolute(allocator);
                try tensor2[tensor_out].unary_absolute(allocator);
                tensor2[tensor_out].realize();
                try tensor1[tensor_out].unary_add(allocator, 1);
                try tensor2[tensor_out].unary_add(allocator, 1);
                tensor2[tensor_out].realize();
                try tensor1[tensor_out].unary_reciprocal(allocator);
                try tensor2[tensor_out].unary_reciprocal(allocator);
                tensor2[tensor_out].realize();
            },
            .unary_max => {
                const u_var: f32 = Pcg.rand_f32();
                try tensor1[tensor_out].unary_max(allocator, u_var);
                try tensor2[tensor_out].unary_max(allocator, u_var);
                tensor2[tensor_out].realize();
            },
            .unary_min => {
                const u_var: f32 = Pcg.rand_f32();
                try tensor1[tensor_out].unary_min(allocator, u_var);
                try tensor2[tensor_out].unary_min(allocator, u_var);
                tensor2[tensor_out].realize();
            },
            .unary_set => {
                const u_var: f32 = Pcg.rand_f32();
                try tensor1[tensor_out].unary_set(allocator, u_var);
                try tensor2[tensor_out].unary_set(allocator, u_var);
                tensor2[tensor_out].realize();
            },
            .unary_random => {
                // Not doing this because I would have to reset the rng
                const u_var: f32 = Pcg.rand_f32();
                try tensor1[tensor_out].unary_set(allocator, u_var);
                try tensor2[tensor_out].unary_set(allocator, u_var);
                tensor2[tensor_out].realize();
            },
            .unary_tanh => {
                try tensor1[tensor_out].unary_tanh(allocator);
                try tensor2[tensor_out].unary_tanh(allocator);
                tensor2[tensor_out].realize();
            },
            .unary_absolute => {
                try tensor1[tensor_out].unary_absolute(allocator);
                try tensor2[tensor_out].unary_absolute(allocator);
                tensor2[tensor_out].realize();
            },
            .unary_sign => {
                try tensor1[tensor_out].unary_sign(allocator);
                try tensor2[tensor_out].unary_sign(allocator);
                tensor2[tensor_out].realize();
            },
            .binary_add => {
                try tensor1[tensor_out].binary_add(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].binary_add(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_subtract => {
                try tensor1[tensor_out].binary_subtract(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].binary_subtract(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_multiply => {
                std.debug.print("{} {}\n", .{ tensor_out, tensor_in });
                try tensor1[tensor_out].binary_multiply(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].binary_multiply(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_divide => {
                try tensor1[tensor_in].unary_absolute(allocator);
                try tensor2[tensor_in].unary_absolute(allocator);
                tensor2[tensor_in].realize();
                try tensor1[tensor_in].unary_add(allocator, 1);
                try tensor2[tensor_in].unary_add(allocator, 1);
                tensor2[tensor_in].realize();
                try tensor1[tensor_out].binary_divide(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].binary_divide(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_max => {
                try tensor1[tensor_out].binary_max(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].binary_max(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_min => {
                try tensor1[tensor_out].binary_min(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].binary_min(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .binary_set => {
                try tensor1[tensor_out].binary_set(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].binary_set(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
            },
            .linary_add => {
                tensor1[tensor_in].move_resize(1, 1, 1, 1);
                tensor2[tensor_in].move_resize(1, 1, 1, 1);
                try tensor1[tensor_out].linary_add(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].linary_add(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
                tensor1[tensor_in].move_resize(a_size, z_size, y_size, x_size);
                tensor2[tensor_in].move_resize(a_size, z_size, y_size, x_size);
            },
            .linary_subtract => {
                tensor1[tensor_in].move_resize(1, 1, 1, 1);
                tensor2[tensor_in].move_resize(1, 1, 1, 1);
                try tensor1[tensor_out].linary_subtract(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].linary_subtract(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
                tensor1[tensor_in].move_resize(a_size, z_size, y_size, x_size);
                tensor2[tensor_in].move_resize(a_size, z_size, y_size, x_size);
            },
            .linary_multiply => {
                tensor1[tensor_in].move_resize(1, 1, 1, 1);
                tensor2[tensor_in].move_resize(1, 1, 1, 1);
                try tensor1[tensor_out].linary_multiply(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].linary_multiply(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
                tensor1[tensor_in].move_resize(a_size, z_size, y_size, x_size);
                tensor2[tensor_in].move_resize(a_size, z_size, y_size, x_size);
            },
            .linary_divide => {
                tensor1[tensor_in].move_resize(1, 1, 1, 1);
                tensor2[tensor_in].move_resize(1, 1, 1, 1);
                try tensor1[tensor_in].unary_absolute(allocator);
                try tensor2[tensor_in].unary_absolute(allocator);
                tensor2[tensor_in].realize();
                try tensor1[tensor_in].unary_add(allocator, 1);
                try tensor2[tensor_in].unary_add(allocator, 1);
                tensor2[tensor_in].realize();
                try tensor1[tensor_out].linary_divide(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].linary_divide(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
                tensor1[tensor_in].move_resize(a_size, z_size, y_size, x_size);
                tensor2[tensor_in].move_resize(a_size, z_size, y_size, x_size);
            },
            .linary_max => {
                tensor1[tensor_in].move_resize(1, 1, 1, 1);
                tensor2[tensor_in].move_resize(1, 1, 1, 1);
                try tensor1[tensor_out].linary_max(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].linary_max(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
                tensor1[tensor_in].move_resize(a_size, z_size, y_size, x_size);
                tensor2[tensor_in].move_resize(a_size, z_size, y_size, x_size);
            },
            .linary_min => {
                tensor1[tensor_in].move_resize(1, 1, 1, 1);
                tensor2[tensor_in].move_resize(1, 1, 1, 1);
                try tensor1[tensor_out].linary_min(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].linary_min(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
                tensor1[tensor_in].move_resize(a_size, z_size, y_size, x_size);
                tensor2[tensor_in].move_resize(a_size, z_size, y_size, x_size);
            },
            .linary_set => {
                tensor1[tensor_in].move_resize(1, 1, 1, 1);
                tensor2[tensor_in].move_resize(1, 1, 1, 1);
                try tensor1[tensor_out].linary_set(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].linary_set(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
                tensor1[tensor_in].move_resize(a_size, z_size, y_size, x_size);
                tensor2[tensor_in].move_resize(a_size, z_size, y_size, x_size);
            },
            .reduce_sum => {
                tensor1[tensor_out].move_resize(1, 1, 1, 1);
                tensor2[tensor_out].move_resize(1, 1, 1, 1);
                try tensor1[tensor_out].reduce_sum(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].reduce_sum(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
                tensor1[tensor_out].move_resize(a_size, z_size, y_size, x_size);
                tensor2[tensor_out].move_resize(a_size, z_size, y_size, x_size);
            },
            .reduce_max => {
                tensor1[tensor_out].move_resize(1, 1, 1, 1);
                tensor2[tensor_out].move_resize(1, 1, 1, 1);
                try tensor1[tensor_out].reduce_max(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].reduce_max(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
                tensor1[tensor_out].move_resize(a_size, z_size, y_size, x_size);
                tensor2[tensor_out].move_resize(a_size, z_size, y_size, x_size);
            },
            .reduce_min => {
                tensor1[tensor_out].move_resize(1, 1, 1, 1);
                tensor2[tensor_out].move_resize(1, 1, 1, 1);
                try tensor1[tensor_out].reduce_min(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].reduce_min(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
                tensor1[tensor_out].move_resize(a_size, z_size, y_size, x_size);
                tensor2[tensor_out].move_resize(a_size, z_size, y_size, x_size);
            },
            .reduce_avg => {
                tensor1[tensor_out].move_resize(1, 1, 1, 1);
                tensor2[tensor_out].move_resize(1, 1, 1, 1);
                try tensor1[tensor_out].reduce_avg(allocator, &tensor1[tensor_in]);
                try tensor2[tensor_out].reduce_avg(allocator, &tensor2[tensor_in]);
                tensor2[tensor_out].realize();
                tensor1[tensor_out].move_resize(a_size, z_size, y_size, x_size);
                tensor2[tensor_out].move_resize(a_size, z_size, y_size, x_size);
            },
        }
    }

    tensor1[tensor_out].realize();
    for (0..a_size * z_size * y_size * x_size) |arg_idx| {
        try assert_eq(tensor1[tensor_out].buffer.values[arg_idx], tensor2[tensor_out].buffer.values[arg_idx]);
    }

    std.debug.print(" passed!\n", .{});
}
