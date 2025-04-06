const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.asserti;

const todo = @import("./util.zig").todo;

const Linearized = @import("./tensor.zig").Linearized;
const Op = @import("./tensor.zig").Op;
const Buffer = @import("./tensor.zig").Buffer;
const Tensor = @import("./tensor.zig").Tensor;

// u add a 1 f'(x)
// ... g'(x)
//
//
// g(x) = f(x) + 1
// f(x) = g(x) - 1
// f'(x) = g'(x)
//
// g(x) = f(x) / h(x)
// f(x) = g(x) * h(x)
// f'(x) = g'(x) * h(x) + g(x) * h'(x)
//
// g(x) = h(f(x)) assume h injective
// f(x) = h^{-1}(g(x))
// f'(x) = h^{-1}'(g(x)) * g'(x)

// $TODO Maybe don't pass in an allocator and just assert there is enough space
/// Append op(s) to `target` that computes the derivative of `op` into `d_out`, with respect to `d_in`.
/// For unary ops `d_out` should be equal to `d_in`
/// Assumes d_out is zeroed appropriatly.
pub fn differentiateOp(allocator: Allocator, op_type: Op.Type, op_u_var: f32, out: Tensor, in: Tensor, d_out: Tensor, d_in: Tensor, temp: ?Tensor) !void {
    if (op_type.isUnary()) {
        assert(d_out.equal(d_in));
    } else if (op_type.isBinary()) {
        assert(d_out.a_size == d_in.a_size);
        assert(d_out.z_size == d_in.z_size);
        assert(d_out.y_size == d_in.y_size);
        assert(d_out.x_size == d_in.x_size);
    } else if (op_type.isExpand()) {
        assert(d_out.a_size == 1);
        assert(d_out.z_size == 1);
        assert(d_out.y_size == 1);
        assert(d_out.x_size == 1);
    } else if (op_type.isReduce()) {
        assert(d_in.a_size == 1);
        assert(d_in.z_size == 1);
        assert(d_in.y_size == 1);
        assert(d_in.x_size == 1);
    } else {
        unreachable;
    }
    assert(out.overlapsAll(d_out));
    assert(in.overlapsAll(d_in));

    if (op_type.isReduce()) {
        try d_out.linearized.capacityEnsure(allocator, d_in.linearized.op_num + (d_in.buffer.a_size * d_in.buffer.z_size * d_in.buffer.y_size * d_in.buffer.x_size));
    } else if (op_type.isUnary() or op_type.isBinary() or op_type.isExpand()) {
        try d_out.linearized.capacityEnsure(allocator, d_in.linearized.op_num + 2);
    } else {
        unreachable;
    }

    switch (op_type) {
        .unary_add, .unary_subtract => d_out.binarySet(d_in),
        .unary_multiply => {
            d_out.binarySet(d_in);
            d_out.unaryDivide(op_u_var);
        },
        .unary_divide => {
            d_out.binarySet(d_in);
            d_out.unaryMultiply(op_u_var);
        },
        .unary_exp => {
            d_out.binarySet(d_in);
            d_out.unaryLog();
        },
        .unary_log => {
            d_out.binarySet(d_in);
            d_out.unaryExp();
        },
        .unary_square => {
            d_out.binarySet(d_in);
            d_out.unarySqrt();
        },
        .unary_sqrt => {
            d_out.binarySet(d_in);
            d_out.unarySquare();
        },
        .unary_reciprocal => {
            d_out.binarySet(d_in);
            d_out.unarySquare();
        },
        .unary_max => todo(@src()),
        .unary_min => todo(@src()),
        .unary_set => d_out.unarySet(0),
        .unary_random => d_out.unarySet(0),
        .unary_tanh => todo(@src()),
        .unary_absolute => todo(@src()),
        .unary_sign => todo(@src()),
        .binary_add => d_out.binarySubtract(d_in),
        .binary_subtract => d_out.binaryAdd(d_out),
        .binary_multiply => d_out.binaryDivide(d_out),
        .binary_divide => d_out.binaryMultiply(d_out),
        .binary_max => todo(@src()),
        .binary_min => todo(@src()),
        .expand_add => {
            if (temp) |t| {
                // $NOTE reduceAvg will provide better numerical stability
                t.reduceSum(d_in);
                d_out.binaryAdd(t);
            } else {
                std.log.err("Expected temporary buffer for differentiating `expand_add`, but got null\n", .{});
            }
        },
        .expand_subtract => {
            if (temp) |t| {
                t.reduceSum(d_in);
                d_out.binarySubtract(t);
            } else {
                std.log.err("Expected temporary buffer for differentiating `expand_subtract`, but got null\n", .{});
            }
        },
    }
}
