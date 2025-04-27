const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const todo = @import("./util.zig").todo;

const Linearized = @import("./tensor.zig").Linearized;
const Op = @import("./tensor.zig").Op;
const Buffer = @import("./tensor.zig").Buffer;
const Tensor = @import("./tensor.zig").Tensor;

// u add a 1 f'(x)
// ... g'(x)
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
// f'(x) = (h^{-1})'(g(x)) * g'(x)
//
// g(x) = h(f(x))
// g'(x) = h'(f(x)) * f'(x)
// f'(x) = g'(x) / h'(f(x))

// $TODO Maybe don't pass in an allocator and just assert there is enough space
/// Append op(s) to `target` that computes the derivative of `op` into `d_out`, with respect to `d_in`.
/// For unary ops `d_out` should be equal to `d_in`
/// Assumes d_out is zeroed appropriatly and the values in `out` where the ones in the forward pass and weren't changed from then.
/// Otherwise the gradients will be wrong.
pub fn differentiateOp(allocator: Allocator, op_type: Op.Type, op_u_var: f32, out: Tensor, d_out: Tensor, d_in: Tensor) !void {
    if (op_type.isUnary()) {
        assert(d_out.buffer.equal(d_in.buffer));
    } else if (op_type.isBinary()) {
        assert(d_out.buffer.id != d_in.buffer.id);
        assert(d_out.buffer.a_size == d_in.buffer.a_size);
        assert(d_out.buffer.z_size == d_in.buffer.z_size);
        assert(d_out.buffer.y_size == d_in.buffer.y_size);
        assert(d_out.buffer.x_size == d_in.buffer.x_size);
    } else if (op_type.isExpand()) {
        assert(d_out.buffer.id != d_in.buffer.id);
        assert(d_out.buffer.a_size == 1);
        assert(d_out.buffer.z_size == 1);
        assert(d_out.buffer.y_size == 1);
        assert(d_out.buffer.x_size == 1);
    } else if (op_type.isReduce()) {
        assert(d_out.buffer.id != d_in.buffer.id);
        assert(d_in.buffer.a_size == 1);
        assert(d_in.buffer.z_size == 1);
        assert(d_in.buffer.y_size == 1);
        assert(d_in.buffer.x_size == 1);
    } else {
        unreachable;
    }
    assert(out.buffer.overlapsAll(d_out.buffer));

    if (op_type.isUnary() or op_type.isBinary() or op_type.isExpand() or op_type.isReduce()) {
        try d_out.linearized.capacityEnsure(allocator, d_in.linearized.op_num + 2);
    } else {
        unreachable;
    }

    // f'(x) = g'(x) / h'(f(x));
    switch (op_type) {
        .unary_add, .unary_subtract => {},
        .unary_multiply => d_out.unaryDivide(op_u_var),
        .unary_divide => d_out.unaryMultiply(op_u_var),
        .unary_exp => {
            d_out.binarySet(out);
            d_out.unaryExp();
            d_out.unaryReciprocal();
            d_out.binaryMultiply(d_in);
        },
        .unary_log => {
            d_out.binarySet(out);
            d_out.binaryMultiply(d_in);
        },
        .unary_square => {
            d_out.binarySet(out);
            d_out.unaryMultiply(2);
            d_out.unaryReciprocal();
            d_out.binaryMultiply(d_in);
        },
        .unary_sqrt => {
            d_out.binarySet(out);
            d_out.unarySqrt();
            d_out.unaryMultiply(2);
            d_out.binaryMultiply(d_in);
        },
        .unary_reciprocal => {
            d_out.binarySet(out);
            d_out.unarySquare();
            d_out.unaryMultiply(-1);
            d_out.binaryMultiply(d_in);
        },
        .unary_max => todo(@src()),
        .unary_min => todo(@src()),
        .unary_set => todo(@src()),
        .unary_random => todo(@src()),
        .unary_tanh => todo(@src()),
        .unary_absolute => todo(@src()),
        .unary_sign => todo(@src()),
        .binary_add => todo(@src()),
        .binary_subtract => todo(@src()),
        .binary_multiply => todo(@src()),
        .binary_divide => todo(@src()),
        .binary_max => todo(@src()),
        .binary_min => todo(@src()),
        .expand_add => todo(@src()),
        .expand_subtract => todo(@src()),
        .expand_multiply => todo(@src()),
        .expand_divide => todo(@src()),
        .expand_max => todo(@src()),
        .expand_min => todo(@src()),
        .expand_set => todo(@src()),
        .reduce_sum => todo(@src()),
        .reduce_avg => todo(@src()),
        .reduce_max => todo(@src()),
        .reduce_min => todo(@src()),
    }
}
