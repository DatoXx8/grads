const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Tensor = @import("./Tensor.zig");
const Linearized = Tensor.Linearized;
const Op = Tensor.Op;
const Buffer = Tensor.Buffer;
const todo = @import("./util.zig").todo;

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
// out_d = in_d / op_deriv(out)

// $TODO Maybe don't pass in an allocator and just assert there is enough space
/// Append op(s) to `d_in` that computes the derivative of `op` into `d_out`, with respect to `d_in`.
/// For unary ops `d_out` should be equal to `d_in`
/// Assumes d_out is zeroed appropriatly and the values in `out` where the ones in the forward pass and weren't changed from then.
/// Otherwise the gradients will be wrong.
pub fn differentiateOp(allocator: Allocator, op_type: Op.Type, op_u_var: f32, out: Tensor, d_out: Tensor, in: Tensor, d_in: Tensor) !void {
    if (op_type.isUnary()) {
        assert(d_out.buffer.equal(d_in.buffer));
        assert(out.buffer.equal(in.buffer));
    } else if (op_type.isBinary()) {
        assert(d_out.buffer.id != d_in.buffer.id);
        assert(d_out.buffer.a_size == d_in.buffer.a_size);
        assert(d_out.buffer.z_size == d_in.buffer.z_size);
        assert(d_out.buffer.y_size == d_in.buffer.y_size);
        assert(d_out.buffer.x_size == d_in.buffer.x_size);
        assert(out.buffer.id != in.buffer.id);
        assert(out.buffer.a_size == in.buffer.a_size);
        assert(out.buffer.z_size == in.buffer.z_size);
        assert(out.buffer.y_size == in.buffer.y_size);
        assert(out.buffer.x_size == in.buffer.x_size);
    } else if (op_type.isExpand()) {
        assert(d_out.buffer.id != d_in.buffer.id);
        assert(d_out.buffer.a_size == 1);
        assert(d_out.buffer.z_size == 1);
        assert(d_out.buffer.y_size == 1);
        assert(d_out.buffer.x_size == 1);
        assert(out.buffer.id != in.buffer.id);
        assert(out.buffer.a_size == 1);
        assert(out.buffer.z_size == 1);
        assert(out.buffer.y_size == 1);
        assert(out.buffer.x_size == 1);
    } else if (op_type.isReduce()) {
        assert(d_out.buffer.id != d_in.buffer.id);
        assert(d_in.buffer.a_size == 1);
        assert(d_in.buffer.z_size == 1);
        assert(d_in.buffer.y_size == 1);
        assert(d_in.buffer.x_size == 1);
        assert(out.buffer.id != in.buffer.id);
        assert(in.buffer.a_size == 1);
        assert(in.buffer.z_size == 1);
        assert(in.buffer.y_size == 1);
        assert(in.buffer.x_size == 1);
    } else {
        unreachable;
    }
    assert(out.buffer.overlapsAll(d_out.buffer));
    assert(in.buffer.overlapsAll(d_in.buffer));
    assert(out.buffer.id != d_out.buffer.id);
    assert(in.buffer.id != d_in.buffer.id);

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
        .unary_set => d_out.unarySet(0),
        .unary_random => d_out.unarySet(0),
        .unary_tanh => {
            d_out.binarySet(out);
            d_out.unarySquare();
            d_out.unaryMultiply(-1);
            d_out.unaryAdd(1);
            d_out.unaryReciprocal();
            d_out.binaryMultiply(d_in);
        },
        .unary_absolute => {
            d_out.binarySet(in);
            d_out.unarySign(); // Reciprocal is irrelevant here because the value are either 1 or -1
            d_out.binaryMultiply(d_in);
        },
        .unary_sign => d_out.unarySet(0), // At 0 this is undefined, so we'll just define it to be 0 like the rest of the values
        .binary_add => d_out.binarySet(d_in),
        .binary_subtract => d_out.binarySet(d_in), // $FIXME This needs to be negated, no?
        .binary_multiply => {
            d_out.binarySet(d_in);
            d_out.binaryDivide(in);
        },
        .binary_divide => {
            d_out.binarySet(d_in);
            d_out.binaryMultiply(in); // $TODO Test if doing a square thing is faster here
            d_out.binaryMultiply(in);
            d_out.unaryMultiply(-1);
        },
        .binary_max => todo(@src()),
        .binary_min => todo(@src()),
        .binary_set => d_out.binarySet(d_in),
        .expand_add => d_out.reduceSum(d_in),
        .expand_subtract => d_out.reduceSum(d_in), // $FIXME This needs to be negated, no?
        .expand_multiply => todo(@src()),
        .expand_divide => todo(@src()),
        .expand_max => todo(@src()),
        .expand_min => todo(@src()),
        .expand_set => d_out.reduceSum(d_in),
        .reduce_sum => d_out.expandSet(d_in),
        .reduce_avg => d_out.expandSet(d_in),
        .reduce_max => todo(@src()),
        .reduce_min => todo(@src()),
    }
}
