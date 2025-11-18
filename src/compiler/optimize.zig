const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

const Linearized = @import("../Linearized.zig");
const Op = Linearized.Op;
const Buffer = @import("../Buffer.zig");
const Vec4 = Buffer.Vec4;
const Pir = @import("Pir.zig");
const Assign = Pir.Assign;
const Inlined = Pir.Inlined;
const Base = Pir.Base;
const ViewOffset = Pir.ViewOffset;
const util = @import("../util.zig");

/// Planned optimization steps
///  none
///  parallelize, inline, split, fuse ops, constant folding, swap
///  SIMD
///  memory optimizer
/// Remaining optimization steps
///  constant folding, swap
///  SIMD
///  memory optimizer
pub const Optimization = union(enum) {
    parallelize: struct {
        left_idx: u32,
        right_idx: u32,
    },
    inlined: struct {
        left_idx: u32,
        right_idx_max_written: u32,
    },
    split: struct {
        idx: u32,
    },
    fuse: struct {
        left_idx: u32,
        right_idx: u32,
    },
};

/// $WARN Inputs are assumed valid, i.e `sqrt(x)^2` for `x < 0` is assumed to never occur and will be optimized to `id(x)`
/// Check if left and right can be merged.
/// Assumes there is no useage of the out buffer of left between the two bases.
fn mergeOpPossible(left: Assign, right: Assign) bool {
    if (left.repeats != right.repeats) return false;
    if (!left.base.out_view.equal(right.base.out_view)) return false;
    if (!left.base.in_view.equal(right.base.in_view)) return false;
    if (!left.size.equal(right.size)) return false;
    if (left.base.out.id != right.base.out.id or left.base.in.id != right.base.in.id) return false;
    if (right.base.kind.isReduce()) return false;
    if (right.base.kind.overwrites()) return true;

    if (left.inlined.num > 0) return false;

    // $TODO This is a really inconvenient way of doing this. Right should be on the outside.
    return switch (left.base.kind) {
        .unary_add, .unary_subtract => return switch (right.base.kind) {
            .unary_add => true,
            .unary_subtract => true,
            else => false,
        },
        .unary_multiply, .unary_divide => return switch (right.base.kind) {
            .unary_multiply => true,
            .unary_divide => true,
            else => false,
        },
        .unary_random => return false,
        .unary_square => return switch (right.base.kind) {
            .unary_sqrt => true,
            .unary_absolute => true,
            else => false,
        },
        .unary_absolute => return switch (right.base.kind) {
            .unary_square => true,
            .unary_absolute => true,
            else => false,
        },
        .unary_sqrt => return right.base.kind == .unary_square,
        .unary_set => return false,
        .unary_exp => return switch (right.base.kind) {
            .unary_log => true,
            .unary_absolute => true,
            else => false,
        },
        .unary_log => return right.base.kind == .unary_exp,
        .unary_max => return right.base.kind == .unary_max,
        .unary_min => return right.base.kind == .unary_min,
        .unary_reciprocal => return switch (right.base.kind) {
            .unary_reciprocal => true,
            .unary_sign => true,
            else => false,
        },
        .unary_sign => return switch (right.base.kind) {
            .unary_reciprocal => true,
            .unary_sign => true,
            else => false,
        },
        .unary_tanh => false,
        .binary_add => return right.base.kind == .binary_subtract,
        .binary_subtract => return right.base.kind == .binary_add,
        .binary_multiply => return right.base.kind == .binary_divide,
        .binary_divide => return right.base.kind == .binary_multiply,
        .binary_max => return right.base.kind == .binary_max,
        .binary_min => return right.base.kind == .binary_min,
        .binary_set => false,
        .expand_add => return right.base.kind == .expand_subtract,
        .expand_subtract => return right.base.kind == .expand_add,
        .expand_multiply => return right.base.kind == .expand_divide,
        .expand_divide => return right.base.kind == .expand_multiply,
        .expand_max => return right.base.kind == .expand_max,
        .expand_min => return right.base.kind == .expand_min,
        .expand_set => false,
        // $TODO Rethink these
        .reduce_avg => false,
        .reduce_max => false,
        .reduce_min => false,
        .reduce_sum => false,
    };
}
/// Modifies `right`
/// Return wether both bases should be removed. Happens if `right(left(x)) == id(x)`
fn mergeOpCombine(left: Assign, right: *Assign) bool {
    assert(mergeOpPossible(left, right.*));

    if (right.base.kind.overwrites()) return false;

    const delete_both: bool = true;
    const delete_first: bool = false;

    // $TODO This is a really inconvenient way of doing this. Right should be on the outside.
    switch (left.base.kind) {
        // $TODO Don't know how I feel about this being a singular case
        .unary_add, .unary_subtract => {
            right.base.u_var = if (left.base.kind == right.base.kind)
                right.base.u_var + left.base.u_var
            else
                right.base.u_var - left.base.u_var;
        },
        // $TODO Don't know how I feel about this being a singular case
        .unary_multiply, .unary_divide => {
            right.base.u_var = if (left.base.kind == right.base.kind)
                right.base.u_var * left.base.u_var
            else
                right.base.u_var / left.base.u_var;
        },
        .unary_square => {
            right.base.kind = switch (right.base.kind) {
                .unary_sqrt => .unary_absolute, // sqrt(x^2) = |x|
                .unary_absolute => .unary_square, // |x^2| = x^2
                else => unreachable,
            };
        },
        .unary_absolute => {
            right.base.kind = switch (right.base.kind) {
                .unary_square => .unary_square, // |x|^2 = x^2
                .unary_absolute => .unary_absolute, // ||x|| = |x|
                else => unreachable,
            };
        },
        .unary_sqrt => switch (right.base.kind) {
            .unary_square => return delete_both, // sqrt(x)^2 = x by assumption of valid input
            else => unreachable,
        },
        .unary_exp => switch (right.base.kind) {
            .unary_log => return delete_both, // log_e(e^x) = id(x)
            .unary_absolute => {
                right.base.kind = .unary_exp; // |e^x| = e^x
            },
            else => unreachable,
        },
        .unary_log => switch (right.base.kind) {
            .unary_exp => return delete_both, // e^(log_e(x)) = id(x) by assumption of valid input
            else => unreachable,
        },
        .unary_max => switch (right.base.kind) {
            .unary_max => {
                right.base.u_var = @max(left.base.u_var, right.base.u_var); // max(max(x, a), b) = max(x, max(a, b))
            },
            else => unreachable,
        },
        .unary_min => switch (right.base.kind) {
            .unary_min => {
                right.base.u_var = @min(left.base.u_var, right.base.u_var); // min(min(x, a), b) = min(x, min(a, b))
            },
            else => unreachable,
        },
        .unary_reciprocal => switch (right.base.kind) {
            .unary_reciprocal => return delete_both, // 1 / (1 / x) = x assumes x != 0
            .unary_sign => {
                right.base.kind = .unary_sign; // sign(1 / x) = sign(x) assumes x != 0
            },
            else => unreachable,
        },
        .unary_sign => switch (right.base.kind) {
            .unary_reciprocal => {
                right.base.kind = .unary_sign; // 1 / sign(x) = sign(x) assumes x != 0
            },
            .unary_sign => {
                right.base.kind = .unary_sign; // sign(sign(x)) = sign(x)
            },
            else => unreachable,
        },
        .binary_add => switch (right.base.kind) {
            .binary_subtract => return delete_both,
            else => unreachable,
        },
        .binary_subtract => switch (right.base.kind) {
            .binary_add => return delete_both,
            else => unreachable,
        },
        .binary_multiply => switch (right.base.kind) {
            .binary_divide => return delete_both,
            else => unreachable,
        },
        .binary_divide => switch (right.base.kind) {
            .binary_multiply => return delete_both,
            else => unreachable,
        },
        .binary_max => {
            right.base.kind = switch (right.base.kind) {
                .binary_max => .binary_max,
                else => unreachable,
            };
        },
        .binary_min => {
            right.base.kind = switch (right.base.kind) {
                .binary_min => .binary_min,
                else => unreachable,
            };
        },
        .expand_add => switch (right.base.kind) {
            .expand_subtract => return delete_both,
            else => unreachable,
        },
        .expand_subtract => switch (right.base.kind) {
            .expand_add => return delete_both,
            else => unreachable,
        },
        .expand_multiply => switch (right.base.kind) {
            .expand_divide => return delete_both,
            else => unreachable,
        },
        .expand_divide => switch (right.base.kind) {
            .expand_multiply => return delete_both,
            else => unreachable,
        },
        .expand_max => {
            right.base.kind = switch (right.base.kind) {
                .expand_max => .expand_max,
                else => unreachable,
            };
        },
        .expand_min => {
            right.base.kind = switch (right.base.kind) {
                .expand_min => .expand_min,
                else => unreachable,
            };
        },
        else => unreachable,
    }
    return delete_first;
}
pub fn mergeOpGather(gpa: Allocator, optimization: *ArrayList(Optimization), pir: Pir) !void {
    var left_idx: u32 = 0;
    while (left_idx < pir.assign_num - 1) : (left_idx += 1) {
        var right_idx: u32 = left_idx + 1;
        while (right_idx < pir.assign_num) : (right_idx += 1) {
            if (mergeOpPossible(pir.assign[left_idx], pir.assign[right_idx])) {
                const fuse: Optimization = .{
                    .fuse = .{
                        .left_idx = left_idx,
                        .right_idx = right_idx,
                    },
                };
                optimization.appendBounded(fuse) catch {
                    try optimization.ensureTotalCapacity(gpa, @max(optimization.capacity * 2, 4)); // Just in case it somehow has a capacity of 0
                    try optimization.appendBounded(fuse);
                };
                break;
            } else {
                const assign_left: Assign = pir.assign[left_idx];
                const base_left: Base = pir.assign[left_idx].base;
                const assign_left_size_out: Vec4 = if (base_left.kind.isReduce())
                    .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                else
                    assign_left.size;
                const assign_left_size_in: Vec4 = if (base_left.kind.isExpand())
                    .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                else
                    assign_left.size;
                const inlined_left: Inlined = pir.assign[left_idx].inlined;

                const assign_right: Assign = pir.assign[right_idx];
                const base_right: Base = pir.assign[right_idx].base;
                const assign_right_size_out: Vec4 = if (base_right.kind.isReduce())
                    .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                else
                    assign_right.size;
                const assign_right_size_in: Vec4 = if (base_right.kind.isExpand())
                    .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                else
                    assign_right.size;
                const inlined_right: Inlined = pir.assign[right_idx].inlined;

                const overlap_out_x: bool = blk: {
                    if (base_left.out.id == base_right.out.id and
                        base_left.out_view.overlaps(1, assign_left_size_out, //
                            base_right.out_view, 1, assign_right_size_out))
                    {
                        break :blk true;
                    }
                    if (base_left.out.id == base_right.in.id and
                        base_left.out_view.overlaps(1, assign_left_size_out, //
                            base_right.in_view, 1, assign_right_size_in) and
                        !base_right.kind.isUnary())
                    {
                        break :blk true;
                    }
                    var inlined_right_idx: u32 = 0;
                    while (inlined_right_idx < inlined_right.num) : (inlined_right_idx += 1) {
                        const inlined_right_size_out: Vec4 = if (inlined_right.base[inlined_right_idx].kind.isReduce())
                            .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                        else
                            assign_right.size;
                        const inlined_right_size_in: Vec4 = if (inlined_right.base[inlined_right_idx].kind.isExpand())
                            .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                        else
                            assign_right.size;
                        if (base_left.out.id == inlined_right.base[inlined_right_idx].out.id and
                            base_left.out_view.overlaps(1, assign_left_size_out, //
                                inlined_right.base[inlined_right_idx].out_view, 1, inlined_right_size_out) and
                            inlined_right.out[inlined_right_idx] == null)
                        {
                            break :blk true;
                        }
                        if (base_left.out.id == inlined_right.base[inlined_right_idx].in.id and
                            base_left.out_view.overlaps(1, assign_left_size_out, //
                                inlined_right.base[inlined_right_idx].in_view, 1, inlined_right_size_in) and
                            !inlined_right.base[inlined_right_idx].kind.isUnary() and
                            inlined_right.in[inlined_right_idx] == null)
                        {
                            break :blk true;
                        }
                    }
                    break :blk false;
                };
                const overlap_x_out: bool = blk: {
                    if (base_left.in.id == base_right.out.id and
                        base_left.in_view.overlaps(1, assign_left_size_in, //
                            base_right.out_view, 1, assign_right_size_out))
                    {
                        break :blk true;
                    }
                    var inlined_left_idx: u32 = 0;
                    while (inlined_left_idx < inlined_left.num) : (inlined_left_idx += 1) {
                        const inlined_left_size_out: Vec4 = if (inlined_left.base[inlined_left_idx].kind.isReduce())
                            .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                        else
                            assign_left.size;
                        const inlined_left_size_in: Vec4 = if (inlined_left.base[inlined_left_idx].kind.isExpand())
                            .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                        else
                            assign_left.size;
                        if (inlined_left.base[inlined_left_idx].out.id == base_right.out.id and
                            inlined_left.base[inlined_left_idx].out_view.overlaps(1, inlined_left_size_out, //
                                base_right.out_view, 1, assign_right_size_out) and
                            inlined_left.out[inlined_left_idx] == null)
                        {
                            break :blk true;
                        }
                        if (inlined_left.base[inlined_left_idx].in.id == base_right.out.id and
                            inlined_left.base[inlined_left_idx].in_view.overlaps(1, inlined_left_size_in, //
                                base_right.out_view, 1, assign_right_size_out) and
                            inlined_left.in[inlined_left_idx] == null and
                            !inlined_left.base[inlined_left_idx].kind.isUnary())
                        {
                            break :blk true;
                        }
                    }
                    break :blk false;
                };

                if (overlap_out_x or overlap_x_out) {
                    break;
                }
            }
        }
    }
}
pub fn mergeOp(gpa: Allocator, pir: *Pir, left_idx: u32, right_idx: u32) void {
    assert(mergeOpPossible(pir.assign[left_idx], pir.assign[right_idx]));
    const merge_both: bool = mergeOpCombine(pir.assign[left_idx], &pir.assign[right_idx]);

    var assign_num_new: u32 = 0;
    for (0..pir.assign_num) |assign_idx| {
        if (assign_idx == left_idx or (assign_idx == right_idx and merge_both)) {
            gpa.free(pir.assign[assign_idx].inlined.base);
            gpa.free(pir.assign[assign_idx].inlined.out);
            gpa.free(pir.assign[assign_idx].inlined.in);
        } else {
            pir.assign[assign_num_new] = pir.assign[assign_idx];
            assign_num_new += 1;
        }
    }
    assert(assign_num_new > 0);
    pir.assign_num = assign_num_new;
}

pub fn inlineOpGather(gpa: Allocator, optimization: *ArrayList(Optimization), pir: Pir) !void {
    var left_idx: u32 = 0;
    left_loop: while (left_idx < pir.assign_num - 1) : (left_idx += 1) {
        const assign_left: Assign = pir.assign[left_idx];
        const base_left: Base = pir.assign[left_idx].base;
        const assign_left_size_out: Vec4 = if (base_left.kind.isReduce())
            .{ .a = 1, .z = 1, .y = 1, .x = 1 }
        else
            pir.assign[left_idx].size;
        const assign_left_size_in: Vec4 = if (base_left.kind.isExpand())
            .{ .a = 1, .z = 1, .y = 1, .x = 1 }
        else
            pir.assign[left_idx].size;
        const inlined_left: Inlined = pir.assign[left_idx].inlined;

        if (base_left.kind.isReduce()) {
            continue :left_loop;
        }

        var inlined_valid: bool = true;
        var right_idx_max_written: u32 = left_idx;

        var right_idx: u32 = left_idx + 1;
        right_loop: while (right_idx < pir.assign_num) : (right_idx += 1) {
            assert(inlined_valid);

            const assign_right: Assign = pir.assign[right_idx];
            const base_right: Base = pir.assign[right_idx].base;
            const assign_right_size_out: Vec4 = if (base_right.kind.isReduce())
                .{ .a = 1, .z = 1, .y = 1, .x = 1 }
            else
                pir.assign[right_idx].size;
            const assign_right_size_in: Vec4 = if (base_right.kind.isExpand())
                .{ .a = 1, .z = 1, .y = 1, .x = 1 }
            else
                pir.assign[right_idx].size;
            const inlined_right: Inlined = pir.assign[right_idx].inlined;

            // $TODO This should be more optimized
            const repeats_different: bool = assign_left.repeats != assign_right.repeats;
            const split_different: bool = false;
            // const split_different: bool = pir.assign[left_idx].split != pir.assign[right_idx].split;

            // $FIXME All of this is probably completely broken if the ViewOffset repeat info is different.
            //  In fact basically all of these are gonna break I think
            const left_out_overwritten: bool = blk: {
                if (base_left.out.id == base_right.out.id and
                    base_left.out_view.overlapsAll(1, assign_left_size_out, //
                        base_right.out_view, 1, assign_right_size_out) and
                    base_right.kind.overwrites())
                {
                    break :blk true;
                }
                var inlined_right_idx: u32 = 0;
                while (inlined_right_idx < inlined_right.num) : (inlined_right_idx += 1) {
                    const inlined_right_size_out: Vec4 = if (inlined_right.base[inlined_right_idx].kind.isReduce())
                        .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                    else
                        assign_right.size;
                    if (base_left.out.id == inlined_right.base[inlined_right_idx].out.id and
                        base_left.out_view.overlapsAll(1, assign_left_size_out, //
                            inlined_right.base[inlined_right_idx].out_view, 1, inlined_right_size_out) and
                        inlined_right.base[inlined_right_idx].kind.overwrites() and
                        inlined_right.out[inlined_right_idx] == null) // This part is probably not necessary
                    {
                        break :blk true;
                    }
                }
                break :blk false;
            };
            const left_out_non_intermdiary_inlined: bool = blk: {
                const base_left_out_intermediary: bool = switch (base_left.out.data().*.kind) {
                    .intermediary => true,
                    .normal => false,
                    .free => unreachable,
                };
                if (base_left.out.id == base_right.in.id and
                    base_left.out_view.overlapsAll(1, assign_left_size_out, //
                        base_right.in_view, 1, assign_right_size_in) and
                    !base_left_out_intermediary and
                    !base_right.kind.isUnary())
                {
                    break :blk true;
                }
                var inlined_right_idx: u32 = 0;
                while (inlined_right_idx < inlined_right.num) : (inlined_right_idx += 1) {
                    const inlined_right_size_in: Vec4 = if (inlined_right.base[inlined_right_idx].kind.isExpand())
                        .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                    else
                        assign_right.size;
                    if (base_left.out.id == inlined_right.base[inlined_right_idx].in.id and
                        base_left.out_view.overlapsAll(1, assign_left_size_out, //
                            inlined_right.base[inlined_right_idx].in_view, 1, inlined_right_size_in) and
                        !base_left_out_intermediary and
                        !inlined_right.base[inlined_right_idx].kind.isUnary() and
                        inlined_right.in[inlined_right_idx] == null) // This part is probably not necessary
                    {
                        break :blk true;
                    }
                }
                break :blk false;
            };
            const partial_overlap_out_x: bool = blk: {
                if (base_left.out.id == base_right.out.id and
                    base_left.out_view.overlapsPartial(1, assign_left_size_out, //
                        base_right.out_view, 1, assign_right_size_out))
                {
                    break :blk true;
                }
                if (base_left.out.id == base_right.in.id and
                    base_left.out_view.overlapsPartial(1, assign_left_size_out, //
                        base_right.in_view, 1, assign_right_size_in) and
                    !base_right.kind.isUnary())
                {
                    break :blk true;
                }
                var inlined_right_idx: u32 = 0;
                while (inlined_right_idx < inlined_right.num) : (inlined_right_idx += 1) {
                    const inlined_right_size_out: Vec4 = if (inlined_right.base[inlined_right_idx].kind.isReduce())
                        .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                    else
                        assign_right.size;
                    const inlined_right_size_in: Vec4 = if (inlined_right.base[inlined_right_idx].kind.isExpand())
                        .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                    else
                        assign_right.size;
                    if (base_left.out.id == inlined_right.base[inlined_right_idx].out.id and
                        base_left.out_view.overlapsPartial(1, assign_left_size_out, //
                            inlined_right.base[inlined_right_idx].out_view, 1, inlined_right_size_out) and
                        inlined_right.out[inlined_right_idx] == null)
                    {
                        break :blk true;
                    }
                    if (base_left.out.id == inlined_right.base[inlined_right_idx].in.id and
                        base_left.out_view.overlapsPartial(1, assign_left_size_out, //
                            inlined_right.base[inlined_right_idx].in_view, 1, inlined_right_size_in) and
                        !inlined_right.base[inlined_right_idx].kind.isUnary() and
                        inlined_right.in[inlined_right_idx] == null)
                    {
                        break :blk true;
                    }
                }
                break :blk false;
            };
            const partial_overlap_x_out: bool = blk: {
                if (base_left.in.id == base_right.out.id and
                    base_left.in_view.overlapsPartial(1, assign_left_size_in, //
                        base_right.out_view, 1, assign_right_size_out))
                {
                    break :blk true;
                }
                var inlined_left_idx: u32 = 0;
                while (inlined_left_idx < inlined_left.num) : (inlined_left_idx += 1) {
                    const inlined_left_size_out: Vec4 = if (inlined_left.base[inlined_left_idx].kind.isReduce())
                        .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                    else
                        assign_left.size;
                    const inlined_left_size_in: Vec4 = if (inlined_left.base[inlined_left_idx].kind.isExpand())
                        .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                    else
                        assign_left.size;
                    if (inlined_left.base[inlined_left_idx].out.id == base_right.out.id and
                        inlined_left.base[inlined_left_idx].out_view.overlapsPartial(1, inlined_left_size_out, //
                            base_right.out_view, 1, assign_right_size_out) and
                        inlined_left.out[inlined_left_idx] == null)
                    {
                        break :blk true;
                    }
                    if (inlined_left.base[inlined_left_idx].in.id == base_right.out.id and
                        inlined_left.base[inlined_left_idx].in_view.overlapsPartial(1, inlined_left_size_in, //
                            base_right.out_view, 1, assign_right_size_out) and
                        inlined_left.in[inlined_left_idx] == null and
                        !inlined_left.base[inlined_left_idx].kind.isUnary())
                    {
                        break :blk true;
                    }
                }
                break :blk false;
            };
            const left_in_written: bool = blk: {
                if (base_left.in.id == base_right.out.id and
                    !base_left.kind.isUnary() and
                    base_left.in_view.overlapsAll(1, assign_left_size_in, //
                        base_right.out_view, 1, assign_right_size_out))
                {
                    break :blk true;
                }
                var inlined_left_idx: u32 = 0;
                while (inlined_left_idx < inlined_left.num) : (inlined_left_idx += 1) {
                    const inlined_left_size_in: Vec4 = if (inlined_left.base[inlined_left_idx].kind.isExpand())
                        .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                    else
                        assign_left.size;
                    if (inlined_left.base[inlined_left_idx].in.id == base_right.out.id and
                        inlined_left.base[inlined_left_idx].in_view.overlapsAll(1, inlined_left_size_in, //
                            base_right.out_view, 1, assign_right_size_out) and
                        inlined_left.in[inlined_left_idx] == null and
                        !inlined_left.base[inlined_left_idx].kind.isUnary())
                    {
                        break :blk true;
                    }
                }
                break :blk false;
            };
            if (repeats_different or split_different or left_out_non_intermdiary_inlined or
                partial_overlap_out_x or partial_overlap_x_out or left_in_written)
            {
                inlined_valid = false;
                break :right_loop;
            }
            if (left_out_overwritten) {
                break :right_loop;
            }
            const left_out_written_to_in: bool = blk: {
                if (base_left.out.id == base_right.in.id and
                    base_left.out_view.overlapsAll(1, assign_left_size_out, //
                        base_right.in_view, 1, assign_right_size_in) and
                    !base_right.kind.isUnary())
                {
                    break :blk true;
                }
                var inlined_right_idx: u32 = 0;
                while (inlined_right_idx < inlined_right.num) : (inlined_right_idx += 1) {
                    const inlined_right_size_in: Vec4 = if (inlined_right.base[inlined_right_idx].kind.isExpand())
                        .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                    else
                        assign_right.size;
                    if (base_left.out.id == inlined_right.base[inlined_right_idx].in.id and
                        base_left.out_view.overlapsAll(1, assign_right_size_out, //
                            inlined_right.base[inlined_right_idx].in_view, 1, inlined_right_size_in) and
                        !inlined_right.base[inlined_right_idx].kind.isUnary() and
                        inlined_right.in[inlined_right_idx] == null)
                    {
                        break :blk true;
                    }
                }
                break :blk false;
            };
            const left_out_written_to_out: bool = blk: {
                if (base_left.out.id == base_right.out.id and
                    base_left.out_view.overlapsAll(1, assign_left_size_out, //
                        base_right.out_view, 1, assign_right_size_out))
                {
                    break :blk true;
                }
                var inlined_right_idx: u32 = 0;
                while (inlined_right_idx < inlined_right.num) : (inlined_right_idx += 1) {
                    const inlined_right_size_out: Vec4 = if (inlined_right.base[inlined_right_idx].kind.isReduce())
                        .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                    else
                        assign_right.size;
                    if (base_left.out.id == inlined_right.base[inlined_right_idx].out.id and
                        base_left.out_view.overlapsAll(1, assign_left_size_out, //
                            inlined_right.base[inlined_right_idx].out_view, 1, inlined_right_size_out) and
                        inlined_right.out[inlined_right_idx] == null)
                    {
                        break :blk true;
                    }
                }
                break :blk false;
            };
            if (left_out_written_to_in) {
                right_idx_max_written = right_idx;
            }
            if (left_out_written_to_out) {
                right_idx_max_written = right_idx;
                break :right_loop;
            }
        }

        if (inlined_valid and right_idx_max_written != left_idx) {
            const inlined: Optimization = .{
                .inlined = .{
                    .left_idx = left_idx,
                    .right_idx_max_written = right_idx_max_written,
                },
            };
            optimization.appendBounded(inlined) catch {
                try optimization.ensureTotalCapacity(gpa, @max(optimization.capacity * 2, 4)); // Just in case it somehow has a capacity of 0
                try optimization.appendBounded(inlined);
            };
        }
    }
}
pub fn inlineOp(gpa: Allocator, pir: *Pir, left_idx: u32, right_idx_max_written: u32) !void {
    assert(left_idx < right_idx_max_written);
    assert(right_idx_max_written < pir.assign_num);

    const base_left: Base = pir.assign[left_idx].base;
    const assign_left_size_out: Vec4 = if (base_left.kind.isReduce())
        .{ .a = 1, .z = 1, .y = 1, .x = 1 }
    else
        pir.assign[left_idx].size;
    const inlined_left: Inlined = pir.assign[left_idx].inlined;

    var right_idx: u32 = left_idx + 1;
    right_loop: while (right_idx <= right_idx_max_written) : (right_idx += 1) {
        const assign_right: Assign = pir.assign[right_idx];
        const base_right: Base = pir.assign[right_idx].base;
        const assign_right_size_out: Vec4 = if (base_right.kind.isReduce())
            .{ .a = 1, .z = 1, .y = 1, .x = 1 }
        else
            pir.assign[right_idx].size;
        const assign_right_size_in: Vec4 = if (base_right.kind.isExpand())
            .{ .a = 1, .z = 1, .y = 1, .x = 1 }
        else
            pir.assign[right_idx].size;
        const inlined_right: *Inlined = &pir.assign[right_idx].inlined;

        const inlined_right_num_initial: u32 = inlined_right.*.num;

        if (base_left.out.id == base_right.out.id and base_left.out_view.overlaps(1, assign_left_size_out, //
            base_right.out_view, 1, assign_right_size_out) and
            inlined_right.*.out_root == null)
        {
            assert(base_left.out_view.overlapsAll(1, assign_left_size_out, //
                base_right.out_view, 1, assign_right_size_out));
            assert(!base_left.out_view.overlapsPartial(1, assign_left_size_out, //
                base_right.out_view, 1, assign_right_size_out));
            assert(right_idx == right_idx_max_written);

            const inlined_right_num_new: u32 = inlined_right_num_initial + inlined_left.num + 1;
            inlined_right.* = .{
                .num = inlined_right_num_new,
                .base = try gpa.realloc(inlined_right.*.base, inlined_right_num_new),
                .out = try gpa.realloc(inlined_right.*.out, inlined_right_num_new),
                .in = try gpa.realloc(inlined_right.*.in, inlined_right_num_new),
                .in_root = inlined_right.*.in_root,
                .out_root = inlined_right_num_new - 1,
            };

            const inlined_right_idx_last: u32 = inlined_right_num_new - 1;
            inlined_right.*.base[inlined_right_idx_last] = base_left;
            inlined_right.*.out[inlined_right_idx_last] =
                if (inlined_left.out_root) |out| out + inlined_right_num_initial else null;
            inlined_right.*.in[inlined_right_idx_last] =
                if (inlined_left.in_root) |in| in + inlined_right_num_initial else null;

            var inlined_left_idx: u32 = 0;
            while (inlined_left_idx < inlined_left.num) : (inlined_left_idx += 1) {
                inlined_right.*.base[inlined_right_num_initial + inlined_left_idx] =
                    inlined_left.base[inlined_left_idx];
                inlined_right.*.out[inlined_right_num_initial + inlined_left_idx] =
                    if (inlined_left.out[inlined_left_idx]) |out| out + inlined_right_num_initial else null;
                inlined_right.*.in[inlined_right_num_initial + inlined_left_idx] =
                    if (inlined_left.in[inlined_left_idx]) |in| in + inlined_right_num_initial else null;
            }

            break :right_loop;
        } else {
            if (base_left.out.id == base_right.in.id and base_left.out_view.overlaps(1, assign_left_size_out, //
                base_right.in_view, 1, assign_right_size_in) and
                inlined_right.*.in_root == null and !base_right.kind.isUnary())
            {
                assert(base_left.out_view.overlapsAll(1, assign_left_size_out, //
                    base_right.in_view, 1, assign_right_size_in));
                assert(!base_left.out_view.overlapsPartial(1, assign_left_size_out, //
                    base_right.in_view, 1, assign_right_size_in));
                assert(switch (base_left.out.data().*.kind) {
                    .normal => right_idx != right_idx_max_written,
                    .intermediary => true,
                    .free => unreachable,
                });

                const inlined_right_num_new: u32 = inlined_right_num_initial + inlined_left.num + 1;
                inlined_right.* = .{
                    .num = inlined_right_num_new,
                    .base = try gpa.realloc(inlined_right.*.base, inlined_right_num_new),
                    .out = try gpa.realloc(inlined_right.*.out, inlined_right_num_new),
                    .in = try gpa.realloc(inlined_right.*.in, inlined_right_num_new),
                    .in_root = inlined_right_num_new - 1,
                    .out_root = inlined_right.*.out_root,
                };

                const inlined_right_idx_last: u32 = inlined_right_num_new - 1;
                inlined_right.*.base[inlined_right_idx_last] = base_left;
                inlined_right.*.out[inlined_right_idx_last] =
                    if (inlined_left.out_root) |out| out + inlined_right_num_initial else null;
                inlined_right.*.in[inlined_right_idx_last] =
                    if (inlined_left.in_root) |in| in + inlined_right_num_initial else null;

                var inlined_left_idx: u32 = 0;
                while (inlined_left_idx < inlined_left.num) : (inlined_left_idx += 1) {
                    inlined_right.*.base[inlined_right_num_initial + inlined_left_idx] =
                        inlined_left.base[inlined_left_idx];
                    inlined_right.*.out[inlined_right_num_initial + inlined_left_idx] =
                        if (inlined_left.out[inlined_left_idx]) |out| out + inlined_right_num_initial else null;
                    inlined_right.*.in[inlined_right_num_initial + inlined_left_idx] =
                        if (inlined_left.in[inlined_left_idx]) |in| in + inlined_right_num_initial else null;
                }
            }

            var inlined_right_idx: u32 = 0;
            while (inlined_right_idx < inlined_right_num_initial) : (inlined_right_idx += 1) {
                const inlined_right_size_out: Vec4 = if (inlined_right.base[inlined_right_idx].kind.isReduce())
                    .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                else
                    assign_right.size;
                const inlined_right_size_in: Vec4 = if (inlined_right.base[inlined_right_idx].kind.isExpand())
                    .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                else
                    assign_right.size;
                if (base_left.out.id == inlined_right.*.base[inlined_right_idx].out.id and
                    inlined_right.*.out[inlined_right_idx] == null and
                    base_left.out_view.overlaps(1, assign_left_size_out, //
                        inlined_right.*.base[inlined_right_idx].out_view, 1, inlined_right_size_out))
                {
                    assert(base_left.out_view.overlapsAll(1, assign_left_size_out, //
                        inlined_right.*.base[inlined_right_idx].out_view, 1, inlined_right_size_out));
                    assert(!base_left.out_view.overlapsPartial(1, assign_left_size_out, //
                        inlined_right.*.base[inlined_right_idx].out_view, 1, inlined_right_size_out));

                    const inlined_right_num_temp: u32 = inlined_right.*.num;
                    const inlined_right_num_new: u32 = inlined_right.*.num + inlined_left.num + 1;
                    inlined_right.* = .{
                        .num = inlined_right_num_new,
                        .base = try gpa.realloc(inlined_right.*.base, inlined_right_num_new),
                        .out = try gpa.realloc(inlined_right.*.out, inlined_right_num_new),
                        .in = try gpa.realloc(inlined_right.*.in, inlined_right_num_new),
                        .in_root = inlined_right.*.in_root,
                        .out_root = inlined_right.*.out_root,
                    };

                    const inlined_right_idx_last: u32 = inlined_right_num_new - 1;

                    inlined_right.*.out[inlined_right_idx] = inlined_right_idx_last;

                    inlined_right.*.base[inlined_right_idx_last] = base_left;
                    inlined_right.*.out[inlined_right_idx_last] =
                        if (inlined_left.out_root) |out| out + inlined_right_num_temp else null;
                    inlined_right.*.in[inlined_right_idx_last] =
                        if (inlined_left.in_root) |in| in + inlined_right_num_temp else null;

                    var inlined_left_idx: u32 = 0;
                    while (inlined_left_idx < inlined_left.num) : (inlined_left_idx += 1) {
                        inlined_right.*.base[inlined_right_num_temp + inlined_left_idx] =
                            inlined_left.base[inlined_left_idx];
                        inlined_right.*.out[inlined_right_num_temp + inlined_left_idx] =
                            if (inlined_left.out[inlined_left_idx]) |out| out + inlined_right_num_temp else null;
                        inlined_right.*.in[inlined_right_num_temp + inlined_left_idx] =
                            if (inlined_left.in[inlined_left_idx]) |in| in + inlined_right_num_temp else null;
                    }
                } else if (base_left.out.id == inlined_right.*.base[inlined_right_idx].in.id and
                    inlined_right.*.in[inlined_right_idx] == null and
                    base_left.out_view.overlaps(1, assign_left_size_out, //
                        inlined_right.*.base[inlined_right_idx].in_view, 1, inlined_right_size_in) and
                    !inlined_right.*.base[inlined_right_idx].kind.isUnary())
                {
                    assert(base_left.out_view.overlapsAll(1, assign_left_size_out, //
                        inlined_right.*.base[inlined_right_idx].in_view, 1, inlined_right_size_in));
                    assert(!base_left.out_view.overlapsPartial(1, assign_left_size_out, //
                        inlined_right.*.base[inlined_right_idx].in_view, 1, inlined_right_size_in));

                    const inlined_right_num_temp: u32 = inlined_right.*.num;
                    const inlined_right_num_new: u32 = inlined_right.*.num + inlined_left.num + 1;
                    inlined_right.* = .{
                        .num = inlined_right_num_new,
                        .base = try gpa.realloc(inlined_right.*.base, inlined_right_num_new),
                        .out = try gpa.realloc(inlined_right.*.out, inlined_right_num_new),
                        .in = try gpa.realloc(inlined_right.*.in, inlined_right_num_new),
                        .in_root = inlined_right.*.in_root,
                        .out_root = inlined_right.*.out_root,
                    };

                    const inlined_right_idx_last: u32 = inlined_right_num_new - 1;

                    inlined_right.*.in[inlined_right_idx] = inlined_right_idx_last;

                    inlined_right.*.base[inlined_right_idx_last] = base_left;
                    inlined_right.*.out[inlined_right_idx_last] =
                        if (inlined_left.out_root) |out| out + inlined_right_num_temp else null;
                    inlined_right.*.in[inlined_right_idx_last] =
                        if (inlined_left.in_root) |in| in + inlined_right_num_temp else null;

                    var inlined_left_idx: u32 = 0;
                    while (inlined_left_idx < inlined_left.num) : (inlined_left_idx += 1) {
                        inlined_right.*.base[inlined_right_num_temp + inlined_left_idx] =
                            inlined_left.base[inlined_left_idx];
                        inlined_right.*.out[inlined_right_num_temp + inlined_left_idx] =
                            if (inlined_left.out[inlined_left_idx]) |out| out + inlined_right_num_temp else null;
                        inlined_right.*.in[inlined_right_num_temp + inlined_left_idx] =
                            if (inlined_left.in[inlined_left_idx]) |in| in + inlined_right_num_temp else null;
                    }
                }
            }
        }
    }

    var assign_num_new: u32 = 0;
    var assign_idx: u32 = 0;
    while (assign_idx < pir.assign_num) : (assign_idx += 1) {
        if (assign_idx == left_idx) {
            gpa.free(pir.assign[assign_idx].inlined.base);
            gpa.free(pir.assign[assign_idx].inlined.out);
            gpa.free(pir.assign[assign_idx].inlined.in);
        } else {
            pir.assign[assign_num_new] = pir.assign[assign_idx];
            assign_num_new += 1;
        }
    }
    assert(assign_num_new == pir.assign_num - 1);
    pir.assign_num = assign_num_new;
}

fn viewOffsetMergePossible(left: Assign, right: Assign) bool {
    if (left.inlined.num != right.inlined.num or !left.size.equal(right.size)) {
        return false;
    }
    const base_num: u32 = 1 + left.inlined.num;

    var base_idx: u32 = 0;
    while (base_idx < base_num) : (base_idx += 1) {
        var left_base: Base = if (base_idx == 0) left.base else left.inlined.base[base_idx - 1];
        const right_base: Base = if (base_idx == 0) right.base else right.inlined.base[base_idx - 1];

        var dim_idx: u32 = 0;
        while (dim_idx < 8) : (dim_idx += 1) {
            const left_wait: *u32 = switch (dim_idx) {
                0 => &left_base.out_view.repeat_wait.a,
                1 => &left_base.out_view.repeat_wait.z,
                2 => &left_base.out_view.repeat_wait.y,
                3 => &left_base.out_view.repeat_wait.x,
                4 => &left_base.in_view.repeat_wait.a,
                5 => &left_base.in_view.repeat_wait.z,
                6 => &left_base.in_view.repeat_wait.y,
                7 => &left_base.in_view.repeat_wait.x,
                else => unreachable,
            };
            const left_stride: *u32 = switch (dim_idx) {
                0 => &left_base.out_view.repeat_stride.a,
                1 => &left_base.out_view.repeat_stride.z,
                2 => &left_base.out_view.repeat_stride.y,
                3 => &left_base.out_view.repeat_stride.x,
                4 => &left_base.in_view.repeat_stride.a,
                5 => &left_base.in_view.repeat_stride.z,
                6 => &left_base.in_view.repeat_stride.y,
                7 => &left_base.in_view.repeat_stride.x,
                else => unreachable,
            };
            const left_reset: *u32 = switch (dim_idx) {
                0 => &left_base.out_view.repeat_reset.a,
                1 => &left_base.out_view.repeat_reset.z,
                2 => &left_base.out_view.repeat_reset.y,
                3 => &left_base.out_view.repeat_reset.x,
                4 => &left_base.in_view.repeat_reset.a,
                5 => &left_base.in_view.repeat_reset.z,
                6 => &left_base.in_view.repeat_reset.y,
                7 => &left_base.in_view.repeat_reset.x,
                else => unreachable,
            };
            const left_offset: u32 = switch (dim_idx) {
                0 => left_base.out_view.viewAtRepeat(left.size, 0).aOffset(),
                1 => left_base.out_view.viewAtRepeat(left.size, 0).zOffset(),
                2 => left_base.out_view.viewAtRepeat(left.size, 0).yOffset(),
                3 => left_base.out_view.viewAtRepeat(left.size, 0).xOffset(),
                4 => left_base.in_view.viewAtRepeat(left.size, 0).aOffset(),
                5 => left_base.in_view.viewAtRepeat(left.size, 0).zOffset(),
                6 => left_base.in_view.viewAtRepeat(left.size, 0).yOffset(),
                7 => left_base.in_view.viewAtRepeat(left.size, 0).xOffset(),
                else => unreachable,
            };
            var right_repeat_idx: u32 = 0;
            while (right_repeat_idx < right.repeats) : (right_repeat_idx += 1) {
                const right_offset: u32 = switch (dim_idx) {
                    0 => right_base.out_view.viewAtRepeat(right.size, right_repeat_idx).aOffset(),
                    1 => right_base.out_view.viewAtRepeat(right.size, right_repeat_idx).zOffset(),
                    2 => right_base.out_view.viewAtRepeat(right.size, right_repeat_idx).yOffset(),
                    3 => right_base.out_view.viewAtRepeat(right.size, right_repeat_idx).xOffset(),
                    4 => right_base.in_view.viewAtRepeat(right.size, right_repeat_idx).aOffset(),
                    5 => right_base.in_view.viewAtRepeat(right.size, right_repeat_idx).zOffset(),
                    6 => right_base.in_view.viewAtRepeat(right.size, right_repeat_idx).yOffset(),
                    7 => right_base.in_view.viewAtRepeat(right.size, right_repeat_idx).xOffset(),
                    else => unreachable,
                };

                if (right_offset < left_offset) {
                    return false;
                }

                if (left_wait.* == ViewOffset.value_none) {
                    assert(left_stride.* == ViewOffset.value_none);
                    if (right_offset == left_offset) {
                        continue;
                    } else {
                        left_wait.* = left.repeats + right_repeat_idx;
                        left_stride.* = right_offset - left_offset;
                    }
                } else {
                    assert(left_stride.* != ViewOffset.value_none);
                    if (left_reset.* == ViewOffset.value_none) {
                        if (left_offset == right_offset) {
                            left_reset.* = left.repeats + right_repeat_idx;
                        } else {
                            if (@divFloor(left.repeats + right_repeat_idx, left_wait.*) * left_stride.* + left_offset == right_offset) {
                                continue;
                            } else {
                                return false;
                            }
                        }
                    } else {
                        if (@divFloor((left.repeats + right_repeat_idx) % left_reset.*, left_wait.*) * left_stride.* + left_offset != right_offset) {
                            return false;
                        }
                    }
                }
            }
        }
    }

    return true;
}
fn viewOffsetMerge(left: *Assign, right: Assign) void {
    assert(viewOffsetMergePossible(left.*, right)); // This is slow and duplicate but just to be sure

    const base_num: u32 = 1 + left.inlined.num;

    var base_idx: u32 = 0;
    while (base_idx < base_num) : (base_idx += 1) {
        const left_base: *Base = if (base_idx == 0) &left.base else &left.inlined.base[base_idx - 1];
        const right_base: Base = if (base_idx == 0) right.base else right.inlined.base[base_idx - 1];

        var dim_idx: u32 = 0;
        dim: while (dim_idx < 8) : (dim_idx += 1) {
            const left_wait: *u32 = switch (dim_idx) {
                0 => &left_base.out_view.repeat_wait.a,
                1 => &left_base.out_view.repeat_wait.z,
                2 => &left_base.out_view.repeat_wait.y,
                3 => &left_base.out_view.repeat_wait.x,
                4 => &left_base.in_view.repeat_wait.a,
                5 => &left_base.in_view.repeat_wait.z,
                6 => &left_base.in_view.repeat_wait.y,
                7 => &left_base.in_view.repeat_wait.x,
                else => unreachable,
            };
            const left_stride: *u32 = switch (dim_idx) {
                0 => &left_base.out_view.repeat_stride.a,
                1 => &left_base.out_view.repeat_stride.z,
                2 => &left_base.out_view.repeat_stride.y,
                3 => &left_base.out_view.repeat_stride.x,
                4 => &left_base.in_view.repeat_stride.a,
                5 => &left_base.in_view.repeat_stride.z,
                6 => &left_base.in_view.repeat_stride.y,
                7 => &left_base.in_view.repeat_stride.x,
                else => unreachable,
            };
            const left_reset: *u32 = switch (dim_idx) {
                0 => &left_base.out_view.repeat_reset.a,
                1 => &left_base.out_view.repeat_reset.z,
                2 => &left_base.out_view.repeat_reset.y,
                3 => &left_base.out_view.repeat_reset.x,
                4 => &left_base.in_view.repeat_reset.a,
                5 => &left_base.in_view.repeat_reset.z,
                6 => &left_base.in_view.repeat_reset.y,
                7 => &left_base.in_view.repeat_reset.x,
                else => unreachable,
            };
            const left_offset: u32 = switch (dim_idx) {
                0 => left_base.out_view.viewAtRepeat(left.size, 0).aOffset(),
                1 => left_base.out_view.viewAtRepeat(left.size, 0).zOffset(),
                2 => left_base.out_view.viewAtRepeat(left.size, 0).yOffset(),
                3 => left_base.out_view.viewAtRepeat(left.size, 0).xOffset(),
                4 => left_base.in_view.viewAtRepeat(left.size, 0).aOffset(),
                5 => left_base.in_view.viewAtRepeat(left.size, 0).zOffset(),
                6 => left_base.in_view.viewAtRepeat(left.size, 0).yOffset(),
                7 => left_base.in_view.viewAtRepeat(left.size, 0).xOffset(),
                else => unreachable,
            };
            var right_repeat_idx: u32 = 0;
            right_repeat: while (right_repeat_idx < right.repeats) : (right_repeat_idx += 1) {
                const right_offset: u32 = switch (dim_idx) {
                    0 => right_base.out_view.viewAtRepeat(right.size, right_repeat_idx).aOffset(),
                    1 => right_base.out_view.viewAtRepeat(right.size, right_repeat_idx).zOffset(),
                    2 => right_base.out_view.viewAtRepeat(right.size, right_repeat_idx).yOffset(),
                    3 => right_base.out_view.viewAtRepeat(right.size, right_repeat_idx).xOffset(),
                    4 => right_base.in_view.viewAtRepeat(right.size, right_repeat_idx).aOffset(),
                    5 => right_base.in_view.viewAtRepeat(right.size, right_repeat_idx).zOffset(),
                    6 => right_base.in_view.viewAtRepeat(right.size, right_repeat_idx).yOffset(),
                    7 => right_base.in_view.viewAtRepeat(right.size, right_repeat_idx).xOffset(),
                    else => unreachable,
                };
                assert(right_offset >= left_offset);

                if (left_wait.* == ViewOffset.value_none) {
                    assert(left_stride.* == ViewOffset.value_none);
                    if (right_offset == left_offset) {
                        continue :right_repeat;
                    } else {
                        left_wait.* = left.repeats + right_repeat_idx;
                        left_stride.* = right_offset - left_offset;
                    }
                } else {
                    assert(left_stride.* != ViewOffset.value_none);
                    if (left_reset.* == ViewOffset.value_none) {
                        if (left_offset == right_offset) {
                            left_reset.* = left.repeats + right_repeat_idx;
                            continue :dim;
                        }
                    }
                }
            }
        }
    }
    left.repeats += right.repeats;
}
pub fn parallelizeGather(gpa: Allocator, optimization: *ArrayList(Optimization), pir: Pir) !void {
    var left_idx: u32 = 0;
    outer: while (left_idx < pir.assign_num - 1) : (left_idx += 1) {
        const assign_left: Assign = pir.assign[left_idx];
        const base_left: Base = pir.assign[left_idx].base;
        const assign_left_size_out: Vec4 = if (base_left.kind.isReduce())
            .{ .a = 1, .z = 1, .y = 1, .x = 1 }
        else
            assign_left.size;
        const assign_left_size_in: Vec4 = if (base_left.kind.isExpand())
            .{ .a = 1, .z = 1, .y = 1, .x = 1 }
        else
            assign_left.size;

        var right_idx: u32 = left_idx + 1;
        while (right_idx < pir.assign_num) : (right_idx += 1) {
            const assign_right: Assign = pir.assign[right_idx];
            const base_right: Base = pir.assign[right_idx].base;
            const assign_right_size_out: Vec4 = if (base_right.kind.isReduce())
                .{ .a = 1, .z = 1, .y = 1, .x = 1 }
            else
                assign_right.size;
            const assign_right_size_in: Vec4 = if (base_right.kind.isExpand())
                .{ .a = 1, .z = 1, .y = 1, .x = 1 }
            else
                assign_right.size;

            const overlap_out_out: bool = base_left.out.id == base_right.out.id and
                base_left.out_view.overlaps(assign_left.repeats, assign_left_size_out, //
                    base_right.out_view, assign_right.repeats, assign_right_size_out);
            const overlap_out_in: bool = base_left.out.id == base_right.in.id and
                base_left.out_view.overlaps(assign_left.repeats, assign_left_size_out, //
                    base_right.in_view, assign_right.repeats, assign_right_size_in);
            const overlap_in_out: bool = base_left.in.id == base_right.out.id and
                base_left.in_view.overlaps(assign_left.repeats, assign_left_size_in, //
                    base_right.out_view, assign_right.repeats, assign_right_size_out);
            const overlap_inline: bool = blk: {
                var inlined_idx: u32 = 0;
                while (inlined_idx < assign_right.inlined.num) : (inlined_idx += 1) {
                    const base_inlined: Base = pir.assign[right_idx].inlined.base[inlined_idx];
                    const assign_right_size_in_inlined: Vec4 = if (base_inlined.kind.isExpand())
                        .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                    else
                        assign_right.size;
                    if (base_left.out.id == base_inlined.in.id and
                        !base_inlined.kind.isUnary() and
                        assign_right.inlined.in[inlined_idx] == null and
                        base_left.out_view.overlaps(assign_left.repeats, assign_left_size_out, //
                            base_inlined.in_view, assign_right.repeats, assign_right_size_in_inlined))
                    {
                        break :blk true;
                    }
                }
                break :blk false;
            };
            if (overlap_out_out or overlap_out_in or overlap_in_out or overlap_inline) {
                continue :outer;
            }

            if (viewOffsetMergePossible(assign_left, assign_right)) {
                var back_idx: u32 = 1;
                while (back_idx < right_idx - left_idx) : (back_idx += 1) {
                    const right_back_idx: u32 = right_idx - back_idx;

                    const assign_right_back: Assign = pir.assign[right_back_idx];
                    const base_right_back: Base = pir.assign[right_back_idx].base;
                    const assign_right_back_size_out: Vec4 = if (base_right_back.kind.isReduce())
                        .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                    else
                        assign_right_back.size;
                    const assign_right_back_size_in: Vec4 = if (base_right_back.kind.isExpand())
                        .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                    else
                        assign_right_back.size;

                    const overlap_out_out_back: bool = base_right_back.out.id == base_right.out.id and
                        base_right_back.out_view.overlaps(assign_right_back.repeats, assign_right_back_size_out, //
                            base_right.out_view, assign_right.repeats, assign_right_size_out);
                    const overlap_out_in_back: bool = base_right_back.out.id == base_right.in.id and
                        base_right_back.out_view.overlaps(assign_right_back.repeats, assign_right_back_size_out, //
                            base_right.in_view, assign_right.repeats, assign_right_size_in);
                    const overlap_in_out_back: bool = base_right_back.in.id == base_right.out.id and
                        base_right_back.in_view.overlaps(assign_right_back.repeats, assign_right_back_size_in, //
                            base_right.out_view, assign_right.repeats, assign_right_size_out);
                    const overlap_inline_back: bool = blk: {
                        var inlined_idx: u32 = 0;
                        while (inlined_idx < assign_right.inlined.num) : (inlined_idx += 1) {
                            const base_inlined: Base = assign_right.inlined.base[inlined_idx];
                            const assign_inlined_size_in: Vec4 = if (base_inlined.kind.isExpand())
                                .{ .a = 1, .z = 1, .y = 1, .x = 1 }
                            else
                                assign_right.size;
                            if (base_right_back.out.id == base_inlined.in.id and
                                !base_inlined.kind.isUnary() and
                                assign_right.inlined.in[inlined_idx] == null and
                                base_right_back.out_view.overlaps(assign_right_back.repeats, assign_right_back_size_out, //
                                    base_inlined.in_view, assign_right.repeats, assign_inlined_size_in))
                            {
                                break :blk true;
                            }
                        }
                        break :blk false;
                    };
                    if (overlap_out_out_back or overlap_out_in_back or
                        overlap_in_out_back or overlap_inline_back)
                    {
                        continue :outer;
                    }
                }

                const parallelized: Optimization = .{
                    .parallelize = .{
                        .left_idx = left_idx,
                        .right_idx = right_idx,
                    },
                };
                optimization.appendBounded(parallelized) catch {
                    try optimization.ensureTotalCapacity(gpa, @max(optimization.capacity * 2, 4)); // Just in case it somehow has a capacity of 0
                    try optimization.appendBounded(parallelized);
                };
                continue :outer;
            }
        }
    }
}
// I don't think there is way to make this faster than O(n^2) unless I make a max loop size, which sucks for large PIRs
pub fn parallelize(gpa: Allocator, pir: *Pir, left_idx: u32, right_idx: u32) !void {
    assert(left_idx < right_idx);
    assert(right_idx <= pir.assign_num);
    assert(viewOffsetMergePossible(pir.assign[left_idx], pir.assign[right_idx]));

    viewOffsetMerge(&pir.assign[left_idx], pir.assign[right_idx]);

    var assign_num_new: u32 = 0;
    var assign_idx: u32 = 0;
    while (assign_idx < pir.assign_num) : (assign_idx += 1) {
        if (assign_idx == right_idx) {
            gpa.free(pir.assign[assign_idx].inlined.base);
            gpa.free(pir.assign[assign_idx].inlined.out);
            gpa.free(pir.assign[assign_idx].inlined.in);
        } else {
            pir.assign[assign_num_new] = pir.assign[assign_idx];
            assign_num_new += 1;
        }
    }
    pir.assign_num = assign_num_new;
}

// $TODO Add in local size as a factor because those are also likely to have some cache coherency
pub fn splitKernelGather(
    gpa: Allocator,
    optimization: *ArrayList(Optimization),
    pir: Pir,
    size_global: u32,
    size_local: u32,
) !void {
    _ = size_local;
    var assign_idx: u32 = 0;
    while (assign_idx < pir.assign_num) : (assign_idx += 1) {
        if (!pir.assign[assign_idx].base.kind.isReduce() and
            pir.assign[assign_idx].repeats < size_global and
            !pir.assign[assign_idx].split)
        {
            const split: Optimization = .{
                .split = .{
                    .idx = assign_idx,
                },
            };
            optimization.appendBounded(split) catch {
                try optimization.ensureTotalCapacity(gpa, @max(optimization.capacity * 2, 4)); // Just in case it somehow has a capacity of 0
                try optimization.appendBounded(split);
            };
        }
    }
}
/// Split work more evenly across kernels. Does nothing to reduce ops as they can't trivially be parallilized (yet (copium))
pub fn splitKernel(pir: *Pir, idx: u32) void {
    pir.assign[idx].split = true;
}

pub fn simd(_: Allocator, pir: *Pir) !void {
    _ = pir;
}

pub fn memoryLayout(_: Allocator, pir: *Pir) !void {
    _ = pir;
}
