const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

const Linearized = @import("../Linearized.zig");
const Op = Linearized.Op;
const Buffer = @import("../Buffer.zig");
const Pir = @import("Pir.zig");
const Assign = Pir.Assign;
const Inlined = Pir.Inlined;
const Base = Pir.Base;
const DimInfo = Pir.DimInfo;
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
    if (left.base.repeats != right.base.repeats) return false;
    if (!left.base.out_dim.equal(right.base.out_dim)) return false;
    if (!left.base.in_dim.equal(right.base.in_dim)) return false;
    if (!left.base.out.equal(right.base.out) or !left.base.in.equal(right.base.in)) return false;
    if (right.base.kind.isReduce()) return false;
    if (right.base.kind.overwrites()) return true;

    if (left.inlined.num != right.inlined.num or
        left.inlined.out_root != right.inlined.out_root or
        left.inlined.in_root != right.inlined.in_root) return false;

    var inlined_idx: u32 = 0;
    while (inlined_idx < left.inlined.num) : (inlined_idx += 1) {
        if (!left.inlined.base[inlined_idx].equal(right.inlined.base[inlined_idx]) or
            left.inlined.out[inlined_idx] != right.inlined.out[inlined_idx] or
            left.inlined.in[inlined_idx] != right.inlined.in[inlined_idx])
        {
            return false;
        }
    }

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
                    try optimization.resize(gpa, @min(optimization.capacity * 2, 4)); // Just in case it somehow has a capacity of 0
                    try optimization.appendBounded(fuse);
                };
                break;
            } else {
                const base_left: Base = pir.assign[left_idx].base;
                const inlined_left: Inlined = pir.assign[left_idx].inlined;

                const base_right: Base = pir.assign[right_idx].base;
                const inlined_right: Inlined = pir.assign[right_idx].inlined;

                const overlap_out_x: bool = blk: {
                    if (base_left.out.id == base_right.out.id and
                        base_left.out.overlaps(base_right.out))
                    {
                        break :blk true;
                    }
                    if (base_left.out.id == base_right.in.id and
                        base_left.out.overlaps(base_right.in) and
                        !base_right.kind.isUnary())
                    {
                        break :blk true;
                    }
                    var inlined_right_idx: u32 = 0;
                    while (inlined_right_idx < inlined_right.num) : (inlined_right_idx += 1) {
                        if (base_left.out.id == inlined_right.base[inlined_right_idx].out.id and
                            base_left.out.overlaps(inlined_right.base[inlined_right_idx].out) and
                            inlined_right.out[inlined_right_idx] == null)
                        {
                            break :blk true;
                        }
                        if (base_left.out.id == inlined_right.base[inlined_right_idx].in.id and
                            base_left.out.overlaps(inlined_right.base[inlined_right_idx].in) and
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
                        base_left.in.overlaps(base_right.out))
                    {
                        break :blk true;
                    }
                    var inlined_left_idx: u32 = 0;
                    while (inlined_left_idx < inlined_left.num) : (inlined_left_idx += 1) {
                        if (inlined_left.base[inlined_left_idx].out.id == base_right.out.id and
                            inlined_left.base[inlined_left_idx].out.overlaps(base_right.out) and
                            inlined_left.out[inlined_left_idx] == null)
                        {
                            break :blk true;
                        }
                        if (inlined_left.base[inlined_left_idx].in.id == base_right.out.id and
                            inlined_left.base[inlined_left_idx].in.overlaps(base_right.out) and
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
        const base_left: Base = pir.assign[left_idx].base;
        const inlined_left: Inlined = pir.assign[left_idx].inlined;

        if (base_left.kind.isReduce()) {
            continue :left_loop;
        }

        var inlined_valid: bool = true;
        var right_idx_max_written: u32 = left_idx;

        var right_idx: u32 = left_idx + 1;
        right_loop: while (right_idx < pir.assign_num) : (right_idx += 1) {
            assert(inlined_valid);

            const base_right: Base = pir.assign[right_idx].base;
            const inlined_right: Inlined = pir.assign[right_idx].inlined;

            // $TODO This should be more optimized
            const repeats_different: bool = base_left.repeats != base_right.repeats;
            const split_different: bool = false;
            // const split_different: bool = pir.assign[left_idx].split != pir.assign[right_idx].split;
            const left_out_overwritten: bool = blk: {
                if (base_left.out.id == base_right.out.id and
                    base_left.out.overlapsAll(base_right.out) and
                    base_right.kind.overwrites())
                {
                    break :blk true;
                }
                var inlined_right_idx: u32 = 0;
                while (inlined_right_idx < inlined_right.num) : (inlined_right_idx += 1) {
                    if (base_left.out.id == inlined_right.base[inlined_right_idx].out.id and
                        base_left.out.overlapsAll(inlined_right.base[inlined_right_idx].out) and
                        inlined_right.base[inlined_right_idx].kind.overwrites() and
                        inlined_right.out[inlined_right_idx] == null) // This part is probably not necessary
                    {
                        break :blk true;
                    }
                }
                break :blk false;
            };
            const left_out_non_intermdiary_inlined: bool = blk: {
                const base_left_out_intermediary: bool = switch (base_left.out.kind) {
                    .intermediary => true,
                    .normal => false,
                };
                if (base_left.out.id == base_right.in.id and
                    base_left.out.overlapsAll(base_right.in) and
                    !base_left_out_intermediary and
                    !base_right.kind.isUnary())
                {
                    break :blk true;
                }
                var inlined_right_idx: u32 = 0;
                while (inlined_right_idx < inlined_right.num) : (inlined_right_idx += 1) {
                    if (base_left.out.id == inlined_right.base[inlined_right_idx].in.id and
                        base_left.out.overlapsAll(inlined_right.base[inlined_right_idx].in) and
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
                    base_left.out.overlapsPartial(base_right.out))
                {
                    break :blk true;
                }
                if (base_left.out.id == base_right.in.id and
                    base_left.out.overlapsPartial(base_right.in) and
                    !base_right.kind.isUnary())
                {
                    break :blk true;
                }
                var inlined_right_idx: u32 = 0;
                while (inlined_right_idx < inlined_right.num) : (inlined_right_idx += 1) {
                    if (base_left.out.id == inlined_right.base[inlined_right_idx].out.id and
                        base_left.out.overlapsPartial(inlined_right.base[inlined_right_idx].out) and
                        inlined_right.out[inlined_right_idx] == null)
                    {
                        break :blk true;
                    }
                    if (base_left.out.id == inlined_right.base[inlined_right_idx].in.id and
                        base_left.out.overlapsPartial(inlined_right.base[inlined_right_idx].in) and
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
                    base_left.in.overlapsPartial(base_right.out))
                {
                    break :blk true;
                }
                var inlined_left_idx: u32 = 0;
                while (inlined_left_idx < inlined_left.num) : (inlined_left_idx += 1) {
                    if (inlined_left.base[inlined_left_idx].out.id == base_right.out.id and
                        inlined_left.base[inlined_left_idx].out.overlapsPartial(base_right.out) and
                        inlined_left.out[inlined_left_idx] == null)
                    {
                        break :blk true;
                    }
                    if (inlined_left.base[inlined_left_idx].in.id == base_right.out.id and
                        inlined_left.base[inlined_left_idx].in.overlapsPartial(base_right.out) and
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
                    base_left.in.overlapsAll(base_right.out))
                {
                    break :blk true;
                }
                var inlined_left_idx: u32 = 0;
                while (inlined_left_idx < inlined_left.num) : (inlined_left_idx += 1) {
                    if (inlined_left.base[inlined_left_idx].in.id == base_right.out.id and
                        inlined_left.base[inlined_left_idx].in.overlapsAll(base_right.out) and
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
                    base_left.out.overlapsAll(base_right.in) and
                    !base_right.kind.isUnary())
                {
                    break :blk true;
                }
                var inlined_right_idx: u32 = 0;
                while (inlined_right_idx < inlined_right.num) : (inlined_right_idx += 1) {
                    if (base_left.out.id == inlined_right.base[inlined_right_idx].in.id and
                        base_left.out.overlapsAll(inlined_right.base[inlined_right_idx].in) and
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
                    base_left.out.overlapsAll(base_right.out))
                {
                    break :blk true;
                }
                var inlined_right_idx: u32 = 0;
                while (inlined_right_idx < inlined_right.num) : (inlined_right_idx += 1) {
                    if (base_left.out.id == inlined_right.base[inlined_right_idx].out.id and
                        base_left.out.overlapsAll(inlined_right.base[inlined_right_idx].out) and
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
                try optimization.resize(gpa, @min(optimization.capacity * 2, 4)); // Just in case it somehow has a capacity of 0
                try optimization.appendBounded(inlined);
            };
        }
    }
}
pub fn inlineOp(gpa: Allocator, pir: *Pir, left_idx: u32, right_idx_max_written: u32) !void {
    assert(left_idx < right_idx_max_written);
    assert(right_idx_max_written < pir.assign_num);

    const base_left: Base = pir.assign[left_idx].base;
    const inlined_left: Inlined = pir.assign[left_idx].inlined;

    var right_idx: u32 = left_idx + 1;
    right_loop: while (right_idx <= right_idx_max_written) : (right_idx += 1) {
        const base_right: Base = pir.assign[right_idx].base;
        const inlined_right: *Inlined = &pir.assign[right_idx].inlined;

        const inlined_right_num_initial: u32 = inlined_right.*.num;

        if (base_left.out.id == base_right.out.id and base_left.out.overlaps(base_right.out) and
            inlined_right.*.out_root == null)
        {
            assert(base_left.out.overlapsAll(base_right.out));
            assert(!base_left.out.overlapsPartial(base_right.out));
            assert(base_left.out.kind == base_right.out.kind);
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
            if (base_left.out.id == base_right.in.id and base_left.out.overlaps(base_right.in) and
                inlined_right.*.in_root == null and !base_right.kind.isUnary())
            {
                assert(base_left.out.overlapsAll(base_right.in));
                assert(!base_left.out.overlapsPartial(base_right.in));
                assert(base_left.out.kind == base_right.in.kind);
                assert(switch (base_left.out.kind) {
                    .normal => right_idx != right_idx_max_written,
                    .intermediary => true,
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
                if (base_left.out.id == inlined_right.*.base[inlined_right_idx].out.id and
                    inlined_right.*.out[inlined_right_idx] == null and
                    base_left.out.overlaps(inlined_right.*.base[inlined_right_idx].out))
                {
                    assert(base_left.out.overlapsAll(inlined_right.*.base[inlined_right_idx].out));
                    assert(!base_left.out.overlapsPartial(inlined_right.*.base[inlined_right_idx].out));

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
                    base_left.out.overlaps(inlined_right.*.base[inlined_right_idx].in) and
                    !inlined_right.*.base[inlined_right_idx].kind.isUnary())
                {
                    assert(base_left.out.overlapsAll(inlined_right.*.base[inlined_right_idx].in));
                    assert(!base_left.out.overlapsPartial(inlined_right.*.base[inlined_right_idx].in));

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

fn dimInfoMergePossible(left: Assign, right: Assign) bool {
    if (left.inlined.num != right.inlined.num) {
        return false;
    }

    const base_num: u32 = 1 + left.inlined.num;

    for (0..base_num) |base_idx| {
        const left_base: Base = if (base_idx == 0) left.base else left.inlined.base[base_idx - 1];
        const right_base: Base = if (base_idx == 0) right.base else right.inlined.base[base_idx - 1];

        if (base_idx != 0) {
            if (left.inlined.out[base_idx - 1] == right.inlined.out[base_idx - 1]) {
                if (left.inlined.out[base_idx - 1] != null) {
                    continue;
                }
            } else {
                return false;
            }
            if (left.inlined.in[base_idx - 1] == right.inlined.in[base_idx - 1]) {
                if (left.inlined.in[base_idx - 1] != null) {
                    continue;
                }
            } else {
                return false;
            }
        }

        if (!left_base.equalNoOffset(right_base)) {
            return false;
        }

        inline for (0..8) |dim_idx| {
            var left_wait: u32 = switch (dim_idx) {
                0 => left_base.out_dim.a_wait,
                1 => left_base.out_dim.z_wait,
                2 => left_base.out_dim.y_wait,
                3 => left_base.out_dim.x_wait,
                4 => left_base.in_dim.a_wait,
                5 => left_base.in_dim.z_wait,
                6 => left_base.in_dim.y_wait,
                7 => left_base.in_dim.x_wait,
                else => unreachable,
            };
            var left_stride: u32 = switch (dim_idx) {
                0 => left_base.out_dim.a_stride,
                1 => left_base.out_dim.z_stride,
                2 => left_base.out_dim.y_stride,
                3 => left_base.out_dim.x_stride,
                4 => left_base.in_dim.a_stride,
                5 => left_base.in_dim.z_stride,
                6 => left_base.in_dim.y_stride,
                7 => left_base.in_dim.x_stride,
                else => unreachable,
            };
            var left_reset: u32 = switch (dim_idx) {
                0 => left_base.out_dim.a_reset,
                1 => left_base.out_dim.z_reset,
                2 => left_base.out_dim.y_reset,
                3 => left_base.out_dim.x_reset,
                4 => left_base.in_dim.a_reset,
                5 => left_base.in_dim.z_reset,
                6 => left_base.in_dim.y_reset,
                7 => left_base.in_dim.x_reset,
                else => unreachable,
            };
            const left_off: u32 = switch (dim_idx) {
                0 => left_base.out.aOffset(),
                1 => left_base.out.zOffset(),
                2 => left_base.out.yOffset(),
                3 => left_base.out.xOffset(),
                4 => left_base.in.aOffset(),
                5 => left_base.in.zOffset(),
                6 => left_base.in.yOffset(),
                7 => left_base.in.xOffset(),
                else => unreachable,
            };
            const right_wait: u32 = switch (dim_idx) {
                0 => right_base.out_dim.a_wait,
                1 => right_base.out_dim.z_wait,
                2 => right_base.out_dim.y_wait,
                3 => right_base.out_dim.x_wait,
                4 => right_base.in_dim.a_wait,
                5 => right_base.in_dim.z_wait,
                6 => right_base.in_dim.y_wait,
                7 => right_base.in_dim.x_wait,
                else => unreachable,
            };
            const right_stride: u32 = switch (dim_idx) {
                0 => right_base.out_dim.a_stride,
                1 => right_base.out_dim.z_stride,
                2 => right_base.out_dim.y_stride,
                3 => right_base.out_dim.x_stride,
                4 => right_base.in_dim.a_stride,
                5 => right_base.in_dim.z_stride,
                6 => right_base.in_dim.y_stride,
                7 => right_base.in_dim.x_stride,
                else => unreachable,
            };
            const right_reset: u32 = switch (dim_idx) {
                0 => right_base.out_dim.a_reset,
                1 => right_base.out_dim.z_reset,
                2 => right_base.out_dim.y_reset,
                3 => right_base.out_dim.x_reset,
                4 => right_base.in_dim.a_reset,
                5 => right_base.in_dim.z_reset,
                6 => right_base.in_dim.y_reset,
                7 => right_base.in_dim.x_reset,
                else => unreachable,
            };
            const right_off_start: u32 = switch (dim_idx) {
                0 => right_base.out.aOffset(),
                1 => right_base.out.zOffset(),
                2 => right_base.out.yOffset(),
                3 => right_base.out.xOffset(),
                4 => right_base.in.aOffset(),
                5 => right_base.in.zOffset(),
                6 => right_base.in.yOffset(),
                7 => right_base.in.xOffset(),
                else => unreachable,
            };
            var right_repeat_idx: u32 = 0;
            while (right_repeat_idx < right_base.repeats) : (right_repeat_idx += 1) {
                const right_off: u32 = blk: {
                    if (right_wait == DimInfo.value_none) {
                        assert(right_stride == DimInfo.value_none);
                        break :blk right_off_start;
                    } else {
                        assert(right_stride != DimInfo.value_none);
                        if (right_reset == DimInfo.value_none) {
                            break :blk @divFloor(right_repeat_idx, right_wait) * right_stride + right_off_start;
                        } else {
                            break :blk @divFloor(right_repeat_idx % right_reset, right_wait) * right_stride + right_off_start;
                        }
                    }
                };

                if (right_off < left_off) {
                    return false;
                }

                if (left_wait == DimInfo.value_none) {
                    assert(left_stride == DimInfo.value_none);
                    if (right_off == left_off) {
                        continue;
                    } else {
                        left_wait = left_base.repeats + right_repeat_idx;
                        left_stride = right_off - left_off;
                    }
                } else {
                    assert(left_stride != DimInfo.value_none);
                    if (left_reset == DimInfo.value_none) {
                        if (left_off == right_off) {
                            left_reset = left_base.repeats + right_repeat_idx;
                        } else {
                            if (@divFloor(left_base.repeats + right_repeat_idx, left_wait) * left_stride + left_off == right_off) {
                                continue;
                            } else {
                                return false;
                            }
                        }
                    } else {
                        if (@divFloor((left_base.repeats + right_repeat_idx) % left_reset, left_wait) * left_stride + left_off != right_off) {
                            return false;
                        }
                    }
                }
            }
        }
    }

    return true;
}
fn dimInfoMerge(left: *Assign, right: Assign) void {
    assert(dimInfoMergePossible(left.*, right)); // This is slow and duplicate but just to be sure

    const base_num: u32 = 1 + left.inlined.num;

    for (0..base_num) |base_idx| {
        const left_base: *Base = if (base_idx == 0) &left.base else &left.inlined.base[base_idx - 1];
        const right_base: *const Base = if (base_idx == 0) &right.base else &right.inlined.base[base_idx - 1];

        outer_dim: for (0..8) |dim_idx| {
            const left_wait: *u32 = switch (dim_idx) {
                0 => &left_base.out_dim.a_wait,
                1 => &left_base.out_dim.z_wait,
                2 => &left_base.out_dim.y_wait,
                3 => &left_base.out_dim.x_wait,
                4 => &left_base.in_dim.a_wait,
                5 => &left_base.in_dim.z_wait,
                6 => &left_base.in_dim.y_wait,
                7 => &left_base.in_dim.x_wait,
                else => unreachable,
            };
            const left_stride: *u32 = switch (dim_idx) {
                0 => &left_base.out_dim.a_stride,
                1 => &left_base.out_dim.z_stride,
                2 => &left_base.out_dim.y_stride,
                3 => &left_base.out_dim.x_stride,
                4 => &left_base.in_dim.a_stride,
                5 => &left_base.in_dim.z_stride,
                6 => &left_base.in_dim.y_stride,
                7 => &left_base.in_dim.x_stride,
                else => unreachable,
            };
            const left_reset: *u32 = switch (dim_idx) {
                0 => &left_base.out_dim.a_reset,
                1 => &left_base.out_dim.z_reset,
                2 => &left_base.out_dim.y_reset,
                3 => &left_base.out_dim.x_reset,
                4 => &left_base.in_dim.a_reset,
                5 => &left_base.in_dim.z_reset,
                6 => &left_base.in_dim.y_reset,
                7 => &left_base.in_dim.x_reset,
                else => unreachable,
            };
            const left_off: u32 = switch (dim_idx) {
                0 => left_base.out.aOffset(),
                1 => left_base.out.zOffset(),
                2 => left_base.out.yOffset(),
                3 => left_base.out.xOffset(),
                4 => left_base.in.aOffset(),
                5 => left_base.in.zOffset(),
                6 => left_base.in.yOffset(),
                7 => left_base.in.xOffset(),
                else => unreachable,
            };
            const right_wait: u32 = switch (dim_idx) {
                0 => right_base.out_dim.a_wait,
                1 => right_base.out_dim.z_wait,
                2 => right_base.out_dim.y_wait,
                3 => right_base.out_dim.x_wait,
                4 => right_base.in_dim.a_wait,
                5 => right_base.in_dim.z_wait,
                6 => right_base.in_dim.y_wait,
                7 => right_base.in_dim.x_wait,
                else => unreachable,
            };
            const right_stride: u32 = switch (dim_idx) {
                0 => right_base.out_dim.a_stride,
                1 => right_base.out_dim.z_stride,
                2 => right_base.out_dim.y_stride,
                3 => right_base.out_dim.x_stride,
                4 => right_base.in_dim.a_stride,
                5 => right_base.in_dim.z_stride,
                6 => right_base.in_dim.y_stride,
                7 => right_base.in_dim.x_stride,
                else => unreachable,
            };
            const right_reset: u32 = switch (dim_idx) {
                0 => right_base.out_dim.a_reset,
                1 => right_base.out_dim.z_reset,
                2 => right_base.out_dim.y_reset,
                3 => right_base.out_dim.x_reset,
                4 => right_base.in_dim.a_reset,
                5 => right_base.in_dim.z_reset,
                6 => right_base.in_dim.y_reset,
                7 => right_base.in_dim.x_reset,
                else => unreachable,
            };
            const right_off_start: u32 = switch (dim_idx) {
                0 => right_base.out.aOffset(),
                1 => right_base.out.zOffset(),
                2 => right_base.out.yOffset(),
                3 => right_base.out.xOffset(),
                4 => right_base.in.aOffset(),
                5 => right_base.in.zOffset(),
                6 => right_base.in.yOffset(),
                7 => right_base.in.xOffset(),
                else => unreachable,
            };
            var right_repeat_idx: u32 = 0;
            inner_repeat: while (right_repeat_idx < right_base.repeats) : (right_repeat_idx += 1) {
                const right_off: u32 = blk: {
                    if (right_wait == DimInfo.value_none) {
                        assert(right_stride == DimInfo.value_none);
                        break :blk right_off_start;
                    } else {
                        assert(right_stride != DimInfo.value_none);
                        if (right_reset == DimInfo.value_none) {
                            break :blk @divFloor(right_repeat_idx, right_wait) * right_stride + right_off_start;
                        } else {
                            break :blk @divFloor(right_repeat_idx % right_reset, right_wait) * right_stride + right_off_start;
                        }
                    }
                };

                assert(right_off >= left_off);

                if (left_wait.* == DimInfo.value_none) {
                    assert(left_stride.* == DimInfo.value_none);
                    if (right_off == left_off) {
                        continue :inner_repeat;
                    } else {
                        left_wait.* = left_base.repeats + right_repeat_idx;
                        left_stride.* = right_off - left_off;
                    }
                } else {
                    assert(left_stride.* != DimInfo.value_none);
                    if (left_reset.* == DimInfo.value_none) {
                        if (left_off == right_off) {
                            left_reset.* = left_base.repeats + right_repeat_idx;
                            continue :outer_dim;
                        }
                    }
                }
            }
        }
        left_base.repeats += right_base.repeats;
    }
}
fn dimInfoOverlap(
    left: Buffer,
    left_dim: DimInfo,
    left_repeats: u32,
    right: Buffer,
    right_dim: DimInfo,
    right_repeats: u32,
) bool {
    var left_idx: u32 = 0;
    while (left_idx < left_repeats) : (left_idx += 1) {
        const a_1: u32 = left.aOffset() + blk: {
            if (left_dim.a_wait == DimInfo.value_none) {
                assert(left_dim.a_stride == DimInfo.value_none);
                break :blk 0;
            } else {
                assert(left_dim.a_stride != DimInfo.value_none);
                if (left_dim.a_reset == DimInfo.value_none) {
                    break :blk @divFloor(left_idx, left_dim.a_wait) * left_dim.a_stride;
                } else {
                    break :blk @divFloor(left_idx % left_dim.a_reset, left_dim.a_wait) * left_dim.a_stride;
                }
            }
        };
        const z_1: u32 = left.zOffset() + blk: {
            if (left_dim.z_wait == DimInfo.value_none) {
                assert(left_dim.z_stride == DimInfo.value_none);
                break :blk 0;
            } else {
                assert(left_dim.z_stride != DimInfo.value_none);
                if (left_dim.z_reset == DimInfo.value_none) {
                    break :blk @divFloor(left_idx, left_dim.z_wait) * left_dim.z_stride;
                } else {
                    break :blk @divFloor(left_idx % left_dim.z_reset, left_dim.z_wait) * left_dim.z_stride;
                }
            }
        };
        const y_1: u32 = left.yOffset() + blk: {
            if (left_dim.y_wait == DimInfo.value_none) {
                assert(left_dim.y_stride == DimInfo.value_none);
                break :blk 0;
            } else {
                assert(left_dim.y_stride != DimInfo.value_none);
                if (left_dim.y_reset == DimInfo.value_none) {
                    break :blk @divFloor(left_idx, left_dim.y_wait) * left_dim.y_stride;
                } else {
                    break :blk @divFloor(left_idx % left_dim.y_reset, left_dim.y_wait) * left_dim.y_stride;
                }
            }
        };
        const x_1: u32 = left.xOffset() + blk: {
            if (left_dim.x_wait == DimInfo.value_none) {
                assert(left_dim.x_stride == DimInfo.value_none);
                break :blk 0;
            } else {
                assert(left_dim.x_stride != DimInfo.value_none);
                if (left_dim.x_reset == DimInfo.value_none) {
                    break :blk @divFloor(left_idx, left_dim.x_wait) * left_dim.x_stride;
                } else {
                    break :blk @divFloor(left_idx % left_dim.x_reset, left_dim.x_wait) * left_dim.x_stride;
                }
            }
        };
        var right_idx: u32 = 0;
        while (right_idx < right_repeats) : (right_idx += 1) {
            const a_2: u32 = right.aOffset() + blk: {
                if (right_dim.a_wait == DimInfo.value_none) {
                    assert(right_dim.a_stride == DimInfo.value_none);
                    break :blk 0;
                } else {
                    assert(right_dim.a_stride != DimInfo.value_none);
                    if (right_dim.a_reset == DimInfo.value_none) {
                        break :blk @divFloor(right_idx, right_dim.a_wait) * right_dim.a_stride;
                    } else {
                        break :blk @divFloor(right_idx % right_dim.a_reset, right_dim.a_wait) * right_dim.a_stride;
                    }
                }
            };
            const z_2: u32 = right.zOffset() + blk: {
                if (right_dim.z_wait == DimInfo.value_none) {
                    assert(right_dim.z_stride == DimInfo.value_none);
                    break :blk 0;
                } else {
                    assert(right_dim.z_stride != DimInfo.value_none);
                    if (right_dim.z_reset == DimInfo.value_none) {
                        break :blk @divFloor(right_idx, right_dim.z_wait) * right_dim.z_stride;
                    } else {
                        break :blk @divFloor(right_idx % right_dim.z_reset, right_dim.z_wait) * right_dim.z_stride;
                    }
                }
            };
            const y_2: u32 = right.yOffset() + blk: {
                if (right_dim.y_wait == DimInfo.value_none) {
                    assert(right_dim.y_stride == DimInfo.value_none);
                    break :blk 0;
                } else {
                    assert(right_dim.y_stride != DimInfo.value_none);
                    if (right_dim.y_reset == DimInfo.value_none) {
                        break :blk @divFloor(right_idx, right_dim.y_wait) * right_dim.y_stride;
                    } else {
                        break :blk @divFloor(right_idx % right_dim.y_reset, right_dim.y_wait) * right_dim.y_stride;
                    }
                }
            };
            const x_2: u32 = right.xOffset() + blk: {
                if (right_dim.x_wait == DimInfo.value_none) {
                    assert(right_dim.x_stride == DimInfo.value_none);
                    break :blk 0;
                } else {
                    assert(right_dim.x_stride != DimInfo.value_none);
                    if (right_dim.x_reset == DimInfo.value_none) {
                        break :blk @divFloor(right_idx, right_dim.x_wait) * right_dim.x_stride;
                    } else {
                        break :blk @divFloor(right_idx % right_dim.x_reset, right_dim.x_wait) * right_dim.x_stride;
                    }
                }
            };
            const overlap: bool = @max(a_1, a_2) < @min(a_1 + left.a_size, a_2 + right.a_size) and
                @max(z_1, z_2) < @min(z_1 + left.z_size, z_2 + right.z_size) and
                @max(y_1, y_2) < @min(y_1 + left.y_size, y_2 + right.y_size) and
                @max(x_1, x_2) < @min(x_1 + left.x_size, x_2 + right.x_size);
            if (overlap) {
                return true;
            }
        }
    }
    return false;
}
pub fn parallelizeGather(gpa: Allocator, optimization: *ArrayList(Optimization), pir: Pir) !void {
    var start_idx: u32 = 0;
    outer: while (start_idx < pir.assign_num - 1) : (start_idx += 1) {
        var search_idx: u32 = start_idx + 1;
        while (search_idx < pir.assign_num) : (search_idx += 1) {
            // I think these conditions are the least restrictive they could be because of the inlining that happens before parallelization
            // At the very least it is not obvious to me how to loosen them.

            const base_search: Base = pir.assign[search_idx].base;
            const base_start: Base = pir.assign[start_idx].base;

            const overlap_out_out: bool = base_start.out.id == base_search.out.id and
                dimInfoOverlap(base_start.out, base_start.out_dim, base_start.repeats, //
                    base_search.out, base_search.out_dim, base_search.repeats);
            const overlap_out_in: bool = base_start.out.id == base_search.in.id and
                dimInfoOverlap(base_start.out, base_start.out_dim, base_start.repeats, //
                    base_search.in, base_search.in_dim, base_search.repeats);
            const overlap_in_out: bool = base_start.in.id == base_search.out.id and
                dimInfoOverlap(base_start.in, base_start.in_dim, base_start.repeats, //
                    base_search.out, base_search.out_dim, base_search.repeats);
            const overlap_inline: bool = blk: {
                var inlined_idx: u32 = 0;
                while (inlined_idx < pir.assign[search_idx].inlined.num) : (inlined_idx += 1) {
                    const base_inlined: Base = pir.assign[search_idx].inlined.base[inlined_idx];
                    if (base_start.out.id == base_inlined.in.id and pir.assign[search_idx].inlined.in[inlined_idx] == null and
                        dimInfoOverlap(base_start.out, base_start.out_dim, base_start.repeats, //
                            base_inlined.in, base_inlined.in_dim, base_inlined.repeats))
                    {
                        break :blk true;
                    }
                }
                break :blk false;
            };
            if (overlap_out_out or overlap_out_in or overlap_in_out or overlap_inline) {
                continue :outer;
            }

            if (dimInfoMergePossible(pir.assign[start_idx], pir.assign[search_idx])) {
                var back_idx: u32 = 1;
                while (back_idx < search_idx - start_idx) : (back_idx += 1) {
                    const search_back_idx: u32 = search_idx - back_idx;

                    const base_search_back: Base = pir.assign[search_back_idx].base;

                    const overlap_out_out_back: bool = base_search_back.out.id == base_search.out.id and
                        dimInfoOverlap(base_search_back.out, base_search_back.out_dim, base_search_back.repeats, //
                            base_search.out, base_search.out_dim, base_search.repeats);
                    const overlap_out_in_back: bool = base_search_back.out.id == base_search.in.id and
                        dimInfoOverlap(base_search_back.out, base_search_back.out_dim, base_search_back.repeats, //
                            base_search.in, base_search.in_dim, base_search.repeats);
                    const overlap_in_out_back: bool = base_search_back.in.id == base_search.out.id and
                        dimInfoOverlap(base_search_back.in, base_search_back.in_dim, base_search_back.repeats, //
                            base_search.out, base_search.out_dim, base_search.repeats);
                    const overlap_inline_back: bool = blk: {
                        var inlined_idx: u32 = 0;
                        while (inlined_idx < pir.assign[search_idx].inlined.num) : (inlined_idx += 1) {
                            const base_inlined: Base = pir.assign[search_idx].inlined.base[inlined_idx];
                            if (base_search_back.out.id == base_inlined.in.id and pir.assign[search_idx].inlined.in[inlined_idx] == null and
                                dimInfoOverlap(base_search_back.out, base_search_back.out_dim, base_search_back.repeats, //
                                    base_inlined.in, base_inlined.in_dim, base_inlined.repeats))
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
                        .left_idx = start_idx,
                        .right_idx = search_idx,
                    },
                };
                optimization.appendBounded(parallelized) catch {
                    try optimization.resize(gpa, @min(optimization.capacity * 2, 4)); // Just in case it somehow has a capacity of 0
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
    assert(dimInfoMergePossible(pir.assign[left_idx], pir.assign[right_idx]));

    dimInfoMerge(&pir.assign[left_idx], pir.assign[right_idx]);

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
            pir.assign[assign_idx].base.repeats < size_global and
            !pir.assign[assign_idx].split)
        {
            const split: Optimization = .{
                .split = .{
                    .idx = assign_idx,
                },
            };
            optimization.appendBounded(split) catch {
                try optimization.resize(gpa, @min(optimization.capacity * 2, 4)); // Just in case it somehow has a capacity of 0
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
