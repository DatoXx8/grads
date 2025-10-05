// $TODO Expressive numerical representation of an optimization such that a casey type optimizer is possible

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const Tensor = @import("../Tensor.zig");
const Op = Tensor.Op;
const Buffer = Tensor.Buffer;
const Pir = @import("Pir.zig");
const Assign = Pir.Assign;
const Base = Pir.Base;
const DimInfo = Pir.DimInfo;

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
        idx: u32,
    },
    split: struct {
        idx: u32,
    },
    fuse: struct {
        left_idx: u32,
        right_idx: u32,
    },
};

/// $WARN Things like `sqrt(x)^2` for `x < 0` are undefined behaviour and will just be optimized to `id(x)`
/// Check if left and right can be merged.
/// Assumes there is no useage of the out buffer of left between the two bases.
fn mergeOpPossible(left: Assign, right: Assign) bool {
    if (left.inlined != null or right.inlined != null) return false; // $TODO Handle this case. Shouldn't be too hard

    if (left.base.repeats != right.base.repeats) return false;
    if (!left.base.out_dim.equal(right.base.out_dim)) return false;
    if (!left.base.in_dim.equal(right.base.in_dim)) return false;
    if (!left.base.out.equal(right.base.out) or !left.base.in.equal(right.base.in)) return false;
    if (right.base.kind.isReduce()) return false;
    if (right.base.kind.overwrites()) return true;

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
pub fn mergeOpGather(allocator: Allocator, optimization: *[]Optimization, optimization_count: *u32, pir: Pir) !void {
    var left_idx: u32 = 0;
    while (left_idx < pir.assign_num - 1) : (left_idx += 1) {
        var right_idx: u32 = left_idx + 1;
        while (right_idx < pir.assign_num) : (right_idx += 1) {
            if (mergeOpPossible(pir.assign[left_idx], pir.assign[right_idx])) {
                defer optimization_count.* += 1;
                if (optimization_count.* == optimization.*.len) {
                    optimization.* = try allocator.realloc(optimization.*, optimization.*.len * 2);
                }
                optimization.*[optimization_count.*] = .{
                    .fuse = .{
                        .left_idx = left_idx,
                        .right_idx = right_idx,
                    },
                };
                break;
            } else {
                const left: Base = pir.assign[left_idx].base;
                const right: Base = pir.assign[right_idx].base;
                // If there is a simulator failure try removing the overlap condition here
                const out_out_conflict = left.out.id == right.out.id and left.out.overlaps(right.out);
                const out_in_conflict = left.out.id == right.in.id and left.out.overlaps(right.in);
                const in_out_conflict = left.in.id == right.out.id and left.out.overlaps(right.in);
                if (out_out_conflict or out_in_conflict or in_out_conflict) {
                    break;
                }
            }
        }
    }
}
// $FIXME This does not handle inlines and repeats
pub fn mergeOp(pir: *Pir, left_idx: u32, right_idx: u32) void {
    assert(mergeOpPossible(pir.assign[left_idx], pir.assign[right_idx]));
    const merge_both: bool = mergeOpCombine(pir.assign[left_idx], &pir.assign[right_idx]);

    var assign_num_new: u32 = 0;
    for (0..pir.assign_num) |assign_idx| {
        if (assign_idx == left_idx or (assign_idx == right_idx and merge_both)) {
            //
        } else {
            pir.assign[assign_num_new] = pir.assign[assign_idx];
            assign_num_new += 1;
        }
    }
    assert(assign_num_new > 0);
    pir.assign_num = assign_num_new;
}

pub fn inlineOpGather(allocator: Allocator, optimization: *[]Optimization, optimization_count: *u32, pir: Pir) !void {
    var assign_idx: u32 = 0;
    outer: while (assign_idx < pir.assign_num - 1) : (assign_idx += 1) {
        var inlineable: bool = false;
        var out_found: bool = false;
        var in_found: bool = false;

        var search_idx: u32 = assign_idx + 1;
        while (search_idx < pir.assign_num) : (search_idx += 1) {
            const overlap_out_out: bool = pir.assign[assign_idx].base.out.id == pir.assign[search_idx].base.out.id and
                pir.assign[assign_idx].base.out.overlapsPartial(pir.assign[search_idx].base.out);
            const overlap_out_in: bool = pir.assign[assign_idx].base.out.id == pir.assign[search_idx].base.in.id and
                pir.assign[assign_idx].base.out.overlapsPartial(pir.assign[search_idx].base.in);
            const overlap_in_out: bool = pir.assign[assign_idx].base.in.id == pir.assign[search_idx].base.out.id and
                pir.assign[assign_idx].base.in.overlaps(pir.assign[search_idx].base.out) and
                !pir.assign[assign_idx].base.kind.isUnary();
            const repeat_different: bool = pir.assign[assign_idx].base.repeats != pir.assign[search_idx].base.repeats;
            const overlap_out_x_inlined: bool = blk: {
                if (pir.assign[search_idx].inlined) |inlined| {
                    var inlined_idx: u32 = 0;
                    while (inlined_idx < inlined.inlined_num) : (inlined_idx += 1) {
                        var overlap: bool = false;
                        if (pir.assign[assign_idx].base.out.id == inlined.base[inlined_idx].out.id and inlined.out[inlined_idx] == null) {
                            if (pir.assign[assign_idx].base.out.overlapsPartial(inlined.base[inlined_idx].out)) {
                                overlap = true;
                            } else if (pir.assign[assign_idx].base.out.overlapsAll(inlined.base[inlined_idx].out)) {
                                out_found = true;
                            }
                        }
                        if (pir.assign[assign_idx].base.out.id == inlined.base[inlined_idx].in.id and inlined.in[inlined_idx] == null) {
                            if (pir.assign[assign_idx].base.out.overlapsPartial(inlined.base[inlined_idx].in)) {
                                overlap = true;
                            } else if (pir.assign[assign_idx].base.out.overlapsAll(inlined.base[inlined_idx].in)) {
                                in_found = true;
                            }
                        }
                        if (overlap) {
                            break :blk false;
                        }
                    }
                }

                break :blk false;
            };
            if (overlap_out_out or overlap_out_in or overlap_in_out or repeat_different or overlap_out_x_inlined) {
                inlineable = false;
                continue :outer;
            }
            if (pir.assign[assign_idx].base.out.equal(pir.assign[search_idx].base.out)) {
                inlineable = true;
                break;
            }
            if (pir.assign[assign_idx].base.out.equal(pir.assign[search_idx].base.in)) {
                inlineable = true;
                out_found = true;
            }
            if (pir.assign[assign_idx].base.in.equal(pir.assign[search_idx].base.out) and !pir.assign[assign_idx].base.kind.isUnary()) {
                inlineable = true;
                in_found = true;
            }
            if (out_found and in_found) {
                break;
            }
        }

        if (inlineable) {
            defer optimization_count.* += 1;
            if (optimization_count.* == optimization.*.len) {
                optimization.* = try allocator.realloc(optimization.*, optimization.*.len * 2);
            }
            optimization.*[optimization_count.*] = .{
                .inlined = .{
                    .idx = assign_idx,
                },
            };
        }
    }
}
pub fn inlineOp(allocator: Allocator, pir: *Pir, left_idx: u32) !void {
    assert(left_idx + 1 < pir.assign_num);

    var search_idx: u32 = left_idx + 1;
    while (search_idx < pir.assign_num) : (search_idx += 1) {
        if (pir.assign[left_idx].base.out.equal(pir.assign[search_idx].base.out)) {
            if (pir.assign[search_idx].base.kind.overwrites()) {
                break;
            }

            const out_root_old: ?u32 = if (pir.assign[left_idx].inlined) |inlined|
                inlined.out_root
            else
                null;
            const inlined_num_start: u32 = if (pir.assign[left_idx].inlined) |inlined|
                inlined.inlined_num
            else
                0;
            const inlined_num_old: u32 = if (pir.assign[search_idx].inlined) |inlined|
                inlined.inlined_num
            else
                0;
            const inlined_num_new: u32 = 1 + inlined_num_start + inlined_num_old;

            if (pir.assign[search_idx].inlined) |*inlined| {
                if (inlined.out_root) |out_root_old_val| {
                    inlined.* = .{
                        .inlined_num = inlined_num_new,
                        .base = try allocator.realloc(inlined.base, inlined_num_new),
                        .out = try allocator.realloc(inlined.out, inlined_num_new),
                        .in = try allocator.realloc(inlined.in, inlined_num_new),
                        .in_root = inlined.in_root,
                        .out_root = inlined.out_root,
                    };
                    inlined.*.out[out_root_old_val] = inlined_num_new - 1;
                } else {
                    inlined.* = .{
                        .inlined_num = inlined_num_new,
                        .base = try allocator.realloc(inlined.base, inlined_num_new),
                        .out = try allocator.realloc(inlined.out, inlined_num_new),
                        .in = try allocator.realloc(inlined.in, inlined_num_new),
                        .in_root = inlined.in_root,
                        .out_root = inlined_num_new - 1,
                    };
                }
            } else {
                assert(inlined_num_old == 0);
                pir.assign[search_idx].inlined = .{
                    .inlined_num = inlined_num_new,
                    .base = try allocator.alloc(Base, inlined_num_new),
                    .out = try allocator.alloc(?u32, inlined_num_new),
                    .in = try allocator.alloc(?u32, inlined_num_new),
                    .in_root = null,
                    .out_root = inlined_num_new - 1,
                };
            }

            assert(pir.assign[search_idx].inlined != null);
            pir.assign[search_idx].inlined.?.in[inlined_num_new - 1] = if (pir.assign[left_idx].inlined) |j| (if (j.in_root) |in| in + inlined_num_old else null) else null;
            pir.assign[search_idx].inlined.?.out[inlined_num_new - 1] = if (out_root_old) |out| out + inlined_num_old else null;
            pir.assign[search_idx].inlined.?.base[inlined_num_new - 1] = pir.assign[left_idx].base;

            if (pir.assign[left_idx].inlined) |inlined| {
                assert(inlined.inlined_num > 0);
                var inlined_idx: u32 = 0;
                while (inlined_idx < inlined.inlined_num) : (inlined_idx += 1) {
                    pir.assign[search_idx].inlined.?.in[inlined_num_old + inlined_idx] = if (inlined.in[inlined_idx]) |in| in + inlined_num_old else null;
                    pir.assign[search_idx].inlined.?.out[inlined_num_old + inlined_idx] = if (inlined.out[inlined_idx]) |out| out + inlined_num_old else null;
                    pir.assign[search_idx].inlined.?.base[inlined_num_old + inlined_idx] = inlined.base[inlined_idx];
                }
            }

            pir.assign[search_idx].inlined.?.inlined_num = inlined_num_new;
            break;
        } else if (pir.assign[left_idx].base.out.equal(pir.assign[search_idx].base.in)) {
            if (!pir.assign[left_idx].base.out.intermediary) {
                // This should never be the case I think
                break;
            }

            const in_root_old: ?u32 = if (pir.assign[left_idx].inlined) |inlined|
                inlined.in_root
            else
                null;
            const inlined_num_start: u32 = if (pir.assign[left_idx].inlined) |inlined|
                inlined.inlined_num
            else
                0;
            const inlined_num_old: u32 = if (pir.assign[search_idx].inlined) |inlined|
                inlined.inlined_num
            else
                0;
            const inlined_num_new: u32 = 1 + inlined_num_start + inlined_num_old;

            if (pir.assign[search_idx].inlined) |*inlined| {
                assert(inlined.in_root == null);
                inlined.* = .{
                    .inlined_num = inlined_num_new,
                    .base = try allocator.realloc(inlined.base, inlined_num_new),
                    .out = try allocator.realloc(inlined.out, inlined_num_new),
                    .in = try allocator.realloc(inlined.in, inlined_num_new),
                    .in_root = inlined_num_new - 1,
                    .out_root = inlined.out_root,
                };
            } else {
                assert(inlined_num_old == 0);
                pir.assign[search_idx].inlined = .{
                    .inlined_num = inlined_num_new,
                    .base = try allocator.alloc(Base, inlined_num_new),
                    .out = try allocator.alloc(?u32, inlined_num_new),
                    .in = try allocator.alloc(?u32, inlined_num_new),
                    .in_root = inlined_num_new - 1,
                    .out_root = null,
                };
            }

            assert(pir.assign[search_idx].inlined != null);
            pir.assign[search_idx].inlined.?.in[inlined_num_new - 1] = if (in_root_old) |in| in + inlined_num_old else null;
            pir.assign[search_idx].inlined.?.out[inlined_num_new - 1] = if (pir.assign[left_idx].inlined) |j| (if (j.out_root) |out| out + inlined_num_old else null) else null;
            pir.assign[search_idx].inlined.?.base[inlined_num_new - 1] = pir.assign[left_idx].base;

            if (pir.assign[left_idx].inlined) |inlined| {
                assert(inlined.inlined_num > 0);
                var inlined_idx: u32 = 0;
                while (inlined_idx < inlined.inlined_num) : (inlined_idx += 1) {
                    pir.assign[search_idx].inlined.?.in[inlined_num_old + inlined_idx] = if (inlined.in[inlined_idx]) |in| in + inlined_num_old else null;
                    pir.assign[search_idx].inlined.?.out[inlined_num_old + inlined_idx] = if (inlined.out[inlined_idx]) |out| out + inlined_num_old else null;
                    pir.assign[search_idx].inlined.?.base[inlined_num_old + inlined_idx] = inlined.base[inlined_idx];
                }
            }

            pir.assign[search_idx].inlined.?.inlined_num = inlined_num_new;
        } else {
            // Inlining non-intermediaries is only allowed if the target has the same out buffer, which is not the case here
            if (!pir.assign[left_idx].base.out.intermediary) {
                continue;
            }
            if (pir.assign[search_idx].inlined) |*inlined| {
                const inlined_num_start: u32 = if (pir.assign[left_idx].inlined) |inlined_old|
                    inlined_old.inlined_num
                else
                    0;
                const inlined_num_old: u32 = inlined.inlined_num;
                const inlined_num_new: u32 = 1 + inlined_num_start + inlined_num_old;

                var inlined_idx: u32 = 0;
                while (inlined_idx < inlined.inlined_num) : (inlined_idx += 1) {
                    if (pir.assign[left_idx].base.out.id == inlined.base[inlined_idx].out.id and inlined.out[inlined_idx] == null) {
                        inlined.out[inlined_idx] = inlined_num_new - 1;

                        if (pir.assign[left_idx].inlined) |inlined_left| {
                            assert(inlined_left.inlined_num > 0);
                            var inlined_left_idx: u32 = 0;
                            while (inlined_left_idx < inlined.inlined_num) : (inlined_left_idx += 1) {
                                inlined.in[inlined_num_old + inlined_left_idx] = if (inlined_left.in[inlined_left_idx]) |in| in + inlined_num_old else null;
                                inlined.out[inlined_num_old + inlined_left_idx] = if (inlined_left.out[inlined_left_idx]) |out| out + inlined_num_old else null;
                                inlined.base[inlined_num_old + inlined_left_idx] = inlined_left.base[inlined_left_idx];
                            }
                        }
                    } else if (pir.assign[left_idx].base.out.id == inlined.base[inlined_idx].in.id and inlined.in[inlined_idx] == null) {
                        inlined.in[inlined_idx] = inlined_num_new - 1;

                        if (pir.assign[left_idx].inlined) |inlined_left| {
                            assert(inlined_left.inlined_num > 0);
                            var inlined_left_idx: u32 = 0;
                            while (inlined_left_idx < inlined.inlined_num) : (inlined_left_idx += 1) {
                                inlined.in[inlined_num_old + inlined_left_idx] = if (inlined_left.in[inlined_left_idx]) |in| in + inlined_num_old else null;
                                inlined.out[inlined_num_old + inlined_left_idx] = if (inlined_left.out[inlined_left_idx]) |out| out + inlined_num_old else null;
                                inlined.base[inlined_num_old + inlined_left_idx] = inlined_left.base[inlined_left_idx];
                            }
                        }
                    }
                }
                inlined.inlined_num = inlined_num_new;
            }
        }
    }

    var assign_num_new: u32 = 0;
    var assign_idx: u32 = 0;
    while (assign_idx < pir.assign_num) : (assign_idx += 1) {
        if (assign_idx == left_idx) {
            if (pir.assign[assign_idx].inlined) |*inlined| {
                allocator.free(inlined.base);
                allocator.free(inlined.out);
                allocator.free(inlined.in);
            }
        } else {
            pir.assign[assign_num_new] = pir.assign[assign_idx];
            assign_num_new += 1;
        }
    }
    pir.assign_num = assign_num_new;
}
fn dimInfoMergePossible(left: Assign, right: Assign) bool {
    if ((if (left.inlined) |i| i.inlined_num else 0) != (if (right.inlined) |i| i.inlined_num else 0)) {
        return false;
    }

    const base_num: u32 = 1 + if (left.inlined) |i| i.inlined_num else 0;

    for (0..base_num) |base_idx| {
        const left_base: Base = if (base_idx == 0) left.base else left.inlined.?.base[base_idx - 1];
        const right_base: Base = if (base_idx == 0) right.base else right.inlined.?.base[base_idx - 1];

        if (base_idx != 0) {
            assert(left.inlined != null);
            assert(right.inlined != null);
            if (left.inlined.?.out[base_idx - 1] == right.inlined.?.out[base_idx - 1]) {
                if (left.inlined.?.out[base_idx - 1] != null) {
                    continue;
                }
            } else {
                return false;
            }
            if (left.inlined.?.in[base_idx - 1] == right.inlined.?.in[base_idx - 1]) {
                if (left.inlined.?.in[base_idx - 1] != null) {
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

    const base_num: u32 = 1 + if (left.inlined) |i| i.inlined_num else 0;

    for (0..base_num) |base_idx| {
        const left_base: *Base = if (base_idx == 0) &left.base else &left.inlined.?.base[base_idx - 1];
        const right_base: *const Base = if (base_idx == 0) &right.base else &right.inlined.?.base[base_idx - 1];

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
fn dimInfoOverlap(left: Buffer, left_dim: DimInfo, left_repeats: u32, right: Buffer, right_dim: DimInfo, right_repeats: u32) bool {
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
pub fn parallelizeGather(allocator: Allocator, optimization: *[]Optimization, optimization_count: *u32, pir: Pir) !void {
    var start_idx: u32 = 0;
    outer: while (start_idx < pir.assign_num - 1) : (start_idx += 1) {
        var search_idx: u32 = start_idx + 1;
        while (search_idx < pir.assign_num) : (search_idx += 1) {
            // I think these conditions are the least restrictive they could be because of the inlining that happens before parallelization
            // At the very least it is not obvious to me how to loosen them.
            const overlap_out_out: bool = pir.assign[start_idx].base.out.id == pir.assign[search_idx].base.out.id and
                dimInfoOverlap(pir.assign[start_idx].base.out, pir.assign[start_idx].base.out_dim, pir.assign[start_idx].base.repeats, //
                    pir.assign[search_idx].base.out, pir.assign[search_idx].base.out_dim, pir.assign[search_idx].base.repeats);
            const overlap_out_in: bool = pir.assign[start_idx].base.out.id == pir.assign[search_idx].base.in.id and
                dimInfoOverlap(pir.assign[start_idx].base.out, pir.assign[start_idx].base.out_dim, pir.assign[start_idx].base.repeats, //
                    pir.assign[search_idx].base.in, pir.assign[search_idx].base.in_dim, pir.assign[search_idx].base.repeats);
            const overlap_in_out: bool = pir.assign[start_idx].base.in.id == pir.assign[search_idx].base.out.id and
                dimInfoOverlap(pir.assign[start_idx].base.in, pir.assign[start_idx].base.in_dim, pir.assign[start_idx].base.repeats, //
                    pir.assign[search_idx].base.out, pir.assign[search_idx].base.out_dim, pir.assign[search_idx].base.repeats);
            const overlap_inline: bool = blk: {
                if (pir.assign[search_idx].inlined) |inlined| {
                    var inlined_idx: u32 = 0;
                    while (inlined_idx < inlined.inlined_num) : (inlined_idx += 1) {
                        if (pir.assign[start_idx].base.out.id == inlined.base[inlined_idx].in.id and inlined.in[inlined_idx] == null and
                            dimInfoOverlap(pir.assign[start_idx].base.out, pir.assign[start_idx].base.out_dim, pir.assign[start_idx].base.repeats, //
                                inlined.base[inlined_idx].in, inlined.base[inlined_idx].in_dim, inlined.base[inlined_idx].repeats))
                        {
                            break :blk true;
                        }
                    }
                }
                break :blk false;
            };
            if (overlap_out_out or overlap_out_in or overlap_in_out or overlap_inline) {
                continue :outer;
            }

            // $FIXME Need to also search backwards from the search_idx to check that nothing overlaps with that

            if (dimInfoMergePossible(pir.assign[start_idx], pir.assign[search_idx])) {
                var back_idx: u32 = 1;
                while (back_idx < search_idx - start_idx) : (back_idx += 1) {
                    const search_back_idx: u32 = search_idx - back_idx;

                    const overlap_out_out_back: bool = pir.assign[search_back_idx].base.out.id == pir.assign[search_idx].base.out.id and
                        dimInfoOverlap(pir.assign[search_back_idx].base.out, pir.assign[search_back_idx].base.out_dim, pir.assign[search_back_idx].base.repeats, //
                            pir.assign[search_idx].base.out, pir.assign[search_idx].base.out_dim, pir.assign[search_idx].base.repeats);
                    const overlap_out_in_back: bool = pir.assign[search_back_idx].base.out.id == pir.assign[search_idx].base.in.id and
                        dimInfoOverlap(pir.assign[search_back_idx].base.out, pir.assign[search_back_idx].base.out_dim, pir.assign[search_back_idx].base.repeats, //
                            pir.assign[search_idx].base.in, pir.assign[search_idx].base.in_dim, pir.assign[search_idx].base.repeats);
                    const overlap_in_out_back: bool = pir.assign[search_back_idx].base.in.id == pir.assign[search_idx].base.out.id and
                        dimInfoOverlap(pir.assign[search_back_idx].base.in, pir.assign[search_back_idx].base.in_dim, pir.assign[search_back_idx].base.repeats, //
                            pir.assign[search_idx].base.out, pir.assign[search_idx].base.out_dim, pir.assign[search_idx].base.repeats);
                    const overlap_inline_back: bool = blk: {
                        if (pir.assign[search_idx].inlined) |inlined| {
                            var inlined_idx: u32 = 0;
                            while (inlined_idx < inlined.inlined_num) : (inlined_idx += 1) {
                                if (pir.assign[search_back_idx].base.out.id == inlined.base[inlined_idx].in.id and inlined.in[inlined_idx] == null and
                                    dimInfoOverlap(pir.assign[search_back_idx].base.out, pir.assign[search_back_idx].base.out_dim, pir.assign[search_back_idx].base.repeats, //
                                        inlined.base[inlined_idx].in, inlined.base[inlined_idx].in_dim, inlined.base[inlined_idx].repeats))
                                {
                                    break :blk true;
                                }
                            }
                        }
                        break :blk false;
                    };
                    if (overlap_out_out_back or overlap_out_in_back or overlap_in_out_back or overlap_inline_back) {
                        continue :outer;
                    }
                }

                defer optimization_count.* += 1;
                if (optimization_count.* == optimization.*.len) {
                    optimization.* = try allocator.realloc(optimization.*, optimization.*.len * 2);
                }
                optimization.*[optimization_count.*] = .{
                    .parallelize = .{
                        .left_idx = start_idx,
                        .right_idx = search_idx,
                    },
                };
                continue :outer;
            }
        }
    }
}
// I don't think there is way to make this faster than O(n^2) unless I make a max loop size, which sucks for large PIRs
pub fn parallelize(allocator: Allocator, pir: *Pir, left_idx: u32, right_idx: u32) !void {
    assert(left_idx < right_idx);
    assert(right_idx <= pir.assign_num);
    assert(dimInfoMergePossible(pir.assign[left_idx], pir.assign[right_idx]));

    dimInfoMerge(&pir.assign[left_idx], pir.assign[right_idx]);

    var assign_num_new: u32 = 0;
    var assign_idx: u32 = 0;
    while (assign_idx < pir.assign_num) : (assign_idx += 1) {
        if (assign_idx == right_idx) {
            if (pir.assign[assign_idx].inlined) |*inlined| {
                allocator.free(inlined.base);
                allocator.free(inlined.out);
                allocator.free(inlined.in);
            }
        } else {
            pir.assign[assign_num_new] = pir.assign[assign_idx];
            assign_num_new += 1;
        }
    }
    pir.assign_num = assign_num_new;
}
// $TODO Add in local size as a factor because those are also likely to have some cache coherency
pub fn splitKernelGather(
    allocator: Allocator,
    optimization: *[]Optimization,
    optimization_count: *u32,
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
            defer optimization_count.* += 1;
            if (optimization_count.* == optimization.*.len) {
                optimization.* = try allocator.realloc(optimization.*, optimization.*.len * 2);
            }
            optimization.*[optimization_count.*] = .{
                .split = .{
                    .idx = assign_idx,
                },
            };
        }
    }
}
/// Split work more evenly across kernels. Does nothing to reduce ops as they can't trivially be parallilized (yet (copium))
pub fn splitKernel(pir: *Pir, idx: u32) void {
    pir.assign[idx].split = true;
}
pub fn simd(allocator: Allocator, pir: *Pir) !void {
    _ = allocator;
    _ = pir;
}
pub fn memoryLayout(allocator: Allocator, pir: *Pir) !void {
    _ = allocator;
    _ = pir;
}
