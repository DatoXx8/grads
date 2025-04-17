// $TODO Maybe just do the parallelize one at O0 anyways, because it is utterly useless without it
// $TODO These levels
// $TODO Expressive numerical representation of an optimization such that a casey type optimizer is possible
// Optimization levels
// O1 - parallelize, inline, split, idempotent functions
// O2 - SIMD
// O3 - memory optimizer

// Optimization levels
// O0 - none
// O1 - parallelize, inline, split, idempotent functions
// O2 - SIMD
// O3 - memory optimizer

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const Op = @import("../tensor.zig").Op;
const Ssa = @import("./ssa.zig").Ssa;
const Assign = @import("./ssa.zig").Assign;
const Base = @import("./ssa.zig").Base;
const DimInfo = @import("./ssa.zig").DimInfo;
const Inlined = @import("./ssa.zig").Inlined;

pub const Optimization = enum(u8) {
    O0,
    O1,
    O2,
    O3,
};

fn inlineOpStep(allocator: Allocator, assign: []Assign, start_idx: u32) !bool {
    assert(start_idx + 1 < assign.len);

    if (assign[start_idx].base.type.isReduce()) return false;

    for (start_idx + 1..assign.len) |assign_idx| {
        // $TODO Currently there is no way to handle partial overlaps. I think you just need to burn the whole thing down if you find one.
        if ((assign[start_idx].base.out.name_offset == assign[assign_idx].base.out.name_offset and
            assign[start_idx].base.out.overlapsPartial(assign[assign_idx].base.out)) or
            (assign[start_idx].base.out.name_offset == assign[assign_idx].base.in.name_offset and
            assign[start_idx].base.out.overlapsPartial(assign[assign_idx].base.in)) or
            (assign[start_idx].base.in.name_offset == assign[assign_idx].base.out.name_offset and
            assign[start_idx].base.in.overlaps(assign[assign_idx].base.out)))
        {
            return false;
        }
        if (assign[start_idx].base.out.equal(assign[assign_idx].base.out)) {
            break;
        }
    }

    var written: bool = false;
    for (start_idx + 1..assign.len) |assign_idx| {
        if (assign[start_idx].base.out.equal(assign[assign_idx].base.out)) {
            if (assign[assign_idx].base.overwrites()) {
                return written;
            }

            const out_root_old: ?u32 = if (assign[start_idx].inlined) |i| i.out_root else null;
            const inlined_num_start: u32 = if (assign[start_idx].inlined) |i| i.inlined_num else 0;
            const inlined_num_old: u32 = if (assign[assign_idx].inlined) |j| j.inlined_num else 0;

            const inlined_num_new: u32 = 1 + inlined_num_start + inlined_num_old;
            if (assign[assign_idx].inlined) |*i| {
                assert(i.out_root == null);
                i.* = .{
                    .inlined_num = inlined_num_new,
                    .base = try allocator.realloc(i.base, inlined_num_new),
                    .out = try allocator.realloc(i.out, inlined_num_new),
                    .in = try allocator.realloc(i.in, inlined_num_new),
                    .in_root = i.in_root,
                    .out_root = inlined_num_new - 1,
                };
            } else {
                assert(inlined_num_old == 0);
                assign[assign_idx].inlined = .{
                    .inlined_num = inlined_num_new,
                    .base = try allocator.alloc(Base, inlined_num_new),
                    .out = try allocator.alloc(?u32, inlined_num_new),
                    .in = try allocator.alloc(?u32, inlined_num_new),
                    .in_root = null,
                    .out_root = inlined_num_new - 1,
                };
            }

            assert(assign[assign_idx].inlined != null);

            assign[assign_idx].inlined.?.in[inlined_num_new - 1] = if (assign[start_idx].inlined) |j| (if (j.in_root) |in| in + inlined_num_old else null) else null;
            assign[assign_idx].inlined.?.out[inlined_num_new - 1] = out_root_old;
            assign[assign_idx].inlined.?.base[inlined_num_new - 1] = assign[start_idx].base;

            if (assign[start_idx].inlined) |j| {
                assert(j.inlined_num > 0);
                for (0..j.inlined_num) |inlined_idx| {
                    assign[assign_idx].inlined.?.in[inlined_num_old + inlined_idx] = if (j.in[inlined_idx]) |in| in + inlined_num_old else null;
                    assign[assign_idx].inlined.?.out[inlined_num_old + inlined_idx] = if (j.out[inlined_idx]) |out| out + inlined_num_old else null;
                    assign[assign_idx].inlined.?.base[inlined_num_old + inlined_idx] = j.base[inlined_idx];
                }
            }

            return true;
        }
        if (assign[start_idx].base.out.equal(assign[assign_idx].base.in)) {
            if (!assign[start_idx].base.out.intermediary) {
                return written;
            }
            const in_root_old: ?u32 = if (assign[start_idx].inlined) |i| i.in_root else null;
            const inlined_num_start: u32 = if (assign[start_idx].inlined) |i| i.inlined_num else 0;
            const inlined_num_old: u32 = if (assign[assign_idx].inlined) |j| j.inlined_num else 0;

            const inlined_num_new: u32 = 1 + inlined_num_start + inlined_num_old;
            if (assign[assign_idx].inlined) |*i| {
                assert(i.in_root == null);
                i.* = .{
                    .inlined_num = inlined_num_new,
                    .base = try allocator.realloc(i.base, inlined_num_new),
                    .out = try allocator.realloc(i.out, inlined_num_new),
                    .in = try allocator.realloc(i.in, inlined_num_new),
                    .in_root = inlined_num_new - 1,
                    .out_root = i.out_root,
                };
            } else {
                assert(inlined_num_old == 0);
                assign[assign_idx].inlined = .{
                    .inlined_num = inlined_num_new,
                    .base = try allocator.alloc(Base, inlined_num_new),
                    .out = try allocator.alloc(?u32, inlined_num_new),
                    .in = try allocator.alloc(?u32, inlined_num_new),
                    .in_root = inlined_num_new - 1,
                    .out_root = null,
                };
            }

            assert(assign[assign_idx].inlined != null);

            assign[assign_idx].inlined.?.in[inlined_num_new - 1] = in_root_old;
            assign[assign_idx].inlined.?.out[inlined_num_new - 1] = if (assign[start_idx].inlined) |j| (if (j.out_root) |out| out + inlined_num_old else null) else null;
            assign[assign_idx].inlined.?.base[inlined_num_new - 1] = assign[start_idx].base;

            if (assign[start_idx].inlined) |j| {
                assert(j.inlined_num > 0);
                for (0..j.inlined_num) |inlined_idx| {
                    assign[assign_idx].inlined.?.in[inlined_num_old + inlined_idx] = if (j.in[inlined_idx]) |in| in + inlined_num_old else null;
                    assign[assign_idx].inlined.?.out[inlined_num_old + inlined_idx] = if (j.out[inlined_idx]) |out| out + inlined_num_old else null;
                    assign[assign_idx].inlined.?.base[inlined_num_old + inlined_idx] = j.base[inlined_idx];
                }
            }

            written = true;
        }
    }

    return written;
}

// $TODO Either make the order irrelevant here or assert the right order
// $TODO This memory management is horrible. Refactor refactor refactor
//  I feel like there should be a really simple way to do this but I for the life of me can not figure it out
pub fn inlineOp(allocator: Allocator, ssa: *Ssa) !void {
    const temp_written: []bool = try allocator.alloc(bool, ssa.assign_num);
    errdefer allocator.free(temp_written);
    defer allocator.free(temp_written);
    @memset(temp_written, false);

    var start_idx: u32 = 0;
    while (start_idx < ssa.assign_num - 1) : (start_idx += 1) {
        temp_written[start_idx] = try inlineOpStep(allocator, ssa.assign, start_idx);
    }

    var assign_idx: u32 = 0;
    var assign_num_new: u32 = 0;
    while (assign_idx < ssa.assign_num) : (assign_idx += 1) {
        if (temp_written[assign_idx]) {
            if (ssa.assign[assign_idx].inlined) |*inlined| {
                allocator.free(inlined.base);
                allocator.free(inlined.out);
                allocator.free(inlined.in);
            }
        } else {
            ssa.assign[assign_num_new] = ssa.assign[assign_idx];
            assign_num_new += 1;
        }
    }
    ssa.assign_num = assign_num_new;
}

// $FIX / $TODO / $PERF / $HACK         don't even know what to mark this shit with
// If you are unlucky with the layout of your offsets then you can get into a situation where the offsets for each assign can't be modeled by a linear function.
// This is a huge issue because other functions that model the offsets can't be found easily and table driven solutions limit the loop size artificially,
// because of the limits on local memory per kernel.
// As a hacky fix we just split the loop up if there is something that can't be modeled linearly. This sucks bad. I hate that I have to do this.
// I am sorry for this terrible shittines, I just can't think of a better solution right now

fn dimInfoMaxLegal(base: []const Base) u32 {
    assert(base.len > 0);

    var max: u32 = @intCast(base.len);

    var dim_info: DimInfo = .{
        .off_out = 0,
        .off_in = 0,
        .a_stride_out = 0,
        .z_stride_out = 0,
        .y_stride_out = 0,
        .x_stride_out = 0,
        .a_stride_in = 0,
        .z_stride_in = 0,
        .y_stride_in = 0,
        .x_stride_in = 0,
        .a_reset_out = @intCast(base.len),
        .z_reset_out = @intCast(base.len),
        .y_reset_out = @intCast(base.len),
        .x_reset_out = @intCast(base.len),
        .a_reset_in = @intCast(base.len),
        .z_reset_in = @intCast(base.len),
        .y_reset_in = @intCast(base.len),
        .x_reset_in = @intCast(base.len),
        .a_wait_out = 1,
        .z_wait_out = 1,
        .y_wait_out = 1,
        .x_wait_out = 1,
        .a_wait_in = 1,
        .z_wait_in = 1,
        .y_wait_in = 1,
        .x_wait_in = 1,
    };

    var loop_idx: u32 = 1;
    while (loop_idx < base.len) : (loop_idx += 1) {
        if (base[loop_idx].out.aOffset() >= base[loop_idx - 1].out.aOffset() or base[loop_idx].out.aOffset() == base[0].out.aOffset() or
            base[loop_idx].out.zOffset() >= base[loop_idx - 1].out.zOffset() or base[loop_idx].out.zOffset() == base[0].out.zOffset() or
            base[loop_idx].out.yOffset() >= base[loop_idx - 1].out.yOffset() or base[loop_idx].out.yOffset() == base[0].out.yOffset() or
            base[loop_idx].out.xOffset() >= base[loop_idx - 1].out.xOffset() or base[loop_idx].out.xOffset() == base[0].out.xOffset() or
            base[loop_idx].in.aOffset() >= base[loop_idx - 1].in.aOffset() or base[loop_idx].in.aOffset() == base[0].in.aOffset() or
            base[loop_idx].in.zOffset() >= base[loop_idx - 1].in.zOffset() or base[loop_idx].in.zOffset() == base[0].in.zOffset() or
            base[loop_idx].in.yOffset() >= base[loop_idx - 1].in.yOffset() or base[loop_idx].in.yOffset() == base[0].in.yOffset() or
            base[loop_idx].in.xOffset() >= base[loop_idx - 1].in.xOffset() or base[loop_idx].in.xOffset() == base[0].in.xOffset())
        {
            continue;
        } else {
            max = loop_idx;
            break;
        }
    }

    var a_left_out, var z_left_out, var y_left_out, var x_left_out = .{ false, false, false, false };
    var a_left_in, var z_left_in, var y_left_in, var x_left_in = .{ false, false, false, false };
    var a_enter_out, var z_enter_out, var y_enter_out, var x_enter_out = .{ false, false, false, false };
    var a_enter_in, var z_enter_in, var y_enter_in, var x_enter_in = .{ false, false, false, false };

    const a_off_out_root: u32 = base[0].out.aOffset();
    const z_off_out_root: u32 = base[0].out.zOffset();
    const y_off_out_root: u32 = base[0].out.yOffset();
    const x_off_out_root: u32 = base[0].out.xOffset();
    const a_off_in_root: u32 = base[0].in.aOffset();
    const z_off_in_root: u32 = base[0].in.zOffset();
    const y_off_in_root: u32 = base[0].in.yOffset();
    const x_off_in_root: u32 = base[0].in.xOffset();

    loop_idx = 1;
    while (loop_idx < max) : (loop_idx += 1) {
        if (a_left_out) {
            if (a_enter_out) {
                if (base[loop_idx].out.aOffset() != (loop_idx % dim_info.a_reset_out) / dim_info.a_wait_out * dim_info.a_stride_out + a_off_out_root) {
                    max = loop_idx;
                    break;
                }
            } else {
                if (base[loop_idx].out.aOffset() == a_off_out_root) {
                    a_enter_out = true;
                    dim_info.a_reset_out = loop_idx;
                } else {
                    if (base[loop_idx].out.aOffset() != (loop_idx / dim_info.a_wait_out) * dim_info.a_stride_out + a_off_out_root) {
                        max = loop_idx;
                        break;
                    }
                }
            }
        } else {
            if (base[loop_idx].out.aOffset() < a_off_out_root) {
                max = loop_idx;
                break;
            } else if (base[loop_idx].out.aOffset() > a_off_out_root) {
                a_left_out = true;
                dim_info.a_wait_out = loop_idx;
                dim_info.a_stride_out = base[loop_idx].out.aOffset() - a_off_out_root;
            }
        }
        if (z_left_out) {
            if (z_enter_out) {
                if (base[loop_idx].out.zOffset() != (loop_idx % dim_info.z_reset_out) / dim_info.z_wait_out * dim_info.z_stride_out + z_off_out_root) {
                    max = loop_idx;
                    break;
                }
            } else {
                if (base[loop_idx].out.zOffset() == z_off_out_root) {
                    z_enter_out = true;
                    dim_info.z_reset_out = loop_idx;
                } else {
                    if (base[loop_idx].out.zOffset() != (loop_idx / dim_info.z_wait_out) * dim_info.z_stride_out + z_off_out_root) {
                        max = loop_idx;
                        break;
                    }
                }
            }
        } else {
            if (base[loop_idx].out.zOffset() < z_off_out_root) {
                max = loop_idx;
                break;
            } else if (base[loop_idx].out.zOffset() > z_off_out_root) {
                z_left_out = true;
                dim_info.z_wait_out = loop_idx;
                dim_info.z_stride_out = base[loop_idx].out.zOffset() - z_off_out_root;
            }
        }
        if (y_left_out) {
            if (y_enter_out) {
                if (base[loop_idx].out.yOffset() != (loop_idx % dim_info.y_reset_out) / dim_info.y_wait_out * dim_info.y_stride_out + y_off_out_root) {
                    max = loop_idx;
                    break;
                }
            } else {
                if (base[loop_idx].out.yOffset() == y_off_out_root) {
                    y_enter_out = true;
                    dim_info.y_reset_out = loop_idx;
                } else {
                    if (base[loop_idx].out.yOffset() != (loop_idx / dim_info.y_wait_out) * dim_info.y_stride_out + y_off_out_root) {
                        max = loop_idx;
                        break;
                    }
                }
            }
        } else {
            if (base[loop_idx].out.yOffset() < y_off_out_root) {
                max = loop_idx;
                break;
            } else if (base[loop_idx].out.yOffset() > y_off_out_root) {
                y_left_out = true;
                dim_info.y_wait_out = loop_idx;
                dim_info.y_stride_out = base[loop_idx].out.yOffset() - y_off_out_root;
            }
        }
        if (x_left_out) {
            if (x_enter_out) {
                if (base[loop_idx].out.xOffset() != (loop_idx % dim_info.x_reset_out) / dim_info.x_wait_out * dim_info.x_stride_out + x_off_out_root) {
                    max = loop_idx;
                    break;
                }
            } else {
                if (base[loop_idx].out.xOffset() == x_off_out_root) {
                    x_enter_out = true;
                    dim_info.x_reset_out = loop_idx;
                } else {
                    if (base[loop_idx].out.xOffset() != (loop_idx / dim_info.x_wait_out) * dim_info.x_stride_out + x_off_out_root) {
                        max = loop_idx;
                        break;
                    }
                }
            }
        } else {
            if (base[loop_idx].out.xOffset() < x_off_out_root) {
                max = loop_idx;
                break;
            } else if (base[loop_idx].out.xOffset() > x_off_out_root) {
                x_left_out = true;
                dim_info.x_wait_out = loop_idx;
                dim_info.x_stride_out = base[loop_idx].out.xOffset() - x_off_out_root;
            }
        }
        if (a_left_in) {
            if (a_enter_in) {
                if (base[loop_idx].in.aOffset() != (loop_idx % dim_info.a_reset_in) / dim_info.a_wait_in * dim_info.a_stride_in + a_off_in_root) {
                    max = loop_idx;
                    break;
                }
            } else {
                if (base[loop_idx].in.aOffset() == a_off_in_root) {
                    a_enter_in = true;
                    dim_info.a_reset_in = loop_idx;
                } else {
                    if (base[loop_idx].in.aOffset() != (loop_idx / dim_info.a_wait_in) * dim_info.a_stride_in + a_off_in_root) {
                        max = loop_idx;
                        break;
                    }
                }
            }
        } else {
            if (base[loop_idx].in.aOffset() < a_off_in_root) {
                max = loop_idx;
                break;
            } else if (base[loop_idx].in.aOffset() > a_off_in_root) {
                a_left_in = true;
                dim_info.a_wait_in = loop_idx;
                dim_info.a_stride_in = base[loop_idx].in.aOffset() - a_off_in_root;
            }
        }
        if (z_left_in) {
            if (z_enter_in) {
                if (base[loop_idx].in.zOffset() != (loop_idx % dim_info.z_reset_in) / dim_info.z_wait_in * dim_info.z_stride_in + z_off_in_root) {
                    max = loop_idx;
                    break;
                }
            } else {
                if (base[loop_idx].in.zOffset() == z_off_in_root) {
                    z_enter_in = true;
                    dim_info.z_reset_in = loop_idx;
                } else {
                    if (base[loop_idx].in.zOffset() != (loop_idx / dim_info.z_wait_in) * dim_info.z_stride_in + z_off_in_root) {
                        max = loop_idx;
                        break;
                    }
                }
            }
        } else {
            if (base[loop_idx].in.zOffset() < z_off_in_root) {
                max = loop_idx;
                break;
            } else if (base[loop_idx].in.zOffset() > z_off_in_root) {
                z_left_in = true;
                dim_info.z_wait_in = loop_idx;
                dim_info.z_stride_in = base[loop_idx].in.zOffset() - z_off_in_root;
            }
        }
        if (y_left_in) {
            if (y_enter_in) {
                if (base[loop_idx].in.yOffset() != (loop_idx % dim_info.y_reset_in) / dim_info.y_wait_in * dim_info.y_stride_in + y_off_in_root) {
                    max = loop_idx;
                    break;
                }
            } else {
                if (base[loop_idx].in.yOffset() == y_off_in_root) {
                    y_enter_in = true;
                    dim_info.y_reset_in = loop_idx;
                } else {
                    if (base[loop_idx].in.yOffset() != (loop_idx / dim_info.y_wait_in) * dim_info.y_stride_in + y_off_in_root) {
                        max = loop_idx;
                        break;
                    }
                }
            }
        } else {
            if (base[loop_idx].in.yOffset() < y_off_in_root) {
                max = loop_idx;
                break;
            } else if (base[loop_idx].in.yOffset() > y_off_in_root) {
                y_left_in = true;
                dim_info.y_wait_in = loop_idx;
                dim_info.y_stride_in = base[loop_idx].in.yOffset() - y_off_in_root;
            }
        }
        if (x_left_in) {
            if (x_enter_in) {
                if (base[loop_idx].in.xOffset() != (loop_idx % dim_info.x_reset_in) / dim_info.x_wait_in * dim_info.x_stride_in + x_off_in_root) {
                    max = loop_idx;
                    break;
                }
            } else {
                if (base[loop_idx].in.xOffset() == x_off_in_root) {
                    x_enter_in = true;
                    dim_info.x_reset_in = loop_idx;
                } else {
                    if (base[loop_idx].in.xOffset() != (loop_idx / dim_info.x_wait_in) * dim_info.x_stride_in + x_off_in_root) {
                        max = loop_idx;
                        break;
                    }
                }
            }
        } else {
            if (base[loop_idx].in.xOffset() < x_off_in_root) {
                max = loop_idx;
                break;
            } else if (base[loop_idx].in.xOffset() > x_off_in_root) {
                x_left_in = true;
                dim_info.x_wait_in = loop_idx;
                dim_info.x_stride_in = base[loop_idx].in.xOffset() - x_off_in_root;
            }
        }
    }

    return max;
}

// $NOTE I don't think there is way to make this faster than O(n^2) unless I make a max loop size
pub fn parallelize(allocator: Allocator, ssa: *Ssa) !void {
    var temp_base: []Base = try allocator.alloc(Base, ssa.assign_num);
    errdefer allocator.free(temp_base);
    defer allocator.free(temp_base);

    var temp_remove: []bool = try allocator.alloc(bool, ssa.assign_num);
    errdefer allocator.free(temp_remove);
    defer allocator.free(temp_remove);
    @memset(temp_remove, false);

    var loop_id: u32 = 1;
    // $FIX Check for equality of inlined trees
    var assign_idx: u32 = 0;
    while (assign_idx < ssa.assign_num) {
        var loop_len: u32 = 0;
        var loop_num: u32 = 0;

        var assign_idx_search: u32 = assign_idx + 1;
        while (2 * assign_idx_search - assign_idx < ssa.assign_num) : (assign_idx_search += 1) {
            if (ssa.assign[assign_idx].base.equalNoOffset(ssa.assign[assign_idx_search].base)) {
                if (ssa.assign[assign_idx].base.out.overlapsPartial(ssa.assign[assign_idx_search].base.out)) {
                    break;
                } else {
                    var equal: bool = true;
                    for (0..assign_idx_search - assign_idx) |assign_off| {
                        const inlined_equal: bool = blk: {
                            if ((ssa.assign[assign_idx + assign_off].inlined == null) !=
                                (ssa.assign[assign_idx_search + assign_off].inlined == null)) break :blk false;
                            if (ssa.assign[assign_idx + assign_off].inlined == null) break :blk true;
                            break :blk ssa.assign[assign_idx + assign_off].inlined.?.inlinedEqualNoOffset( //
                                ssa.assign[assign_idx_search + assign_off].inlined.?);
                        };
                        if (!(ssa.assign[assign_idx + assign_off].base.equalNoOffset(ssa.assign[assign_idx_search + assign_off].base) and
                            inlined_equal) or
                            ssa.assign[assign_idx + assign_off].base.out.overlaps(ssa.assign[assign_idx_search + assign_off].base.out))
                        {
                            equal = false;
                            break;
                        }
                    }
                    if (equal) {
                        loop_len = assign_idx_search - assign_idx;
                        break;
                    } else {
                        continue;
                    }
                }
            }
        }

        if (loop_len == 0) {
            loop_len = 1;
            loop_num = 1;
        } else {
            loop_num = 1;
            for (1..@divFloor(ssa.assign_num - assign_idx, loop_len)) |loop_idx| {
                var equal: bool = true;
                // $TODO This is stupidly slow. There has to be a faster way to do this. This is so bad it might aswell be a FIXME
                // $TODO Also some overlaps might be ok if the things are intermediaries
                for (0..loop_len) |assign_off| blk: {
                    for (0..loop_num) |loop_idx_search| {
                        for (0..loop_len) |assign_off_search| {
                            if (assign_off == assign_off_search) {
                                const inlined_equal: bool = block: {
                                    if ((ssa.assign[assign_idx + loop_idx * loop_len + assign_off].inlined == null) !=
                                        (ssa.assign[assign_idx + loop_idx_search * loop_len + assign_off_search].inlined == null)) break :block false;
                                    if (ssa.assign[assign_idx + loop_idx * loop_len + assign_off].inlined == null) break :block true;
                                    break :block ssa.assign[assign_idx + loop_idx * loop_len + assign_off].inlined.?.inlinedEqualNoOffset( //
                                        ssa.assign[assign_idx + loop_idx_search * loop_len + assign_off_search].inlined.?);
                                };
                                if (!(ssa.assign[assign_idx + loop_idx_search * loop_len + assign_off_search].base.equalNoOffset( //
                                    ssa.assign[assign_idx + loop_idx * loop_len + assign_off].base) and inlined_equal) or
                                    ssa.assign[assign_idx + loop_idx_search * loop_len + assign_off_search].base.out.overlaps( //
                                    ssa.assign[assign_idx + loop_idx * loop_len + assign_off].base.out))
                                {
                                    equal = false;
                                    break :blk;
                                }
                            } else {
                                if (ssa.assign[assign_idx + loop_idx_search * loop_len + assign_off_search].base.out.name_offset ==
                                    ssa.assign[assign_idx + loop_idx * loop_len + assign_off].base.out.name_offset and
                                    ssa.assign[assign_idx + loop_idx_search * loop_len + assign_off_search].base.out.overlaps( //
                                    ssa.assign[assign_idx + loop_idx * loop_len + assign_off].base.out))
                                {
                                    equal = false;
                                    break :blk;
                                }
                            }
                        }
                        if (!equal) {
                            break;
                        }
                    }
                }
                if (equal) {
                    loop_num += 1;
                } else {
                    break;
                }
            }

            for (0..loop_len) |inner_idx| {
                for (0..loop_num) |loop_idx| {
                    temp_base[loop_idx] = ssa.assign[assign_idx + inner_idx + loop_idx * loop_len].base;
                }
                loop_num = @min(loop_num, dimInfoMaxLegal(temp_base[0..loop_num]));

                if (ssa.assign[assign_idx + inner_idx].inlined) |*inlined| {
                    for (0..inlined.inlined_num) |inlined_idx| {
                        for (0..loop_num) |loop_idx| {
                            temp_base[loop_idx] = ssa.assign[assign_idx + inner_idx + loop_idx * loop_len].inlined.?.base[inlined_idx];
                        }
                        loop_num = @min(loop_num, dimInfoMaxLegal(temp_base[0..loop_num]));
                    }
                }
            }

            for (0..loop_len) |inner_idx| {
                for (0..loop_num) |loop_idx| {
                    temp_base[loop_idx] = ssa.assign[assign_idx + inner_idx + loop_idx * loop_len].base;
                    if (loop_idx != 0) {
                        temp_remove[assign_idx + inner_idx + loop_idx * loop_len] = true;
                    }
                }
                ssa.assign[assign_idx + inner_idx].base.dim_info = DimInfo.init(temp_base[0..loop_num]);
                ssa.assign_loop_id[assign_idx + inner_idx] = loop_id;

                if (ssa.assign[assign_idx + inner_idx].inlined) |*inlined| {
                    for (0..inlined.inlined_num) |inlined_idx| {
                        for (0..loop_num) |loop_idx| {
                            temp_base[loop_idx] = ssa.assign[assign_idx + inner_idx + loop_idx * loop_len].inlined.?.base[inlined_idx];
                        }
                        inlined.base[inlined_idx].dim_info = DimInfo.init(temp_base[0..loop_num]);
                    }
                }

                ssa.assign_loop_num[loop_id] = loop_num;
            }

            loop_id += 1;
        }

        assert(loop_len >= 1);
        assert(loop_num >= 1);
        assign_idx += loop_len * loop_num;
    }
    var assign_num_new: u32 = 0;
    assign_idx = 0;
    while (assign_idx < ssa.assign_num) : (assign_idx += 1) {
        if (temp_remove[assign_idx]) {
            if (ssa.assign[assign_idx].inlined) |*inlined| {
                allocator.free(inlined.base);
                allocator.free(inlined.out);
                allocator.free(inlined.in);
            }
        } else {
            ssa.assign[assign_num_new] = ssa.assign[assign_idx];
            ssa.assign_loop_id[assign_num_new] = ssa.assign_loop_id[assign_idx];
            assign_num_new += 1;
        }
    }
    ssa.assign_num = assign_num_new;
}
pub fn splitKernel(allocator: Allocator, ssa: *Ssa) !void {
    _ = allocator;
    _ = ssa;
}
pub fn simd(allocator: Allocator, ssa: *Ssa) !void {
    _ = allocator;
    _ = ssa;
}
pub fn memoryLayout(allocator: Allocator, ssa: *Ssa) !void {
    _ = allocator;
    _ = ssa;
}
