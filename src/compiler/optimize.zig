// $TODO Maybe just do the parallelize one at O0 anyways, because it is utterly useless without it
// $TODO These levels
// $TODO Expressive numerical representation of an optimization such that a casey type optimizer is possible
// Optimization levels
// O1 - parallelize, split, idempotent functions
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
        if ((assign[start_idx].base.out.id == assign[assign_idx].base.out.id and
            assign[start_idx].base.out.overlapsPartial(assign[assign_idx].base.out)) or
            (assign[start_idx].base.out.id == assign[assign_idx].base.in.id and
                assign[start_idx].base.out.overlapsPartial(assign[assign_idx].base.in)) or
            (assign[start_idx].base.in.id == assign[assign_idx].base.out.id and
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
            assign[assign_idx].inlined.?.out[inlined_num_new - 1] = if (out_root_old) |out| out + inlined_num_old else null;
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

            assign[assign_idx].inlined.?.in[inlined_num_new - 1] = if (in_root_old) |in| in + inlined_num_old else null;
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

// $TODO Support reordering
fn dimInfoMergePossible(base: Base, merge: Base) bool {
    assert(base.out.id == merge.out.id);
    assert(base.in.id == merge.in.id);

    // $NOTE Can't just check the total offset because base=[0,1,0,0] and merge=[1,0,0,0] would be allowed then
    const out_offset_invalid: bool = base.out.aOffset() < base.out.aOffset() or base.out.zOffset() < base.out.zOffset() or
        base.out.yOffset() < base.out.yOffset() or base.out.xOffset() < base.out.xOffset();
    const in_offset_invalid: bool = base.in.aOffset() < base.in.aOffset() or base.in.zOffset() < base.in.zOffset() or
        base.in.yOffset() < base.in.yOffset() or base.in.xOffset() < base.in.xOffset();
    if (out_offset_invalid or in_offset_invalid) return false;

    inline for (0..8) |dim_idx| {
        const wait: u32 = switch (dim_idx) {
            0 => base.dim_info.a_wait_out,
            1 => base.dim_info.z_wait_out,
            2 => base.dim_info.y_wait_out,
            3 => base.dim_info.x_wait_out,
            4 => base.dim_info.a_wait_in,
            5 => base.dim_info.z_wait_in,
            6 => base.dim_info.y_wait_in,
            7 => base.dim_info.x_wait_in,
        };
        const stride: u32 = switch (dim_idx) {
            0 => base.dim_info.a_stride_out,
            1 => base.dim_info.z_stride_out,
            2 => base.dim_info.y_stride_out,
            3 => base.dim_info.x_stride_out,
            4 => base.dim_info.a_stride_in,
            5 => base.dim_info.z_stride_in,
            6 => base.dim_info.y_stride_in,
            7 => base.dim_info.x_stride_in,
        };
        const reset: u32 = switch (dim_idx) {
            0 => base.dim_info.a_reset_out,
            1 => base.dim_info.z_reset_out,
            2 => base.dim_info.y_reset_out,
            3 => base.dim_info.x_reset_out,
            4 => base.dim_info.a_reset_in,
            5 => base.dim_info.z_reset_in,
            6 => base.dim_info.y_reset_in,
            7 => base.dim_info.x_reset_in,
        };
        const off_base: u32 = switch (dim_idx) {
            0 => base.dim_info.off_out,
            1 => base.dim_info.off_out,
            2 => base.dim_info.off_out,
            3 => base.dim_info.off_out,
            4 => base.dim_info.off_in,
            5 => base.dim_info.off_in,
            6 => base.dim_info.off_in,
            7 => base.dim_info.off_in,
        };
        const off_merge: u32 = switch (dim_idx) {
            0 => merge.dim_info.off_out,
            1 => merge.dim_info.off_out,
            2 => merge.dim_info.off_out,
            3 => merge.dim_info.off_out,
            4 => merge.dim_info.off_in,
            5 => merge.dim_info.off_in,
            6 => merge.dim_info.off_in,
            7 => merge.dim_info.off_in,
        };
        if (wait == DimInfo.value_none) {
            assert(stride == DimInfo.value_none);
        } else {
            assert(stride != DimInfo.value_none);
            if (reset == DimInfo.value_none) {
                if (@divFloor(base.dim_info.repeats + 1, wait) * stride != off_merge and off_base != off_merge) {
                    return false;
                }
            } else {
                if (@divFloor((base.dim_info.repeats + 1) % reset, wait) * stride != off_merge) {
                    return false;
                }
            }
        }
    }

    return true;
}
fn dimInfoMerge(base: DimInfo, merge: Base) DimInfo {
    _ = base;
    _ = merge;
    return undefined;
}

/// Returns 0 in case nothing was parallelized, assign_loop_id in case an assign got added to the an existant loop and assign_loop_id + 1 in case a new one had to be created
fn parallelizeStep(ssa: *Ssa, start_idx: u32, assign_loop_id: u32) u32 {
    var assign_idx: u32 = start_idx;
    while (assign_idx < ssa.assign_num) : (assign_idx += 1) {
        if (ssa.assign[start_idx].base.out.id == ssa.assign[assign_idx].base.out.id and
            ssa.assign[start_idx].base.out.overlaps(ssa.assign[assign_idx].base.out)) break;
        if (ssa.assign[start_idx].base.out.id == ssa.assign[assign_idx].base.in.id and
            ssa.assign[start_idx].base.out.overlaps(ssa.assign[assign_idx].base.in)) break;

        if (ssa.assign[start_idx].base.out.id == ssa.assign[assign_idx].base.out.id and
            ssa.assign[start_idx].base.in.id == ssa.assign[assign_idx].base.in.id and
            dimInfoMergePossible(ssa.assign[start_idx].base, ssa.assign[assign_idx].base))
        {
            ssa.assign[assign_idx].base.dim_info = dimInfoMerge(ssa.assign[start_idx].base.dim_info, ssa.assign[assign_idx]);
            if (ssa.assign_loop_id[start_idx] == 0) {
                ssa.assign_loop_id[assign_idx] = assign_loop_id;
                return assign_loop_id + 1;
            } else {
                ssa.assign_loop_id[assign_idx] = ssa.assign_loop_id[start_idx];
                return assign_loop_id;
            }
        }
    }
    return 0;
}
// $NOTE I don't think there is way to make this faster than O(n^2) unless I make a max loop size, which sucks for large SSAs
pub fn parallelize(allocator: Allocator, ssa: *Ssa) !void {
    var temp_remove: []bool = try allocator.alloc(bool, ssa.assign_num);
    errdefer allocator.free(temp_remove);
    defer allocator.free(temp_remove);

    var assign_idx: u32 = 0;
    var assign_loop_id: u32 = 1;
    while (assign_idx < ssa.assign_num) : (assign_idx += 1) {
        const assign_loop_id_writen: u32 = parallelizeStep(ssa, assign_idx, assign_loop_id);
        if (assign_loop_id_writen != 0) {
            assign_loop_id = assign_loop_id_writen;
            temp_remove[assign_idx] = true;
        } else {
            temp_remove[assign_idx] = false;
        }
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
// pub fn parallelize(allocator: Allocator, ssa: *Ssa) !void {
//     var temp_base: []Base = try allocator.alloc(Base, ssa.assign_num);
//     errdefer allocator.free(temp_base);
//     defer allocator.free(temp_base);
//
//     var temp_remove: []bool = try allocator.alloc(bool, ssa.assign_num);
//     errdefer allocator.free(temp_remove);
//     defer allocator.free(temp_remove);
//     @memset(temp_remove, false);
//
//     var loop_id: u32 = 1;
//     var assign_idx: u32 = 0;
//     while (assign_idx < ssa.assign_num) {
//         var loop_len: u32 = 0;
//         var loop_num: u32 = 0;
//
//         var assign_idx_search: u32 = assign_idx + 1;
//         while (2 * assign_idx_search - assign_idx < ssa.assign_num) : (assign_idx_search += 1) {
//             if (ssa.assign[assign_idx].base.equalNoOffset(ssa.assign[assign_idx_search].base)) {
//                 if (ssa.assign[assign_idx].base.out.overlapsPartial(ssa.assign[assign_idx_search].base.out)) {
//                     break;
//                 } else {
//                     var equal: bool = true;
//                     for (0..assign_idx_search - assign_idx) |assign_off| {
//                         const inlined_equal: bool = blk: {
//                             if ((ssa.assign[assign_idx + assign_off].inlined == null) !=
//                                 (ssa.assign[assign_idx_search + assign_off].inlined == null)) break :blk false;
//                             if (ssa.assign[assign_idx + assign_off].inlined == null) break :blk true;
//                             break :blk ssa.assign[assign_idx + assign_off].inlined.?.inlinedEqualNoOffset( //
//                                 ssa.assign[assign_idx_search + assign_off].inlined.?);
//                         };
//                         if (!(ssa.assign[assign_idx + assign_off].base.equalNoOffset(ssa.assign[assign_idx_search + assign_off].base) and
//                             inlined_equal) or
//                             ssa.assign[assign_idx + assign_off].base.out.overlaps(ssa.assign[assign_idx_search + assign_off].base.out))
//                         {
//                             equal = false;
//                             break;
//                         }
//                     }
//                     if (equal) {
//                         loop_len = assign_idx_search - assign_idx;
//                         break;
//                     } else {
//                         continue;
//                     }
//                 }
//             }
//         }
//
//         if (loop_len == 0) {
//             loop_len = 1;
//             loop_num = 1;
//         } else {
//             loop_num = 1;
//             for (1..@divFloor(ssa.assign_num - assign_idx, loop_len)) |loop_idx| {
//                 var equal: bool = true;
//                 // $TODO This is stupidly slow. There has to be a faster way to do this. This is so bad it might aswell be a FIXME
//                 // $TODO Also some overlaps might be ok if the things are intermediaries
//                 for (0..loop_len) |assign_off| blk: {
//                     for (0..loop_num) |loop_idx_search| {
//                         for (0..loop_len) |assign_off_search| {
//                             if (assign_off == assign_off_search) {
//                                 const inlined_equal: bool = block: {
//                                     if ((ssa.assign[assign_idx + loop_idx * loop_len + assign_off].inlined == null) !=
//                                         (ssa.assign[assign_idx + loop_idx_search * loop_len + assign_off_search].inlined == null)) break :block false;
//                                     if (ssa.assign[assign_idx + loop_idx * loop_len + assign_off].inlined == null) break :block true;
//                                     break :block ssa.assign[assign_idx + loop_idx * loop_len + assign_off].inlined.?.inlinedEqualNoOffset( //
//                                         ssa.assign[assign_idx + loop_idx_search * loop_len + assign_off_search].inlined.?);
//                                 };
//                                 if (!(ssa.assign[assign_idx + loop_idx_search * loop_len + assign_off_search].base.equalNoOffset( //
//                                     ssa.assign[assign_idx + loop_idx * loop_len + assign_off].base) and inlined_equal) or
//                                     ssa.assign[assign_idx + loop_idx_search * loop_len + assign_off_search].base.out.overlaps( //
//                                         ssa.assign[assign_idx + loop_idx * loop_len + assign_off].base.out))
//                                 {
//                                     equal = false;
//                                     break :blk;
//                                 }
//                             } else {
//                                 if (ssa.assign[assign_idx + loop_idx_search * loop_len + assign_off_search].base.out.id ==
//                                     ssa.assign[assign_idx + loop_idx * loop_len + assign_off].base.out.id and
//                                     ssa.assign[assign_idx + loop_idx_search * loop_len + assign_off_search].base.out.overlaps( //
//                                         ssa.assign[assign_idx + loop_idx * loop_len + assign_off].base.out))
//                                 {
//                                     equal = false;
//                                     break :blk;
//                                 }
//                                 // $NOTE / $FIXME I hate this condition, but it fixes rng=1745145740864090 opt=O1.
//                                 // Maybe there is a less restrictive condition
//                                 if (ssa.assign[assign_idx + loop_idx_search * loop_len + assign_off_search].base.in.id ==
//                                     ssa.assign[assign_idx + loop_idx * loop_len + assign_off].base.out.id and
//                                     ssa.assign[assign_idx + loop_idx_search * loop_len + assign_off_search].base.in.overlaps( //
//                                         ssa.assign[assign_idx + loop_idx * loop_len + assign_off].base.out))
//                                 {
//                                     equal = false;
//                                     break :blk;
//                                 }
//                                 // $NOTE / $FIXME I doubly hate this condition, but it fixes rng=1748555540748849 opt=O1.
//                                 // Maybe there is a less restrictive condition x2. This one happens because not every combination is tested symmetrically, which sucks
//                                 //  but isn't trivially fixed
//                                 if (ssa.assign[assign_idx + loop_idx_search * loop_len + assign_off_search].base.out.id ==
//                                     ssa.assign[assign_idx + loop_idx * loop_len + assign_off].base.in.id and
//                                     ssa.assign[assign_idx + loop_idx_search * loop_len + assign_off_search].base.out.overlaps( //
//                                         ssa.assign[assign_idx + loop_idx * loop_len + assign_off].base.in))
//                                 {
//                                     equal = false;
//                                     break :blk;
//                                 }
//                             }
//                         }
//                         if (!equal) {
//                             break;
//                         }
//                     }
//                 }
//                 if (equal) {
//                     loop_num += 1;
//                 } else {
//                     break;
//                 }
//             }
//
//             for (0..loop_len) |inner_idx| {
//                 for (0..loop_num) |loop_idx| {
//                     temp_base[loop_idx] = ssa.assign[assign_idx + inner_idx + loop_idx * loop_len].base;
//                 }
//                 loop_num = @min(loop_num, dimInfoMaxLegal(temp_base[0..loop_num]));
//
//                 if (ssa.assign[assign_idx + inner_idx].inlined) |*inlined| {
//                     for (0..inlined.inlined_num) |inlined_idx| {
//                         for (0..loop_num) |loop_idx| {
//                             temp_base[loop_idx] = ssa.assign[assign_idx + inner_idx + loop_idx * loop_len].inlined.?.base[inlined_idx];
//                         }
//                         loop_num = @min(loop_num, dimInfoMaxLegal(temp_base[0..loop_num]));
//                     }
//                 }
//             }
//
//             for (0..loop_len) |inner_idx| {
//                 for (0..loop_num) |loop_idx| {
//                     temp_base[loop_idx] = ssa.assign[assign_idx + inner_idx + loop_idx * loop_len].base;
//                     if (loop_idx != 0) {
//                         temp_remove[assign_idx + inner_idx + loop_idx * loop_len] = true;
//                     }
//                 }
//                 ssa.assign[assign_idx + inner_idx].base.dim_info = DimInfo.init(temp_base[0..loop_num]);
//                 ssa.assign_loop_id[assign_idx + inner_idx] = loop_id;
//
//                 if (ssa.assign[assign_idx + inner_idx].inlined) |*inlined| {
//                     for (0..inlined.inlined_num) |inlined_idx| {
//                         for (0..loop_num) |loop_idx| {
//                             temp_base[loop_idx] = ssa.assign[assign_idx + inner_idx + loop_idx * loop_len].inlined.?.base[inlined_idx];
//                         }
//                         inlined.base[inlined_idx].dim_info = DimInfo.init(temp_base[0..loop_num]);
//                     }
//                 }
//
//                 ssa.assign_loop_num[loop_id] = loop_num;
//             }
//
//             loop_id += 1;
//         }
//
//         assert(loop_len >= 1);
//         assert(loop_num >= 1);
//         assign_idx += loop_len * loop_num;
//     }
//     var assign_num_new: u32 = 0;
//     assign_idx = 0;
//     while (assign_idx < ssa.assign_num) : (assign_idx += 1) {
//         if (temp_remove[assign_idx]) {
//             if (ssa.assign[assign_idx].inlined) |*inlined| {
//                 allocator.free(inlined.base);
//                 allocator.free(inlined.out);
//                 allocator.free(inlined.in);
//             }
//         } else {
//             ssa.assign[assign_num_new] = ssa.assign[assign_idx];
//             ssa.assign_loop_id[assign_num_new] = ssa.assign_loop_id[assign_idx];
//             assign_num_new += 1;
//         }
//     }
//     ssa.assign_num = assign_num_new;
// }
