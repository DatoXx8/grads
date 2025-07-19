// $TODO Expressive numerical representation of an optimization such that a casey type optimizer is possible
// $TODO These levels
// Optimization levels
// O1 - split
// O2 - SIMD, float range based analysis
// O3 - memory optimizer

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const Tensor = @import("../Tensor.zig");
const Op = Tensor.Op;
const Buffer = Tensor.Buffer;
const Pir = @import("./Pir.zig");
const Assign = Pir.Assign;
const Base = Pir.Base;
const DimInfo = Pir.DimInfo;

/// $WARN O0 is **really** slow
/// Planned optimization steps
/// O0 - none
/// O1 - parallelize, inline, split
/// O2 - SIMD, float range based analysis
/// O3 - memory optimizer
pub const Optimization = enum(u8) { O0, O1, O2, O3 };

fn inlineOpStep(allocator: Allocator, assign: []Assign, start_idx: u32) !bool {
    assert(start_idx + 1 < assign.len);

    if (assign[start_idx].base.type.isReduce()) return false;

    var out_found: bool = false;
    var in_found: bool = false;
    var assign_idx: u32 = start_idx + 1;
    while (assign_idx < assign.len) : (assign_idx += 1) {
        // I don't think there is no way to handle partial overlaps. AFAICT you just need to burn the whole thing down if you find that case
        if ((assign[start_idx].base.out.id == assign[assign_idx].base.out.id and
            assign[start_idx].base.out.overlapsPartial(assign[assign_idx].base.out)) or
            (assign[start_idx].base.out.id == assign[assign_idx].base.in.id and
                assign[start_idx].base.out.overlapsPartial(assign[assign_idx].base.in)) or
            (assign[start_idx].base.in.id == assign[assign_idx].base.out.id and
                assign[start_idx].base.in.overlaps(assign[assign_idx].base.out) and
                !assign[start_idx].base.type.isUnary()))
        {
            return false;
        }
        // If I didn't do it like this I would have to check every single already inlined op for overlaps, which I think is more expensive than testing these conditions here.
        if (assign[start_idx].base.out.equal(assign[assign_idx].base.out)) {
            out_found = true;
        }
        if (assign[start_idx].base.in.equal(assign[assign_idx].base.out) and !assign[start_idx].base.type.isUnary()) {
            in_found = true;
        }
        if (out_found and in_found) {
            break;
        }
    }

    var written: bool = false;
    assign_idx = start_idx + 1;
    while (assign_idx < assign.len) : (assign_idx += 1) {
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
pub fn inlineOp(allocator: Allocator, pir: *Pir) !void {
    const temp_written: []bool = try allocator.alloc(bool, pir.assign_num);
    errdefer allocator.free(temp_written);
    defer allocator.free(temp_written);
    @memset(temp_written, false);

    var start_idx: u32 = 0;
    while (start_idx < pir.assign_num - 1) : (start_idx += 1) {
        temp_written[start_idx] = try inlineOpStep(allocator, pir.assign, start_idx);
    }

    var assign_idx: u32 = 0;
    var assign_num_new: u32 = 0;
    while (assign_idx < pir.assign_num) : (assign_idx += 1) {
        if (temp_written[assign_idx]) {
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

// $FIX / $TODO / $PERF / $HACK         don't even know what to mark this shit with
// If you are unlucky with the layout of your offsets then you can get into a situation where the offsets for each assign can't be modeled by a linear function.
// This is a huge issue because other functions that model the offsets can't be found easily and table driven solutions limit the loop size artificially,
// because of the limits on local memory per kernel.
// As a hacky fix we just split the loop up if there is something that can't be modeled linearly. This sucks bad. I hate that I have to do this.
// I am sorry for this terrible shittines, I just can't think of a better solution right now

// $TODO Support reordering
fn dimInfoMergePossible(base: Assign, merge: Assign) bool {
    if ((if (base.inlined) |i| i.inlined_num else 0) != (if (merge.inlined) |i| i.inlined_num else 0)) {
        return false;
    }

    const base_num: u32 = 1 + if (base.inlined) |i| i.inlined_num else 0;

    for (0..base_num) |base_idx| {
        const pre: Base = if (base_idx == 0) base.base else base.inlined.?.base[base_idx - 1];
        const post: Base = if (base_idx == 0) merge.base else merge.inlined.?.base[base_idx - 1];
        if (!pre.equalNoOffset(post)) {
            return false;
        }

        inline for (0..8) |dim_idx| {
            const wait: u32 = switch (dim_idx) {
                0 => pre.out_dim.a_wait,
                1 => pre.out_dim.z_wait,
                2 => pre.out_dim.y_wait,
                3 => pre.out_dim.x_wait,
                4 => pre.in_dim.a_wait,
                5 => pre.in_dim.z_wait,
                6 => pre.in_dim.y_wait,
                7 => pre.in_dim.x_wait,
                else => unreachable,
            };
            const stride: u32 = switch (dim_idx) {
                0 => pre.out_dim.a_stride,
                1 => pre.out_dim.z_stride,
                2 => pre.out_dim.y_stride,
                3 => pre.out_dim.x_stride,
                4 => pre.in_dim.a_stride,
                5 => pre.in_dim.z_stride,
                6 => pre.in_dim.y_stride,
                7 => pre.in_dim.x_stride,
                else => unreachable,
            };
            const reset: u32 = switch (dim_idx) {
                0 => pre.out_dim.a_reset,
                1 => pre.out_dim.z_reset,
                2 => pre.out_dim.y_reset,
                3 => pre.out_dim.x_reset,
                4 => pre.in_dim.a_reset,
                5 => pre.in_dim.z_reset,
                6 => pre.in_dim.y_reset,
                7 => pre.in_dim.x_reset,
                else => unreachable,
            };
            const off_pre: u32 = switch (dim_idx) {
                0 => pre.out.aOffset(),
                1 => pre.out.zOffset(),
                2 => pre.out.yOffset(),
                3 => pre.out.xOffset(),
                4 => pre.in.aOffset(),
                5 => pre.in.zOffset(),
                6 => pre.in.yOffset(),
                7 => pre.in.xOffset(),
                else => unreachable,
            };
            const off_merge: u32 = switch (dim_idx) {
                0 => post.out.aOffset(),
                1 => post.out.zOffset(),
                2 => post.out.yOffset(),
                3 => post.out.xOffset(),
                4 => post.in.aOffset(),
                5 => post.in.zOffset(),
                6 => post.in.yOffset(),
                7 => post.in.xOffset(),
                else => unreachable,
            };

            if (off_merge < off_pre) {
                return false;
            }

            if (wait == DimInfo.value_none) {
                assert(stride == DimInfo.value_none);
            } else {
                assert(stride != DimInfo.value_none);
                if (reset == DimInfo.value_none) {
                    if (off_pre != off_merge and @divFloor(pre.repeats, wait) * stride != (off_merge - off_pre)) {
                        return false;
                    }
                } else {
                    if (@divFloor((pre.repeats) % reset, wait) * stride != (off_merge - off_pre)) {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}
fn dimInfoMerge(base: Assign, merge: *Assign) void {
    assert((base.inlined == null) == (merge.inlined == null));
    assert((if (base.inlined) |i| i.inlined_num else 0) == (if (merge.inlined) |i| i.inlined_num else 0));
    // assert(dimInfoMergePossible(base, merge.*)); // Might be a bit slow to check every time

    const base_num: u32 = 1 + if (base.inlined) |i| i.inlined_num else 0;

    for (0..base_num) |base_idx| {
        const pre: *const Base = if (base_idx == 0) &base.base else &base.inlined.?.base[base_idx - 1];
        const post: *Base = if (base_idx == 0) &merge.base else &merge.inlined.?.base[base_idx - 1];

        inline for (0..2) |buffer_idx| {
            const pre_buffer: Buffer = if (buffer_idx == 0) pre.out else pre.in;
            const post_buffer: Buffer = if (buffer_idx == 0) post.out else post.in;
            var modified: DimInfo = if (buffer_idx == 0) pre.out_dim else pre.in_dim;
            inline for (0..4) |dim_idx| {
                const wait: *u32 = switch (dim_idx) {
                    0 => &modified.a_wait,
                    1 => &modified.z_wait,
                    2 => &modified.y_wait,
                    3 => &modified.x_wait,
                    else => unreachable,
                };
                const stride: *u32 = switch (dim_idx) {
                    0 => &modified.a_stride,
                    1 => &modified.z_stride,
                    2 => &modified.y_stride,
                    3 => &modified.x_stride,
                    else => unreachable,
                };
                const reset: *u32 = switch (dim_idx) {
                    0 => &modified.a_reset,
                    1 => &modified.z_reset,
                    2 => &modified.y_reset,
                    3 => &modified.x_reset,
                    else => unreachable,
                };
                const off_pre: u32 = switch (dim_idx) {
                    0 => pre_buffer.aOffset(),
                    1 => pre_buffer.zOffset(),
                    2 => pre_buffer.yOffset(),
                    3 => pre_buffer.xOffset(),
                    else => unreachable,
                };
                const off_post: u32 = switch (dim_idx) {
                    0 => post_buffer.aOffset(),
                    1 => post_buffer.zOffset(),
                    2 => post_buffer.yOffset(),
                    3 => post_buffer.xOffset(),
                    else => unreachable,
                };
                if (wait.* == DimInfo.value_none) {
                    assert(stride.* == DimInfo.value_none);
                    if (off_pre != off_post) {
                        assert(off_pre < off_post);
                        wait.* = pre.repeats;
                        stride.* = off_post - off_pre;
                    }
                } else {
                    assert(stride.* != DimInfo.value_none);
                    if (reset.* == DimInfo.value_none) {
                        if (off_pre == off_post) {
                            reset.* = pre.repeats; // $FIXME This might need to be repeats + 1
                        }
                    }
                }
            }

            if (buffer_idx == 0) {
                post.out_dim = modified;
            } else {
                post.in_dim = modified;
            }
        }

        post.out = pre.out;
        post.in = pre.in;
        post.repeats = pre.repeats + 1;
    }
}

// Assumes `this` and `base` hold the base offsets
fn dimInfoOverlap(this: Buffer, this_dim: DimInfo, this_repeats: u32, target: Buffer, target_dim: DimInfo, target_repeats: u32) bool {
    const this_a_reset: u32 = if (this_dim.a_reset == DimInfo.value_none) DimInfo.reset_default else this_dim.a_reset;
    const this_z_reset: u32 = if (this_dim.z_reset == DimInfo.value_none) DimInfo.reset_default else this_dim.z_reset;
    const this_y_reset: u32 = if (this_dim.y_reset == DimInfo.value_none) DimInfo.reset_default else this_dim.y_reset;
    const this_x_reset: u32 = if (this_dim.x_reset == DimInfo.value_none) DimInfo.reset_default else this_dim.x_reset;
    const this_a_wait: u32 = if (this_dim.a_wait == DimInfo.value_none) DimInfo.wait_default else this_dim.a_wait;
    const this_z_wait: u32 = if (this_dim.z_wait == DimInfo.value_none) DimInfo.wait_default else this_dim.z_wait;
    const this_y_wait: u32 = if (this_dim.y_wait == DimInfo.value_none) DimInfo.wait_default else this_dim.y_wait;
    const this_x_wait: u32 = if (this_dim.x_wait == DimInfo.value_none) DimInfo.wait_default else this_dim.x_wait;
    const this_a_stride: u32 = if (this_dim.a_stride == DimInfo.value_none) DimInfo.stride_default else this_dim.a_stride;
    const this_z_stride: u32 = if (this_dim.z_stride == DimInfo.value_none) DimInfo.stride_default else this_dim.z_stride;
    const this_y_stride: u32 = if (this_dim.y_stride == DimInfo.value_none) DimInfo.stride_default else this_dim.y_stride;
    const this_x_stride: u32 = if (this_dim.x_stride == DimInfo.value_none) DimInfo.stride_default else this_dim.x_stride;

    const target_a_reset: u32 = if (target_dim.a_reset == DimInfo.value_none) DimInfo.reset_default else target_dim.a_reset;
    const target_z_reset: u32 = if (target_dim.z_reset == DimInfo.value_none) DimInfo.reset_default else target_dim.z_reset;
    const target_y_reset: u32 = if (target_dim.y_reset == DimInfo.value_none) DimInfo.reset_default else target_dim.y_reset;
    const target_x_reset: u32 = if (target_dim.x_reset == DimInfo.value_none) DimInfo.reset_default else target_dim.x_reset;
    const target_a_wait: u32 = if (target_dim.a_wait == DimInfo.value_none) DimInfo.wait_default else target_dim.a_wait;
    const target_z_wait: u32 = if (target_dim.z_wait == DimInfo.value_none) DimInfo.wait_default else target_dim.z_wait;
    const target_y_wait: u32 = if (target_dim.y_wait == DimInfo.value_none) DimInfo.wait_default else target_dim.y_wait;
    const target_x_wait: u32 = if (target_dim.x_wait == DimInfo.value_none) DimInfo.wait_default else target_dim.x_wait;
    const target_a_stride: u32 = if (target_dim.a_stride == DimInfo.value_none) DimInfo.stride_default else target_dim.a_stride;
    const target_z_stride: u32 = if (target_dim.z_stride == DimInfo.value_none) DimInfo.stride_default else target_dim.z_stride;
    const target_y_stride: u32 = if (target_dim.y_stride == DimInfo.value_none) DimInfo.stride_default else target_dim.y_stride;
    const target_x_stride: u32 = if (target_dim.x_stride == DimInfo.value_none) DimInfo.stride_default else target_dim.x_stride;

    var this_idx: u32 = 0;
    while (this_idx < this_repeats) : (this_idx += 1) {
        const a_1: u32 = this.aOffset() + (this_idx % this_a_reset) / this_a_wait * this_a_stride;
        const z_1: u32 = this.zOffset() + (this_idx % this_z_reset) / this_z_wait * this_z_stride;
        const y_1: u32 = this.yOffset() + (this_idx % this_y_reset) / this_y_wait * this_y_stride;
        const x_1: u32 = this.xOffset() + (this_idx % this_x_reset) / this_x_wait * this_x_stride;

        var target_idx: u32 = 0;
        while (target_idx < target_repeats) : (target_idx += 1) {
            const a_2: u32 = target.aOffset() + (target_idx % target_a_reset) / target_a_wait * target_a_stride;
            const z_2: u32 = target.zOffset() + (target_idx % target_z_reset) / target_z_wait * target_z_stride;
            const y_2: u32 = target.yOffset() + (target_idx % target_y_reset) / target_y_wait * target_y_stride;
            const x_2: u32 = target.xOffset() + (target_idx % target_x_reset) / target_x_wait * target_x_stride;
            const overlap: bool = @max(a_1, a_2) < @min(a_1 + this.a_size, a_2 + target.a_size) and
                @max(z_1, z_2) < @min(z_1 + this.z_size, z_2 + target.z_size) and
                @max(y_1, y_2) < @min(y_1 + this.y_size, y_2 + target.y_size) and
                @max(x_1, x_2) < @min(x_1 + this.x_size, x_2 + target.x_size);
            if (overlap) {
                return true;
            }
        }
    }
    return false;
}

/// Returns 0 in case nothing was parallelized, assign_loop_id in case an assign got added to the an existant loop and assign_loop_id + 1 in case a new one had to be created
fn parallelizeStep(pir: *Pir, start_idx: u32) bool {
    var assign_idx: u32 = start_idx + 1;

    while (assign_idx < pir.assign_num) : (assign_idx += 1) {
        // I think these conditions are the least restrictive they could be because of the inlining that happens before parallelization
        // At the very least it is not obvious to me how to loosen them.
        const overlap_out_out: bool = pir.assign[start_idx].base.out.id == pir.assign[assign_idx].base.out.id and
            dimInfoOverlap(pir.assign[start_idx].base.out, pir.assign[start_idx].base.out_dim, pir.assign[start_idx].base.repeats, //
                pir.assign[assign_idx].base.out, pir.assign[assign_idx].base.out_dim, pir.assign[assign_idx].base.repeats);
        const overlap_out_in: bool = pir.assign[start_idx].base.out.id == pir.assign[assign_idx].base.in.id and
            dimInfoOverlap(pir.assign[start_idx].base.out, pir.assign[start_idx].base.out_dim, pir.assign[start_idx].base.repeats, //
                pir.assign[assign_idx].base.in, pir.assign[assign_idx].base.in_dim, pir.assign[assign_idx].base.repeats);
        const overlap_in_out: bool = pir.assign[start_idx].base.in.id == pir.assign[assign_idx].base.out.id and
            dimInfoOverlap(pir.assign[start_idx].base.in, pir.assign[start_idx].base.in_dim, pir.assign[start_idx].base.repeats, //
                pir.assign[assign_idx].base.out, pir.assign[assign_idx].base.out_dim, pir.assign[assign_idx].base.repeats);
        const overlap_inline: bool = blk: {
            if (pir.assign[assign_idx].inlined) |inlined| {
                for (0..inlined.inlined_num) |inlined_idx| {
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
            break;
        }

        if (dimInfoMergePossible(pir.assign[start_idx], pir.assign[assign_idx])) {
            dimInfoMerge(pir.assign[start_idx], &pir.assign[assign_idx]);
            return true;
        }
    }
    return false;
}
// I don't think there is way to make this faster than O(n^2) unless I make a max loop size, which sucks for large SSAs
pub fn parallelize(allocator: Allocator, pir: *Pir) !void {
    var temp_remove: []bool = try allocator.alloc(bool, pir.assign_num);
    errdefer allocator.free(temp_remove);
    defer allocator.free(temp_remove);

    var assign_idx: u32 = 0;
    while (assign_idx < pir.assign_num) : (assign_idx += 1) {
        temp_remove[assign_idx] = parallelizeStep(pir, assign_idx);
    }

    var assign_num_new: u32 = 0;
    assign_idx = 0;
    while (assign_idx < pir.assign_num) : (assign_idx += 1) {
        if (temp_remove[assign_idx]) {
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
pub fn splitKernel(allocator: Allocator, pir: *Pir) !void {
    _ = allocator;
    _ = pir;
}
pub fn simd(allocator: Allocator, pir: *Pir) !void {
    _ = allocator;
    _ = pir;
}
pub fn memoryLayout(allocator: Allocator, pir: *Pir) !void {
    _ = allocator;
    _ = pir;
}
