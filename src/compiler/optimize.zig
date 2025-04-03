// $TODO Maybe just do the parallelize one at O0 anyways, because it is utterly useless without it
// $TODO These levels
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
// $TODO Either make the order irrelevant here or assert the right order
// $TODO This memory management is horrible. Refactor refactor refactor
//  I feel like there should be a really simple way to do this but I for the life of me can not figure it out
pub fn inlineOp(allocator: Allocator, ssa: *Ssa) !void {
    var inlined_eligible: []bool = try allocator.alloc(bool, ssa.assign_num);
    errdefer allocator.free(inlined_eligible);
    defer allocator.free(inlined_eligible);

    for (0..ssa.assign_num) |assign_idx| {
        inlined_eligible[assign_idx] = if (ssa.assign[assign_idx].base.type.isReduce() or
            !ssa.assign[assign_idx].base.out.intermediary)
            false
        else blk: {
            for (assign_idx + 1..ssa.assign_num) |assign_idx_search| {
                // $TODO Currently there is no way to handle partial overlaps. I think you just need to burn the whole thing down if you find one.
                if ((ssa.assign[assign_idx].base.out.equal(ssa.assign[assign_idx_search].base.out) and
                    ssa.assign[assign_idx].base.out.overlapsPartial(ssa.assign[assign_idx_search].base.out)) or
                    (ssa.assign[assign_idx].base.out.equal(ssa.assign[assign_idx_search].base.in) and
                        ssa.assign[assign_idx].base.out.overlapsPartial(ssa.assign[assign_idx_search].base.in)))
                {
                    break :blk false;
                }

                if (ssa.assign[assign_idx].base.out.equal(ssa.assign[assign_idx_search].base.out) and
                    ssa.assign[assign_idx].base.out.overlapsAll(ssa.assign[assign_idx_search].base.out) and
                    ssa.assign[assign_idx_search].base.overwrites())
                {
                    break :blk true;
                }
            }
            break :blk true;
        };
    }

    // $TODO I should only really save the indices but because I store them in a non global array that information gets lost when saving in Inlined struct
    const temp_base: []Base = try allocator.alloc(Base, ssa.assign_num);
    const temp_out: []?u32 = try allocator.alloc(?u32, ssa.assign_num);
    const temp_in: []?u32 = try allocator.alloc(?u32, ssa.assign_num);
    const temp_idx: []?u32 = try allocator.alloc(?u32, ssa.assign_num);
    const temp_already: []bool = try allocator.alloc(bool, ssa.assign_num);
    errdefer {
        allocator.free(temp_base);
        allocator.free(temp_out);
        allocator.free(temp_in);
        allocator.free(temp_idx);
        allocator.free(temp_already);
    }
    defer {
        allocator.free(temp_base);
        allocator.free(temp_out);
        allocator.free(temp_in);
        allocator.free(temp_idx);
        allocator.free(temp_already);
    }
    @memset(temp_already, false);

    var temp_num: u32 = 0;
    var assign_idx: u32 = 0;
    while (assign_idx < ssa.assign_num) : (assign_idx += 1) {
        if (!inlined_eligible[assign_idx] or temp_already[assign_idx]) {
            continue;
        }

        temp_base[0] = ssa.assign[assign_idx].base;
        temp_num = 1;
        temp_in[0] = null;
        temp_out[0] = null;
        temp_idx[0] = assign_idx;

        var assign_idx_search: u32 = assign_idx + 1;
        while (assign_idx_search < ssa.assign_num) : (assign_idx_search += 1) {
            if (temp_already[assign_idx_search]) {
                continue;
            }

            if (ssa.assign[assign_idx].base.out.equal(ssa.assign[assign_idx_search].base.out) and
                ssa.assign[assign_idx_search].base.overwrites()) break;

            if (ssa.assign[assign_idx].base.out.equal(ssa.assign[assign_idx_search].base.out) and
                ssa.assign[assign_idx].base.out.overlapsAll(ssa.assign[assign_idx_search].base.out))
            {
                if (ssa.assign[assign_idx_search].inlined) |*inlined| {
                    assert(inlined.out_root == null);
                    assert(inlined.in_root != null);

                    for (0..inlined.inlined_num) |inlined_idx| {
                        temp_in[temp_num + inlined_idx] = if (inlined.in[inlined_idx]) |in| in + temp_num else null;
                        temp_out[temp_num + inlined_idx] = if (inlined.out[inlined_idx]) |out| out + temp_num else null;
                        temp_base[temp_num + inlined_idx] = inlined.base[inlined_idx];
                        // $NOTE It's fine that this information is lost because it's already marked for deletion anyways
                        temp_idx[temp_num + inlined_idx] = null;
                    }
                    temp_num += inlined.inlined_num;
                    temp_in[temp_num] = inlined.in_root.? + temp_num;
                } else {
                    temp_in[temp_num] = null;
                }

                temp_base[temp_num] = ssa.assign[assign_idx_search].base;
                temp_out[temp_num] = if (temp_num == 0) null else temp_num - 1;
                temp_idx[temp_num] = assign_idx_search;
                temp_num += 1;
            } else if (ssa.assign[assign_idx].base.out.name_offset == ssa.assign[assign_idx_search].base.in.name_offset and
                (ssa.assign[assign_idx].base.out.overlaps(ssa.assign[assign_idx_search].base.in)))
            {
                // $FIXME This will cause bugs if the inlined assignments are already written somewhere
                if (!ssa.assign[assign_idx].base.out.overlapsAll(ssa.assign[assign_idx_search].base.in)) {
                    temp_num = 1;
                    break;
                }

                if (ssa.assign[assign_idx_search].inlined) |*inlined| {
                    assert(inlined.in_root == null);
                    assert(inlined.out_root == null or inlined.out_root != null); // Just to make it explicit what is expected here

                    inlined.out = try allocator.realloc(inlined.out, inlined.inlined_num + temp_num);
                    inlined.in = try allocator.realloc(inlined.in, inlined.inlined_num + temp_num);
                    inlined.base = try allocator.realloc(inlined.base, inlined.inlined_num + temp_num);
                    inlined.in_root = inlined.inlined_num + (temp_num - 1);
                } else {
                    ssa.assign[assign_idx_search].inlined = .{
                        .base = try allocator.alloc(Base, temp_num),
                        .out = try allocator.alloc(?u32, temp_num),
                        .in = try allocator.alloc(?u32, temp_num),
                        .out_root = null,
                        .in_root = temp_num - 1,
                        .inlined_num = 0,
                    };
                }

                for (0..temp_num) |inlined_idx| {
                    const inlined_num: u32 = ssa.assign[assign_idx_search].inlined.?.inlined_num;
                    if (temp_idx[inlined_idx]) |idx| {
                        temp_already[idx] = true;
                    }

                    ssa.assign[assign_idx_search].inlined.?.base[inlined_num + inlined_idx] = temp_base[inlined_idx];
                    ssa.assign[assign_idx_search].inlined.?.in[inlined_num + inlined_idx] = temp_in[inlined_idx];
                    ssa.assign[assign_idx_search].inlined.?.out[inlined_num + inlined_idx] = temp_out[inlined_idx];
                }
                ssa.assign[assign_idx_search].inlined.?.inlined_num += temp_num;
            }
        }

        if (temp_num == 1) continue;

        if (temp_already[temp_idx[temp_num - 1].?]) {
            // Do nothing I guess?
        } else {
            // $TODO Come to think of it if this case is hit then I guess the ops are completely redundant and can be deleted
            const target_idx: u32 = temp_idx[temp_num - 1].?;

            temp_base[temp_num - 1] = undefined;
            temp_out[temp_num - 1] = null;
            temp_in[temp_num - 1] = null;
            temp_idx[temp_num - 1] = null;
            temp_num -= 1;

            if (ssa.assign[target_idx].inlined) |*inlined| {
                assert(inlined.out_root == null);
                assert(inlined.in_root == null or inlined.in_root != null); // Just to make it explicit what is expected here

                inlined.base = try allocator.realloc(inlined.base, inlined.inlined_num + temp_num);
                inlined.out = try allocator.realloc(inlined.out, inlined.inlined_num + temp_num);
                inlined.in = try allocator.realloc(inlined.in, inlined.inlined_num + temp_num);
                inlined.out_root = inlined.inlined_num + (temp_num - 1);
            } else {
                ssa.assign[target_idx].inlined = .{
                    .base = try allocator.alloc(Base, temp_num),
                    .out = try allocator.alloc(?u32, temp_num),
                    .in = try allocator.alloc(?u32, temp_num),
                    .out_root = temp_num - 1,
                    .in_root = null,
                    .inlined_num = 0,
                };
            }

            const target_num: u32 = ssa.assign[target_idx].inlined.?.inlined_num;
            for (0..temp_num) |inlined_idx| {
                ssa.assign[target_idx].inlined.?.base[target_num + inlined_idx] = temp_base[inlined_idx];
                ssa.assign[target_idx].inlined.?.out[target_num + inlined_idx] = temp_out[inlined_idx];
                ssa.assign[target_idx].inlined.?.in[target_num + inlined_idx] = temp_in[inlined_idx];
                if (temp_idx[inlined_idx]) |idx| {
                    temp_already[idx] = true;
                }
            }

            ssa.assign[target_idx].inlined.?.inlined_num += temp_num;
        }
    }

    assign_idx = 0;
    var assign_num_new: u32 = 0;
    while (assign_idx < ssa.assign_num) : (assign_idx += 1) {
        if (temp_already[assign_idx]) {
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
                if (base[loop_idx].out.aOffset() < a_off_out_root or
                    base[loop_idx].out.aOffset() - a_off_out_root != (loop_idx % dim_info.a_reset_out) / dim_info.a_wait_out * dim_info.a_stride_out)
                {
                    max = loop_idx;
                    break;
                }
            } else {
                if (base[loop_idx].out.aOffset() == a_off_out_root) {
                    a_enter_out = true;
                    dim_info.a_reset_out = loop_idx;
                } else {
                    if (base[loop_idx].out.aOffset() < a_off_out_root or
                        base[loop_idx].out.aOffset() - a_off_out_root != (loop_idx / dim_info.a_wait_out) * dim_info.a_stride_out)
                    {
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
                if (base[loop_idx].out.zOffset() < z_off_out_root or
                    base[loop_idx].out.zOffset() - z_off_out_root != (loop_idx % dim_info.z_reset_out) / dim_info.z_wait_out * dim_info.z_stride_out)
                {
                    max = loop_idx;
                    break;
                }
            } else {
                if (base[loop_idx].out.zOffset() == z_off_out_root) {
                    z_enter_out = true;
                    dim_info.z_reset_out = loop_idx;
                } else {
                    if (base[loop_idx].out.zOffset() < z_off_out_root or
                        base[loop_idx].out.zOffset() - z_off_out_root != (loop_idx / dim_info.z_wait_out) * dim_info.z_stride_out)
                    {
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
                if (base[loop_idx].out.yOffset() < y_off_out_root or
                    base[loop_idx].out.yOffset() - y_off_out_root != (loop_idx % dim_info.y_reset_out) / dim_info.y_wait_out * dim_info.y_stride_out)
                {
                    max = loop_idx;
                    break;
                }
            } else {
                if (base[loop_idx].out.yOffset() == y_off_out_root) {
                    y_enter_out = true;
                    dim_info.y_reset_out = loop_idx;
                } else {
                    if (base[loop_idx].out.yOffset() < y_off_out_root or
                        base[loop_idx].out.yOffset() - y_off_out_root != (loop_idx / dim_info.y_wait_out) * dim_info.y_stride_out)
                    {
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
                if (base[loop_idx].out.xOffset() < x_off_out_root or
                    base[loop_idx].out.xOffset() - x_off_out_root != (loop_idx % dim_info.x_reset_out) / dim_info.x_wait_out * dim_info.x_stride_out)
                {
                    max = loop_idx;
                    break;
                }
            } else {
                if (base[loop_idx].out.xOffset() == x_off_out_root) {
                    x_enter_out = true;
                    dim_info.x_reset_out = loop_idx;
                } else {
                    if (base[loop_idx].out.xOffset() < x_off_out_root or
                        base[loop_idx].out.xOffset() - x_off_out_root != (loop_idx / dim_info.x_wait_out) * dim_info.x_stride_out)
                    {
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
                if (base[loop_idx].in.aOffset() < a_off_in_root or
                    base[loop_idx].in.aOffset() - a_off_in_root != (loop_idx % dim_info.a_reset_in) / dim_info.a_wait_in * dim_info.a_stride_in)
                {
                    max = loop_idx;
                    break;
                }
            } else {
                if (base[loop_idx].in.aOffset() == a_off_in_root) {
                    a_enter_in = true;
                    dim_info.a_reset_in = loop_idx;
                } else {
                    if (base[loop_idx].in.aOffset() < a_off_in_root or
                        base[loop_idx].in.aOffset() - a_off_in_root != (loop_idx / dim_info.a_wait_in) * dim_info.a_stride_in)
                    {
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
                if (base[loop_idx].in.zOffset() < z_off_in_root or
                    base[loop_idx].in.zOffset() - z_off_in_root != (loop_idx % dim_info.z_reset_in) / dim_info.z_wait_in * dim_info.z_stride_in)
                {
                    max = loop_idx;
                    break;
                }
            } else {
                if (base[loop_idx].in.zOffset() == z_off_in_root) {
                    z_enter_in = true;
                    dim_info.z_reset_in = loop_idx;
                } else {
                    if (base[loop_idx].in.zOffset() < z_off_in_root or
                        base[loop_idx].in.zOffset() - z_off_in_root != (loop_idx / dim_info.z_wait_in) * dim_info.z_stride_in)
                    {
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
                if (base[loop_idx].in.yOffset() < y_off_in_root or
                    base[loop_idx].in.yOffset() - y_off_in_root != (loop_idx % dim_info.y_reset_in) / dim_info.y_wait_in * dim_info.y_stride_in)
                {
                    max = loop_idx;
                    break;
                }
            } else {
                if (base[loop_idx].in.yOffset() == y_off_in_root) {
                    y_enter_in = true;
                    dim_info.y_reset_in = loop_idx;
                } else {
                    if (base[loop_idx].in.yOffset() < y_off_in_root or
                        base[loop_idx].in.yOffset() - y_off_in_root != (loop_idx / dim_info.y_wait_in) * dim_info.y_stride_in)
                    {
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
                if (base[loop_idx].in.xOffset() < x_off_in_root or
                    base[loop_idx].in.xOffset() - x_off_in_root != (loop_idx % dim_info.x_reset_in) / dim_info.x_wait_in * dim_info.x_stride_in)
                {
                    max = loop_idx;
                    break;
                }
            } else {
                if (base[loop_idx].in.xOffset() == x_off_in_root) {
                    x_enter_in = true;
                    dim_info.x_reset_in = loop_idx;
                } else {
                    if (base[loop_idx].in.xOffset() < x_off_in_root or
                        base[loop_idx].in.xOffset() - x_off_in_root != (loop_idx / dim_info.x_wait_in) * dim_info.x_stride_in)
                    {
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
            if (ssa.assign[assign_idx].base.equals(ssa.assign[assign_idx_search].base)) {
                if (ssa.assign[assign_idx].base.out.overlapsPartial(ssa.assign[assign_idx_search].base.out)) {
                    break;
                } else {
                    var equal: bool = true;
                    for (0..(assign_idx_search - assign_idx)) |assign_off| {
                        if (!ssa.assign[assign_idx + assign_off].base.equals(ssa.assign[assign_idx_search + assign_off].base) or
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
                                if (!ssa.assign[assign_idx + loop_idx_search * loop_len + assign_off_search].base.equals( //
                                    ssa.assign[assign_idx + loop_idx * loop_len + assign_off].base) or
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
