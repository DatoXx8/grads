// TODO: Maybe just do the parallelize one at O0 anyways, because it is utterly useless without it
// TODO: These levels
// Optimization levels
// O1 - parallelize, inline, split
// O2 - SIMD
// O3 - memory optimizer

// Optimization levels
// O0 - none
// O1 - parallelize, inline, split
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
    // TODO: Either make the order irrelevant here or assert the right order
    // TODO: This memory management is horrible. Refactor refactor refactor
    //  I feel like there should be a really simple way to do this but I for the life of me can not figure it out
    pub fn inlineOp(this: @This(), allocator: Allocator, ssa: *Ssa) !void {
        assert(this == .O1 or this == .O2 or this == .O3);

        var inlined_eligible: []bool = try allocator.alloc(bool, ssa.assign_num);
        errdefer allocator.free(inlined_eligible);
        defer allocator.free(inlined_eligible);

        var inlined_temp_num: usize = 0;

        for (0..ssa.assign_num) |assign_idx| {
            inlined_eligible[assign_idx] = if (ssa.assign[assign_idx].base.type.isReduce() or
                !ssa.assign[assign_idx].base.out.intermediary)
                false
            else blk: {
                for (assign_idx + 1..ssa.assign_num) |assign_idx_search| {
                    // TODO: Currently there is no way to handle partial overlaps. I think you just need to burn the whole thing down if you find one.
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

        // TODO: I should only really save the indices but because I store them in a non global array that information gets lost when saving in Inlined struct
        const inlined_temp: []Base = try allocator.alloc(Base, ssa.assign_num);
        const inlined_temp_out: []?usize = try allocator.alloc(?usize, ssa.assign_num);
        const inlined_temp_in: []?usize = try allocator.alloc(?usize, ssa.assign_num);
        const inlined_temp_idx: []?usize = try allocator.alloc(?usize, ssa.assign_num);
        const inlined_already: []bool = try allocator.alloc(bool, ssa.assign_num);
        errdefer {
            allocator.free(inlined_temp);
            allocator.free(inlined_temp_out);
            allocator.free(inlined_temp_in);
            allocator.free(inlined_temp_idx);
            allocator.free(inlined_already);
        }
        defer {
            allocator.free(inlined_temp);
            allocator.free(inlined_temp_out);
            allocator.free(inlined_temp_in);
            allocator.free(inlined_temp_idx);
            allocator.free(inlined_already);
        }
        @memset(inlined_already, false);

        for (0..ssa.assign_num) |assign_idx| {
            if (!inlined_eligible[assign_idx] or inlined_already[assign_idx]) {
                continue;
            }

            inlined_temp[0] = ssa.assign[assign_idx].base;
            inlined_temp_num = 1;
            inlined_temp_in[0] = null;
            inlined_temp_out[0] = null;
            inlined_temp_idx[0] = assign_idx;

            for (assign_idx + 1..ssa.assign_num) |assign_idx_search| {
                if (inlined_already[assign_idx_search]) {
                    continue;
                }

                if (ssa.assign[assign_idx].base.out.equal(ssa.assign[assign_idx_search].base.out) and
                    ssa.assign[assign_idx_search].base.overwrites()) break;

                if (ssa.assign[assign_idx].base.out.equal(ssa.assign[assign_idx_search].base.out) and
                    ssa.assign[assign_idx].base.out.overlapsAll(ssa.assign[assign_idx_search].base.out))
                {
                    if (ssa.assign[assign_idx_search].inlined) |*inlined| {
                        assert(inlined.inlined_out_base == null);
                        assert(inlined.inlined_in_base != null);

                        for (0..inlined.inlined_num) |inlined_idx| {
                            inlined_temp_in[inlined_temp_num + inlined_idx] = if (inlined.inlined_in[inlined_idx]) |in| in + inlined_temp_num else null;
                            inlined_temp_out[inlined_temp_num + inlined_idx] = if (inlined.inlined_out[inlined_idx]) |out| out + inlined_temp_num else null;
                            inlined_temp[inlined_temp_num + inlined_idx] = inlined.base[inlined_idx];
                            // NOTE: It's fine that this information is lost because it's already marked for deletion anyways
                            inlined_temp_idx[inlined_temp_num + inlined_idx] = null;
                        }
                        inlined_temp_num += inlined.inlined_num;
                        inlined_temp_in[inlined_temp_num] = inlined.inlined_in_base.? + inlined_temp_num;
                    } else {
                        inlined_temp_in[inlined_temp_num] = null;
                    }

                    inlined_temp[inlined_temp_num] = ssa.assign[assign_idx_search].base;
                    inlined_temp_out[inlined_temp_num] = if (inlined_temp_num == 0) null else inlined_temp_num - 1;
                    inlined_temp_idx[inlined_temp_num] = assign_idx_search;
                    inlined_temp_num += 1;
                } else if (ssa.assign[assign_idx].base.out.equal(ssa.assign[assign_idx_search].base.in) and
                    ssa.assign[assign_idx].base.out.overlapsAll(ssa.assign[assign_idx_search].base.in))
                {
                    if (ssa.assign[assign_idx_search].inlined) |*inlined| {
                        assert(inlined.inlined_in_base == null);
                        assert(inlined.inlined_out_base == null or inlined.inlined_out_base != null); // Just to make it explicit what is expected here

                        inlined.inlined_out = try allocator.realloc(inlined.inlined_out, inlined.inlined_num + inlined_temp_num);
                        inlined.inlined_in = try allocator.realloc(inlined.inlined_in, inlined.inlined_num + inlined_temp_num);
                        inlined.base = try allocator.realloc(inlined.base, inlined.inlined_num + inlined_temp_num);
                        inlined.inlined_in_base = inlined.inlined_num + (inlined_temp_num - 1);
                    } else {
                        ssa.assign[assign_idx_search].inlined = .{
                            .base = try allocator.alloc(Base, inlined_temp_num),
                            .inlined_out = try allocator.alloc(?usize, inlined_temp_num),
                            .inlined_in = try allocator.alloc(?usize, inlined_temp_num),
                            .inlined_out_base = null,
                            .inlined_in_base = inlined_temp_num - 1,
                            .inlined_num = 0,
                        };
                    }

                    for (0..inlined_temp_num) |inlined_idx| {
                        const inlined_num: usize = ssa.assign[assign_idx_search].inlined.?.inlined_num;
                        if (inlined_temp_idx[inlined_idx]) |idx| {
                            inlined_already[idx] = true;
                        }

                        ssa.assign[assign_idx_search].inlined.?.base[inlined_num + inlined_idx] = inlined_temp[inlined_idx];
                        ssa.assign[assign_idx_search].inlined.?.inlined_in[inlined_num + inlined_idx] = inlined_temp_in[inlined_idx];
                        ssa.assign[assign_idx_search].inlined.?.inlined_out[inlined_num + inlined_idx] = inlined_temp_out[inlined_idx];
                    }
                    ssa.assign[assign_idx_search].inlined.?.inlined_num += inlined_temp_num;
                }
            }

            if (inlined_temp_num == 1) continue;

            if (inlined_already[inlined_temp_idx[inlined_temp_num - 1].?]) {
                // Do nothing I guess?
            } else {
                // TODO: Come to think of it if this case is hit then I guess the ops are completely redundant and can be deleted
                const target_idx: usize = inlined_temp_idx[inlined_temp_num - 1].?;

                inlined_temp[inlined_temp_num - 1] = undefined;
                inlined_temp_out[inlined_temp_num - 1] = null;
                inlined_temp_in[inlined_temp_num - 1] = null;
                inlined_temp_idx[inlined_temp_num - 1] = null;
                inlined_temp_num -= 1;

                if (ssa.assign[target_idx].inlined) |*inlined| {
                    assert(inlined.inlined_out_base == null);
                    assert(inlined.inlined_in_base == null or inlined.inlined_in_base != null); // Just to make it explicit what is expected here

                    inlined.base = try allocator.realloc(inlined.base, inlined.inlined_num + inlined_temp_num);
                    inlined.inlined_out = try allocator.realloc(inlined.inlined_out, inlined.inlined_num + inlined_temp_num);
                    inlined.inlined_in = try allocator.realloc(inlined.inlined_in, inlined.inlined_num + inlined_temp_num);
                    inlined.inlined_out_base = inlined.inlined_num + (inlined_temp_num - 1);
                } else {
                    ssa.assign[target_idx].inlined = .{
                        .base = try allocator.alloc(Base, inlined_temp_num),
                        .inlined_out = try allocator.alloc(?usize, inlined_temp_num),
                        .inlined_in = try allocator.alloc(?usize, inlined_temp_num),
                        .inlined_out_base = inlined_temp_num - 1,
                        .inlined_in_base = null,
                        .inlined_num = 0,
                    };
                }

                const target_num: usize = ssa.assign[target_idx].inlined.?.inlined_num;
                for (0..inlined_temp_num) |inlined_idx| {
                    ssa.assign[target_idx].inlined.?.base[target_num + inlined_idx] = inlined_temp[inlined_idx];
                    ssa.assign[target_idx].inlined.?.inlined_out[target_num + inlined_idx] = inlined_temp_out[inlined_idx];
                    ssa.assign[target_idx].inlined.?.inlined_in[target_num + inlined_idx] = inlined_temp_in[inlined_idx];
                    if (inlined_temp_idx[inlined_idx]) |idx| {
                        inlined_already[idx] = true;
                    }
                }

                ssa.assign[target_idx].inlined.?.inlined_num += inlined_temp_num;
            }
        }

        var assign_num_new: usize = 0;
        for (0..ssa.assign_num) |assign_idx| {
            if (inlined_already[assign_idx]) {
                if (ssa.assign[assign_idx].inlined) |*inlined| {
                    allocator.free(inlined.base);
                    allocator.free(inlined.inlined_out);
                    allocator.free(inlined.inlined_in);
                }
            } else {
                ssa.assign[assign_num_new] = ssa.assign[assign_idx];
                assign_num_new += 1;
            }
        }
        ssa.assign_num = assign_num_new;
    }
    // NOTE: I don't think there is way to make this faster than O(n^2) unless I make a max loop size
    pub fn parallelize(this: @This(), allocator: Allocator, ssa: *Ssa) !void {
        assert(this == .O1 or this == .O2 or this == .O3);

        var base_temp: []Base = try allocator.alloc(Base, ssa.assign_num);
        errdefer allocator.free(base_temp);
        defer allocator.free(base_temp);

        var remove_temp: []bool = try allocator.alloc(bool, ssa.assign_num);
        errdefer allocator.free(remove_temp);
        defer allocator.free(remove_temp);
        @memset(remove_temp, false);

        var loop_id: u32 = 1;

        // TODO: Check for equality of inlined trees
        var assign_idx: u32 = 0;
        while (assign_idx < ssa.assign_num) : (assign_idx += 1) {
            var loop_len: u32 = 0;
            var loop_num: u32 = 0;

            var assign_idx_search: u32 = assign_idx + 1;
            while (2 * assign_idx_search - assign_idx < ssa.assign_num) : (assign_idx_search += 1) {
                if (ssa.assign[assign_idx].base.equals(ssa.assign[assign_idx_search].base) and
                    !ssa.assign[assign_idx].base.out.overlaps(ssa.assign[assign_idx_search].base.out))
                {
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

            if (loop_len == 0) {
                loop_len = 1;
                loop_num = 1;
            } else {
                loop_num = 1;
                for (1..@divFloor(ssa.assign_num - assign_idx, loop_len)) |loop_idx| {
                    var equal: bool = true;
                    var assign_off: usize = 0;
                    // TODO: Might need to check for overlap between every single iterations... yikes...
                    while (assign_off < loop_len) : (assign_off += 1) {
                        if (!ssa.assign[assign_idx + assign_off].base.equals(ssa.assign[assign_idx + loop_idx * loop_len + assign_off].base) or
                            ssa.assign[assign_idx + assign_off].base.out.overlaps(ssa.assign[assign_idx + loop_idx * loop_len + assign_off].base.out))
                        {
                            equal = false;
                            break;
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
                        base_temp[loop_idx] = ssa.assign[assign_idx + inner_idx + loop_idx * loop_len].base;
                        if (loop_idx != 0) {
                            remove_temp[assign_idx + inner_idx + loop_idx * loop_len] = true;
                        }
                    }
                    ssa.assign[assign_idx + inner_idx].base.dim_info = DimInfo.init(base_temp[0..loop_num]);
                    ssa.assign_loop_id[assign_idx + inner_idx] = loop_id;

                    if (ssa.assign[assign_idx + inner_idx].inlined) |*inlined| {
                        for (0..inlined.inlined_num) |inlined_idx| {
                            for (0..loop_num) |loop_idx| {
                                base_temp[loop_idx] = ssa.assign[assign_idx + inner_idx + loop_idx * loop_len].inlined.?.base[inlined_idx];
                            }
                            inlined.base[inner_idx].dim_info = DimInfo.init(base_temp[0..loop_num]);
                        }
                    }

                    ssa.assign_loop_num[loop_id] = loop_num;
                }

                loop_id += 1;
                assign_idx += (loop_len * loop_num) - 1;
            }
        }
        var assign_num_new: usize = 0;
        assign_idx = 0;
        while (assign_idx < ssa.assign_num) : (assign_idx += 1) {
            if (remove_temp[assign_idx]) {
                if (ssa.assign[assign_idx].inlined) |*inlined| {
                    allocator.free(inlined.base);
                    allocator.free(inlined.inlined_out);
                    allocator.free(inlined.inlined_in);
                }
            } else {
                ssa.assign[assign_num_new] = ssa.assign[assign_idx];
                ssa.assign_loop_id[assign_num_new] = ssa.assign_loop_id[assign_idx];
                assign_num_new += 1;
            }
        }
        ssa.assign_num = assign_num_new;
    }
    pub fn splitKernel(this: @This(), allocator: Allocator, ssa: *Ssa) !void {
        assert(this == .O1 or this == .O2 or this == .O3);
        _ = allocator;
        _ = ssa;
    }
    pub fn simd(this: @This(), allocator: Allocator, ssa: *Ssa) !void {
        assert(this == .O2 or this == .O3);
        _ = allocator;
        _ = ssa;
    }
    pub fn memoryLayout(this: @This(), allocator: Allocator, ssa: *Ssa) !void {
        assert(this == .O3);
        _ = allocator;
        _ = ssa;
    }
};
