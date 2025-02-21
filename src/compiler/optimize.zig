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
const DimInfo = @import("./ssa.zig").DimInfo;

pub const Optimization = enum(u8) {
    O0,
    O1,
    O2,
    O3,
    pub fn inlineOp(this: @This(), allocator: Allocator, ssa: *Ssa) !void {
        assert(this == .O1 or this == .O2 or this == .O3);
        _ = allocator;
        _ = ssa;
    }
    // NOTE: I don't think there is way to make this faster than O(n^2) unless I make a max loop size
    pub fn parallelize(this: @This(), allocator: Allocator, ssa: *Ssa) !void {
        assert(this == .O1 or this == .O2 or this == .O3);

        var base_temp: []Assign.Base = try allocator.alloc(Assign.Base, ssa.assign_num);
        errdefer allocator.free(base_temp);
        defer allocator.free(base_temp);

        var remove_temp: []bool = try allocator.alloc(bool, ssa.assign_num);
        errdefer allocator.free(remove_temp);
        defer allocator.free(remove_temp);
        @memset(remove_temp, false);

        var loop_id: u32 = 1;

        var assign_idx: u32 = 0;
        while (assign_idx < ssa.assign_num) : (assign_idx += 1) {
            var loop_len: u32 = 0;
            var loop_num: u32 = 0;

            var assign_idx_search: u32 = assign_idx + 1;
            while (2 * assign_idx_search - assign_idx < ssa.assign_num) : (assign_idx_search += 1) {
                if (ssa.assign[assign_idx].base.equals(ssa.assign[assign_idx_search].base) and
                    (!ssa.assign[assign_idx].base.out.overlaps(ssa.assign[assign_idx_search].base.out) or
                    ssa.assign[assign_idx].base.out.intermediary))
                {
                    var equal: bool = true;
                    for (0..(assign_idx_search - assign_idx)) |assign_off| {
                        if (!ssa.assign[assign_idx + assign_off].base.equals(ssa.assign[assign_idx_search + assign_off].base) or
                            (ssa.assign[assign_idx + assign_off].base.out.overlaps(ssa.assign[assign_idx_search + assign_off].base.out) and
                            !ssa.assign[assign_idx].base.out.intermediary))
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
                            (ssa.assign[assign_idx + assign_off].base.out.overlaps(ssa.assign[assign_idx + loop_idx * loop_len + assign_off].base.out) and
                            !ssa.assign[assign_idx + assign_off].base.out.intermediary))
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
                    ssa.assign[assign_idx + inner_idx].base.dim_info = DimInfo.init(base_temp[0..loop_len]);
                    ssa.assign_loop_id[assign_idx + inner_idx] = loop_id;
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
                continue;
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
