// TODO: Maybe make Ofast where accuracy can get sacrificed for speed
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

const Op = @import("../tensor.zig").Op;
const Ssa = @import("./ssa.zig").Ssa;

pub const Optimization = enum(u8) {
    O0,
    O1,
    O2,
    O3,
    pub fn inlineOp(this: @This(), allocator: std.mem.Allocator, ssa: *Ssa) !void {
        assert(this == .O1 or this == .O2 or this == .O3);

        if (ssa.assign_num < 2) {
            return;
        }

        const assign_used: []bool = try allocator.alloc(bool, ssa.assign_num);
        errdefer allocator.free(assign_used);
        defer allocator.free(assign_used);

        const assign_temp: []Ssa.Assign.Base = try allocator.alloc(Ssa.Assign.Base, ssa.assign_num);
        errdefer allocator.free(assign_temp);
        defer allocator.free(assign_temp);

        const assign_type: []Ssa.Assign.Inlined.Type = try allocator.alloc(Ssa.Assign.Inlined.Type, ssa.assign_num);
        errdefer allocator.free(assign_type);
        defer allocator.free(assign_type);

        var assign_temp_num: usize = 0;

        for (0..ssa.assign_num) |assign_idx_reverse| {
            const assign_idx: usize = ssa.assign_num - (assign_idx_reverse + 1);

            if (assign_used[assign_idx]) {
                continue;
            }

            for (0..assign_idx) |assign_idx_search_reverse| {
                const assign_idx_search: usize = assign_idx - (assign_idx_search_reverse + 1);

                // TODO: Make it possible to inline reduce ops
                if (ssa.assign[assign_idx_search].base.type.isReduce()) {
                    continue;
                }

                if (ssa.assign[assign_idx].base.out.name_offset == ssa.assign[assign_idx_search].base.out.name_offset) {
                    if (ssa.assign[assign_idx].base.out.overlapsAll(ssa.assign[assign_idx_search].base.out)) {
                        assign_used[assign_idx_search] = true;
                        assign_temp[assign_temp_num] = ssa.assign[assign_idx_search].base;
                        assign_type[assign_temp_num] = .out;
                        assign_temp_num += 1;
                    } else {
                        break;
                    }
                } else if (ssa.assign[assign_idx].base.in.name_offset == ssa.assign[assign_idx_search].base.out.name_offset) {
                    if (ssa.assign[assign_idx].base.in.overlapsAll(ssa.assign[assign_idx_search].base.out)) {
                        assign_used[assign_idx_search] = true;
                        assign_temp[assign_temp_num] = ssa.assign[assign_idx_search].base;
                        assign_type[assign_temp_num] = .in;
                        assign_temp_num += 1;
                    } else {
                        break;
                    }
                } else {
                    continue;
                }

                if (ssa.assign[assign_idx_search].base.overwrites() and
                    (ssa.assign[assign_idx].base.out.name_offset == ssa.assign[assign_idx_search].base.out.name_offset or
                    ssa.assign[assign_idx].base.in.name_offset == ssa.assign[assign_idx_search].base.out.name_offset))
                {
                    break;
                }
            }

            if (assign_temp_num == 0) {
                ssa.assign[assign_idx].inlined = null;
            } else {
                ssa.assign[assign_idx].inlined = .{
                    .base = try allocator.dupe(Ssa.Assign.Base, assign_temp[0..assign_temp_num]),
                    .type = try allocator.dupe(Ssa.Assign.Inlined.Type, assign_type[0..assign_temp_num]),
                };
                assign_temp_num = 0;
            }
        }

        var assign_num_new: usize = 0;
        for (0..ssa.assign_num) |assign_idx| {
            if (!assign_used[assign_idx]) {
                ssa.assign[assign_num_new] = ssa.assign[assign_idx];
                assign_num_new += 1;
            }
        }
        ssa.assign_num = assign_num_new;
    }
    pub fn parallelize(this: @This(), allocator: std.mem.Allocator, ssa: *Ssa) !void {
        assert(this == .O1 or this == .O2 or this == .O3);
        _ = allocator;
        _ = ssa;
    }
    pub fn splitKernel(this: @This(), allocator: std.mem.Allocator, ssa: *Ssa) !void {
        assert(this == .O1 or this == .O2 or this == .O3);
        _ = allocator;
        _ = ssa;
    }
    pub fn simd(this: @This(), allocator: std.mem.Allocator, ssa: *Ssa) !void {
        assert(this == .O2 or this == .O3);
        _ = allocator;
        _ = ssa;
    }
    pub fn memoryLayout(this: @This(), allocator: std.mem.Allocator, ssa: *Ssa) !void {
        assert(this == .O3);
        _ = allocator;
        _ = ssa;
    }
};
