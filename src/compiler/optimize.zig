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

                std.debug.print("{} {}\n", .{ assign_idx, assign_idx_search });

                // TODO: Make it possible to inline reduce ops
                if (ssa.assign[assign_idx_search].base.type.isReduce()) {
                    std.debug.print("SKIP REDUCE\n", .{});
                    continue;
                }

                if (ssa.assign[assign_idx].base.out.name_offset == ssa.assign[assign_idx_search].base.out.name_offset) {
                    assert(ssa.assign[assign_idx].base.out.overlapsAll(ssa.assign[assign_idx_search].base.out));
                    assign_used[assign_idx_search] = true;
                    assign_temp[assign_temp_num] = ssa.assign[assign_idx_search].base;
                    assign_type[assign_temp_num] = .out;
                    assign_temp_num += 1;
                } else if (ssa.assign[assign_idx].base.in.name_offset == ssa.assign[assign_idx_search].base.out.name_offset) {
                    assert(ssa.assign[assign_idx].base.in.overlapsAll(ssa.assign[assign_idx_search].base.out));
                    assign_used[assign_idx_search] = true;
                    assign_temp[assign_temp_num] = ssa.assign[assign_idx_search].base;
                    assign_type[assign_temp_num] = .in;
                    assign_temp_num += 1;
                } else {
                    continue;
                }

                // TODO: Need to check that the ops operate on the same values
                //  Unsure how to handle partial overlaps
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

// var loop_len: usize = 0;
// // TODO: Maybe make this a constant like 4096 or something.
// const loop_len_max: usize = @divFloor(linearized.op_num - op_search_idx, 2);
// if (loop_len_max != 0) {
//     for (1..loop_len_max) |loop_search| {
//         const match_potential: bool = linearized.op[op_search_idx].equal(linearized.op[op_search_idx + loop_search]) and
//             !linearized.op[op_search_idx].overlaps(linearized.op[op_search_idx + loop_search]);
//         var match: bool = true;
//         if (match_potential) {
//             for (1..loop_search) |op_idx| {
//                 if (linearized.op[op_search_idx + op_idx].equal(linearized.op[op_search_idx + loop_search + op_idx]) and
//                     !linearized.op[op_search_idx].overlaps(linearized.op[op_search_idx + loop_search]))
//                 {
//                     continue;
//                 } else {
//                     match = false;
//                     break;
//                 }
//             }
//             if (match) {
//                 loop_len = loop_search;
//                 break;
//             }
//         }
//     }
// }
//
// var layer_loop_id_curr: usize = 0;
//
// var loop_num: usize = 1;
// if (loop_len == 0) {
//     loop_len = 1;
//     loop_num = 1;
//     layer_loop_id_curr = 0;
// } else {
//     const loop_num_max: usize = @divFloor(linearized.op_num - op_search_idx, loop_len);
//     for (1..loop_num_max) |loop_idx| {
//         var same: bool = true;
//         for (0..loop_len) |op_idx| {
//             if (!linearized.op[op_search_idx + op_idx].equal(linearized.op[op_search_idx + op_idx + loop_idx * loop_len]) or
//                 linearized.op[op_search_idx + op_idx].overlaps(linearized.op[op_search_idx + op_idx + loop_idx * loop_len]))
//             {
//                 same = false;
//                 break;
//             }
//         }
//         if (same) {
//             loop_num += 1;
//         } else {
//             break;
//         }
//     }
//
//     // If this isn't the case some *really* weird things are going on
//     assert(loop_num <= loop_num_max);
//
//     if (loop_num == 1) {
//         layer_loop_id_curr = 0;
//     } else {
//         layer_loop_id_curr = layer_loop_id_tracker;
//         layer_loop_id_tracker += 1;
//     }
// }
