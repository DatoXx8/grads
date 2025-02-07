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
        var inlined_temp_num: usize = 0;
        var inlined_temp: []Ssa.Assignment = try allocator.alloc(Ssa.Assignment, ssa.assignment_num);
        defer allocator.free(inlined_temp);
        var inlined_temp_used: []bool = try allocator.alloc(bool, ssa.assignment_num);
        defer allocator.free(inlined_temp_used);
        @memset(inlined_temp_used, false);

        for (0..ssa.assignment_num) |assignment_idx| {
            inlined_temp[inlined_temp_num] = ssa.assignment[assignment_idx];
            inlined_temp_used[inlined_temp_num] = true;
            inlined_temp[inlined_temp_num].print(4, 0, null);
            inlined_temp_num += 1;
        }
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
