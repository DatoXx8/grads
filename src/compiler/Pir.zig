const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const ArrayList = std.ArrayList;

const Linearized = @import("../Linearized.zig");
const Op = Linearized.Op;
const Buffer = @import("../Buffer.zig");
const Vec4 = Buffer.Vec4;
const View = Buffer.View;
const Stride = Buffer.Stride;
const opt = @import("optimize.zig");
const Optimization = opt.Optimization;
const util = @import("../util.zig");

const VGpu = @import("VGpu.zig");

pub const ViewOffset = struct {
    pub const value_none: u32 = std.math.maxInt(u32);
    pub const wait_default: u32 = 1;
    pub const stride_default: u32 = 0;
    /// Just the highest bit
    pub const reset_default: u32 = ~(value_none >> 1);
    offset: u32,
    stride: Stride,
    repeat_stride: Vec4,
    repeat_wait: Vec4,
    repeat_reset: Vec4,
    pub fn fromView(view: View) ViewOffset {
        return .{
            .offset = view.offset,
            .stride = view.stride,
            .repeat_stride = .{ .a = value_none, .z = value_none, .y = value_none, .x = value_none },
            .repeat_wait = .{ .a = value_none, .z = value_none, .y = value_none, .x = value_none },
            .repeat_reset = .{ .a = value_none, .z = value_none, .y = value_none, .x = value_none },
        };
    }
    pub fn removeDefault(view_offset: *ViewOffset) void {
        view_offset.repeat_stride.a =
            if (view_offset.repeat_stride.a == value_none) stride_default else view_offset.repeat_stride.a;
        view_offset.repeat_stride.z =
            if (view_offset.repeat_stride.z == value_none) stride_default else view_offset.repeat_stride.z;
        view_offset.repeat_stride.y =
            if (view_offset.repeat_stride.y == value_none) stride_default else view_offset.repeat_stride.y;
        view_offset.repeat_stride.x =
            if (view_offset.repeat_stride.x == value_none) stride_default else view_offset.repeat_stride.x;
        view_offset.repeat_wait.a =
            if (view_offset.repeat_wait.a == value_none) wait_default else view_offset.repeat_wait.a;
        view_offset.repeat_wait.z =
            if (view_offset.repeat_wait.z == value_none) wait_default else view_offset.repeat_wait.z;
        view_offset.repeat_wait.y =
            if (view_offset.repeat_wait.y == value_none) wait_default else view_offset.repeat_wait.y;
        view_offset.repeat_wait.x =
            if (view_offset.repeat_wait.x == value_none) wait_default else view_offset.repeat_wait.x;
        view_offset.repeat_reset.a =
            if (view_offset.repeat_reset.a == value_none) reset_default else view_offset.repeat_reset.a;
        view_offset.repeat_reset.z =
            if (view_offset.repeat_reset.z == value_none) reset_default else view_offset.repeat_reset.z;
        view_offset.repeat_reset.y =
            if (view_offset.repeat_reset.y == value_none) reset_default else view_offset.repeat_reset.y;
        view_offset.repeat_reset.x =
            if (view_offset.repeat_reset.x == value_none) reset_default else view_offset.repeat_reset.x;
    }
    pub fn viewAtRepeat(view_offset: ViewOffset, size: Vec4, repeat_idx: u32) View {
        var view_offset_no_default: ViewOffset = view_offset;
        view_offset_no_default.removeDefault();
        return .{
            .size = size,
            .stride = view_offset.stride,
            .offset = view_offset.offset +
                (repeat_idx % view_offset_no_default.repeat_reset.a) / view_offset_no_default.repeat_wait.a *
                    view_offset_no_default.repeat_stride.a * view_offset_no_default.stride.a +
                (repeat_idx % view_offset_no_default.repeat_reset.z) / view_offset_no_default.repeat_wait.z *
                    view_offset_no_default.repeat_stride.z * view_offset_no_default.stride.z +
                (repeat_idx % view_offset_no_default.repeat_reset.y) / view_offset_no_default.repeat_wait.y *
                    view_offset_no_default.repeat_stride.y * view_offset_no_default.stride.y +
                (repeat_idx % view_offset_no_default.repeat_reset.x) / view_offset_no_default.repeat_wait.x *
                    view_offset_no_default.repeat_stride.x,
        };
    }
    pub fn equal(view_offset_1: ViewOffset, view_offset_2: ViewOffset) bool {
        return view_offset_1.offset == view_offset_2.offset and
            view_offset_1.stride.equal(view_offset_2.stride) and
            view_offset_1.repeat_stride.equal(view_offset_2.repeat_stride) and
            view_offset_1.repeat_wait.equal(view_offset_2.repeat_wait) and
            view_offset_1.repeat_reset.equal(view_offset_2.repeat_reset);
    }
    pub fn overlaps(
        view_offset_1: ViewOffset,
        repeat_1: u32,
        size_1: Vec4,
        view_offset_2: ViewOffset,
        repeat_2: u32,
        size_2: Vec4,
    ) bool {
        var repeat_1_idx: u32 = 0;
        while (repeat_1_idx < repeat_1) : (repeat_1_idx += 1) {
            const view_1: View = view_offset_1.viewAtRepeat(size_1, repeat_1_idx);
            var repeat_2_idx: u32 = 0;
            while (repeat_2_idx < repeat_2) : (repeat_2_idx += 1) {
                const view_2: View = view_offset_2.viewAtRepeat(size_2, repeat_2_idx);

                if (view_1.overlaps(view_2)) return true;
            }
        }
        return false;
    }
    pub fn overlapsAll(
        view_offset_1: ViewOffset,
        repeat_1: u32,
        size_1: Vec4,
        view_offset_2: ViewOffset,
        repeat_2: u32,
        size_2: Vec4,
    ) bool {
        var repeat_1_idx: u32 = 0;
        while (repeat_1_idx < repeat_1) : (repeat_1_idx += 1) {
            const view_1: View = view_offset_1.viewAtRepeat(size_1, repeat_1_idx);
            var repeat_2_idx: u32 = 0;
            while (repeat_2_idx < repeat_2) : (repeat_2_idx += 1) {
                const view_2: View = view_offset_2.viewAtRepeat(size_2, repeat_2_idx);

                if (view_1.overlapsAll(view_2)) return true;
            }
        }
        return false;
    }
    pub fn overlapsPartial(
        view_offset_1: ViewOffset,
        repeat_1: u32,
        size_1: Vec4,
        view_offset_2: ViewOffset,
        repeat_2: u32,
        size_2: Vec4,
    ) bool {
        var repeat_1_idx: u32 = 0;
        while (repeat_1_idx < repeat_1) : (repeat_1_idx += 1) {
            const view_1: View = view_offset_1.viewAtRepeat(size_1, repeat_1_idx);
            var repeat_2_idx: u32 = 0;
            while (repeat_2_idx < repeat_2) : (repeat_2_idx += 1) {
                const view_2: View = view_offset_2.viewAtRepeat(size_2, repeat_2_idx);

                if (view_1.overlapsPartial(view_2)) return true;
            }
        }
        return false;
    }
    pub fn print(view_offset: ViewOffset, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            util.log.print("{s}ViewOffset {s}\n", .{ [1]u8{' '} ** offset, text });
        }
        util.log.print("{s}off => ({d:10}, {d:10}, {d:10}, {d:10}) = {}\n", .{
            " " ** (offset + padding), //
            view_offset.offset / view_offset.stride.a,
            view_offset.offset % view_offset.stride.a / view_offset.stride.z,
            view_offset.offset % view_offset.stride.z / view_offset.stride.y,
            view_offset.offset % view_offset.stride.y / view_offset.stride.x,
            view_offset.offset,
        });
        util.log.print("{s}str => ({d:10}, {d:10}, {d:10}, {d:10})\n", .{
            " " ** (offset + padding), //
            view_offset.repeat_stride.a,
            view_offset.repeat_stride.z,
            view_offset.repeat_stride.y,
            view_offset.repeat_stride.x,
        });
        util.log.print("{s}res => ({d:10}, {d:10}, {d:10}, {d:10})\n", .{
            " " ** (offset + padding), //
            view_offset.repeat_reset.a,
            view_offset.repeat_reset.z,
            view_offset.repeat_reset.y,
            view_offset.repeat_reset.x,
        });
        util.log.print("{s}wai => ({d:10}, {d:10}, {d:10}, {d:10})\n", .{
            " " ** (offset + padding), //
            view_offset.repeat_wait.a,
            view_offset.repeat_wait.z,
            view_offset.repeat_wait.y,
            view_offset.repeat_wait.x,
        });
    }
};
/// The basic thing the Assignment does without any funny business
pub const Base = struct {
    out: Buffer,
    in: Buffer,
    out_view: ViewOffset, // Only need to store the information to conmpute the offsets because the size is stored in the Assign
    in_view: ViewOffset,

    kind: Op.Kind,
    u_var: f32,
    pub fn print(base: Base, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            util.log.print("{s}Base {s}\n", .{ " " ** offset, text });
        }
        if (base.kind.isUnary()) {
            util.log.print("{s}{s} \"{s}\" {d}\n", .{
                " " ** (offset + padding),
                @tagName(base.kind),
                base.out.name(),
                base.u_var,
            });
            base.out_view.print(padding, padding + offset, null);
        } else {
            util.log.print("{s}{s} \"{s}\" \"{s}\"\n", .{
                " " ** (offset + padding),
                @tagName(base.kind),
                base.out.name(),
                base.in.name(),
            });
            base.out_view.print(padding, padding + offset, null);
            base.in_view.print(padding, padding + offset, null);
        }
    }
};
/// Tree representation of inlined ops
pub const Inlined = struct {
    base: []Base,
    out: []?u32,
    in: []?u32,
    out_root: ?u32,
    in_root: ?u32,
    num: u32,
    pub inline fn inlinedEqual(inlined: Inlined, target: Inlined) bool {
        assert(inlined.base.len == inlined.out.len);
        assert(inlined.base.len == inlined.in.len);
        assert(target.base.len == target.out.len);
        assert(target.base.len == target.in.len);
        if (inlined.base.len != target.base.len or inlined.num != target.num) {
            return false;
        }
        for (0..inlined.base.len) |inlined_idx| {
            if (!inlined.base[inlined_idx].out.equal(target.base[inlined_idx].out) or
                !inlined.base[inlined_idx].in.equal(target.base[inlined_idx].in) or
                inlined.out[inlined_idx] != target.out[inlined_idx] or
                inlined.in[inlined_idx] != target.in[inlined_idx])
            {
                return false;
            }
        }
        return true;
    }
    pub inline fn inlinedEqualNoOffset(inlined: Inlined, target: Inlined) bool {
        assert(inlined.base.len == inlined.out.len);
        assert(inlined.base.len == inlined.in.len);
        assert(target.base.len == target.out.len);
        assert(target.base.len == target.in.len);
        if (inlined.base.len != target.base.len or inlined.num != target.num) {
            return false;
        }
        for (0..inlined.base.len) |inlined_idx| {
            if (!inlined.base[inlined_idx].out.equalNoOffset(target.base[inlined_idx].out) or
                !inlined.base[inlined_idx].in.equalNoOffset(target.base[inlined_idx].in) or
                inlined.out[inlined_idx] != target.out[inlined_idx] or
                inlined.in[inlined_idx] != target.in[inlined_idx])
            {
                return false;
            }
        }
        return true;
    }
};
// $TODO This is where tiling and SIMD stuff should happen
// Want to be able to specify the per-kernel size
// Want to be able to unroll the per-dim loops
// Want to be able to specify that an assign block should be SIMD (Meaning the unrolled per-dim loops)
pub const Split = bool;
pub const Assign = struct {
    repeats: u32,
    size: Vec4,
    base: Base,
    inlined: Inlined,
    split: Split,
    pub fn print(assign: Assign, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            util.log.print("{s}Assign {s}\n", .{ " " ** offset, text });
        }
        util.log.print("{s}({}, {}, {}, {}) Repeats: {}\n", .{
            " " ** (offset + padding),
            assign.size.a,
            assign.size.z,
            assign.size.y,
            assign.size.x,
            assign.repeats,
        });
        assign.base.print(padding, offset, null);

        if (assign.inlined.num > 0) {
            util.log.print("{s}Inlined out_base {?} in_base {?} inlined_num {}\n", //
                .{ " " ** (offset + padding), assign.inlined.out_root, assign.inlined.in_root, assign.inlined.num });
            var inlined_idx: u32 = 0;
            while (inlined_idx < assign.inlined.num) : (inlined_idx += 1) {
                util.log.print("{s}({}) out -> {?} in -> {?}\n", .{ " " ** (offset + padding), inlined_idx, assign.inlined.out[inlined_idx], assign.inlined.in[inlined_idx] });
                assign.inlined.base[inlined_idx].print(padding, padding + offset, null);
            }
        }
        if (assign.split) {
            util.log.print("{s}Splitting\n", .{" " ** (offset + padding)});
        }
    }
};
pub const Pir = @This();
assign: []Assign,
assign_num: u32,
pub fn alloc(
    gpa: Allocator,
    linearized: Linearized,
    depth_max: u32,
    size_global: u32,
    size_local: u32,
) !Pir {
    assert(linearized.num > 0);

    const assign: []Assign = try gpa.alloc(Assign, linearized.num);
    errdefer gpa.free(assign);

    for (0..linearized.num) |op_idx| {
        assign[op_idx] = .{
            .repeats = 1,
            .size = linearized.op[op_idx].view_dual.size,
            .base = .{
                .out = linearized.op[op_idx].out,
                .in = linearized.op[op_idx].in,
                .u_var = linearized.op[op_idx].u_var,
                .kind = linearized.op[op_idx].kind,
                .out_view = ViewOffset.fromView(linearized.op[op_idx].view_dual.viewOut()),
                .in_view = ViewOffset.fromView(linearized.op[op_idx].view_dual.viewIn()),
            },
            .split = false,
            .inlined = .{
                .num = 0,
                .out_root = null,
                .in_root = null,
                .base = &.{},
                .out = &.{},
                .in = &.{},
            },
        };
    }

    // Why was this ever here? Just to group assignments that could be on the same layer?
    // std.mem.sort(Assign, assign, {}, layerLessThan);

    var pir: Pir = .{
        .assign = assign,
        .assign_num = @intCast(assign.len),
    };

    const vgpu: VGpu = .{ // $FIXME This is just a dummy.
        .detail = .simple,
        .features = .{
            .simd_width_bytes = 128,
            .cache_l1_kb = 32,
            .cache_l2_kb = 512,
        },
    };

    try pir.optimize(gpa, depth_max, vgpu, size_global, size_local);
    pir.removeDefault();

    return pir;
}
pub fn copy(pir: Pir, gpa: Allocator) !Pir {
    var result: Pir = .{
        .assign = try gpa.alloc(Assign, pir.assign.len),
        .assign_num = pir.assign_num,
    };

    for (0..pir.assign_num) |assign_idx| {
        result.assign[assign_idx] = .{
            .repeats = pir.assign[assign_idx].repeats,
            .size = pir.assign[assign_idx].size,
            .base = pir.assign[assign_idx].base,
            .inlined = if (pir.assign[assign_idx].inlined.num > 0)
                .{
                    .in_root = pir.assign[assign_idx].inlined.in_root,
                    .out_root = pir.assign[assign_idx].inlined.out_root,
                    .num = pir.assign[assign_idx].inlined.num,
                    .base = gpa.dupe(Base, pir.assign[assign_idx].inlined.base) catch |err| {
                        @branchHint(.cold);
                        for (0..assign_idx) |free_idx| {
                            gpa.free(pir.assign[free_idx].inlined.base);
                            gpa.free(pir.assign[free_idx].inlined.out);
                            gpa.free(pir.assign[free_idx].inlined.in);
                        }
                        return err;
                    },
                    .out = gpa.dupe(?u32, pir.assign[assign_idx].inlined.out) catch |err| {
                        @branchHint(.cold);
                        for (0..assign_idx) |free_idx| {
                            gpa.free(pir.assign[free_idx].inlined.base);
                            gpa.free(pir.assign[free_idx].inlined.out);
                            gpa.free(pir.assign[free_idx].inlined.in);
                        }
                        gpa.free(pir.assign[assign_idx].inlined.base);
                        return err;
                    },
                    .in = gpa.dupe(?u32, pir.assign[assign_idx].inlined.in) catch |err| {
                        @branchHint(.cold);
                        for (0..assign_idx) |free_idx| {
                            gpa.free(pir.assign[free_idx].inlined.base);
                            gpa.free(pir.assign[free_idx].inlined.out);
                            gpa.free(pir.assign[free_idx].inlined.in);
                        }
                        gpa.free(pir.assign[assign_idx].inlined.base);
                        gpa.free(pir.assign[assign_idx].inlined.out);
                        return err;
                    },
                }
            else
                .{
                    .base = &.{},
                    .out = &.{},
                    .in = &.{},
                    .out_root = null,
                    .in_root = null,
                    .num = 0,
                },
            .split = pir.assign[assign_idx].split,
        };
    }

    return result;
}
pub fn free(pir: Pir, gpa: Allocator) void {
    for (0..pir.assign_num) |assign_idx| {
        if (pir.assign[assign_idx].inlined) |*inlined| {
            gpa.free(inlined.base);
            gpa.free(inlined.out);
            gpa.free(inlined.in);
        }
        // Split has nothing to free rn
        if (pir.assign[assign_idx].block) |block| {
            _ = block;
            unreachable;
        }
        if (pir.assign[assign_idx].simd) |simd| {
            _ = simd;
            unreachable;
        }
    }
    gpa.free(pir.assign);
}
fn removeDefault(pir: *Pir) void {
    for (0..pir.assign_num) |assign_idx| {
        pir.assign[assign_idx].base.out_view.removeDefault();
        pir.assign[assign_idx].base.in_view.removeDefault();

        for (0..pir.assign[assign_idx].inlined.num) |inlined_idx| {
            pir.assign[assign_idx].inlined.base[inlined_idx].out_view.removeDefault();
            pir.assign[assign_idx].inlined.base[inlined_idx].in_view.removeDefault();
        }
    }
}

/// Simple greedy search over the field of possible optimizations
fn optimize(pir: *Pir, gpa: Allocator, depth_max: u32, vgpu: VGpu, size_global: u32, size_local: u32) !void {
    const optimization_len_initial: u32 = 128; // Pretty arbitrary
    var optimization: ArrayList(Optimization) = try .initCapacity(gpa, optimization_len_initial);
    defer optimization.deinit(gpa);

    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a: Allocator = arena.allocator();

    // $TODO This is so unoptimized it might worthy of a fix me tag
    //  Would ne nice to only have to compute a diff between two iterations of this loop, but I suspect that would be very complicated
    var cost_curr: u64 = vgpu.costEstimate(pir.*, size_global, size_local);

    const stage_max: u32 = 3;
    var stage: u32 = 0;
    var depth_idx: u32 = 0;
    while (depth_idx < depth_max and stage < stage_max) {
        optimization.clearRetainingCapacity();

        // $TODO Can these be done incrementally?
        switch (stage) {
            0 => {
                try opt.mergeOpGather(gpa, &optimization, pir.*);
            },
            1 => {
                try opt.parallelizeGather(gpa, &optimization, pir.*);
                try opt.inlineOpGather(gpa, &optimization, pir.*);
            },
            2 => {
                try opt.splitKernelGather(gpa, &optimization, pir.*, size_global, size_local);
            },
            else => unreachable,
        }

        if (optimization.items.len == 0) {
            stage += 1;
            continue;
        }

        var cost_next_best: u64 = cost_curr;
        var cost_next_best_idx: u32 = 0; // Easy filter with cost_next_best == cost_curr

        var optimization_idx: u32 = 0;
        while (optimization_idx < optimization.items.len) : (optimization_idx += 1) {
            defer _ = arena.reset(.retain_capacity);

            var pir_temp: Pir = try pir.copy(a);

            switch (optimization.items[optimization_idx]) {
                .parallelize => |parallelize| {
                    assert(stage == 1);
                    try opt.parallelize(a, &pir_temp, parallelize.left_idx, parallelize.right_idx);
                },
                .inlined => |inlined| {
                    assert(stage == 1);
                    try opt.inlineOp(a, &pir_temp, inlined.left_idx, inlined.right_idx_max_written);
                },
                .fuse => |fuse| {
                    assert(stage == 0);
                    opt.mergeOp(a, &pir_temp, fuse.left_idx, fuse.right_idx);
                },
                .split => |split| {
                    assert(stage == 2);
                    opt.splitKernel(&pir_temp, split.idx);
                },
            }

            const cost_next: u64 = vgpu.costEstimate(pir_temp, size_global, size_local);

            if (cost_next < cost_next_best) {
                cost_next_best = cost_next;
                cost_next_best_idx = optimization_idx;
            }
        }

        if (cost_next_best == cost_curr) {
            stage += 1;
            break; // No better optimization found
        } else {
            depth_idx += 1;
            switch (optimization.items[cost_next_best_idx]) {
                .parallelize => |parallelize| {
                    assert(stage == 1);
                    try opt.parallelize(gpa, pir, parallelize.left_idx, parallelize.right_idx);
                },
                .inlined => |inlined| {
                    assert(stage == 1);
                    try opt.inlineOp(gpa, pir, inlined.left_idx, inlined.right_idx_max_written);
                },
                .fuse => |fuse| {
                    assert(stage == 0);
                    opt.mergeOp(gpa, pir, fuse.left_idx, fuse.right_idx);
                },
                .split => |split| {
                    assert(stage == 2);
                    opt.splitKernel(pir, split.idx);
                },
            }
            cost_curr = cost_next_best;
        }
    }
}
pub fn print(pir: Pir, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
    if (name) |text| {
        util.log.print("{s}PIR {s}\n", .{ " " ** offset, text });
    } else {
        util.log.print("{s}PIR\n", .{" " ** offset});
    }
    for (0..pir.assign_num) |assign_idx| {
        util.log.print("{s}[{}] =>\n", .{ " " ** offset, assign_idx });
        pir.assign[assign_idx].print(padding, offset + padding, null);
    }
}

// Assign:
//  Remove `out` and `in` fields in favor of zig compiler Pool style index approach
//      The byte at that index specifies what the input / output are: buffer, assign etc.
//      Here the child "assign" can actually be smaller as repeats don't need to be saved. Should probably actually store the repeats in the Pool above the `Assign` struct, avoids having duplicate code for checking equality with AssignInlined (Doesn't exist in this case) This also removes the need for an `Inlined` struct OffsetGen only needs to exist in leaf assigns (store as idx, encoding could work but that complicates comparison, actually maybe comparisons only make sense between the same type of assign) Pir: One Pir still corresponds to one program
//  Underlying structure needs to be changed entirely obviously
//  Storing the `Base` struct in a normal array still feels better though

// Fuse ops, parallelize ops, constant folding etc. can remain largely unchanged

// Split:
//  This one is tricky now. I felt like it should just output more ops
//  however that makes things very tricky when dim sizes aren't a multiple of the global / local size
//  Old way should work still, but it feels weirder
//  Could have an identifier in the Pool that is like a thin wrapper over base
//  Meaning you have like ...{split, count, base_idx}...

// Pool:
//  identifiers: AssignInOut, AssignIn, AssignOut, Assign, Split, Buffer, BufferIntermediary, none
//      ------- AssignX means that X is *not* inlined in that case
//  Unsure how to handle freeing memory here, could do free list with merging on adjacency

// Forgot to record git hash :^(
// profile_compiler: rng=11714000457094572098...
// Time:   2.3557ms +-   0.7386ms linearized
//
// Gathering :               +0ns
// Copying   :               +0ns
// Cost      :               +0ns
// Optimizing:               +0ns
// Time: 724.1800us +-  50.0590us d=0
//
// Gathering :         +2818019ns
// Copying   :        +14147167ns
// Cost      :        +28918162ns
// Optimizing:         +1537003ns
// Time: 716.6720us +-  36.8470us d=1
//
// Gathering :        +26789020ns
// Copying   :       +139502661ns
// Cost      :       +283695473ns
// Optimizing:        +15072778ns
// Time: 696.8480us +-  57.9700us d=10
//
// Gathering :       +202526986ns
// Copying   :      +1122446910ns
// Cost      :      +1822257750ns
// Optimizing:        +88829460ns
// Time: 549.7720us +-  38.3900us d=100
//
// Gathering :       +299521111ns
// Copying   :      +1439456366ns
// Cost      :      +2175965461ns
// Optimizing:        +96041078ns
// Time: 312.0550us +-  20.1140us d=1000

// 6bf9a126cb2bc2c8068befc9690615bd4733e992
// profile_compiler: rng=11714000457094572098...
// Time:   2.8005ms +-   0.0431ms linearized
//
// Gathering :               +0ns
// Copying   :               +0ns
// Cost      :               +0ns
// Optimizing:               +0ns
// Time: 715.8550us +-  38.1960us d=0
//
// Gathering :         +5841578ns
// Copying   :        +13768849ns
// Cost      :        +17046366ns
// Optimizing:         +2583720ns
// Time: 707.6160us +-  18.5350us d=1
//
// Gathering :        +58253662ns
// Copying   :       +136962502ns
// Cost      :       +170673309ns
// Optimizing:        +25991059ns
// Time: 680.3150us +-  23.6710us d=10
//
// Gathering :       +546229150ns
// Copying   :       +985414552ns
// Cost      :      +1175031847ns
// Optimizing:       +188669409ns
// Time: 324.4180us +-  40.5340us d=100
//
// Gathering :       +848090734ns
// Copying   :      +1312283259ns
// Cost      :      +1414031724ns
// Optimizing:       +213331728ns
// Time: 183.3490us +-   9.0080us d=1000

// 78dc36594c5c149d04f8d7f5339be5dd9744d21f
// profile_compiler: rng=11714000457094572098...
// Time:   2.8283ms +-   0.0679ms linearized
//
// Gathering :               +0ns
// Copying   :               +0ns
// Cost      :               +0ns
// Optimizing:               +0ns
// Time: 735.7960us +-  83.4950us d=0
//
// Gathering :         +1461340ns
// Copying   :          +518449ns
// Cost      :          +245310ns
// Optimizing:          +220244ns
// Time: 722.6050us +-  72.4110us d=1
//
// Gathering :        +14153816ns
// Copying   :         +4050156ns
// Cost      :         +2335733ns
// Optimizing:         +2033501ns
// Time: 683.3290us +-  73.9890us d=10
//
// Gathering :        +56773023ns
// Copying   :         +9516437ns
// Cost      :         +5467612ns
// Optimizing:         +4930680ns
// Time: 348.7810us +-  51.4790us d=100
//
// Gathering :        +57639502ns
// Copying   :        +23574353ns
// Cost      :         +9108663ns
// Optimizing:         +5175524ns
// Time:  94.3810us +-  25.8020us d=1000
