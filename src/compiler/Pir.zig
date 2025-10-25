const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Linearized = @import("../Linearized.zig");
const Op = Linearized.Op;
const Buffer = @import("../Buffer.zig");
const opt = @import("optimize.zig");
const Optimization = opt.Optimization;

const VGpu = @import("VGpu.zig");

pub const DimInfo = struct {
    pub const value_none: u32 = std.math.maxInt(u32);
    pub const wait_default: u32 = 1;
    pub const stride_default: u32 = 0;
    /// Just the highest bit
    pub const reset_default: u32 = ~(value_none >> 1);
    off: u32,
    a_stride: u32,
    z_stride: u32,
    y_stride: u32,
    x_stride: u32,
    a_reset: u32,
    z_reset: u32,
    y_reset: u32,
    x_reset: u32,
    a_wait: u32,
    z_wait: u32,
    y_wait: u32,
    x_wait: u32,
    pub fn init(offset: u32) DimInfo {
        return .{
            .off = offset,
            .a_stride = value_none,
            .z_stride = value_none,
            .y_stride = value_none,
            .x_stride = value_none,
            .a_reset = value_none,
            .z_reset = value_none,
            .y_reset = value_none,
            .x_reset = value_none,
            .a_wait = value_none,
            .z_wait = value_none,
            .y_wait = value_none,
            .x_wait = value_none,
        };
    }
    pub fn equal(dim_info: DimInfo, target: DimInfo) bool {
        return dim_info.off == target.off and
            dim_info.a_stride == target.a_stride and dim_info.z_stride == target.z_stride and
            dim_info.y_stride == target.y_stride and dim_info.x_stride == target.x_stride and
            dim_info.a_wait == target.a_wait and dim_info.z_wait == target.z_wait and
            dim_info.y_wait == target.y_wait and dim_info.x_wait == target.x_wait and
            dim_info.a_reset == target.a_reset and dim_info.z_reset == target.z_reset and
            dim_info.y_reset == target.y_reset and dim_info.x_reset == target.x_reset;
    }
    pub fn print(dim_info: DimInfo, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}DimInfo {s}\n", .{ [1]u8{' '} ** offset, text });
        }
        std.debug.print("{s}str => ({d:10}, {d:10}, {d:10}, {d:10})\n", .{
            " " ** (offset + padding), //
            dim_info.a_stride, dim_info.z_stride, dim_info.y_stride, dim_info.x_stride, //
        });
        std.debug.print("{s}res => ({d:10}, {d:10}, {d:10}, {d:10})\n", .{
            " " ** (offset + padding), //
            dim_info.a_reset, dim_info.z_reset, dim_info.y_reset, dim_info.x_reset, //
        });
        std.debug.print("{s}wai => ({d:10}, {d:10}, {d:10}, {d:10})\n", .{
            " " ** (offset + padding), //
            dim_info.a_wait, dim_info.z_wait, dim_info.y_wait, dim_info.x_wait, //
        });
    }
};
// I removed the dependency layers for now, because they just weren't used anywhere. Add those back if necessary.
/// The basic thing the Assignment does without any funny business
pub const Base = struct {
    out: Buffer,
    in: Buffer,
    kind: Op.Kind,
    u_var: f32,
    repeats: u32,
    out_dim: DimInfo,
    in_dim: DimInfo,
    pub inline fn equal(base: Base, target: Base) bool {
        return base.out.equal(target.out) and base.in.equal(target.in) and
            base.kind == target.kind and base.u_var == target.u_var;
    }
    pub inline fn equalNoOffset(base: Base, target: Base) bool {
        return base.out.equalNoOffset(target.out) and base.in.equalNoOffset(target.in) and
            base.kind == target.kind and base.u_var == target.u_var;
    }
    pub fn print(base: Base, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Base {s}\n", .{ " " ** offset, text });
        }
        if (base.kind.isUnary()) {
            std.debug.print("{s}U {s} ({d} {d} {d} {d}) [{d}, {d}, {d}, {d} = {d}] {} \"{s}\" {d}\n", .{
                " " ** (offset + padding),
                switch (base.kind) {
                    .unary_add => "add",
                    .unary_subtract => "sub",
                    .unary_multiply => "mul",
                    .unary_divide => "div",
                    .unary_exp => "exp",
                    .unary_log => "log",
                    .unary_square => "sqr",
                    .unary_sqrt => "sqt",
                    .unary_reciprocal => "rcp",
                    .unary_max => "max",
                    .unary_min => "min",
                    .unary_set => "set",
                    .unary_random => "rng",
                    .unary_tanh => "tanh",
                    .unary_absolute => "abs",
                    .unary_sign => "sgn",
                    else => unreachable,
                },
                base.out.a_size,
                base.out.z_size,
                base.out.y_size,
                base.out.x_size,
                base.out.aOffset(),
                base.out.zOffset(),
                base.out.yOffset(),
                base.out.xOffset(),
                base.out.offset,
                base.out.intermediary,
                base.out.name(),
                base.u_var,
            });
        } else {
            const op_kind: u8 = if (base.kind.isBinary()) 'B' else (if (base.kind.isExpand()) 'E' else (if (base.kind.isReduce()) 'R' else unreachable));
            std.debug.print("{s}{c} {s} ({d} {d} {d} {d}) [{d}, {d}, {d}, {d} = {d}] {} \"{s}\" ({d} {d} {d} {d}) [{d}, {d}, {d}, {d} = {d}] {} \"{s}\"\n", .{
                " " ** (offset + padding),
                op_kind,
                switch (base.kind) {
                    .binary_add => "add",
                    .binary_subtract => "sub",
                    .binary_multiply => "mul",
                    .binary_divide => "div",
                    .binary_max => "max",
                    .binary_min => "min",
                    .binary_set => "set",
                    .expand_add => "add",
                    .expand_subtract => "sub",
                    .expand_multiply => "mul",
                    .expand_divide => "div",
                    .expand_max => "max",
                    .expand_min => "min",
                    .expand_set => "set",
                    .reduce_sum => "sum",
                    .reduce_max => "max",
                    .reduce_min => "min",
                    .reduce_avg => "avg",
                    else => unreachable,
                },
                base.out.a_size,
                base.out.z_size,
                base.out.y_size,
                base.out.x_size,
                base.out.aOffset(),
                base.out.zOffset(),
                base.out.yOffset(),
                base.out.xOffset(),
                base.out.offset,
                base.out.intermediary,
                base.out.name(),
                base.in.a_size,
                base.in.z_size,
                base.in.y_size,
                base.in.x_size,
                base.in.aOffset(),
                base.in.zOffset(),
                base.in.yOffset(),
                base.in.xOffset(),
                base.in.offset,
                base.in.intermediary,
                base.in.name(),
            });
        }
        std.debug.print("{s}Repeats {}\n", .{ " " ** (offset + padding), base.repeats });
        base.out_dim.print(padding, padding + offset, "out_dim");
        base.in_dim.print(padding, padding + offset, "in_dim");
    }
};
// $TODO Maybe make all these slices from a global / per ssa buffer
/// Tree representation of inlined ops
pub const Inlined = struct {
    base: []Base,
    out: []?u32,
    in: []?u32,
    out_root: ?u32,
    in_root: ?u32,
    inlined_num: u32,
    pub inline fn inlinedEqual(inlined: Inlined, target: Inlined) bool {
        assert(inlined.base.len == inlined.out.len);
        assert(inlined.base.len == inlined.in.len);
        assert(target.base.len == target.out.len);
        assert(target.base.len == target.in.len);
        if (inlined.base.len != target.base.len) {
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
        if (inlined.base.len != target.base.len) {
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
/// Wether or not to split a single operation across kernels
pub const Split = bool;
/// Describes how to utilize blocks for better caching
pub const Block = struct {
    //
};
/// Describes the SIMD width and tells to codegen to actually use SIMD
pub const Simd = struct {
    //
};
/// Essentialy this is one unit of work
pub const Assign = struct {
    base: Base,
    inlined: ?Inlined, // $TODO This should not be optional
    split: Split,
    block: ?Block,
    simd: ?Simd,
    pub fn print(assign: Assign, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Assign {s}\n", .{ " " ** offset, text });
        }
        assign.base.print(padding, offset, null);
        if (assign.inlined) |inlined| {
            std.debug.print("{s}Inlined out_base {?} in_base {?} inlined_num {}\n", //
                .{ " " ** (offset + padding), inlined.out_root, inlined.in_root, inlined.inlined_num });
            for (0..inlined.inlined_num) |inlined_idx| {
                std.debug.print("{s}({}) out -> {?} in -> {?}\n", .{ " " ** (offset + padding), inlined_idx, inlined.out[inlined_idx], inlined.in[inlined_idx] });
                inlined.base[inlined_idx].print(padding, padding + offset, null);
            }
        }
        if (assign.split) {
            std.debug.print("{s}Splitting\n", .{" " ** (offset + padding)});
        }
        if (assign.block) |block| {
            _ = block;
            unreachable;
        }
        if (assign.simd) |simd| {
            _ = simd;
            unreachable;
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
    assert(linearized.op_num > 0);

    const assign: []Assign = try gpa.alloc(Assign, linearized.op_num);
    errdefer gpa.free(assign);

    for (0..linearized.op_num) |op_idx| {
        assign[op_idx] = .{
            .base = .{
                .kind = linearized.op[op_idx].kind,
                .u_var = linearized.op[op_idx].u_var,
                .out = linearized.op[op_idx].out,
                .in = linearized.op[op_idx].in,
                .out_dim = DimInfo.init(assign[op_idx].base.out.offset),
                .in_dim = DimInfo.init(assign[op_idx].base.in.offset),
                .repeats = 1,
            },
            .inlined = null,
            .split = false,
            .simd = null,
            .block = null,
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
            .base = pir.assign[assign_idx].base,
            .inlined = if (pir.assign[assign_idx].inlined) |inlined|
                .{
                    .in_root = inlined.in_root,
                    .out_root = inlined.out_root,
                    .inlined_num = inlined.inlined_num,
                    .base = gpa.dupe(Base, inlined.base) catch |err| {
                        @branchHint(.cold);
                        for (0..assign_idx) |free_idx| {
                            if (pir.assign[free_idx].inlined) |inlined_free| {
                                gpa.free(inlined_free.base);
                                gpa.free(inlined_free.out);
                                gpa.free(inlined_free.in);
                            }
                        }
                        return err;
                    },
                    .out = gpa.dupe(?u32, inlined.out) catch |err| {
                        @branchHint(.cold);
                        for (0..assign_idx) |free_idx| {
                            if (pir.assign[free_idx].inlined) |inlined_free| {
                                gpa.free(inlined_free.base);
                                gpa.free(inlined_free.out);
                                gpa.free(inlined_free.in);
                            }
                        }
                        gpa.free(inlined.base);
                        return err;
                    },
                    .in = gpa.dupe(?u32, inlined.in) catch |err| {
                        @branchHint(.cold);
                        for (0..assign_idx) |free_idx| {
                            if (pir.assign[free_idx].inlined) |inlined_free| {
                                gpa.free(inlined_free.base);
                                gpa.free(inlined_free.out);
                                gpa.free(inlined_free.in);
                            }
                        }
                        gpa.free(inlined.base);
                        gpa.free(inlined.out);
                        return err;
                    },
                }
            else
                null,
            .split = pir.assign[assign_idx].split,
            .simd = pir.assign[assign_idx].simd,
            .block = pir.assign[assign_idx].block,
        };
        assert(pir.assign[assign_idx].simd == null);
        assert(pir.assign[assign_idx].block == null);
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
        inline for (0..2) |dim_idx| {
            const dim_info: *DimInfo = if (dim_idx == 0)
                &pir.assign[assign_idx].base.out_dim
            else
                &pir.assign[assign_idx].base.in_dim;
            if (dim_info.a_wait == DimInfo.value_none) dim_info.a_wait = DimInfo.wait_default;
            if (dim_info.z_wait == DimInfo.value_none) dim_info.z_wait = DimInfo.wait_default;
            if (dim_info.y_wait == DimInfo.value_none) dim_info.y_wait = DimInfo.wait_default;
            if (dim_info.x_wait == DimInfo.value_none) dim_info.x_wait = DimInfo.wait_default;

            if (dim_info.a_stride == DimInfo.value_none) dim_info.a_stride = DimInfo.stride_default;
            if (dim_info.z_stride == DimInfo.value_none) dim_info.z_stride = DimInfo.stride_default;
            if (dim_info.y_stride == DimInfo.value_none) dim_info.y_stride = DimInfo.stride_default;
            if (dim_info.x_stride == DimInfo.value_none) dim_info.x_stride = DimInfo.stride_default;

            if (dim_info.a_reset == DimInfo.value_none) dim_info.a_reset = DimInfo.reset_default;
            if (dim_info.z_reset == DimInfo.value_none) dim_info.z_reset = DimInfo.reset_default;
            if (dim_info.y_reset == DimInfo.value_none) dim_info.y_reset = DimInfo.reset_default;
            if (dim_info.x_reset == DimInfo.value_none) dim_info.x_reset = DimInfo.reset_default;
            if (pir.assign[assign_idx].inlined) |inlined| {
                for (0..inlined.inlined_num) |inlined_idx| {
                    const inlined_info: *DimInfo = if (dim_idx == 0)
                        &inlined.base[inlined_idx].out_dim
                    else
                        &inlined.base[inlined_idx].in_dim;
                    if (inlined_info.a_wait == DimInfo.value_none) inlined_info.a_wait = DimInfo.wait_default;
                    if (inlined_info.z_wait == DimInfo.value_none) inlined_info.z_wait = DimInfo.wait_default;
                    if (inlined_info.y_wait == DimInfo.value_none) inlined_info.y_wait = DimInfo.wait_default;
                    if (inlined_info.x_wait == DimInfo.value_none) inlined_info.x_wait = DimInfo.wait_default;

                    if (inlined_info.a_stride == DimInfo.value_none) inlined_info.a_stride = DimInfo.stride_default;
                    if (inlined_info.z_stride == DimInfo.value_none) inlined_info.z_stride = DimInfo.stride_default;
                    if (inlined_info.y_stride == DimInfo.value_none) inlined_info.y_stride = DimInfo.stride_default;
                    if (inlined_info.x_stride == DimInfo.value_none) inlined_info.x_stride = DimInfo.stride_default;

                    if (inlined_info.a_reset == DimInfo.value_none) inlined_info.a_reset = DimInfo.reset_default;
                    if (inlined_info.z_reset == DimInfo.value_none) inlined_info.z_reset = DimInfo.reset_default;
                    if (inlined_info.y_reset == DimInfo.value_none) inlined_info.y_reset = DimInfo.reset_default;
                    if (inlined_info.x_reset == DimInfo.value_none) inlined_info.x_reset = DimInfo.reset_default;
                }
            }
        }
    }
}

// profile_compiler: rng=11714000457094572098...
// Time:   2.3557ms +-   0.7386ms linearized
// Gathering :               +0ns
// Copying   :               +0ns
// Cost      :               +0ns
// Optimizing:               +0ns
//
// Time: 724.1800us +-  50.0590us d=0
//
//
// Gathering :         +2818019ns
// Copying   :        +14147167ns
// Cost      :        +28918162ns
// Optimizing:         +1537003ns
//
// Time: 716.6720us +-  36.8470us d=1
//
//
// Gathering :        +26789020ns
// Copying   :       +139502661ns
// Cost      :       +283695473ns
// Optimizing:        +15072778ns
//
// Time: 696.8480us +-  57.9700us d=10
//
//
// Gathering :       +202526986ns
// Copying   :      +1122446910ns
// Cost      :      +1822257750ns
// Optimizing:        +88829460ns
//
// Time: 549.7720us +-  38.3900us d=100
//
//
// Gathering :       +299521111ns
// Copying   :      +1439456366ns
// Cost      :      +2175965461ns
// Optimizing:        +96041078ns
//
// Time: 312.0550us +-  20.1140us d=1000

// For d=1000 there are 354784 * 3 copies for inline fields alone!

/// Simple greedy search over the field of possible optimizations
fn optimize(pir: *Pir, gpa: Allocator, depth_max: u32, vgpu: VGpu, size_global: u32, size_local: u32) !void {
    const optimization_len_initial: u32 = 128; // Pretty arbitrary
    var optimization: []Optimization = try gpa.alloc(Optimization, optimization_len_initial);
    defer gpa.free(optimization);

    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a: Allocator = arena.allocator();

    // $TODO This is so unoptimized it might worthy of a fix me tag
    //  Would ne nice to only have to compute a diff between two iterations of this loop, but I suspect that would be very complicated

    var cost_curr: u64 = vgpu.costEstimate(pir.*, size_global, size_local);
    var depth_idx: u32 = 0;
    while (depth_idx < depth_max) : (depth_idx += 1) {
        // $TODO Can this be done incrementally?

        var optimization_count: u32 = 0;
        try opt.parallelizeGather(gpa, &optimization, &optimization_count, pir.*);
        try opt.inlineOpGather(gpa, &optimization, &optimization_count, pir.*);
        try opt.mergeOpGather(gpa, &optimization, &optimization_count, pir.*);
        try opt.splitKernelGather(gpa, &optimization, &optimization_count, pir.*, size_global, size_local);

        if (optimization_count == 0) {
            break;
        }

        var cost_next_best: u64 = cost_curr;
        var cost_next_best_idx: u32 = 0; // Easy filter with cost_next_best == cost_curr

        var optimization_idx: u32 = 0;
        while (optimization_idx < optimization_count) : (optimization_idx += 1) {
            defer _ = arena.reset(.retain_capacity);

            var pir_temp: Pir = try pir.copy(a);

            switch (optimization[optimization_idx]) {
                .parallelize => |parallelize| {
                    try opt.parallelize(a, &pir_temp, parallelize.left_idx, parallelize.right_idx);
                },
                .inlined => |inlined| {
                    try opt.inlineOp(a, &pir_temp, inlined.left_idx);
                },
                .split => |split| {
                    opt.splitKernel(&pir_temp, split.idx);
                },
                .fuse => |fuse| {
                    opt.mergeOp(&pir_temp, fuse.left_idx, fuse.right_idx);
                },
            }

            const cost_next: u64 = vgpu.costEstimate(pir_temp, size_global, size_local);

            if (cost_next < cost_next_best) {
                cost_next_best = cost_next;
                cost_next_best_idx = optimization_idx;
            }
        }

        if (cost_next_best == cost_curr) {
            break; // No better optimization found
        } else {
            switch (optimization[cost_next_best_idx]) {
                .parallelize => |parallelize| {
                    try opt.parallelize(gpa, pir, parallelize.left_idx, parallelize.right_idx);
                },
                .inlined => |inlined| {
                    try opt.inlineOp(gpa, pir, inlined.left_idx);
                },
                .split => |split| {
                    opt.splitKernel(pir, split.idx);
                },
                .fuse => |fuse| {
                    opt.mergeOp(pir, fuse.left_idx, fuse.right_idx);
                },
            }
            cost_curr = cost_next_best;
        }
    }
}
pub fn print(pir: Pir, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
    if (name) |text| {
        std.debug.print("{s}PIR {s}\n", .{ " " ** offset, text });
    } else {
        std.debug.print("{s}PIR\n", .{" " ** offset});
    }
    for (0..pir.assign_num) |assign_idx| {
        std.debug.print("{s}[{}] => \n", .{ " " ** offset, assign_idx });
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
