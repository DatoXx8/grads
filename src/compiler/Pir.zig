const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Tensor = @import("../Tensor.zig");
const Op = Tensor.Op;
const Linearized = Tensor.Linearized;
const Buffer = Tensor.Buffer;
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
    pub fn equal(this: DimInfo, target: DimInfo) bool {
        return this.off == target.off and
            this.a_stride == target.a_stride and this.z_stride == target.z_stride and
            this.y_stride == target.y_stride and this.x_stride == target.x_stride and
            this.a_wait == target.a_wait and this.z_wait == target.z_wait and
            this.y_wait == target.y_wait and this.x_wait == target.x_wait and
            this.a_reset == target.a_reset and this.z_reset == target.z_reset and
            this.y_reset == target.y_reset and this.x_reset == target.x_reset;
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}DimInfo {s}\n", .{ [1]u8{' '} ** offset, text });
        }
        std.debug.print("{s}str => ({d:10}, {d:10}, {d:10}, {d:10})\n", .{
            " " ** (offset + padding), //
            this.a_stride, this.z_stride, this.y_stride, this.x_stride, //
        });
        std.debug.print("{s}res => ({d:10}, {d:10}, {d:10}, {d:10})\n", .{
            " " ** (offset + padding), //
            this.a_reset, this.z_reset, this.y_reset, this.x_reset, //
        });
        std.debug.print("{s}wai => ({d:10}, {d:10}, {d:10}, {d:10})\n", .{
            " " ** (offset + padding), //
            this.a_wait, this.z_wait, this.y_wait, this.x_wait, //
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
    pub inline fn equal(this: @This(), target: @This()) bool {
        return this.out.equal(target.out) and this.in.equal(target.in) and
            this.kind == target.kind and this.u_var == target.u_var;
    }
    pub inline fn equalNoOffset(this: @This(), target: @This()) bool {
        return this.out.equalNoOffset(target.out) and this.in.equalNoOffset(target.in) and
            this.kind == target.kind and this.u_var == target.u_var;
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Base {s}\n", .{ " " ** offset, text });
        }
        if (this.kind.isUnary()) {
            std.debug.print("{s}U {s} ({d} {d} {d} {d}) [{d}, {d}, {d}, {d} = {d}] {} \"{s}\" {d}\n", .{
                " " ** (offset + padding),
                switch (this.kind) {
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
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.aOffset(),
                this.out.zOffset(),
                this.out.yOffset(),
                this.out.xOffset(),
                this.out.offset,
                this.out.intermediary,
                this.out.name(),
                this.u_var,
            });
        } else {
            const op_kind: u8 = if (this.kind.isBinary()) 'B' else (if (this.kind.isExpand()) 'E' else (if (this.kind.isReduce()) 'R' else unreachable));
            std.debug.print("{s}{c} {s} ({d} {d} {d} {d}) [{d}, {d}, {d}, {d} = {d}] {} \"{s}\" ({d} {d} {d} {d}) [{d}, {d}, {d}, {d} = {d}] {} \"{s}\"\n", .{
                " " ** (offset + padding),
                op_kind,
                switch (this.kind) {
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
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.aOffset(),
                this.out.zOffset(),
                this.out.yOffset(),
                this.out.xOffset(),
                this.out.offset,
                this.out.intermediary,
                this.out.name(),
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.aOffset(),
                this.in.zOffset(),
                this.in.yOffset(),
                this.in.xOffset(),
                this.in.offset,
                this.in.intermediary,
                this.in.name(),
            });
        }
        std.debug.print("{s}Repeats {}\n", .{ " " ** (offset + padding), this.repeats });
        this.out_dim.print(padding, padding + offset, "out_dim");
        this.in_dim.print(padding, padding + offset, "in_dim");
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
    pub inline fn inlinedEqual(this: @This(), target: Inlined) bool {
        assert(this.base.len == this.out.len);
        assert(this.base.len == this.in.len);
        assert(target.base.len == target.out.len);
        assert(target.base.len == target.in.len);
        if (this.base.len != target.base.len) {
            return false;
        }
        for (0..this.base.len) |inlined_idx| {
            if (!this.base[inlined_idx].out.equal(target.base[inlined_idx].out) or
                !this.base[inlined_idx].in.equal(target.base[inlined_idx].in) or
                this.out[inlined_idx] != target.out[inlined_idx] or
                this.in[inlined_idx] != target.in[inlined_idx])
            {
                return false;
            }
        }
        return true;
    }
    pub inline fn inlinedEqualNoOffset(this: @This(), target: Inlined) bool {
        assert(this.base.len == this.out.len);
        assert(this.base.len == this.in.len);
        assert(target.base.len == target.out.len);
        assert(target.base.len == target.in.len);
        if (this.base.len != target.base.len) {
            return false;
        }
        for (0..this.base.len) |inlined_idx| {
            if (!this.base[inlined_idx].out.equalNoOffset(target.base[inlined_idx].out) or
                !this.base[inlined_idx].in.equalNoOffset(target.base[inlined_idx].in) or
                this.out[inlined_idx] != target.out[inlined_idx] or
                this.in[inlined_idx] != target.in[inlined_idx])
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
    inlined: ?Inlined,
    split: Split,
    block: ?Block,
    simd: ?Simd,
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Assign {s}\n", .{ " " ** offset, text });
        }
        this.base.print(padding, offset, null);
        if (this.inlined) |inlined| {
            std.debug.print("{s}Inlined out_base {?} in_base {?}\n", .{ " " ** (offset + padding), inlined.out_root, inlined.in_root });
            for (0..inlined.inlined_num) |inlined_idx| {
                std.debug.print("{s}({}) out -> {?} in -> {?}\n", .{ " " ** (offset + padding), inlined_idx, inlined.out[inlined_idx], inlined.in[inlined_idx] });
                inlined.base[inlined_idx].print(padding, padding + offset, null);
            }
        }
        if (this.split) {
            std.debug.print("{s}Splitting\n", .{" " ** (offset + padding)});
        }
        if (this.block) |block| {
            _ = block;
            unreachable;
        }
        if (this.simd) |simd| {
            _ = simd;
            unreachable;
        }
    }
};
pub const Pir = @This();
assign: []Assign,
assign_num: u32,
pub fn alloc(
    allocator: Allocator,
    linearized: Linearized,
    depth_max: u32,
    size_global: u32,
    size_local: u32,
) !Pir {
    assert(linearized.op_num > 0);

    const assign: []Assign = try allocator.alloc(Assign, linearized.op_num);
    errdefer allocator.free(assign);

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
    try pir.optimize(allocator, depth_max, vgpu, size_global, size_local);
    pir.removeDefault();

    return pir;
}
// $TODO This should really not involve any allocator calls when switching to Pool based impl
pub fn copy(this: Pir, allocator: Allocator) !Pir {
    var result: Pir = .{
        .assign = try allocator.alloc(Assign, this.assign.len),
        .assign_num = this.assign_num,
    };

    for (0..this.assign_num) |assign_idx| {
        result.assign[assign_idx] = .{
            .base = this.assign[assign_idx].base,
            .inlined = if (this.assign[assign_idx].inlined) |inlined|
                .{
                    .in_root = inlined.in_root,
                    .out_root = inlined.out_root,
                    .inlined_num = inlined.inlined_num,
                    .base = try allocator.dupe(Base, inlined.base), // $FIXME If this fails memory leaks
                    .out = try allocator.dupe(?u32, inlined.out), // $FIXME If this fails memory leaks
                    .in = try allocator.dupe(?u32, inlined.in), // $FIXME If this fails memory leaks
                }
            else
                null,
            .split = this.assign[assign_idx].split,
            .simd = null,
            .block = null,
        };
        assert(this.assign[assign_idx].simd == null);
        assert(this.assign[assign_idx].block == null);
    }

    return result;
}
pub fn free(this: *@This(), allocator: Allocator) void {
    for (0..this.assign_num) |assign_idx| {
        if (this.assign[assign_idx].inlined) |*inlined| {
            allocator.free(inlined.base);
            allocator.free(inlined.out);
            allocator.free(inlined.in);
        }
        // Split has nothing to free rn
        if (this.assign[assign_idx].block) |block| {
            _ = block;
            unreachable;
        }
        if (this.assign[assign_idx].simd) |simd| {
            _ = simd;
            unreachable;
        }
    }
    allocator.free(this.assign);
}
fn removeDefault(this: *@This()) void {
    for (0..this.assign_num) |assign_idx| {
        inline for (0..2) |dim_idx| {
            const dim_info: *DimInfo = if (dim_idx == 0)
                &this.assign[assign_idx].base.out_dim
            else
                &this.assign[assign_idx].base.in_dim;
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
            if (this.assign[assign_idx].inlined) |inlined| {
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
/// Simple greedy search over the field of possible optimizations
fn optimize(this: *Pir, allocator: Allocator, depth_max: u32, vgpu: VGpu, size_global: u32, size_local: u32) !void {
    const optimization_len_initial: u32 = 128; // Pretty arbitrary
    var optimization_count: u32 = 0;
    var optimization: []Optimization = try allocator.alloc(Optimization, optimization_len_initial);
    defer allocator.free(optimization);

    // $TODO This is so unoptimized and horrible it might worthy of a fix me tag

    var cost_curr: u64 = vgpu.costEstimate(this.*, size_global, size_local);
    var depth_idx: u32 = 0;
    while (depth_idx < depth_max) : (depth_idx += 1) {
        optimization_count = 0;
        try opt.parallelizeGather(allocator, &optimization, &optimization_count, this.*);
        try opt.inlineOpGather(allocator, &optimization, &optimization_count, this.*); // $FIXME This doesn't seem to do anything
        try opt.mergeOpGather(allocator, &optimization, &optimization_count, this.*);
        try opt.splitKernelGather(allocator, &optimization, &optimization_count, this.*, size_global, size_local);

        if (optimization_count == 0) {
            break;
        }

        var cost_next_best: u64 = cost_curr;
        var cost_next_best_idx: u32 = 0; // Easy filter with cost_next_best == cost_curr

        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();
        const a: Allocator = arena.allocator();

        var optimization_idx: u32 = 0;
        while (optimization_idx < optimization_count) : (optimization_idx += 1) {
            defer _ = arena.reset(.retain_capacity);

            var pir_temp: Pir = try this.copy(a);

            switch (optimization[optimization_idx]) {
                .parallelize => |parallelize| {
                    try opt.parallelize(a, &pir_temp, parallelize.left_idx, parallelize.right_idx);
                },
                .inlined => |inlined| {
                    try opt.inlineOp(a, &pir_temp, inlined.idx);
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
                    try opt.parallelize(allocator, this, parallelize.left_idx, parallelize.right_idx);
                },
                .inlined => |inlined| {
                    try opt.inlineOp(allocator, this, inlined.idx);
                },
                .split => |split| {
                    opt.splitKernel(this, split.idx);
                },
                .fuse => |fuse| {
                    opt.mergeOp(this, fuse.left_idx, fuse.right_idx);
                },
            }
            cost_curr = cost_next_best;
        }
    }
}
pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
    if (name) |text| {
        std.debug.print("{s}PIR {s}\n", .{ " " ** offset, text });
    } else {
        std.debug.print("{s}PIR\n", .{" " ** offset});
    }
    for (0..this.assign_num) |assign_idx| {
        std.debug.print("{s}[{}] => \n", .{ " " ** offset, assign_idx });
        this.assign[assign_idx].print(padding, offset + padding, null);
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
