const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Tensor = @import("../Tensor.zig");
const Op = Tensor.Op;
const Linearized = Tensor.Linearized;
const Buffer = Tensor.Buffer;
const opt = @import("optimize.zig");
const Optimization = opt.Optimization;

pub const DimInfo = struct {
    pub const value_none: u32 = std.math.maxInt(u32);
    pub const wait_default: u32 = 1;
    pub const stride_default: u32 = 0;
    pub const reset_default: u32 = ~(value_none >> 1); // Just the highest bit
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
    type: Op.Type,
    u_var: f32,
    // $FIXME kind of unhappy this is separate but having it in dim_info would store it twice for no reason
    repeats: u32,
    out_dim: DimInfo,
    in_dim: DimInfo,
    // $TODO Unsure how to check the layer things
    pub inline fn equal(this: @This(), target: @This()) bool {
        return this.out.equal(target.out) and this.in.equal(target.in) and
            this.type == target.type and this.u_var == target.u_var;
    }
    pub inline fn equalNoOffset(this: @This(), target: @This()) bool {
        return this.out.equalNoOffset(target.out) and this.in.equalNoOffset(target.in) and
            this.type == target.type and this.u_var == target.u_var;
    }
    /// Not a great name. Essentially this returns wether the result is independant of what was in `this.out` before
    pub inline fn overwrites(this: @This()) bool {
        // I did this with a switch statement so that you are forced to handle this in case you add a new op
        return switch (this.type) {
            .unary_add => false,
            .unary_subtract => false,
            .unary_multiply => false,
            .unary_divide => false,
            .unary_exp => false,
            .unary_log => false,
            .unary_square => false,
            .unary_sqrt => false,
            .unary_reciprocal => false,
            .unary_max => false,
            .unary_min => false,
            .unary_set => true,
            .unary_random => true,
            .unary_tanh => false,
            .unary_absolute => false,
            .unary_sign => false,
            .binary_add => false,
            .binary_subtract => false,
            .binary_multiply => false,
            .binary_divide => false,
            .binary_max => false,
            .binary_min => false,
            .binary_set => true,
            .expand_add => false,
            .expand_subtract => false,
            .expand_multiply => false,
            .expand_divide => false,
            .expand_max => false,
            .expand_min => false,
            .expand_set => true,
            .reduce_sum => true,
            .reduce_max => true,
            .reduce_avg => true,
            .reduce_min => true,
        };
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Base {s}\n", .{ " " ** offset, text });
        }
        if (this.type.isUnary()) {
            std.debug.print("{s}U {s} ({d} {d} {d} {d}) [{d}, {d}, {d}, {d} = {d}] {} \"{s}\" {d}\n", .{
                " " ** (offset + padding),
                switch (this.type) {
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
            const op_kind: u8 = if (this.type.isBinary()) 'B' else (if (this.type.isExpand()) 'E' else (if (this.type.isReduce()) 'R' else unreachable));
            std.debug.print("{s}{c} {s} ({d} {d} {d} {d}) [{d}, {d}, {d}, {d} = {d}] {} \"{s}\" ({d} {d} {d} {d}) [{d}, {d}, {d}, {d} = {d}] {} \"{s}\"\n", .{
                " " ** (offset + padding),
                op_kind,
                switch (this.type) {
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
pub const Split = struct {
    //
};

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
    split: ?Split,
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
        if (this.split) |split| {
            _ = split;
            unreachable;
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
pub fn alloc(allocator: Allocator, linearized: Linearized, optimization: Optimization) !Pir {
    assert(linearized.op_num > 0);

    const assign: []Assign = try allocator.alloc(Assign, linearized.op_num);
    errdefer allocator.free(assign);

    for (0..linearized.op_num) |op_idx| {
        assign[op_idx] = .{
            .base = .{
                .type = linearized.op[op_idx].type,
                .u_var = linearized.op[op_idx].u_var,
                .out = linearized.op[op_idx].out,
                .in = linearized.op[op_idx].in,
                .out_dim = DimInfo.init(assign[op_idx].base.out.offset),
                .in_dim = DimInfo.init(assign[op_idx].base.in.offset),
                .repeats = 1,
            },
            .inlined = null,
            .split = null,
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
    try pir.optimize(allocator, optimization);
    pir.removeDefault();

    return pir;
}
pub fn free(this: *@This(), allocator: Allocator) void {
    for (0..this.assign_num) |assign_idx| {
        if (this.assign[assign_idx].inlined) |*inlined| {
            allocator.free(inlined.base);
            allocator.free(inlined.in);
            allocator.free(inlined.out);
        }
        if (this.assign[assign_idx].split) |split| {
            _ = split;
            unreachable;
        }
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
fn optimize(this: *@This(), allocator: Allocator, optimization: Optimization) !void {
    if (optimization == .O0) {
        return;
    }

    try opt.inlineOp(allocator, this);
    try opt.parallelize(allocator, this);
    try opt.splitKernel(allocator, this);

    if (optimization == .O1) {
        return;
    }

    try opt.simd(allocator, this);

    if (optimization == .O2) {
        return;
    }

    try opt.memoryLayout(allocator, this);
}
pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
    if (name) |text| {
        std.debug.print("{s}SSA {s}\n", .{ " " ** offset, text });
    } else {
        std.debug.print("{s}SSA\n", .{" " ** offset});
    }
    for (0..this.assign_num) |assign_idx| {
        std.debug.print("{s}[{}] => \n", .{ " " ** offset, assign_idx });
        this.assign[assign_idx].print(padding, offset + padding, null);
    }
}
