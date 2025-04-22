const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Op = @import("../tensor.zig").Op;
const Linearized = @import("../tensor.zig").Linearized;
const Buffer = @import("../tensor.zig").Buffer;
const opt = @import("./optimize.zig");
const Optimization = opt.Optimization;

pub const DimInfo = struct {
    off_out: u32,
    off_in: u32,
    a_stride_out: u32,
    z_stride_out: u32,
    y_stride_out: u32,
    x_stride_out: u32,
    a_stride_in: u32,
    z_stride_in: u32,
    y_stride_in: u32,
    x_stride_in: u32,
    a_reset_out: u32,
    z_reset_out: u32,
    y_reset_out: u32,
    x_reset_out: u32,
    a_reset_in: u32,
    z_reset_in: u32,
    y_reset_in: u32,
    x_reset_in: u32,
    a_wait_out: u32,
    z_wait_out: u32,
    y_wait_out: u32,
    x_wait_out: u32,
    a_wait_in: u32,
    z_wait_in: u32,
    y_wait_in: u32,
    x_wait_in: u32,
    pub fn init(base: []const Base) DimInfo {
        assert(base.len > 0);

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
            assert(base[loop_idx].out.aOffset() >= base[loop_idx - 1].out.aOffset() or base[loop_idx].out.aOffset() == base[0].out.aOffset());
            assert(base[loop_idx].out.zOffset() >= base[loop_idx - 1].out.zOffset() or base[loop_idx].out.zOffset() == base[0].out.zOffset());
            assert(base[loop_idx].out.yOffset() >= base[loop_idx - 1].out.yOffset() or base[loop_idx].out.yOffset() == base[0].out.yOffset());
            assert(base[loop_idx].out.xOffset() >= base[loop_idx - 1].out.xOffset() or base[loop_idx].out.xOffset() == base[0].out.xOffset());
            assert(base[loop_idx].in.aOffset() >= base[loop_idx - 1].in.aOffset() or base[loop_idx].in.aOffset() == base[0].in.aOffset());
            assert(base[loop_idx].in.zOffset() >= base[loop_idx - 1].in.zOffset() or base[loop_idx].in.zOffset() == base[0].in.zOffset());
            assert(base[loop_idx].in.yOffset() >= base[loop_idx - 1].in.yOffset() or base[loop_idx].in.yOffset() == base[0].in.yOffset());
            assert(base[loop_idx].in.xOffset() >= base[loop_idx - 1].in.xOffset() or base[loop_idx].in.xOffset() == base[0].in.xOffset());
        }

        var a_left_out, var z_left_out, var y_left_out, var x_left_out = .{ false, false, false, false };
        var a_left_in, var z_left_in, var y_left_in, var x_left_in = .{ false, false, false, false };
        var a_enter_out, var z_enter_out, var y_enter_out, var x_enter_out = .{ false, false, false, false };
        var a_enter_in, var z_enter_in, var y_enter_in, var x_enter_in = .{ false, false, false, false };

        loop_idx = 1;
        while (loop_idx < base.len) : (loop_idx += 1) {
            if (!a_left_out and base[0].out.aOffset() != base[loop_idx].out.aOffset()) {
                a_left_out = true;
                dim_info.a_wait_out = loop_idx;
                dim_info.a_stride_out = base[loop_idx].out.aOffset() - base[0].out.aOffset();
            } else if (a_left_out and !a_enter_out and base[0].out.aOffset() == base[loop_idx].out.aOffset()) {
                a_enter_out = true;
                dim_info.a_reset_out = loop_idx;
            }
            if (!z_left_out and base[0].out.zOffset() != base[loop_idx].out.zOffset()) {
                z_left_out = true;
                dim_info.z_wait_out = loop_idx;
                dim_info.z_stride_out = base[loop_idx].out.zOffset() - base[0].out.zOffset();
            } else if (z_left_out and !z_enter_out and base[0].out.zOffset() == base[loop_idx].out.zOffset()) {
                z_enter_out = true;
                dim_info.z_reset_out = loop_idx;
            }
            if (!y_left_out and base[0].out.yOffset() != base[loop_idx].out.yOffset()) {
                y_left_out = true;
                dim_info.y_wait_out = loop_idx;
                dim_info.y_stride_out = base[loop_idx].out.yOffset() - base[0].out.yOffset();
            } else if (y_left_out and !y_enter_out and base[0].out.yOffset() == base[loop_idx].out.yOffset()) {
                y_enter_out = true;
                dim_info.y_reset_out = loop_idx;
            }
            if (!x_left_out and base[0].out.xOffset() != base[loop_idx].out.xOffset()) {
                x_left_out = true;
                dim_info.x_wait_out = loop_idx;
                dim_info.x_stride_out = base[loop_idx].out.xOffset() - base[0].out.xOffset();
            } else if (x_left_out and !x_enter_out and base[0].out.xOffset() == base[loop_idx].out.xOffset()) {
                x_enter_out = true;
                dim_info.x_reset_out = loop_idx;
            }
            if (!a_left_in and base[0].in.aOffset() != base[loop_idx].in.aOffset()) {
                a_left_in = true;
                dim_info.a_wait_in = loop_idx;
                dim_info.a_stride_in = base[loop_idx].in.aOffset() - base[0].in.aOffset();
            } else if (a_left_in and !a_enter_in and base[0].in.aOffset() == base[loop_idx].in.aOffset()) {
                a_enter_in = true;
                dim_info.a_reset_in = loop_idx;
            }
            if (!z_left_in and base[0].in.zOffset() != base[loop_idx].in.zOffset()) {
                z_left_in = true;
                dim_info.z_wait_in = loop_idx;
                dim_info.z_stride_in = base[loop_idx].in.zOffset() - base[0].in.zOffset();
            } else if (z_left_in and !z_enter_in and base[0].in.zOffset() == base[loop_idx].in.zOffset()) {
                z_enter_in = true;
                dim_info.z_reset_in = loop_idx;
            }
            if (!y_left_in and base[0].in.yOffset() != base[loop_idx].in.yOffset()) {
                y_left_in = true;
                dim_info.y_wait_in = loop_idx;
                dim_info.y_stride_in = base[loop_idx].in.yOffset() - base[0].in.yOffset();
            } else if (y_left_in and !y_enter_in and base[0].in.yOffset() == base[loop_idx].in.yOffset()) {
                y_enter_in = true;
                dim_info.y_reset_in = loop_idx;
            }
            if (!x_left_in and base[0].in.xOffset() != base[loop_idx].in.xOffset()) {
                x_left_in = true;
                dim_info.x_wait_in = loop_idx;
                dim_info.x_stride_in = base[loop_idx].in.xOffset() - base[0].in.xOffset();
            } else if (x_left_in and !x_enter_in and base[0].in.xOffset() == base[loop_idx].in.xOffset()) {
                x_enter_in = true;
                dim_info.x_reset_in = loop_idx;
            }
        }

        dim_info.off_out = base[0].out.offset;
        dim_info.off_in = base[0].in.offset;

        return dim_info;
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}DimInfo {s}\n", .{ [1]u8{' '} ** offset, text });
        }
        std.debug.print("{s}str => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
            " " ** (offset + padding), //
            this.a_stride_out, this.z_stride_out, this.y_stride_out, this.x_stride_out, //
            this.a_stride_in,  this.z_stride_in,  this.y_stride_in,  this.x_stride_in,
        });
        std.debug.print("{s}res => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
            " " ** (offset + padding), //
            this.a_reset_out, this.z_reset_out, this.y_reset_out, this.x_reset_out, //
            this.a_reset_in,  this.z_reset_in,  this.y_reset_in,  this.x_reset_in,
        });
        std.debug.print("{s}wai => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
            " " ** (offset + padding), //
            this.a_wait_out, this.z_wait_out, this.y_wait_out, this.x_wait_out, //
            this.a_wait_in,  this.z_wait_in,  this.y_wait_in,  this.x_wait_in,
        });
    }
};

/// The basic thing the Assignment does without any funny business
pub const Base = struct {
    out: Buffer,
    in: Buffer,
    type: Op.Type,
    u_var: f32,
    dim_info: DimInfo,
    // $NOTE If you need more dependency layers than this there's some funky stuff going on
    layer_out: u32,
    layer_in: u32,
    pub inline fn layer(this: @This()) u32 {
        return @max(this.layer_out, this.layer_in);
    }
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
        // $NOTE I did this with a switch statement so that you are forced to handle this in case you add a new op
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
            std.debug.print("{s}U {s} ({d} {d} {d} {d}) [{d}, {d}, {d}, {d} = {d}] {} \"{s}\" {d} {}\n", .{
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
                this.layer(),
            });
        } else {
            const op_kind: u8 = if (this.type.isBinary()) 'B' else (if (this.type.isExpand()) 'E' else (if (this.type.isReduce()) 'R' else unreachable));
            std.debug.print("{s}{c} {s} ({d} {d} {d} {d}) [{d}, {d}, {d}, {d} = {d}] {} \"{s}\" ({d} {d} {d} {d}) [{d}, {d}, {d}, {d} = {d}] {} \"{s}\" {}\n", .{
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
                this.layer(),
            });
        }
        this.dim_info.print(padding, padding + offset, null);
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

pub const Ssa = struct {
    assign: []Assign,
    assign_num: u32,
    assign_loop_id: []u32,
    /// Indexed by the id from above
    assign_loop_num: []u32,
    pub fn alloc(allocator: Allocator, linearized: Linearized) !Ssa {
        assert(linearized.op_num > 0);

        var layer_read = std.AutoArrayHashMap(u64, u32).init(allocator);
        var layer_write = std.AutoArrayHashMap(u64, u32).init(allocator);
        errdefer {
            layer_read.deinit();
            layer_write.deinit();
        }
        defer {
            layer_read.deinit();
            layer_write.deinit();
        }

        const assign: []Assign = try allocator.alloc(Assign, linearized.op_num);
        const assign_loop_id: []u32 = try allocator.alloc(u32, linearized.op_num);
        const assign_loop_num: []u32 = try allocator.alloc(u32, linearized.op_num);
        errdefer {
            allocator.free(assign);
            allocator.free(assign_loop_id);
            allocator.free(assign_loop_num);
        }
        @memset(assign_loop_id, 0);
        // $NOTE I don't think the other values should be initialised to prevent strange reads
        assign_loop_num[0] = 1;

        for (0..linearized.op_num) |op_idx| {
            const layer_out: u32 = @max(
                layer_write.get(linearized.op[op_idx].out.id) orelse 0,
                layer_read.get(linearized.op[op_idx].out.id) orelse 0,
            );
            const layer_in: u32 = layer_write.get(linearized.op[op_idx].in.id) orelse 0;
            const layer_idx: u32 = @max(layer_out, layer_in);

            assign[op_idx] = .{
                .base = .{
                    .type = linearized.op[op_idx].type,
                    .u_var = linearized.op[op_idx].u_var,
                    .out = linearized.op[op_idx].out,
                    .in = linearized.op[op_idx].in,
                    .layer_out = layer_out,
                    .layer_in = layer_in,
                    .dim_info = undefined,
                },
                .inlined = null,
                .split = null,
                .simd = null,
                .block = null,
            };

            assign[op_idx].base.dim_info = DimInfo.init(&[1]Base{assign[op_idx].base});

            // $NOTE This overwrites the data if it already existed
            try layer_write.put(assign[op_idx].base.out.id, layer_idx + 1);
            try layer_read.put(assign[op_idx].base.in.id, layer_idx + 1);
        }

        // $NOTE Why was this ever here? Just to group assignments that could be on the same layer?
        // std.mem.sort(Assign, assign, {}, layerLessThan);

        return .{
            .assign = assign,
            .assign_num = @truncate(assign.len),
            .assign_loop_id = assign_loop_id,
            .assign_loop_num = assign_loop_num,
        };
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
        allocator.free(this.assign_loop_id);
        allocator.free(this.assign_loop_num);
    }
    pub fn optimize(this: *@This(), allocator: Allocator, optimization: Optimization) !void {
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
            const id: u32 = this.assign_loop_id[assign_idx];
            std.debug.print("{s}[{}] => Loop id {} num {}\n", .{ " " ** offset, assign_idx, id, this.assign_loop_num[id] });
            this.assign[assign_idx].print(padding, offset + padding, null);
        }
    }
};
