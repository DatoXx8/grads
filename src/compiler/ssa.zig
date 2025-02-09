const std = @import("std");

const assert = std.debug.assert;

const Op = @import("../tensor.zig").Op;
const Linearized = @import("../tensor.zig").Linearized;
const Buffer = @import("../tensor.zig").Buffer;

const Optimization = @import("./optimize.zig").Optimization;

pub const Ssa = struct {
    pub const DimInfo = struct {
        off_out: usize,
        off_in: usize,
        a_idx_out: usize,
        z_idx_out: usize,
        y_idx_out: usize,
        x_idx_out: usize,
        a_idx_in: usize,
        z_idx_in: usize,
        y_idx_in: usize,
        x_idx_in: usize,
        a_stride_out: usize,
        z_stride_out: usize,
        y_stride_out: usize,
        x_stride_out: usize,
        a_stride_in: usize,
        z_stride_in: usize,
        y_stride_in: usize,
        x_stride_in: usize,
        a_reset_out: usize,
        z_reset_out: usize,
        y_reset_out: usize,
        x_reset_out: usize,
        a_reset_in: usize,
        z_reset_in: usize,
        y_reset_in: usize,
        x_reset_in: usize,
        a_wait_out: usize,
        z_wait_out: usize,
        y_wait_out: usize,
        x_wait_out: usize,
        a_wait_in: usize,
        z_wait_in: usize,
        y_wait_in: usize,
        x_wait_in: usize,
        pub fn init(op: []Op, loop_num: usize) DimInfo {
            assert(loop_num > 0);
            assert(op.len == loop_num);

            var dim_info: DimInfo = .{
                .off_out = 0,
                .off_in = 0,
                .a_idx_out = 0,
                .z_idx_out = 0,
                .y_idx_out = 0,
                .x_idx_out = 0,
                .a_idx_in = 0,
                .z_idx_in = 0,
                .y_idx_in = 0,
                .x_idx_in = 0,
                .a_stride_out = 0,
                .z_stride_out = 0,
                .y_stride_out = 0,
                .x_stride_out = 0,
                .a_stride_in = 0,
                .z_stride_in = 0,
                .y_stride_in = 0,
                .x_stride_in = 0,
                .a_reset_out = loop_num,
                .z_reset_out = loop_num,
                .y_reset_out = loop_num,
                .x_reset_out = loop_num,
                .a_reset_in = loop_num,
                .z_reset_in = loop_num,
                .y_reset_in = loop_num,
                .x_reset_in = loop_num,
                .a_wait_out = 1,
                .z_wait_out = 1,
                .y_wait_out = 1,
                .x_wait_out = 1,
                .a_wait_in = 1,
                .z_wait_in = 1,
                .y_wait_in = 1,
                .x_wait_in = 1,
            };

            for (1..loop_num) |loop_idx| {
                if (op[dim_info.a_idx_out].out.aOffset() > op[loop_idx].out.aOffset()) {
                    dim_info.a_idx_out = loop_idx;
                }
                if (op[dim_info.z_idx_out].out.zOffset() > op[loop_idx].out.zOffset()) {
                    dim_info.z_idx_out = loop_idx;
                }
                if (op[dim_info.y_idx_out].out.yOffset() > op[loop_idx].out.yOffset()) {
                    dim_info.y_idx_out = loop_idx;
                }
                if (op[dim_info.x_idx_out].out.xOffset() > op[loop_idx].out.xOffset()) {
                    dim_info.x_idx_out = loop_idx;
                }
                if (op[dim_info.a_idx_in].in.aOffset() > op[loop_idx].in.aOffset()) {
                    dim_info.a_idx_in = loop_idx;
                }
                if (op[dim_info.z_idx_in].in.zOffset() > op[loop_idx].in.zOffset()) {
                    dim_info.z_idx_in = loop_idx;
                }
                if (op[dim_info.y_idx_in].in.yOffset() > op[loop_idx].in.yOffset()) {
                    dim_info.y_idx_in = loop_idx;
                }
                if (op[dim_info.x_idx_in].in.xOffset() > op[loop_idx].in.xOffset()) {
                    dim_info.x_idx_in = loop_idx;
                }
            }

            var a_left_out, var z_left_out, var y_left_out, var x_left_out = .{ false, false, false, false };
            var a_left_in, var z_left_in, var y_left_in, var x_left_in = .{ false, false, false, false };
            var a_enter_out, var z_enter_out, var y_enter_out, var x_enter_out = .{ false, false, false, false };
            var a_enter_in, var z_enter_in, var y_enter_in, var x_enter_in = .{ false, false, false, false };

            for (1..loop_num) |loop_idx| {
                if (!a_left_out and op[dim_info.a_idx_out].out.aOffset() !=
                    op[(loop_idx + dim_info.a_idx_out) % loop_num].out.aOffset())
                {
                    a_left_out = true;
                    dim_info.a_wait_out = loop_idx;
                    dim_info.a_stride_out = op[(loop_idx + dim_info.a_idx_out) % loop_num].out.aOffset() -
                        op[dim_info.a_idx_out].out.aOffset();
                } else if (a_left_out and !a_enter_out and op[dim_info.a_idx_out].out.aOffset() ==
                    op[(loop_idx + dim_info.a_idx_out) % loop_num].out.aOffset())
                {
                    a_enter_out = true;
                    dim_info.a_reset_out = loop_idx;
                }
                if (!z_left_out and op[dim_info.z_idx_out].out.zOffset() !=
                    op[(loop_idx + dim_info.z_idx_out) % loop_num].out.zOffset())
                {
                    z_left_out = true;
                    dim_info.z_wait_out = loop_idx;
                    dim_info.z_stride_out = op[(loop_idx + dim_info.z_idx_out) % loop_num].out.zOffset() -
                        op[dim_info.z_idx_out].out.zOffset();
                } else if (z_left_out and !z_enter_out and op[dim_info.z_idx_out].out.zOffset() ==
                    op[(loop_idx + dim_info.z_idx_out) % loop_num].out.zOffset())
                {
                    z_enter_out = true;
                    dim_info.z_reset_out = loop_idx;
                }
                if (!y_left_out and op[dim_info.y_idx_out].out.yOffset() !=
                    op[(loop_idx + dim_info.y_idx_out) % loop_num].out.yOffset())
                {
                    y_left_out = true;
                    dim_info.y_wait_out = loop_idx;
                    dim_info.y_stride_out = op[(loop_idx + dim_info.y_idx_out) % loop_num].out.yOffset() -
                        op[dim_info.y_idx_out].out.yOffset();
                } else if (y_left_out and !y_enter_out and op[dim_info.y_idx_out].out.yOffset() ==
                    op[(loop_idx + dim_info.y_idx_out) % loop_num].out.yOffset())
                {
                    y_enter_out = true;
                    dim_info.y_reset_out = loop_idx;
                }
                if (!x_left_out and op[dim_info.x_idx_out].out.xOffset() !=
                    op[(loop_idx + dim_info.x_idx_out) % loop_num].out.xOffset())
                {
                    x_left_out = true;
                    dim_info.x_wait_out = loop_idx;
                    dim_info.x_stride_out = op[(loop_idx + dim_info.x_idx_out) % loop_num].out.xOffset() -
                        op[dim_info.x_idx_out].out.xOffset();
                } else if (x_left_out and !x_enter_out and op[dim_info.x_idx_out].out.xOffset() ==
                    op[(loop_idx + dim_info.x_idx_out) % loop_num].out.xOffset())
                {
                    x_enter_out = true;
                    dim_info.x_reset_out = loop_idx;
                }
                if (!a_left_in and op[dim_info.a_idx_in].in.aOffset() !=
                    op[(loop_idx + dim_info.a_idx_in) % loop_num].in.aOffset())
                {
                    a_left_in = true;
                    dim_info.a_wait_in = loop_idx;
                    dim_info.a_stride_in = op[(loop_idx + dim_info.a_idx_in) % loop_num].in.aOffset() -
                        op[dim_info.a_idx_in].in.aOffset();
                } else if (a_left_in and !a_enter_in and op[dim_info.a_idx_in].in.aOffset() ==
                    op[(loop_idx + dim_info.a_idx_in) % loop_num].in.aOffset())
                {
                    a_enter_in = true;
                    dim_info.a_reset_in = loop_idx;
                }
                if (!z_left_in and op[dim_info.z_idx_in].in.zOffset() !=
                    op[(loop_idx + dim_info.z_idx_in) % loop_num].in.zOffset())
                {
                    z_left_in = true;
                    dim_info.z_wait_in = loop_idx;
                    dim_info.z_stride_in = op[(loop_idx + dim_info.z_idx_in) % loop_num].in.zOffset() -
                        op[dim_info.z_idx_in].in.zOffset();
                } else if (z_left_in and !z_enter_in and op[dim_info.z_idx_in].in.zOffset() ==
                    op[(loop_idx + dim_info.z_idx_in) % loop_num].in.zOffset())
                {
                    z_enter_in = true;
                    dim_info.z_reset_in = loop_idx;
                }
                if (!y_left_in and op[dim_info.y_idx_in].in.yOffset() !=
                    op[(loop_idx + dim_info.y_idx_in) % loop_num].in.yOffset())
                {
                    y_left_in = true;
                    dim_info.y_wait_in = loop_idx;
                    dim_info.y_stride_in = op[(loop_idx + dim_info.y_idx_in) % loop_num].in.yOffset() -
                        op[dim_info.y_idx_in].in.yOffset();
                } else if (y_left_in and !y_enter_in and op[dim_info.y_idx_in].in.yOffset() ==
                    op[(loop_idx + dim_info.y_idx_in) % loop_num].in.yOffset())
                {
                    y_enter_in = true;
                    dim_info.y_reset_in = loop_idx;
                }
                if (!x_left_in and op[dim_info.x_idx_in].in.xOffset() !=
                    op[(loop_idx + dim_info.x_idx_in) % loop_num].in.xOffset())
                {
                    x_left_in = true;
                    dim_info.x_wait_in = loop_idx;
                    dim_info.x_stride_in = op[(loop_idx + dim_info.x_idx_in) % loop_num].in.xOffset() -
                        op[dim_info.x_idx_in].in.xOffset();
                } else if (x_left_in and !x_enter_in and op[dim_info.x_idx_in].in.xOffset() ==
                    op[(loop_idx + dim_info.x_idx_in) % loop_num].in.xOffset())
                {
                    x_enter_in = true;
                    dim_info.x_reset_in = loop_idx;
                }
            }

            dim_info.off_out = op[dim_info.a_idx_out].out.aOffset() * op[dim_info.a_idx_out].out.a_stride + //
                op[dim_info.z_idx_out].out.zOffset() * op[dim_info.z_idx_out].out.z_stride + //
                op[dim_info.y_idx_out].out.yOffset() * op[dim_info.y_idx_out].out.y_stride + //
                op[dim_info.x_idx_out].out.xOffset() * op[dim_info.x_idx_out].out.x_stride;
            dim_info.off_in = op[dim_info.a_idx_in].in.aOffset() * op[dim_info.a_idx_in].in.a_stride + //
                op[dim_info.z_idx_in].in.zOffset() * op[dim_info.z_idx_in].in.z_stride + //
                op[dim_info.y_idx_in].in.yOffset() * op[dim_info.y_idx_in].in.y_stride + //
                op[dim_info.x_idx_in].in.xOffset() * op[dim_info.x_idx_in].in.x_stride;

            return dim_info;
        }
        pub fn print(this: *@This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
            if (name) |text| {
                std.debug.print("{s}DimInfo {s}\n", .{ [1]u8{' '} ** offset, text });
            }
            std.debug.print("{s}str => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
                " " ** (offset + padding), //
                this.a_stride_out, this.z_stride_out, this.y_stride_out, this.x_stride_out, //
                this.a_stride_out, this.z_stride_out, this.y_stride_out, this.x_stride_out,
            });
            std.debug.print("{s}res => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
                " " ** (offset + padding), //
                this.a_reset_out, this.z_reset_out, this.y_reset_out, this.x_reset_out, //
                this.a_reset_out, this.z_reset_out, this.y_reset_out, this.x_reset_out,
            });
            std.debug.print("{s}wai => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
                " " ** (offset + padding), //
                this.a_wait_out, this.z_wait_out, this.y_wait_out, this.x_wait_out, //
                this.a_wait_out, this.z_wait_out, this.y_wait_out, this.x_wait_out,
            });
            std.debug.print("{s}idx => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
                " " ** (offset + padding), //
                this.a_idx_out, this.z_idx_out, this.y_idx_out, this.x_idx_out, //
                this.a_idx_out, this.z_idx_out, this.y_idx_out, this.x_idx_out,
            });
        }
    };
    pub const Assignment = struct {
        pub const Base = struct {
            type: Op.Type,
            out: Buffer,
            in: Buffer,
            u_var: f32,
            layer_out: usize,
            layer_in: usize,
            dim_info: DimInfo,
            pub inline fn layer(this: *const @This()) usize {
                return @max(this.layer_out, this.layer_in);
            }
            /// Returns wether the `this` assignment overwrites every value in `target` with values not dependant on those in `target`
            pub inline fn overwrites(this: *const @This(), target: *const @This()) bool {
                const a_low_this: usize = this.out.a_size + this.out.a_offset;
                const z_low_this: usize = this.out.z_size + this.out.z_offset;
                const y_low_this: usize = this.out.y_size + this.out.y_offset;
                const x_low_this: usize = this.out.x_size + this.out.x_offset;
                const a_low_target: usize = target.out.a_size + target.out.a_offset;
                const z_low_target: usize = target.out.z_size + target.out.z_offset;
                const y_low_target: usize = target.out.y_size + target.out.y_offset;
                const x_low_target: usize = target.out.x_size + target.out.x_offset;
                return this.out.name_offset == target.out.name_offset and this.type.isStandalone() and
                    a_low_this == a_low_target and z_low_this == z_low_target and
                    y_low_this == y_low_target and x_low_this == x_low_target;
            }
            pub fn print(this: *const @This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
                // TODO: Also print the dim info
                if (name) |text| {
                    std.debug.print("{s}Base {s}\n", .{ " " ** (offset), text });
                }
                if (this.type.isUnary()) {
                    std.debug.print("{s}{} U {s} ({d} {d} {d} {d}) [{d} {d} {d} {d} = {d}] \"{s}\" {} {d}\n", .{
                        " " ** (offset + padding),
                        this.layer(),
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
                        this.out.name(),
                        this.layer_out,
                        this.u_var,
                    });
                } else {
                    const op_kind: u8 = if (this.type.isBinary()) 'B' else (if (this.type.isLinary()) 'L' else 'R');
                    std.debug.print("{s}{} {c} {s} ({d} {d} {d} {d}) [{d} {d} {d} {d} = {d}] {} \"{s}\" ({d} {d} {d} {d}) [{d} {d} {d} {d} = {d}] {} \"{s}\"\n", .{
                        " " ** (offset + padding),
                        this.layer(),
                        op_kind,
                        switch (this.type) {
                            .binary_add => "add",
                            .binary_subtract => "sub",
                            .binary_multiply => "mul",
                            .binary_divide => "div",
                            .binary_max => "max",
                            .binary_min => "min",
                            .binary_set => "set",
                            .linary_add => "add",
                            .linary_subtract => "sub",
                            .linary_multiply => "mul",
                            .linary_divide => "div",
                            .linary_max => "max",
                            .linary_min => "min",
                            .linary_set => "set",
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
                        this.layer_out,
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
                        this.layer_in,
                        this.in.name(),
                    });
                }
            }
        };
        pub const Inlined = struct {
            pub const Type = enum(u8) { in, out };
            type: []Type,
            base: []Base,
            pub fn print(this: *const @This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
                // TODO: Also print the dim info
                if (name) |text| {
                    std.debug.print("{s}Inlined {s}\n", .{ " " ** (offset), text });
                }
                for (0..this.base.len) |inlined_idx| {
                    const inlined_type: u8 = switch (this.type[inlined_idx]) {
                        .out => 'o',
                        .in => 'i',
                    };
                    std.debug.print("{s}({}) => {c} ", .{ " " ** (offset + padding), inlined_idx, inlined_type });
                    this.base[inlined_idx].print(0, 0, null);
                }
            }
        };
        pub const Split = struct {
            splittable: bool,
        };
        // TODO: Simd is not really an accurate name, it utilizes simd by block forming
        //  but I am not really sure how to name that
        pub const Simd = struct {
            a_block: usize,
            z_block: usize,
            y_block: usize,
            x_block: usize,
        };
        pub const Memory = struct {
            // TODO: Don't even know what to put here yet
            //  I guess it's supposed to be like an advanced mix of SIMD block forms with Split
        };
        base: Base,
        inlined: ?Inlined,
        split: ?Split,
        simd: ?Simd,
        memory: ?Memory,
        pub fn print(this: *const @This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
            // TODO: Also print the dim info and optimizations
            if (name) |text| {
                std.debug.print("{s}Assignment {s}\n", .{ " " ** (offset), text });
            }
            this.base.print(padding, padding + offset, null);
            if (this.inlined) |inlined| {
                std.debug.print("{s}Inlined\n", .{" " ** (padding + offset)});
                inlined.print(padding, padding + offset, null);
            }
        }
    };
    assignment: []Assignment,
    assignment_num: usize,
    pub fn alloc(allocator: std.mem.Allocator, linearized: Linearized) !Ssa {
        // Don't think hashmaps are avoidable sadly :^(
        var assignment_layer_write = std.AutoHashMap(usize, usize).init(allocator);
        defer assignment_layer_write.deinit();

        var assignment_layer_read = std.AutoHashMap(usize, usize).init(allocator);
        defer assignment_layer_read.deinit();

        var assignment: []Assignment = try allocator.alloc(Assignment, linearized.op_num);

        for (0..linearized.op_num) |op_idx| {
            const layer_out: usize = @max(
                assignment_layer_write.get(linearized.op[op_idx].out.name_offset) orelse 0,
                assignment_layer_read.get(linearized.op[op_idx].out.name_offset) orelse 0,
            );
            const layer_in: usize = assignment_layer_write.get(linearized.op[op_idx].in.name_offset) orelse 0;
            const layer_idx: usize = @max(layer_out, layer_in);

            assignment[op_idx] = .{
                .base = .{
                    .type = linearized.op[op_idx].type,
                    .u_var = linearized.op[op_idx].u_var,
                    .out = linearized.op[op_idx].out,
                    .in = linearized.op[op_idx].in,
                    .layer_out = layer_out,
                    .layer_in = layer_in,
                    .dim_info = DimInfo.init(linearized.op[op_idx .. op_idx + 1], 1),
                },
                .inlined = null,
                .split = null,
                .simd = null,
                .memory = null,
            };

            // This overwrites the data if it already existed
            try assignment_layer_write.put(assignment[op_idx].base.out.name_offset, layer_idx + 1);
            try assignment_layer_read.put(assignment[op_idx].base.in.name_offset, layer_idx + 1);

            if (op_idx == 0) {
                assert(assignment[0].base.layer() == 0);
            } else {
                assert(assignment[op_idx].base.layer() >= assignment[op_idx - 1].base.layer());
            }
        }

        return .{
            .assignment = assignment,
            .assignment_num = assignment.len,
        };
    }
    pub fn free(this: *@This(), allocator: std.mem.Allocator) void {
        for (this.assignment) |*assignment| {
            if (assignment.inlined) |*inlined| {
                allocator.free(inlined.base);
                allocator.free(inlined.type);
            }
        }
        allocator.free(this.assignment);
    }
    pub fn optimize(this: *@This(), allocator: std.mem.Allocator, optimization: Optimization) !void {
        if (optimization == .O0) {
            return;
        }
        try optimization.inlineOp(allocator, this);
        try optimization.parallelize(allocator, this);
        try optimization.splitKernel(allocator, this);
        if (optimization == .O1) {
            return;
        }
        try optimization.simd(allocator, this);
        if (optimization == .O2) {
            return;
        }
        try optimization.memoryLayout(allocator, this);
    }
    pub fn print(this: *const @This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}SSA {s}\n", .{ [1]u8{' '} ** offset, text });
        } else {
            std.debug.print("{s}SSA\n", .{[1]u8{' '} ** offset});
        }
        for (0..this.assignment_num) |assignment_idx| {
            std.debug.print("{s}[{}] => ", .{ [1]u8{' '} ** (offset + padding), assignment_idx });
            this.assignment[assignment_idx].print(0, 0, null);
        }
    }
    pub fn debug(this: *const @This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}SSA {s}\n", .{ [1]u8{' '} ** offset, text });
        } else {
            std.debug.print("{s}SSA\n", .{[1]u8{' '} ** offset});
        }
        for (0..this.assignment_num) |assignment_idx| {
            std.debug.print("{s}[{}] => \n", .{ [1]u8{' '} ** (offset + padding), assignment_idx });
            this.assignment[assignment_idx].print(padding, offset + padding, null);
        }
    }
};
