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
        pub fn print(this: *@This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
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
};
