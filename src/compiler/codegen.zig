// TODO: These levels
// Optimization levels
// O1 - inline, split, merge kernels
// O2 - fuse along all axis
// O3 - memory optimizer (SLOW!!!)

// Optimiuation levels
// O0 - none
// O1 - inline, split, merge kernels
// O2 - fuse along all axis
// O3 - memory optimizer (SLOW!!!)

const std = @import("std");

const Pir = @import("./pir.zig").Pir;
const DimInfo = @import("./pir.zig").DimInfo;

const assert = std.debug.assert;

const buffer_name_size = @import("../tensor.zig").buffer_name_size;
const Op = @import("../tensor.zig").Op;

const ClMem = @import("../runtimes/cl.zig").ClMem;

const Args = @import("./kernel.zig").Args;
const kernel_name = @import("../runtimes/cl.zig").kernel_name;
const kernel_name_c = @import("../runtimes/cl.zig").kernel_name_c;

pub const Optimization = enum(u8) {
    O0,
    O1,
    O2,
    O3,
};

/// Expand buffer if necessary and set new bytes to 0
fn capacityEnsure(allocator: anytype, source: []u8, offset: usize, padding: usize) ![]u8 {
    if (source.len - offset < padding) {
        const len_old: usize = source.len;
        const new: []u8 = try allocator.realloc(source, len_old * 2);
        @memset(new[len_old..], 0);
        return new;
    } else {
        return source;
    }
}

/// Write format string to buffer and ensure there is at least `padding` bytes left
fn writeBuffer(allocator: anytype, source: *[]u8, offset: *usize, padding: usize, comptime fmt: []const u8, args: anytype) !void {
    // TODO: Validate that there is enough space for this and expand if there isn't
    offset.* += (try std.fmt.bufPrint(source.*[offset.*..], fmt, args)).len;
    source.* = try capacityEnsure(allocator, source.*, offset.*, padding);
}

/// Generate computation for per-op indices
fn generateIndex(allocator: anytype, source: *[]u8, offset: *usize, padding: usize, pir: Pir, repeat_idx: usize) !void {
    const op = pir.op;
    const dim_info = pir.dim_info;
    for (0..pir.op_num) |op_idx| {
        const offset_out: usize = dim_info[op_idx].off_a_out * op[op_idx].out.a_stride +
            dim_info[op_idx].off_z_out * op[op_idx].out.z_stride +
            dim_info[op_idx].off_y_out * op[op_idx].out.y_stride +
            dim_info[op_idx].off_x_out * op[op_idx].out.x_stride;
        try writeBuffer(
            allocator,
            source,
            offset,
            padding,
            "int {s}_{}_{} = (id+{})%{}/{}*{}+(id+{})%{}/{}*{}+(id+{})%{}/{}*{}+(id+{})%{}/{}*{}+{};\n",
            .{
                op[op_idx].out.name, repeat_idx, op_idx, //
                dim_info[op_idx].idx_a_out, dim_info[op_idx].res_a_out, //
                dim_info[op_idx].wai_a_out, dim_info[op_idx].str_a_out * op[op_idx].out.a_stride,
                dim_info[op_idx].idx_z_out, dim_info[op_idx].res_z_out,
                dim_info[op_idx].wai_z_out, dim_info[op_idx].str_z_out * op[op_idx].out.z_stride,
                dim_info[op_idx].idx_y_out, dim_info[op_idx].res_y_out,
                dim_info[op_idx].wai_y_out, dim_info[op_idx].str_y_out * op[op_idx].out.y_stride,
                dim_info[op_idx].idx_x_out, dim_info[op_idx].res_x_out,
                dim_info[op_idx].wai_x_out, dim_info[op_idx].str_x_out * op[op_idx].out.x_stride,
                offset_out,
            },
        );
        if (!op[op_idx].isUnary()) {
            const offset_in: usize = dim_info[op_idx].off_a_in * op[op_idx].in.a_stride +
                dim_info[op_idx].off_z_in * op[op_idx].in.z_stride +
                dim_info[op_idx].off_y_in * op[op_idx].in.y_stride +
                dim_info[op_idx].off_x_in * op[op_idx].in.x_stride;
            try writeBuffer(
                allocator,
                source,
                offset,
                padding,
                "int {s}_{}_{} = (id+{})%{}/{}*{}+(id+{})%{}/{}*{}+(id+{})%{}/{}*{}+(id+{})%{}/{}*{}+{};\n",
                .{
                    op[op_idx].in.name, repeat_idx, op_idx, //
                    dim_info[op_idx].idx_a_in, dim_info[op_idx].res_a_in, //
                    dim_info[op_idx].wai_a_in, dim_info[op_idx].str_a_in * op[op_idx].in.a_stride,
                    dim_info[op_idx].idx_z_in, dim_info[op_idx].res_z_in,
                    dim_info[op_idx].wai_z_in, dim_info[op_idx].str_z_in * op[op_idx].in.z_stride,
                    dim_info[op_idx].idx_y_in, dim_info[op_idx].res_y_in,
                    dim_info[op_idx].wai_y_in, dim_info[op_idx].str_y_in * op[op_idx].in.y_stride,
                    dim_info[op_idx].idx_x_in, dim_info[op_idx].res_x_in,
                    dim_info[op_idx].wai_x_in, dim_info[op_idx].str_x_in * op[op_idx].in.x_stride,
                    offset_in,
                },
            );
        }
    }
}

/// Generate a line of OpenCL code setting up the op. Like setting to -INFINITY for reduce_max
fn generateOpHeader(
    allocator: anytype,
    source: *[]u8,
    offset: *usize,
    padding: usize,
    op: Op,
    repeat_idx: usize,
    op_idx: usize,
) !void {
    switch (op.type) {
        .reduce_sum => {
            try writeBuffer(allocator, source, offset, padding, "{s}[{s}_{}_{}] = 0;\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
            });
        },
        .reduce_avg => {
            try writeBuffer(allocator, source, offset, padding, "{s}[{s}_{}_{}] = 0;\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
            });
        },
        .reduce_max => {
            try writeBuffer(allocator, source, offset, padding, "{s}[{s}_{}_{}] = -INFINITY;\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
            });
        },
        .reduce_min => {
            try writeBuffer(allocator, source, offset, padding, "{s}[{s}_{}_{}] = INFINITY;\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
            });
        },
        else => {},
    }
}

/// Do post op calculations. Currently only used for dividing by the size of the `in` buffer for reduce_avg
fn generateOpFooter(
    allocator: anytype,
    source: *[]u8,
    offset: *usize,
    padding: usize,
    op: Op,
    repeat_idx: usize,
    op_idx: usize,
) !void {
    switch (op.type) {
        .reduce_avg => {
            try writeBuffer(allocator, source, offset, padding, "{s}[{s}_{}_{}] /= {};\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                op.in.a_size * op.in.z_size * op.in.y_size * op.in.x_size,
            });
        },
        else => {},
    }
}

fn generateOpPrefix(
    allocator: anytype,
    source: *[]u8,
    offset: *usize,
    padding: usize,
    op: Op,
    inline_info: Pir.Inline,
    repeat_idx: usize,
    op_idx: usize,
    offset_out: usize,
) !void {
    switch (op.type) {
        .unary_add => {
            try writeBuffer(allocator, source, offset, padding, "({d} + ", .{
                op.u_var,
            });
        },
        .unary_subtract => {
            try writeBuffer(allocator, source, offset, padding, "({d} - ", .{
                op.u_var,
            });
        },
        .unary_multiply => {
            try writeBuffer(allocator, source, offset, padding, "({d} * ", .{
                op.u_var,
            });
        },
        .unary_divide => {
            try writeBuffer(allocator, source, offset, padding, "(", .{});
        },
        .unary_exp => {
            try writeBuffer(allocator, source, offset, padding, "exp(", .{});
        },
        .unary_log => {
            try writeBuffer(allocator, source, offset, padding, "log(", .{});
        },
        .unary_square => {
            // TODO: Refactor this to be a multiplication instead of an exponentiation algorithm
            try writeBuffer(allocator, source, offset, padding, "pow(", .{});
        },
        .unary_sqrt => {
            try writeBuffer(allocator, source, offset, padding, "sqrt(", .{});
        },
        .unary_reciprocal => {
            try writeBuffer(allocator, source, offset, padding, "1 / (", .{});
        },
        .unary_max => {
            try writeBuffer(allocator, source, offset, padding, "fmax((float){d}, ", .{
                op.u_var,
            });
        },
        .unary_min => {
            try writeBuffer(allocator, source, offset, padding, "fmin((float){d}, ", .{
                op.u_var,
            });
        },
        .unary_set => {
            try writeBuffer(allocator, source, offset, padding, "(", .{});
        },
        .unary_random => {
            try writeBuffer(allocator, source, offset, padding, "(", .{});
            unreachable;
        },
        .unary_tanh => {
            try writeBuffer(allocator, source, offset, padding, "tanh(", .{});
        },
        .unary_absolute => {
            try writeBuffer(allocator, source, offset, padding, "fabs(", .{});
        },
        .unary_sign => {
            try writeBuffer(allocator, source, offset, padding, "(", .{});
            unreachable;
        },
        .binary_add => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, "({s}[{s}_{}_{} + {}] + ", .{
                    op.out.name,
                    op.out.name,
                    repeat_idx,
                    op_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, padding, "(", .{});
            }
        },
        .binary_subtract => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, "({s}[{s}_{}_{} + {}] - ", .{
                    op.out.name,
                    op.out.name,
                    repeat_idx,
                    op_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, padding, "(", .{});
            }
        },
        .binary_multiply => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, "({s}[{s}_{}_{} + {}] * ", .{
                    op.out.name,
                    op.out.name,
                    repeat_idx,
                    op_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, padding, "(", .{});
            }
        },
        .binary_divide => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, "({s}[{s}_{}_{} + {}] / ", .{
                    op.out.name,
                    op.out.name,
                    repeat_idx,
                    op_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, padding, "(", .{});
            }
        },
        .binary_max => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, "fmax({s}[{s}_{}_{} + {}], ", .{
                    op.out.name,
                    op.out.name,
                    repeat_idx,
                    op_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, padding, "fmax(", .{});
            }
        },
        .binary_min => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, "fmin({s}[{s}_{}_{} + {}], ", .{
                    op.out.name,
                    op.out.name,
                    repeat_idx,
                    op_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, padding, "fmin(", .{});
            }
        },
        .binary_set => {
            try writeBuffer(allocator, source, offset, padding, "(", .{});
        },
        .linary_add => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, "({s}[{s}_{}_{}] + ", .{
                    op.out.name,
                    op.out.name,
                    repeat_idx,
                    op_idx,
                });
            } else {
                try writeBuffer(allocator, source, offset, padding, "(", .{});
            }
        },
        .linary_subtract => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, "({s}[{s}_{}_{}] - ", .{
                    op.out.name,
                    op.out.name,
                    repeat_idx,
                    op_idx,
                });
            } else {
                try writeBuffer(allocator, source, offset, padding, "(", .{});
            }
        },
        .linary_multiply => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, "({s}[{s}_{}_{}] * ", .{
                    op.out.name,
                    op.out.name,
                    repeat_idx,
                    op_idx,
                });
            } else {
                try writeBuffer(allocator, source, offset, padding, "(", .{});
            }
        },
        .linary_divide => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, "({s}[{s}_{}_{}] / ", .{
                    op.out.name,
                    op.out.name,
                    repeat_idx,
                    op_idx,
                });
            } else {
                try writeBuffer(allocator, source, offset, padding, "(", .{});
            }
        },
        .linary_max => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, "fmax({s}[{s}_{}_{}], ", .{
                    op.out.name,
                    op.out.name,
                    repeat_idx,
                    op_idx,
                });
            } else {
                try writeBuffer(allocator, source, offset, padding, "fmax(", .{});
            }
        },
        .linary_min => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, "fmin({s}[{s}_{}_{}], ", .{
                    op.out.name,
                    op.out.name,
                    repeat_idx,
                    op_idx,
                });
            } else {
                try writeBuffer(allocator, source, offset, padding, "fmin(", .{});
            }
        },
        .linary_set => {
            try writeBuffer(allocator, source, offset, padding, "(", .{});
        },
        .reduce_sum => {
            try writeBuffer(allocator, source, offset, padding, "({s}[{s}_{}_{}] + ", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
            });
        },
        .reduce_avg => {
            try writeBuffer(allocator, source, offset, padding, "({s}[{s}_{}_{}] + ", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
            });
        },
        .reduce_max => {
            try writeBuffer(allocator, source, offset, padding, "fmax({s}[{s}_{}_{}], ", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
            });
        },
        .reduce_min => {
            try writeBuffer(allocator, source, offset, padding, "fmin({s}[{s}_{}_{}], ", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
            });
        },
    }
}

/// Generate a line of OpenCL code computing one entry of `op.out`
fn generateOpPostfix(
    allocator: anytype,
    source: *[]u8,
    offset: *usize,
    padding: usize,
    op: Op,
    inline_info: Pir.Inline,
    repeat_idx: usize,
    op_idx: usize,
    offset_in: usize,
) !void {
    switch (op.type) {
        .unary_add => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
        .unary_subtract => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
        .unary_multiply => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
        .unary_divide => {
            try writeBuffer(allocator, source, offset, padding, "/ {})", .{op.u_var});
        },
        .unary_exp => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
        .unary_log => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
        .unary_square => {
            // TODO: Refactor this to be a multiplication instead of an exponentiation algorithm where that is advantageous
            try writeBuffer(allocator, source, offset, padding, ", 2)", .{});
        },
        .unary_sqrt => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
        .unary_reciprocal => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
        .unary_max => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
        .unary_min => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
        .unary_set => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
        .unary_random => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
            unreachable;
        },
        .unary_tanh => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
        .unary_absolute => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
        .unary_sign => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
            unreachable;
        },
        .binary_add => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, padding, " + {s}[{s}_{}_{} + {}])", .{
                    op.in.name,
                    op.in.name,
                    repeat_idx,
                    op_idx,
                    offset_in,
                });
            }
        },
        .binary_subtract => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, padding, " - {s}[{s}_{}_{} + {}])", .{
                    op.in.name,
                    op.in.name,
                    repeat_idx,
                    op_idx,
                    offset_in,
                });
            }
        },
        .binary_multiply => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, padding, " * {s}[{s}_{}_{} + {}])", .{
                    op.in.name,
                    op.in.name,
                    repeat_idx,
                    op_idx,
                    offset_in,
                });
            }
        },
        .binary_divide => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, padding, " / {s}[{s}_{}_{} + {}])", .{
                    op.in.name,
                    op.in.name,
                    repeat_idx,
                    op_idx,
                    offset_in,
                });
            }
        },
        .binary_max => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, padding, ", {s}[{s}_{}_{} + {}])", .{
                    op.in.name,
                    op.in.name,
                    repeat_idx,
                    op_idx,
                    offset_in,
                });
            }
        },
        .binary_min => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, padding, ", {s}[{s}_{}_{} + {}])", .{
                    op.in.name,
                    op.in.name,
                    repeat_idx,
                    op_idx,
                    offset_in,
                });
            }
        },
        .binary_set => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
        .linary_add => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, padding, " + {s}[{s}_{}_{}])", .{
                    op.in.name,
                    op.in.name,
                    repeat_idx,
                    op_idx,
                });
            }
        },
        .linary_subtract => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, padding, " - {s}[{s}_{}_{}])", .{
                    op.in.name,
                    op.in.name,
                    repeat_idx,
                    op_idx,
                });
            }
        },
        .linary_multiply => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, padding, " * {s}[{s}_{}_{}])", .{
                    op.in.name,
                    op.in.name,
                    repeat_idx,
                    op_idx,
                });
            }
        },
        .linary_divide => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, padding, " / {s}[{s}_{}_{}])", .{
                    op.in.name,
                    op.in.name,
                    repeat_idx,
                    op_idx,
                });
            }
        },
        .linary_max => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, padding, ", {s}[{s}_{}_{}])", .{
                    op.in.name,
                    op.in.name,
                    repeat_idx,
                    op_idx,
                });
            }
        },
        .linary_min => {
            if (inline_info == .none or inline_info == .in) {
                try writeBuffer(allocator, source, offset, padding, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, padding, ", {s}[{s}_{}_{}])", .{
                    op.in.name,
                    op.in.name,
                    repeat_idx,
                    op_idx,
                });
            }
        },
        .linary_set => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
        .reduce_sum => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
        .reduce_avg => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
        .reduce_max => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
        .reduce_min => {
            try writeBuffer(allocator, source, offset, padding, ")", .{});
        },
    }
}

fn generateOpAssign(
    allocator: anytype,
    source: *[]u8,
    offset: *usize,
    padding: usize,
    op: Op,
    repeat_idx: usize,
    op_idx: usize,
    offset_out: usize,
) !void {
    if (op.isReduce()) {
        assert(offset_out == 0);
    }
    try writeBuffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = ", .{
        op.out.name,
        op.out.name,
        repeat_idx,
        op_idx,
        offset_out,
    });
}

fn generateOpBody(
    allocator: anytype,
    source: *[]u8,
    offset: *usize,
    padding: usize,
    op_slice: []Op,
    inline_info_slice: []Pir.Inline,
    repeat_idx: usize,
    kernel_op_idx: usize,
    a: usize,
    z: usize,
    y: usize,
    x: usize,
) !void {
    assert(inline_info_slice[0] == .none);
    for (0..op_slice.len) |loop_idx| {
        const offset_out: usize = if (op_slice[loop_idx].isReduce()) 0 else //
        op_slice[loop_idx].out.at(a, z, y, x) - op_slice[loop_idx].out.offset;
        try generateOpPrefix(allocator, source, offset, padding, op_slice[loop_idx], inline_info_slice[loop_idx], //
            repeat_idx, kernel_op_idx + loop_idx, offset_out);
    }
    if (op_slice[0].isLinary()) {
        try writeBuffer(allocator, source, offset, padding, "{s}[{s}_{}_{}]", .{
            op_slice[0].in.name,
            op_slice[0].in.name,
            repeat_idx,
            kernel_op_idx,
        });
    } else {
        const offset_in: usize = op_slice[op_slice.len - 1].in.at(a, z, y, x) - op_slice[op_slice.len - 1].in.offset;
        try writeBuffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}]", .{
            op_slice[0].in.name,
            op_slice[0].in.name,
            repeat_idx,
            kernel_op_idx,
            offset_in,
        });
    }
    for (0..op_slice.len) |loop_idx| {
        const offset_in: usize = if (op_slice[loop_idx].isLinary()) 0 else //
        op_slice[loop_idx].in.at(a, z, y, x) - op_slice[loop_idx].in.offset;
        try generateOpPostfix(allocator, source, offset, padding, op_slice[loop_idx], inline_info_slice[loop_idx], //
            repeat_idx, kernel_op_idx + loop_idx, offset_in);
    }
}

fn generateOp(allocator: anytype, source: *[]u8, offset: *usize, padding: usize, pir: Pir, repeat_idx: usize) !void {
    var kernel_op_idx: usize = 0;
    for (0..pir.op_num) |_| {
        var kernel_op_idx_top: usize = kernel_op_idx + 1;
        for (kernel_op_idx + 1..pir.op_num) |inline_idx| {
            if (pir.inline_type[inline_idx] == .none) {
                break;
            } else {
                kernel_op_idx_top += 1;
            }
        }

        const op_slice: []Op = pir.op[kernel_op_idx..kernel_op_idx_top];
        const inline_info_slice: []Pir.Inline = pir.inline_type[kernel_op_idx..kernel_op_idx_top];

        assert(pir.inline_type[kernel_op_idx] == .none);

        // Every other op can not be a reduce op so it does not have a op header
        try generateOpHeader(allocator, source, offset, padding, op_slice[0], repeat_idx, kernel_op_idx);

        // To deal with reduce and linary ops.
        const a_max: usize = if (op_slice[0].isReduce()) op_slice[0].in.a_size else op_slice[0].out.a_size;
        const z_max: usize = if (op_slice[0].isReduce()) op_slice[0].in.z_size else op_slice[0].out.z_size;
        const y_max: usize = if (op_slice[0].isReduce()) op_slice[0].in.y_size else op_slice[0].out.y_size;
        const x_max: usize = if (op_slice[0].isReduce()) op_slice[0].in.x_size else op_slice[0].out.x_size;

        for (0..a_max) |a| {
            for (0..z_max) |z| {
                for (0..y_max) |y| {
                    for (0..x_max) |x| {
                        const offset_assign = if (op_slice[0].isReduce()) 0 else //
                        op_slice[0].out.at(a, z, y, x) - op_slice[0].out.offset;
                        try generateOpAssign(allocator, source, offset, padding, op_slice[0], repeat_idx, //
                            kernel_op_idx, offset_assign);
                        try generateOpBody(allocator, source, offset, padding, op_slice, inline_info_slice, //
                            repeat_idx, kernel_op_idx, a, z, y, x);
                        try writeBuffer(allocator, source, offset, padding, ";\n", .{});
                    }
                }
            }
        }

        try generateOpFooter(allocator, source, offset, padding, op_slice[0], repeat_idx, kernel_op_idx);

        kernel_op_idx = kernel_op_idx_top;

        if (kernel_op_idx >= pir.op_num) {
            break;
        }
    }
}

// TODO: Clean up this Args nonsense and the file structure. The way it currently is, it makes little to no sense to have cl.zig in a seperate directory
/// Create the source for a kernel computing `pir`
pub fn generate(allocator: anytype, pir: Pir, args: Args, size_global: usize, size_local: usize) ![]u8 {
    assert(args.arg_num > 0);
    assert(size_global % size_local == 0);
    assert(size_global > 0);
    assert(size_local > 0);
    // This might not be necessary because I think it is implied by the 3 above
    assert(size_global >= size_local);

    const capacity_min: usize = 2048;
    const padding: usize = 1024;
    var offset: usize = 0;

    var source: []u8 = try allocator.alloc(u8, capacity_min);
    @memset(source[0..], 0);

    try writeBuffer(allocator, &source, &offset, padding, "__kernel void {s}(", .{kernel_name});
    for (0..args.arg_num) |arg_idx| {
        if (arg_idx == 0) {
            try writeBuffer(allocator, &source, &offset, padding, "__global float *{s}", .{args.arg_name[arg_idx]});
        } else {
            try writeBuffer(allocator, &source, &offset, padding, ", __global float *{s}", .{args.arg_name[arg_idx]});
        }
    }
    try writeBuffer(allocator, &source, &offset, padding, ") {{\n", .{});

    const repeat_leftover: bool = (pir.repeat_num % size_global) != 0;
    // Not using std.math.divCeil here because it is kind of silly that it can error
    const repeat_kernel: usize = @divFloor(pir.repeat_num, size_global) + @intFromBool(repeat_leftover);

    try writeBuffer(allocator, &source, &offset, padding, "__const int gid = get_global_id(0);\n", .{});
    try writeBuffer(allocator, &source, &offset, padding, "int id;\n", .{});

    for (0..repeat_kernel) |repeat_idx| {
        if (repeat_leftover and repeat_idx == repeat_kernel - 1) {
            try writeBuffer(allocator, &source, &offset, padding, "if(gid < {}) {{\n", .{pir.repeat_num % size_global});
        }

        if (repeat_idx == 0) {
            try writeBuffer(allocator, &source, &offset, padding, "id = gid;\n", .{});
        } else {
            try writeBuffer(allocator, &source, &offset, padding, "id += {};\n", .{size_global});
        }

        try generateIndex(allocator, &source, &offset, padding, pir, repeat_idx);
        try generateOp(allocator, &source, &offset, padding, pir, repeat_idx);

        if (repeat_leftover and repeat_idx == repeat_kernel - 1) {
            try writeBuffer(allocator, &source, &offset, padding, "}}\n", .{});
        }
    }

    try writeBuffer(allocator, &source, &offset, padding, "}}\n", .{});

    return source;
}
