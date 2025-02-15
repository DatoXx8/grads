const std = @import("std");

const Ssa = @import("./ssa.zig").Ssa;
const DimInfo = Ssa.DimInfo;
const Assign = Ssa.Assign;

const assert = std.debug.assert;

const buffer_name_size = @import("../tensor.zig").buffer_name_size;
const Op = @import("../tensor.zig").Op;
const nameFromOffset = @import("../tensor.zig").Buffer.nameFromOffset;

const ClMem = @import("../runtimes/cl.zig").ClMem;

const Args = @import("./kernel.zig").Args;

pub const kernel_base_name = "kern{}";
pub const capacity_min: usize = 2048;
const padding: usize = 1024;

/// Expand buffer if necessary and set new bytes to 0
fn capacityEnsure(allocator: std.mem.Allocator, source: []u8, offset: usize) ![]u8 {
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
fn writeBuffer(allocator: std.mem.Allocator, source: *[]u8, offset: *usize, comptime fmt: []const u8, args: anytype) !void {
    // TODO: Validate that there is enough space for this and expand if there isn't
    offset.* += (try std.fmt.bufPrint(source.*[offset.*..], fmt, args)).len;
    source.* = try capacityEnsure(allocator, source.*, offset.*);
}

/// Generate computation for per-op indices
fn generateIndexSingular(
    allocator: std.mem.Allocator,
    source: *[]u8,
    offset: *usize,
    base: Assign.Base,
    loop_idx: usize,
    assign_idx: usize,
    inlined_idx: usize,
) !void {
    const dim_info: DimInfo = base.dim_info;
    try writeBuffer(
        allocator,
        source,
        offset,
        "int {s}_{}_{}_{} = (id+{})%{}/{}*{}+(id+{})%{}/{}*{}+(id+{})%{}/{}*{}+(id+{})%{}/{}*{}+{};\n",
        .{
            base.out.name(), loop_idx, assign_idx, inlined_idx, //
            dim_info.a_idx_out,  dim_info.a_reset_out, //
            dim_info.a_wait_out, dim_info.a_stride_out * base.out.a_stride,
            dim_info.z_idx_out,  dim_info.z_reset_out,
            dim_info.z_wait_out, dim_info.z_stride_out * base.out.z_stride,
            dim_info.y_idx_out,  dim_info.y_reset_out,
            dim_info.y_wait_out, dim_info.y_stride_out * base.out.y_stride,
            dim_info.x_idx_out,  dim_info.x_reset_out,
            dim_info.x_wait_out, dim_info.x_stride_out * base.out.x_stride,
            dim_info.off_out,
        },
    );
    if (!base.type.isUnary()) {
        try writeBuffer(
            allocator,
            source,
            offset,
            "int {s}_{}_{}_{} = (id+{})%{}/{}*{}+(id+{})%{}/{}*{}+(id+{})%{}/{}*{}+(id+{})%{}/{}*{}+{};\n",
            .{
                base.in.name(), loop_idx, assign_idx, inlined_idx, //
                dim_info.a_idx_in,  dim_info.a_reset_in, //
                dim_info.a_wait_in, dim_info.a_stride_in * base.in.a_stride,
                dim_info.z_idx_in,  dim_info.z_reset_in,
                dim_info.z_wait_in, dim_info.z_stride_in * base.in.z_stride,
                dim_info.y_idx_in,  dim_info.y_reset_in,
                dim_info.y_wait_in, dim_info.y_stride_in * base.in.y_stride,
                dim_info.x_idx_in,  dim_info.x_reset_in,
                dim_info.x_wait_in, dim_info.x_stride_in * base.in.x_stride,
                dim_info.off_in,
            },
        );
    }
}
/// Generate computation for per-op indices for an entire layer
fn generateIndex(
    allocator: std.mem.Allocator,
    source: *[]u8,
    offset: *usize,
    layer: []Assign,
    loop_idx: usize,
) !void {
    for (0..layer.len) |assign_idx| {
        try generateIndexSingular(allocator, source, offset, layer[assign_idx].base, loop_idx, assign_idx, 0);
        if (layer[assign_idx].inlined) |*inlined| {
            for (0..inlined.base.len) |inlined_idx| {
                try generateIndexSingular(allocator, source, offset, inlined.base[inlined_idx], loop_idx, assign_idx, 1 + inlined_idx);
            }
        }
    }
}

/// Generate a line of OpenCL code setting up the assign. Like setting to -INFINITY for reduce_max
fn generateHeader(
    allocator: std.mem.Allocator,
    source: *[]u8,
    offset: *usize,
    base: Assign.Base,
    loop_idx: usize,
    assign_idx: usize,
) !void {
    switch (base.type) {
        .reduce_sum => {
            try writeBuffer(allocator, source, offset, "{s}[{s}_{}_{}_0] = 0;\n", .{
                base.out.name(),
                base.out.name(),
                loop_idx,
                assign_idx,
            });
        },
        .reduce_avg => {
            try writeBuffer(allocator, source, offset, "{s}[{s}_{}_{}_0] = 0;\n", .{
                base.out.name(),
                base.out.name(),
                loop_idx,
                assign_idx,
            });
        },
        .reduce_max => {
            try writeBuffer(allocator, source, offset, "{s}[{s}_{}_{}_0] = -INFINITY;\n", .{
                base.out.name(),
                base.out.name(),
                loop_idx,
                assign_idx,
            });
        },
        .reduce_min => {
            try writeBuffer(allocator, source, offset, "{s}[{s}_{}_{}_0] = INFINITY;\n", .{
                base.out.name(),
                base.out.name(),
                loop_idx,
                assign_idx,
            });
        },
        else => {},
    }
}

/// Do post assign calculations. Currently only used for dividing by the size of the `in` buffer for reduce_avg
fn generateFooter(
    allocator: std.mem.Allocator,
    source: *[]u8,
    offset: *usize,
    base: Assign.Base,
    loop_idx: usize,
    assign_idx: usize,
) !void {
    switch (base.type) {
        .reduce_avg => {
            try writeBuffer(allocator, source, offset, "{s}[{s}_{}_{}_0] /= {};\n", .{
                base.out.name(),
                base.out.name(),
                loop_idx,
                assign_idx,
                base.in.a_size * base.in.z_size * base.in.y_size * base.in.x_size,
            });
        },
        else => {},
    }
}

fn generatePrefix(
    allocator: std.mem.Allocator,
    source: *[]u8,
    offset: *usize,
    base: Assign.Base,
    inlined_type: Ssa.Assign.Inlined.Type,
    loop_idx: usize,
    assign_idx: usize,
    inlined_idx: usize,
    offset_out: usize,
) !void {
    switch (base.type) {
        .unary_add => {
            try writeBuffer(allocator, source, offset, "({d:16} + ", .{
                base.u_var,
            });
        },
        .unary_subtract => {
            try writeBuffer(allocator, source, offset, "(", .{});
        },
        .unary_multiply => {
            try writeBuffer(allocator, source, offset, "({d:16} * ", .{
                base.u_var,
            });
        },
        .unary_divide => {
            try writeBuffer(allocator, source, offset, "(", .{});
        },
        .unary_exp => {
            try writeBuffer(allocator, source, offset, "exp(", .{});
        },
        .unary_log => {
            try writeBuffer(allocator, source, offset, "log(", .{});
        },
        .unary_square => {
            // TODO: Refactor this to be a multiplication instead of an exponentiation algorithm
            try writeBuffer(allocator, source, offset, "pow(", .{});
        },
        .unary_sqrt => {
            try writeBuffer(allocator, source, offset, "sqrt(", .{});
        },
        .unary_reciprocal => {
            try writeBuffer(allocator, source, offset, "1 / (", .{});
        },
        .unary_max => {
            try writeBuffer(allocator, source, offset, "fmax((float){d:16}, ", .{
                base.u_var,
            });
        },
        .unary_min => {
            try writeBuffer(allocator, source, offset, "fmin((float){d:16}, ", .{
                base.u_var,
            });
        },
        .unary_set => {
            try writeBuffer(allocator, source, offset, "(", .{});
        },
        .unary_random => {
            try writeBuffer(allocator, source, offset, "(", .{});
            unreachable;
        },
        .unary_tanh => {
            try writeBuffer(allocator, source, offset, "tanh(", .{});
        },
        .unary_absolute => {
            try writeBuffer(allocator, source, offset, "fabs(", .{});
        },
        .unary_sign => {
            try writeBuffer(allocator, source, offset, "(", .{});
            unreachable;
        },
        .binary_add => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{}_{} + {}] + ", .{
                    base.out.name(),
                    base.out.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, "(", .{});
            }
        },
        .binary_subtract => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{}_{} + {}] - ", .{
                    base.out.name(),
                    base.out.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, "(", .{});
            }
        },
        .binary_multiply => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{}_{} + {}] * ", .{
                    base.out.name(),
                    base.out.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, "(", .{});
            }
        },
        .binary_divide => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{}_{} + {}] / ", .{
                    base.out.name(),
                    base.out.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, "(", .{});
            }
        },
        .binary_max => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, "fmax({s}[{s}_{}_{}_{} + {}], ", .{
                    base.out.name(),
                    base.out.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, "fmax(", .{});
            }
        },
        .binary_min => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, "fmin({s}[{s}_{}_{}_{} + {}], ", .{
                    base.out.name(),
                    base.out.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, "fmin(", .{});
            }
        },
        .binary_set => {
            try writeBuffer(allocator, source, offset, "(", .{});
        },
        .linary_add => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{}_{} + {}] + ", .{
                    base.out.name(),
                    base.out.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, "(", .{});
            }
        },
        .linary_subtract => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{}_{} + {}] - ", .{
                    base.out.name(),
                    base.out.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, "(", .{});
            }
        },
        .linary_multiply => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{}_{} + {}] * ", .{
                    base.out.name(),
                    base.out.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, "(", .{});
            }
        },
        .linary_divide => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{}_{} + {}] / ", .{
                    base.out.name(),
                    base.out.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, "(", .{});
            }
        },
        .linary_max => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, "fmax({s}[{s}_{}_{}_{} + {}], ", .{
                    base.out.name(),
                    base.out.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, "fmax(", .{});
            }
        },
        .linary_min => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, "fmin({s}[{s}_{}_{}_{} + {}], ", .{
                    base.out.name(),
                    base.out.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_out,
                });
            } else {
                try writeBuffer(allocator, source, offset, "fmin(", .{});
            }
        },
        .linary_set => {
            try writeBuffer(allocator, source, offset, "(", .{});
        },
        .reduce_sum => {
            try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{}_{}] + ", .{
                base.out.name(),
                base.out.name(),
                loop_idx,
                assign_idx,
                inlined_idx,
            });
        },
        .reduce_avg => {
            try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{}_{}] + ", .{
                base.out.name(),
                base.out.name(),
                loop_idx,
                assign_idx,
                inlined_idx,
            });
        },
        .reduce_max => {
            try writeBuffer(allocator, source, offset, "fmax({s}[{s}_{}_{}_{}], ", .{
                base.out.name(),
                base.out.name(),
                loop_idx,
                assign_idx,
                inlined_idx,
            });
        },
        .reduce_min => {
            try writeBuffer(allocator, source, offset, "fmin({s}[{s}_{}_{}_{}], ", .{
                base.out.name(),
                base.out.name(),
                loop_idx,
                assign_idx,
                inlined_idx,
            });
        },
    }
}

/// Generate a line of OpenCL code computing one entry of `assign.out`
fn generatePostfix(
    allocator: std.mem.Allocator,
    source: *[]u8,
    offset: *usize,
    base: Assign.Base,
    inlined_type: Ssa.Assign.Inlined.Type,
    loop_idx: usize,
    assign_idx: usize,
    inlined_idx: usize,
    offset_in: usize,
) !void {
    switch (base.type) {
        .unary_add => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .unary_subtract => {
            try writeBuffer(allocator, source, offset, " - ({d:16}))", .{base.u_var});
        },
        .unary_multiply => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .unary_divide => {
            try writeBuffer(allocator, source, offset, " / ({d:16}))", .{base.u_var});
        },
        .unary_exp => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .unary_log => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .unary_square => {
            // TODO: Refactor this to be a multiplication instead of an exponentiation algorithm where that is advantageous
            try writeBuffer(allocator, source, offset, ", 2)", .{});
        },
        .unary_sqrt => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .unary_reciprocal => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .unary_max => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .unary_min => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .unary_set => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .unary_random => {
            try writeBuffer(allocator, source, offset, ")", .{});
            unreachable;
        },
        .unary_tanh => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .unary_absolute => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .unary_sign => {
            try writeBuffer(allocator, source, offset, ")", .{});
            unreachable;
        },
        .binary_add => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, " + {s}[{s}_{}_{}_{} + {}])", .{
                    base.in.name(),
                    base.in.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_in,
                });
            }
        },
        .binary_subtract => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, " - {s}[{s}_{}_{}_{} + {}])", .{
                    base.in.name(),
                    base.in.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_in,
                });
            }
        },
        .binary_multiply => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, " * {s}[{s}_{}_{}_{} + {}])", .{
                    base.in.name(),
                    base.in.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_in,
                });
            }
        },
        .binary_divide => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, " / {s}[{s}_{}_{}_{} + {}])", .{
                    base.in.name(),
                    base.in.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_in,
                });
            }
        },
        .binary_max => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, ", {s}[{s}_{}_{}_{} + {}])", .{
                    base.in.name(),
                    base.in.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_in,
                });
            }
        },
        .binary_min => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, ", {s}[{s}_{}_{}_{} + {}])", .{
                    base.in.name(),
                    base.in.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                    offset_in,
                });
            }
        },
        .binary_set => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .linary_add => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, " + {s}[{s}_{}_{}_{}])", .{
                    base.in.name(),
                    base.in.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                });
            }
        },
        .linary_subtract => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, " - {s}[{s}_{}_{}_{}])", .{
                    base.in.name(),
                    base.in.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                });
            }
        },
        .linary_multiply => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, " * {s}[{s}_{}_{}_{}])", .{
                    base.in.name(),
                    base.in.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                });
            }
        },
        .linary_divide => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, " / {s}[{s}_{}_{}_{}])", .{
                    base.in.name(),
                    base.in.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                });
            }
        },
        .linary_max => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, ", {s}[{s}_{}_{}_{}])", .{
                    base.in.name(),
                    base.in.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                });
            }
        },
        .linary_min => {
            if (inlined_type == .none or inlined_type == .out) {
                try writeBuffer(allocator, source, offset, ")", .{});
            } else {
                try writeBuffer(allocator, source, offset, ", {s}[{s}_{}_{}_{}])", .{
                    base.in.name(),
                    base.in.name(),
                    loop_idx,
                    assign_idx,
                    inlined_idx,
                });
            }
        },
        .linary_set => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .reduce_sum => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .reduce_avg => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .reduce_max => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .reduce_min => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
    }
}

fn generateAssign(
    allocator: std.mem.Allocator,
    source: *[]u8,
    offset: *usize,
    base: Assign.Base,
    loop_idx: usize,
    assign_idx: usize,
    offset_out: usize,
) !void {
    if (base.type.isReduce()) {
        assert(offset_out == 0);
    }
    try writeBuffer(allocator, source, offset, "{s}[{s}_{}_{}_0 + {}] = ", .{
        base.out.name(),
        base.out.name(),
        loop_idx,
        assign_idx,
        offset_out,
    });
}

fn generateBody(
    allocator: std.mem.Allocator,
    source: *[]u8,
    offset: *usize,
    assign: Assign,
    loop_idx: usize,
    assign_idx: usize,
    a: usize,
    z: usize,
    y: usize,
    x: usize,
) !void {
    const assign_num: usize = 1 + if (assign.inlined) |inlined| inlined.base.len else 0;
    for (0..assign_num) |inlined_idx| {
        if (inlined_idx == 0) {
            const base: Assign.Base = assign.base;
            const offset_out: usize = if (base.type.isReduce()) 0 else base.out.at(a, z, y, x) - base.out.offset;
            try generatePrefix(allocator, source, offset, base, .none, loop_idx, assign_idx, inlined_idx, offset_out);
        } else {
            const base: Assign.Base = if (inlined_idx == 0) assign.base else assign.inlined.?.base[inlined_idx - 1];
            const inlined_type: Assign.Inlined.Type = assign.inlined.?.type[inlined_idx - 1];
            const offset_out: usize = if (base.type.isReduce()) 0 else base.out.at(a, z, y, x) - base.out.offset;
            try generatePrefix(allocator, source, offset, base, inlined_type, loop_idx, assign_idx, inlined_idx, offset_out);
        }
    }

    const root: Assign.Base = if (assign_num == 1) assign.base else assign.inlined.?.base[assign_num - 2];
    if (root.type.isLinary()) {
        try writeBuffer(allocator, source, offset, "{s}[{s}_{}_{}_{}]", .{
            root.in.name(),
            root.in.name(),
            loop_idx,
            assign_idx,
            assign_num - 1,
        });
    } else {
        if (root.type == .unary_set) {
            try writeBuffer(allocator, source, offset, "{d}", .{root.u_var});
        } else {
            const offset_in: usize = root.in.at(a, z, y, x) - root.in.offset;
            try writeBuffer(allocator, source, offset, "{s}[{s}_{}_{}_{} + {}]", .{
                root.in.name(),
                root.in.name(),
                loop_idx,
                assign_idx,
                assign_num - 1,
                offset_in,
            });
        }
    }

    for (0..assign_num) |inlined_idx_reverse| {
        const inlined_idx: usize = assign_num - (inlined_idx_reverse + 1);
        if (inlined_idx == 0) {
            const base: Assign.Base = assign.base;
            const offset_in: usize = if (base.type.isLinary()) 0 else base.in.at(a, z, y, x) - base.in.offset;
            try generatePostfix(allocator, source, offset, base, .none, loop_idx, assign_idx, inlined_idx, offset_in);
        } else {
            const base: Assign.Base = assign.inlined.?.base[inlined_idx - 1];
            const inlined_type: Assign.Inlined.Type = assign.inlined.?.type[inlined_idx - 1];
            const offset_in: usize = if (base.type.isLinary()) 0 else base.in.at(a, z, y, x) - base.in.offset;
            try generatePostfix(allocator, source, offset, base, inlined_type, loop_idx, assign_idx, inlined_idx, offset_in);
        }
    }
}

fn generateOp(allocator: std.mem.Allocator, source: *[]u8, offset: *usize, layer: []Ssa.Assign, loop_idx: usize) !void {
    for (0..layer.len) |assign_idx| {
        const base: Assign.Base = layer[assign_idx].base;
        // Every other op can not be a reduce op so it does not have a op header
        try generateHeader(allocator, source, offset, base, loop_idx, assign_idx);

        // To deal with reduce and linary ops.
        const a_max: usize = if (base.type.isReduce()) base.in.a_size else base.out.a_size;
        const z_max: usize = if (base.type.isReduce()) base.in.z_size else base.out.z_size;
        const y_max: usize = if (base.type.isReduce()) base.in.y_size else base.out.y_size;
        const x_max: usize = if (base.type.isReduce()) base.in.x_size else base.out.x_size;

        for (0..a_max) |a| {
            for (0..z_max) |z| {
                for (0..y_max) |y| {
                    for (0..x_max) |x| {
                        const offset_assign = if (base.type.isReduce()) 0 else base.out.at(a, z, y, x) - base.out.offset;
                        try generateAssign(allocator, source, offset, base, loop_idx, assign_idx, offset_assign);
                        try generateBody(allocator, source, offset, layer[assign_idx], loop_idx, assign_idx, a, z, y, x);
                        try writeBuffer(allocator, source, offset, ";\n", .{});
                    }
                }
            }
        }

        try generateFooter(allocator, source, offset, base, loop_idx, assign_idx);
    }
}

// TODO: Clean up the file structure. The way it currently is, it makes little to no sense to have cl.zig in a seperate directory
// TODO: compileKernel is a bad name
// TODO: allocate the source to be the max size it could be before hand and then the entire codegen could just be that one allocation
/// Create the source for a kernel computing all assigns in `layer` if it is a singular layer and otherwise it computes the loop described by the layers
pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: *[]u8,
    source_len: *usize,
    layer: []Ssa.Assign,
    loop_id: usize,
    loop_num: usize,
    args: Args,
    kernel_name: []u8,
    size_global: usize,
    size_local: usize,
) !void {
    assert(args.arg_mem.len == args.arg_name_offset.len);
    assert(size_global % size_local == 0);
    assert(size_global > 0);
    assert(size_local > 0);
    assert(size_global >= size_local); // This might not be necessary because I think it is implied by the 3 above
    assert(layer.len > 0);
    // TODO: There has to be a way to do this without an if statement
    if (loop_id == 0) {
        assert(loop_num == 1);
    } else {
        assert(loop_num > 1);
    }

    source.* = try capacityEnsure(allocator, source.*, source_len.*);
    try writeBuffer(allocator, source, source_len, "__kernel void {s}(", .{kernel_name});
    for (0..args.arg_mem.len) |arg_idx| {
        const name: [buffer_name_size]u8 = nameFromOffset(args.arg_name_offset[arg_idx]);
        if (arg_idx == 0) {
            try writeBuffer(allocator, source, source_len, "__global float *{s}", .{name});
        } else {
            try writeBuffer(allocator, source, source_len, ", __global float *{s}", .{name});
        }
    }
    try writeBuffer(allocator, source, source_len, ") {{\n", .{});

    const loop_leftover: bool = (loop_num % size_global) != 0;
    // NOTE: Not using std.math.divCeil here because it is kind of silly that it can error
    const loop_kernel: usize = @divFloor(loop_num, size_global) + @intFromBool(loop_leftover);

    try writeBuffer(allocator, source, source_len, "__const int gid = get_global_id(0);\n", .{});
    try writeBuffer(allocator, source, source_len, "int id;\n", .{});

    for (0..loop_kernel) |loop_idx| {
        if (loop_leftover and loop_idx == loop_kernel - 1) {
            try writeBuffer(allocator, source, source_len, "if(gid < {}) {{\n", .{loop_num % size_global});
        }

        if (loop_idx == 0) {
            try writeBuffer(allocator, source, source_len, "id = gid;\n", .{});
        } else {
            try writeBuffer(allocator, source, source_len, "id += {};\n", .{size_global});
        }

        // TODO: Reduce allocation by allocating the max size for this assign
        try generateIndex(allocator, source, source_len, layer, loop_idx);
        try generateOp(allocator, source, source_len, layer, loop_idx);

        if (loop_leftover and loop_idx == loop_kernel - 1) {
            try writeBuffer(allocator, source, source_len, "}}\n", .{});
        }
    }

    try writeBuffer(allocator, source, source_len, "}}\n", .{});
}
