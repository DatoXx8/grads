const std = @import("std");

const Ssa = @import("./ssa.zig").Ssa;
const DimInfo = Ssa.DimInfo;
const Assignment = Ssa.Assignment;

const assert = std.debug.assert;

const buffer_name_size = @import("../tensor.zig").buffer_name_size;
const Op = @import("../tensor.zig").Op;

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

// TODO: Might need to make my own format string in case the std lib does allocations in formatting and on top of that I could translate the entire kernel in one allocation and format

/// Write format string to buffer and ensure there is at least `padding` bytes left
fn writeBuffer(allocator: std.mem.Allocator, source: *[]u8, offset: *usize, comptime fmt: []const u8, args: anytype) !void {
    // TODO: Validate that there is enough space for this and expand if there isn't
    offset.* += (try std.fmt.bufPrint(source.*[offset.*..], fmt, args)).len;
    source.* = try capacityEnsure(allocator, source.*, offset.*);
}

/// Generate computation for per-op indices
fn generateIndex(
    allocator: std.mem.Allocator,
    source: *[]u8,
    offset: *usize,
    layer: []Assignment,
    loop_idx: usize,
) !void {
    for (0..layer.len) |assignment_idx| {
        const assignment: Assignment.Base = layer[assignment_idx].base;
        const dim_info: DimInfo = assignment.dim_info;
        try writeBuffer(
            allocator,
            source,
            offset,
            "int {s}_{}_{} = (id+{})%{}/{}*{}+(id+{})%{}/{}*{}+(id+{})%{}/{}*{}+(id+{})%{}/{}*{}+{};\n",
            .{
                assignment.out.name, loop_idx, assignment_idx, //
                dim_info.a_idx_out,  dim_info.a_reset_out, //
                dim_info.a_wait_out, dim_info.a_stride_out * assignment.out.a_stride,
                dim_info.z_idx_out,  dim_info.z_reset_out,
                dim_info.z_wait_out, dim_info.z_stride_out * assignment.out.z_stride,
                dim_info.y_idx_out,  dim_info.y_reset_out,
                dim_info.y_wait_out, dim_info.y_stride_out * assignment.out.y_stride,
                dim_info.x_idx_out,  dim_info.x_reset_out,
                dim_info.x_wait_out, dim_info.x_stride_out * assignment.out.x_stride,
                dim_info.off_out,
            },
        );
        if (!assignment.type.isUnary()) {
            try writeBuffer(
                allocator,
                source,
                offset,
                "int {s}_{}_{} = (id+{})%{}/{}*{}+(id+{})%{}/{}*{}+(id+{})%{}/{}*{}+(id+{})%{}/{}*{}+{};\n",
                .{
                    assignment.in.name, loop_idx, assignment_idx, //
                    dim_info.a_idx_in,  dim_info.a_reset_in, //
                    dim_info.a_wait_in, dim_info.a_stride_in * assignment.in.a_stride,
                    dim_info.z_idx_in,  dim_info.z_reset_in,
                    dim_info.z_wait_in, dim_info.z_stride_in * assignment.in.z_stride,
                    dim_info.y_idx_in,  dim_info.y_reset_in,
                    dim_info.y_wait_in, dim_info.y_stride_in * assignment.in.y_stride,
                    dim_info.x_idx_in,  dim_info.x_reset_in,
                    dim_info.x_wait_in, dim_info.x_stride_in * assignment.in.x_stride,
                    dim_info.off_in,
                },
            );
        }
    }
}

/// Generate a line of OpenCL code setting up the assignment. Like setting to -INFINITY for reduce_max
fn generateHeader(
    allocator: std.mem.Allocator,
    source: *[]u8,
    offset: *usize,
    base: Assignment.Base,
    loop_idx: usize,
    assignment_idx: usize,
) !void {
    switch (base.type) {
        .reduce_sum => {
            try writeBuffer(allocator, source, offset, "{s}[{s}_{}_{}] = 0;\n", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
            });
        },
        .reduce_avg => {
            try writeBuffer(allocator, source, offset, "{s}[{s}_{}_{}] = 0;\n", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
            });
        },
        .reduce_max => {
            try writeBuffer(allocator, source, offset, "{s}[{s}_{}_{}] = -INFINITY;\n", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
            });
        },
        .reduce_min => {
            try writeBuffer(allocator, source, offset, "{s}[{s}_{}_{}] = INFINITY;\n", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
            });
        },
        else => {},
    }
}

/// Do post assignment calculations. Currently only used for dividing by the size of the `in` buffer for reduce_avg
fn generateFooter(
    allocator: std.mem.Allocator,
    source: *[]u8,
    offset: *usize,
    base: Assignment.Base,
    loop_idx: usize,
    assignment_idx: usize,
) !void {
    switch (base.type) {
        .reduce_avg => {
            try writeBuffer(allocator, source, offset, "{s}[{s}_{}_{}] /= {};\n", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
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
    base: Assignment.Base,
    // inline_info: Pir.Inline,
    loop_idx: usize,
    assignment_idx: usize,
    offset_out: usize,
) !void {
    switch (base.type) {
        .unary_add => {
            try writeBuffer(allocator, source, offset, "({d} + ", .{
                base.u_var,
            });
        },
        .unary_subtract => {
            try writeBuffer(allocator, source, offset, "({d} - ", .{
                base.u_var,
            });
        },
        .unary_multiply => {
            try writeBuffer(allocator, source, offset, "({d} * ", .{
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
            try writeBuffer(allocator, source, offset, "fmax((float){d}, ", .{
                base.u_var,
            });
        },
        .unary_min => {
            try writeBuffer(allocator, source, offset, "fmin((float){d}, ", .{
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
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{} + {}] + ", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
                offset_out,
            });
            // } else {
            //     try writeBuffer(allocator, source, offset, "(", .{});
            // }
        },
        .binary_subtract => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{} + {}] - ", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
                offset_out,
            });
            // } else {
            //     try writeBuffer(allocator, source, offset, "(", .{});
            // }
        },
        .binary_multiply => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{} + {}] * ", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
                offset_out,
            });
            // } else {
            //     try writeBuffer(allocator, source, offset, "(", .{});
            // }
        },
        .binary_divide => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{} + {}] / ", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
                offset_out,
            });
            // } else {
            //     try writeBuffer(allocator, source, offset, "(", .{});
            // }
        },
        .binary_max => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, "fmax({s}[{s}_{}_{} + {}], ", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
                offset_out,
            });
            // } else {
            //     try writeBuffer(allocator, source, offset, "fmax(", .{});
            // }
        },
        .binary_min => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, "fmin({s}[{s}_{}_{} + {}], ", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
                offset_out,
            });
            // } else {
            //     try writeBuffer(allocator, source, offset, "fmin(", .{});
            // }
        },
        .binary_set => {
            try writeBuffer(allocator, source, offset, "(", .{});
        },
        .linary_add => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{}] + ", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
            });
            // } else {
            //     try writeBuffer(allocator, source, offset, "(", .{});
            // }
        },
        .linary_subtract => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{}] - ", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
            });
            // } else {
            //     try writeBuffer(allocator, source, offset, "(", .{});
            // }
        },
        .linary_multiply => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{}] * ", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
            });
            // } else {
            //     try writeBuffer(allocator, source, offset, "(", .{});
            // }
        },
        .linary_divide => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{}] / ", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
            });
            // } else {
            //     try writeBuffer(allocator, source, offset, "(", .{});
            // }
        },
        .linary_max => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, "fmax({s}[{s}_{}_{}], ", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
            });
            // } else {
            //     try writeBuffer(allocator, source, offset, "fmax(", .{});
            // }
        },
        .linary_min => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, "fmin({s}[{s}_{}_{}], ", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
            });
            // } else {
            //     try writeBuffer(allocator, source, offset, "fmin(", .{});
            // }
        },
        .linary_set => {
            try writeBuffer(allocator, source, offset, "(", .{});
        },
        .reduce_sum => {
            try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{}] + ", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
            });
        },
        .reduce_avg => {
            try writeBuffer(allocator, source, offset, "({s}[{s}_{}_{}] + ", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
            });
        },
        .reduce_max => {
            try writeBuffer(allocator, source, offset, "fmax({s}[{s}_{}_{}], ", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
            });
        },
        .reduce_min => {
            try writeBuffer(allocator, source, offset, "fmin({s}[{s}_{}_{}], ", .{
                base.out.name,
                base.out.name,
                loop_idx,
                assignment_idx,
            });
        },
    }
}

/// Generate a line of OpenCL code computing one entry of `assignment.out`
fn generatePostfix(
    allocator: std.mem.Allocator,
    source: *[]u8,
    offset: *usize,
    base: Assignment.Base,
    // inline_info: Pir.Inline,
    loop_idx: usize,
    assignment_idx: usize,
    offset_in: usize,
) !void {
    _ = loop_idx;
    _ = assignment_idx;
    _ = offset_in;
    switch (base.type) {
        .unary_add => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .unary_subtract => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .unary_multiply => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .unary_divide => {
            try writeBuffer(allocator, source, offset, "/ {})", .{base.u_var});
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
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, ")", .{});
            // } else {
            //     try writeBuffer(allocator, source, offset, " + {s}[{s}_{}_{} + {}])", .{
            //         assignment.in.name,
            //         assignment.in.name,
            //         loop_idx,
            //         assignment_idx,
            //         offset_in,
            //     });
            // }
        },
        .binary_subtract => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, ")", .{});
            // } else {
            //     try writeBuffer(allocator, source, offset, " - {s}[{s}_{}_{} + {}])", .{
            //         assignment.in.name,
            //         assignment.in.name,
            //         loop_idx,
            //         assignment_idx,
            //         offset_in,
            //     });
            // }
        },
        .binary_multiply => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .binary_divide => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, ")", .{});
            // } else {
            //     try writeBuffer(allocator, source, offset, " / {s}[{s}_{}_{} + {}])", .{
            //         assignment.in.name,
            //         assignment.in.name,
            //         loop_idx,
            //         assignment_idx,
            //         offset_in,
            //     });
            // }
        },
        .binary_max => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, ")", .{});
            // } else {
            //     try writeBuffer(allocator, source, offset, ", {s}[{s}_{}_{} + {}])", .{
            //         assignment.in.name,
            //         assignment.in.name,
            //         loop_idx,
            //         assignment_idx,
            //         offset_in,
            //     });
            // }
        },
        .binary_min => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, ")", .{});
            // } else {
            //     try writeBuffer(allocator, source, offset, ", {s}[{s}_{}_{} + {}])", .{
            //         assignment.in.name,
            //         assignment.in.name,
            //         loop_idx,
            //         assignment_idx,
            //         offset_in,
            //     });
            // }
        },
        .binary_set => {
            try writeBuffer(allocator, source, offset, ")", .{});
        },
        .linary_add => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, ")", .{});
            // } else {
            //     try writeBuffer(allocator, source, offset, " + {s}[{s}_{}_{}])", .{
            //         assignment.in.name,
            //         assignment.in.name,
            //         loop_idx,
            //         assignment_idx,
            //     });
            // }
        },
        .linary_subtract => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, ")", .{});
            // } else {
            //     try writeBuffer(allocator, source, offset, " - {s}[{s}_{}_{}])", .{
            //         assignment.in.name,
            //         assignment.in.name,
            //         loop_idx,
            //         assignment_idx,
            //     });
            // }
        },
        .linary_multiply => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, ")", .{});
            // } else {
            //     try writeBuffer(allocator, source, offset, " * {s}[{s}_{}_{}])", .{
            //         assignment.in.name,
            //         assignment.in.name,
            //         loop_idx,
            //         assignment_idx,
            //     });
            // }
        },
        .linary_divide => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, ")", .{});
            // } else {
            //     try writeBuffer(allocator, source, offset, " / {s}[{s}_{}_{}])", .{
            //         assignment.in.name,
            //         assignment.in.name,
            //         loop_idx,
            //         assignment_idx,
            //     });
            // }
        },
        .linary_max => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, ")", .{});
            // } else {
            //     try writeBuffer(allocator, source, offset, ", {s}[{s}_{}_{}])", .{
            //         assignment.in.name,
            //         assignment.in.name,
            //         loop_idx,
            //         assignment_idx,
            //     });
            // }
        },
        .linary_min => {
            // if (inline_info == .none or inline_info == .in) {
            try writeBuffer(allocator, source, offset, ")", .{});
            // } else {
            //     try writeBuffer(allocator, source, offset, ", {s}[{s}_{}_{}])", .{
            //         assignment.in.name,
            //         assignment.in.name,
            //         loop_idx,
            //         assignment_idx,
            //     });
            // }
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
    base: Assignment.Base,
    loop_idx: usize,
    assignment_idx: usize,
    offset_out: usize,
) !void {
    if (base.type.isReduce()) {
        assert(offset_out == 0);
    }
    try writeBuffer(allocator, source, offset, "{s}[{s}_{}_{} + {}] = ", .{
        base.out.name,
        base.out.name,
        loop_idx,
        assignment_idx,
        offset_out,
    });
}

fn generateBody(
    allocator: std.mem.Allocator,
    source: *[]u8,
    offset: *usize,
    base: Assignment.Base,
    loop_idx: usize,
    assignment_idx: usize,
    a: usize,
    z: usize,
    y: usize,
    x: usize,
) !void {
    const offset_out: usize = if (base.type.isReduce()) 0 else //
    base.out.at(a, z, y, x) - base.out.offset;
    try generatePrefix(allocator, source, offset, base, //
        loop_idx, assignment_idx, offset_out);
    if (base.type.isLinary()) {
        try writeBuffer(allocator, source, offset, "{s}[{s}_{}_{}]", .{
            base.in.name,
            base.in.name,
            loop_idx,
            assignment_idx,
        });
    } else {
        const offset_in: usize = base.in.at(a, z, y, x) - base.in.offset;
        try writeBuffer(allocator, source, offset, "{s}[{s}_{}_{} + {}]", .{
            base.in.name,
            base.in.name,
            loop_idx,
            assignment_idx,
            offset_in,
        });
    }
    const offset_in: usize = if (base.type.isLinary()) 0 else //
    base.in.at(a, z, y, x) - base.in.offset;
    try generatePostfix(allocator, source, offset, base, //
        loop_idx, assignment_idx, offset_in);
}

fn generateOp(allocator: std.mem.Allocator, source: *[]u8, offset: *usize, layer: []Ssa.Assignment, loop_idx: usize) !void {
    for (0..layer.len) |assignment_idx| {
        const base: Assignment.Base = layer[assignment_idx].base;
        // Every other op can not be a reduce op so it does not have a op header
        try generateHeader(allocator, source, offset, base, loop_idx, assignment_idx);

        // To deal with reduce and linary ops.
        const a_max: usize = if (base.type.isReduce()) base.in.a_size else base.out.a_size;
        const z_max: usize = if (base.type.isReduce()) base.in.z_size else base.out.z_size;
        const y_max: usize = if (base.type.isReduce()) base.in.y_size else base.out.y_size;
        const x_max: usize = if (base.type.isReduce()) base.in.x_size else base.out.x_size;

        for (0..a_max) |a| {
            for (0..z_max) |z| {
                for (0..y_max) |y| {
                    for (0..x_max) |x| {
                        const offset_assign = if (base.type.isReduce()) 0 else //
                        base.out.at(a, z, y, x) - base.out.offset;
                        try generateAssign(allocator, source, offset, base, loop_idx, //
                            assignment_idx, offset_assign);
                        try generateBody(allocator, source, offset, base, //
                            loop_idx, assignment_idx, a, z, y, x);
                        try writeBuffer(allocator, source, offset, ";\n", .{});
                    }
                }
            }
        }

        try generateFooter(allocator, source, offset, base, loop_idx, assignment_idx);
    }
}

// TODO: Clean up the file structure. The way it currently is, it makes little to no sense to have cl.zig in a seperate directory
// TODO: compileKernel is a bad name
/// Create the source for a kernel computing all assignments in `layer` if it is a singular layer and otherwise it computes the loop described by the layers
pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: *[]u8,
    source_len: *usize,
    layer: []Ssa.Assignment,
    loop_id: usize,
    loop_num: usize,
    args: Args,
    kernel_name: []u8,
    size_global: usize,
    size_local: usize,
) !void {
    assert(args.arg_num > 0);
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
    for (0..args.arg_num) |arg_idx| {
        if (arg_idx == 0) {
            try writeBuffer(allocator, source, source_len, "__global float *{s}", .{args.arg_name[arg_idx]});
        } else {
            try writeBuffer(allocator, source, source_len, ", __global float *{s}", .{args.arg_name[arg_idx]});
        }
    }
    try writeBuffer(allocator, source, source_len, ") {{\n", .{});

    const loop_leftover: bool = (loop_num % size_global) != 0;
    // Not using std.math.divCeil here because it is kind of silly that it can error
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

        // TODO: Reduce allocation by allocating the max size for this assignment
        try generateIndex(allocator, source, source_len, layer, loop_idx);
        try generateOp(allocator, source, source_len, layer, loop_idx);

        if (loop_leftover and loop_idx == loop_kernel - 1) {
            try writeBuffer(allocator, source, source_len, "}}\n", .{});
        }
    }

    try writeBuffer(allocator, source, source_len, "}}\n", .{});
}
