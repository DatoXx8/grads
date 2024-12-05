// TODO: Optimisation
// Optimisation levels
// O0 - none
// O1 - inline, split, merge kernels
// O2 - fuse along all axis
// O3 - memory optimizer (SLOW!!!)

const std = @import("std");

const Pir = @import("./pir.zig").Pir;
const DimInfo = @import("./pir.zig").DimInfo;

const assert = @import("../util.zig").assert;

const buffer_name_size = @import("../tensor.zig").buffer_name_size;
const Op = @import("../tensor.zig").Op;

const ClMem = @import("../runtimes/cl.zig").ClMem;

const Args = @import("./kernel.zig").Args;
const kernel_name = @import("../runtimes/cl.zig").kernel_name;
const kernel_name_c = @import("../runtimes/cl.zig").kernel_name_c;

pub const Optimisation = enum(u8) {
    O0,
    O1,
    O2,
    O3,
};

/// Expand buffer if necessary and set new bytes to 0
fn capacity_ensure(allocator: anytype, source: []u8, offset: usize, padding: usize) ![]u8 {
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
fn write_buffer(allocator: anytype, source: *[]u8, offset: *usize, padding: usize, comptime fmt: []const u8, args: anytype) !void {
    // TODO: Validate that there is enough space for this and expand if there isn't
    offset.* += (try std.fmt.bufPrint(source.*[offset.*..], fmt, args)).len;
    source.* = try capacity_ensure(allocator, source.*, offset.*, padding);
}

/// Generate computation for per-op indices
fn generate_index(
    allocator: anytype,
    source: *[]u8,
    offset: *usize,
    padding: usize,
    op: Op,
    dim_info: DimInfo,
    repeat_idx: usize,
    op_idx: usize,
) !void {
    try write_buffer(allocator, source, offset, padding, "int {s}_{}_{} = id%{}/{}*{}+id%{}/{}*{}+id%{}/{}*{}+id%{}/{}*{}+{};\n", .{
        op.out.name,        repeat_idx,         op_idx, //
        dim_info.res_a_out, dim_info.wai_a_out, dim_info.str_a_out * op.out.a_stride,
        dim_info.res_z_out, dim_info.wai_z_out, dim_info.str_z_out * op.out.z_stride,
        dim_info.res_y_out, dim_info.wai_y_out, dim_info.str_y_out * op.out.y_stride,
        dim_info.res_a_out, dim_info.wai_a_out, dim_info.str_a_out * op.out.a_stride,
        dim_info.off_out,
    });
    if (!op.is_unary()) {
        try write_buffer(allocator, source, offset, padding, "int {s}_{}_{} = id%{}/{}*{}+id%{}/{}*{}+id%{}/{}*{}+id%{}/{}*{}+{};\n", .{
            op.in.name,        repeat_idx,        op_idx, //
            dim_info.res_a_in, dim_info.wai_a_in, dim_info.str_a_in * op.in.a_stride,
            dim_info.res_z_in, dim_info.wai_z_in, dim_info.str_z_in * op.in.z_stride,
            dim_info.res_y_in, dim_info.wai_y_in, dim_info.str_y_in * op.in.y_stride,
            dim_info.res_a_in, dim_info.wai_a_in, dim_info.str_a_in * op.in.a_stride,
            dim_info.off_in,
        });
    }
}

/// Generate a line of OpenCL code setting up the op. Like setting to -INFINITY for reduce_max
fn generate_op_header(
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
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{}] = 0;\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
            });
        },
        .reduce_avg => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{}] = 0;\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
            });
        },
        .reduce_max => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{}] = -INFINITY;\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
            });
        },
        .reduce_min => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{}] = INFINITY;\n", .{
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
fn generate_op_footer(
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
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{}] /= {};\n", .{
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
/// Generate a line of OpenCL code computing one entry of `op.out`
fn generate_op_singular(
    allocator: anytype,
    source: *[]u8,
    offset: *usize,
    padding: usize,
    op: Op,
    repeat_idx: usize,
    op_idx: usize,
    offset_out: usize,
    offset_in: usize,
) !void {
    switch (op.type) {
        .unary_add => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] += {d};\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.u_var,
            });
        },
        .unary_subtract => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] -= {d};\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.u_var,
            });
        },
        .unary_multiply => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] *= {d};\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.u_var,
            });
        },
        .unary_divide => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] /= {d};\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.u_var,
            });
        },
        .unary_exp => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = exp({s}[{s}_{}_{} + {}]);\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
            });
        },
        .unary_log => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = log({s}[{s}_{}_{} + {}]);\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
            });
        },
        .unary_square => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] *= {s}[{s}_{}_{} + {}];\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
            });
        },
        .unary_sqrt => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = sqrt({s}[{s}_{}_{} + {}]);\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
            });
        },
        .unary_reciprocal => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = 1 / {s}[{s}_{}_{} + {}];\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
            });
        },
        .unary_max => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = fmax({s}[{s}_{}_{} + {}], (float){d});\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.u_var,
            });
        },
        .unary_min => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = fmin({s}[{s}_{}_{} + {}], (float){d});\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.u_var,
            });
        },
        .unary_set => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = {d};\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.u_var,
            });
        },
        .unary_random => {
            assert(false);
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = {s}[{s}_{}_{} + {}];\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
            });
        },
        .unary_tanh => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = tanh({s}[{s}_{}_{} + {}]);\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
            });
        },
        .unary_absolute => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = fabs({s}[{s}_{}_{} + {}]);\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
            });
        },
        .unary_sign => {
            assert(false);
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = {s}[{s}_{}_{} + {}];\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
            });
        },
        .binary_add => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] += {s}[{s}_{}_{} + {}];\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
                offset_in,
            });
        },
        .binary_subtract => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] -= {s}[{s}_{}_{} + {}];\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
                offset_in,
            });
        },
        .binary_multiply => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] *= {s}[{s}_{}_{} + {}];\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
                offset_in,
            });
        },
        .binary_divide => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] /= {s}[{s}_{}_{} + {}];\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
                offset_in,
            });
        },
        .binary_max => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = fmax({s}[{s}_{}_{} + {}], {s}[{s}_{}_{} + {}]);\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
                offset_in,
            });
        },
        .binary_min => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = fmin({s}[{s}_{}_{} + {}], {s}[{s}_{}_{} + {}]);\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
                offset_in,
            });
        },
        .binary_set => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = {s}[{s}_{}_{} + {}];\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
                offset_in,
            });
        },
        .linary_add => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] += {s}[{s}_{}_{}];\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
            });
        },
        .linary_subtract => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] -= {s}[{s}_{}_{}];\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
            });
        },
        .linary_multiply => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] *= {s}[{s}_{}_{}];\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
            });
        },
        .linary_divide => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] /= {s}[{s}_{}_{}];\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
            });
        },
        .linary_max => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = fmax({s}[{s}_{}_{} + {}], {s}[{s}_{}_{}]);\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
            });
        },
        .linary_min => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = fmin({s}[{s}_{}_{} + {}], {s}[{s}_{}_{}]);\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
            });
        },
        .linary_set => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{} + {}] = {s}[{s}_{}_{}];\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                offset_out,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
            });
        },
        .reduce_sum => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{}] += {s}[{s}_{}_{} + {}];\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
                offset_in,
            });
        },
        .reduce_avg => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{}] += {s}[{s}_{}_{} + {}];\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
                offset_in,
            });
        },
        .reduce_max => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{}] = fmax({s}[{s}_{}_{}], {s}[{s}_{}_{} + {}]);\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
                offset_in,
            });
        },
        .reduce_min => {
            try write_buffer(allocator, source, offset, padding, "{s}[{s}_{}_{}] = fmin({s}[{s}_{}_{}], {s}[{s}_{}_{} + {}]);\n", .{
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                op.out.name,
                op.out.name,
                repeat_idx,
                op_idx,
                op.in.name,
                op.in.name,
                repeat_idx,
                op_idx,
                offset_in,
            });
        },
    }
}

fn generate_op(allocator: anytype, source: *[]u8, offset: *usize, padding: usize, op: Op, repeat_idx: usize, op_idx: usize) !void {
    // To deal with reduce and linary ops.
    const a_max: u32 = @max(op.out.a_size, op.in.a_size);
    const z_max: u32 = @max(op.out.z_size, op.in.z_size);
    const y_max: u32 = @max(op.out.y_size, op.in.y_size);
    const x_max: u32 = @max(op.out.x_size, op.in.x_size);

    try generate_op_header(allocator, source, offset, padding, op, repeat_idx, op_idx);
    for (0..a_max) |a| {
        for (0..z_max) |z| {
            for (0..y_max) |y| {
                for (0..x_max) |x| {
                    // TODO: Might have to deal with size = 1 for reduce and linary
                    var offset_out: usize = 0;
                    if (op.is_reduce()) {
                        offset_out = op.out.at(0, 0, 0, 0);
                    } else {
                        offset_out = op.out.at(a, z, y, x);
                    }
                    var offset_in: usize = 0;
                    if (op.is_linary()) {
                        offset_in = op.in.at(0, 0, 0, 0);
                    } else {
                        offset_in = op.in.at(a, z, y, x);
                    }
                    try generate_op_singular(allocator, source, offset, padding, op, repeat_idx, op_idx, offset_out, offset_in);
                }
            }
        }
    }
    try generate_op_footer(allocator, source, offset, padding, op, repeat_idx, op_idx);
}

// TODO: Clean up this Args nonsense and the file structure. The way it currently is, it makes little to no sense to have cl.zig in a seperate directory
/// Create the source for a kernel computing `pir`
pub fn generate(allocator: anytype, pir: Pir, args: Args, size_global: u32, size_local: u32, optimisation: Optimisation) ![]u8 {
    assert(optimisation == .O0);
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

    try write_buffer(allocator, &source, &offset, padding, "__kernel void {s}(", .{kernel_name});
    for (0..args.arg_num) |arg_idx| {
        if (arg_idx == 0) {
            try write_buffer(allocator, &source, &offset, padding, "__global float *{s}", .{args.arg_name[arg_idx]});
        } else {
            try write_buffer(allocator, &source, &offset, padding, ", __global float *{s}", .{args.arg_name[arg_idx]});
        }
    }
    try write_buffer(allocator, &source, &offset, padding, ") {{\n", .{});

    const repeat_leftover: bool = (pir.repeat_num % size_global) != 0;
    // Not using std.math.divCeil here because it is kind of silly that it can error
    const repeat_kernel: u32 = switch (repeat_leftover) {
        true => @divFloor(pir.repeat_num, size_global) + 1,
        false => @divFloor(pir.repeat_num, size_global),
    };

    try write_buffer(allocator, &source, &offset, padding, "__const int gid = get_global_id(0);\n", .{});
    try write_buffer(allocator, &source, &offset, padding, "int id;\n", .{});

    for (0..repeat_kernel) |repeat_idx| {
        if (repeat_leftover and repeat_idx == repeat_kernel - 1) {
            try write_buffer(allocator, &source, &offset, padding, "if(gid < {}) {{\n", .{pir.repeat_num % size_global});
        }

        if (repeat_idx == 0) {
            try write_buffer(allocator, &source, &offset, padding, "id = gid;\n", .{});
        } else {
            try write_buffer(allocator, &source, &offset, padding, "id += {};\n", .{size_global});
        }
        for (0..pir.op_num) |op_idx| {
            try generate_index(allocator, &source, &offset, padding, pir.op[op_idx], pir.dim_info[op_idx], repeat_idx, op_idx);
            try generate_op(allocator, &source, &offset, padding, pir.op[op_idx], repeat_idx, op_idx);
        }

        if (repeat_leftover and repeat_idx == repeat_kernel - 1) {
            try write_buffer(allocator, &source, &offset, padding, "}}\n", .{});
        }
    }

    try write_buffer(allocator, &source, &offset, padding, "}}\n", .{});

    std.debug.print("Source:\n{s}\n", .{source});
    return source;
}
