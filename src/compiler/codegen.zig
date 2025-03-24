const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const bufPrint = std.fmt.bufPrint;

const todo = @import("../util.zig").todo;

const Ssa = @import("./ssa.zig").Ssa;
const Assign = @import("./ssa.zig").Assign;
const Base = @import("./ssa.zig").Base;
const DimInfo = @import("./ssa.zig").DimInfo;
const Inlined = @import("./ssa.zig").Inlined;

const buffer_name_size = @import("../tensor.zig").buffer_name_size;
const Op = @import("../tensor.zig").Op;
const nameFromOffset = @import("../tensor.zig").Buffer.nameFromOffset;

const ClMem = @import("../runtimes/cl.zig").ClMem;

const Args = @import("./kernel.zig").Args;

pub const kernel_base_name = "kern{}";
pub const source_padding = 4096;

// $TODO Make my own format string implementation, can't really get faster trivially without changing behaviour, which I don't really mind

/// Expand buffer if necessary and set new bytes to 0
fn capacityEnsure(allocator: Allocator, source: *[]u8, offset: usize) Allocator.Error!void {
    if (source.len - offset < source_padding) {
        const len_old: usize = source.len;
        source.* = try allocator.realloc(source.*, len_old * 2);
        @memset(source.*[len_old..], 0);
    }
}

const WriteSourceError = Allocator.Error || std.fmt.BufPrintError;
/// Write format string to buffer and ensure there is at least `padding` bytes left
fn writeSource(allocator: Allocator, source: *[]u8, offset: *usize, comptime fmt: []const u8, args: anytype) WriteSourceError!void {
    // $TODO Validate that there is enough space for this and expand if there isn't
    const written = try bufPrint(source.*[offset.*..], fmt, args);
    offset.* += written.len;
    try capacityEnsure(allocator, source, offset.*);
}

fn writeIndices(allocator: Allocator, source: *[]u8, offset: *usize, assign: Assign, kernel_loop_idx: usize, assign_idx: usize) WriteSourceError!void {
    const inlined_num: u32 = 1 + (if (assign.inlined) |inlined| inlined.inlined_num else 0);
    var inlined_idx: u32 = 0;
    while (inlined_idx < inlined_num) : (inlined_idx += 1) {
        const base: Base = if (inlined_idx == 0) assign.base else assign.inlined.?.base[inlined_idx - 1];
        const dim_info: DimInfo = base.dim_info;
        try writeSource(
            allocator,
            source,
            offset,
            "int {s}_{}_{}_{} = (id+{})/{}*{}+(id+{})/{}*{}+(id+{})/{}*{}+(id+{})/{}*{}+{};\n",
            .{
                base.out.name(), kernel_loop_idx, assign_idx, inlined_idx, //
                dim_info.a_reset_out, dim_info.a_wait_out, dim_info.a_stride_out * base.out.a_stride, //
                dim_info.z_reset_out, dim_info.z_wait_out, dim_info.z_stride_out * base.out.z_stride, //
                dim_info.y_reset_out, dim_info.y_wait_out, dim_info.y_stride_out * base.out.y_stride, //
                dim_info.x_reset_out, dim_info.x_wait_out, dim_info.x_stride_out * base.out.x_stride, //
                dim_info.off_out,
            },
        );
        if (!base.type.isUnary()) {
            try writeSource(
                allocator,
                source,
                offset,
                "int {s}_{}_{}_{} = (id+{})/{}*{}+(id+{})/{}*{}+(id+{})/{}*{}+(id+{})/{}*{}+{};\n",
                .{
                    base.in.name(), kernel_loop_idx, assign_idx, inlined_idx, //
                    dim_info.a_reset_in, dim_info.a_wait_in, dim_info.a_stride_in * base.in.a_stride, //
                    dim_info.z_reset_in, dim_info.z_wait_in, dim_info.z_stride_in * base.in.z_stride, //
                    dim_info.y_reset_in, dim_info.y_wait_in, dim_info.y_stride_in * base.in.y_stride, //
                    dim_info.x_reset_in, dim_info.x_wait_in, dim_info.x_stride_in * base.in.x_stride, //
                    dim_info.off_in,
                },
            );
        }
    }
}

fn writeAssignPrefix(allocator: Allocator, source: *[]u8, offset: *usize, base: Base) WriteSourceError!void {
    switch (base.type) {
        .unary_add,
        .unary_subtract,
        .unary_multiply,
        .unary_divide,
        .unary_set,
        .binary_add,
        .binary_subtract,
        .binary_multiply,
        .binary_divide,
        .binary_set,
        .linary_add,
        .linary_subtract,
        .linary_multiply,
        .linary_divide,
        .linary_set,
        .reduce_sum,
        .reduce_avg,
        => {
            try writeSource(allocator, source, offset, "(", .{});
        },
        .unary_exp => {
            try writeSource(allocator, source, offset, "exp(", .{});
        },
        .unary_log => {
            try writeSource(allocator, source, offset, "log(", .{});
        },
        .unary_square => {
            try writeSource(allocator, source, offset, "pow(", .{});
        },
        .unary_sqrt => {
            try writeSource(allocator, source, offset, "sqrt(", .{});
        },
        .unary_reciprocal => {
            try writeSource(allocator, source, offset, "(1/", .{});
        },
        .unary_max,
        .binary_max,
        .linary_max,
        .reduce_max,
        => {
            try writeSource(allocator, source, offset, "fmax(", .{});
        },
        .unary_min,
        .binary_min,
        .linary_min,
        .reduce_min,
        => {
            try writeSource(allocator, source, offset, "fmin(", .{});
        },
        .unary_random => {
            todo(@src());
        },
        .unary_tanh => {
            try writeSource(allocator, source, offset, "tanh(", .{});
        },
        .unary_absolute => {
            try writeSource(allocator, source, offset, "fabs(", .{});
        },
        .unary_sign => {
            todo(@src());
        },
    }
}

fn writeAssignMidfix(allocator: Allocator, source: *[]u8, offset: *usize, base: Base) WriteSourceError!void {
    switch (base.type) {
        .unary_add,
        .binary_add,
        .linary_add,
        .reduce_sum,
        .reduce_avg,
        => {
            try writeSource(allocator, source, offset, "+", .{});
        },
        .unary_subtract,
        .binary_subtract,
        .linary_subtract,
        => {
            try writeSource(allocator, source, offset, "-", .{});
        },
        .unary_multiply,
        .binary_multiply,
        .linary_multiply,
        => {
            try writeSource(allocator, source, offset, "*", .{});
        },
        .unary_divide,
        .binary_divide,
        .linary_divide,
        => {
            try writeSource(allocator, source, offset, "/", .{});
        },
        .unary_reciprocal,
        .unary_exp,
        .unary_log,
        .unary_square,
        .unary_sqrt,
        .unary_tanh,
        .unary_absolute,
        .unary_set,
        .binary_set,
        .linary_set,
        => {},
        .unary_max,
        .binary_max,
        .linary_max,
        .reduce_max,
        .unary_min,
        .binary_min,
        .linary_min,
        .reduce_min,
        => {
            try writeSource(allocator, source, offset, ",", .{});
        },
        .unary_random => {
            todo(@src());
        },
        .unary_sign => {
            todo(@src());
        },
    }
}

fn writeAssignPostfix(allocator: Allocator, source: *[]u8, offset: *usize, base: Base) WriteSourceError!void {
    switch (base.type) {
        .unary_add,
        .unary_subtract,
        .unary_multiply,
        .unary_divide,
        .unary_max,
        .unary_min,
        => {
            try writeSource(allocator, source, offset, "((float){d}))", .{base.u_var});
        },
        .unary_square,
        => {
            try writeSource(allocator, source, offset, ",2)", .{});
        },
        .unary_set,
        .unary_exp,
        .unary_log,
        .unary_sqrt,
        .unary_reciprocal,
        .unary_random,
        .unary_tanh,
        .unary_absolute,
        .unary_sign,
        .binary_add,
        .binary_subtract,
        .binary_multiply,
        .binary_divide,
        .binary_max,
        .binary_min,
        .binary_set,
        .linary_add,
        .linary_subtract,
        .linary_multiply,
        .linary_divide,
        .linary_max,
        .linary_min,
        .linary_set,
        .reduce_sum,
        .reduce_max,
        .reduce_avg,
        .reduce_min,
        => {
            try writeSource(allocator, source, offset, ")", .{});
        },
    }
}

fn writeAssignOutBase(
    allocator: Allocator,
    source: *[]u8,
    offset: *usize,
    base: Base,
    kernel_loop_idx: u32,
    assign_idx: u32,
    inlined_idx_curr: u32,
    offset_out: u32,
) WriteSourceError!void {
    if (base.type.isReduce()) {
        assert(inlined_idx_curr == 0);
        assert(offset_out == 0);
    }
    switch (base.type) {
        .unary_add,
        .unary_subtract,
        .unary_multiply,
        .unary_divide,
        .unary_exp,
        .unary_log,
        .unary_square,
        .unary_sqrt,
        .unary_reciprocal,
        .unary_max,
        .unary_min,
        .unary_random,
        .unary_tanh,
        .unary_absolute,
        .unary_sign,
        .binary_add,
        .binary_subtract,
        .binary_multiply,
        .binary_divide,
        .binary_max,
        .binary_min,
        .linary_add,
        .linary_subtract,
        .linary_multiply,
        .linary_divide,
        .linary_max,
        .linary_min,
        .reduce_sum,
        .reduce_max,
        .reduce_avg,
        .reduce_min,
        => {
            try writeSource(allocator, source, offset, "{s}[{s}_{}_{}_{}+{}]", .{
                base.out.name(),
                base.out.name(),
                kernel_loop_idx,
                assign_idx,
                inlined_idx_curr,
                offset_out,
            });
        },
        .unary_set => {
            try writeSource(allocator, source, offset, "({d})", .{base.u_var});
        },
        .binary_set,
        .linary_set,
        => {},
    }
}

fn writeAssignInBase(
    allocator: Allocator,
    source: *[]u8,
    offset: *usize,
    base: Base,
    kernel_loop_idx: u32,
    assign_idx: u32,
    inlined_idx_curr: u32,
    offset_in: u32,
) WriteSourceError!void {
    if (base.type.isLinary()) {
        assert(offset_in == 0);
    }
    switch (base.type) {
        .binary_add,
        .binary_subtract,
        .binary_multiply,
        .binary_divide,
        .binary_max,
        .binary_min,
        .linary_add,
        .linary_subtract,
        .linary_multiply,
        .linary_divide,
        .linary_max,
        .linary_min,
        .reduce_sum,
        .reduce_max,
        .reduce_avg,
        .reduce_min,
        .binary_set,
        .linary_set,
        => {
            try writeSource(allocator, source, offset, "{s}[{s}_{}_{}_{}+{}]", .{
                base.in.name(),
                base.in.name(),
                kernel_loop_idx,
                assign_idx,
                inlined_idx_curr,
                offset_in,
            });
        },
        .unary_add,
        .unary_subtract,
        .unary_multiply,
        .unary_divide,
        .unary_max,
        .unary_min,
        .unary_set,
        .unary_exp,
        .unary_log,
        .unary_square,
        .unary_sqrt,
        .unary_reciprocal,
        .unary_random,
        .unary_tanh,
        .unary_absolute,
        .unary_sign,
        => unreachable,
    }
}

// $NOTE inlined_idx_curr is 0 if it is the base assign and actual inlined index + 1 otherwise
fn writeAssignOut(
    allocator: Allocator,
    source: *[]u8,
    offset: *usize,
    inlined: Inlined,
    kernel_loop_idx: u32,
    assign_idx: u32,
    inlined_idx_curr: u32,
    a: u32,
    z: u32,
    y: u32,
    x: u32,
) WriteSourceError!void {
    const inlined_idx_actual: u32 = inlined_idx_curr - 1;

    try writeAssignPrefix(allocator, source, offset, inlined.base[inlined_idx_actual]);

    if (inlined.out[inlined_idx_actual]) |inlined_out| {
        try writeAssignOut(allocator, source, offset, inlined, kernel_loop_idx, assign_idx, inlined_out + 1, a, z, y, x);
    } else {
        const base_relevant: Base = inlined.base[inlined_idx_actual];
        const offset_out: u32 = if (base_relevant.type.isReduce()) 0 else base_relevant.out.at(a, z, y, x) - base_relevant.out.offset;
        try writeAssignOutBase(allocator, source, offset, base_relevant, kernel_loop_idx, assign_idx, inlined_idx_curr, offset_out);
    }

    try writeAssignMidfix(allocator, source, offset, inlined.base[inlined_idx_actual]);

    if (inlined.in[inlined_idx_actual]) |inlined_in| {
        try writeAssignIn(allocator, source, offset, inlined, kernel_loop_idx, assign_idx, inlined_in + 1, a, z, y, x);
    } else {
        const base_relevant: Base = inlined.base[inlined_idx_actual];
        const offset_in: u32 = if (base_relevant.type.isReduce()) 0 else base_relevant.in.at(a, z, y, x) - base_relevant.in.offset;
        try writeAssignInBase(allocator, source, offset, base_relevant, kernel_loop_idx, assign_idx, inlined_idx_curr, offset_in);
    }

    try writeAssignPostfix(allocator, source, offset, inlined.base[inlined_idx_actual]);
}

// $NOTE inlined_idx_curr is 0 if it is the base assign and actual inlined index + 1 otherwise
fn writeAssignIn(
    allocator: Allocator,
    source: *[]u8,
    offset: *usize,
    inlined: Inlined,
    kernel_loop_idx: u32,
    assign_idx: u32,
    inlined_idx_curr: u32,
    a: u32,
    z: u32,
    y: u32,
    x: u32,
) WriteSourceError!void {
    const inlined_idx_actual: u32 = inlined_idx_curr - 1;

    try writeAssignPrefix(allocator, source, offset, inlined.base[inlined_idx_actual]);

    if (inlined.out[inlined_idx_actual]) |inlined_out| {
        try writeAssignOut(allocator, source, offset, inlined, kernel_loop_idx, assign_idx, inlined_out + 1, a, z, y, x);
    } else {
        const base_relevant: Base = inlined.base[inlined_idx_actual];
        const offset_out: u32 = if (base_relevant.type.isReduce()) 0 else base_relevant.out.at(a, z, y, x) - base_relevant.out.offset;
        try writeAssignOutBase(allocator, source, offset, base_relevant, kernel_loop_idx, assign_idx, inlined_idx_curr, offset_out);
    }

    try writeAssignMidfix(allocator, source, offset, inlined.base[inlined_idx_actual]);

    if (inlined.in[inlined_idx_actual]) |inlined_in| {
        try writeAssignIn(allocator, source, offset, inlined, kernel_loop_idx, assign_idx, inlined_in + 1, a, z, y, x);
    } else {
        const base_relevant: Base = inlined.base[inlined_idx_actual];
        const offset_in: u32 = if (base_relevant.type.isReduce()) 0 else base_relevant.in.at(a, z, y, x) - base_relevant.in.offset;
        try writeAssignInBase(allocator, source, offset, base_relevant, kernel_loop_idx, assign_idx, inlined_idx_curr, offset_in);
    }

    try writeAssignPostfix(allocator, source, offset, inlined.base[inlined_idx_actual]);
}

fn writeAssign(allocator: Allocator, source: *[]u8, offset: *usize, assign: Assign, kernel_loop_idx: u32, assign_idx: u32) WriteSourceError!void {
    if (assign.base.type.isReduce()) {
        try writeSource(allocator, source, offset, "{s}[{s}_{}_{}_{}+{}]={s};\n", .{
            assign.base.out.name(),
            assign.base.out.name(),
            kernel_loop_idx,
            assign_idx,
            0,
            0,
            switch (assign.base.type) {
                .reduce_sum => "0",
                .reduce_avg => "0",
                .reduce_max => "-INFINITY",
                .reduce_min => "INFINITY",
                else => unreachable,
            },
        });
    }

    const a_size: u32 = if (assign.base.type.isReduce()) assign.base.in.a_size else assign.base.out.a_size;
    const z_size: u32 = if (assign.base.type.isReduce()) assign.base.in.z_size else assign.base.out.z_size;
    const y_size: u32 = if (assign.base.type.isReduce()) assign.base.in.y_size else assign.base.out.y_size;
    const x_size: u32 = if (assign.base.type.isReduce()) assign.base.in.x_size else assign.base.out.x_size;

    var a: u32 = 0;
    while (a < a_size) : (a += 1) {
        var z: u32 = 0;
        while (z < z_size) : (z += 1) {
            var y: u32 = 0;
            while (y < y_size) : (y += 1) {
                var x: u32 = 0;
                while (x < x_size) : (x += 1) {
                    const offset_out: u32 = if (assign.base.type.isReduce()) 0 else assign.base.out.at(a, z, y, x) - assign.base.out.offset;

                    try writeSource(allocator, source, offset, "{s}[{s}_{}_{}_{}+{}] = ", .{
                        assign.base.out.name(),
                        assign.base.out.name(),
                        kernel_loop_idx,
                        assign_idx,
                        0,
                        offset_out,
                    });

                    try writeAssignPrefix(allocator, source, offset, assign.base);

                    if (assign.inlined) |inlined| {
                        if (inlined.out_root) |inlined_out| {
                            try writeAssignOut(allocator, source, offset, inlined, kernel_loop_idx, assign_idx, inlined_out + 1, a, z, y, x);
                        } else {
                            try writeAssignOutBase(allocator, source, offset, assign.base, kernel_loop_idx, assign_idx, 0, offset_out);
                        }
                    } else {
                        try writeAssignOutBase(allocator, source, offset, assign.base, kernel_loop_idx, assign_idx, 0, offset_out);
                    }

                    try writeAssignMidfix(allocator, source, offset, assign.base);

                    if (assign.inlined) |inlined| {
                        if (inlined.in_root) |inlined_in| {
                            try writeAssignIn(allocator, source, offset, inlined, kernel_loop_idx, assign_idx, inlined_in + 1, a, z, y, x);
                        } else {
                            const offset_in: u32 = if (assign.base.type.isLinary()) 0 else assign.base.in.at(a, z, y, x) - assign.base.in.offset;
                            try writeAssignInBase(allocator, source, offset, assign.base, kernel_loop_idx, assign_idx, 0, offset_in);
                        }
                    } else {
                        if (!assign.base.type.isUnary()) {
                            const offset_in: u32 = if (assign.base.type.isLinary()) 0 else assign.base.in.at(a, z, y, x) - assign.base.in.offset;
                            try writeAssignInBase(allocator, source, offset, assign.base, kernel_loop_idx, assign_idx, 0, offset_in);
                        }
                    }

                    try writeAssignPostfix(allocator, source, offset, assign.base);

                    try writeSource(allocator, source, offset, ";\n", .{});
                }
            }
        }
    }

    if (assign.base.type == .reduce_avg) {
        try writeSource(allocator, source, offset, "{s}[{s}_{}_{}_{}+{}]/={d};\n", .{
            assign.base.out.name(),
            assign.base.out.name(),
            kernel_loop_idx,
            assign_idx,
            0,
            0,
            @as(f64, @floatFromInt(a_size * z_size * y_size * x_size)),
        });
    }
}

pub fn compileKernel(
    allocator: Allocator,
    source: *[]u8,
    offset: *usize,
    assign: []const Assign,
    assign_loop_num: u32,
    kernel_name: []const u8,
    kernel_args: Args,
    size_global: u32,
    size_local: u32,
) WriteSourceError!void {
    assert(assign_loop_num > 0);
    assert(size_global > 0);
    assert(size_local > 0);
    assert(size_global % size_local == 0);

    try writeSource(allocator, source, offset, "__kernel void {s}(", .{kernel_name});
    assert(kernel_args.arg_mem.len == kernel_args.arg_name_offset.len);
    for (0..kernel_args.arg_name_offset.len) |arg_idx| {
        if (arg_idx == 0) {
            try writeSource(allocator, source, offset, "__global float *{s}", .{nameFromOffset(kernel_args.arg_name_offset[arg_idx])});
        } else {
            try writeSource(allocator, source, offset, ", __global float *{s}", .{nameFromOffset(kernel_args.arg_name_offset[arg_idx])});
        }
    }

    try writeSource(allocator, source, offset, ") {{\n", .{});
    try writeSource(allocator, source, offset, "const int gid = get_global_id(0);\n", .{});
    try writeSource(allocator, source, offset, "int id;\n", .{});

    const kernel_loop_leftover: bool = (assign_loop_num % size_global) != 0;
    const kernel_loop_num: u32 = @divFloor(assign_loop_num, size_global) + @intFromBool(kernel_loop_leftover);

    var kernel_loop_idx: u32 = 0;
    while (kernel_loop_idx < kernel_loop_num) : (kernel_loop_idx += 1) {
        if (kernel_loop_idx == 0) {
            try writeSource(allocator, source, offset, "id = gid;\n", .{});
        } else {
            try writeSource(allocator, source, offset, "id += {};\n", .{size_global});
        }

        if (kernel_loop_idx == kernel_loop_num - 1 and kernel_loop_leftover) {
            try writeSource(allocator, source, offset, "if(gid < {}) {{\n", .{assign_loop_num % size_global});
        }

        var assign_idx: u32 = 0;
        while (assign_idx < assign.len) : (assign_idx += 1) {
            try writeIndices(allocator, source, offset, assign[assign_idx], kernel_loop_idx, assign_idx);
            try writeAssign(allocator, source, offset, assign[assign_idx], kernel_loop_idx, assign_idx);
        }

        if (kernel_loop_idx == kernel_loop_num - 1 and kernel_loop_leftover) {
            try writeSource(allocator, source, offset, "}}\n", .{});
        }

        try capacityEnsure(allocator, source, offset.*);
    }

    try writeSource(allocator, source, offset, "}}\n", .{});
    try capacityEnsure(allocator, source, offset.*);
}
