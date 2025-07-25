const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const bufPrint = std.fmt.bufPrint;

const Program = @import("../Program.zig");
const source_padding = Program.source_padding;
const Args = Program.Args;
const Pir = @import("../Pir.zig");
const Assign = Pir.Assign;
const Base = Pir.Base;
const DimInfo = Pir.DimInfo;
const Inlined = Pir.Inlined;
const Tensor = @import("../../Tensor.zig");
const buffer_name_size = Tensor.buffer_name_size;
const Op = Tensor.Op;
const nameFromId = Tensor.Buffer.nameFromId;
const todo = @import("../../util.zig").todo;

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

fn writeIndices(allocator: Allocator, source: *[]u8, offset: *usize, assign: Assign, kernel_loop_idx: usize) WriteSourceError!void {
    const inlined_num: u32 = 1 + (if (assign.inlined) |inlined| inlined.inlined_num else 0);
    var inlined_idx: u32 = 0;
    while (inlined_idx < inlined_num) : (inlined_idx += 1) {
        const base: Base = if (inlined_idx == 0) assign.base else assign.inlined.?.base[inlined_idx - 1];
        const out_dim: DimInfo = base.out_dim;
        const in_dim: DimInfo = base.in_dim;
        try writeSource(
            allocator,
            source,
            offset,
            "int {s}_{}_{} = (id%{})/{}*{}+(id%{})/{}*{}+(id%{})/{}*{}+(id%{})/{}*{}+{};\n",
            .{
                base.out.name(), kernel_loop_idx, inlined_idx, //
                out_dim.a_reset, out_dim.a_wait, out_dim.a_stride * base.out.a_stride, //
                out_dim.z_reset, out_dim.z_wait, out_dim.z_stride * base.out.z_stride, //
                out_dim.y_reset, out_dim.y_wait, out_dim.y_stride * base.out.y_stride, //
                out_dim.x_reset, out_dim.x_wait, out_dim.x_stride * base.out.x_stride, //
                out_dim.off,
            },
        );
        if (!base.type.isUnary()) {
            try writeSource(
                allocator,
                source,
                offset,
                "int {s}_{}_{} = (id%{})/{}*{}+(id%{})/{}*{}+(id%{})/{}*{}+(id%{})/{}*{}+{};\n",
                .{
                    base.in.name(), kernel_loop_idx, inlined_idx, //
                    in_dim.a_reset, in_dim.a_wait, in_dim.a_stride * base.in.a_stride, //
                    in_dim.z_reset, in_dim.z_wait, in_dim.z_stride * base.in.z_stride, //
                    in_dim.y_reset, in_dim.y_wait, in_dim.y_stride * base.in.y_stride, //
                    in_dim.x_reset, in_dim.x_wait, in_dim.x_stride * base.in.x_stride, //
                    in_dim.off,
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
        .expand_add,
        .expand_subtract,
        .expand_multiply,
        .expand_divide,
        .expand_set,
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
        .expand_max,
        .reduce_max,
        => {
            try writeSource(allocator, source, offset, "fmax(", .{});
        },
        .unary_min,
        .binary_min,
        .expand_min,
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
        .expand_add,
        .reduce_sum,
        .reduce_avg,
        => {
            try writeSource(allocator, source, offset, "+", .{});
        },
        .unary_subtract,
        .binary_subtract,
        .expand_subtract,
        => {
            try writeSource(allocator, source, offset, "-", .{});
        },
        .unary_multiply,
        .binary_multiply,
        .expand_multiply,
        => {
            try writeSource(allocator, source, offset, "*", .{});
        },
        .unary_divide,
        .binary_divide,
        .expand_divide,
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
        .expand_set,
        => {},
        .unary_max,
        .binary_max,
        .expand_max,
        .reduce_max,
        .unary_min,
        .binary_min,
        .expand_min,
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
        .expand_add,
        .expand_subtract,
        .expand_multiply,
        .expand_divide,
        .expand_max,
        .expand_min,
        .expand_set,
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
        .expand_add,
        .expand_subtract,
        .expand_multiply,
        .expand_divide,
        .expand_max,
        .expand_min,
        .reduce_sum,
        .reduce_max,
        .reduce_avg,
        .reduce_min,
        => {
            try writeSource(allocator, source, offset, "{s}[{s}_{}_{}+{}]", .{
                base.out.name(),
                base.out.name(),
                kernel_loop_idx,
                inlined_idx_curr,
                offset_out,
            });
        },
        .unary_set => {
            try writeSource(allocator, source, offset, "((float){d})", .{base.u_var});
        },
        .binary_set,
        .expand_set,
        => {},
    }
}

fn writeAssignInBase(
    allocator: Allocator,
    source: *[]u8,
    offset: *usize,
    base: Base,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    offset_in: u32,
) WriteSourceError!void {
    if (base.type.isExpand()) {
        assert(offset_in == 0);
    }
    switch (base.type) {
        .binary_add,
        .binary_subtract,
        .binary_multiply,
        .binary_divide,
        .binary_max,
        .binary_min,
        .expand_add,
        .expand_subtract,
        .expand_multiply,
        .expand_divide,
        .expand_max,
        .expand_min,
        .reduce_sum,
        .reduce_max,
        .reduce_avg,
        .reduce_min,
        .binary_set,
        .expand_set,
        => {
            try writeSource(allocator, source, offset, "{s}[{s}_{}_{}+{}]", .{
                base.in.name(),
                base.in.name(),
                kernel_loop_idx,
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

// inlined_idx_curr is 0 if it is the base assign and actual inlined index + 1 otherwise
fn writeAssignOut(
    allocator: Allocator,
    source: *[]u8,
    offset: *usize,
    inlined: Inlined,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    a: u32,
    z: u32,
    y: u32,
    x: u32,
) WriteSourceError!void {
    const inlined_idx_actual: u32 = inlined_idx_curr - 1;

    try writeAssignPrefix(allocator, source, offset, inlined.base[inlined_idx_actual]);

    const base_relevant: Base = inlined.base[inlined_idx_actual];
    const a_out: u32 = if (base_relevant.type.isReduce()) 0 else a;
    const z_out: u32 = if (base_relevant.type.isReduce()) 0 else z;
    const y_out: u32 = if (base_relevant.type.isReduce()) 0 else y;
    const x_out: u32 = if (base_relevant.type.isReduce()) 0 else x;
    if (inlined.out[inlined_idx_actual]) |inlined_out| {
        try writeAssignOut(allocator, source, offset, inlined, kernel_loop_idx,  inlined_out + 1, a_out, z_out, y_out, x_out);
    } else {
        const offset_out: u32 = base_relevant.out.at(a_out, z_out, y_out, x_out) - base_relevant.out.offset;
        try writeAssignOutBase(allocator, source, offset, base_relevant, kernel_loop_idx,  inlined_idx_curr, offset_out);
    }

    try writeAssignMidfix(allocator, source, offset, inlined.base[inlined_idx_actual]);

    const a_in: u32 = if (base_relevant.type.isExpand()) 0 else a;
    const z_in: u32 = if (base_relevant.type.isExpand()) 0 else z;
    const y_in: u32 = if (base_relevant.type.isExpand()) 0 else y;
    const x_in: u32 = if (base_relevant.type.isExpand()) 0 else x;
    if (inlined.in[inlined_idx_actual]) |inlined_in| {
        try writeAssignIn(allocator, source, offset, inlined, kernel_loop_idx,  inlined_in + 1, a_in, z_in, y_in, x_in);
    } else {
        if (!base_relevant.type.isUnary()) {
            const offset_in: u32 = base_relevant.in.at(a_in, z_in, y_in, x_in) - base_relevant.in.offset;
            try writeAssignInBase(allocator, source, offset, base_relevant, kernel_loop_idx,  inlined_idx_curr, offset_in);
        }
    }

    try writeAssignPostfix(allocator, source, offset, inlined.base[inlined_idx_actual]);
}

// inlined_idx_curr is 0 if it is the base assign and actual inlined index + 1 otherwise
fn writeAssignIn(
    allocator: Allocator,
    source: *[]u8,
    offset: *usize,
    inlined: Inlined,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    a: u32,
    z: u32,
    y: u32,
    x: u32,
) WriteSourceError!void {
    const inlined_idx_actual: u32 = inlined_idx_curr - 1;

    try writeAssignPrefix(allocator, source, offset, inlined.base[inlined_idx_actual]);

    const base_relevant: Base = inlined.base[inlined_idx_actual];
    const a_out: u32 = if (base_relevant.type.isReduce()) 0 else a;
    const z_out: u32 = if (base_relevant.type.isReduce()) 0 else z;
    const y_out: u32 = if (base_relevant.type.isReduce()) 0 else y;
    const x_out: u32 = if (base_relevant.type.isReduce()) 0 else x;
    if (inlined.out[inlined_idx_actual]) |inlined_out| {
        try writeAssignOut(allocator, source, offset, inlined, kernel_loop_idx,  inlined_out + 1, a_out, z_out, y_out, x_out);
    } else {
        const offset_out: u32 = base_relevant.out.at(a_out, z_out, y_out, x_out) - base_relevant.out.offset;
        try writeAssignOutBase(allocator, source, offset, base_relevant, kernel_loop_idx,  inlined_idx_curr, offset_out);
    }

    try writeAssignMidfix(allocator, source, offset, inlined.base[inlined_idx_actual]);

    const a_in: u32 = if (base_relevant.type.isExpand()) 0 else a;
    const z_in: u32 = if (base_relevant.type.isExpand()) 0 else z;
    const y_in: u32 = if (base_relevant.type.isExpand()) 0 else y;
    const x_in: u32 = if (base_relevant.type.isExpand()) 0 else x;
    if (inlined.in[inlined_idx_actual]) |inlined_in| {
        try writeAssignIn(allocator, source, offset, inlined, kernel_loop_idx,  inlined_in + 1, a_in, z_in, y_in, x_in);
    } else {
        if (!base_relevant.type.isUnary()) {
            const offset_in: u32 = base_relevant.in.at(a_in, z_in, y_in, x_in) - base_relevant.in.offset;
            try writeAssignInBase(allocator, source, offset, base_relevant, kernel_loop_idx,  inlined_idx_curr, offset_in);
        }
    }

    try writeAssignPostfix(allocator, source, offset, inlined.base[inlined_idx_actual]);
}

fn writeAssign(allocator: Allocator, source: *[]u8, offset: *usize, assign: Assign, kernel_loop_idx: u32) WriteSourceError!void {
    if (assign.base.type.isReduce()) {
        try writeSource(allocator, source, offset, "{s}[{s}_{}_{}+{}]={s};\n", .{
            assign.base.out.name(),
            assign.base.out.name(),
            kernel_loop_idx,
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
                    const a_out: u32 = if (assign.base.type.isReduce()) 0 else a;
                    const z_out: u32 = if (assign.base.type.isReduce()) 0 else z;
                    const y_out: u32 = if (assign.base.type.isReduce()) 0 else y;
                    const x_out: u32 = if (assign.base.type.isReduce()) 0 else x;

                    const offset_out: u32 = assign.base.out.at(a_out, z_out, y_out, x_out) - assign.base.out.offset;

                    try writeSource(allocator, source, offset, "{s}[{s}_{}_{}+{}] = ", .{
                        assign.base.out.name(),
                        assign.base.out.name(),
                        kernel_loop_idx,
                        0,
                        offset_out,
                    });

                    try writeAssignPrefix(allocator, source, offset, assign.base);

                    if (assign.inlined) |inlined| {
                        if (inlined.out_root) |inlined_out| {
                            try writeAssignOut(allocator, source, offset, inlined, kernel_loop_idx, inlined_out + 1, a_out, z_out, y_out, x_out);
                        } else {
                            try writeAssignOutBase(allocator, source, offset, assign.base, kernel_loop_idx, 0, offset_out);
                        }
                    } else {
                        try writeAssignOutBase(allocator, source, offset, assign.base, kernel_loop_idx, 0, offset_out);
                    }

                    try writeAssignMidfix(allocator, source, offset, assign.base);

                    const a_in: u32 = if (assign.base.type.isExpand()) 0 else a;
                    const z_in: u32 = if (assign.base.type.isExpand()) 0 else z;
                    const y_in: u32 = if (assign.base.type.isExpand()) 0 else y;
                    const x_in: u32 = if (assign.base.type.isExpand()) 0 else x;

                    if (assign.inlined) |inlined| {
                        if (inlined.in_root) |inlined_in| {
                            try writeAssignIn(allocator, source, offset, inlined, kernel_loop_idx, inlined_in + 1, a_in, z_in, y_in, x_in);
                        } else {
                            if (!assign.base.type.isUnary()) {
                                const offset_in: u32 = if (assign.base.type.isExpand()) 0 else assign.base.in.at(a_in, z_in, y_in, x_in) - assign.base.in.offset;
                                try writeAssignInBase(allocator, source, offset, assign.base, kernel_loop_idx,  0, offset_in);
                            }
                        }
                    } else {
                        if (!assign.base.type.isUnary()) {
                            const offset_in: u32 = if (assign.base.type.isExpand()) 0 else assign.base.in.at(a_in, z_in, y_in, x_in) - assign.base.in.offset;
                            try writeAssignInBase(allocator, source, offset, assign.base, kernel_loop_idx,  0, offset_in);
                        }
                    }

                    try writeAssignPostfix(allocator, source, offset, assign.base);

                    try writeSource(allocator, source, offset, ";\n", .{});
                }
            }
        }
    }

    if (assign.base.type == .reduce_avg) {
        try writeSource(allocator, source, offset, "{s}[{s}_{}_{}+{}]/={d};\n", .{
            assign.base.out.name(),
            assign.base.out.name(),
            kernel_loop_idx,
            0,
            0,
            @as(f64, @floatFromInt(a_size * z_size * y_size * x_size)),
        });
    }
}

pub fn assignCompile(
    _: *anyopaque,
    allocator: Allocator,
    source: *[]u8,
    offset: *usize,
    assign: Assign,
    kernel_name: []const u8,
    kernel_args: Args,
    size_global: u32,
    size_local: u32,
) ?void {
    assert(assign.base.repeats > 0);
    assert(size_global > 0);
    assert(size_local > 0);
    assert(size_global % size_local == 0);

    writeSource(allocator, source, offset, "__kernel void {s}(", .{kernel_name}) catch return null;
    assert(kernel_args.arg_mem.len == kernel_args.arg_id.len);
    for (0..kernel_args.arg_num) |arg_idx| {
        if (arg_idx == 0) {
            writeSource(allocator, source, offset, "__global float *{s}", .{nameFromId(kernel_args.arg_id[arg_idx])}) catch return null;
        } else {
            writeSource(allocator, source, offset, ", __global float *{s}", .{nameFromId(kernel_args.arg_id[arg_idx])}) catch return null;
        }
    }

    writeSource(allocator, source, offset, ") {{\n", .{}) catch return null;
    writeSource(allocator, source, offset, "const int gid = get_global_id(0);\n", .{}) catch return null;
    writeSource(allocator, source, offset, "int id;\n", .{}) catch return null;

    const kernel_loop_leftover: bool = (assign.base.repeats % size_global) != 0;
    const kernel_loop_num: u32 = @divFloor(assign.base.repeats, size_global) + @intFromBool(kernel_loop_leftover);

    var kernel_loop_idx: u32 = 0;
    while (kernel_loop_idx < kernel_loop_num) : (kernel_loop_idx += 1) {
        if (kernel_loop_idx == 0) {
            writeSource(allocator, source, offset, "id = gid;\n", .{}) catch return null;
        } else {
            writeSource(allocator, source, offset, "id += {};\n", .{size_global}) catch return null;
        }

        if (kernel_loop_idx == kernel_loop_num - 1 and kernel_loop_leftover) {
            writeSource(allocator, source, offset, "if(gid < {}) {{\n", .{assign.base.repeats % size_global}) catch return null;
        }

        writeIndices(allocator, source, offset, assign, kernel_loop_idx) catch return null;
        writeAssign(allocator, source, offset, assign, kernel_loop_idx) catch return null;

        if (kernel_loop_idx == kernel_loop_num - 1 and kernel_loop_leftover) {
            writeSource(allocator, source, offset, "}}\n", .{}) catch return null;
        }

        capacityEnsure(allocator, source, offset.*) catch return null;
    }

    writeSource(allocator, source, offset, "}}\n", .{}) catch return null;
    capacityEnsure(allocator, source, offset.*) catch return null;
}
