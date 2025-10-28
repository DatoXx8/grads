const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const bufPrint = std.fmt.bufPrint;

const Program = @import("../Program.zig");
const Args = Program.Args;
const Pir = @import("../Pir.zig");
const Assign = Pir.Assign;
const Base = Pir.Base;
const DimInfo = Pir.DimInfo;
const Inlined = Pir.Inlined;
const Buffer = @import("../../Buffer.zig");
const buffer_name_size = Buffer.buffer_name_size;
const nameFromId = Buffer.nameFromId;
const Linearized = @import("../../Linearized.zig");
const Op = Linearized.Op;
const todo = @import("../../util.zig").todo;
const Runtime = @import("Runtime.zig");
const RuntimeCl = Runtime.RuntimeCl;

const source_padding: u32 = 4 * 1024;
/// Write format string to buffer and ensure there is at least `padding` bytes left
fn writeSource(
    gpa: Allocator,
    source: *[]u8,
    offset: *usize,
    comptime fmt: []const u8,
    args: anytype,
) Allocator.Error!void {
    const written = bufPrint(source.*[offset.*..], fmt, args) catch unreachable;
    offset.* += written.len;
    if (source.len - offset.* <= source_padding) {
        source.* = try gpa.realloc(source.*, source.len * 2);
    }
}
fn writeIndices(
    gpa: Allocator,
    source: *[]u8,
    offset: *usize,
    assign: Assign,
    kernel_loop_idx: usize,
) Allocator.Error!void {
    const inlined_num: u32 = 1 + assign.inlined.inlined_num;
    var inlined_idx: u32 = 0;
    while (inlined_idx < inlined_num) : (inlined_idx += 1) {
        const base: Base = if (inlined_idx == 0) assign.base else assign.inlined.base[inlined_idx - 1];
        const out_dim: DimInfo = base.out_dim;
        const in_dim: DimInfo = base.in_dim;
        try writeSource(
            gpa,
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
        if (!base.kind.isUnary()) {
            try writeSource(
                gpa,
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
fn writeAssignPrefix(gpa: Allocator, source: *[]u8, offset: *usize, base: Base) Allocator.Error!void {
    switch (base.kind) {
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
            try writeSource(gpa, source, offset, "(", .{});
        },
        .unary_exp => {
            try writeSource(gpa, source, offset, "exp(", .{});
        },
        .unary_log => {
            try writeSource(gpa, source, offset, "log(", .{});
        },
        .unary_square => {
            try writeSource(gpa, source, offset, "pow(", .{});
        },
        .unary_sqrt => {
            try writeSource(gpa, source, offset, "sqrt(", .{});
        },
        .unary_reciprocal => {
            try writeSource(gpa, source, offset, "(1/", .{});
        },
        .unary_max,
        .binary_max,
        .expand_max,
        .reduce_max,
        => {
            try writeSource(gpa, source, offset, "fmax(", .{});
        },
        .unary_min,
        .binary_min,
        .expand_min,
        .reduce_min,
        => {
            try writeSource(gpa, source, offset, "fmin(", .{});
        },
        .unary_random => {
            todo(@src());
        },
        .unary_tanh => {
            try writeSource(gpa, source, offset, "tanh(", .{});
        },
        .unary_absolute => {
            try writeSource(gpa, source, offset, "fabs(", .{});
        },
        .unary_sign => {
            todo(@src());
        },
    }
}
fn writeAssignMidfix(gpa: Allocator, source: *[]u8, offset: *usize, base: Base) Allocator.Error!void {
    switch (base.kind) {
        .unary_add,
        .binary_add,
        .expand_add,
        .reduce_sum,
        .reduce_avg,
        => {
            try writeSource(gpa, source, offset, "+", .{});
        },
        .unary_subtract,
        .binary_subtract,
        .expand_subtract,
        => {
            try writeSource(gpa, source, offset, "-", .{});
        },
        .unary_multiply,
        .binary_multiply,
        .expand_multiply,
        => {
            try writeSource(gpa, source, offset, "*", .{});
        },
        .unary_divide,
        .binary_divide,
        .expand_divide,
        => {
            try writeSource(gpa, source, offset, "/", .{});
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
            try writeSource(gpa, source, offset, ",", .{});
        },
        .unary_random => {
            todo(@src());
        },
        .unary_sign => {
            todo(@src());
        },
    }
}
fn writeAssignPostfix(gpa: Allocator, source: *[]u8, offset: *usize, base: Base) Allocator.Error!void {
    switch (base.kind) {
        .unary_add,
        .unary_subtract,
        .unary_multiply,
        .unary_divide,
        .unary_max,
        .unary_min,
        => {
            try writeSource(gpa, source, offset, "((float){d}))", .{base.u_var});
        },
        .unary_square,
        => {
            try writeSource(gpa, source, offset, ",2)", .{});
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
            try writeSource(gpa, source, offset, ")", .{});
        },
    }
}
fn writeAssignOutBase(
    ga: Allocator,
    source: *[]u8,
    offset: *usize,
    base: Base,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    offset_out: u32,
) Allocator.Error!void {
    if (base.kind.isReduce()) {
        assert(inlined_idx_curr == 0);
        assert(offset_out == 0);
    }
    switch (base.kind) {
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
            try writeSource(ga, source, offset, "{s}[{s}_{}_{}+{}]", .{
                base.out.name(),
                base.out.name(),
                kernel_loop_idx,
                inlined_idx_curr,
                offset_out,
            });
        },
        .unary_set => {
            try writeSource(ga, source, offset, "((float){d})", .{base.u_var});
        },
        .binary_set,
        .expand_set,
        => {},
    }
}
fn writeAssignInBase(
    gpa: Allocator,
    source: *[]u8,
    offset: *usize,
    base: Base,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    offset_in: u32,
) Allocator.Error!void {
    if (base.kind.isExpand()) {
        assert(offset_in == 0);
    }
    switch (base.kind) {
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
            try writeSource(gpa, source, offset, "{s}[{s}_{}_{}+{}]", .{
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
    gpa: Allocator,
    source: *[]u8,
    offset: *usize,
    inlined: Inlined,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    a: u32,
    z: u32,
    y: u32,
    x: u32,
) Allocator.Error!void {
    const inlined_idx_actual: u32 = inlined_idx_curr - 1;

    try writeAssignPrefix(gpa, source, offset, inlined.base[inlined_idx_actual]);

    const base_relevant: Base = inlined.base[inlined_idx_actual];
    const a_out: u32 = if (base_relevant.kind.isReduce()) 0 else a;
    const z_out: u32 = if (base_relevant.kind.isReduce()) 0 else z;
    const y_out: u32 = if (base_relevant.kind.isReduce()) 0 else y;
    const x_out: u32 = if (base_relevant.kind.isReduce()) 0 else x;
    if (inlined.out[inlined_idx_actual]) |inlined_out| {
        try writeAssignOut(gpa, source, offset, inlined, kernel_loop_idx, inlined_out + 1, a_out, z_out, y_out, x_out);
    } else {
        const offset_out: u32 = base_relevant.out.at(a_out, z_out, y_out, x_out) - base_relevant.out.offset;
        try writeAssignOutBase(gpa, source, offset, base_relevant, kernel_loop_idx, inlined_idx_curr, offset_out);
    }

    try writeAssignMidfix(gpa, source, offset, inlined.base[inlined_idx_actual]);

    const a_in: u32 = if (base_relevant.kind.isExpand()) 0 else a;
    const z_in: u32 = if (base_relevant.kind.isExpand()) 0 else z;
    const y_in: u32 = if (base_relevant.kind.isExpand()) 0 else y;
    const x_in: u32 = if (base_relevant.kind.isExpand()) 0 else x;
    if (inlined.in[inlined_idx_actual]) |inlined_in| {
        try writeAssignIn(gpa, source, offset, inlined, kernel_loop_idx, inlined_in + 1, a_in, z_in, y_in, x_in);
    } else {
        if (!base_relevant.kind.isUnary()) {
            const offset_in: u32 = base_relevant.in.at(a_in, z_in, y_in, x_in) - base_relevant.in.offset;
            try writeAssignInBase(gpa, source, offset, base_relevant, kernel_loop_idx, inlined_idx_curr, offset_in);
        }
    }

    try writeAssignPostfix(gpa, source, offset, inlined.base[inlined_idx_actual]);
}
// inlined_idx_curr is 0 if it is the base assign and actual inlined index + 1 otherwise
fn writeAssignIn(
    gpa: Allocator,
    source: *[]u8,
    offset: *usize,
    inlined: Inlined,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    a: u32,
    z: u32,
    y: u32,
    x: u32,
) Allocator.Error!void {
    const inlined_idx_actual: u32 = inlined_idx_curr - 1;

    try writeAssignPrefix(gpa, source, offset, inlined.base[inlined_idx_actual]);

    const base_relevant: Base = inlined.base[inlined_idx_actual];
    const a_out: u32 = if (base_relevant.kind.isReduce()) 0 else a;
    const z_out: u32 = if (base_relevant.kind.isReduce()) 0 else z;
    const y_out: u32 = if (base_relevant.kind.isReduce()) 0 else y;
    const x_out: u32 = if (base_relevant.kind.isReduce()) 0 else x;
    if (inlined.out[inlined_idx_actual]) |inlined_out| {
        try writeAssignOut(gpa, source, offset, inlined, kernel_loop_idx, inlined_out + 1, a_out, z_out, y_out, x_out);
    } else {
        const offset_out: u32 = base_relevant.out.at(a_out, z_out, y_out, x_out) - base_relevant.out.offset;
        try writeAssignOutBase(gpa, source, offset, base_relevant, kernel_loop_idx, inlined_idx_curr, offset_out);
    }

    try writeAssignMidfix(gpa, source, offset, inlined.base[inlined_idx_actual]);

    const a_in: u32 = if (base_relevant.kind.isExpand()) 0 else a;
    const z_in: u32 = if (base_relevant.kind.isExpand()) 0 else z;
    const y_in: u32 = if (base_relevant.kind.isExpand()) 0 else y;
    const x_in: u32 = if (base_relevant.kind.isExpand()) 0 else x;
    if (inlined.in[inlined_idx_actual]) |inlined_in| {
        try writeAssignIn(gpa, source, offset, inlined, kernel_loop_idx, inlined_in + 1, a_in, z_in, y_in, x_in);
    } else {
        if (!base_relevant.kind.isUnary()) {
            const offset_in: u32 = base_relevant.in.at(a_in, z_in, y_in, x_in) - base_relevant.in.offset;
            try writeAssignInBase(gpa, source, offset, base_relevant, kernel_loop_idx, inlined_idx_curr, offset_in);
        }
    }

    try writeAssignPostfix(gpa, source, offset, inlined.base[inlined_idx_actual]);
}
fn writeAssign(gpa: Allocator, source: *[]u8, offset: *usize, assign: Assign, kernel_loop_idx: u32) Allocator.Error!void {
    if (assign.base.kind.isReduce()) {
        try writeSource(gpa, source, offset, "{s}[{s}_{}_{}+{}]={s};\n", .{
            assign.base.out.name(),
            assign.base.out.name(),
            kernel_loop_idx,
            0,
            0,
            switch (assign.base.kind) {
                .reduce_sum => "0",
                .reduce_avg => "0",
                .reduce_max => "-INFINITY",
                .reduce_min => "INFINITY",
                else => unreachable,
            },
        });
    }

    const a_size: u32 = if (assign.base.kind.isReduce()) assign.base.in.a_size else assign.base.out.a_size;
    const z_size: u32 = if (assign.base.kind.isReduce()) assign.base.in.z_size else assign.base.out.z_size;
    const y_size: u32 = if (assign.base.kind.isReduce()) assign.base.in.y_size else assign.base.out.y_size;
    const x_size: u32 = if (assign.base.kind.isReduce()) assign.base.in.x_size else assign.base.out.x_size;

    var a: u32 = 0;
    while (a < a_size) : (a += 1) {
        var z: u32 = 0;
        while (z < z_size) : (z += 1) {
            var y: u32 = 0;
            while (y < y_size) : (y += 1) {
                var x: u32 = 0;
                while (x < x_size) : (x += 1) {
                    const a_out: u32 = if (assign.base.kind.isReduce()) 0 else a;
                    const z_out: u32 = if (assign.base.kind.isReduce()) 0 else z;
                    const y_out: u32 = if (assign.base.kind.isReduce()) 0 else y;
                    const x_out: u32 = if (assign.base.kind.isReduce()) 0 else x;

                    const offset_out: u32 = assign.base.out.at(a_out, z_out, y_out, x_out) - assign.base.out.offset;

                    try writeSource(gpa, source, offset, "{s}[{s}_{}_{}+{}] = ", .{
                        assign.base.out.name(),
                        assign.base.out.name(),
                        kernel_loop_idx,
                        0,
                        offset_out,
                    });

                    try writeAssignPrefix(gpa, source, offset, assign.base);

                    if (assign.inlined.out_root) |inlined_out| {
                        try writeAssignOut(gpa, source, offset, assign.inlined, kernel_loop_idx, inlined_out + 1, a_out, z_out, y_out, x_out);
                    } else {
                        try writeAssignOutBase(gpa, source, offset, assign.base, kernel_loop_idx, 0, offset_out);
                    }

                    try writeAssignMidfix(gpa, source, offset, assign.base);

                    const a_in: u32 = if (assign.base.kind.isExpand()) 0 else a;
                    const z_in: u32 = if (assign.base.kind.isExpand()) 0 else z;
                    const y_in: u32 = if (assign.base.kind.isExpand()) 0 else y;
                    const x_in: u32 = if (assign.base.kind.isExpand()) 0 else x;

                    if (assign.inlined.in_root) |inlined_in| {
                        try writeAssignIn(gpa, source, offset, assign.inlined, kernel_loop_idx, inlined_in + 1, a_in, z_in, y_in, x_in);
                    } else {
                        if (!assign.base.kind.isUnary()) {
                            const offset_in: u32 = if (assign.base.kind.isExpand()) 0 else assign.base.in.at(a_in, z_in, y_in, x_in) - assign.base.in.offset;
                            try writeAssignInBase(gpa, source, offset, assign.base, kernel_loop_idx, 0, offset_in);
                        }
                    }

                    try writeAssignPostfix(gpa, source, offset, assign.base);

                    try writeSource(gpa, source, offset, ";\n", .{});
                }
            }
        }
    }

    if (assign.base.kind == .reduce_avg) {
        try writeSource(gpa, source, offset, "{s}[{s}_{}_{}+{}]/={d};\n", .{
            assign.base.out.name(),
            assign.base.out.name(),
            kernel_loop_idx,
            0,
            0,
            @as(f64, @floatFromInt(a_size * z_size * y_size * x_size)),
        });
    }
}
fn writeAssignOutBaseBlock(
    gpa: Allocator,
    source: *[]u8,
    offset: *usize,
    base: Base,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    kernel_block_idx: u32,
) Allocator.Error!void {
    if (base.kind.isReduce()) {
        assert(inlined_idx_curr == 0);
        assert(kernel_block_idx == 0);
    }
    switch (base.kind) {
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
            try writeSource(gpa, source, offset, "{s}[{s}_{}_{}+{s}_{}_{}_{}]", .{
                base.out.name(),
                base.out.name(),
                kernel_loop_idx,
                inlined_idx_curr,
                base.out.name(),
                kernel_loop_idx,
                inlined_idx_curr,
                kernel_block_idx,
            });
        },
        .unary_set => {
            try writeSource(gpa, source, offset, "((float){d})", .{base.u_var});
        },
        .binary_set,
        .expand_set,
        => {},
    }
}
fn writeAssignInBaseBlock(
    gpa: Allocator,
    source: *[]u8,
    offset: *usize,
    base: Base,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    kernel_block_idx: u32,
) Allocator.Error!void {
    switch (base.kind) {
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
            try writeSource(gpa, source, offset, "{s}[{s}_{}_{}+{s}_{}_{}_{}]", .{
                base.in.name(),
                base.in.name(),
                kernel_loop_idx,
                inlined_idx_curr,
                base.in.name(),
                kernel_loop_idx,
                inlined_idx_curr,
                kernel_block_idx,
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
// $TODO Refactor this horrible shit
fn writeIndicesBlock(
    gpa: Allocator,
    source: *[]u8,
    offset: *usize,
    assign: Assign,
    kernel_loop_idx: u32,
    kernel_block_idx: u32,
) Allocator.Error!void {
    const inlined_num: u32 = 1 + assign.inlined.inlined_num;
    var inlined_idx: u32 = 0;
    while (inlined_idx < inlined_num) : (inlined_idx += 1) {
        const base: Base = if (inlined_idx == 0) assign.base else assign.inlined.base[inlined_idx - 1];
        if (base.kind.isReduce()) {
            try writeSource(
                gpa,
                source,
                offset,
                "int {s}_{}_{}_{} = {};\n",
                .{ base.out.name(), kernel_loop_idx, inlined_idx, kernel_block_idx, 0 },
            );
        } else {
            try writeSource(
                gpa,
                source,
                offset,
                "int {s}_{}_{}_{} = (id%{})/{}*{}+(id%{})/{}*{}+(id%{})/{}*{}+(id%{})/{}*{};\n",
                .{
                    base.out.name(), kernel_loop_idx, inlined_idx, kernel_block_idx, //
                    base.out.a_size * base.out.z_size * base.out.y_size * base.out.x_size, base.out.z_size * base.out.y_size * base.out.x_size, if (base.out.a_size == 1) 0 else base.out.a_stride, //
                    base.out.z_size * base.out.y_size * base.out.x_size,                   base.out.y_size * base.out.x_size,                   if (base.out.z_size == 1) 0 else base.out.z_stride,
                    base.out.y_size * base.out.x_size,                                     base.out.x_size,                                     if (base.out.y_size == 1) 0 else base.out.y_stride,
                    base.out.x_size,                                                       1,                                                   if (base.out.x_size == 1) 0 else base.out.x_stride,
                },
            );
        }
        if (!base.kind.isUnary()) {
            if (base.kind.isExpand()) {
                try writeSource(
                    gpa,
                    source,
                    offset,
                    "int {s}_{}_{}_{} = 0;\n",
                    .{ base.in.name(), kernel_loop_idx, inlined_idx, kernel_block_idx },
                );
            } else {
                try writeSource(
                    gpa,
                    source,
                    offset,
                    "int {s}_{}_{}_{} = (id%{})/{}*{}+(id%{})/{}*{}+(id%{})/{}*{}+(id%{})/{}*{};\n",
                    .{
                        base.in.name(), kernel_loop_idx, inlined_idx, kernel_block_idx, //
                        base.in.a_size * base.in.z_size * base.in.y_size * base.in.x_size, base.in.z_size * base.in.y_size * base.in.x_size, if (base.in.a_size == 1) 0 else base.in.a_stride, //
                        base.in.z_size * base.in.y_size * base.in.x_size,                  base.in.y_size * base.in.x_size,                  if (base.in.z_size == 1) 0 else base.in.z_stride,
                        base.in.y_size * base.in.x_size,                                   base.in.x_size,                                   if (base.in.y_size == 1) 0 else base.in.y_stride,
                        base.in.x_size,                                                    1,                                                if (base.in.x_size == 1) 0 else base.in.x_stride,
                    },
                );
            }
        }
    }
}
// inlined_idx_curr is 0 if it is the base assign and actual inlined index + 1 otherwise
fn writeAssignOutBlock(
    gpa: Allocator,
    source: *[]u8,
    offset: *usize,
    inlined: Inlined,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    kernel_block_idx: u32,
) Allocator.Error!void {
    const inlined_idx_actual: u32 = inlined_idx_curr - 1;

    try writeAssignPrefix(gpa, source, offset, inlined.base[inlined_idx_actual]);

    const base_relevant: Base = inlined.base[inlined_idx_actual];
    if (inlined.out[inlined_idx_actual]) |inlined_out| {
        try writeAssignOutBlock(gpa, source, offset, inlined, kernel_loop_idx, inlined_out + 1, kernel_block_idx);
    } else {
        try writeAssignOutBaseBlock(gpa, source, offset, base_relevant, kernel_loop_idx, inlined_idx_curr, kernel_block_idx);
    }

    try writeAssignMidfix(gpa, source, offset, inlined.base[inlined_idx_actual]);

    if (inlined.in[inlined_idx_actual]) |inlined_in| {
        try writeAssignInBlock(gpa, source, offset, inlined, kernel_loop_idx, inlined_in + 1, kernel_block_idx);
    } else {
        if (!base_relevant.kind.isUnary()) {
            try writeAssignInBaseBlock(gpa, source, offset, base_relevant, kernel_loop_idx, inlined_idx_curr, kernel_block_idx);
        }
    }

    try writeAssignPostfix(gpa, source, offset, inlined.base[inlined_idx_actual]);
}
// inlined_idx_curr is 0 if it is the base assign and actual inlined index + 1 otherwise
fn writeAssignInBlock(
    gpa: Allocator,
    source: *[]u8,
    offset: *usize,
    inlined: Inlined,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    kernel_block_idx: u32,
) Allocator.Error!void {
    const inlined_idx_actual: u32 = inlined_idx_curr - 1;

    try writeAssignPrefix(gpa, source, offset, inlined.base[inlined_idx_actual]);

    const base_relevant: Base = inlined.base[inlined_idx_actual];
    if (inlined.out[inlined_idx_actual]) |inlined_out| {
        try writeAssignOutBlock(gpa, source, offset, inlined, kernel_loop_idx, inlined_out + 1, kernel_block_idx);
    } else {
        try writeAssignOutBaseBlock(gpa, source, offset, base_relevant, kernel_loop_idx, inlined_idx_curr, kernel_block_idx);
    }

    try writeAssignMidfix(gpa, source, offset, inlined.base[inlined_idx_actual]);

    if (inlined.in[inlined_idx_actual]) |inlined_in| {
        try writeAssignInBlock(gpa, source, offset, inlined, kernel_loop_idx, inlined_in + 1, kernel_block_idx);
    } else {
        if (!base_relevant.kind.isUnary()) {
            try writeAssignInBaseBlock(gpa, source, offset, base_relevant, kernel_loop_idx, inlined_idx_curr, kernel_block_idx);
        }
    }

    try writeAssignPostfix(gpa, source, offset, inlined.base[inlined_idx_actual]);
}
fn writeAssignBlock(
    gpa: Allocator,
    source: *[]u8,
    offset: *usize,
    assign: Assign,
    kernel_loop_idx: u32,
    kernel_block_idx: u32,
) Allocator.Error!void {
    assert(!assign.base.kind.isReduce());

    try writeSource(gpa, source, offset, "{s}[{s}_{}_{}+{s}_{}_{}_{}] = ", .{
        assign.base.out.name(),
        assign.base.out.name(),
        kernel_loop_idx,
        0,
        assign.base.out.name(),
        kernel_loop_idx,
        0,
        kernel_block_idx,
    });

    try writeAssignPrefix(gpa, source, offset, assign.base);

    if (assign.inlined.out_root) |inlined_out| {
        try writeAssignOutBlock(gpa, source, offset, assign.inlined, kernel_loop_idx, inlined_out + 1, kernel_block_idx);
    } else {
        try writeAssignOutBaseBlock(gpa, source, offset, assign.base, kernel_loop_idx, 0, kernel_block_idx);
    }

    try writeAssignMidfix(gpa, source, offset, assign.base);

    if (assign.inlined.in_root) |inlined_in| {
        try writeAssignInBlock(gpa, source, offset, assign.inlined, kernel_loop_idx, inlined_in + 1, kernel_block_idx);
    } else {
        if (!assign.base.kind.isUnary()) {
            try writeAssignInBaseBlock(gpa, source, offset, assign.base, kernel_loop_idx, 0, kernel_block_idx);
        }
    }

    try writeAssignPostfix(gpa, source, offset, assign.base);

    try writeSource(gpa, source, offset, ";\n", .{});
}

pub fn assignCompile(
    _: *anyopaque,
    gpa: Allocator,
    source: *[]u8,
    offset: *usize,
    assign: Assign,
    name: []const u8,
    args: Args,
    size_global: u32,
    size_local: u32,
) Allocator.Error!void {
    assert(assign.base.repeats > 0);
    assert(size_global > 0);
    assert(size_local > 0);
    assert(size_global % size_local == 0);

    try writeSource(gpa, source, offset, "__kernel void {s}(", .{name});
    assert(args.arg_mem.len == args.arg_id.len);
    for (0..args.arg_num) |arg_idx| {
        const arg_name: [buffer_name_size]u8 = nameFromId(args.arg_id[arg_idx]);
        if (arg_idx == 0) {
            try writeSource(gpa, source, offset, "__global float *{s}", .{arg_name});
        } else {
            try writeSource(gpa, source, offset, ", __global float *{s}", .{arg_name});
        }
    }

    try writeSource(gpa, source, offset, ") {{\n" ++
        "const int gid = get_global_id(0);\n" ++
        "int id;\n", .{});

    // $TODO Merge these cases in to 1 case. Should not really be that difficult
    if (assign.split) {
        const a_size: u32 = if (assign.base.kind.isReduce()) assign.base.in.a_size else assign.base.out.a_size;
        const z_size: u32 = if (assign.base.kind.isReduce()) assign.base.in.z_size else assign.base.out.z_size;
        const y_size: u32 = if (assign.base.kind.isReduce()) assign.base.in.y_size else assign.base.out.y_size;
        const x_size: u32 = if (assign.base.kind.isReduce()) assign.base.in.x_size else assign.base.out.x_size;
        const size: u32 = a_size * z_size * y_size * x_size;
        const size_with_repeats: u32 = size * assign.base.repeats;

        const kernel_block_leftover: u32 = size_with_repeats % size_global;
        const kernel_block_size: u32 = std.math.divCeil(u32, size_with_repeats, size_global) catch unreachable;

        assert(kernel_block_size <= size);

        var kernel_block_idx: u32 = 0;
        while (kernel_block_idx < kernel_block_size) : (kernel_block_idx += 1) {
            if (kernel_block_idx == kernel_block_size - 1 and kernel_block_leftover != 0) {
                try writeSource(gpa, source, offset, "if(gid<{}) {{\n", .{kernel_block_leftover});
            }

            try writeSource(gpa, source, offset, "id = (gid+{})/{};\n", .{ size_global * kernel_block_idx, size });
            try writeIndices(gpa, source, offset, assign, kernel_block_idx);
            try writeSource(gpa, source, offset, "id = gid+{};\n", .{size_global * kernel_block_idx});
            try writeIndicesBlock(gpa, source, offset, assign, kernel_block_idx, kernel_block_idx);
            try writeAssignBlock(gpa, source, offset, assign, kernel_block_idx, kernel_block_idx);

            if (kernel_block_idx == kernel_block_size - 1 and kernel_block_leftover != 0) {
                try writeSource(gpa, source, offset, "}}\n", .{});
            }
        }
    } else {
        const kernel_loop_leftover: u32 = (assign.base.repeats) % size_global;
        const kernel_loop_num: u32 = @divFloor(assign.base.repeats, size_global) + @intFromBool(kernel_loop_leftover != 0);

        var kernel_loop_idx: u32 = 0;
        while (kernel_loop_idx < kernel_loop_num) : (kernel_loop_idx += 1) {
            try writeSource(gpa, source, offset, "id = gid+{};\n", .{size_global * kernel_loop_idx});

            if (kernel_loop_idx == kernel_loop_num - 1 and kernel_loop_leftover != 0) {
                try writeSource(gpa, source, offset, "if(gid < {}) {{\n", .{kernel_loop_leftover});
            }

            try writeIndices(gpa, source, offset, assign, kernel_loop_idx);
            try writeAssign(gpa, source, offset, assign, kernel_loop_idx);

            if (kernel_loop_idx == kernel_loop_num - 1 and kernel_loop_leftover != 0) {
                try writeSource(gpa, source, offset, "}}\n", .{});
            }
        }
    }

    try writeSource(gpa, source, offset, "}}\n", .{});
}
