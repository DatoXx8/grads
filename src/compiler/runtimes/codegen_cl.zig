const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const bufPrint = std.fmt.bufPrint;

const Program = @import("../Program.zig");
const length_int_max = Program.length_int_max;
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
const Runtime = @import("Runtime.zig");
const RuntimeCl = Runtime.RuntimeCl;

/// Write format string to buffer and ensure there is at least `padding` bytes left
fn writeSource(source: *[]u8, offset: *usize, comptime fmt: []const u8, args: anytype) void {
    const written = bufPrint(source.*[offset.*..], fmt, args) catch unreachable;
    offset.* += written.len;
}
fn writeIndices(source: *[]u8, offset: *usize, assign: Assign, kernel_loop_idx: usize) void {
    const inlined_num: u32 = 1 + (if (assign.inlined) |inlined| inlined.inlined_num else 0);
    var inlined_idx: u32 = 0;
    while (inlined_idx < inlined_num) : (inlined_idx += 1) {
        const base: Base = if (inlined_idx == 0) assign.base else assign.inlined.?.base[inlined_idx - 1];
        const out_dim: DimInfo = base.out_dim;
        const in_dim: DimInfo = base.in_dim;
        writeSource(
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
            writeSource(
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
fn writeAssignPrefix(source: *[]u8, offset: *usize, base: Base) void {
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
            writeSource(source, offset, "(", .{});
        },
        .unary_exp => {
            writeSource(source, offset, "exp(", .{});
        },
        .unary_log => {
            writeSource(source, offset, "log(", .{});
        },
        .unary_square => {
            writeSource(source, offset, "pow(", .{});
        },
        .unary_sqrt => {
            writeSource(source, offset, "sqrt(", .{});
        },
        .unary_reciprocal => {
            writeSource(source, offset, "(1/", .{});
        },
        .unary_max,
        .binary_max,
        .expand_max,
        .reduce_max,
        => {
            writeSource(source, offset, "fmax(", .{});
        },
        .unary_min,
        .binary_min,
        .expand_min,
        .reduce_min,
        => {
            writeSource(source, offset, "fmin(", .{});
        },
        .unary_random => {
            todo(@src());
        },
        .unary_tanh => {
            writeSource(source, offset, "tanh(", .{});
        },
        .unary_absolute => {
            writeSource(source, offset, "fabs(", .{});
        },
        .unary_sign => {
            todo(@src());
        },
    }
}
fn writeAssignMidfix(source: *[]u8, offset: *usize, base: Base) void {
    switch (base.type) {
        .unary_add,
        .binary_add,
        .expand_add,
        .reduce_sum,
        .reduce_avg,
        => {
            writeSource(source, offset, "+", .{});
        },
        .unary_subtract,
        .binary_subtract,
        .expand_subtract,
        => {
            writeSource(source, offset, "-", .{});
        },
        .unary_multiply,
        .binary_multiply,
        .expand_multiply,
        => {
            writeSource(source, offset, "*", .{});
        },
        .unary_divide,
        .binary_divide,
        .expand_divide,
        => {
            writeSource(source, offset, "/", .{});
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
            writeSource(source, offset, ",", .{});
        },
        .unary_random => {
            todo(@src());
        },
        .unary_sign => {
            todo(@src());
        },
    }
}
fn writeAssignPostfix(source: *[]u8, offset: *usize, base: Base) void {
    switch (base.type) {
        .unary_add,
        .unary_subtract,
        .unary_multiply,
        .unary_divide,
        .unary_max,
        .unary_min,
        => {
            writeSource(source, offset, "((float){d}))", .{base.u_var});
        },
        .unary_square,
        => {
            writeSource(source, offset, ",2)", .{});
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
            writeSource(source, offset, ")", .{});
        },
    }
}
fn writeAssignOutBase(source: *[]u8, offset: *usize, base: Base, kernel_loop_idx: u32, inlined_idx_curr: u32, offset_out: u32) void {
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
            writeSource(source, offset, "{s}[{s}_{}_{}+{}]", .{
                base.out.name(),
                base.out.name(),
                kernel_loop_idx,
                inlined_idx_curr,
                offset_out,
            });
        },
        .unary_set => {
            writeSource(source, offset, "((float){d})", .{base.u_var});
        },
        .binary_set,
        .expand_set,
        => {},
    }
}
fn writeAssignInBase(
    source: *[]u8,
    offset: *usize,
    base: Base,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    offset_in: u32,
) void {
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
            writeSource(source, offset, "{s}[{s}_{}_{}+{}]", .{
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
    source: *[]u8,
    offset: *usize,
    inlined: Inlined,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    a: u32,
    z: u32,
    y: u32,
    x: u32,
) void {
    const inlined_idx_actual: u32 = inlined_idx_curr - 1;

    writeAssignPrefix(source, offset, inlined.base[inlined_idx_actual]);

    const base_relevant: Base = inlined.base[inlined_idx_actual];
    const a_out: u32 = if (base_relevant.type.isReduce()) 0 else a;
    const z_out: u32 = if (base_relevant.type.isReduce()) 0 else z;
    const y_out: u32 = if (base_relevant.type.isReduce()) 0 else y;
    const x_out: u32 = if (base_relevant.type.isReduce()) 0 else x;
    if (inlined.out[inlined_idx_actual]) |inlined_out| {
        writeAssignOut(source, offset, inlined, kernel_loop_idx, inlined_out + 1, a_out, z_out, y_out, x_out);
    } else {
        const offset_out: u32 = base_relevant.out.at(a_out, z_out, y_out, x_out) - base_relevant.out.offset;
        writeAssignOutBase(source, offset, base_relevant, kernel_loop_idx, inlined_idx_curr, offset_out);
    }

    writeAssignMidfix(source, offset, inlined.base[inlined_idx_actual]);

    const a_in: u32 = if (base_relevant.type.isExpand()) 0 else a;
    const z_in: u32 = if (base_relevant.type.isExpand()) 0 else z;
    const y_in: u32 = if (base_relevant.type.isExpand()) 0 else y;
    const x_in: u32 = if (base_relevant.type.isExpand()) 0 else x;
    if (inlined.in[inlined_idx_actual]) |inlined_in| {
        writeAssignIn(source, offset, inlined, kernel_loop_idx, inlined_in + 1, a_in, z_in, y_in, x_in);
    } else {
        if (!base_relevant.type.isUnary()) {
            const offset_in: u32 = base_relevant.in.at(a_in, z_in, y_in, x_in) - base_relevant.in.offset;
            writeAssignInBase(source, offset, base_relevant, kernel_loop_idx, inlined_idx_curr, offset_in);
        }
    }

    writeAssignPostfix(source, offset, inlined.base[inlined_idx_actual]);
}
// inlined_idx_curr is 0 if it is the base assign and actual inlined index + 1 otherwise
fn writeAssignIn(
    source: *[]u8,
    offset: *usize,
    inlined: Inlined,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    a: u32,
    z: u32,
    y: u32,
    x: u32,
) void {
    const inlined_idx_actual: u32 = inlined_idx_curr - 1;

    writeAssignPrefix(source, offset, inlined.base[inlined_idx_actual]);

    const base_relevant: Base = inlined.base[inlined_idx_actual];
    const a_out: u32 = if (base_relevant.type.isReduce()) 0 else a;
    const z_out: u32 = if (base_relevant.type.isReduce()) 0 else z;
    const y_out: u32 = if (base_relevant.type.isReduce()) 0 else y;
    const x_out: u32 = if (base_relevant.type.isReduce()) 0 else x;
    if (inlined.out[inlined_idx_actual]) |inlined_out| {
        writeAssignOut(source, offset, inlined, kernel_loop_idx, inlined_out + 1, a_out, z_out, y_out, x_out);
    } else {
        const offset_out: u32 = base_relevant.out.at(a_out, z_out, y_out, x_out) - base_relevant.out.offset;
        writeAssignOutBase(source, offset, base_relevant, kernel_loop_idx, inlined_idx_curr, offset_out);
    }

    writeAssignMidfix(source, offset, inlined.base[inlined_idx_actual]);

    const a_in: u32 = if (base_relevant.type.isExpand()) 0 else a;
    const z_in: u32 = if (base_relevant.type.isExpand()) 0 else z;
    const y_in: u32 = if (base_relevant.type.isExpand()) 0 else y;
    const x_in: u32 = if (base_relevant.type.isExpand()) 0 else x;
    if (inlined.in[inlined_idx_actual]) |inlined_in| {
        writeAssignIn(source, offset, inlined, kernel_loop_idx, inlined_in + 1, a_in, z_in, y_in, x_in);
    } else {
        if (!base_relevant.type.isUnary()) {
            const offset_in: u32 = base_relevant.in.at(a_in, z_in, y_in, x_in) - base_relevant.in.offset;
            writeAssignInBase(source, offset, base_relevant, kernel_loop_idx, inlined_idx_curr, offset_in);
        }
    }

    writeAssignPostfix(source, offset, inlined.base[inlined_idx_actual]);
}
fn writeAssign(source: *[]u8, offset: *usize, assign: Assign, kernel_loop_idx: u32) void {
    if (assign.base.type.isReduce()) {
        writeSource(source, offset, "{s}[{s}_{}_{}+{}]={s};\n", .{
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

                    writeSource(source, offset, "{s}[{s}_{}_{}+{}] = ", .{
                        assign.base.out.name(),
                        assign.base.out.name(),
                        kernel_loop_idx,
                        0,
                        offset_out,
                    });

                    writeAssignPrefix(source, offset, assign.base);

                    if (assign.inlined) |inlined| {
                        if (inlined.out_root) |inlined_out| {
                            writeAssignOut(source, offset, inlined, kernel_loop_idx, inlined_out + 1, a_out, z_out, y_out, x_out);
                        } else {
                            writeAssignOutBase(source, offset, assign.base, kernel_loop_idx, 0, offset_out);
                        }
                    } else {
                        writeAssignOutBase(source, offset, assign.base, kernel_loop_idx, 0, offset_out);
                    }

                    writeAssignMidfix(source, offset, assign.base);

                    const a_in: u32 = if (assign.base.type.isExpand()) 0 else a;
                    const z_in: u32 = if (assign.base.type.isExpand()) 0 else z;
                    const y_in: u32 = if (assign.base.type.isExpand()) 0 else y;
                    const x_in: u32 = if (assign.base.type.isExpand()) 0 else x;

                    if (assign.inlined) |inlined| {
                        if (inlined.in_root) |inlined_in| {
                            writeAssignIn(source, offset, inlined, kernel_loop_idx, inlined_in + 1, a_in, z_in, y_in, x_in);
                        } else {
                            if (!assign.base.type.isUnary()) {
                                const offset_in: u32 = if (assign.base.type.isExpand()) 0 else assign.base.in.at(a_in, z_in, y_in, x_in) - assign.base.in.offset;
                                writeAssignInBase(source, offset, assign.base, kernel_loop_idx, 0, offset_in);
                            }
                        }
                    } else {
                        if (!assign.base.type.isUnary()) {
                            const offset_in: u32 = if (assign.base.type.isExpand()) 0 else assign.base.in.at(a_in, z_in, y_in, x_in) - assign.base.in.offset;
                            writeAssignInBase(source, offset, assign.base, kernel_loop_idx, 0, offset_in);
                        }
                    }

                    writeAssignPostfix(source, offset, assign.base);

                    writeSource(source, offset, ";\n", .{});
                }
            }
        }
    }

    if (assign.base.type == .reduce_avg) {
        writeSource(source, offset, "{s}[{s}_{}_{}+{}]/={d};\n", .{
            assign.base.out.name(),
            assign.base.out.name(),
            kernel_loop_idx,
            0,
            0,
            @as(f64, @floatFromInt(a_size * z_size * y_size * x_size)),
        });
    }
}
fn writeAssignOutBaseBlock(source: *[]u8, offset: *usize, base: Base, kernel_loop_idx: u32, inlined_idx_curr: u32, kernel_block_idx: u32) void {
    if (base.type.isReduce()) {
        assert(inlined_idx_curr == 0);
        assert(kernel_block_idx == 0);
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
            writeSource(source, offset, "{s}[{s}_{}_{}+{s}_{}_{}_{}]", .{
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
            writeSource(source, offset, "((float){d})", .{base.u_var});
        },
        .binary_set,
        .expand_set,
        => {},
    }
}
fn writeAssignInBaseBlock(
    source: *[]u8,
    offset: *usize,
    base: Base,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    kernel_block_idx: u32,
) void {
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
            writeSource(source, offset, "{s}[{s}_{}_{}+{s}_{}_{}_{}]", .{
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
fn writeIndicesBlock(source: *[]u8, offset: *usize, assign: Assign, kernel_loop_idx: u32, kernel_block_idx: u32) void {
    const inlined_num: u32 = 1 + (if (assign.inlined) |inlined| inlined.inlined_num else 0);
    var inlined_idx: u32 = 0;
    while (inlined_idx < inlined_num) : (inlined_idx += 1) {
        const base: Base = if (inlined_idx == 0) assign.base else assign.inlined.?.base[inlined_idx - 1];
        if (base.type.isReduce()) {
            writeSource(
                source,
                offset,
                "int {s}_{}_{}_{} = {};\n",
                .{ base.out.name(), kernel_loop_idx, inlined_idx, kernel_block_idx, 0 },
            );
        } else {
            writeSource(
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
        if (!base.type.isUnary()) {
            if (base.type.isExpand()) {
                writeSource(
                    source,
                    offset,
                    "int {s}_{}_{}_{} = 0;\n",
                    .{ base.in.name(), kernel_loop_idx, inlined_idx, kernel_block_idx },
                );
            } else {
                writeSource(
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
    source: *[]u8,
    offset: *usize,
    inlined: Inlined,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    kernel_block_idx: u32,
) void {
    const inlined_idx_actual: u32 = inlined_idx_curr - 1;

    writeAssignPrefix(source, offset, inlined.base[inlined_idx_actual]);

    const base_relevant: Base = inlined.base[inlined_idx_actual];
    if (inlined.out[inlined_idx_actual]) |inlined_out| {
        writeAssignOutBlock(source, offset, inlined, kernel_loop_idx, inlined_out + 1, kernel_block_idx);
    } else {
        writeAssignOutBaseBlock(source, offset, base_relevant, kernel_loop_idx, inlined_idx_curr, kernel_block_idx);
    }

    writeAssignMidfix(source, offset, inlined.base[inlined_idx_actual]);

    if (inlined.in[inlined_idx_actual]) |inlined_in| {
        writeAssignInBlock(source, offset, inlined, kernel_loop_idx, inlined_in + 1, kernel_block_idx);
    } else {
        if (!base_relevant.type.isUnary()) {
            writeAssignInBaseBlock(source, offset, base_relevant, kernel_loop_idx, inlined_idx_curr, kernel_block_idx);
        }
    }

    writeAssignPostfix(source, offset, inlined.base[inlined_idx_actual]);
}
// inlined_idx_curr is 0 if it is the base assign and actual inlined index + 1 otherwise
fn writeAssignInBlock(
    source: *[]u8,
    offset: *usize,
    inlined: Inlined,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    kernel_block_idx: u32,
) void {
    const inlined_idx_actual: u32 = inlined_idx_curr - 1;

    writeAssignPrefix(source, offset, inlined.base[inlined_idx_actual]);

    const base_relevant: Base = inlined.base[inlined_idx_actual];
    if (inlined.out[inlined_idx_actual]) |inlined_out| {
        writeAssignOutBlock(source, offset, inlined, kernel_loop_idx, inlined_out + 1, kernel_block_idx);
    } else {
        writeAssignOutBaseBlock(source, offset, base_relevant, kernel_loop_idx, inlined_idx_curr, kernel_block_idx);
    }

    writeAssignMidfix(source, offset, inlined.base[inlined_idx_actual]);

    if (inlined.in[inlined_idx_actual]) |inlined_in| {
        writeAssignInBlock(source, offset, inlined, kernel_loop_idx, inlined_in + 1, kernel_block_idx);
    } else {
        if (!base_relevant.type.isUnary()) {
            writeAssignInBaseBlock(source, offset, base_relevant, kernel_loop_idx, inlined_idx_curr, kernel_block_idx);
        }
    }

    writeAssignPostfix(source, offset, inlined.base[inlined_idx_actual]);
}
fn writeAssignBlock(source: *[]u8, offset: *usize, assign: Assign, kernel_loop_idx: u32, kernel_block_idx: u32) void {
    assert(!assign.base.type.isReduce());

    writeSource(source, offset, "{s}[{s}_{}_{}+{s}_{}_{}_{}] = ", .{
        assign.base.out.name(),
        assign.base.out.name(),
        kernel_loop_idx,
        0,
        assign.base.out.name(),
        kernel_loop_idx,
        0,
        kernel_block_idx,
    });

    writeAssignPrefix(source, offset, assign.base);

    if (assign.inlined) |inlined| {
        if (inlined.out_root) |inlined_out| {
            writeAssignOutBlock(source, offset, inlined, kernel_loop_idx, inlined_out + 1, kernel_block_idx);
        } else {
            writeAssignOutBaseBlock(source, offset, assign.base, kernel_loop_idx, 0, kernel_block_idx);
        }
    } else {
        writeAssignOutBaseBlock(source, offset, assign.base, kernel_loop_idx, 0, kernel_block_idx);
    }

    writeAssignMidfix(source, offset, assign.base);

    if (assign.inlined) |inlined| {
        if (inlined.in_root) |inlined_in| {
            writeAssignInBlock(source, offset, inlined, kernel_loop_idx, inlined_in + 1, kernel_block_idx);
        } else {
            if (!assign.base.type.isUnary()) {
                writeAssignInBaseBlock(source, offset, assign.base, kernel_loop_idx, 0, kernel_block_idx);
            }
        }
    } else {
        if (!assign.base.type.isUnary()) {
            writeAssignInBaseBlock(source, offset, assign.base, kernel_loop_idx, 0, kernel_block_idx);
        }
    }

    writeAssignPostfix(source, offset, assign.base);

    writeSource(source, offset, ";\n", .{});
}
/// This one does not return the buffers included. That only gets added if there is no inlined
/// thing for that specific buffer
fn assignCompileBytesBase(base: Base) u32 {
    return switch (base.type) {
        .unary_add => "(+)".len,
        .unary_subtract => "(-)".len,
        .unary_multiply => "(*)".len,
        .unary_divide => "(/)".len,
        .unary_exp => "exp()".len,
        .unary_log => "log()".len,
        .unary_square => "pow(,2)".len,
        .unary_sqrt => "sqrt()".len,
        .unary_reciprocal => "(1/)".len,
        .unary_max => "fmax(,)".len,
        .unary_min => "fmin(,)".len,
        .unary_set => "()".len,
        .unary_random => unreachable, // $TODO: Do this
        .unary_tanh => "tanh()".len,
        .unary_absolute => "fabs()".len,
        .unary_sign => unreachable, // $TODO: Do this
        .binary_add => "(+)".len,
        .binary_subtract => "(-)".len,
        .binary_multiply => "(*)".len,
        .binary_divide => "(/)".len,
        .binary_max => "fmax(,)".len,
        .binary_min => "fmin(,)".len,
        .binary_set => "()".len,
        .expand_add => "(+)".len,
        .expand_subtract => "(-)".len,
        .expand_multiply => "(*)".len,
        .expand_divide => "(/)".len,
        .expand_max => "fmax(,)".len,
        .expand_min => "fmin(,)".len,
        .expand_set => "()".len,
        .reduce_sum => "()".len,
        .reduce_max => "()".len,
        .reduce_avg => "()".len,
        .reduce_min => "()".len,
    };
}
pub fn assignCompileBytes(_: *anyopaque, assign: Assign, name_len_max: u32, args: Args, size_global: u32, _: u32) u32 {
    const boilerplate_kernel: []const u8 =
        "__kernel void () {\n" ++
        "const int gid = get_global_id(0);\n" ++
        "int id;\n" ++
        "id = gid;\n" ++
        "}\n";
    const boilerplate_argument: []const u8 = ", global float *";
    const length_header: u32 = @intCast(boilerplate_kernel.len + name_len_max + args.arg_num * (boilerplate_argument.len + buffer_name_size));
    const boilerplate_conditional: []const u8 = "if(gid < ) {\n" ++ "}\n";
    const leangth_conditional: u32 = if (assign.base.repeats % size_global > 0)
        @intCast(boilerplate_conditional.len + length_int_max)
    else
        0;
    const boilerplate_index: []const u8 = "int ___ = (id%)/*+(id%)/*+(id%)/*+(id%)/*+;\n"; // 4 * 3 + 1 + 3 int_width_max + 1 * buffer_name_size
    const length_index: u32 = @intCast(boilerplate_index.len + 16 * length_int_max + buffer_name_size *
        (1 + if (assign.inlined) |i| i.inlined_num else 0));
    const boilerplate_assign_outer: []const u8 = "[___+] += ();\n"; // buffer_name_size + 4 * int_width_max. += is in case of reduce
    const length_assign_root: u32 = (if (assign.inlined) |inlined|
        @as(u32, @intFromBool(inlined.in_root == null)) + @as(u32, @intFromBool(inlined.out_root == null))
    else
        2) * (buffer_name_size + 4 * length_int_max);
    const length_assign_outer: u32 = @intCast(boilerplate_assign_outer.len + buffer_name_size + 4 * length_int_max +
        length_assign_root);
    var length_assign_inner: u32 = assignCompileBytesBase(assign.base);
    if (assign.inlined) |inlined| {
        for (0..inlined.inlined_num) |inlined_idx| {
            length_assign_inner += assignCompileBytesBase(inlined.base[inlined_idx]) +
                (buffer_name_size + 4 * length_int_max) *
                    (@as(u32, @intFromBool(inlined.in[inlined_idx] == null)) +
                        @as(u32, @intFromBool(inlined.out[inlined_idx] == null)));
        }
    }

    const a_size: u32 = if (assign.base.type.isReduce()) assign.base.in.a_size else assign.base.out.a_size;
    const z_size: u32 = if (assign.base.type.isReduce()) assign.base.in.z_size else assign.base.out.z_size;
    const y_size: u32 = if (assign.base.type.isReduce()) assign.base.in.y_size else assign.base.out.y_size;
    const x_size: u32 = if (assign.base.type.isReduce()) assign.base.in.x_size else assign.base.out.x_size;

    const length_assign: u32 = a_size * z_size * y_size * x_size * (length_assign_outer + length_assign_inner);

    return length_header + leangth_conditional + length_index + length_assign;
}
pub fn assignCompile(
    this: *anyopaque,
    source: *[]u8,
    offset: *usize,
    assign: Assign,
    name: []const u8,
    args: Args,
    size_global: u32,
    size_local: u32,
) void {
    assert(assign.base.repeats > 0);
    assert(size_global > 0);
    assert(size_local > 0);
    assert(size_global % size_local == 0);

    const state: *RuntimeCl = @ptrCast(@alignCast(this));
    assert(offset.* + assignCompileBytes(state, assign, @intCast(name.len), args, size_global, size_local) < source.len);

    writeSource(source, offset, "__kernel void {s}(", .{name});
    assert(args.arg_mem.len == args.arg_id.len);
    for (0..args.arg_num) |arg_idx| {
        const arg_name: [buffer_name_size]u8 = nameFromId(args.arg_id[arg_idx]);
        if (arg_idx == 0) {
            writeSource(source, offset, "__global float *{s}", .{arg_name});
        } else {
            writeSource(source, offset, ", __global float *{s}", .{arg_name});
        }
    }

    writeSource(source, offset, ") {{\n" ++
        "const int gid = get_global_id(0);\n" ++
        "int id;\n", .{});

    // $TODO Merge these cases in to 1 case. Should not really be that difficult
    if (assign.split) {
        const a_size: u32 = if (assign.base.type.isReduce()) assign.base.in.a_size else assign.base.out.a_size;
        const z_size: u32 = if (assign.base.type.isReduce()) assign.base.in.z_size else assign.base.out.z_size;
        const y_size: u32 = if (assign.base.type.isReduce()) assign.base.in.y_size else assign.base.out.y_size;
        const x_size: u32 = if (assign.base.type.isReduce()) assign.base.in.x_size else assign.base.out.x_size;
        const size: u32 = a_size * z_size * y_size * x_size;
        const size_with_repeats: u32 = size * assign.base.repeats;

        const kernel_block_leftover: u32 = size_with_repeats % size_global;
        const kernel_block_size: u32 = std.math.divCeil(u32, size_with_repeats, size_global) catch unreachable;

        assert(kernel_block_size <= size);

        var kernel_block_idx: u32 = 0;
        while (kernel_block_idx < kernel_block_size) : (kernel_block_idx += 1) {
            if (kernel_block_idx == kernel_block_size - 1 and kernel_block_leftover != 0) {
                writeSource(source, offset, "if(gid<{}) {{\n", .{kernel_block_leftover});
            }

            writeSource(source, offset, "id = (gid+{})/{};\n", .{ size_global * kernel_block_idx, size });
            writeIndices(source, offset, assign, kernel_block_idx);
            writeSource(source, offset, "id = gid+{};\n", .{size_global * kernel_block_idx});
            writeIndicesBlock(source, offset, assign, kernel_block_idx, kernel_block_idx);
            writeAssignBlock(source, offset, assign, kernel_block_idx, kernel_block_idx);

            if (kernel_block_idx == kernel_block_size - 1 and kernel_block_leftover != 0) {
                writeSource(source, offset, "}}\n", .{});
            }
        }
    } else {
        const kernel_loop_leftover: u32 = (assign.base.repeats) % size_global;
        const kernel_loop_num: u32 = @divFloor(assign.base.repeats, size_global) + @intFromBool(kernel_loop_leftover != 0);

        var kernel_loop_idx: u32 = 0;
        while (kernel_loop_idx < kernel_loop_num) : (kernel_loop_idx += 1) {
            writeSource(source, offset, "id = gid+{};\n", .{size_global * kernel_loop_idx});

            if (kernel_loop_idx == kernel_loop_num - 1 and kernel_loop_leftover != 0) {
                writeSource(source, offset, "if(gid < {}) {{\n", .{kernel_loop_leftover});
            }

            writeIndices(source, offset, assign, kernel_loop_idx);
            writeAssign(source, offset, assign, kernel_loop_idx);

            if (kernel_loop_idx == kernel_loop_num - 1 and kernel_loop_leftover != 0) {
                writeSource(source, offset, "}}\n", .{});
            }
        }
    }

    writeSource(source, offset, "}}\n", .{});
}
