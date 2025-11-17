const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const bufPrint = std.fmt.bufPrint;
const ArrayList = std.ArrayList;

const Program = @import("../Program.zig");
const Args = Program.Args;
const Pir = @import("../Pir.zig");
const Assign = Pir.Assign;
const Base = Pir.Base;
const ViewOffset = Pir.ViewOffset;
const Inlined = Pir.Inlined;
const Buffer = @import("../../Buffer.zig");
const Vec4 = Buffer.Vec4;
const buffer_name_size = Buffer.buffer_name_size;
const Linearized = @import("../../Linearized.zig");
const Op = Linearized.Op;
const util = @import("../../util.zig");
const Runtime = @import("Runtime.zig");
const RuntimeCl = Runtime.RuntimeCl;

const source_padding: u32 = 4 * 1024;
/// Write format string to buffer and ensure there is at least `padding` bytes left
fn writeSource(gpa: Allocator, source: *ArrayList(u8), comptime fmt: []const u8, args: anytype) Allocator.Error!void {
    source.printBounded(fmt, args) catch {
        try source.ensureUnusedCapacity(gpa, source_padding);
        try source.printBounded(fmt, args);
    };
    try source.ensureUnusedCapacity(gpa, source_padding);
}
fn writeIndices(
    gpa: Allocator,
    source: *ArrayList(u8),
    assign: Assign,
    kernel_loop_idx: usize,
) Allocator.Error!void {
    const inlined_num: u32 = 1 + assign.inlined.num;
    var inlined_idx: u32 = 0;
    while (inlined_idx < inlined_num) : (inlined_idx += 1) {
        const base: Base = if (inlined_idx == 0) assign.base else assign.inlined.base[inlined_idx - 1];
        const out_view: ViewOffset = base.out_view;
        const in_view: ViewOffset = base.in_view;
        try writeSource(
            gpa,
            source,
            "int {s}_{}_{} = (id%{})/{}*{}+(id%{})/{}*{}+(id%{})/{}*{}+(id%{})/{}*{}+{};\n",
            .{
                base.out.name(), kernel_loop_idx, inlined_idx, //
                out_view.repeat_reset.a, out_view.repeat_wait.a, out_view.repeat_stride.a * out_view.stride.a, //
                out_view.repeat_reset.z, out_view.repeat_wait.z, out_view.repeat_stride.z * out_view.stride.z, //
                out_view.repeat_reset.y, out_view.repeat_wait.y, out_view.repeat_stride.y * out_view.stride.y, //
                out_view.repeat_reset.x, out_view.repeat_wait.x, out_view.repeat_stride.x * out_view.stride.x, //
                out_view.offset,
            },
        );
        if (!base.kind.isUnary()) {
            try writeSource(
                gpa,
                source,
                "int {s}_{}_{} = (id%{})/{}*{}+(id%{})/{}*{}+(id%{})/{}*{}+(id%{})/{}*{}+{};\n",
                .{
                    base.in.name(), kernel_loop_idx, inlined_idx, //
                    in_view.repeat_reset.a, in_view.repeat_wait.a, in_view.repeat_stride.a * out_view.stride.a, //
                    in_view.repeat_reset.z, in_view.repeat_wait.z, in_view.repeat_stride.z * out_view.stride.z, //
                    in_view.repeat_reset.y, in_view.repeat_wait.y, in_view.repeat_stride.y * out_view.stride.y, //
                    in_view.repeat_reset.x, in_view.repeat_wait.x, in_view.repeat_stride.x * out_view.stride.x, //
                    in_view.offset,
                },
            );
        }
    }
}
fn writeAssignPrefix(gpa: Allocator, source: *ArrayList(u8), base: Base) Allocator.Error!void {
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
            try writeSource(gpa, source, "(", .{});
        },
        .unary_exp => {
            try writeSource(gpa, source, "exp(", .{});
        },
        .unary_log => {
            try writeSource(gpa, source, "log(", .{});
        },
        .unary_square => {
            try writeSource(gpa, source, "pow(", .{});
        },
        .unary_sqrt => {
            try writeSource(gpa, source, "sqrt(", .{});
        },
        .unary_reciprocal => {
            try writeSource(gpa, source, "(1/", .{});
        },
        .unary_max,
        .binary_max,
        .expand_max,
        .reduce_max,
        => {
            try writeSource(gpa, source, "fmax(", .{});
        },
        .unary_min,
        .binary_min,
        .expand_min,
        .reduce_min,
        => {
            try writeSource(gpa, source, "fmin(", .{});
        },
        .unary_random => {
            util.todo(@src());
        },
        .unary_tanh => {
            try writeSource(gpa, source, "tanh(", .{});
        },
        .unary_absolute => {
            try writeSource(gpa, source, "fabs(", .{});
        },
        .unary_sign => {
            util.todo(@src());
        },
    }
}
fn writeAssignMidfix(gpa: Allocator, source: *ArrayList(u8), base: Base) Allocator.Error!void {
    switch (base.kind) {
        .unary_add,
        .binary_add,
        .expand_add,
        .reduce_sum,
        .reduce_avg,
        => {
            try writeSource(gpa, source, "+", .{});
        },
        .unary_subtract,
        .binary_subtract,
        .expand_subtract,
        => {
            try writeSource(gpa, source, "-", .{});
        },
        .unary_multiply,
        .binary_multiply,
        .expand_multiply,
        => {
            try writeSource(gpa, source, "*", .{});
        },
        .unary_divide,
        .binary_divide,
        .expand_divide,
        => {
            try writeSource(gpa, source, "/", .{});
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
            try writeSource(gpa, source, ",", .{});
        },
        .unary_random => {
            util.todo(@src());
        },
        .unary_sign => {
            util.todo(@src());
        },
    }
}
fn writeAssignPostfix(gpa: Allocator, source: *ArrayList(u8), base: Base) Allocator.Error!void {
    switch (base.kind) {
        .unary_add,
        .unary_subtract,
        .unary_multiply,
        .unary_divide,
        .unary_max,
        .unary_min,
        => {
            try writeSource(gpa, source, "((float){d}))", .{base.u_var});
        },
        .unary_square,
        => {
            try writeSource(gpa, source, ",2)", .{});
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
            try writeSource(gpa, source, ")", .{});
        },
    }
}
fn writeAssignOutBase(
    ga: Allocator,
    source: *ArrayList(u8),
    base: Base,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    size: Vec4,
    offset: Vec4,
) Allocator.Error!void {
    if (base.kind.isReduce()) {
        assert(inlined_idx_curr == 0);
        assert(offset.equal(.{ .a = 0, .z = 0, .y = 0, .x = 0 }));
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
            try writeSource(ga, source, "{s}[{s}_{}_{}+{}]", .{
                base.out.name(),
                base.out.name(),
                kernel_loop_idx,
                inlined_idx_curr,
                base.out_view.viewAtRepeat(size, 0).at(offset),
            });
        },
        .unary_set => {
            try writeSource(ga, source, "((float){d})", .{base.u_var});
        },
        .binary_set,
        .expand_set,
        => {},
    }
}
fn writeAssignInBase(
    gpa: Allocator,
    source: *ArrayList(u8),
    base: Base,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    size: Vec4,
    offset: Vec4,
) Allocator.Error!void {
    if (base.kind.isExpand()) {
        assert(offset.equal(.{ .a = 0, .z = 0, .y = 0, .x = 0 }));
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
            try writeSource(gpa, source, "{s}[{s}_{}_{}+{}]", .{
                base.in.name(),
                base.in.name(),
                kernel_loop_idx,
                inlined_idx_curr,
                base.in_view.viewAtRepeat(size, 0).at(offset),
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
    source: *ArrayList(u8),
    inlined: Inlined,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    size: Vec4,
    offset: Vec4,
) Allocator.Error!void {
    const inlined_idx_actual: u32 = inlined_idx_curr - 1;

    try writeAssignPrefix(gpa, source, inlined.base[inlined_idx_actual]);

    const base_relevant: Base = inlined.base[inlined_idx_actual];
    const offset_out: Vec4 = if (base_relevant.kind.isReduce())
        .{ .a = 0, .z = 0, .y = 0, .x = 0 }
    else
        offset;
    if (inlined.out[inlined_idx_actual]) |inlined_out| {
        try writeAssignOut(gpa, source, inlined, kernel_loop_idx, inlined_out + 1, size, offset_out);
    } else {
        try writeAssignOutBase(gpa, source, base_relevant, kernel_loop_idx, inlined_idx_curr, size, offset_out);
    }

    try writeAssignMidfix(gpa, source, inlined.base[inlined_idx_actual]);

    const offset_in: Vec4 = if (base_relevant.kind.isExpand())
        .{ .a = 0, .z = 0, .y = 0, .x = 0 }
    else
        offset;
    if (inlined.in[inlined_idx_actual]) |inlined_in| {
        try writeAssignIn(gpa, source, inlined, kernel_loop_idx, inlined_in + 1, size, offset_in);
    } else {
        if (!base_relevant.kind.isUnary()) {
            try writeAssignInBase(gpa, source, base_relevant, kernel_loop_idx, inlined_idx_curr, size, offset_in);
        }
    }

    try writeAssignPostfix(gpa, source, inlined.base[inlined_idx_actual]);
}
// inlined_idx_curr is 0 if it is the base assign and actual inlined index + 1 otherwise
fn writeAssignIn(
    gpa: Allocator,
    source: *ArrayList(u8),
    inlined: Inlined,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    size: Vec4,
    offset: Vec4,
) Allocator.Error!void {
    const inlined_idx_actual: u32 = inlined_idx_curr - 1;

    try writeAssignPrefix(gpa, source, inlined.base[inlined_idx_actual]);

    const base_relevant: Base = inlined.base[inlined_idx_actual];
    const offset_out: Vec4 = if (base_relevant.kind.isReduce())
        .{ .a = 0, .z = 0, .y = 0, .x = 0 }
    else
        offset;
    if (inlined.out[inlined_idx_actual]) |inlined_out| {
        try writeAssignOut(gpa, source, inlined, kernel_loop_idx, inlined_out + 1, size, offset_out);
    } else {
        try writeAssignOutBase(gpa, source, base_relevant, kernel_loop_idx, inlined_idx_curr, size, offset_out);
    }

    try writeAssignMidfix(gpa, source, inlined.base[inlined_idx_actual]);

    const offset_in: Vec4 = if (base_relevant.kind.isExpand())
        .{ .a = 0, .z = 0, .y = 0, .x = 0 }
    else
        offset;
    if (inlined.in[inlined_idx_actual]) |inlined_in| {
        try writeAssignIn(gpa, source, inlined, kernel_loop_idx, inlined_in + 1, size, offset_in);
    } else {
        if (!base_relevant.kind.isUnary()) {
            try writeAssignInBase(gpa, source, base_relevant, kernel_loop_idx, inlined_idx_curr, size, offset_in);
        }
    }

    try writeAssignPostfix(gpa, source, inlined.base[inlined_idx_actual]);
}
fn writeAssign(gpa: Allocator, source: *ArrayList(u8), assign: Assign, kernel_loop_idx: u32) Allocator.Error!void {
    if (assign.base.kind.isReduce()) {
        try writeSource(gpa, source, "{s}[{s}_{}_{}+{}]={s};\n", .{
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

    const size: Vec4 = assign.size;

    var a: u32 = 0;
    while (a < size.a) : (a += 1) {
        var z: u32 = 0;
        while (z < size.z) : (z += 1) {
            var y: u32 = 0;
            while (y < size.y) : (y += 1) {
                var x: u32 = 0;
                while (x < size.x) : (x += 1) {
                    const offset_out: Vec4 = if (assign.base.kind.isReduce())
                        .{ .a = 0, .z = 0, .y = 0, .x = 0 }
                    else
                        .{ .a = a, .z = z, .y = y, .x = x };

                    try writeSource(gpa, source, "{s}[{s}_{}_{}+{}] = ", .{
                        assign.base.out.name(),
                        assign.base.out.name(),
                        kernel_loop_idx,
                        0,
                        assign.base.out_view.viewAtRepeat(assign.size, 0).at(offset_out),
                    });

                    try writeAssignPrefix(gpa, source, assign.base);

                    if (assign.inlined.out_root) |inlined_out| {
                        try writeAssignOut(gpa, source, assign.inlined, kernel_loop_idx, inlined_out + 1, size, offset_out);
                    } else {
                        try writeAssignOutBase(gpa, source, assign.base, kernel_loop_idx, 0, size, offset_out);
                    }

                    try writeAssignMidfix(gpa, source, assign.base);

                    const offset_in: Vec4 = if (assign.base.kind.isExpand())
                        .{ .a = 0, .z = 0, .y = 0, .x = 0 }
                    else
                        .{ .a = a, .z = z, .y = y, .x = x };

                    if (assign.inlined.in_root) |inlined_in| {
                        try writeAssignIn(gpa, source, assign.inlined, kernel_loop_idx, inlined_in + 1, size, offset_in);
                    } else {
                        if (!assign.base.kind.isUnary()) {
                            try writeAssignInBase(gpa, source, assign.base, kernel_loop_idx, 0, size, offset_in);
                        }
                    }

                    try writeAssignPostfix(gpa, source, assign.base);

                    try writeSource(gpa, source, ";\n", .{});
                }
            }
        }
    }

    if (assign.base.kind == .reduce_avg) {
        try writeSource(gpa, source, "{s}[{s}_{}_{}+{}]/={d};\n", .{
            assign.base.out.name(),
            assign.base.out.name(),
            kernel_loop_idx,
            0,
            0,
            @as(f64, @floatFromInt(assign.size.productOfElements())),
        });
    }
}
fn writeAssignOutBaseBlock(
    gpa: Allocator,
    source: *ArrayList(u8),
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
            try writeSource(gpa, source, "{s}[{s}_{}_{}+{s}_{}_{}_{}]", .{
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
            try writeSource(gpa, source, "((float){d})", .{base.u_var});
        },
        .binary_set,
        .expand_set,
        => {},
    }
}
fn writeAssignInBaseBlock(
    gpa: Allocator,
    source: *ArrayList(u8),
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
            try writeSource(gpa, source, "{s}[{s}_{}_{}+{s}_{}_{}_{}]", .{
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
    source: *ArrayList(u8),
    assign: Assign,
    kernel_loop_idx: u32,
    kernel_block_idx: u32,
) Allocator.Error!void {
    const inlined_num: u32 = 1 + assign.inlined.num;
    var inlined_idx: u32 = 0;
    while (inlined_idx < inlined_num) : (inlined_idx += 1) {
        const base: Base = if (inlined_idx == 0) assign.base else assign.inlined.base[inlined_idx - 1];
        const out_view: ViewOffset = base.out_view;
        if (base.kind.isReduce()) {
            try writeSource(
                gpa,
                source,
                "int {s}_{}_{}_{} = 0;\n",
                .{ base.out.name(), kernel_loop_idx, inlined_idx, kernel_block_idx },
            );
        } else {
            const size: Vec4 = assign.size;
            try writeSource(
                gpa,
                source,
                "int {s}_{}_{}_{} = (id%{})/{}*{}+(id%{})/{}*{}+(id%{})/{}*{}+(id%{})/{}*{};\n",
                .{
                    base.out.name(), kernel_loop_idx, inlined_idx, kernel_block_idx, //
                    size.a * size.z * size.y * size.x, size.z * size.y * size.x, out_view.stride.a, //
                    size.z * size.y * size.x,          size.y * size.x,          out_view.stride.z,
                    size.y * size.x,                   size.x,                   out_view.stride.y,
                    size.x,                            1,                        out_view.stride.x,
                },
            );
        }
        if (!base.kind.isUnary()) {
            const in_view: ViewOffset = base.in_view;
            if (base.kind.isExpand()) {
                try writeSource(
                    gpa,
                    source,
                    "int {s}_{}_{}_{} = 0;\n",
                    .{ base.in.name(), kernel_loop_idx, inlined_idx, kernel_block_idx },
                );
            } else {
                const size: Vec4 = assign.size;
                try writeSource(
                    gpa,
                    source,
                    "int {s}_{}_{}_{} = (id%{})/{}*{}+(id%{})/{}*{}+(id%{})/{}*{}+(id%{})/{}*{};\n",
                    .{
                        base.in.name(), kernel_loop_idx, inlined_idx, kernel_block_idx, //
                        size.a * size.z * size.y * size.x, size.z * size.y * size.x, in_view.stride.a, //
                        size.z * size.y * size.x,          size.y * size.x,          in_view.stride.z,
                        size.y * size.x,                   size.x,                   in_view.stride.y,
                        size.x,                            1,                        in_view.stride.x,
                    },
                );
            }
        }
    }
}
// inlined_idx_curr is 0 if it is the base assign and actual inlined index + 1 otherwise
fn writeAssignOutBlock(
    gpa: Allocator,
    source: *ArrayList(u8),
    inlined: Inlined,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    kernel_block_idx: u32,
) Allocator.Error!void {
    const inlined_idx_actual: u32 = inlined_idx_curr - 1;

    try writeAssignPrefix(gpa, source, inlined.base[inlined_idx_actual]);

    const base_relevant: Base = inlined.base[inlined_idx_actual];
    if (inlined.out[inlined_idx_actual]) |inlined_out| {
        try writeAssignOutBlock(gpa, source, inlined, kernel_loop_idx, inlined_out + 1, kernel_block_idx);
    } else {
        try writeAssignOutBaseBlock(gpa, source, base_relevant, kernel_loop_idx, inlined_idx_curr, kernel_block_idx);
    }

    try writeAssignMidfix(gpa, source, inlined.base[inlined_idx_actual]);

    if (inlined.in[inlined_idx_actual]) |inlined_in| {
        try writeAssignInBlock(gpa, source, inlined, kernel_loop_idx, inlined_in + 1, kernel_block_idx);
    } else {
        if (!base_relevant.kind.isUnary()) {
            try writeAssignInBaseBlock(gpa, source, base_relevant, kernel_loop_idx, inlined_idx_curr, kernel_block_idx);
        }
    }

    try writeAssignPostfix(gpa, source, inlined.base[inlined_idx_actual]);
}
// inlined_idx_curr is 0 if it is the base assign and actual inlined index + 1 otherwise
fn writeAssignInBlock(
    gpa: Allocator,
    source: *ArrayList(u8),
    inlined: Inlined,
    kernel_loop_idx: u32,
    inlined_idx_curr: u32,
    kernel_block_idx: u32,
) Allocator.Error!void {
    const inlined_idx_actual: u32 = inlined_idx_curr - 1;

    try writeAssignPrefix(gpa, source, inlined.base[inlined_idx_actual]);

    const base_relevant: Base = inlined.base[inlined_idx_actual];
    if (inlined.out[inlined_idx_actual]) |inlined_out| {
        try writeAssignOutBlock(gpa, source, inlined, kernel_loop_idx, inlined_out + 1, kernel_block_idx);
    } else {
        try writeAssignOutBaseBlock(gpa, source, base_relevant, kernel_loop_idx, inlined_idx_curr, kernel_block_idx);
    }

    try writeAssignMidfix(gpa, source, inlined.base[inlined_idx_actual]);

    if (inlined.in[inlined_idx_actual]) |inlined_in| {
        try writeAssignInBlock(gpa, source, inlined, kernel_loop_idx, inlined_in + 1, kernel_block_idx);
    } else {
        if (!base_relevant.kind.isUnary()) {
            try writeAssignInBaseBlock(gpa, source, base_relevant, kernel_loop_idx, inlined_idx_curr, kernel_block_idx);
        }
    }

    try writeAssignPostfix(gpa, source, inlined.base[inlined_idx_actual]);
}
fn writeAssignBlock(
    gpa: Allocator,
    source: *ArrayList(u8),
    assign: Assign,
    kernel_loop_idx: u32,
    kernel_block_idx: u32,
) Allocator.Error!void {
    assert(!assign.base.kind.isReduce());

    try writeSource(gpa, source, "{s}[{s}_{}_{}+{s}_{}_{}_{}] = ", .{
        assign.base.out.name(),
        assign.base.out.name(),
        kernel_loop_idx,
        0,
        assign.base.out.name(),
        kernel_loop_idx,
        0,
        kernel_block_idx,
    });

    try writeAssignPrefix(gpa, source, assign.base);

    if (assign.inlined.out_root) |inlined_out| {
        try writeAssignOutBlock(gpa, source, assign.inlined, kernel_loop_idx, inlined_out + 1, kernel_block_idx);
    } else {
        try writeAssignOutBaseBlock(gpa, source, assign.base, kernel_loop_idx, 0, kernel_block_idx);
    }

    try writeAssignMidfix(gpa, source, assign.base);

    if (assign.inlined.in_root) |inlined_in| {
        try writeAssignInBlock(gpa, source, assign.inlined, kernel_loop_idx, inlined_in + 1, kernel_block_idx);
    } else {
        if (!assign.base.kind.isUnary()) {
            try writeAssignInBaseBlock(gpa, source, assign.base, kernel_loop_idx, 0, kernel_block_idx);
        }
    }

    try writeAssignPostfix(gpa, source, assign.base);

    try writeSource(gpa, source, ";\n", .{});
}

pub fn assignCompile(
    _: *anyopaque,
    gpa: Allocator,
    source: *ArrayList(u8),
    assign: Assign,
    name: []const u8,
    args: Args,
    size_global: u32,
    size_local: u32,
) Allocator.Error!void {
    assert(assign.repeats > 0);
    assert(size_global > 0);
    assert(size_local > 0);
    assert(size_global % size_local == 0);

    try writeSource(gpa, source, "__kernel void {s}(", .{name});
    for (args.arg_buffer, 0..) |arg_buffer, arg_idx| {
        const arg_name: [buffer_name_size]u8 = arg_buffer.name();
        if (arg_idx == 0) {
            try writeSource(gpa, source, "__global float *{s}", .{arg_name});
        } else {
            try writeSource(gpa, source, ", __global float *{s}", .{arg_name});
        }
    }

    try writeSource(gpa, source, ") {{\n" ++
        "const int gid = get_global_id(0);\n" ++
        "int id;\n", .{});

    // $TODO Merge these cases in to 1 case. Should not really be that difficult
    if (assign.split) {
        const size_with_repeats: u32 = assign.size.productOfElements() * assign.repeats;

        const kernel_block_leftover: u32 = size_with_repeats % size_global;
        const kernel_block_size: u32 = std.math.divCeil(u32, size_with_repeats, size_global) catch unreachable;

        assert(kernel_block_size <= assign.size.productOfElements());

        var kernel_block_idx: u32 = 0;
        while (kernel_block_idx < kernel_block_size) : (kernel_block_idx += 1) {
            if (kernel_block_idx == kernel_block_size - 1 and kernel_block_leftover != 0) {
                try writeSource(gpa, source, "if(gid<{}) {{\n", .{kernel_block_leftover});
            }

            try writeSource(gpa, source, "id = (gid+{})/{};\n", .{ size_global * kernel_block_idx, assign.size.productOfElements() });
            try writeIndices(gpa, source, assign, kernel_block_idx);
            try writeSource(gpa, source, "id = gid+{};\n", .{size_global * kernel_block_idx});
            try writeIndicesBlock(gpa, source, assign, kernel_block_idx, kernel_block_idx);
            try writeAssignBlock(gpa, source, assign, kernel_block_idx, kernel_block_idx);

            if (kernel_block_idx == kernel_block_size - 1 and kernel_block_leftover != 0) {
                try writeSource(gpa, source, "}}\n", .{});
            }
        }
    } else {
        const kernel_loop_leftover: u32 = (assign.repeats) % size_global;
        const kernel_loop_num: u32 = @divFloor(assign.repeats, size_global) + @intFromBool(kernel_loop_leftover != 0);

        var kernel_loop_idx: u32 = 0;
        while (kernel_loop_idx < kernel_loop_num) : (kernel_loop_idx += 1) {
            try writeSource(gpa, source, "id = gid+{};\n", .{size_global * kernel_loop_idx});

            if (kernel_loop_idx == kernel_loop_num - 1 and kernel_loop_leftover != 0) {
                try writeSource(gpa, source, "if(gid < {}) {{\n", .{kernel_loop_leftover});
            }

            try writeIndices(gpa, source, assign, kernel_loop_idx);
            try writeAssign(gpa, source, assign, kernel_loop_idx);

            if (kernel_loop_idx == kernel_loop_num - 1 and kernel_loop_leftover != 0) {
                try writeSource(gpa, source, "}}\n", .{});
            }
        }
    }

    try writeSource(gpa, source, "}}\n", .{});
}
