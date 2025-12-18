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
        try source.ensureTotalCapacity(gpa, source.capacity * 2);
        try source.printBounded(fmt, args);
    };
    try source.ensureUnusedCapacity(gpa, source_padding);
}
fn writeIndex(
    gpa: Allocator,
    source: *ArrayList(u8),
    view: ViewOffset,
    buffer: Buffer,
    block_idx: u32,
    inlined_idx: ?u32,
) Allocator.Error!void {
    var first_dim: bool = true;
    try writeSource(gpa, source, "int {s}_{}_{?} = ", .{ buffer.name(), block_idx, inlined_idx });
    if (view.repeat_stride.a != 0) {
        try writeSource(gpa, source, "(id%{})/{}*{}", //
            .{ view.repeat_reset.a, view.repeat_wait.a, view.repeat_stride.a * view.stride.a });
        first_dim = false;
    }
    if (view.repeat_stride.z != 0) {
        if (first_dim) {
            try writeSource(gpa, source, "(id%{})/{}*{}", //
                .{ view.repeat_reset.z, view.repeat_wait.z, view.repeat_stride.z * view.stride.z });
        } else {
            try writeSource(gpa, source, "+(id%{})/{}*{}", //
                .{ view.repeat_reset.z, view.repeat_wait.z, view.repeat_stride.z * view.stride.z });
        }
        first_dim = false;
    }
    if (view.repeat_stride.y != 0) {
        if (first_dim) {
            try writeSource(gpa, source, "(id%{})/{}*{}", //
                .{ view.repeat_reset.y, view.repeat_wait.y, view.repeat_stride.y * view.stride.y });
        } else {
            try writeSource(gpa, source, "+(id%{})/{}*{}", //
                .{ view.repeat_reset.y, view.repeat_wait.y, view.repeat_stride.y * view.stride.y });
        }
        first_dim = false;
    }
    if (view.repeat_stride.x != 0) {
        if (first_dim) {
            try writeSource(gpa, source, "(id%{})/{}*{}", //
                .{ view.repeat_reset.x, view.repeat_wait.x, view.repeat_stride.x });
        } else {
            try writeSource(gpa, source, "+(id%{})/{}*{}", //
                .{ view.repeat_reset.x, view.repeat_wait.x, view.repeat_stride.x });
        }
        first_dim = false;
    }
    if (first_dim) {
        try writeSource(gpa, source, "0;\n", .{});
    } else {
        try writeSource(gpa, source, ";\n", .{});
    }
}
fn writeIndicesInlined(
    gpa: Allocator,
    source: *ArrayList(u8),
    inlined: Inlined,
    size: Vec4,
    block_idx: u32,
    inlined_idx: u32,
) Allocator.Error!void {
    const base_relevant: Base = inlined.base[inlined_idx];
    const out_size: Vec4 = if (base_relevant.kind.isReduce())
        .{ .a = 1, .z = 1, .y = 1, .x = 1 }
    else
        size;
    if (inlined.out[inlined_idx]) |inlined_out| {
        try writeIndicesInlined(gpa, source, inlined, out_size, block_idx, inlined_out);
    } else {
        try writeIndex(gpa, source, base_relevant.out_view, base_relevant.out, block_idx, inlined_idx);
    }
    if (!base_relevant.kind.isUnary()) {
        const in_size: Vec4 = if (base_relevant.kind.isExpand())
            .{ .a = 1, .z = 1, .y = 1, .x = 1 }
        else
            size;
        if (inlined.in[inlined_idx]) |inlined_in| {
            try writeIndicesInlined(gpa, source, inlined, in_size, block_idx, inlined_in);
        } else {
            try writeIndex(gpa, source, base_relevant.in_view, base_relevant.in, block_idx, inlined_idx);
        }
    }
}
fn writeIndices(
    gpa: Allocator,
    source: *ArrayList(u8),
    assign: Assign,
    block_idx: u32,
) Allocator.Error!void {
    const out_size: Vec4 = if (assign.base.kind.isReduce())
        .{ .a = 1, .z = 1, .y = 1, .x = 1 }
    else
        assign.size;
    try writeIndex(gpa, source, assign.base.out_view, assign.base.out, block_idx, null);
    if (assign.inlined.out_root) |inlined_out| {
        try writeIndicesInlined(gpa, source, assign.inlined, out_size, block_idx, inlined_out);
    }
    if (!assign.base.kind.isUnary()) {
        const in_size: Vec4 = if (assign.base.kind.isExpand())
            .{ .a = 1, .z = 1, .y = 1, .x = 1 }
        else
            assign.size;
        if (assign.inlined.in_root) |inlined_in| {
            try writeIndicesInlined(gpa, source, assign.inlined, in_size, block_idx, inlined_in);
        } else {
            try writeIndex(gpa, source, assign.base.in_view, assign.base.in, block_idx, null);
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
    gpa: Allocator,
    source: *ArrayList(u8),
    base: Base,
    block_idx: u32,
    inlined_idx: ?u32,
    size: Vec4,
    offset: Vec4,
) Allocator.Error!void {
    if (base.kind.isReduce()) {
        assert(inlined_idx == 0);
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
            try writeSource(gpa, source, "{s}[{s}_{}_{?}+{}]", .{
                base.out.name(),
                base.out.name(),
                block_idx,
                inlined_idx,
                base.out_view.viewAtRepeat(size, 0).at(offset),
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
fn writeAssignInBase(
    gpa: Allocator,
    source: *ArrayList(u8),
    base: Base,
    block_idx: u32,
    inlined_idx: ?u32,
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
            try writeSource(gpa, source, "{s}[{s}_{}_{?}+{}]", .{
                base.in.name(),
                base.in.name(),
                block_idx,
                inlined_idx,
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
fn writeAssignInlined(
    gpa: Allocator,
    source: *ArrayList(u8),
    inlined: Inlined,
    block_idx: u32,
    inlined_idx: u32,
    size: Vec4,
    offset: Vec4,
) Allocator.Error!void {
    try writeAssignPrefix(gpa, source, inlined.base[inlined_idx]);

    const base_relevant: Base = inlined.base[inlined_idx];
    const offset_out: Vec4 = if (base_relevant.kind.isReduce())
        .{ .a = 0, .z = 0, .y = 0, .x = 0 }
    else
        offset;
    if (inlined.out[inlined_idx]) |inlined_out| {
        try writeAssignInlined(gpa, source, inlined, block_idx, inlined_out, size, offset_out);
    } else {
        try writeAssignOutBase(gpa, source, base_relevant, block_idx, inlined_idx, size, offset_out);
    }

    try writeAssignMidfix(gpa, source, inlined.base[inlined_idx]);

    const offset_in: Vec4 = if (base_relevant.kind.isExpand())
        .{ .a = 0, .z = 0, .y = 0, .x = 0 }
    else
        offset;
    if (inlined.in[inlined_idx]) |inlined_in| {
        try writeAssignInlined(gpa, source, inlined, block_idx, inlined_in, size, offset_in);
    } else {
        if (!base_relevant.kind.isUnary()) {
            try writeAssignInBase(gpa, source, base_relevant, block_idx, inlined_idx, size, offset_in);
        }
    }

    try writeAssignPostfix(gpa, source, inlined.base[inlined_idx]);
}
fn writeAssign(gpa: Allocator, source: *ArrayList(u8), assign: Assign, block_idx: u32) Allocator.Error!void {
    assert(assign.split.block_split.equal(.splat(1)));
    try writeSource(gpa, source, "for(int a=0;a<{};a++) {{\n", .{assign.size.a});
    try writeSource(gpa, source, "for(int z=0;z<{};z++) {{\n", .{assign.size.z});
    try writeSource(gpa, source, "for(int y=0;y<{};y++) {{\n", .{assign.size.y});
    try writeSource(gpa, source, "for(int x=0;x<{};x++) {{\n", .{assign.size.x});
    try writeSource(gpa, source, "{s}[{s}_{}_{?}] = ", .{
        assign.base.out.name(),
        assign.base.out.name(),
        block_idx,
        @as(?u32, null),
    });
    try writeSource(gpa, source, ";\n", .{});
    try writeSource(gpa, source, "}}\n", .{});
    try writeSource(gpa, source, "}}\n", .{});
    try writeSource(gpa, source, "}}\n", .{});
    try writeSource(gpa, source, "}}\n", .{});
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

    const block_num: u32 = assign.split.block_split.productOfElements() * assign.repeats;
    const block_per_kernel_common: u32 = std.math.divFloor(u32, block_num, size_global) catch unreachable;
    const block_per_kernel_leftover: u32 = block_num % size_global;
    // Probably not very optimal, but at least it communicates intent clearly
    const block_per_kernel_total: u32 = if (block_per_kernel_leftover == 0)
        block_per_kernel_common
    else
        block_per_kernel_common + 1;

    var block_idx: u32 = 0;
    while (block_idx < block_per_kernel_total) : (block_idx += 1) {
        if (block_idx == block_per_kernel_common) {
            try writeSource(gpa, source, "if(gid<{}) {{\n", .{block_per_kernel_leftover});
        }
        try writeSource(gpa, source, "id = gid + {};\n", .{size_global * block_idx});
        try writeIndices(gpa, source, assign, block_idx);
        try writeAssign(gpa, source, assign, block_idx);
        if (block_idx == block_per_kernel_common) {
            try writeSource(gpa, source, "}}\n", .{});
        }
    }

    // if (assign.split) {
    //     const size_with_repeats: u32 = assign.size.productOfElements() * assign.repeats;
    //
    //     const kernel_block_leftover: u32 = size_with_repeats % size_global;
    //     const kernel_block_size: u32 = std.math.divCeil(u32, size_with_repeats, size_global) catch unreachable;
    //
    //     assert(kernel_block_size <= assign.size.productOfElements());
    //
    //     var kernel_block_idx: u32 = 0;
    //     while (kernel_block_idx < kernel_block_size) : (kernel_block_idx += 1) {
    //         if (kernel_block_idx == kernel_block_size - 1 and kernel_block_leftover != 0) {
    //             try writeSource(gpa, source, "if(gid<{}) {{\n", .{kernel_block_leftover});
    //         }
    //
    //         try writeSource(gpa, source, "id = (gid+{})/{};\n", .{ size_global * kernel_block_idx, assign.size.productOfElements() });
    //         try writeIndices(gpa, source, assign, kernel_block_idx);
    //         try writeSource(gpa, source, "id = gid+{};\n", .{size_global * kernel_block_idx});
    //         try writeIndicesBlock(gpa, source, assign, kernel_block_idx, kernel_block_idx);
    //         try writeAssignBlock(gpa, source, assign, kernel_block_idx, kernel_block_idx);
    //
    //         if (kernel_block_idx == kernel_block_size - 1 and kernel_block_leftover != 0) {
    //             try writeSource(gpa, source, "}}\n", .{});
    //         }
    //     }
    // } else {
    //     const kernel_loop_leftover: u32 = (assign.repeats) % size_global;
    //     const kernel_loop_num: u32 = @divFloor(assign.repeats, size_global) + @intFromBool(kernel_loop_leftover != 0);
    //
    //     var kernel_loop_idx: u32 = 0;
    //     while (kernel_loop_idx < kernel_loop_num) : (kernel_loop_idx += 1) {
    //         try writeSource(gpa, source, "id = gid+{};\n", .{size_global * kernel_loop_idx});
    //
    //         if (kernel_loop_idx == kernel_loop_num - 1 and kernel_loop_leftover != 0) {
    //             try writeSource(gpa, source, "if(gid < {}) {{\n", .{kernel_loop_leftover});
    //         }
    //
    //         try writeIndices(gpa, source, assign, kernel_loop_idx);
    //         try writeAssign(gpa, source, assign, kernel_loop_idx);
    //
    //         if (kernel_loop_idx == kernel_loop_num - 1 and kernel_loop_leftover != 0) {
    //             try writeSource(gpa, source, "}}\n", .{});
    //         }
    //     }
    // }

    try writeSource(gpa, source, "}}\n", .{});

    util.log.print("{s}\n", .{source.items});
    std.posix.exit(0);
}
