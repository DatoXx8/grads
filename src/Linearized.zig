const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const Buffer = @import("Buffer.zig");
const Id = Buffer.Id;
const Vec4 = Buffer.Vec4;
const View = Buffer.View;
const Data = Buffer.Data;

const Program = @import("compiler/Program.zig");
const Memory = Program.Memory;
const Runtime = @import("compiler/runtimes/Runtime.zig");
const util = @import("util.zig");

pub const Linearized = @This();

pub const ViewDual = struct {
    size: Vec4,
    out_stride: Vec4,
    in_stride: Vec4,
    out_offset: u32,
    in_offset: u32,
    pub fn viewOut(view_dual: ViewDual) View {
        return .{
            .size = view_dual.size,
            .stride = view_dual.out_stride,
            .offset = view_dual.out_offset,
        };
    }
    pub fn viewOutReduce(view_dual: ViewDual) View {
        return .{
            .size = .{ .a = 1, .z = 1, .y = 1, .x = 1 },
            .stride = .{ .a = 1, .z = 1, .y = 1, .x = 1 },
            .offset = view_dual.out_offset,
        };
    }
    pub fn viewIn(view_dual: ViewDual) View {
        return .{
            .size = view_dual.size,
            .stride = view_dual.in_stride,
            .offset = view_dual.in_offset,
        };
    }
    pub fn viewInExpand(view_dual: ViewDual) View {
        return .{
            .size = .{ .a = 1, .z = 1, .y = 1, .x = 1 },
            .stride = .{ .a = 1, .z = 1, .y = 1, .x = 1 },
            .offset = view_dual.in_offset,
        };
    }
};
pub const Op = struct {
    pub const Kind = enum(u8) {
        unary_add,
        unary_subtract,
        unary_multiply,
        unary_divide,
        unary_exp,
        unary_log,
        unary_square,
        unary_sqrt,
        unary_reciprocal,
        unary_max,
        unary_min,
        unary_set,
        unary_random,
        unary_tanh,
        unary_absolute,
        unary_sign,
        binary_add,
        binary_subtract,
        binary_multiply,
        binary_divide,
        binary_max,
        binary_min,
        binary_set,
        expand_add,
        expand_subtract,
        expand_multiply,
        expand_divide,
        expand_max,
        expand_min,
        expand_set,
        reduce_sum,
        reduce_max,
        reduce_avg,
        reduce_min,
        pub inline fn isUnary(kind: Kind) bool {
            return switch (kind) {
                .unary_add, .unary_subtract, .unary_multiply, .unary_divide => true,
                .unary_exp, .unary_log, .unary_square, .unary_sqrt, .unary_reciprocal => true,
                .unary_max, .unary_min, .unary_set, .unary_random => true,
                .unary_tanh, .unary_absolute, .unary_sign => true,
                .binary_add, .binary_subtract, .binary_multiply, .binary_divide => false,
                .binary_max, .binary_min, .binary_set => false,
                .expand_add, .expand_subtract, .expand_multiply, .expand_divide => false,
                .expand_max, .expand_min, .expand_set => false,
                .reduce_sum, .reduce_max, .reduce_avg, .reduce_min => false,
            };
        }
        pub inline fn isBinary(kind: Kind) bool {
            return switch (kind) {
                .unary_add, .unary_subtract, .unary_multiply, .unary_divide => false,
                .unary_exp, .unary_log, .unary_square, .unary_sqrt, .unary_reciprocal => false,
                .unary_max, .unary_min, .unary_set, .unary_random => false,
                .unary_tanh, .unary_absolute, .unary_sign => false,
                .binary_add, .binary_subtract, .binary_multiply, .binary_divide => true,
                .binary_max, .binary_min, .binary_set => true,
                .expand_add, .expand_subtract, .expand_multiply, .expand_divide => false,
                .expand_max, .expand_min, .expand_set => false,
                .reduce_sum, .reduce_max, .reduce_avg, .reduce_min => false,
            };
        }
        pub inline fn isExpand(kind: Kind) bool {
            return switch (kind) {
                .unary_add, .unary_subtract, .unary_multiply, .unary_divide => false,
                .unary_exp, .unary_log, .unary_square, .unary_sqrt, .unary_reciprocal => false,
                .unary_max, .unary_min, .unary_set, .unary_random => false,
                .unary_tanh, .unary_absolute, .unary_sign => false,
                .binary_add, .binary_subtract, .binary_multiply, .binary_divide => false,
                .binary_max, .binary_min, .binary_set => false,
                .expand_add, .expand_subtract, .expand_multiply, .expand_divide => true,
                .expand_max, .expand_min, .expand_set => true,
                .reduce_sum, .reduce_max, .reduce_avg, .reduce_min => false,
            };
        }
        pub inline fn isReduce(kind: Kind) bool {
            return switch (kind) {
                .unary_add, .unary_subtract, .unary_multiply, .unary_divide => false,
                .unary_exp, .unary_log, .unary_square, .unary_sqrt, .unary_reciprocal => false,
                .unary_max, .unary_min, .unary_set, .unary_random => false,
                .unary_tanh, .unary_absolute, .unary_sign => false,
                .binary_add, .binary_subtract, .binary_multiply, .binary_divide => false,
                .binary_max, .binary_min, .binary_set => false,
                .expand_add, .expand_subtract, .expand_multiply, .expand_divide => false,
                .expand_max, .expand_min, .expand_set => false,
                .reduce_sum, .reduce_max, .reduce_avg, .reduce_min => true,
            };
        }
        /// Not a great name. Essentially this returns wether the result is independant of what was in `this.out` before
        pub fn overwrites(kind: Kind) bool {
            return switch (kind) {
                .unary_add, .unary_subtract, .unary_multiply, .unary_divide => false,
                .unary_exp, .unary_log, .unary_square, .unary_sqrt => false,
                .unary_reciprocal, .unary_max, .unary_min => false,
                .unary_set, .unary_random => true,
                .unary_tanh, .unary_absolute, .unary_sign => false,
                .binary_add, .binary_subtract, .binary_multiply, .binary_divide => false,
                .binary_max, .binary_min => false,
                .binary_set => true,
                .expand_add, .expand_subtract, .expand_multiply, .expand_divide => false,
                .expand_max, .expand_min => false,
                .expand_set => true,
                .reduce_sum, .reduce_max, .reduce_avg, .reduce_min => true,
            };
        }
    };

    kind: Kind,
    u_var: f32,
    out: Buffer,
    in: Buffer,
    view_dual: ViewDual,

    /// Don't use this for more than one-off operations because this is pretty slow
    pub fn realize(op: Op) void {
        const out_data: Data = op.out.data().*;
        const out_view: View = op.view_dual.viewOut();
        const in_data: Data = op.in.data().*;
        const in_view: View = op.view_dual.viewIn();

        const offset_zero: Vec4 = .{ .a = 0, .z = 0, .y = 0, .x = 0 };

        switch (op.kind) {
            .reduce_sum => {
                out_data.values[out_view.at(offset_zero)] = 0;
            },
            .reduce_max => {
                out_data.values[out_view.at(offset_zero)] = -std.math.inf(f32);
            },
            .reduce_min => {
                out_data.values[out_view.at(offset_zero)] = std.math.inf(f32);
            },
            .reduce_avg => {
                out_data.values[out_view.at(offset_zero)] = 0;
            },
            else => {},
        }

        // $FIXME When implementing unary_random this needs to be changed
        var rng: ?std.Random.Pcg = if (op.kind == .unary_random)
            std.Random.Pcg.init(@as(u32, @bitCast(op.u_var)))
        else
            null;

        // Just to be clear I know that putting the loop outside might make it slower because you have to go through the switch statement every time, but
        // the branch predictor is likely to have an extremely easy time predicting the branches since it's the same every single time.
        // Which should mean that as long as your CPU even has a branch predictor it should cause very little to no performance impact.
        // I measured it by running some arbitrary ops and there was no measurable difference
        const size: Vec4 = op.view_dual.size;
        var a: u32 = 0;
        while (a < size.a) : (a += 1) {
            var z: u32 = 0;
            while (z < size.z) : (z += 1) {
                var y: u32 = 0;
                while (y < size.y) : (y += 1) {
                    var x: u32 = 0;
                    while (x < size.x) : (x += 1) {
                        const offset: Vec4 = .{ .a = a, .z = z, .y = y, .x = x };
                        switch (op.kind) {
                            .unary_add => {
                                out_data.values[out_view.at(offset)] += op.u_var;
                            },
                            .unary_subtract => {
                                out_data.values[out_view.at(offset)] -= op.u_var;
                            },
                            .unary_multiply => {
                                out_data.values[out_view.at(offset)] *= op.u_var;
                            },
                            .unary_divide => {
                                out_data.values[out_view.at(offset)] /= op.u_var;
                            },
                            .unary_exp => {
                                out_data.values[out_view.at(offset)] =
                                    @exp(out_data.values[out_view.at(offset)]);
                            },
                            .unary_log => {
                                out_data.values[out_view.at(offset)] =
                                    @log(out_data.values[out_view.at(offset)]);
                            },
                            .unary_square => {
                                out_data.values[out_view.at(offset)] *=
                                    out_data.values[out_view.at(offset)];
                            },
                            .unary_sqrt => {
                                out_data.values[out_view.at(offset)] =
                                    @sqrt(out_data.values[out_view.at(offset)]);
                            },
                            .unary_reciprocal => {
                                out_data.values[out_view.at(offset)] =
                                    1 / out_data.values[out_view.at(offset)];
                            },
                            .unary_max => {
                                out_data.values[out_view.at(offset)] =
                                    @max(out_data.values[out_view.at(offset)], op.u_var);
                            },
                            .unary_min => {
                                out_data.values[out_view.at(offset)] =
                                    @min(out_data.values[out_view.at(offset)], op.u_var);
                            },
                            .unary_set => {
                                out_data.values[out_view.at(offset)] = op.u_var;
                            },
                            .unary_random => {
                                // $TODO Make my own PCG implementation that can do SIMD
                                out_data.values[out_view.at(offset)] =
                                    rng.?.random().floatNorm(f32);
                            },
                            .unary_tanh => {
                                out_data.values[out_view.at(offset)] =
                                    std.math.tanh(out_data.values[out_view.at(offset)]);
                            },
                            .unary_absolute => {
                                out_data.values[out_view.at(offset)] =
                                    @abs(out_data.values[out_view.at(offset)]);
                            },
                            .unary_sign => {
                                if (out_data.values[out_view.at(offset)] > 0) {
                                    out_data.values[out_view.at(offset)] = 1;
                                } else if (out_data.values[out_view.at(offset)] < 0) {
                                    out_data.values[out_view.at(offset)] = -1;
                                } else {
                                    out_data.values[out_view.at(offset)] = 0;
                                }
                                // $fn signVector(comptime T: kind, vector @Vector(4, T)) @Vector(4, i32) {
                                //     const zero = @splat(4, @as(T, 0));
                                //     const one = @splat(4, @as(T, 1));
                                //     const neg_one = @splat(4, @as(T, -1));
                                //
                                //     const cmp_lt = @Vector(4, bool)(vector < zero);
                                //     const cmp_gt = @Vector(4, bool)(vector > zero);
                                //
                                //     return @select(cmp_lt, neg_one, @select(cmp_gt, one, @splat(4, @as(i32, 0))));
                                // }
                            },
                            .binary_add => {
                                out_data.values[out_view.at(offset)] +=
                                    in_data.values[in_view.at(offset)];
                            },
                            .binary_subtract => {
                                out_data.values[out_view.at(offset)] -=
                                    in_data.values[in_view.at(offset)];
                            },
                            .binary_multiply => {
                                out_data.values[out_view.at(offset)] *=
                                    in_data.values[in_view.at(offset)];
                            },
                            .binary_divide => {
                                out_data.values[out_view.at(offset)] /=
                                    in_data.values[in_view.at(offset)];
                            },
                            .binary_max => {
                                out_data.values[out_view.at(offset)] =
                                    @max(out_data.values[out_view.at(offset)], //
                                    in_data.values[in_view.at(offset)]);
                            },
                            .binary_min => {
                                out_data.values[out_view.at(offset)] =
                                    @min(out_data.values[out_view.at(offset)], //
                                    in_data.values[in_view.at(offset)]);
                            },
                            .binary_set => {
                                out_data.values[out_view.at(offset)] =
                                    in_data.values[in_view.at(offset)];
                            },
                            .expand_add => {
                                out_data.values[out_view.at(offset)] +=
                                    in_data.values[in_view.at(offset_zero)];
                            },
                            .expand_subtract => {
                                out_data.values[out_view.at(offset)] -=
                                    in_data.values[in_view.at(offset_zero)];
                            },
                            .expand_multiply => {
                                out_data.values[out_view.at(offset)] *=
                                    in_data.values[in_view.at(offset_zero)];
                            },
                            .expand_divide => {
                                out_data.values[out_view.at(offset)] /=
                                    in_data.values[in_view.at(offset_zero)];
                            },
                            .expand_max => {
                                out_data.values[out_view.at(offset)] =
                                    @max(out_data.values[out_view.at(offset)], //
                                    in_data.values[in_view.at(offset_zero)]);
                            },
                            .expand_min => {
                                out_data.values[out_view.at(offset)] =
                                    @min(out_data.values[out_view.at(offset)], //
                                    in_data.values[in_view.at(offset_zero)]);
                            },
                            .expand_set => {
                                out_data.values[out_view.at(offset)] =
                                    in_data.values[in_view.at(offset_zero)];
                            },
                            .reduce_sum => {
                                out_data.values[out_view.at(offset_zero)] +=
                                    in_data.values[in_view.at(offset)];
                            },
                            .reduce_max => {
                                out_data.values[out_view.at(offset_zero)] =
                                    @max(out_data.values[out_view.at(offset_zero)], //
                                    in_data.values[in_view.at(offset)]);
                            },
                            .reduce_min => {
                                out_data.values[out_view.at(offset_zero)] =
                                    @min(out_data.values[out_view.at(offset_zero)], //
                                    in_data.values[in_view.at(offset)]);
                            },
                            .reduce_avg => {
                                out_data.values[out_view.at(offset_zero)] +=
                                    in_data.values[in_view.at(offset)];
                            },
                        }
                    }
                }
            }
        }
        if (op.kind == .reduce_avg) {
            out_data.values[out_view.at(offset_zero)] /=
                @as(f32, @floatFromInt(in_view.size.productOfElements()));
        }
    }
    pub fn print(op: Op, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            util.log.print("{s}{s} ", .{ " " ** (padding + offset), text });
        } else {
            util.log.print("{s}", .{" " ** (padding + offset)});
        }
        if (op.kind.isUnary()) {
            util.log.print("{s} ({}, {}, {}, {}) [{} = ({}, {}, {}, {})] \"{s}\" {d}\n", .{
                @tagName(op.kind),
                op.view_dual.size.a,
                op.view_dual.size.z,
                op.view_dual.size.y,
                op.view_dual.size.x,
                op.view_dual.out_offset,
                op.view_dual.viewOut().aOffset(),
                op.view_dual.viewOut().zOffset(),
                op.view_dual.viewOut().yOffset(),
                op.view_dual.viewOut().xOffset(),
                Buffer.name(op.out),
                op.u_var,
            });
        } else {
            util.log.print("{s} ({}, {}, {}, {}) [{} = ({}, {}, {}, {})] \"{s}\" [{} = ({}, {}, {}, {})] \"{s}\"\n", .{
                @tagName(op.kind),
                op.view_dual.size.a,
                op.view_dual.size.z,
                op.view_dual.size.y,
                op.view_dual.size.x,
                op.view_dual.out_offset,
                op.view_dual.viewOut().aOffset(),
                op.view_dual.viewOut().zOffset(),
                op.view_dual.viewOut().yOffset(),
                op.view_dual.viewOut().xOffset(),
                Buffer.name(op.out),
                op.view_dual.in_offset,
                op.view_dual.viewIn().aOffset(),
                op.view_dual.viewIn().zOffset(),
                op.view_dual.viewIn().yOffset(),
                op.view_dual.viewIn().xOffset(),
                Buffer.name(op.in),
            });
        }
    }
};

op: []Op,
num: u32,
pub fn alloc(gpa: Allocator, capacity: u32) !Linearized {
    return .{
        .num = 0,
        .op = try gpa.alloc(Op, capacity),
    };
}
pub fn capacityEnsure(linearized: *Linearized, gpa: Allocator, capacity: u32) !void {
    if (linearized.op.len < linearized.num + capacity) {
        linearized.op = try gpa.realloc(linearized.op, linearized.num + capacity);
    }
}
pub fn free(linearized: *Linearized, gpa: Allocator) void {
    linearized.op_num = 0;
    gpa.free(linearized.op);
}
pub fn clear(linearized: *Linearized) void {
    linearized.num = 0;
}
pub fn run(linearized: *Linearized) void {
    var op_idx: u32 = 0;
    while (op_idx < linearized.num) : (op_idx += 1) {
        linearized.op[op_idx].realize();
    }
}
pub fn realize(linearized: *Linearized) void {
    linearized.run();
    linearized.clear();
}
pub fn append(linearized: *Linearized, op: Op) void {
    assert(linearized.num < linearized.op.len);
    linearized.op[linearized.num] = op;
    linearized.num += 1;
}
pub fn concat(linearized: *Linearized, source: *Linearized) void {
    assert(linearized.num + source.num <= linearized.op.len);
    var op_idx: u32 = 0;
    while (op_idx < source.num) : (op_idx += 1) {
        linearized.op[linearized.num + op_idx] = source.op[op_idx];
    }
    linearized.num += source.num;
    source.clear();
}
pub fn print(linearized: Linearized, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
    if (name) |text| {
        util.log.print("{s}Linearized = {s}\n", .{ " " ** offset, text });
    } else {
        util.log.print("{s}Linearized\n", .{" " ** offset});
    }
    if (linearized.num == 0) {
        util.log.print("{s}[] => empty\n", .{" " ** (offset + padding)});
    } else {
        for (0..linearized.num) |op_idx| {
            util.log.print("{s}[{}] => ", .{ " " ** (offset + padding), op_idx });
            linearized.op[op_idx].print(0, 0, null);
        }
    }
}
pub fn unaryAdd(linearized: *Linearized, buffer: Buffer, u_var: f32) void {
    assert(!std.math.isNan(u_var));
    assert(!std.math.isInf(u_var));
    linearized.append(.{
        .view_dual = .{
            .out_offset = buffer.view().offset,
            .out_stride = buffer.view().stride,
            .in_offset = buffer.view().offset,
            .in_stride = buffer.view().stride,
            .size = buffer.view().size,
        },
        .in = buffer,
        .out = buffer,
        .kind = .unary_add,
        .u_var = u_var,
    });
}
pub fn unarySubtract(linearized: *Linearized, buffer: Buffer, u_var: f32) void {
    assert(!std.math.isNan(u_var));
    assert(!std.math.isInf(u_var));
    linearized.append(.{
        .view_dual = .{
            .out_offset = buffer.view().offset,
            .out_stride = buffer.view().stride,
            .in_offset = buffer.view().offset,
            .in_stride = buffer.view().stride,
            .size = buffer.view().size,
        },
        .in = buffer,
        .out = buffer,
        .kind = .unary_subtract,
        .u_var = u_var,
    });
}
pub fn unaryMultiply(linearized: *Linearized, buffer: Buffer, u_var: f32) void {
    assert(!std.math.isNan(u_var));
    assert(!std.math.isInf(u_var));
    linearized.append(.{
        .view_dual = .{
            .out_offset = buffer.view().offset,
            .out_stride = buffer.view().stride,
            .in_offset = buffer.view().offset,
            .in_stride = buffer.view().stride,
            .size = buffer.view().size,
        },
        .in = buffer,
        .out = buffer,
        .kind = .unary_multiply,
        .u_var = u_var,
    });
}
pub fn unaryDivide(linearized: *Linearized, buffer: Buffer, u_var: f32) void {
    assert(!std.math.isNan(u_var));
    assert(!std.math.isInf(u_var));
    linearized.append(.{
        .view_dual = .{
            .out_offset = buffer.view().offset,
            .out_stride = buffer.view().stride,
            .in_offset = buffer.view().offset,
            .in_stride = buffer.view().stride,
            .size = buffer.view().size,
        },
        .in = buffer,
        .out = buffer,
        .kind = .unary_divide,
        .u_var = u_var,
    });
}
pub fn unaryExp(linearized: *Linearized, buffer: Buffer) void {
    linearized.append(.{
        .view_dual = .{
            .out_offset = buffer.view().offset,
            .out_stride = buffer.view().stride,
            .in_offset = buffer.view().offset,
            .in_stride = buffer.view().stride,
            .size = buffer.view().size,
        },
        .in = buffer,
        .out = buffer,
        .kind = .unary_exp,
        .u_var = 0,
    });
}
pub fn unaryLog(linearized: *Linearized, buffer: Buffer) void {
    linearized.append(.{
        .view_dual = .{
            .out_offset = buffer.view().offset,
            .out_stride = buffer.view().stride,
            .in_offset = buffer.view().offset,
            .in_stride = buffer.view().stride,
            .size = buffer.view().size,
        },
        .in = buffer,
        .out = buffer,
        .kind = .unary_log,
        .u_var = 0,
    });
}
pub fn unarySquare(linearized: *Linearized, buffer: Buffer) void {
    linearized.append(.{
        .view_dual = .{
            .out_offset = buffer.view().offset,
            .out_stride = buffer.view().stride,
            .in_offset = buffer.view().offset,
            .in_stride = buffer.view().stride,
            .size = buffer.view().size,
        },
        .in = buffer,
        .out = buffer,
        .kind = .unary_square,
        .u_var = 0,
    });
}
pub fn unarySqrt(linearized: *Linearized, buffer: Buffer) void {
    linearized.append(.{
        .view_dual = .{
            .out_offset = buffer.view().offset,
            .out_stride = buffer.view().stride,
            .in_offset = buffer.view().offset,
            .in_stride = buffer.view().stride,
            .size = buffer.view().size,
        },
        .in = buffer,
        .out = buffer,
        .kind = .unary_sqrt,
        .u_var = 0,
    });
}
pub fn unaryReciprocal(linearized: *Linearized, buffer: Buffer) void {
    linearized.append(.{
        .view_dual = .{
            .out_offset = buffer.view().offset,
            .out_stride = buffer.view().stride,
            .in_offset = buffer.view().offset,
            .in_stride = buffer.view().stride,
            .size = buffer.view().size,
        },
        .in = buffer,
        .out = buffer,
        .kind = .unary_reciprocal,
        .u_var = 0,
    });
}
pub fn unaryMax(linearized: *Linearized, buffer: Buffer, u_var: f32) void {
    assert(!std.math.isNan(u_var));
    assert(!std.math.isInf(u_var));
    linearized.append(.{
        .view_dual = .{
            .out_offset = buffer.view().offset,
            .out_stride = buffer.view().stride,
            .in_offset = buffer.view().offset,
            .in_stride = buffer.view().stride,
            .size = buffer.view().size,
        },
        .in = buffer,
        .out = buffer,
        .kind = .unary_max,
        .u_var = u_var,
    });
}
pub fn unaryMin(linearized: *Linearized, buffer: Buffer, u_var: f32) void {
    assert(!std.math.isNan(u_var));
    assert(!std.math.isInf(u_var));
    linearized.append(.{
        .view_dual = .{
            .out_offset = buffer.view().offset,
            .out_stride = buffer.view().stride,
            .in_offset = buffer.view().offset,
            .in_stride = buffer.view().stride,
            .size = buffer.view().size,
        },
        .in = buffer,
        .out = buffer,
        .kind = .unary_min,
        .u_var = u_var,
    });
}
pub fn unarySet(linearized: *Linearized, buffer: Buffer, u_var: f32) void {
    assert(!std.math.isNan(u_var));
    assert(!std.math.isInf(u_var));
    linearized.append(.{
        .view_dual = .{
            .out_offset = buffer.view().offset,
            .out_stride = buffer.view().stride,
            .in_offset = buffer.view().offset,
            .in_stride = buffer.view().stride,
            .size = buffer.view().size,
        },
        .in = buffer,
        .out = buffer,
        .kind = .unary_set,
        .u_var = u_var,
    });
}
// $TODO Decide if this explicit seed thing is actually any good at all.
//  I don't really want to make the user think about it, but the explicit-nes is also nice
/// Here u_var is the seed of the prng
pub fn unaryRandom(linearized: *Linearized, buffer: Buffer, u_var: u32) void {
    linearized.append(.{
        .view_dual = .{
            .out_offset = buffer.view().offset,
            .out_stride = buffer.view().stride,
            .in_offset = buffer.view().offset,
            .in_stride = buffer.view().stride,
            .size = buffer.view().size,
        },
        .in = buffer,
        .out = buffer,
        .kind = .unary_random,
        .u_var = @bitCast(u_var),
    });
}
pub fn unaryTanh(linearized: *Linearized, buffer: Buffer) void {
    linearized.append(.{
        .view_dual = .{
            .out_offset = buffer.view().offset,
            .out_stride = buffer.view().stride,
            .in_offset = buffer.view().offset,
            .in_stride = buffer.view().stride,
            .size = buffer.view().size,
        },
        .in = buffer,
        .out = buffer,
        .kind = .unary_tanh,
        .u_var = 0,
    });
}
pub fn unaryAbsolute(linearized: *Linearized, buffer: Buffer) void {
    linearized.append(.{
        .view_dual = .{
            .out_offset = buffer.view().offset,
            .out_stride = buffer.view().stride,
            .in_offset = buffer.view().offset,
            .in_stride = buffer.view().stride,
            .size = buffer.view().size,
        },
        .in = buffer,
        .out = buffer,
        .kind = .unary_absolute,
        .u_var = 0,
    });
}
// This could be changed to (sign_bit * 2) - 1, which gives a different result for 0, but who cares
pub fn unarySign(linearized: *Linearized, buffer: Buffer) void {
    linearized.append(.{
        .view_dual = .{
            .out_offset = buffer.view().offset,
            .out_stride = buffer.view().stride,
            .in_offset = buffer.view().offset,
            .in_stride = buffer.view().stride,
            .size = buffer.view().size,
        },
        .in = buffer,
        .out = buffer,
        .kind = .unary_sign,
        .u_var = 0,
    });
}
pub fn binaryAdd(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(out.view().size.equal(in.view().size));
    linearized.append(.{
        .view_dual = .{
            .size = out.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .binary_add,
        .u_var = 0,
    });
}
pub fn binarySubtract(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(out.view().size.equal(in.view().size));
    linearized.append(.{
        .view_dual = .{
            .size = out.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .binary_subtract,
        .u_var = 0,
    });
}
pub fn binaryMultiply(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(out.view().size.equal(in.view().size));
    linearized.append(.{
        .view_dual = .{
            .size = out.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .binary_multiply,
        .u_var = 0,
    });
}
pub fn binaryDivide(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(out.view().size.equal(in.view().size));
    linearized.append(.{
        .view_dual = .{
            .size = out.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .binary_divide,
        .u_var = 0,
    });
}
pub fn binaryMax(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(out.view().size.equal(in.view().size));
    linearized.append(.{
        .view_dual = .{
            .size = out.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .binary_max,
        .u_var = 0,
    });
}
pub fn binaryMin(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(out.view().size.equal(in.view().size));
    linearized.append(.{
        .view_dual = .{
            .size = out.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .binary_min,
        .u_var = 0,
    });
}
pub fn binarySet(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(out.view().size.equal(in.view().size));
    linearized.append(.{
        .view_dual = .{
            .size = out.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .binary_set,
        .u_var = 0,
    });
}
pub fn expandAdd(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(in.view().size.equal(.{ .a = 1, .z = 1, .y = 1, .x = 1 }));
    linearized.append(.{
        .view_dual = .{
            .size = out.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .expand_add,
        .u_var = 0,
    });
}
pub fn expandSubtract(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(in.view().size.equal(.{ .a = 1, .z = 1, .y = 1, .x = 1 }));
    linearized.append(.{
        .view_dual = .{
            .size = out.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .expand_subtract,
        .u_var = 0,
    });
}
pub fn expandMultiply(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(in.view().size.equal(.{ .a = 1, .z = 1, .y = 1, .x = 1 }));
    linearized.append(.{
        .view_dual = .{
            .size = out.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .expand_multiply,
        .u_var = 0,
    });
}
pub fn expandDivide(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(in.view().size.equal(.{ .a = 1, .z = 1, .y = 1, .x = 1 }));
    linearized.append(.{
        .view_dual = .{
            .size = out.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .expand_divide,
        .u_var = 0,
    });
}
pub fn expandMax(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(in.view().size.equal(.{ .a = 1, .z = 1, .y = 1, .x = 1 }));
    linearized.append(.{
        .view_dual = .{
            .size = out.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .expand_max,
        .u_var = 0,
    });
}
pub fn expandMin(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(in.view().size.equal(.{ .a = 1, .z = 1, .y = 1, .x = 1 }));
    linearized.append(.{
        .view_dual = .{
            .size = out.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .expand_min,
        .u_var = 0,
    });
}
pub fn expandSet(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(in.view().size.equal(.{ .a = 1, .z = 1, .y = 1, .x = 1 }));
    linearized.append(.{
        .view_dual = .{
            .size = out.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .expand_set,
        .u_var = 0,
    });
}
pub fn reduceSum(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(out.view().size.equal(.{ .a = 1, .z = 1, .y = 1, .x = 1 }));
    linearized.append(.{
        .view_dual = .{
            .size = in.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .reduce_sum,
        .u_var = 0,
    });
}
pub fn reduceMax(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(out.view().size.equal(.{ .a = 1, .z = 1, .y = 1, .x = 1 }));
    linearized.append(.{
        .view_dual = .{
            .size = in.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .reduce_max,
        .u_var = 0,
    });
}
pub fn reduceMin(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(out.view().size.equal(.{ .a = 1, .z = 1, .y = 1, .x = 1 }));
    linearized.append(.{
        .view_dual = .{
            .size = in.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .reduce_min,
        .u_var = 0,
    });
}
pub fn reduceAvg(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.id != in.id);
    assert(out.view().size.equal(.{ .a = 1, .z = 1, .y = 1, .x = 1 }));
    linearized.append(.{
        .view_dual = .{
            .size = in.view().size,
            .out_offset = out.view().offset,
            .out_stride = out.view().stride,
            .in_offset = in.view().offset,
            .in_stride = in.view().stride,
        },
        .out = out,
        .in = in,
        .kind = .reduce_avg,
        .u_var = 0,
    });
}
