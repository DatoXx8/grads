const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const Buffer = @import("Buffer.zig");

const Program = @import("compiler/Program.zig");
const Memory = Program.Memory;
const Runtime = @import("compiler/runtimes/Runtime.zig");

pub const Linearized = @This();

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
        pub inline fn isUnary(this: @This()) bool {
            return switch (this) {
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
        pub inline fn isBinary(this: @This()) bool {
            return switch (this) {
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
        pub inline fn isExpand(this: @This()) bool {
            return switch (this) {
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
        pub inline fn isReduce(this: @This()) bool {
            return switch (this) {
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
        pub fn overwrites(this: @This()) bool {
            return switch (this) {
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
    // When using this as a seed for the unary_random op, the integer value will be cast back and forth with @bitCast,
    //  here's hoping some NaN floating point magic doesn't ruin that
    u_var: f32,
    out: Buffer,
    in: Buffer,
    // $TODO Make this sucker SIMD-able
    pub fn realize(this: @This()) void {
        if (this.kind.isUnary()) {
            // In buffer is just a copy of out buffer, basically just a sanity check.
            assert(this.out.a_size == this.in.a_size);
            assert(this.out.z_size == this.in.z_size);
            assert(this.out.y_size == this.in.y_size);
            assert(this.out.x_size == this.in.x_size);
        } else if (this.kind.isBinary()) {
            assert(this.out.a_size == this.in.a_size);
            assert(this.out.z_size == this.in.z_size);
            assert(this.out.y_size == this.in.y_size);
            assert(this.out.x_size == this.in.x_size);
        } else if (this.kind.isExpand()) {
            assert(this.in.a_size == 1);
            assert(this.in.z_size == 1);
            assert(this.in.y_size == 1);
            assert(this.in.x_size == 1);
        } else if (this.kind.isReduce()) {
            assert(this.out.a_size == 1);
            assert(this.out.z_size == 1);
            assert(this.out.y_size == 1);
            assert(this.out.x_size == 1);
        } else {
            unreachable;
        }

        switch (this.kind) {
            .reduce_sum => {
                this.out.values[this.out.at(0, 0, 0, 0)] = 0;
            },
            .reduce_max => {
                this.out.values[this.out.at(0, 0, 0, 0)] = -std.math.inf(f32);
            },
            .reduce_min => {
                this.out.values[this.out.at(0, 0, 0, 0)] = std.math.inf(f32);
            },
            .reduce_avg => {
                this.out.values[this.out.at(0, 0, 0, 0)] = 0;
            },
            else => {},
        }

        var rng: ?std.Random.Pcg = if (this.kind == .unary_random)
            std.Random.Pcg.init(@as(u32, @bitCast(this.u_var)))
        else
            null;

        // Just to be clear I know that putting the loop outside might make it slower because you have to go through the switch statement every time, but
        // the branch predictor is likely to have an extremely easy time predicting the branches since it's the same every single time.
        // Which should mean that as long as your CPU even has a branch predictor it should cause very little to no performance impact.
        // I measured it by running some arbitrary ops and there was no measurable difference
        const a_size: u32 = if (this.kind.isReduce()) this.in.a_size else this.out.a_size;
        const z_size: u32 = if (this.kind.isReduce()) this.in.z_size else this.out.z_size;
        const y_size: u32 = if (this.kind.isReduce()) this.in.y_size else this.out.y_size;
        const x_size: u32 = if (this.kind.isReduce()) this.in.x_size else this.out.x_size;
        var a: u32 = 0;
        while (a < a_size) : (a += 1) {
            var z: u32 = 0;
            while (z < z_size) : (z += 1) {
                var y: u32 = 0;
                while (y < y_size) : (y += 1) {
                    var x: u32 = 0;
                    while (x < x_size) : (x += 1) {
                        switch (this.kind) {
                            .unary_add => {
                                this.out.values[this.out.at(a, z, y, x)] += this.u_var;
                            },
                            .unary_subtract => {
                                this.out.values[this.out.at(a, z, y, x)] -= this.u_var;
                            },
                            .unary_multiply => {
                                this.out.values[this.out.at(a, z, y, x)] *= this.u_var;
                            },
                            .unary_divide => {
                                this.out.values[this.out.at(a, z, y, x)] /= this.u_var;
                            },
                            .unary_exp => {
                                this.out.values[this.out.at(a, z, y, x)] =
                                    @exp(this.out.values[this.out.at(a, z, y, x)]);
                            },
                            .unary_log => {
                                this.out.values[this.out.at(a, z, y, x)] =
                                    @log(this.out.values[this.out.at(a, z, y, x)]);
                            },
                            .unary_square => {
                                this.out.values[this.out.at(a, z, y, x)] *=
                                    this.out.values[this.out.at(a, z, y, x)];
                            },
                            .unary_sqrt => {
                                this.out.values[this.out.at(a, z, y, x)] =
                                    @sqrt(this.out.values[this.out.at(a, z, y, x)]);
                            },
                            .unary_reciprocal => {
                                this.out.values[this.out.at(a, z, y, x)] =
                                    1 / this.out.values[this.out.at(a, z, y, x)];
                            },
                            .unary_max => {
                                this.out.values[this.out.at(a, z, y, x)] =
                                    @max(this.out.values[this.out.at(a, z, y, x)], this.u_var);
                            },
                            .unary_min => {
                                this.out.values[this.out.at(a, z, y, x)] =
                                    @min(this.out.values[this.out.at(a, z, y, x)], this.u_var);
                            },
                            .unary_set => {
                                this.out.values[this.out.at(a, z, y, x)] = this.u_var;
                            },
                            .unary_random => {
                                // $TODO Make my own PCG implementation that can do SIMD
                                this.out.values[this.out.at(a, z, y, x)] =
                                    rng.?.random().floatNorm(f32);
                            },
                            .unary_tanh => {
                                this.out.values[this.out.at(a, z, y, x)] =
                                    std.math.tanh(this.out.values[this.out.at(a, z, y, x)]);
                            },
                            .unary_absolute => {
                                this.out.values[this.out.at(a, z, y, x)] =
                                    @abs(this.out.values[this.out.at(a, z, y, x)]);
                            },
                            .unary_sign => {
                                if (this.out.values[this.out.at(a, z, y, x)] > 0) {
                                    this.out.values[this.out.at(a, z, y, x)] = 1;
                                } else if (this.out.values[this.out.at(a, z, y, x)] < 0) {
                                    this.out.values[this.out.at(a, z, y, x)] = -1;
                                } else {
                                    this.out.values[this.out.at(a, z, y, x)] = 0;
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
                                this.out.values[this.out.at(a, z, y, x)] +=
                                    this.in.values[this.in.at(a, z, y, x)];
                            },
                            .binary_subtract => {
                                this.out.values[this.out.at(a, z, y, x)] -=
                                    this.in.values[this.in.at(a, z, y, x)];
                            },
                            .binary_multiply => {
                                this.out.values[this.out.at(a, z, y, x)] *=
                                    this.in.values[this.in.at(a, z, y, x)];
                            },
                            .binary_divide => {
                                this.out.values[this.out.at(a, z, y, x)] /=
                                    this.in.values[this.in.at(a, z, y, x)];
                            },
                            .binary_max => {
                                this.out.values[this.out.at(a, z, y, x)] =
                                    @max(this.out.values[this.out.at(a, z, y, x)], //
                                    this.in.values[this.in.at(a, z, y, x)]);
                            },
                            .binary_min => {
                                this.out.values[this.out.at(a, z, y, x)] =
                                    @min(this.out.values[this.out.at(a, z, y, x)], //
                                    this.in.values[this.in.at(a, z, y, x)]);
                            },
                            .binary_set => {
                                this.out.values[this.out.at(a, z, y, x)] =
                                    this.in.values[this.in.at(a, z, y, x)];
                            },
                            .expand_add => {
                                this.out.values[this.out.at(a, z, y, x)] +=
                                    this.in.values[this.in.at(0, 0, 0, 0)];
                            },
                            .expand_subtract => {
                                this.out.values[this.out.at(a, z, y, x)] -=
                                    this.in.values[this.in.at(0, 0, 0, 0)];
                            },
                            .expand_multiply => {
                                this.out.values[this.out.at(a, z, y, x)] *=
                                    this.in.values[this.in.at(0, 0, 0, 0)];
                            },
                            .expand_divide => {
                                this.out.values[this.out.at(a, z, y, x)] /=
                                    this.in.values[this.in.at(0, 0, 0, 0)];
                            },
                            .expand_max => {
                                this.out.values[this.out.at(a, z, y, x)] =
                                    @max(this.out.values[this.out.at(a, z, y, x)], //
                                    this.in.values[this.in.at(0, 0, 0, 0)]);
                            },
                            .expand_min => {
                                this.out.values[this.out.at(a, z, y, x)] =
                                    @min(this.out.values[this.out.at(a, z, y, x)], //
                                    this.in.values[this.in.at(0, 0, 0, 0)]);
                            },
                            .expand_set => {
                                this.out.values[this.out.at(a, z, y, x)] =
                                    this.in.values[this.in.at(0, 0, 0, 0)];
                            },
                            .reduce_sum => {
                                this.out.values[this.out.at(0, 0, 0, 0)] +=
                                    this.in.values[this.in.at(a, z, y, x)];
                            },
                            .reduce_max => {
                                this.out.values[this.out.at(0, 0, 0, 0)] =
                                    @max(this.out.values[this.out.at(0, 0, 0, 0)], //
                                    this.in.values[this.in.at(a, z, y, x)]);
                            },
                            .reduce_min => {
                                this.out.values[this.out.at(0, 0, 0, 0)] =
                                    @min(this.out.values[this.out.at(0, 0, 0, 0)], //
                                    this.in.values[this.in.at(a, z, y, x)]);
                            },
                            .reduce_avg => {
                                this.out.values[this.out.at(0, 0, 0, 0)] +=
                                    this.in.values[this.in.at(a, z, y, x)];
                            },
                        }
                    }
                }
            }
        }
        if (this.kind == .reduce_avg) {
            this.out.values[this.out.at(0, 0, 0, 0)] /=
                @as(f32, @floatFromInt(this.in.a_size * this.in.z_size * this.in.y_size * this.in.x_size));
        }
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}{s} ", .{ " " ** (padding + offset), text });
        } else {
            std.debug.print("{s}", .{" " ** (padding + offset)});
        }
        if (this.kind.isUnary()) {
            std.debug.print("U {s} ({d} {d} {d} {d}) [{d} {d} {d} {d} = {d}] \"{s}\" {d}\n", .{
                switch (this.kind) {
                    .unary_add => "add",
                    .unary_subtract => "sub",
                    .unary_multiply => "mul",
                    .unary_divide => "div",
                    .unary_exp => "exp",
                    .unary_log => "log",
                    .unary_square => "sqr",
                    .unary_sqrt => "sqt",
                    .unary_reciprocal => "rcp",
                    .unary_max => "max",
                    .unary_min => "min",
                    .unary_set => "set",
                    .unary_random => "rng",
                    .unary_tanh => "tanh",
                    .unary_absolute => "abs",
                    .unary_sign => "sgn",
                    else => unreachable,
                },
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.aOffset(),
                this.out.zOffset(),
                this.out.yOffset(),
                this.out.xOffset(),
                this.out.offset,
                this.out.name(),
                this.u_var,
            });
        } else {
            const op_kind: u8 = if (this.kind.isBinary())
                'B'
            else
                (if (this.kind.isExpand())
                    'E'
                else
                    'R');
            std.debug.print("{c} {s} ({d} {d} {d} {d}) [{d} {d} {d} {d} = {d}] \"{s}\" ({d} {d} {d} {d}) [{d} {d} {d} {d} = {d}] \"{s}\"\n", .{
                op_kind,
                switch (this.kind) {
                    .binary_add => "add",
                    .binary_subtract => "sub",
                    .binary_multiply => "mul",
                    .binary_divide => "div",
                    .binary_max => "max",
                    .binary_min => "min",
                    .binary_set => "set",
                    .expand_add => "add",
                    .expand_subtract => "sub",
                    .expand_multiply => "mul",
                    .expand_divide => "div",
                    .expand_max => "max",
                    .expand_min => "min",
                    .expand_set => "set",
                    .reduce_sum => "sum",
                    .reduce_max => "max",
                    .reduce_min => "min",
                    .reduce_avg => "avg",
                    else => unreachable,
                },
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.aOffset(),
                this.out.zOffset(),
                this.out.yOffset(),
                this.out.xOffset(),
                this.out.offset,
                this.out.name(),
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.aOffset(),
                this.in.zOffset(),
                this.in.yOffset(),
                this.in.xOffset(),
                this.in.offset,
                this.in.name(),
            });
        }
    }
};

op: []Op,
op_num: u32,
pub fn alloc(gpa: Allocator, capacity: u32) !Linearized {
    return .{
        .op_num = 0,
        .op = try gpa.alloc(Op, capacity),
    };
}
pub fn capacityEnsure(linearized: *Linearized, gpa: Allocator, capacity: u32) !void {
    if (linearized.op.len < linearized.op_num + capacity) {
        linearized.op = try gpa.realloc(linearized.op, linearized.op_num + capacity);
    }
}
pub fn free(linearized: *Linearized(), gpa: Allocator) void {
    linearized.op_num = 0;
    gpa.free(linearized.op);
}
pub fn clear(linearized: *Linearized) void {
    linearized.op_num = 0;
}
pub fn run(linearized: *Linearized) void {
    var op_idx: u32 = 0;
    while (op_idx < linearized.op_num) : (op_idx += 1) {
        linearized.op[op_idx].realize();
    }
}
pub fn realize(linearized: *Linearized) void {
    linearized.run();
    linearized.clear();
}
pub fn append(linearized: *Linearized, op: Op) void {
    assert(linearized.op_num < linearized.op.len);
    linearized.op[linearized.op_num] = op;
    linearized.op_num += 1;
}
pub fn concat(linearized: *Linearized, source: *Linearized) void {
    assert(linearized.op_num + source.op_num <= linearized.op.len);
    var op_idx: u32 = 0;
    while (op_idx < source.op_num) : (op_idx += 1) {
        linearized.op[linearized.op_num + op_idx] = source.op[op_idx];
    }
    linearized.op_num += source.op_num;
    source.clear();
}
pub fn print(linearized: Linearized, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
    if (name) |text| {
        std.debug.print("{s}Linearized = {s}\n", .{ " " ** offset, text });
    } else {
        std.debug.print("{s}Linearized\n", .{" " ** offset});
    }
    if (linearized.op_num == 0) {
        std.debug.print("{s}[] => empty\n", .{" " ** (offset + padding)});
    } else {
        for (0..linearized.op_num) |op_idx| {
            std.debug.print("{s}[{}] => ", .{ " " ** (offset + padding), op_idx });
            linearized.op[op_idx].print(0, 0, null);
        }
    }
}
// $TODO Maybe get rid of all of these?
pub fn unaryAdd(linearized: *Linearized, buffer: Buffer, u_var: f32) void {
    assert(!std.math.isNan(u_var));
    assert(!std.math.isInf(u_var));
    linearized.append(.{
        .out = buffer,
        .in = buffer,
        .kind = .unary_add,
        .u_var = u_var,
    });
}
pub fn unarySubtract(linearized: *Linearized, buffer: Buffer, u_var: f32) void {
    assert(!std.math.isNan(u_var));
    assert(!std.math.isInf(u_var));
    linearized.append(.{
        .out = buffer,
        .in = buffer,
        .kind = .unary_subtract,
        .u_var = u_var,
    });
}
pub fn unaryMultiply(linearized: *Linearized, buffer: Buffer, u_var: f32) void {
    assert(!std.math.isNan(u_var));
    assert(!std.math.isInf(u_var));
    linearized.append(.{
        .out = buffer,
        .in = buffer,
        .kind = .unary_multiply,
        .u_var = u_var,
    });
}
pub fn unaryDivide(linearized: *Linearized, buffer: Buffer, u_var: f32) void {
    assert(!std.math.isNan(u_var));
    assert(!std.math.isInf(u_var));
    linearized.append(.{
        .out = buffer,
        .in = buffer,
        .kind = .unary_divide,
        .u_var = u_var,
    });
}
pub fn unaryExp(linearized: *Linearized, buffer: Buffer) void {
    linearized.append(.{
        .out = buffer,
        .in = buffer,
        .kind = .unary_exp,
        .u_var = 0,
    });
}
pub fn unaryLog(linearized: *Linearized, buffer: Buffer) void {
    linearized.append(.{
        .out = buffer,
        .in = buffer,
        .kind = .unary_log,
        .u_var = 0,
    });
}
pub fn unarySquare(linearized: *Linearized, buffer: Buffer) void {
    linearized.append(.{
        .out = buffer,
        .in = buffer,
        .kind = .unary_square,
        .u_var = 0,
    });
}
pub fn unarySqrt(linearized: *Linearized, buffer: Buffer) void {
    linearized.append(.{
        .out = buffer,
        .in = buffer,
        .kind = .unary_sqrt,
        .u_var = 0,
    });
}
pub fn unaryReciprocal(linearized: *Linearized, buffer: Buffer) void {
    linearized.append(.{
        .out = buffer,
        .in = buffer,
        .kind = .unary_reciprocal,
        .u_var = 0,
    });
}
pub fn unaryMax(linearized: *Linearized, buffer: Buffer, u_var: f32) void {
    assert(!std.math.isNan(u_var));
    assert(!std.math.isInf(u_var));
    linearized.append(.{
        .out = buffer,
        .in = buffer,
        .kind = .unary_max,
        .u_var = u_var,
    });
}
pub fn unaryMin(linearized: *Linearized, buffer: Buffer, u_var: f32) void {
    assert(!std.math.isNan(u_var));
    assert(!std.math.isInf(u_var));
    linearized.append(.{
        .out = buffer,
        .in = buffer,
        .kind = .unary_min,
        .u_var = u_var,
    });
}
pub fn unarySet(linearized: *Linearized, buffer: Buffer, u_var: f32) void {
    assert(!std.math.isNan(u_var));
    assert(!std.math.isInf(u_var));
    linearized.append(.{
        .out = buffer,
        .in = buffer,
        .kind = .unary_set,
        .u_var = u_var,
    });
}
// $TODO Decide if this explicit seed thing is actually any good at all.
//  I don't really want to make the user think about it, but the explicit-nes is also nice
/// Here u_var is the seed of the prng
pub fn unaryRandom(linearized: *Linearized, buffer: Buffer, u_var: u32) void {
    linearized.append(.{
        .out = buffer,
        .in = buffer,
        .kind = .unary_random,
        .u_var = @bitCast(u_var),
    });
}
pub fn unaryTanh(linearized: *Linearized, buffer: Buffer) void {
    linearized.append(.{
        .out = buffer,
        .in = buffer,
        .kind = .unary_tanh,
        .u_var = 0,
    });
}
pub fn unaryAbsolute(linearized: *Linearized, buffer: Buffer) void {
    linearized.append(.{
        .out = buffer,
        .in = buffer,
        .kind = .unary_absolute,
        .u_var = 0,
    });
}
// This could be changed to (sign_bit * 2) - 1, which gives a different result for 0, but who cares
pub fn unarySign(linearized: *Linearized, buffer: Buffer) void {
    linearized.append(.{
        .out = buffer,
        .in = buffer,
        .kind = .unary_sign,
        .u_var = 0,
    });
}
pub fn binaryAdd(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.a_size == in.a_size);
    assert(out.z_size == in.z_size);
    assert(out.y_size == in.y_size);
    assert(out.x_size == in.x_size);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .binary_add,
        .u_var = 0,
    });
}
pub fn binarySubtract(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.a_size == in.a_size);
    assert(out.z_size == in.z_size);
    assert(out.y_size == in.y_size);
    assert(out.x_size == in.x_size);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .binary_subtract,
        .u_var = 0,
    });
}
pub fn binaryMultiply(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.a_size == in.a_size);
    assert(out.z_size == in.z_size);
    assert(out.y_size == in.y_size);
    assert(out.x_size == in.x_size);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .binary_multiply,
        .u_var = 0,
    });
}
pub fn binaryDivide(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.a_size == in.a_size);
    assert(out.z_size == in.z_size);
    assert(out.y_size == in.y_size);
    assert(out.x_size == in.x_size);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .binary_divide,
        .u_var = 0,
    });
}
pub fn binaryMax(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.a_size == in.a_size);
    assert(out.z_size == in.z_size);
    assert(out.y_size == in.y_size);
    assert(out.x_size == in.x_size);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .binary_max,
        .u_var = 0,
    });
}
pub fn binaryMin(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.a_size == in.a_size);
    assert(out.z_size == in.z_size);
    assert(out.y_size == in.y_size);
    assert(out.x_size == in.x_size);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .binary_min,
        .u_var = 0,
    });
}
pub fn binarySet(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.a_size == in.a_size);
    assert(out.z_size == in.z_size);
    assert(out.y_size == in.y_size);
    assert(out.x_size == in.x_size);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .binary_set,
        .u_var = 0,
    });
}
pub fn expandAdd(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(in.a_size == 1);
    assert(in.z_size == 1);
    assert(in.y_size == 1);
    assert(in.x_size == 1);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .expand_add,
        .u_var = 0,
    });
}
pub fn expandSubtract(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(in.a_size == 1);
    assert(in.z_size == 1);
    assert(in.y_size == 1);
    assert(in.x_size == 1);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .expand_subtract,
        .u_var = 0,
    });
}
pub fn expandMultiply(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(in.a_size == 1);
    assert(in.z_size == 1);
    assert(in.y_size == 1);
    assert(in.x_size == 1);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .expand_multiply,
        .u_var = 0,
    });
}
pub fn expandDivide(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(in.a_size == 1);
    assert(in.z_size == 1);
    assert(in.y_size == 1);
    assert(in.x_size == 1);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .expand_divide,
        .u_var = 0,
    });
}
pub fn expandMax(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(in.a_size == 1);
    assert(in.z_size == 1);
    assert(in.y_size == 1);
    assert(in.x_size == 1);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .expand_max,
        .u_var = 0,
    });
}
pub fn expandMin(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(in.a_size == 1);
    assert(in.z_size == 1);
    assert(in.y_size == 1);
    assert(in.x_size == 1);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .expand_min,
        .u_var = 0,
    });
}
pub fn expandSet(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(in.a_size == 1);
    assert(in.z_size == 1);
    assert(in.y_size == 1);
    assert(in.x_size == 1);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .expand_set,
        .u_var = 0,
    });
}
pub fn reduceSum(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.a_size == 1);
    assert(out.z_size == 1);
    assert(out.y_size == 1);
    assert(out.x_size == 1);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .reduce_sum,
        .u_var = 0,
    });
}
pub fn reduceMax(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.a_size == 1);
    assert(out.z_size == 1);
    assert(out.y_size == 1);
    assert(out.x_size == 1);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .reduce_max,
        .u_var = 0,
    });
}
pub fn reduceMin(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.a_size == 1);
    assert(out.z_size == 1);
    assert(out.y_size == 1);
    assert(out.x_size == 1);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .reduce_min,
        .u_var = 0,
    });
}
pub fn reduceAvg(linearized: *Linearized, out: Buffer, in: Buffer) void {
    assert(out.a_size == 1);
    assert(out.z_size == 1);
    assert(out.y_size == 1);
    assert(out.x_size == 1);
    linearized.append(.{
        .out = out,
        .in = in,
        .kind = .reduce_avg,
        .u_var = 0,
    });
}
