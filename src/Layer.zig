const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const Optimization = @import("compiler/optimize.zig").Optimization;
const Program = @import("compiler/Program.zig");
const Runtime = @import("compiler/runtimes/Runtime.zig");
const Linearized = @import("Linearized.zig");
const Op = Linearized.Op;
const Buffer = @import("Buffer.zig");
const View = Buffer.View;
const Vec4 = Buffer.Vec4;
const util = @import("util.zig");

pub const Activation = struct {
    pub const Kind = enum(u8) {
        none,
        relu,
        sigmoid,
        relu_clipped,
        relu_leaky,
        silu,
        gelu,
        tanh,
    };
    pub const relu_leaky_factor: f32 = 0.1;
    pub const relu_clipped_factor: f32 = 1;
    comptime {
        assert(relu_leaky_factor > 0);
        assert(relu_clipped_factor > 0);
    }
    temp: Buffer,
    t: Activation.Kind,
    pub fn alloc(runtime: Runtime, gpa: Allocator, arena: Allocator, t: Activation.Kind, size: Vec4) !Activation {
        assert(size.a == 1);
        assert(size.z > 0);
        assert(size.y > 0);
        assert(size.x > 0);
        return .{
            .t = t,
            .temp = switch (t) {
                .none => try Buffer.alloc(runtime, gpa, arena, size, .intermediary),
                .relu => try Buffer.alloc(runtime, gpa, arena, size, .intermediary),
                .sigmoid => try Buffer.alloc(runtime, gpa, arena, size, .intermediary),
                .relu_clipped => try Buffer.alloc(runtime, gpa, arena, size, .intermediary),
                .relu_leaky => try Buffer.alloc(runtime, gpa, arena, size, .intermediary),
                .silu => try Buffer.alloc(runtime, gpa, arena, size, .intermediary),
                .gelu => try Buffer.alloc(runtime, gpa, arena, size, .intermediary),
                .tanh => try Buffer.alloc(runtime, gpa, arena, size, .intermediary),
            },
        };
    }
    pub fn free(actiavtion: Activation, runtime: Runtime) void {
        actiavtion.temp.free(runtime);
    }
    pub fn forward(activation: *Activation, linearized: *Linearized, values: Buffer) void {
        switch (activation.t) {
            .none => {},
            .relu => {
                linearized.unaryMax(values, 0);
            },
            .sigmoid => {
                linearized.unaryMultiply(values, -1);
                linearized.unaryExp(values);
                linearized.unaryAdd(values, 1);
                linearized.unaryReciprocal(values);
            },
            .relu_clipped => {
                linearized.unaryMax(values, 0);
                linearized.unaryMin(values, relu_clipped_factor);
            },
            .relu_leaky => {
                linearized.binarySet(activation.temp, values);
                linearized.unaryMultiply(activation.temp, relu_leaky_factor);
                linearized.binaryMax(values, activation.temp);
            },
            .silu => {
                linearized.binarySet(activation.temp, values);
                linearized.unaryMultiply(activation.temp, -1);
                linearized.unaryExp(activation.temp);
                linearized.unaryAdd(activation.temp, 1);
                linearized.unaryReciprocal(activation.temp);
                linearized.binaryMultiply(values, activation.temp);
            },
            .gelu => {
                linearized.binarySet(activation.temp, values);
                linearized.unaryMultiply(activation.temp, -1.702);
                linearized.unaryExp(activation.temp);
                linearized.unaryAdd(activation.temp, 1);
                linearized.unaryReciprocal(activation.temp);
                linearized.binaryMultiply(values, activation.temp);
            },
            .tanh => {
                linearized.unaryTanh(values);
            },
        }
    }
    pub fn backward(activation: Activation, linearized: *Linearized, values: Buffer, values_g: Buffer) void {
        switch (activation.t) {
            .none => {},
            .relu => {
                linearized.binarySet(activation.temp, values);
                linearized.unarySign(activation.temp);
                linearized.binaryMultiply(values_g, activation.temp);
            },
            .sigmoid => {
                linearized.unarySet(activation.temp, 1);
                linearized.binarySubtract(activation.temp, values);
                linearized.binaryMultiply(activation.temp, values);
                linearized.binaryMultiply(values_g, activation.temp);
            },
            .relu_clipped => {
                linearized.binarySet(activation.temp, values);
                linearized.unarySubtract(activation.temp, relu_clipped_factor);
                linearized.unaryAbsolute(activation.temp);
                linearized.unarySign(activation.temp);
                linearized.binaryMultiply(values_g, activation.temp);
            },
            .relu_leaky => {
                util.todo(@src());
            },
            .silu => {
                util.todo(@src());
            },
            .gelu => {
                util.todo(@src());
            },
            .tanh => {
                linearized.binarySet(activation.temp, values);
                linearized.unarySquare(activation.temp);
                linearized.unaryMultiply(activation.temp, -1);
                linearized.unaryAdd(activation.temp, 1);
                linearized.binaryMultiply(values_g, activation.temp);
            },
        }
    }
};
pub const Dense = struct {
    size_in: u32,
    size_out: u32,

    weights: Buffer,
    weights_g: Buffer,
    biases: Buffer,
    biases_g: Buffer,

    temp_in: Buffer,
    temp_out: Buffer,
    temp_full: Buffer,
    pub fn alloc(runtime: Runtime, gpa: Allocator, arena: Allocator, size_in: u32, size_out: u32) !Dense {
        assert(size_in > 0);
        assert(size_out > 0);
        return .{
            .size_in = size_in,
            .size_out = size_out,
            .weights = try Buffer.alloc(runtime, gpa, arena, .{ .a = 1, .z = 1, .y = size_in, .x = size_out }, .normal),
            .weights_g = try Buffer.alloc(runtime, gpa, arena, .{ .a = 1, .z = 1, .y = size_in, .x = size_out }, .normal),
            .biases = try Buffer.alloc(runtime, gpa, arena, .{ .a = 1, .z = 1, .y = 1, .x = size_out }, .normal),
            .biases_g = try Buffer.alloc(runtime, gpa, arena, .{ .a = 1, .z = 1, .y = 1, .x = size_out }, .normal),
            .temp_in = try Buffer.alloc(runtime, gpa, arena, .{ .a = 1, .z = 1, .y = size_in, .x = 1 }, .intermediary),
            .temp_out = try Buffer.alloc(runtime, gpa, arena, .{ .a = 1, .z = 1, .y = 1, .x = size_out }, .intermediary),
            .temp_full = try Buffer.alloc(runtime, gpa, arena, .{ .a = 1, .z = 1, .y = size_in, .x = size_out }, .intermediary),
        };
    }
    pub fn free(dense: Dense, runtime: Runtime) void {
        dense.weights.free(runtime);
        dense.weights_g.free(runtime);
        dense.biases.free(runtime);
        dense.biases_g.free(runtime);
        dense.temp_in.free(runtime);
        dense.temp_out.free(runtime);
        dense.temp_full.free(runtime);
    }
    pub fn forward(dense: *Dense, linearized: *Linearized, in: Buffer, out: *Buffer) void {
        assert(in.view().size.equal(Vec4.splat(1).setY(dense.size_in)));
        assert(out.view().size.equal(Vec4.splat(1).setX(dense.size_out)));

        dense.weights.moveResize(Vec4.splat(1).setY(dense.size_in));
        out.moveResize(Vec4.splat(1));

        var row_idx: u32 = 0;
        while (row_idx < dense.size_out) : (row_idx += 1) {
            dense.weights.moveOffset(Vec4.splat(0).setX(row_idx));
            out.moveOffset(Vec4.splat(0).setX(row_idx));
            linearized.binarySet(dense.temp_in, dense.weights);
            linearized.binaryMultiply(dense.temp_in, in);
            linearized.reduceSum(out.*, dense.temp_in);
        }

        dense.weights.moveResize(.{ .a = 1, .z = 1, .y = dense.size_in, .x = dense.size_out });
        dense.weights.moveOffset(Vec4.splat(0));
        out.moveResize(.{ .a = 1, .z = 1, .y = 1, .x = dense.size_out });
        out.moveOffset(Vec4.splat(0));
        linearized.binaryAdd(out.*, dense.biases);
    }
    pub fn backward(dense: *Dense, linearized: *Linearized, in: Buffer, in_g: *Buffer, out_g: Buffer) void {
        assert(in.view().size.equal(Vec4.splat(1).setY(dense.size_in)));
        assert(in_g.view().overlapsAll(in.view().*));
        assert(out_g.view().size.equal(Vec4.splat(1).setX(dense.size_out)));

        linearized.binaryAdd(dense.biases_g, out_g);

        dense.temp_full.moveResize(Vec4.splat(1).setX(dense.size_out));
        var column_idx: u32 = 0;
        while (column_idx < dense.size_in) : (column_idx += 1) {
            dense.temp_full.moveOffset(Vec4.splat(0).setY(column_idx));
            linearized.binarySet(dense.temp_full, out_g);
        }
        dense.temp_full.moveResize(Vec4.splat(1).setY(dense.size_in));
        var row_idx: u32 = 0;
        while (row_idx < dense.size_out) : (row_idx += 1) {
            dense.temp_full.moveOffset(Vec4.splat(0).setX(row_idx));
            linearized.binaryMultiply(dense.temp_full, in);
        }
        dense.temp_full.moveResize(.{ .a = 1, .z = 1, .y = dense.size_in, .x = dense.size_out });
        dense.temp_full.moveOffset(Vec4.splat(0));
        linearized.binaryAdd(dense.weights_g, dense.temp_full);

        in_g.moveResize(Vec4.splat(1));
        dense.weights.moveResize(Vec4.splat(1).setX(dense.size_out));
        column_idx = 0;
        while (column_idx < dense.size_in) : (column_idx += 1) {
            in_g.moveOffset(Vec4.splat(0).setY(column_idx));
            dense.weights.moveOffset(Vec4.splat(0).setY(column_idx));
            linearized.binarySet(dense.temp_out, dense.weights);
            linearized.binaryMultiply(dense.temp_out, out_g);
            linearized.reduceSum(in_g.*, dense.temp_out); // Could do avg here for move numerical stability
        }
        in_g.moveResize(Vec4.splat(1).setY(dense.size_in));
        in_g.moveOffset(Vec4.splat(0));
        dense.weights.moveResize(.{ .a = 1, .z = 1, .y = dense.size_in, .x = dense.size_out });
        dense.weights.moveOffset(Vec4.splat(0));
    }
    pub fn print(dense: Dense, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            util.log.print("{s}Dense {s}\n", .{ " " ** offset, text });
        } else {
            util.log.print("{s}Dense\n", .{" " ** offset});
        }
        util.log.print("{s}In {}, Out {}\n", //
            .{ " " ** (offset + padding), dense.size_in, dense.size_out });
    }
    pub fn debug(dense: Dense, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        dense.print(padding, offset, name);
        dense.weights.print(padding, padding + offset, "weights");
        dense.weights_g.print(padding, padding + offset, "weights_g");
        dense.biases.print(padding, padding + offset, "biases");
        dense.biases_g.print(padding, padding + offset, "biases_g");
        dense.temp_in.print(padding, padding + offset, "temp_in");
        dense.temp_out.print(padding, padding + offset, "temp_out");
        dense.temp_full.print(padding, padding + offset, "temp_full");
    }
};
pub const Convolution = struct {
    size_in: Vec4,
    filters: u32,
    kernel_size: u32,
    kernel_stride: u32,
    kernel_padding: u32,

    weights: Buffer,
    weights_g: Buffer,
    biases: Buffer,
    biases_g: Buffer,

    temp_input_padded: Buffer,
    temp_grad_padded: Buffer,
    temp_kernel: Buffer,
    temp_single: Buffer,
    /// Size is the per dimension size **`without`** padding
    pub inline fn sizeNew(dim_size: u32, size: u32, stride: u32, padding: u32) u32 {
        assert(dim_size >= size);
        return @divFloor(dim_size + 2 * padding - size, stride) + 1;
    }
    pub fn alloc(
        runtime: Runtime,
        gpa: Allocator,
        arena: Allocator,
        size_in: Vec4,
        filters: u32,
        kernel_size: u32,
        kernel_stride: u32,
        kernel_padding: u32,
    ) !Convolution {
        assert(filters > 0);
        assert(kernel_stride > 0);
        assert(kernel_size > 0);
        assert(size_in.a == 1);
        assert(size_in.z > 0);
        assert(size_in.y >= kernel_size);
        assert(size_in.y >= kernel_size);
        return .{
            .size_in = size_in,
            .filters = filters,
            .kernel_size = kernel_size,
            .kernel_stride = kernel_stride,
            .kernel_padding = kernel_padding,
            .weights = try Buffer.alloc(runtime, gpa, arena, .{
                .a = filters,
                .z = size_in.z,
                .y = kernel_size,
                .x = kernel_size,
            }, .normal),
            .weights_g = try Buffer.alloc(runtime, gpa, arena, .{
                .a = filters,
                .z = size_in.z,
                .y = kernel_size,
                .x = kernel_size,
            }, .normal),
            .biases = try Buffer.alloc(runtime, gpa, arena, Vec4.splat(1).setA(filters), .normal),
            .biases_g = try Buffer.alloc(runtime, gpa, arena, Vec4.splat(1).setA(filters), .normal),
            .temp_input_padded = try Buffer.alloc(runtime, gpa, arena, .{
                .a = 1,
                .z = size_in.z,
                .y = size_in.y + 2 * kernel_padding,
                .x = size_in.x + 2 * kernel_padding,
            }, .intermediary),
            .temp_grad_padded = try Buffer.alloc(runtime, gpa, arena, .{
                .a = 1,
                .z = size_in.z,
                .y = size_in.y + 2 * kernel_padding,
                .x = size_in.x + 2 * kernel_padding,
            }, .intermediary),
            .temp_kernel = try Buffer.alloc(runtime, gpa, arena, .{
                .a = 1,
                .z = size_in.z,
                .y = kernel_size,
                .x = kernel_size,
            }, .intermediary),
            .temp_single = try Buffer.alloc(runtime, gpa, arena, Vec4.splat(1), .intermediary),
        };
    }
    pub fn free(convolution: Convolution, runtime: Runtime) void {
        convolution.weights.free(runtime);
        convolution.weights_g.free(runtime);
        convolution.biases.free(runtime);
        convolution.biases_g.free(runtime);
        convolution.temp_input_padded.free(runtime);
        convolution.temp_grad_padded.free(runtime);
        convolution.temp_kernel.free(runtime);
        convolution.temp_single.free(runtime);
    }
    pub fn forward(convolution: *Convolution, linearized: *Linearized, in: Buffer, out: *Buffer) void {
        assert(1 == in.view().size.a);
        assert(1 == out.view().size.a);
        assert(convolution.size_in.equal(out.view().size));
        assert(convolution.filters == out.view().size.z);
        assert(out.view().size.y ==
            sizeNew(convolution.size_in.y, convolution.kernel_size, convolution.kernel_stride, convolution.kernel_padding));
        assert(out.view().size.x ==
            sizeNew(convolution.size_in.x, convolution.kernel_size, convolution.kernel_stride, convolution.kernel_padding));

        const y_out: u32 = out.view().size.y;
        const x_out: u32 = out.view().size.x;

        out.moveResize(Vec4.splat(1));
        convolution.biases.moveResize(Vec4.splat(1));
        convolution.weights.moveResize(.{
            .a = 1,
            .z = convolution.size_in.z,
            .y = convolution.kernel_size,
            .x = convolution.kernel_size,
        });
        convolution.temp_input_padded.moveResize(convolution.size_in);
        convolution.temp_input_padded.moveOffset(.{
            .a = 0,
            .z = 0,
            .y = convolution.kernel_padding,
            .x = convolution.kernel_padding,
        });
        linearized.binarySet(convolution.temp_input_padded, in);
        convolution.temp_input_padded.moveResize(.{
            .a = 1,
            .z = convolution.size_in.z,
            .y = convolution.kernel_size,
            .x = convolution.kernel_size,
        });
        var filter_idx: u32 = 0;
        while (filter_idx < convolution.filters) : (filter_idx += 1) {
            convolution.biases.moveOffset(Vec4.splat(0).setA(filter_idx));
            convolution.weights.moveOffset(Vec4.splat(0).setA(filter_idx));
            var y_out_idx: u32 = 0;
            while (y_out_idx < y_out) : (y_out_idx += 1) {
                var x_out_idx: u32 = 0;
                while (x_out_idx < x_out) : (x_out_idx += 1) {
                    out.moveOffset(.{ .a = 0, .z = filter_idx, .y = y_out_idx, .x = x_out_idx });
                    convolution.temp_input_padded.moveOffset(.{
                        .a = 0,
                        .z = 0,
                        .y = y_out_idx * convolution.kernel_stride,
                        .x = x_out_idx * convolution.kernel_stride,
                    });
                    linearized.binarySet(convolution.temp_kernel, convolution.temp_input_padded);
                    linearized.binaryMultiply(convolution.temp_kernel, convolution.weights);
                    linearized.reduceSum(out.*, convolution.temp_kernel);
                    linearized.binaryAdd(out.*, convolution.biases);
                }
            }
        }
        convolution.biases.moveResize(Vec4.splat(1).setA(convolution.filters));
        convolution.biases.moveOffset(Vec4.splat(0));
        convolution.weights.moveResize(.{
            .a = convolution.filters,
            .z = convolution.size_in.z,
            .y = convolution.kernel_size,
            .x = convolution.kernel_size,
        });
        convolution.weights.moveOffset(Vec4.splat(0));
        out.moveResize(.{ .a = 1, .z = convolution.filters, .y = y_out, .x = x_out });
        out.moveOffset(Vec4.splat(0));
        convolution.temp_input_padded.moveResize(.{
            .a = 1,
            .z = convolution.size_in.z,
            .y = y_out + 2 * convolution.kernel_padding,
            .x = x_out + 2 * convolution.kernel_padding,
        });
        convolution.temp_input_padded.moveOffset(Vec4.splat(0));
    }
    pub fn backward(convolution: *Convolution, linearized: *Linearized, in: Buffer, in_g: *Buffer, out: Buffer, out_g: *Buffer) void {
        assert(convolution.size_in.a == 1);
        assert(convolution.size_in.equal(in.view().size));
        assert(in.view().overlapsAll(in_g.view().*));
        assert(out.view().size.z == convolution.filters);
        assert(out.view().size.y == sizeNew(convolution.size_in.y, convolution.kernel_size, //
            convolution.kernel_stride, convolution.kernel_padding));
        assert(out.view().size.x == sizeNew(convolution.size_in.x, convolution.kernel_size, //
            convolution.kernel_stride, convolution.kernel_padding));
        assert(out.view().overlapsAll(out_g.view().*));

        const y_out: u32 = out.view().size.y;
        const x_out: u32 = out.view().size.x;

        convolution.biases_g.moveResize(Vec4.splat(1));
        out_g.moveResize(.{ .a = 1, .z = 1, .y = y_out, .x = x_out });
        var filter_idx: u32 = 0;
        while (filter_idx < convolution.filters) : (filter_idx += 1) {
            convolution.biases_g.moveOffset(Vec4.splat(0).setA(filter_idx));
            out_g.moveOffset(Vec4.splat(0).setZ(filter_idx));
            linearized.reduceSum(convolution.temp_single, out_g.*); // Could do avg here for move numerical stability
            linearized.binaryAdd(convolution.biases_g, convolution.temp_single);
        }
        convolution.biases_g.moveResize(Vec4.splat(1).setA(convolution.filters));
        convolution.biases_g.moveOffset(Vec4.splat(0));

        out_g.moveResize(Vec4.splat(1));
        out_g.moveOffset(Vec4.splat(0));
        convolution.weights_g.moveResize(.{
            .a = 1,
            .z = convolution.size_in.z,
            .y = convolution.kernel_size,
            .x = convolution.kernel_size,
        });
        convolution.temp_input_padded.moveResize(.{
            .a = 1,
            .z = convolution.size_in.z,
            .y = convolution.kernel_size,
            .x = convolution.kernel_size,
        });
        filter_idx = 0;
        while (filter_idx < convolution.filters) : (filter_idx += 1) {
            convolution.weights_g.moveOffset(Vec4.splat(0).setA(filter_idx));
            var y_out_idx: u32 = 0;
            while (y_out_idx < y_out) : (y_out_idx += 1) {
                var x_out_idx: u32 = 0;
                while (x_out_idx < x_out) : (x_out_idx += 1) {
                    out_g.moveOffset(.{ .a = 0, .z = filter_idx, .y = y_out_idx, .x = x_out_idx });
                    convolution.temp_input_padded.moveOffset(.{
                        .a = 0,
                        .z = 0,
                        .y = y_out_idx * convolution.kernel_stride,
                        .x = x_out_idx * convolution.kernel_stride,
                    });
                    linearized.binarySet(convolution.temp_kernel, convolution.temp_input_padded);
                    linearized.expandMultiply(convolution.temp_kernel, out_g.*);
                    linearized.binaryAdd(convolution.weights_g, convolution.temp_kernel);
                }
            }
        }
        convolution.weights_g.moveResize(.{
            .a = convolution.filters,
            .z = convolution.size_in.z,
            .y = convolution.kernel_size,
            .x = convolution.kernel_size,
        });
        convolution.weights_g.moveOffset(Vec4.splat(0));

        out_g.moveResize(Vec4.splat(1));
        out_g.moveOffset(Vec4.splat(0));
        convolution.weights.moveResize(.{
            .a = 1,
            .z = convolution.size_in.z,
            .y = convolution.kernel_size,
            .x = convolution.kernel_size,
        });
        convolution.weights.moveOffset(Vec4.splat(0));
        convolution.temp_grad_padded.moveResize(.{
            .a = 1,
            .z = convolution.size_in.z,
            .y = convolution.kernel_size,
            .x = convolution.kernel_size,
        });
        convolution.temp_grad_padded.moveOffset(Vec4.splat(0));
        filter_idx = 0;
        while (filter_idx < convolution.filters) : (filter_idx += 1) {
            convolution.weights.moveOffset(Vec4.splat(0).setA(filter_idx));
            var y_out_idx: u32 = 0;
            while (y_out_idx < y_out) : (y_out_idx += 1) {
                var x_out_idx: u32 = 0;
                while (x_out_idx < x_out) : (x_out_idx += 1) {
                    out_g.moveOffset(.{ .a = 0, .z = filter_idx, .y = y_out_idx, .x = x_out_idx });
                    convolution.temp_grad_padded.moveOffset(.{
                        .a = 0,
                        .z = 0,
                        .y = y_out_idx * convolution.kernel_padding,
                        .x = x_out_idx * convolution.kernel_padding,
                    });
                    linearized.binarySet(convolution.temp_kernel, convolution.weights);
                    linearized.expandMultiply(convolution.temp_kernel, out_g.*);
                    linearized.binaryAdd(convolution.temp_grad_padded, convolution.temp_kernel);
                }
            }
        }
        out_g.moveResize(.{ .a = 1, .z = convolution.filters, .y = y_out, .x = x_out });
        out_g.moveOffset(Vec4.splat(0));
        convolution.weights.moveResize(.{
            .a = convolution.filters,
            .z = convolution.size_in.z,
            .y = convolution.kernel_size,
            .x = convolution.kernel_size,
        });
        convolution.weights.moveOffset(Vec4.splat(0));
        convolution.temp_grad_padded.moveResize(convolution.size_in);
        convolution.temp_grad_padded.moveOffset(.{
            .a = 0,
            .z = 0,
            .y = convolution.kernel_padding,
            .x = convolution.kernel_padding,
        });
        linearized.binaryAdd(in_g.*, convolution.temp_grad_padded);
        convolution.temp_grad_padded.moveResize(.{
            .a = 1,
            .z = convolution.size_in.z,
            .y = convolution.size_in.y + 2 * convolution.kernel_padding,
            .x = convolution.size_in.x + 2 * convolution.kernel_padding,
        });
        convolution.temp_grad_padded.moveOffset(Vec4.splat(0));
    }
    pub fn print(convolution: Convolution, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            util.log.print("{s}Convolution {s}\n", .{ " " ** offset, text });
        } else {
            util.log.print("{s}Convolution\n", .{" " ** offset});
        }
        util.log.print("{s}In (1, {}, {}, {}), Size {}, Stride {}, Padding {}\n", .{
            " " ** (offset + padding), //
            convolution.z_in,        convolution.y_in,          convolution.x_in, //
            convolution.kernel_size, convolution.kernel_stride, convolution.kernel_padding,
        });
    }
    pub fn debug(convolution: Convolution, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        convolution.print(padding, offset, name);
        convolution.weights.print(padding, padding + offset, "weights");
        convolution.weights_g.print(padding, padding + offset, "weights_g");
        convolution.biases.print(padding, padding + offset, "biases");
        convolution.biases_g.print(padding, padding + offset, "biases_g");
        convolution.temp_input_padded.print(padding, padding + offset, "temp_input_padded");
        convolution.temp_grad_padded.print(padding, padding + offset, "temp_grad_padded");
        convolution.temp_kernel.print(padding, padding + offset, "temp_kernel");
        convolution.temp_single.print(padding, padding + offset, "temp_single");
    }
};
pub const Reduce = struct {
    pub const Kind = enum(u8) { sum, avg, max, min };
    size_in: Vec4,
    kernel_size: u32,
    kernel_stride: u32,
    t: Reduce.Kind,
    pub inline fn sizeNew(dim_size: u32, size: u32, stride: u32) u32 {
        assert(dim_size >= size);
        return @divFloor(dim_size - size, stride) + 1;
    }
    pub fn init(size_in: Vec4, kernel_size: u32, kernel_stride: u32, t: Reduce.Kind) Reduce {
        assert(size_in.a == 1);
        assert(size_in.z > 0);
        assert(size_in.y > 0);
        assert(size_in.x > 0);
        assert(kernel_size > 0);
        assert(kernel_stride > 0);
        assert(size_in.y >= kernel_size);
        assert(size_in.x >= kernel_size);
        return .{
            .size_in = size_in,
            .kernel_size = kernel_size,
            .kernel_stride = kernel_stride,
            .t = t,
        };
    }
    pub fn forward(reduce: *Reduce, linearized: *Linearized, in: *Buffer, out: *Buffer) void {
        assert(in.view().size.a == 1);
        assert(in.view().size.equal(reduce.size_in));
        assert(out.view().size.a == 1);
        assert(out.view().size.z == in.view().size.z);
        assert(out.view().size.y ==
            sizeNew(in.view().size.y, reduce.kernel_size, reduce.kernel_stride));
        assert(out.view().size.x ==
            sizeNew(in.view().size.x, reduce.kernel_size, reduce.kernel_stride));

        const y_out: u32 = out.view().size.y;
        const x_out: u32 = out.view().size.x;
        in.moveResize(.{ .a = 1, .z = 1, .y = reduce.kernel_size, .x = reduce.kernel_size });
        out.moveResize(Vec4.splat(1));
        var z_idx: u32 = 0;
        while (z_idx < reduce.size_in.z) : (z_idx += 1) {
            var y_out_idx: u32 = 0;
            while (y_out_idx < y_out) : (y_out_idx += 1) {
                var x_out_idx: u32 = 0;
                while (x_out_idx < x_out) : (x_out_idx += 1) {
                    in.moveOffset(.{
                        .a = 0,
                        .z = z_idx,
                        .y = y_out_idx * reduce.kernel_stride,
                        .x = x_out_idx * reduce.kernel_stride,
                    });
                    out.moveOffset(.{ .a = 0, .z = z_idx, .y = y_out_idx, .x = x_out_idx });
                    switch (reduce.t) {
                        .sum => linearized.reduceSum(out.*, in.*),
                        .avg => linearized.reduceAvg(out.*, in.*),
                        .max => linearized.reduceMax(out.*, in.*),
                        .min => linearized.reduceMin(out.*, in.*),
                    }
                }
            }
        }
    }
    // $FIXME This backprop is just straight up wrong
    //     but it at least **somewhat** approximates the correct solution.
    pub fn backward(reduce: *Reduce, linearized: *Linearized, in_g: *Buffer, out_g: *Buffer) void {
        assert(in_g.view().size.a == 1);
        assert(in_g.view().size.equal(reduce.size_in));
        assert(out_g.view().size.a == 1);
        assert(out_g.view().size.z == in_g.view().size.z);
        assert(out_g.view().size.y == sizeNew(in_g.view().size.y, reduce.kernel_size, reduce.kernel_stride));
        assert(out_g.view().size.x == sizeNew(in_g.view().size.x, reduce.kernel_size, reduce.kernel_stride));
        if (reduce.t != .sum) util.todo(@src());

        const size_y_out: u32 = out_g.view().size.y;
        const size_x_out: u32 = out_g.view().size.x;

        in_g.moveResize(.{ .a = 1, .z = 1, .y = reduce.kernel_size, .x = reduce.kernel_size });
        out_g.moveResize(Vec4.splat(1));

        var z_idx: u32 = 0;
        while (z_idx < reduce.size_in.z) : (z_idx += 1) {
            var y_out_idx: u32 = 0;
            while (y_out_idx < size_y_out) : (y_out_idx += 1) {
                var x_out_idx: u32 = 0;
                while (x_out_idx < size_x_out) : (x_out_idx += 1) {
                    in_g.moveOffset(.{
                        .a = 0,
                        .z = z_idx,
                        .y = y_out_idx * reduce.kernel_stride,
                        .x = x_out_idx * reduce.kernel_stride,
                    });
                    out_g.moveOffset(.{ .a = 0, .z = z_idx, .y = y_out_idx, .x = x_out_idx });
                    linearized.expandAdd(in_g.*, out_g.*);
                }
            }
        }
        in_g.moveResize(.{ .a = 1, .z = reduce.size_in.z, .y = reduce.size_in.y, .x = reduce.size_in.x });
        in_g.moveOffset(Vec4.splat(0));
        out_g.moveResize(.{ .a = 1, .z = reduce.size_in.z, .y = size_y_out, .x = size_x_out });
        out_g.moveOffset(Vec4.splat(0));
    }
    pub fn print(reduce: Reduce, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            util.log.print("{s}Reduce {s}\n", .{ " " ** offset, text });
        } else {
            util.log.print("{s}Reduce\n", .{" " ** offset});
        }
        util.log.print("{s}In (1, {}, {}, {}), Size {}, Stride {}\n", .{
            " " ** (offset + padding), //
            reduce.z_in,        reduce.y_in,          reduce.x_in, //
            reduce.kernel_size, reduce.kernel_stride,
        });
    }
};
pub const Split = struct {
    filters: u32,
    size_in: Vec4,

    weights: Buffer,
    weights_g: Buffer,
    biases: Buffer,
    biases_g: Buffer,

    temp_in: Buffer,
    pub fn alloc(runtime: Runtime, gpa: Allocator, arena: Allocator, size_in: Vec4, filters: u32) !Split {
        assert(filters > 0);
        assert(size_in.a == 1);
        assert(size_in.z > 0);
        assert(size_in.y > 0);
        assert(size_in.x > 0);
        const size_in_with_filters: Vec4 = .{
            .a = filters,
            .z = size_in.z,
            .y = size_in.y,
            .x = size_in.x,
        };
        return .{
            .filters = filters,
            .size_in = size_in,
            .weights = try Buffer.alloc(runtime, gpa, arena, size_in_with_filters, .normal),
            .weights_g = try Buffer.alloc(runtime, gpa, arena, size_in_with_filters, .normal),
            .biases = try Buffer.alloc(runtime, gpa, arena, size_in_with_filters, .normal),
            .biases_g = try Buffer.alloc(runtime, gpa, arena, size_in_with_filters, .normal),
            .temp_in = try Buffer.alloc(runtime, gpa, arena, size_in, .intermediary),
        };
    }
    pub fn free(split: Split, runtime: Runtime) void {
        split.weights.free(runtime);
        split.weights_g.free(runtime);
        split.biases.free(runtime);
        split.biases_g.free(runtime);
        split.temp_in.free(runtime);
    }
    pub fn forward(split: *Split, linearized: *Linearized, in: Buffer, out: *Buffer) void {
        const out_view: View = out.view().*;
        const in_view: View = in.view().*;
        assert(in_view.size.a == 1);
        assert(in_view.size.equal(split.size_in));
        assert(out_view.size.a == 1);
        assert(out_view.size.equal(split.size_in.setZ(split.size_in.z * split.filters)));

        split.weights.moveResize(split.size_in);
        split.biases.moveResize(split.size_in);
        out.moveResize(split.size_in);

        var filter_idx: u32 = 0;
        while (filter_idx < split.filters) : (filter_idx += 1) {
            split.weights.moveOffset(Vec4.splat(0).setA(filter_idx));
            split.biases.moveOffset(Vec4.splat(0).setA(filter_idx));
            out.moveOffset(Vec4.splat(0).setZ(filter_idx * split.size_in.z));
            linearized.binarySet(out.*, in);
            linearized.binaryMultiply(out.*, split.weights);
            linearized.binaryAdd(out.*, split.biases);
        }
        split.weights.moveResize(split.size_in.setA(split.filters));
        split.weights.moveOffset(Vec4.splat(0));
        split.biases.moveResize(split.size_in.setA(split.filters));
        split.biases.moveOffset(Vec4.splat(0));
        out.moveResize(split.size_in.setZ(split.size_in.z * split.filters));
        out.moveOffset(Vec4.splat(0));
    }
    pub fn backward(split: *Split, linearized: *Linearized, in: Buffer, in_g: Buffer, out_g: *Buffer) void {
        const in_view: View = in.view().*;
        const in_g_view: View = in_g.view().*;
        const out_g_view: View = out_g.view().*;
        assert(in_view.size.equal(split.size_in));
        assert(in_view.overlapsAll(in_g_view));
        assert(out_g_view.size.a == 1);
        assert(out_g_view.size.equal(split.size_in.setZ(split.size_in.z * split.filters)));

        split.biases_g.moveResize(split.size_in.setZ(split.size_in.z * split.filters));
        linearized.binarySet(split.biases_g, out_g.*);
        split.biases_g.moveResize(split.size_in.setA(split.filters));

        split.weights_g.moveResize(split.size_in);
        out_g.moveResize(split.size_in);
        var filter_idx: u32 = 0;
        while (filter_idx < split.filters) : (filter_idx += 1) {
            split.weights_g.moveOffset(Vec4.splat(0).setA(filter_idx));
            out_g.moveOffset(Vec4.splat(0).setZ(filter_idx * split.filters));
            linearized.binarySet(split.temp_in, out_g.*);
            linearized.binaryMultiply(split.temp_in, in);
            linearized.binaryAdd(split.weights_g, split.temp_in);
        }
        split.weights_g.moveResize(split.size_in.setA(split.filters));
        split.weights_g.moveOffset(Vec4.splat(0));

        split.weights.moveResize(split.size_in);
        filter_idx = 0;
        while (filter_idx < split.filters) : (filter_idx += 1) {
            split.weights.moveOffset(Vec4.splat(0).setA(filter_idx));
            out_g.moveOffset(Vec4.splat(0).setZ(filter_idx * split.filters));
            linearized.binarySet(split.temp_in, out_g.*);
            linearized.binaryMultiply(split.temp_in, split.weights);
            linearized.binaryAdd(in_g, split.temp_in);
        }
        split.weights.moveResize(split.size_in.setA(split.filters));
        split.weights.moveOffset(Vec4.splat(0));
        out_g.moveResize(split.size_in.setZ(split.size_in.z * split.filters));
        out_g.moveOffset(Vec4.splat(0));
    }
    pub fn print(split: Split, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            util.log.print("{s}Reduce {s}\n", .{ " " ** offset, text });
        } else {
            util.log.print("{s}Reduce\n", .{" " ** offset});
        }
        util.log.print("{s}In (1, {}, {}, {}), filters {}\n", .{
            " " ** (offset + padding),
            split.z_in,
            split.y_in,
            split.x_in,
            split.filters,
        });
    }
    pub fn debug(split: Split, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        split.print(padding, offset, name);
        split.weights.print(padding, offset + padding, "weights");
        split.weights_g.print(padding, offset + padding, "weights_g");
        split.biases.print(padding, offset + padding, "biases");
        split.biases_g.print(padding, offset + padding, "biases_g");
        split.temp_in.print(padding, offset + padding, "temp_in");
    }
};
pub const Residual = struct {
    pub const Kind = enum(u8) {
        identity,
    };
    t: Residual.Kind,
    in_layer: u32,
    pub fn forward(residual: *Residual, in: *Buffer, out: *Buffer) void {
        assert(in.buffer.overlapsAll(out.buffer));
        assert(in.buffer.id != out.buffer.id);
        switch (residual.t) {
            .identity => {
                out.binaryAdd(&in);
            },
        }
    }
    pub fn backward(residual: *Residual, in_g: *Buffer, out_g: *Buffer) void {
        assert(in_g.buffer.overlapsAll(out_g));
        assert(in_g.buffer.id != out_g.buffer.id);
        switch (residual.t) {
            .identity => {
                in_g.binaryAdd(&out_g);
            },
        }
    }
    pub fn print(residual: Residual, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            util.log.print("{s}Reduce {s}\n", .{ " " ** offset, text });
        } else {
            util.log.print("{s}Reduce\n", .{" " ** offset});
        }
        util.log.print("{s}In \"{s}\", Out \"{s}\"\n", .{
            " " ** (offset + padding), //
            residual.in.buffer.name(), residual.out.buffer.name(), //
        });
    }
    pub fn debug(residual: Residual, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        residual.print(padding, offset, name);
        residual.in.print(padding, offset + padding, "in");
        residual.out.print(padding, offset + padding, "out");
    }
};

pub const Layer = @This();
// $TODO Maybe just add values, values_g, activation and norming to the sub-structs
//  This would make the layer just the `tag` field which could be nicer
pub const Kind = enum(u8) { dense, convolution, reduce, split, residual };
tag: union(Kind) {
    dense: Dense,
    convolution: Convolution,
    reduce: Reduce,
    split: Split,
    residual: Residual,
},
values: Buffer,
values_g: Buffer,
activation: Activation,
pub const Config = union(Kind) {
    dense: struct {
        size_out: u32,
        activation_kind: Activation.Kind,
    },
    convolution: struct {
        filters: u32,
        kernel_size: u32,
        kernel_stride: u32,
        kernel_padding: u32,
        activation_kind: Activation.Kind,
    },
    reduce: struct {
        kernel_size: u32,
        kernel_stride: u32,
        t: Reduce.Kind,
    },
    split: struct {
        filters: u32,
        activation_kind: Activation.Kind,
    },
    residual: struct {
        in_layer: u32,
        t: Residual.Kind,
    },
};
