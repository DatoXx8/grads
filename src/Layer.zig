const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const Optimization = @import("compiler/optimize.zig").Optimization;
const Program = @import("compiler/Program.zig");
const Runtime = @import("compiler/runtimes/Runtime.zig");
const Linearized = @import("Linearized.zig");
const Op = Linearized.Op;
const Buffer = @import("Buffer.zig");
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
    pub fn alloc(runtime: Runtime, arena: Allocator, t: Activation.Kind, z_in: u32, y_in: u32, x_in: u32) !Activation {
        return .{
            .t = t,
            .temp = switch (t) {
                .none => try Buffer.alloc(runtime, arena, 1, z_in, y_in, x_in, .intermediary),
                .relu => try Buffer.alloc(runtime, arena, 1, z_in, y_in, x_in, .intermediary),
                .sigmoid => try Buffer.alloc(runtime, arena, 1, z_in, y_in, x_in, .intermediary),
                .relu_clipped => try Buffer.alloc(runtime, arena, 1, z_in, y_in, x_in, .intermediary),
                .relu_leaky => try Buffer.alloc(runtime, arena, 1, z_in, y_in, x_in, .intermediary),
                .silu => try Buffer.alloc(runtime, arena, 1, z_in, y_in, x_in, .intermediary),
                .gelu => try Buffer.alloc(runtime, arena, 1, z_in, y_in, x_in, .intermediary),
                .tanh => try Buffer.alloc(runtime, arena, 1, z_in, y_in, x_in, .intermediary),
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
    pub fn alloc(runtime: Runtime, arena: Allocator, size_in: u32, size_out: u32) !Dense {
        assert(size_in > 0);
        assert(size_out > 0);
        return .{
            .size_in = size_in,
            .size_out = size_out,
            .weights = try Buffer.alloc(runtime, arena, 1, 1, size_in, size_out, .normal),
            .weights_g = try Buffer.alloc(runtime, arena, 1, 1, size_in, size_out, .normal),
            .biases = try Buffer.alloc(runtime, arena, 1, 1, 1, size_out, .normal),
            .biases_g = try Buffer.alloc(runtime, arena, 1, 1, 1, size_out, .normal),
            .temp_in = try Buffer.alloc(runtime, arena, 1, 1, size_in, 1, .intermediary),
            .temp_out = try Buffer.alloc(runtime, arena, 1, 1, 1, size_out, .intermediary),
            .temp_full = try Buffer.alloc(runtime, arena, 1, 1, size_in, size_out, .intermediary),
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
        assert(in.a_size == 1);
        assert(in.z_size == 1);
        assert(in.y_size == dense.size_in);
        assert(in.x_size == 1);
        assert(out.a_size == 1);
        assert(out.z_size == 1);
        assert(out.y_size == 1);
        assert(out.x_size == dense.size_out);

        dense.weights.moveResize(1, 1, dense.size_in, 1);
        out.moveResize(1, 1, 1, 1);

        var row_idx: u32 = 0;
        while (row_idx < dense.size_out) : (row_idx += 1) {
            dense.weights.moveOffset(0, 0, 0, row_idx);
            out.moveOffset(0, 0, 0, row_idx);
            linearized.binarySet(dense.temp_in, dense.weights);
            linearized.binaryMultiply(dense.temp_in, in);
            linearized.reduceSum(out.*, dense.temp_in);
        }

        dense.weights.moveResize(1, 1, dense.size_in, dense.size_out);
        dense.weights.moveOffset(0, 0, 0, 0);
        out.moveResize(1, 1, 1, dense.size_out);
        out.moveOffset(0, 0, 0, 0);
        linearized.binaryAdd(out.*, dense.biases);
    }
    pub fn backward(dense: *Dense, linearized: *Linearized, in: Buffer, in_g: *Buffer, out_g: Buffer) void {
        assert(in.a_size == 1);
        assert(in.z_size == 1);
        assert(in.y_size == dense.size_in);
        assert(in.x_size == 1);
        assert(in_g.overlapsAll(in));
        assert(out_g.a_size == 1);
        assert(out_g.z_size == 1);
        assert(out_g.y_size == 1);
        assert(out_g.x_size == dense.size_out);

        linearized.binaryAdd(dense.biases_g, out_g);

        dense.temp_full.moveResize(1, 1, 1, dense.size_out);
        var column_idx: u32 = 0;
        while (column_idx < dense.size_in) : (column_idx += 1) {
            dense.temp_full.moveOffset(0, 0, column_idx, 0);
            linearized.binarySet(dense.temp_full, out_g);
        }
        dense.temp_full.moveResize(1, 1, dense.size_in, 1);
        var row_idx: u32 = 0;
        while (row_idx < dense.size_out) : (row_idx += 1) {
            dense.temp_full.moveOffset(0, 0, 0, row_idx);
            linearized.binaryMultiply(dense.temp_full, in);
        }
        dense.temp_full.moveResize(1, 1, dense.size_in, dense.size_out);
        dense.temp_full.moveOffset(0, 0, 0, 0);
        linearized.binaryAdd(dense.weights_g, dense.temp_full);

        in_g.moveResize(1, 1, 1, 1);
        dense.weights.moveResize(1, 1, 1, dense.size_out);
        column_idx = 0;
        while (column_idx < dense.size_in) : (column_idx += 1) {
            in_g.moveOffset(0, 0, column_idx, 0);
            dense.weights.moveOffset(0, 0, column_idx, 0);
            linearized.binarySet(dense.temp_out, dense.weights);
            linearized.binaryMultiply(dense.temp_out, out_g);
            linearized.reduceSum(in_g.*, dense.temp_out); // Could do avg here for move numerical stability
        }
        in_g.moveResize(1, 1, dense.size_in, 1);
        in_g.moveOffset(0, 0, 0, 0);
        dense.weights.moveResize(1, 1, dense.size_in, dense.size_out);
        dense.weights.moveOffset(0, 0, 0, 0);
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
    z_in: u32,
    y_in: u32,
    x_in: u32,
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
        arena: Allocator,
        z_in: u32,
        y_in: u32,
        x_in: u32,
        filters: u32,
        kernel_size: u32,
        kernel_stride: u32,
        kernel_padding: u32,
    ) !Convolution {
        assert(filters > 0);
        assert(kernel_stride > 0);
        assert(kernel_size > 0);
        assert(z_in > 0);
        assert(y_in >= kernel_size);
        assert(x_in >= kernel_size);
        return .{
            .z_in = z_in,
            .y_in = y_in,
            .x_in = x_in,
            .filters = filters,
            .kernel_size = kernel_size,
            .kernel_stride = kernel_stride,
            .kernel_padding = kernel_padding,
            .weights = try Buffer.alloc(runtime, arena, filters, z_in, kernel_size, kernel_size, .normal),
            .weights_g = try Buffer.alloc(runtime, arena, filters, z_in, kernel_size, kernel_size, .normal),
            .biases = try Buffer.alloc(runtime, arena, filters, 1, 1, 1, .normal),
            .biases_g = try Buffer.alloc(runtime, arena, filters, 1, 1, 1, .normal),
            .temp_input_padded = try Buffer.alloc(runtime, arena, 1, z_in, //
                y_in + 2 * kernel_padding, x_in + 2 * kernel_padding, .intermediary),
            .temp_grad_padded = try Buffer.alloc(runtime, arena, 1, z_in, //
                y_in + 2 * kernel_padding, x_in + 2 * kernel_padding, .intermediary),
            .temp_kernel = try Buffer.alloc(runtime, arena, 1, z_in, kernel_size, kernel_size, .intermediary),
            .temp_single = try Buffer.alloc(runtime, arena, 1, 1, 1, 1, .intermediary),
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
        assert(1 == in.a_size);
        assert(convolution.z_in == in.z_size);
        assert(convolution.y_in == in.y_size);
        assert(convolution.x_in == in.x_size);
        assert(convolution.filters == out.z_size);
        assert(out.y_size == sizeNew(convolution.y_in, convolution.kernel_size, convolution.kernel_stride, convolution.kernel_padding));
        assert(out.x_size == sizeNew(convolution.x_in, convolution.kernel_size, convolution.kernel_stride, convolution.kernel_padding));

        const y_out: u32 = out.y_size;
        const x_out: u32 = out.x_size;

        out.moveResize(1, 1, 1, 1);
        convolution.biases.moveResize(1, 1, 1, 1);
        convolution.weights.moveResize(1, convolution.z_in, convolution.kernel_size, convolution.kernel_size);
        convolution.temp_input_padded.moveResize(1, convolution.z_in, convolution.y_in, convolution.x_in);
        convolution.temp_input_padded.moveOffset(0, 0, convolution.kernel_padding, convolution.kernel_padding);
        linearized.binarySet(convolution.temp_input_padded, in);
        convolution.temp_input_padded.moveResize(1, convolution.z_in, convolution.kernel_size, convolution.kernel_size);
        var filter_idx: u32 = 0;
        while (filter_idx < convolution.filters) : (filter_idx += 1) {
            convolution.biases.moveOffset(filter_idx, 0, 0, 0);
            convolution.weights.moveOffset(filter_idx, 0, 0, 0);
            var y_out_idx: u32 = 0;
            while (y_out_idx < y_out) : (y_out_idx += 1) {
                var x_out_idx: u32 = 0;
                while (x_out_idx < x_out) : (x_out_idx += 1) {
                    out.moveOffset(0, filter_idx, y_out_idx, x_out_idx);
                    convolution.temp_input_padded.moveOffset(0, 0, y_out_idx * convolution.kernel_stride, //
                        x_out_idx * convolution.kernel_stride);
                    linearized.binarySet(convolution.temp_kernel, convolution.temp_input_padded);
                    linearized.binaryMultiply(convolution.temp_kernel, convolution.weights);
                    linearized.reduceSum(out.*, convolution.temp_kernel);
                    linearized.binaryAdd(out.*, convolution.biases);
                }
            }
        }
        convolution.biases.moveResize(convolution.filters, 1, 1, 1);
        convolution.biases.moveOffset(0, 0, 0, 0);
        convolution.weights.moveResize(convolution.filters, convolution.z_in, convolution.kernel_size, convolution.kernel_size);
        convolution.weights.moveOffset(0, 0, 0, 0);
        out.moveResize(1, convolution.filters, y_out, x_out);
        out.moveOffset(0, 0, 0, 0);
        convolution.temp_input_padded.moveResize(1, convolution.z_in, //
            y_out + 2 * convolution.kernel_padding, x_out + 2 * convolution.kernel_padding);
        convolution.temp_input_padded.moveOffset(0, 0, 0, 0);
    }
    pub fn backward(convolution: *Convolution, linearized: *Linearized, in: Buffer, in_g: *Buffer, out: Buffer, out_g: *Buffer) void {
        assert(1 == in.a_size);
        assert(convolution.z_in == in.z_size);
        assert(convolution.y_in == in.y_size);
        assert(convolution.x_in == in.x_size);
        assert(in.overlapsAll(in_g.*));
        assert(convolution.filters == out.z_size);
        assert(out.y_size == sizeNew(convolution.y_in, convolution.kernel_size, convolution.kernel_stride, convolution.kernel_padding));
        assert(out.x_size == sizeNew(convolution.x_in, convolution.kernel_size, convolution.kernel_stride, convolution.kernel_padding));
        assert(out.overlapsAll(out_g.*));

        const y_out: u32 = out.y_size;
        const x_out: u32 = out.x_size;

        convolution.biases_g.moveResize(1, 1, 1, 1);
        out_g.moveResize(1, 1, y_out, x_out);
        var filter_idx: u32 = 0;
        while (filter_idx < convolution.filters) : (filter_idx += 1) {
            convolution.biases_g.moveOffset(filter_idx, 0, 0, 0);
            out_g.moveOffset(0, filter_idx, 0, 0);
            linearized.reduceSum(convolution.temp_single, out_g.*); // Could do avg here for move numerical stability
            linearized.binaryAdd(convolution.biases_g, convolution.temp_single);
        }
        convolution.biases_g.moveResize(convolution.filters, 1, 1, 1);
        convolution.biases_g.moveOffset(0, 0, 0, 0);

        out_g.moveResize(1, 1, 1, 1);
        out_g.moveOffset(0, 0, 0, 0);
        convolution.weights_g.moveResize(1, convolution.z_in, convolution.kernel_size, convolution.kernel_size);
        convolution.temp_input_padded.moveResize(1, convolution.z_in, convolution.kernel_size, convolution.kernel_size);
        filter_idx = 0;
        while (filter_idx < convolution.filters) : (filter_idx += 1) {
            convolution.weights_g.moveOffset(filter_idx, 0, 0, 0);
            var y_out_idx: u32 = 0;
            while (y_out_idx < y_out) : (y_out_idx += 1) {
                var x_out_idx: u32 = 0;
                while (x_out_idx < x_out) : (x_out_idx += 1) {
                    out_g.moveOffset(0, filter_idx, y_out_idx, x_out_idx);
                    convolution.temp_input_padded.moveOffset(0, 0, y_out_idx * convolution.kernel_stride, x_out_idx * convolution.kernel_stride);
                    linearized.binarySet(convolution.temp_kernel, convolution.temp_input_padded);
                    linearized.expandMultiply(convolution.temp_kernel, out_g.*);
                    linearized.binaryAdd(convolution.weights_g, convolution.temp_kernel);
                }
            }
        }
        convolution.weights_g.moveResize(convolution.filters, convolution.z_in, convolution.kernel_size, convolution.kernel_size);
        convolution.weights_g.moveOffset(0, 0, 0, 0);

        out_g.moveResize(1, 1, 1, 1);
        out_g.moveOffset(0, 0, 0, 0);
        convolution.weights.moveResize(1, convolution.z_in, convolution.kernel_size, convolution.kernel_size);
        convolution.weights.moveOffset(0, 0, 0, 0);
        convolution.temp_grad_padded.moveResize(1, convolution.z_in, convolution.kernel_size, convolution.kernel_size);
        convolution.temp_grad_padded.moveOffset(0, 0, 0, 0);
        filter_idx = 0;
        while (filter_idx < convolution.filters) : (filter_idx += 1) {
            convolution.weights.moveOffset(filter_idx, 0, 0, 0);
            var y_out_idx: u32 = 0;
            while (y_out_idx < y_out) : (y_out_idx += 1) {
                var x_out_idx: u32 = 0;
                while (x_out_idx < x_out) : (x_out_idx += 1) {
                    out_g.moveOffset(0, filter_idx, y_out_idx, x_out_idx);
                    convolution.temp_grad_padded.moveOffset(0, 0, y_out_idx * convolution.kernel_padding, //
                        x_out_idx * convolution.kernel_padding);
                    linearized.binarySet(convolution.temp_kernel, convolution.weights);
                    linearized.expandMultiply(convolution.temp_kernel, out_g.*);
                    linearized.binaryAdd(convolution.temp_grad_padded, convolution.temp_kernel);
                }
            }
        }
        out_g.moveResize(1, convolution.filters, y_out, x_out);
        out_g.moveOffset(0, 0, 0, 0);
        convolution.weights.moveResize(convolution.filters, convolution.z_in, convolution.kernel_size, convolution.kernel_size);
        convolution.weights.moveOffset(0, 0, 0, 0);
        convolution.temp_grad_padded.moveResize(1, convolution.z_in, convolution.y_in, convolution.x_in);
        convolution.temp_grad_padded.moveOffset(0, 0, convolution.kernel_padding, convolution.kernel_padding);
        linearized.binaryAdd(in_g.*, convolution.temp_grad_padded);
        convolution.temp_grad_padded.moveResize(1, convolution.z_in, convolution.y_in + 2 * convolution.kernel_padding, //
            convolution.x_in + 2 * convolution.kernel_padding);
        convolution.temp_grad_padded.moveOffset(0, 0, 0, 0);
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
    z_in: u32,
    y_in: u32,
    x_in: u32,
    kernel_size: u32,
    kernel_stride: u32,
    t: Reduce.Kind,
    pub inline fn sizeNew(dim_size: u32, size: u32, stride: u32) u32 {
        assert(dim_size >= size);
        return @divFloor(dim_size - size, stride) + 1;
    }
    pub fn init(z_in: u32, y_in: u32, x_in: u32, kernel_size: u32, kernel_stride: u32, t: Reduce.Kind) Reduce {
        assert(z_in > 0);
        assert(y_in > 0);
        assert(x_in > 0);
        assert(kernel_size > 0);
        assert(kernel_stride > 0);
        assert(y_in >= kernel_size);
        assert(x_in >= kernel_size);
        return .{
            .z_in = z_in,
            .y_in = y_in,
            .x_in = x_in,
            .kernel_size = kernel_size,
            .kernel_stride = kernel_stride,
            .t = t,
        };
    }
    pub fn forward(reduce: *Reduce, linearized: *Linearized, in: *Buffer, out: *Buffer) void {
        assert(in.a_size == 1);
        assert(in.z_size == reduce.z_in);
        assert(in.y_size == reduce.x_in);
        assert(in.x_size == reduce.y_in);
        assert(out.a_size == 1);
        assert(out.z_size == in.z_size);
        assert(out.y_size == sizeNew(in.y_size, reduce.kernel_size, reduce.kernel_stride));
        assert(out.x_size == sizeNew(in.x_size, reduce.kernel_size, reduce.kernel_stride));

        const y_out: u32 = out.y_size;
        const x_out: u32 = out.x_size;
        in.moveResize(1, 1, reduce.kernel_size, reduce.kernel_size);
        out.moveResize(1, 1, 1, 1);
        var z_idx: u32 = 0;
        while (z_idx < reduce.z_in) : (z_idx += 1) {
            var y_out_idx: u32 = 0;
            while (y_out_idx < y_out) : (y_out_idx += 1) {
                var x_out_idx: u32 = 0;
                while (x_out_idx < x_out) : (x_out_idx += 1) {
                    in.moveOffset(0, z_idx, y_out_idx * reduce.kernel_stride, x_out_idx * reduce.kernel_stride);
                    out.moveOffset(0, z_idx, y_out_idx, x_out_idx);
                    switch (reduce.t) { // The branch predictor should do this ezpz
                        .sum => linearized.reduceSum(out.*, in.*),
                        .avg => linearized.reduceAvg(out.*, in.*),
                        .max => linearized.reduceMax(out.*, in.*),
                        .min => linearized.reduceMin(out.*, in.*),
                    }
                }
            }
        }
    }
    /// $FIXME This backprop is just straight up wrong
    ///     but it at least **somewhat** approximates the correct solution.
    pub fn backward(reduce: *Reduce, linearized: *Linearized, in_g: *Buffer, out_g: *Buffer) void {
        assert(in_g.a_size == 1);
        assert(in_g.z_size == reduce.z_in);
        assert(in_g.y_size == reduce.x_in);
        assert(in_g.x_size == reduce.y_in);
        assert(out_g.a_size == 1);
        assert(out_g.z_size == in_g.z_size);
        assert(out_g.y_size == sizeNew(in_g.y_size, reduce.kernel_size, reduce.kernel_stride));
        assert(out_g.x_size == sizeNew(in_g.x_size, reduce.kernel_size, reduce.kernel_stride));
        if (reduce.t != .sum) util.todo(@src());

        in_g.moveResize(1, 1, reduce.kernel_size, reduce.kernel_size);
        out_g.moveResize(1, 1, 1, 1);

        const y_out: u32 = out_g.y_size;
        const x_out: u32 = out_g.x_size;
        in_g.moveResize(1, 1, reduce.kernel_size, reduce.kernel_size);
        out_g.moveResize(1, 1, 1, 1);
        var z_idx: u32 = 0;
        while (z_idx < reduce.z_in) : (z_idx += 1) {
            var y_out_idx: u32 = 0;
            while (y_out_idx < y_out) : (y_out_idx += 1) {
                var x_out_idx: u32 = 0;
                while (x_out_idx < x_out) : (x_out_idx += 1) {
                    in_g.moveOffset(0, z_idx, y_out_idx * reduce.kernel_stride, x_out_idx * reduce.kernel_stride);
                    out_g.moveOffset(0, z_idx, y_out_idx, x_out_idx);
                    linearized.expandAdd(in_g.*, out_g.*);
                }
            }
        }
        in_g.moveResize(1, reduce.z_in, reduce.y_in, reduce.x_in);
        in_g.moveOffset(0, 0, 0, 0);
        out_g.moveResize(1, reduce.z_in, y_out, x_out);
        out_g.moveOffset(0, 0, 0, 0);
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
    z_in: u32,
    y_in: u32,
    x_in: u32,

    weights: Buffer,
    weights_g: Buffer,
    biases: Buffer,
    biases_g: Buffer,

    temp_in: Buffer,
    pub fn alloc(runtime: Runtime, arena: Allocator, filters: u32, z_in: u32, y_in: u32, x_in: u32) !Split {
        assert(filters > 0);
        assert(z_in > 0);
        assert(y_in > 0);
        assert(x_in > 0);
        return .{
            .filters = filters,
            .z_in = z_in,
            .y_in = y_in,
            .x_in = x_in,
            .weights = try Buffer.alloc(runtime, arena, filters, z_in, y_in, x_in, .normal),
            .weights_g = try Buffer.alloc(runtime, arena, filters, z_in, y_in, x_in, .normal),
            .biases = try Buffer.alloc(runtime, arena, filters, z_in, y_in, x_in, .normal),
            .biases_g = try Buffer.alloc(runtime, arena, filters, z_in, y_in, x_in, .normal),
            .temp_in = try Buffer.alloc(runtime, arena, 1, z_in, y_in, x_in, .intermediary),
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
        assert(in.a_size == 1);
        assert(in.z_size == split.z_in);
        assert(in.y_size == split.y_in);
        assert(in.x_size == split.x_in);
        assert(out.a_size == 1);
        assert(out.z_size == split.z_in * split.filters);
        assert(out.y_size == split.y_in);
        assert(out.x_size == split.x_in);

        split.weights.moveResize(1, split.z_in, split.y_in, split.x_in);
        split.biases.moveResize(1, split.z_in, split.y_in, split.x_in);
        out.moveResize(1, split.z_in, split.y_in, split.x_in);
        var filter_idx: u32 = 0;
        while (filter_idx < split.filters) : (filter_idx += 1) {
            split.weights.moveOffset(filter_idx, 0, 0, 0);
            split.biases.moveOffset(filter_idx, 0, 0, 0);
            out.moveOffset(0, filter_idx * split.z_in, 0, 0);
            linearized.binarySet(out.*, in);
            linearized.binaryMultiply(out.*, split.weights);
            linearized.binaryAdd(out.*, split.biases);
        }
        split.weights.moveResize(split.filters, split.z_in, split.y_in, split.x_in);
        split.weights.moveOffset(0, 0, 0, 0);
        split.biases.moveResize(split.filters, split.z_in, split.y_in, split.x_in);
        split.biases.moveOffset(0, 0, 0, 0);
        out.moveResize(1, split.z_in * split.filters, split.y_in, split.x_in);
        out.moveOffset(0, 0, 0, 0);
    }
    pub fn backward(split: *Split, linearized: *Linearized, in: Buffer, in_g: Buffer, out_g: *Buffer) void {
        assert(in.a_size == 1);
        assert(in.z_size == split.z_in);
        assert(in.y_size == split.y_in);
        assert(in.x_size == split.x_in);
        assert(in.overlapsAll(in_g));
        assert(out_g.a_size == 1);
        assert(out_g.z_size == split.z_in * split.filters);
        assert(out_g.y_size == split.y_in);
        assert(out_g.x_size == split.x_in);

        split.biases_g.moveResize(1, split.z_in * split.filters, split.y_in, split.x_in);
        linearized.binarySet(split.biases_g, out_g.*);
        split.biases_g.moveResize(split.filters, split.z_in, split.y_in, split.x_in);

        split.weights_g.moveResize(1, split.z_in, split.y_in, split.x_in);
        out_g.moveResize(1, split.z_in, split.y_in, split.x_in);
        var filter_idx: u32 = 0;
        while (filter_idx < split.filters) : (filter_idx += 1) {
            split.weights_g.moveOffset(filter_idx, 0, 0, 0);
            out_g.moveOffset(0, filter_idx * split.filters, 0, 0);
            linearized.binarySet(split.temp_in, out_g.*);
            linearized.binaryMultiply(split.temp_in, in);
            linearized.binaryAdd(split.weights_g, split.temp_in);
        }
        split.weights_g.moveResize(split.filters, split.z_in, split.y_in, split.x_in);
        split.weights_g.moveOffset(0, 0, 0, 0);

        split.weights.moveResize(1, split.z_in, split.y_in, split.x_in);
        filter_idx = 0;
        while (filter_idx < split.filters) : (filter_idx += 1) {
            split.weights.moveOffset(filter_idx, 0, 0, 0);
            out_g.moveOffset(0, filter_idx * split.filters, 0, 0);
            linearized.binarySet(split.temp_in, out_g.*);
            linearized.binaryMultiply(split.temp_in, split.weights);
            linearized.binaryAdd(in_g, split.temp_in);
        }
        split.weights.moveResize(split.filters, split.z_in, split.y_in, split.x_in);
        split.weights.moveOffset(0, 0, 0, 0);
        out_g.moveResize(1, split.z_in * split.filters, split.y_in, split.x_in);
        out_g.moveOffset(0, 0, 0, 0);
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
