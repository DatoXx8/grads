const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const Optimization = @import("./compiler/optimize.zig").Optimization;
const Program = @import("./compiler/Program.zig");
const cl = @import("./runtimes/cl.zig");
const ClContext = cl.ClContext;
const ClDevice = cl.ClDevice;
const ClCommandQueue = cl.ClCommandQueue;
const Tensor = @import("./Tensor.zig");
const Linearized = Tensor.Linearized;
const Op = Tensor.Op;
const Buffer = Tensor.Buffer;
const todo = @import("./util.zig").todo;

pub const Activation = struct {
    pub const Type = enum(u32) {
        none,
        relu,
        sigmoid,
        relu_clipped,
        relu_leaky,
        silu,
        gelu,
        tanh, // 1 - tanh^2
    };
    pub const relu_leaky_factor: f32 = 0.1;
    pub const relu_clipped_factor: f32 = 1;
    comptime {
        assert(relu_leaky_factor > 0);
        assert(relu_clipped_factor > 0);
    }
    temp: Tensor,
    t: Activation.Type,
    pub fn alloc(allocator: Allocator, t: Activation.Type, z_in: u32, y_in: u32, x_in: u32, context: ClContext) !Activation {
        return .{
            .t = t,
            .temp = switch (t) {
                .none => try Tensor.allocIntermediary(allocator, 1, z_in, y_in, x_in, context, 1),
                .relu => try Tensor.allocIntermediary(allocator, 1, z_in, y_in, x_in, context, 1),
                .sigmoid => try Tensor.allocIntermediary(allocator, 1, z_in, y_in, x_in, context, 1),
                .relu_clipped => try Tensor.allocIntermediary(allocator, 1, z_in, y_in, x_in, context, 1),
                .relu_leaky => try Tensor.allocIntermediary(allocator, 1, z_in, y_in, x_in, context, 1),
                .silu => try Tensor.allocIntermediary(allocator, 1, z_in, y_in, x_in, context, 1),
                .gelu => try Tensor.allocIntermediary(allocator, 1, z_in, y_in, x_in, context, 1),
                .tanh => try Tensor.allocIntermediary(allocator, 1, z_in, y_in, x_in, context, 1),
            },
        };
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
        this.temp.free(allocator);
    }
    pub fn forward(this: *@This(), values: *Tensor) void {
        switch (this.t) {
            .none => {},
            .relu => {
                values.unaryMax(0);
            },
            .sigmoid => {
                values.unaryMultiply(-1);
                values.unaryExp();
                values.unaryAdd(1);
                values.unaryReciprocal();
            },
            .relu_clipped => {
                values.unaryMax(0);
                values.unaryMin(relu_clipped_factor);
            },
            .relu_leaky => {
                this.temp.binarySet(values);
                this.temp.unaryMultiply(relu_leaky_factor);
                values.binaryMax(&this.temp);
            },
            .silu => {
                this.temp.binarySet(values);
                this.temp.unaryMultiply(-1);
                this.temp.unaryExp();
                this.temp.unaryAdd(1);
                this.temp.unaryReciprocal();
                values.binaryMultiply(&this.temp);
            },
            .gelu => {
                this.temp.binarySet(values);
                this.temp.unaryMultiply(-1.702);
                this.temp.unaryExp();
                this.temp.unaryAdd(1);
                this.temp.unaryReciprocal();
                values.binaryMultiply(&this.temp);
            },
            .tanh => {
                values.unaryTanh();
            },
        }
    }
    pub fn backward(this: *@This(), values: *Tensor, values_g: *Tensor) void {
        switch (this.t) {
            .none => {},
            .relu => {
                this.temp.binarySet(values);
                this.temp.unarySign();
                values_g.binaryMultiply(&this.temp);
            },
            .sigmoid => {
                this.temp.unarySet(1);
                this.temp.binarySubtract(values);
                this.temp.binaryMultiply(values);
                values_g.binaryMultiply(&this.temp);
            },
            .relu_clipped => {
                this.temp.binarySet(values);
                this.temp.unarySubtract(relu_clipped_factor);
                this.temp.unaryAbsolute();
                this.temp.unarySign();
                values_g.binaryMultiply(&this.temp);
            },
            .relu_leaky => {
                todo(@src());
            },
            .silu => {
                todo(@src());
            },
            .gelu => {
                todo(@src());
            },
            .tanh => {
                this.temp.binarySet(values);
                this.temp.unarySquare();
                this.temp.unaryMultiply(-1);
                this.temp.unaryAdd(1);
                values_g.binaryMultiply(&this.temp);
            },
        }
    }
};
pub const Dense = struct {
    size_in: u32,
    size_out: u32,

    weights: Tensor,
    weights_g: Tensor,
    biases: Tensor,
    biases_g: Tensor,

    temp_in: Tensor,
    temp_out: Tensor,
    temp_full: Tensor,
    pub fn alloc(allocator: Allocator, size_in: u32, size_out: u32, context: ClContext) !Dense {
        assert(size_in > 0);
        assert(size_out > 0);
        return .{
            .size_in = size_in,
            .size_out = size_out,
            .weights = try Tensor.alloc(allocator, 1, 1, size_in, size_out, context, 2),
            .weights_g = try Tensor.alloc(allocator, 1, 1, size_in, size_out, context, size_in + size_out + 1),
            .biases = try Tensor.alloc(allocator, 1, 1, 1, size_out, context, 2),
            .biases_g = try Tensor.alloc(allocator, 1, 1, 1, size_out, context, 2),
            .temp_in = try Tensor.allocIntermediary(allocator, 1, 1, size_in, 1, context, 2),
            .temp_out = try Tensor.allocIntermediary(allocator, 1, 1, 1, size_out, context, 2),
            .temp_full = try Tensor.allocIntermediary(allocator, 1, 1, size_in, size_out, context, size_in + size_out),
        };
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
        this.weights.free(allocator);
        this.weights_g.free(allocator);
        this.biases.free(allocator);
        this.biases_g.free(allocator);
        this.temp_in.free(allocator);
        this.temp_out.free(allocator);
        this.temp_full.free(allocator);
    }
    pub fn forward(this: *@This(), in: *Tensor, out: *Tensor) void {
        assert(in.buffer.a_size == 1);
        assert(in.buffer.z_size == 1);
        assert(in.buffer.y_size == this.size_in);
        assert(in.buffer.x_size == 1);
        assert(out.buffer.a_size == 1);
        assert(out.buffer.z_size == 1);
        assert(out.buffer.y_size == 1);
        assert(out.buffer.x_size == this.size_out);

        this.weights.moveResize(1, 1, this.size_in, 1);
        out.moveResize(1, 1, 1, 1);

        var row_idx: u32 = 0;
        while (row_idx < this.size_out) : (row_idx += 1) {
            this.weights.moveOffset(0, 0, 0, row_idx);
            out.moveOffset(0, 0, 0, row_idx);
            this.temp_in.binarySet(&this.weights);
            this.temp_in.binaryMultiply(in);
            out.reduceSum(&this.temp_in);
        }

        this.weights.moveResize(1, 1, this.size_in, this.size_out);
        this.weights.moveOffset(0, 0, 0, 0);
        out.moveResize(1, 1, 1, this.size_out);
        out.moveOffset(0, 0, 0, 0);
        out.binaryAdd(&this.biases);
    }
    pub fn backward(this: *@This(), in: *Tensor, in_g: *Tensor, out_g: *Tensor) void {
        assert(in.buffer.a_size == 1);
        assert(in.buffer.z_size == 1);
        assert(in.buffer.y_size == this.size_in);
        assert(in.buffer.x_size == 1);
        assert(in_g.buffer.overlapsAll(in.buffer));
        assert(out_g.buffer.a_size == 1);
        assert(out_g.buffer.z_size == 1);
        assert(out_g.buffer.y_size == 1);
        assert(out_g.buffer.x_size == this.size_out);

        this.biases_g.binaryAdd(out_g);

        this.temp_full.moveResize(1, 1, 1, this.size_out);
        var column_idx: u32 = 0;
        while (column_idx < this.size_in) : (column_idx += 1) {
            this.temp_full.moveOffset(0, 0, column_idx, 0);
            this.temp_full.binarySet(out_g);
        }
        this.temp_full.moveResize(1, 1, this.size_in, 1);
        var row_idx: u32 = 0;
        while (row_idx < this.size_out) : (row_idx += 1) {
            this.temp_full.moveOffset(0, 0, 0, row_idx);
            this.temp_full.binaryMultiply(in);
        }
        this.temp_full.moveResize(1, 1, this.size_in, this.size_out);
        this.temp_full.moveOffset(0, 0, 0, 0);
        this.weights_g.binaryAdd(&this.temp_full);

        in_g.moveResize(1, 1, 1, 1);
        this.weights.moveResize(1, 1, 1, this.size_out);
        column_idx = 0;
        while (column_idx < this.size_in) : (column_idx += 1) {
            in_g.moveOffset(0, 0, column_idx, 0);
            this.weights.moveOffset(0, 0, column_idx, 0);
            this.temp_out.binarySet(&this.weights);
            this.temp_out.binaryMultiply(out_g);
            in_g.reduceSum(&this.temp_out); // Could do avg here for move numerical stability
        }
        in_g.moveResize(1, 1, this.size_in, 1);
        in_g.moveOffset(0, 0, 0, 0);
        this.weights.moveResize(1, 1, this.size_in, this.size_out);
        this.weights.moveOffset(0, 0, 0, 0);
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Dense {s}\n", .{ " " ** offset, text });
        } else {
            std.debug.print("{s}Dense\n", .{" " ** offset});
        }
        std.debug.print("{s}In {}, Out {}\n", .{ " " ** (offset + padding), this.size_in, this.size_out });
    }
    pub fn debug(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        this.print(padding, offset, name);
        this.weights.print(padding, padding + offset, "weights");
        this.weights_g.print(padding, padding + offset, "weights_g");
        this.biases.print(padding, padding + offset, "biases");
        this.biases_g.print(padding, padding + offset, "biases_g");
        this.temp_in.print(padding, padding + offset, "temp_in");
        this.temp_out.print(padding, padding + offset, "temp_out");
        this.temp_full.print(padding, padding + offset, "temp_full");
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

    weights: Tensor,
    weights_g: Tensor,
    biases: Tensor,
    biases_g: Tensor,

    temp_input_padded: Tensor,
    temp_grad_padded: Tensor,
    temp_kernel: Tensor,
    temp_single: Tensor,
    /// Size is the per dimension size **`without`** padding
    pub inline fn sizeNew(dim_size: u32, kernel_size: u32, kernel_stride: u32, kernel_padding: u32) u32 {
        assert(dim_size >= kernel_size);
        return @divFloor(dim_size + 2 * kernel_padding - kernel_size, kernel_stride) + 1;
    }
    pub fn alloc(
        allocator: Allocator,
        z_in: u32,
        y_in: u32,
        x_in: u32,
        filters: u32,
        kernel_size: u32,
        kernel_stride: u32,
        kernel_padding: u32,
        context: ClContext,
    ) !Convolution {
        assert(filters > 0);
        assert(kernel_stride > 0);
        assert(kernel_size > 0);
        assert(z_in > 0);
        assert(y_in >= kernel_size);
        assert(x_in >= kernel_size);
        const y_out: u32 = sizeNew(y_in, kernel_size, kernel_stride, kernel_padding);
        const x_out: u32 = sizeNew(x_in, kernel_size, kernel_stride, kernel_padding);
        return .{
            .z_in = z_in,
            .y_in = y_in,
            .x_in = x_in,
            .filters = filters,
            .kernel_size = kernel_size,
            .kernel_stride = kernel_stride,
            .kernel_padding = kernel_padding,
            .weights = try Tensor.alloc(allocator, filters, z_in, kernel_size, kernel_size, context, 2),
            .weights_g = try Tensor.alloc(allocator, filters, z_in, kernel_size, kernel_size, context, 3 * filters * y_out * x_out),
            .biases = try Tensor.alloc(allocator, filters, 1, 1, 1, context, 2),
            .biases_g = try Tensor.alloc(allocator, filters, 1, 1, 1, context, 2 * filters),
            .temp_input_padded = try Tensor.alloc(allocator, 1, z_in, y_in + 2 * kernel_padding, x_in + 2 * kernel_padding, context, 1),
            .temp_grad_padded = try Tensor.alloc(allocator, 1, z_in, y_in + 2 * kernel_padding, x_in + 2 * kernel_padding, context, 3 * filters * y_out * x_out),
            .temp_kernel = try Tensor.alloc(allocator, 1, z_in, kernel_size, kernel_size, context, 2),
            .temp_single = try Tensor.alloc(allocator, 1, 1, 1, 1, context, 1),
        };
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
        this.weights.free(allocator);
        this.weights_g.free(allocator);
        this.biases.free(allocator);
        this.biases_g.free(allocator);
        this.temp_input_padded.free(allocator);
        this.temp_grad_padded.free(allocator);
        this.temp_kernel.free(allocator);
        this.temp_single.free(allocator);
    }
    pub fn forward(this: *@This(), in: *Tensor, out: *Tensor) void {
        assert(1 == in.buffer.a_size);
        assert(this.z_in == in.buffer.z_size);
        assert(this.y_in == in.buffer.y_size);
        assert(this.x_in == in.buffer.x_size);
        assert(this.filters == out.buffer.z_size);
        assert(out.buffer.y_size == sizeNew(this.y_in, this.kernel_size, this.kernel_stride, this.kernel_padding));
        assert(out.buffer.x_size == sizeNew(this.x_in, this.kernel_size, this.kernel_stride, this.kernel_padding));

        const y_out: u32 = out.buffer.y_size;
        const x_out: u32 = out.buffer.x_size;

        this.biases.moveResize(1, 1, 1, 1);
        this.weights.moveResize(1, this.z_in, this.kernel_size, this.kernel_size);
        this.temp_input_padded.moveResize(1, this.z_in, this.y_in, this.x_in);
        this.temp_input_padded.moveOffset(0, 0, this.kernel_padding, this.kernel_padding);
        this.temp_input_padded.binarySet(in);
        this.temp_input_padded.moveResize(1, this.z_in, this.kernel_size, this.kernel_size);
        var filter_idx: u32 = 0;
        while (filter_idx < this.filters) : (filter_idx += 1) {
            this.biases.moveOffset(filter_idx, 0, 0, 0);
            this.weights.moveOffset(filter_idx, 0, 0, 0);
            var y_out_idx: u32 = 0;
            while (y_out_idx < y_out) : (y_out_idx += 1) {
                var x_out_idx: u32 = 0;
                while (x_out_idx < x_out) : (x_out_idx += 1) {
                    out.moveOffset(0, filter_idx, y_out_idx, x_out_idx);
                    this.temp_input_padded.moveOffset(0, 0, y_out_idx * this.kernel_stride, x_out_idx * this.kernel_stride);
                    this.temp_kernel.binarySet(&this.temp_input_padded);
                    this.temp_kernel.binaryMultiply(&this.weights);
                    out.reduceSum(&this.temp_kernel);
                    out.binaryAdd(&this.biases);
                }
            }
        }
        this.biases.moveResize(this.filters, 1, 1, 1);
        this.biases.moveOffset(0, 0, 0, 0);
        this.weights.moveResize(this.filters, this.z_in, this.y_in, this.x_in);
        this.weights.moveOffset(0, 0, 0, 0);
        out.moveResize(1, this.z_in * this.filters, y_out, x_out);
        out.moveOffset(0, 0, 0, 0);
        this.temp_input_padded.moveResize(1, this.z_in * this.filters, y_out + 2 * this.kernel_padding, x_out + 2 * this.kernel_padding);
        this.temp_input_padded.moveOffset(0, 0, 0, 0);
    }
    pub fn backward(this: *@This(), in: *Tensor, in_g: *Tensor, out: *Tensor, out_g: *Tensor) void {
        assert(1 == in.buffer.a_size);
        assert(this.z_in == in.buffer.z_size);
        assert(this.y_in == in.buffer.y_size);
        assert(this.x_in == in.buffer.x_size);
        assert(in.buffer.overlapsAll(in_g.buffer));
        assert(this.filters == out.buffer.z_size);
        assert(out.buffer.y_size == sizeNew(this.y_in, this.kernel_size, this.kernel_stride, this.kernel_padding));
        assert(out.buffer.x_size == sizeNew(this.x_in, this.kernel_size, this.kernel_stride, this.kernel_padding));
        assert(out.buffer.overlapsAll(out_g.buffer));

        const y_out: u32 = out.buffer.y_size;
        const x_out: u32 = out.buffer.x_size;

        this.biases_g.moveResize(1, 1, 1, 1);
        out_g.moveResize(1, 1, y_out, x_out);
        var filter_idx: u32 = 0;
        while (filter_idx < this.filters) : (filter_idx += 1) {
            this.biases_g.moveOffset(filter_idx, 0, 0, 0);
            out_g.moveOffset(0, filter_idx, 0, 0);
            this.temp_single.reduceSum(out_g); // Could do avg here for move numerical stability
            this.biases_g.binaryAdd(&this.temp_single);
        }
        this.biases_g.moveResize(1, this.filters, 1, 1);
        this.biases_g.moveOffset(0, 0, 0, 0);

        out_g.moveResize(1, this.filters, y_out, x_out);
        out_g.moveOffset(0, 0, 0, 0);
        this.weights_g.moveResize(1, this.z_in, this.kernel_size, this.kernel_size);
        this.temp_input_padded.moveResize(1, this.z_in, this.kernel_size, this.kernel_size);
        filter_idx = 0;
        while (filter_idx < this.filters) : (filter_idx += 1) {
            this.weights_g.moveOffset(filter_idx, 0, 0, 0);
            var y_out_idx: u32 = 0;
            while (y_out_idx < y_out) : (y_out_idx += 1) {
                var x_out_idx: u32 = 0;
                while (x_out_idx < x_out) : (x_out_idx += 1) {
                    out_g.moveOffset(0, filter_idx, y_out_idx, x_out_idx);
                    this.temp_input_padded.moveOffset(0, 0, y_out_idx * this.kernel_stride, x_out_idx * this.kernel_stride);
                    this.temp_kernel.binarySet(&this.temp_input_padded);
                    this.temp_kernel.expandMultiply(out_g);
                    this.weights_g.binaryAdd(&this.temp_kernel);
                }
            }
        }

        out_g.moveResize(1, 1, 1, 1);
        out_g.moveOffset(0, 0, 0, 0);
        this.weights.moveResize(1, this.z_in, this.kernel_size, this.kernel_size);
        this.weights.moveOffset(0, 0, 0, 0);
        this.temp_grad_padded.moveResize(1, this.z_in, this.kernel_size, this.kernel_size);
        this.temp_grad_padded.moveOffset(0, 0, 0, 0);
        filter_idx = 0;
        while (filter_idx < this.filters) : (filter_idx += 1) {
            this.weights.moveOffset(filter_idx, 0, 0, 0);
            var y_out_idx: u32 = 0;
            while (y_out_idx < y_out) : (y_out_idx += 1) {
                var x_out_idx: u32 = 0;
                while (x_out_idx < x_out) : (x_out_idx += 1) {
                    out_g.moveOffset(0, filter_idx, y_out_idx, x_out_idx);
                    this.temp_grad_padded.moveOffset(0, 0, y_out_idx * this.kernel_padding, x_out_idx * this.kernel_padding);
                    this.temp_kernel.binarySet(&this.weights);
                    this.temp_kernel.expandMultiply(out_g);
                    this.temp_grad_padded.binaryAdd(&this.temp_kernel);
                }
            }
        }
        out_g.moveResize(1, this.filters, y_out, x_out);
        out_g.moveOffset(0, 0, 0, 0);
        this.weights.moveResize(this.filters, this.z_in, this.kernel_size, this.kernel_size);
        this.weights.moveOffset(0, 0, 0, 0);
        this.temp_grad_padded.moveResize(1, this.z_in, this.y_in, this.x_in);
        this.temp_grad_padded.moveOffset(0, 0, this.kernel_padding, this.kernel_padding);
        in_g.binaryAdd(&this.temp_grad_padded);
        this.temp_grad_padded.moveResize(1, this.z_in, this.y_in + 2 * this.kernel_padding, this.x_in + 2 * this.kernel_padding);
        this.temp_grad_padded.moveOffset(0, 0, 0, 0);
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Convolution {s}\n", .{ " " ** offset, text });
        } else {
            std.debug.print("{s}Convolution\n", .{" " ** offset});
        }
        std.debug.print("{s}In (1, {}, {}, {}), Size {}, Stride {}, Padding {}\n", .{
            " " ** (offset + padding), //
            this.z_in, this.y_in, this.x_in, //
            this.kernel_size, this.kernel_stride, this.kernel_padding, //
        });
    }
    pub fn debug(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        this.print(padding, offset, name);
        this.weights.print(padding, padding + offset, "weights");
        this.weights_g.print(padding, padding + offset, "weights_g");
        this.biases.print(padding, padding + offset, "biases");
        this.biases_g.print(padding, padding + offset, "biases_g");
        this.temp_input_padded.print(padding, padding + offset, "temp_input_padded");
        this.temp_grad_padded.print(padding, padding + offset, "temp_grad_padded");
        this.temp_kernel.print(padding, padding + offset, "temp_kernel");
        this.temp_single.print(padding, padding + offset, "temp_single");
    }
};
pub const Reduce = struct {
    pub const Type = enum(u32) { sum, avg, max, min };
    z_in: u32,
    y_in: u32,
    x_in: u32,
    kernel_size: u32,
    kernel_stride: u32,
    t: Reduce.Type,
    pub inline fn sizeNew(dim_size: u32, kernel_size: u32, kernel_stride: u32) u32 {
        assert(dim_size >= kernel_size);
        return @divFloor(dim_size - kernel_size, kernel_stride) + 1;
    }
    pub fn init(z_in: u32, y_in: u32, x_in: u32, kernel_size: u32, kernel_stride: u32, t: Reduce.Type) Reduce {
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
    pub fn forward(this: *@This(), in: *Tensor, out: *Tensor) void {
        assert(in.buffer.a_size == 1);
        assert(in.buffer.z_size == this.z_in);
        assert(in.buffer.y_size == this.x_in);
        assert(in.buffer.x_size == this.y_in);
        assert(out.buffer.a_size == 1);
        assert(out.buffer.z_size == in.buffer.z_size);
        assert(out.buffer.y_size == sizeNew(in.buffer.y_size, this.kernel_size, this.kernel_stride));
        assert(out.buffer.x_size == sizeNew(in.buffer.x_size, this.kernel_size, this.kernel_stride));

        const y_out: u32 = out.buffer.y_size;
        const x_out: u32 = out.buffer.x_size;
        in.moveResize(1, 1, this.kernel_size, this.kernel_size);
        out.moveResize(1, 1, 1, 1);
        var z_idx: u32 = 0;
        while (z_idx < this.z_in) : (z_idx += 1) {
            var y_out_idx: u32 = 0;
            while (y_out_idx < y_out) : (y_out_idx += 1) {
                var x_out_idx: u32 = 0;
                while (x_out_idx < x_out) : (x_out_idx += 1) {
                    in.moveOffset(0, z_idx, y_out_idx * this.kernel_stride, x_out_idx * this.kernel_stride);
                    out.moveOffset(0, z_idx, y_out_idx, x_out_idx);
                    switch (this.t) { // The branch predictor should do this ezpz
                        .sum => out.reduceSum(in),
                        .avg => out.reduceAvg(in),
                        .max => out.reduceMax(in),
                        .min => out.reduceMin(in),
                    }
                }
            }
        }
    }
    /// $FIXME This backprop is just straight up wrong, but it at least **somewhat** approximates the correct solution.
    pub fn backward(this: *@This(), in_g: *Tensor, out_g: *Tensor) void {
        assert(in_g.buffer.a_size == 1);
        assert(in_g.buffer.z_size == this.z_in);
        assert(in_g.buffer.y_size == this.x_in);
        assert(in_g.buffer.x_size == this.y_in);
        assert(out_g.buffer.a_size == 1);
        assert(out_g.buffer.z_size == in_g.buffer.z_size);
        assert(out_g.buffer.y_size == sizeNew(in_g.buffer.y_size, this.kernel_size, this.kernel_stride));
        assert(out_g.buffer.x_size == sizeNew(in_g.buffer.x_size, this.kernel_size, this.kernel_stride));
        if (this.t != .sum) todo(@src());

        in_g.moveResize(1, 1, this.kernel_size, this.kernel_size);
        out_g.moveResize(1, 1, 1, 1);

        const y_out: u32 = out_g.buffer.y_size;
        const x_out: u32 = out_g.buffer.x_size;
        in_g.moveResize(1, 1, this.kernel_size, this.kernel_size);
        out_g.moveResize(1, 1, 1, 1);
        var z_idx: u32 = 0;
        while (z_idx < this.z_in) : (z_idx += 1) {
            var y_out_idx: u32 = 0;
            while (y_out_idx < y_out) : (y_out_idx += 1) {
                var x_out_idx: u32 = 0;
                while (x_out_idx < x_out) : (x_out_idx += 1) {
                    in_g.moveOffset(0, z_idx, y_out_idx * this.kernel_stride, x_out_idx * this.kernel_stride);
                    out_g.moveOffset(0, z_idx, y_out_idx, x_out_idx);
                    in_g.expandAdd(out_g);
                }
            }
        }
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Reduce {s}\n", .{ " " ** offset, text });
        } else {
            std.debug.print("{s}Reduce\n", .{" " ** offset});
        }
        std.debug.print("{s}In (1, {}, {}, {}), Size {}, Stride {}\n", .{
            " " ** (offset + padding), //
            this.z_in,        this.y_in,          this.x_in, //
            this.kernel_size, this.kernel_stride,
        });
    }
};
pub const Split = struct {
    filters: u32,
    z_in: u32,
    y_in: u32,
    x_in: u32,

    weights: Tensor,
    weights_g: Tensor,
    biases: Tensor,
    biases_g: Tensor,

    temp_in: Tensor,
    pub fn alloc(allocator: Allocator, filters: u32, z_in: u32, y_in: u32, x_in: u32, context: ClContext) !Split {
        assert(filters > 0);
        assert(z_in > 0);
        assert(y_in > 0);
        assert(x_in > 0);
        return .{
            .filters = filters,
            .z_in = z_in,
            .y_in = y_in,
            .x_in = x_in,
            .weights = try Tensor.alloc(allocator, filters, z_in, y_in, x_in, context, 2),
            .weights_g = try Tensor.alloc(allocator, filters, z_in, y_in, x_in, context, 3 * filters),
            .biases = try Tensor.alloc(allocator, filters, z_in, y_in, x_in, context, 2),
            .biases_g = try Tensor.alloc(allocator, filters, z_in, y_in, x_in, context, 2),
            .temp_in = try Tensor.alloc(allocator, 1, z_in, y_in, x_in, context, 2),
        };
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
        this.weights.free(allocator);
        this.weights_g.free(allocator);
        this.biases.free(allocator);
        this.biases_g.free(allocator);
        this.temp_in.free(allocator);
    }
    pub fn forward(this: *@This(), in: *Tensor, out: *Tensor) void {
        assert(in.buffer.a_size == 1);
        assert(in.buffer.z_size == this.z_in);
        assert(in.buffer.y_size == this.y_in);
        assert(in.buffer.x_size == this.x_in);
        assert(out.buffer.a_size == 1);
        assert(out.buffer.z_size == this.z_in * this.filters);
        assert(out.buffer.y_size == this.y_in);
        assert(out.buffer.x_size == this.x_in);

        in.moveResize(1, this.z_in, this.y_in, this.x_in);
        out.moveResize(1, this.z_in, this.y_in, this.x_in);
        var filter_idx: u32 = 0;
        while (filter_idx < this.filters) : (filter_idx += 1) {
            in.moveOffset(filter_idx, 0, 0, 0);
            out.moveOffset(0, filter_idx * this.filters, 0, 0);
            out.binarySet(in);
            out.binaryMultiply(&this.weights);
            out.binaryAdd(&this.biases);
        }
        in.moveResize(this.filters, this.z_in, this.y_in, this.x_in);
        in.moveOffset(0, 0, 0, 0);
        out.moveResize(1, this.z_in * this.filters, this.y_in, this.x_in);
        out.moveOffset(0, 0, 0, 0);
    }
    pub fn backward(this: *@This(), in: *Tensor, in_g: *Tensor, out_g: *Tensor) void {
        assert(in.buffer.a_size == 1);
        assert(in.buffer.z_size == this.z_in);
        assert(in.buffer.y_size == this.y_in);
        assert(in.buffer.x_size == this.x_in);
        assert(in.buffer.overlapsAll(in_g.buffer));
        assert(out_g.buffer.a_size == 1);
        assert(out_g.buffer.z_size == this.z_in * this.filters);
        assert(out_g.buffer.y_size == this.y_in);
        assert(out_g.buffer.x_size == this.x_in);

        this.biases_g.moveResize(1, this.z_in * this.filters, this.y_in, this.x_in);
        this.biases_g.binarySet(out_g);
        this.biases_g.moveResize(this.filters, this.z_in, this.y_in, this.x_in);

        this.weights_g.moveResize(1, this.z_in, this.y_in, this.x_in);
        out_g.moveResize(1, this.z_in, this.y_in, this.x_in);
        var filter_idx: u32 = 0;
        while (filter_idx < this.filters) : (filter_idx += 1) {
            this.weights_g.moveOffset(filter_idx, 0, 0, 0);
            out_g.moveOffset(0, filter_idx * this.filters, 0, 0);
            this.temp_in.binarySet(out_g);
            this.temp_in.binaryMultiply(in);
            this.weights_g.binaryAdd(&this.temp_in);
        }
        this.weights_g.moveResize(this.filters, this.z_in, this.y_in, this.x_in);
        this.weights_g.moveOffset(0, 0, 0, 0);

        this.weights.moveResize(1, this.z_in, this.y_in, this.x_in);
        filter_idx = 0;
        while (filter_idx < this.filters) : (filter_idx += 1) {
            this.weights.moveOffset(filter_idx, 0, 0, 0);
            out_g.moveOffset(0, filter_idx * this.filters, 0, 0);
            this.temp_in.binarySet(out_g);
            this.temp_in.binaryMultiply(&this.weights);
            in_g.binaryAdd(&this.temp_in);
        }
        this.weights.moveResize(this.filters, this.z_in, this.y_in, this.x_in);
        this.weights.moveOffset(0, 0, 0, 0);
        out_g.moveResize(1, this.z_in * this.filters, this.y_in, this.x_in);
        out_g.moveOffset(0, 0, 0, 0);
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Reduce {s}\n", .{ " " ** offset, text });
        } else {
            std.debug.print("{s}Reduce\n", .{" " ** offset});
        }
        std.debug.print("{s}In (1, {}, {}, {}), filters {}\n", .{
            " " ** (offset + padding), //
            this.z_in,    this.y_in, this.x_in, //
            this.filters,
        });
    }
    pub fn debug(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        this.print(padding, offset, name);
        this.weights.print(padding, offset + padding, "weights");
        this.weights_g.print(padding, offset + padding, "weights_g");
        this.biases.print(padding, offset + padding, "biases");
        this.biases_g.print(padding, offset + padding, "biases_g");
        this.temp_in.print(padding, offset + padding, "temp_in");
    }
};
pub const Residual = struct {
    pub const Type = enum(u32) {
        identity,
    };
    t: Residual.Type,
    in_layer: u32,
    pub fn forward(this: *@This(), in: *Tensor, out: *Tensor) void {
        assert(in.buffer.overlapsAll(out.buffer));
        assert(in.buffer.id != out.buffer.id);
        switch (this.t) {
            .identity => {
                out.binaryAdd(&in);
            },
        }
    }
    pub fn backward(this: *@This(), in_g: *Tensor, out_g: *Tensor) void {
        assert(in_g.buffer.overlapsAll(out_g));
        assert(in_g.buffer.id != out_g.buffer.id);
        switch (this.t) {
            .identity => {
                in_g.binaryAdd(&out_g);
            },
        }
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Reduce {s}\n", .{ " " ** offset, text });
        } else {
            std.debug.print("{s}Reduce\n", .{" " ** offset});
        }
        std.debug.print("{s}In \"{s}\", Out \"{s}\"\n", .{
            " " ** (offset + padding), //
            this.in.buffer.name(), this.out.buffer.name(), //
        });
    }
    pub fn debug(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        this.print(padding, offset, name);
        this.in.print(padding, offset + padding, "in");
        this.out.print(padding, offset + padding, "out");
    }
};

pub const Layer = @This();
// $TODO Maybe just add values, values_g, activation and norming to the sub-structs so that Layer itself can be the union
pub const Type = enum { dense, convolution, reduce, split, residual };
tag: union(Type) {
    dense: Dense,
    convolution: Convolution,
    reduce: Reduce,
    split: Split,
    residual: Residual,
},
values: Tensor,
values_g: Tensor,
activation: Activation,
pub const Config = union(Type) {
    dense: struct {
        size_out: u32,
        activation_type: Activation.Type,
    },
    convolution: struct {
        filters: u32,
        kernel_size: u32,
        kernel_stride: u32,
        kernel_padding: u32,
        activation_type: Activation.Type,
    },
    reduce: struct {
        kernel_size: u32,
        kernel_stride: u32,
        t: Reduce.Type,
    },
    split: struct {
        filters: u32,
        activation_type: Activation.Type,
    },
    residual: struct {
        in_layer: u32,
        t: Residual.Type,
    },
};
