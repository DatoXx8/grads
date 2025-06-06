const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const Optimization = @import("./compiler/optimize.zig").Optimization;
const Program = @import("./compiler/program.zig").Program;
const ClContext = @import("./runtimes/cl.zig").ClContext;
const ClDevice = @import("./runtimes/cl.zig").ClDevice;
const ClCommandQueue = @import("./runtimes/cl.zig").ClCommandQueue;
const Tensor = @import("./tensor.zig").Tensor;
const Linearized = @import("./tensor.zig").Linearized;
const Op = @import("./tensor.zig").Op;

// $TODO Make this file more compact

// $TODO Make a leaky variant that is max(tanh(x),x). It has a very interesting shape
/// Has to be `none` for reduce layers
pub const Activation = struct {
    const leaky_factor: f32 = 0.1;
    pub const Type = enum(u8) {
        none,
        relu,
        sigmoid,
        tanh,
        silu,
        gelu,
        leaky,
    };
    t: Activation.Type,
    intermediary: ?Tensor,
    pub fn alloc(allocator: Allocator, t: Activation.Type, a: u32, z: u32, y: u32, x: u32, context: ClContext) !Activation {
        return .{
            .t = t,
            .intermediary = switch (t) {
                .none => null,
                .relu => try Tensor.allocIntermediary(allocator, a, z, y, x, context),
                .sigmoid => try Tensor.allocIntermediary(allocator, a, z, y, x, context),
                .tanh => null,
                .silu => try Tensor.allocIntermediary(allocator, a, z, y, x, context),
                .gelu => try Tensor.allocIntermediary(allocator, a, z, y, x, context),
                .leaky => try Tensor.allocIntermediary(allocator, a, z, y, x, context),
            },
        };
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
        if (this.intermediary) |intermediary| {
            intermediary.free(allocator);
        }
    }
    pub fn forward(this: *@This(), input: *Tensor) void {
        switch (this.t) {
            .none => {},
            .relu => {
                input.unaryMax(0);
            },
            .sigmoid => {
                input.unaryMultiply(-1);
                input.unaryExp();
                input.unaryAdd(1);
                input.unaryReciprocal();
            },
            .tanh => {
                input.unaryTanh();
            },
            .silu => {
                this.intermediary.?.binarySet(input);
                input.unaryMultiply(-1);
                input.unaryExp();
                input.unaryAdd(1);
                input.unaryReciprocal();
                input.binaryMultiply(&this.intermediary.?);
            },
            .gelu => {
                this.intermediary.?.binarySet(input);
                input.unaryMultiply(-1.702);
                input.unaryExp();
                input.unaryAdd(1);
                input.unaryReciprocal();
                input.binaryMultiply(&this.intermediary.?);
            },
            .leaky => {
                this.intermediary.?.binarySet(input);
                this.intermediary.?.unaryMultiply(Activation.leaky_factor);
                input.binaryMax(&this.intermediary.?);
            },
        }
    }
    pub fn backward(this: *@This(), input: *Tensor, input_g: *Tensor) void {
        switch (this.t) {
            .none => {},
            .relu => {
                this.intermediary.?.binarySet(input);
                this.intermediary.?.unarySign();
                input_g.binaryMultiply(&this.intermediary.?);
            },
            .sigmoid => {
                this.intermediary.?.binarySet(input);
                this.intermediary.?.unaryMultiply(-1);
                this.intermediary.?.unaryAdd(1);
                this.intermediary.?.binaryMultiply(input);
                input_g.binaryMultiply(&this.intermediary.?);
            },
            .tanh => {
                input_g.unarySquare();
                input_g.unaryMultiply(-1);
                input_g.unaryAdd(1);
            },
            .silu => {
                unreachable;
            },
            .gelu => {
                unreachable;
            },
            .leaky => {
                this.intermediary.?.binarySet(input);
                this.intermediary.?.unarySign();
                this.intermediary.?.unaryMax(Activation.leaky_factor);
                input_g.binaryMultiply(&this.intermediary.?);
            },
        }
    }
};
pub const Norm = struct {
    pub const Type = enum(u8) {
        none,
        layer,
        batch,
        simple,
        softmax,
    };
    type: Norm.Type,
    mean: ?Tensor,
    variance: ?Tensor,
    max: ?Tensor,
    pub fn alloc(allocator: Allocator, t: Norm.Type, a: u32, z: u32, y: u32, x: u32, context: ClContext) !Norm {
        return switch (t) {
            .none => .{
                .type = t,
                .mean = null,
                .variance = null,
                .max = null,
            },
            .layer => .{
                .type = t,
                .mean = try Tensor.allocIntermediary(allocator, a, z, y, x, context),
                .variance = try Tensor.allocIntermediary(allocator, a, z, y, x, context),
                .max = null,
            },
            .batch => .{
                .type = t,
                .mean = try Tensor.allocIntermediary(allocator, a, z, y, x, context),
                .variance = try Tensor.allocIntermediary(allocator, a, z, y, x, context),
                .max = null,
            },
            .simple => .{
                .type = t,
                .mean = null,
                .variance = null,
                .max = try Tensor.allocIntermediary(allocator, 1, 1, 1, 1, context),
            },
            .softmax => .{
                .type = t,
                // Repurposed to be the place in which to exp the values
                .mean = null,
                .variance = null,
                .max = try Tensor.allocIntermediary(allocator, 1, 1, 1, 1, context),
            },
        };
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
        switch (this.type) {
            .none => {},
            .layer => {
                this.mean.?.free(allocator);
                this.variance.?.free(allocator);
            },
            .batch => {
                this.mean.?.free(allocator);
                this.variance.?.free(allocator);
            },
            .simple => {
                this.max.?.free(allocator);
            },
            .softmax => {
                this.max.?.free(allocator);
            },
        }
    }
    pub fn forward(this: *@This(), allocator: Allocator, input: Tensor) !void {
        switch (this.type) {
            .none => {},
            .layer => {
                unreachable;
            },
            .batch => {
                unreachable;
            },
            .simple => {
                this.max.?.reduceMax(allocator, input);
                input.expandDivide(allocator, this.max.?);
            },
            .softmax => {
                input.unaryExp(allocator);
                this.max.?.reduceSum(allocator, input);
                input.expandDivide(allocator, this.max.?);
            },
        }
    }
    pub fn backward(this: *@This(), allocator: Allocator, input: Tensor, input_g: Tensor) !void {
        _ = allocator;
        _ = input;
        _ = input_g;
        switch (this.type) {
            .none => {},
            .layer => {
                unreachable;
            },
            .batch => {
                unreachable;
            },
            .simple => {},
            .softmax => {
                unreachable;
            },
        }
    }
};
pub const Dense = struct {
    size_input: u32,
    size_output: u32,

    weights: Tensor,
    biases: Tensor,
    weights_g: Tensor,
    biases_g: Tensor,

    temp_input: Tensor,
    temp_output: Tensor,
    temp_full: Tensor,

    pub fn alloc(allocator: Allocator, size_input: u32, size_output: u32, context: ClContext) !Dense {
        assert(size_input > 0);
        assert(size_output > 0);

        return .{
            .size_input = size_input,
            .size_output = size_output,
            .weights = try Tensor.alloc(allocator, 1, 1, size_input, size_output, context),
            .biases = try Tensor.alloc(allocator, 1, 1, 1, size_output, context),
            .weights_g = try Tensor.alloc(allocator, 1, 1, size_input, size_output, context),
            .biases_g = try Tensor.alloc(allocator, 1, 1, 1, size_output, context),
            .temp_input = try Tensor.allocIntermediary(allocator, 1, 1, size_input, 1, context),
            .temp_output = try Tensor.allocIntermediary(allocator, 1, 1, 1, size_output, context),
            .temp_full = try Tensor.allocIntermediary(allocator, 1, 1, size_input, size_output, context),
        };
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
        this.weights.free(allocator);
        this.biases.free(allocator);
        this.weights_g.free(allocator);
        this.biases_g.free(allocator);
        this.temp_input.free(allocator);
        this.temp_output.free(allocator);
        this.temp_full.free(allocator);
    }
    pub fn forward(this: *@This(), input: *Tensor, output: *Tensor) void {
        assert(input.buffer.a_size == 1);

        const z_in: u32 = input.buffer.z_size;
        const y_in: u32 = input.buffer.y_size;
        const x_in: u32 = input.buffer.x_size;
        const z_out: u32 = output.buffer.z_size;
        const y_out: u32 = output.buffer.y_size;
        const x_out: u32 = output.buffer.x_size;

        input.moveReshape(1, 1, this.size_input, 1);
        this.weights.moveResize(1, 1, this.size_input, 1);
        output.moveResize(1, 1, 1, 1);

        var row_idx: u32 = 0;
        while (row_idx < this.size_output) : (row_idx += 1) {
            this.weights.moveOffset(0, 0, 0, row_idx);
            output.moveOffset(0, 0, 0, row_idx);

            this.temp_input.binarySet(&this.weights);
            this.temp_input.binaryMultiply(input);
            output.reduceSum(&this.temp_input);
        }

        input.moveReshape(1, z_in, y_in, x_in);
        input.moveOffset(0, 0, 0, 0);
        output.moveResize(1, z_out, y_out, x_out);
        output.moveOffset(0, 0, 0, 0);
        this.weights.moveResize(1, 1, this.size_input, this.size_output);
        this.weights.moveOffset(0, 0, 0, 0);

        output.binaryAdd(&this.biases);
    }
    pub fn backward(this: *@This(), input: *Tensor, input_g: *Tensor, output_g: *Tensor) void {
        const z_in: u32 = input.buffer.z_size;
        const y_in: u32 = input.buffer.y_size;
        const x_in: u32 = input.buffer.x_size;
        // Biases
        this.biases_g.binaryAdd(output_g);
        // Weights
        this.temp_full.moveResize(1, 1, 1, this.size_output);
        var column_idx: u32 = 0;
        while (column_idx < this.size_input) : (column_idx += 1) {
            this.temp_full.moveOffset(0, 0, column_idx, 0);
            this.temp_full.binarySet(output_g);
        }
        this.temp_full.moveResize(1, 1, this.size_input, 1);
        input.moveReshape(1, 1, this.size_input, 1);
        var row_idx: u32 = 0;
        while (row_idx < this.size_output) : (row_idx += 1) {
            this.temp_full.moveOffset(0, 0, 0, row_idx);
            this.temp_full.binaryMultiply(input);
        }
        this.temp_full.moveResize(1, 1, this.size_input, this.size_output);
        this.temp_full.moveOffset(0, 0, 0, 0);
        input.moveReshape(1, z_in, y_in, x_in);
        this.weights_g.binaryAdd(&this.temp_full);
        // Previous activation
        input_g.moveReshape(1, 1, this.size_input, 1);
        input_g.moveResize(1, 1, 1, 1);
        this.weights.moveReshape(1, 1, 1, this.size_output);
        column_idx = 0;
        while (column_idx < this.size_input) : (column_idx += 1) {
            input_g.moveOffset(0, 0, column_idx, 0);
            this.weights.moveOffset(0, 0, column_idx, 0);
            this.temp_output.binarySet(&this.weights);
            this.temp_output.binaryMultiply(output_g);
            // Could use sum or avg here, sum it technically more accurate but avg is more stable
            input_g.reduceSum(&this.temp_output);
            // input_g.reduceAvg(this.temp_output);
        }
        input_g.moveReshape(1, z_in, y_in, x_in);
        input_g.moveOffset(0, 0, 0, 0);
        this.weights.moveResize(1, 1, this.size_input, this.size_output);
        this.weights.moveOffset(0, 0, 0, 0);
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Dense {s}\n", .{ [1]u8{' '} ** offset, text });
        } else {
            std.debug.print("{s}Dense\n", .{[1]u8{' '} ** offset});
        }
        std.debug.print("{s}In {} Out {}\n", .{ [1]u8{' '} ** (offset + padding), this.size_input, this.size_output });
        std.debug.print("{s}Biases ({}, {}, {}, {})\n", .{
            [1]u8{' '} ** (offset + padding), //
            this.biases.buffer.a_inherent, this.biases.buffer.z_inherent, //
            this.biases.buffer.y_inherent, this.biases.buffer.x_inherent,
        });
        std.debug.print("{s}Weights ({}, {}, {}, {})\n", .{
            [1]u8{' '} ** (offset + padding), //
            this.weights.buffer.a_inherent, this.weights.buffer.z_inherent, //
            this.weights.buffer.y_inherent, this.weights.buffer.x_inherent,
        });
    }
    pub fn debug(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Dense {s}\n", .{ [1]u8{' '} ** offset, text });
        } else {
            std.debug.print("{s}Dense\n", .{[1]u8{' '} ** offset});
        }
        std.debug.print("{s}In {} Out {}\n", .{ [1]u8{' '} ** (offset + padding), this.size_input, this.size_output });
        this.biases.print(padding, offset + padding, "biases");
        this.biases_g.print(padding, offset + padding, "biases_g");
        this.weights.print(padding, offset + padding, "weights");
        this.weights_g.print(padding, offset + padding, "weights_g");
        this.temp_input.print(padding, offset + padding, "temp_input");
        this.temp_output.print(padding, offset + padding, "temp_output");
        this.temp_full.print(padding, offset + padding, "temp_full");
    }
};
pub const Convolution = struct {
    z: u32,
    y: u32,
    x: u32,

    filters: u32,
    kernel_size: u32,
    kernel_stride: u32,
    kernel_padding: u32,

    weights: Tensor,
    biases: Tensor,
    weights_g: Tensor,
    biases_g: Tensor,

    temp_input_padded: Tensor,
    temp_grad_padded: Tensor,
    temp_kernel: Tensor,
    temp_single: Tensor,

    pub fn alloc(
        allocator: Allocator,
        z: u32,
        y: u32,
        x: u32,
        filters: u32,
        kernel_size: u32,
        kernel_stride: u32,
        kernel_padding: u32,
        context: ClContext,
    ) !Convolution {
        assert(filters > 0);
        assert(kernel_size > 0);
        assert(kernel_stride > 0);
        assert(kernel_padding >= 0);
        assert(z > 0);
        assert(y == x);
        assert(y >= kernel_size);
        assert(x >= kernel_size);
        return .{
            .z = z,
            .y = y,
            .x = x,
            .filters = filters,
            .kernel_size = kernel_size,
            .kernel_padding = kernel_padding,
            .kernel_stride = kernel_stride,
            .biases = try Tensor.alloc(allocator, filters, 1, 1, 1, context),
            .biases_g = try Tensor.alloc(allocator, filters, 1, 1, 1, context),
            .weights = try Tensor.alloc(allocator, filters, z, kernel_size, kernel_size, context),
            .weights_g = try Tensor.alloc(allocator, filters, z, kernel_size, kernel_size, context),
            .temp_input_padded = try Tensor.allocIntermediary(allocator, 1, z, y + 2 * kernel_padding, //
                x + 2 * kernel_padding, context),
            .temp_grad_padded = try Tensor.allocIntermediary(allocator, 1, z, y + 2 * kernel_padding, //
                x + 2 * kernel_padding, context),
            .temp_kernel = try Tensor.allocIntermediary(allocator, 1, z, kernel_size, kernel_size, context),
            .temp_single = try Tensor.allocIntermediary(allocator, 1, 1, 1, 1, context),
        };
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
        this.biases.free(allocator);
        this.biases_g.free(allocator);
        this.weights.free(allocator);
        this.weights_g.free(allocator);
        this.temp_input_padded.free(allocator);
        this.temp_grad_padded.free(allocator);
        this.temp_kernel.free(allocator);
        this.temp_single.free(allocator);
    }
    pub fn forward(this: *@This(), input: *Tensor, output: *Tensor) void {
        assert(input.buffer.a_size == 1);
        assert(input.buffer.a_size == output.buffer.a_size);

        const z_in: u32 = input.buffer.y_size;
        const y_in: u32 = input.buffer.y_size;
        const x_in: u32 = input.buffer.x_size;
        const z_out: u32 = output.buffer.y_size;
        const y_out: u32 = output.buffer.y_size;
        const x_out: u32 = output.buffer.x_size;

        const x_in_max: u32 = input.buffer.x_size + this.kernel_padding - 1;
        const y_in_max: u32 = input.buffer.y_size + this.kernel_padding - 1;

        // $TODO Figure out a way to remove these padded temporary buffers. Could make the normal buffers padded and just downsize, but that makes things complicated.

        input.moveOffset(0, 0, 0, 0);
        this.biases.moveResize(1, 1, 1, 1);
        this.weights.moveResize(1, z_in, this.kernel_size, this.kernel_size);
        output.moveResize(1, 1, 1, 1);
        this.temp_input_padded.moveResize(1, z_in, y_in, x_in);
        this.temp_input_padded.moveOffset(0, 0, this.kernel_padding, this.kernel_padding);
        this.temp_input_padded.binarySet(input);
        this.temp_input_padded.moveResize(1, z_in, this.kernel_size, this.kernel_size);

        var filter_idx: u32 = 0;
        while (filter_idx < this.filters) : (filter_idx += 1) {
            this.biases.moveOffset(filter_idx, 0, 0, 0);
            this.weights.moveOffset(filter_idx, 0, 0, 0);
            var y_out_idx: u32 = 0;
            var y_in_idx: u32 = 0;
            while (y_in_idx < @divFloor(y_in_max, this.kernel_stride)) : (y_in_idx += this.kernel_stride) {
                var x_out_idx: u32 = 0;
                var x_in_idx: u32 = 0;
                while (x_in_idx < @divFloor(x_in_max, this.kernel_stride)) : (x_in_idx += this.kernel_stride) {
                    output.moveOffset(0, filter_idx, y_out_idx, x_out_idx);
                    this.temp_input_padded.moveOffset(0, 0, y_in_idx, x_in_idx);
                    this.temp_kernel.binarySet(&this.temp_input_padded);
                    this.temp_kernel.binaryMultiply(&this.weights);
                    output.reduceSum(&this.temp_kernel);
                    output.binaryAdd(&this.biases);
                    x_out_idx += 1;
                }
                y_out_idx += 1;
            }
        }
        this.biases.moveResize(this.filters, 1, 1, 1);
        this.biases.moveOffset(0, 0, 0, 0);
        this.weights.moveResize(this.filters, this.z, this.kernel_size, this.kernel_size);
        this.weights.moveOffset(0, 0, 0, 0);
        output.moveResize(1, z_out, y_out, x_out);
        output.moveOffset(0, 0, 0, 0);
        this.temp_input_padded.moveResize(1, z_out, y_out + 2 * this.kernel_padding, x_out + 2 * this.kernel_padding);
        this.temp_input_padded.moveOffset(0, 0, 0, 0);
    }
    // This backprop is per sample, but I guess if i iterate over the a dimension in the training output then I can do all of this with a singular function call.
    // That would remove the need for temp_full
    pub fn backward(this: *@This(), input: *Tensor, input_g: *Tensor, output: *Tensor, output_g: *Tensor) void {
        const z_out: u32 = output.buffer.z_size;
        const y_out: u32 = output.buffer.y_size;
        const x_out: u32 = output.buffer.x_size;
        const z_in: u32 = input.buffer.z_size;
        const y_in: u32 = input.buffer.y_size;
        const x_in: u32 = input.buffer.x_size;
        // Biases
        this.biases_g.moveResize(1, 1, 1, 1);
        output_g.moveResize(1, 1, y_out, x_out);
        var filter_idx: u32 = 0;
        while (filter_idx < this.filters) : (filter_idx += 1) {
            this.biases_g.moveOffset(filter_idx, 0, 0, 0);
            output_g.moveOffset(0, filter_idx, 0, 0);
            // Could do avg here for better numerical stability
            this.temp_single.reduceSum(output_g);
            this.biases_g.binaryAdd(&this.temp_single);
        }
        this.biases_g.moveResize(this.filters, 1, 1, 1);
        this.biases_g.moveOffset(0, 0, 0, 0);
        output_g.moveResize(1, z_out, y_out, x_out);
        output_g.moveOffset(0, 0, 0, 0);
        // Weights
        output_g.moveResize(1, 1, 1, 1);
        output_g.moveOffset(0, 0, 0, 0);
        this.weights_g.moveResize(1, z_in, this.kernel_size, this.kernel_size);
        this.weights_g.moveOffset(0, 0, 0, 0);
        this.temp_input_padded.moveResize(1, z_in, this.kernel_size, this.kernel_size);
        this.temp_input_padded.moveOffset(0, 0, 0, 0);
        filter_idx = 0;
        while (filter_idx < this.filters) : (filter_idx += 1) {
            this.weights_g.moveOffset(filter_idx, 0, 0, 0);
            var y_out_idx: u32 = 0;
            var y_in_idx: u32 = 0;
            while (y_out_idx < y_out) : (y_out_idx += 1) {
                var x_out_idx: u32 = 0;
                var x_in_idx: u32 = 0;
                while (x_out_idx < x_out) : (x_out_idx += 1) {
                    output_g.moveOffset(0, filter_idx, y_out_idx, x_out_idx);
                    this.temp_input_padded.moveOffset(0, 0, y_in_idx, x_in_idx);
                    this.temp_kernel.binarySet(&this.temp_input_padded);
                    this.temp_kernel.expandMultiply(output_g);
                    this.weights_g.binaryAdd(&this.temp_kernel);

                    x_in_idx += this.kernel_stride;
                }
                y_in_idx += this.kernel_stride;
            }
        }
        output.moveResize(1, z_out, y_out, x_out);
        output.moveOffset(0, 0, 0, 0);
        this.weights.moveResize(this.filters, z_in, this.kernel_size, this.kernel_size);
        this.weights.moveOffset(0, 0, 0, 0);
        this.temp_grad_padded.moveResize(1, z_in, y_in, x_in);
        this.temp_grad_padded.moveOffset(0, 0, this.kernel_padding, this.kernel_padding);

        input_g.binarySet(&this.temp_grad_padded);

        this.temp_grad_padded.moveResize(1, z_in, //
            y_in + 2 * this.kernel_padding, x_in + 2 * this.kernel_padding);
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Convolution {s}\n", .{ [1]u8{' '} ** offset, text });
        } else {
            std.debug.print("{s}Convolution\n", .{[1]u8{' '} ** offset});
        }
        const out_z: u32 = @divFloor(this.z + 2 * this.kernel_padding, this.kernel_stride);
        const out_y: u32 = @divFloor(this.y + 2 * this.kernel_padding, this.kernel_stride);
        const out_x: u32 = @divFloor(this.x + 2 * this.kernel_padding, this.kernel_stride);
        std.debug.print("{s}In (1, {}, {}, {}) Out (1, {}, {}, {})\n", .{
            [1]u8{' '} ** (offset + padding), //
            this.z, this.y, this.x, //
            out_z,  out_y,  out_x,
        });
        std.debug.print("{s}Filters {} Size {} Stride {} Padding {}\n", .{
            [1]u8{' '} ** (offset + padding), //
            this.filters,       this.kernel_size, //
            this.kernel_stride, this.kernel_padding,
        });
        std.debug.print("{s}Biases ({}, {}, {}, {})\n", .{
            [1]u8{' '} ** (offset + padding), //
            this.biases.buffer.a_inherent, this.biases.buffer.z_inherent, //
            this.biases.buffer.y_inherent, this.biases.buffer.x_inherent,
        });
        std.debug.print("{s}Weights ({}, {}, {}, {})\n", .{
            [1]u8{' '} ** (offset + padding), //
            this.weights.buffer.a_inherent, this.weights.buffer.z_inherent, //
            this.weights.buffer.y_inherent, this.weights.buffer.x_inherent,
        });
    }
    pub fn debug(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Convolution {s}\n", .{ [1]u8{' '} ** offset, text });
        } else {
            std.debug.print("{s}Convolution\n", .{[1]u8{' '} ** offset});
        }
        const out_z: u32 = @divFloor(this.z + 2 * this.kernel_padding - 2 * this.kernel_size, this.kernel_stride);
        const out_y: u32 = @divFloor(this.y + 2 * this.kernel_padding - 2 * this.kernel_size, this.kernel_stride);
        const out_x: u32 = @divFloor(this.x + 2 * this.kernel_padding - 2 * this.kernel_size, this.kernel_stride);

        std.debug.print("{s}In (1, {}, {}, {}) Out (1, {}, {}, {})\n", .{
            [1]u8{' '} ** (offset + padding), //
            this.z, this.y, this.x, //
            out_z,  out_y,  out_x,
        });
        std.debug.print("{s}Filters {} Size {} Stride {} Padding {}\n", .{
            [1]u8{' '} ** (offset + padding), //
            this.filters,       this.kernel_size, //
            this.kernel_stride, this.kernel_padding,
        });
        this.biases.print(padding + offset, offset, "biases");
        this.biases_g.print(padding + offset, offset, "biases_g");
        this.weights.print(padding + offset, offset, "weights");
        this.weights_g.print(padding + offset, offset, "weights_g");
        this.temp_input_padded.print(padding + offset, offset, "temp_input_padded");
        this.temp_grad_padded.print(padding + offset, offset, "temp_grad_padded");
        this.temp_kernel.print(padding + offset, offset, "temp_kernel");
        this.temp_single.print(padding + offset, offset, "temp_single");
    }
};
pub const Reduce = struct {
    pub const Type = enum(u8) {
        sum,
        avg,
        max,
        min,
    };
    t: Reduce.Type,
    z: u32,
    y: u32,
    x: u32,
    kernel_size: u32,
    kernel_stride: u32,
    // The point of the init is only to check that the initialisation uses valid values
    pub fn init(t: Reduce.Type, z: u32, y: u32, x: u32, kernel_size: u32, kernel_stride: u32) Reduce {
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);
        assert(kernel_size > 0);
        assert(kernel_stride > 0);
        assert(kernel_size <= y);
        assert(kernel_size <= x);
        return .{
            .t = t,
            .z = z,
            .y = y,
            .x = x,
            .kernel_size = kernel_size,
            .kernel_stride = kernel_stride,
        };
    }
    pub fn forward(this: @This(), input: *Tensor, output: *Tensor) void {
        assert(input.buffer.a_size == 1);
        assert(input.buffer.a_size == 1);

        const z_out: u32 = output.buffer.z_size;
        const y_out: u32 = output.buffer.y_size;
        const x_out: u32 = output.buffer.x_size;

        input.moveOffset(0, 0, 0, 0);
        output.moveResize(1, 1, 1, 1);
        output.moveOffset(0, 0, 0, 0);

        var channel_idx: u32 = 0;
        while (channel_idx < this.z) : (channel_idx += 1) {
            var y_out_idx: u32 = 0;
            var y_in_idx: u32 = 0;
            while (y_in_idx < @divFloor(this.y - this.kernel_size + 1, this.kernel_stride)) : (y_in_idx += this.kernel_stride) {
                var x_out_idx: u32 = 0;
                var x_in_idx: u32 = 0;
                while (x_in_idx < @divFloor(this.x - this.kernel_size + 1, this.kernel_stride)) : (x_in_idx += this.kernel_stride) {
                    input.moveOffset(0, channel_idx, y_in_idx, x_in_idx);
                    output.moveOffset(0, channel_idx, y_out_idx, x_out_idx);
                    switch (this.t) {
                        .sum => output.reduceSum(input),
                        .avg => output.reduceAvg(input),
                        .max => output.reduceMax(input),
                        .min => output.reduceMin(input),
                    }
                    x_out_idx += 1;
                }
                y_out_idx += 1;
            }
        }

        input.moveOffset(0, 0, 0, 0);
        output.moveResize(1, z_out, y_out, x_out);
        output.moveOffset(0, 0, 0, 0);
    }
    /// $TODO This is a mega hack that just is just a loose approximation of the real backprop
    pub fn backward(this: @This(), input_g: *Tensor, output_g: *Tensor) void {
        const z_in: u32 = input_g.buffer.z_size;
        const y_in: u32 = input_g.buffer.y_size;
        const x_in: u32 = input_g.buffer.x_size;
        const z_out: u32 = output_g.buffer.z_size;
        const y_out: u32 = output_g.buffer.y_size;
        const x_out: u32 = output_g.buffer.x_size;
        input_g.moveResize(1, 1, this.kernel_size, this.kernel_size);
        output_g.moveResize(1, 1, 1, 1);

        var channel_idx: u32 = 0;
        while (channel_idx < this.z) : (channel_idx += 1) {
            var y_out_idx: u32 = 0;
            var y_in_idx: u32 = 0;
            while (y_out_idx < this.y) : (y_out_idx += 1) {
                var x_out_idx: u32 = 0;
                var x_in_idx: u32 = 0;
                while (x_out_idx < this.x) : (x_out_idx += 1) {
                    input_g.moveOffset(0, channel_idx, y_in_idx, x_in_idx);
                    output_g.moveOffset(0, channel_idx, y_out_idx, x_out_idx);
                    input_g.expandAdd(output_g);
                    x_in_idx += this.kernel_stride;
                }
                y_in_idx += this.kernel_stride;
            }
        }

        input_g.moveResize(1, z_in, y_in, x_in);
        input_g.moveOffset(0, 0, 0, 0);
        output_g.moveResize(1, z_out, y_out, x_out);
        output_g.moveOffset(0, 0, 0, 0);
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Reduce {s}\n", .{ [1]u8{' '} ** offset, text });
        } else {
            std.debug.print("{s}Reduce\n", .{[1]u8{' '} ** offset});
        }
        const z_out: u32 = @divFloor(this.z - 2 * this.kernel_size, this.kernel_stride);
        const y_out: u32 = @divFloor(this.y - 2 * this.kernel_size, this.kernel_stride);
        const x_out: u32 = @divFloor(this.x - 2 * this.kernel_size, this.kernel_stride);
        std.debug.print("{s}Size {} Stride {} In (1, {}, {}, {}) Out (1, {}, {}, {})\n", .{
            [1]u8{' '} ** (offset + padding), //
            this.kernel_size, this.kernel_stride, //
            this.z,           this.y,
            this.x,           z_out,
            y_out,            x_out,
        });
    }
};
pub const Split = struct {
    filters: u32,
    z: u32,
    y: u32,
    x: u32,

    weights: Tensor,
    biases: Tensor,
    weights_g: Tensor,
    biases_g: Tensor,

    temp_input: Tensor,

    pub fn alloc(allocator: Allocator, filters: u32, z: u32, y: u32, x: u32, context: ClContext) !Split {
        assert(filters > 0);
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);
        return .{
            .filters = filters,
            .z = z,
            .y = y,
            .x = x,
            .weights = try Tensor.alloc(allocator, filters, z, y, x, context),
            .weights_g = try Tensor.alloc(allocator, filters, z, y, x, context),
            .biases = try Tensor.alloc(allocator, filters, z, y, x, context),
            .biases_g = try Tensor.alloc(allocator, filters, z, y, x, context),
            .temp_input = try Tensor.allocIntermediary(allocator, 1, z, y, x, context),
        };
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
        this.weights.free(allocator);
        this.weights_g.free(allocator);
        this.biases.free(allocator);
        this.biases_g.free(allocator);
        this.temp_input.free(allocator);
    }
    pub fn forward(this: *@This(), input: *Tensor, output: *Tensor) void {
        assert(input.buffer.a_size == 1);
        assert(input.buffer.a_size == output.buffer.a_size);
        assert(input.buffer.z_size * this.filters == output.buffer.z_size);
        assert(input.buffer.y_size == output.buffer.y_size);
        assert(input.buffer.x_size == output.buffer.x_size);

        const z_out: u32 = output.buffer.z_size;
        const y_out: u32 = output.buffer.y_size;
        const x_out: u32 = output.buffer.x_size;
        const z_in: u32 = input.buffer.z_size;
        const y_in: u32 = input.buffer.y_size;
        const x_in: u32 = input.buffer.x_size;

        input.moveOffset(0, 0, 0, 0);
        output.moveResize(1, z_in, y_out, x_out);
        output.moveOffset(0, 0, 0, 0);
        this.weights.moveResize(1, z_in, y_in, x_in);
        this.weights.moveOffset(0, 0, 0, 0);
        this.biases.moveResize(1, z_in, y_in, x_in);
        this.biases.moveOffset(0, 0, 0, 0);

        var filter_idx: u32 = 0;
        while (filter_idx < this.filters) : (filter_idx += 1) {
            output.moveOffset(0, filter_idx * z_in, 0, 0);
            this.weights.moveOffset(filter_idx, 0, 0, 0);
            this.biases.moveOffset(filter_idx, 0, 0, 0);
            output.binarySet(input);
            output.binaryMultiply(&this.weights);
            output.binaryAdd(&this.biases);
        }

        input.moveResize(1, z_in, y_in, x_in);
        input.moveOffset(0, 0, 0, 0);
        output.moveResize(1, z_out, y_out, x_out);
        output.moveOffset(0, 0, 0, 0);
        this.weights.moveResize(this.filters, z_in, y_in, x_in);
        this.weights.moveOffset(0, 0, 0, 0);
        this.biases.moveResize(this.filters, z_in, y_in, x_in);
        this.biases.moveOffset(0, 0, 0, 0);
    }
    pub fn backward(this: *@This(), input: *Tensor, input_g: *Tensor, output_g: *Tensor) void {
        this.biases_g.moveResize(1, this.z * this.filters, this.y, this.x);
        this.biases_g.moveOffset(0, 0, 0, 0);
        this.biases_g.binarySet(output_g);
        this.biases_g.moveResize(this.filters, this.z, this.y, this.x);
        this.biases_g.moveOffset(0, 0, 0, 0);

        this.weights_g.moveResize(1, this.z, this.y, this.x);
        output_g.moveResize(1, this.z, this.y, this.x);
        var filter_idx: u32 = 0;
        while (filter_idx < this.filters) : (filter_idx += 1) {
            this.weights_g.moveOffset(filter_idx, 0, 0, 0);
            output_g.moveOffset(0, filter_idx * this.z, 0, 0);
            this.temp_input.binarySet(output_g);
            this.temp_input.binaryMultiply(input);
            this.weights_g.binaryAdd(&this.temp_input);
        }
        this.weights_g.moveResize(this.filters, this.z, this.y, this.x);
        this.weights_g.moveOffset(0, 0, 0, 0);
        output_g.moveResize(1, this.filters * this.z, this.y, this.x);
        output_g.moveOffset(0, 0, 0, 0);

        output_g.moveResize(1, this.z, this.y, this.x);
        this.weights.moveResize(1, this.z, this.y, this.x);
        filter_idx = 0;
        while (filter_idx < this.filters) : (filter_idx += 1) {
            this.weights.moveOffset(filter_idx, 0, 0, 0);
            output_g.moveOffset(0, filter_idx * this.z, 0, 0);
            this.temp_input.binarySet(output_g);
            this.temp_input.binaryMultiply(&this.weights);
            input_g.binaryAdd(&this.temp_input);
        }
        this.weights.moveResize(this.filters, this.z, this.y, this.x);
        this.weights.moveOffset(0, 0, 0, 0);
        output_g.moveResize(1, this.filters * this.z, this.y, this.x);
        output_g.moveOffset(0, 0, 0, 0);
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Split {s}\n", .{ [1]u8{' '} ** offset, text });
        } else {
            std.debug.print("{s}Split\n", .{[1]u8{' '} ** offset});
        }

        std.debug.print("{s}Z {} Y {} X {} Filters {}\n", .{
            [1]u8{' '} ** (offset + padding),
            this.z,
            this.y,
            this.x,
            this.filters,
        });
        this.biases.print(padding, offset + padding, "biases");
        this.biases_g.print(padding, offset + padding, "biases_g");
        this.weights.print(padding, offset + padding, "weights");
        this.weights_g.print(padding, offset + padding, "weights_g");
        this.temp_input.print(padding, offset + padding, "temp_input");
    }
    pub fn debug(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Split {s}\n", .{ [1]u8{' '} ** offset, text });
        } else {
            std.debug.print("{s}Split\n", .{[1]u8{' '} ** offset});
        }

        std.debug.print("{s}Z {} Y {} X {} Filters {}\n", .{
            [1]u8{' '} ** (offset + padding), //
            this.z,
            this.y,
            this.x,
            this.filters,
        });
        this.biases.print(padding, offset + padding, "biases");
        this.biases_g.print(padding, offset + padding, "biases_g");
        this.weights.print(padding, offset + padding, "weights");
        this.weights_g.print(padding, offset + padding, "weights_g");
        this.temp_input.print(padding, offset + padding, "temp_input");
    }
};
pub const Residual = struct {
    pub const Type = enum(u8) {
        identity,
        convolution,
        dense,
        reduce,
        split,
    };
    const Connection = union(Residual.Type) {
        identity: void,
        convolution: Convolution,
        dense: Dense,
        reduce: Reduce,
        split: Split,
    };
    t: Residual.Type,
    connection: Residual.Connection,
    layer: u32,
    pub fn allocIdentity(layer: u32) Residual {
        return .{
            .layer = layer,
            .t = .identity,
            .connection = .{
                .identity = @as(void, {}),
            },
        };
    }
    pub fn allocDense(layer: u32, allocator: Allocator, size_in: u32, size_out: u32, context: ClContext) !Residual {
        return .{
            .layer = layer,
            .t = .dense,
            .connection = .{
                .dense = try Dense.alloc(allocator, size_in, size_out, context),
            },
        };
    }
    pub fn allocConvolution(
        layer: u32,
        allocator: Allocator,
        z: u32,
        y: u32,
        x: u32,
        filters: u32,
        kernel_size: u32,
        kernel_stride: u32,
        kernel_padding: u32,
        context: ClContext,
    ) !Residual {
        return .{
            .layer = layer,
            .t = .convolution,
            .connection = .{
                .convolution = try Convolution.alloc(allocator, z, y, x, filters, kernel_size, kernel_stride, kernel_padding, context),
            },
        };
    }
    pub fn allocReduce(layer: u32, t: Reduce.Type, z: u32, y: u32, x: u32, kernel_size: u32, kernel_stride: u32) !Residual {
        return .{
            .layer = layer,
            .t = .reduce,
            .connection = .{
                .reduce = Reduce.init(t, z, y, x, kernel_size, kernel_stride),
            },
        };
    }
    pub fn allocSplit(layer: u32, allocator: Allocator, filters: u32, z: u32, y: u32, x: u32, context: ClContext) !Residual {
        return .{
            .layer = layer,
            .t = .split,
            .connection = .{
                .split = try Split.alloc(allocator, filters, z, y, x, context),
            },
        };
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
        switch (this.connection) {
            .identity => {},
            .convolution => this.connection.convolution.free(allocator),
            .dense => this.connection.dense.free(allocator),
            .reduce => {},
            .split => {},
        }
    }
    pub fn forward(this: *@This(), input: *Tensor, output: *Tensor) void {
        assert(input.buffer.a_size == 1);
        assert(input.buffer.a_size == output.buffer.a_size);
        assert(input.buffer.z_size == output.buffer.z_size);
        assert(input.buffer.y_size == output.buffer.y_size);
        assert(input.buffer.x_size == output.buffer.x_size);

        input.moveOffset(0, 0, 0, 0);
        output.moveOffset(0, 0, 0, 0);

        switch (this.t) {
            .identity => output.binaryAdd(input),
            .convolution => unreachable,
            .dense => unreachable,
            .reduce => unreachable,
            .split => unreachable,
        }
    }
    pub fn backward(this: *@This(), input_g: *Tensor, output_g: *Tensor) void {
        assert(input_g.buffer.a_size == 1);
        assert(input_g.buffer.a_size == output_g.buffer.a_size);
        assert(input_g.buffer.z_size == output_g.buffer.z_size);
        assert(input_g.buffer.y_size == output_g.buffer.y_size);
        assert(input_g.buffer.x_size == output_g.buffer.x_size);

        input_g.moveOffset(0, 0, 0, 0);
        output_g.moveOffset(0, 0, 0, 0);

        switch (this.t) {
            .identity => input_g.binaryAdd(output_g),
            .convolution => unreachable,
            .dense => unreachable,
            .reduce => unreachable,
            .split => unreachable,
        }
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Residual {s}\n", .{ [1]u8{' '} ** offset, text });
        } else {
            std.debug.print("{s}Residual\n", .{[1]u8{' '} ** offset});
        }

        std.debug.print("{s}Type {}\n", .{ [1]u8{' '} ** (offset + padding), this.t });
        switch (this.connection) {
            .identity => {},
            .convolution => this.connection.convolution.print(padding, offset + padding, "convolution"),
            .dense => this.connection.dense.print(padding, offset + padding, "dense"),
            .reduce => this.connection.reduce.print(padding, offset + padding, "reduce"),
            .split => this.connection.split.print(padding, offset + padding, "split"),
        }
    }
    pub fn debug(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]u8) void {
        if (name) |text| {
            std.debug.print("{s}Residual {s}\n", .{ [1]u8{' '} ** offset, text });
        } else {
            std.debug.print("{s}Residual\n", .{[1]u8{' '} ** offset});
        }

        std.debug.print("{s}Type {}\n", .{ [1]u8{' '} ** (offset + padding), this.t });
        switch (this.connection) {
            .identity => {},
            .convolution => this.connection.convolution.debug(padding, offset + padding, "convolution"),
            .dense => this.connection.dense.debug(padding, offset + padding, "dense"),
            .reduce => this.connection.reduce.print(padding, offset + padding, "reduce"),
            .split => this.connection.split.debug(padding, offset + padding, "split"),
        }
    }
};
pub const Neuralnet = struct {
    pub const Type = enum(u8) {
        dense,
        convolution,
        reduce,
        split,
        residual,
    };
    pub const Layer = struct {
        /// The config is to only have the info needed when provided with the previous layer
        pub const Config = union(Neuralnet.Type) {
            dense: struct {
                size_out: u32,
                activation: Activation.Type,
            },
            convolution: struct {
                filters: u32,
                kernel_size: u32,
                kernel_stride: u32,
                kernel_padding: u32,
                activation: Activation.Type,
            },
            reduce: struct {
                kernel_size: u32,
                kernel_stride: u32,
                t: Reduce.Type,
                // activation Activation.Type,
            },
            split: struct {
                filters: u32,
                activation: Activation.Type,
            },
            residual: struct {
                t: Residual.Type,
                layer: u32,
                filters: u32,
                kernel_padding: u32,
                kernel_stride: u32,
                kernel_size: u32,
                size_out: u32,
                reduce_t: Reduce.Type,
                // activation Activation.Type,
            },
        };
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
        pub fn alloc(allocator: Allocator, z: u32, y: u32, x: u32, config: Config, context: ClContext) !Layer {
            switch (config) {
                .dense => {
                    return .{
                        .activation = try Activation.alloc(allocator, config.dense.activation, 1, 1, 1, config.dense.size_out, context),
                        .tag = .{ .dense = try Dense.alloc(allocator, z * y * x, config.dense.size_out, context) },
                        .values = try Tensor.alloc(allocator, 1, 1, 1, config.dense.size_out, context),
                        .values_g = try Tensor.alloc(allocator, 1, 1, 1, config.dense.size_out, context),
                    };
                },
                .convolution => {
                    const z_new: u32 = config.convolution.filters;
                    const y_new: u32 = @divFloor(y + 2 * config.convolution.kernel_padding - config.convolution.kernel_size, //
                        config.convolution.kernel_stride) + 1;
                    const x_new: u32 = @divFloor(x + 2 * config.convolution.kernel_padding - config.convolution.kernel_size, //
                        config.convolution.kernel_stride) + 1;
                    return .{
                        .activation = try Activation.alloc(allocator, config.convolution.activation, 1, z_new, y_new, x_new, context),
                        .tag = .{
                            .convolution = try Convolution.alloc(allocator, z, y, x, config.convolution.filters, //
                                config.convolution.kernel_size, config.convolution.kernel_stride, //
                                config.convolution.kernel_padding, context),
                        },
                        .values = try Tensor.alloc(allocator, 1, z_new, y_new, x_new, context),
                        .values_g = try Tensor.alloc(allocator, 1, z_new, y_new, x_new, context),
                    };
                },
                .reduce => {
                    const z_new: u32 = z;
                    const y_new: u32 = @divFloor(y - config.reduce.kernel_size, config.reduce.kernel_stride) + 1;
                    const x_new: u32 = @divFloor(x - config.reduce.kernel_size, config.reduce.kernel_stride) + 1;
                    return .{
                        .activation = try Activation.alloc(allocator, .none, 1, z_new, y_new, x_new, context),
                        .tag = .{
                            .reduce = Reduce.init(config.reduce.t, z, y, x, config.reduce.kernel_size, //
                                config.reduce.kernel_stride),
                        },
                        .values = try Tensor.alloc(allocator, 1, z_new, y_new, x_new, context),
                        .values_g = try Tensor.alloc(allocator, 1, z_new, y_new, x_new, context),
                    };
                },
                .split => {
                    return .{
                        .activation = try Activation.alloc(allocator, config.split.activation, 1, z * config.split.filters, y, x, context),
                        .tag = .{ .split = try Split.alloc(allocator, config.split.filters, z, y, x, context) },
                        .values = try Tensor.alloc(allocator, 1, z * config.split.filters, y, x, context),
                        .values_g = try Tensor.alloc(allocator, 1, z * config.split.filters, y, x, context),
                    };
                },
                .residual => {
                    return .{
                        .activation = try Activation.alloc(allocator, .none, 1, z, y, x, context),
                        .tag = .{
                            .residual = switch (config.residual.t) {
                                .identity => Residual.allocIdentity(config.residual.layer),
                                .dense => try Residual.allocConvolution(config.residual.layer, allocator, z, y, x, //
                                    config.residual.filters, config.residual.kernel_size, config.residual.kernel_stride, //
                                    config.residual.kernel_padding, context),
                                .convolution => try Residual.allocDense(config.residual.layer, allocator, z * y * x, //
                                    config.residual.size_out, context),
                                .reduce => try Residual.allocReduce(config.residual.layer, config.residual.reduce_t, //
                                    z, y, x, config.residual.kernel_size, config.residual.kernel_stride),
                                .split => try Residual.allocSplit(config.residual.layer, allocator, config.residual.filters, //
                                    z, y, x, context),
                            },
                        },
                        .values = try Tensor.alloc(allocator, 1, z, y, x, context),
                        .values_g = try Tensor.alloc(allocator, 1, z, y, x, context),
                    };
                },
            }
        }
        pub fn free(this: *@This(), allocator: Allocator) void {
            switch (this.tag) {
                .dense => this.tag.dense.free(allocator),
                .convolution => this.tag.convolution.free(allocator),
                .reduce => {},
                .split => this.tag.split.free(allocator),
                .residual => this.tag.residual.free(allocator),
            }
            if (this.activation.intermediary) |*intermediary| {
                intermediary.free(allocator);
            }
            this.values.free(allocator);
            this.values_g.free(allocator);
        }
        pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]u8) void {
            if (name) |text| {
                std.debug.print("{s}Layer {s}\n", .{ [1]u8{' '} ** offset, text });
            } else {
                std.debug.print("{s}Layer\n", .{[1]u8{' '} ** offset});
            }

            std.debug.print("{s}Type {}\n", .{ [1]u8{' '} ** (padding + offset), this.activation.t });
            switch (this.tag) {
                .dense => this.tag.dense.print(padding, offset + padding, null),
                .convolution => this.tag.convolution.print(padding, offset + padding, null),
                .reduce => this.tag.reduce.print(padding, offset + padding, null),
                .split => this.tag.split.print(padding, offset + padding, null),
                .residual => this.tag.residual.print(padding, offset + padding, null),
            }
        }
        pub fn debug(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]u8) void {
            if (name) |text| {
                std.debug.print("{s}Layer {s}\n", .{ [1]u8{' '} ** offset, text });
            } else {
                std.debug.print("{s}Layer\n", .{[1]u8{' '} ** offset});
            }

            std.debug.print("{s}Type {}\n", .{ [1]u8{' '} ** (padding + offset), this.activation.t });
            this.values.print(padding, padding + offset, "values");
            this.values_g.print(padding, padding + offset, "values_g");
            switch (this.tag) {
                .dense => this.tag.dense.debug(padding, offset + padding, null),
                .convolution => this.tag.convolution.debug(padding, offset + padding, null),
                .reduce => this.tag.reduce.print(padding, offset + padding, null),
                .split => this.tag.split.debug(padding, offset + padding, null),
                .residual => this.tag.residual.debug(padding, offset + padding, null),
            }
        }
    };
    input: Tensor,
    layers: []Layer,
    forward_cpu: Linearized,
    backward_cpu: Linearized,
    learn_cpu: Linearized,
    forward_cl: Program,
    backward_cl: Program,
    learn_cl: Program,
    // $TODO These suckers need to be refactored badly
    // $TODO When allocating from here it should really always just use O3, or at least not allow O0 since that actually *so* slow
    pub fn alloc(
        allocator: Allocator,
        input: Tensor,
        config: []const Layer.Config,
        size_global: u32,
        size_local: u32,
        context: ClContext,
        device: ClDevice,
        queue: ClCommandQueue,
    ) !Neuralnet {
        assert(config.len > 0);
        assert(size_global > 0);
        assert(size_local > 0);
        assert(size_global % size_local == 0);

        var layers: []Layer = try allocator.alloc(Layer, config.len);
        var z_previous: u32 = input.buffer.z_size;
        var y_previous: u32 = input.buffer.y_size;
        var x_previous: u32 = input.buffer.x_size;
        for (0..layers.len) |layer_idx| {
            layers[layer_idx] = try Layer.alloc(allocator, z_previous, y_previous, x_previous, config[layer_idx], context);
            z_previous = layers[layer_idx].values.buffer.z_size;
            y_previous = layers[layer_idx].values.buffer.y_size;
            x_previous = layers[layer_idx].values.buffer.x_size;
        }
        var previous_values: Tensor = input;
        for (0..layers.len) |layer_idx| {
            // $TODO Just in case I search for TODO more than FIXME!!!!! FIXME FIXME FIXME
            // $FIXME Need to add the capacity for the activation function
            try layers[layer_idx].values.linearized.capacityEnsure(
                allocator,
                previous_values.linearized.op_num + switch (layers[layer_idx].tag) {
                    .dense => 3 * layers[layer_idx].tag.dense.size_output + 1,
                    .convolution => 4 * layers[layer_idx].values.buffer.z_size * //
                        layers[layer_idx].values.buffer.y_size * layers[layer_idx].values.buffer.x_size + 1,
                    .reduce => layers[layer_idx].values.buffer.z_size * //
                        layers[layer_idx].values.buffer.y_size * layers[layer_idx].values.buffer.x_size,
                    .split => 3 * layers[layer_idx].tag.split.filters,
                    .residual => 1,
                } + switch (layers[layer_idx].activation.t) {
                    .none => @as(u32, 0),
                    .relu => @as(u32, 1),
                    .sigmoid => @as(u32, 4),
                    .tanh => @as(u32, 1),
                    .silu => @as(u32, 6),
                    .gelu => @as(u32, 6),
                    .leaky => @as(u32, 3),
                },
            );

            // Just to force the correct order of operations
            layers[layer_idx].values.dependOn(&previous_values);

            switch (layers[layer_idx].tag) {
                .dense => {
                    layers[layer_idx].tag.dense.forward(&previous_values, &layers[layer_idx].values);
                },
                .convolution => {
                    layers[layer_idx].tag.convolution.forward(&previous_values, &layers[layer_idx].values);
                },
                .reduce => {
                    layers[layer_idx].tag.reduce.forward(&previous_values, &layers[layer_idx].values);
                },
                .split => {
                    layers[layer_idx].tag.split.forward(&previous_values, &layers[layer_idx].values);
                },
                .residual => {
                    layers[layer_idx].tag.residual.forward(&layers[layers[layer_idx].tag.residual.layer].values, //
                        &layers[layer_idx].values);
                },
            }

            layers[layer_idx].activation.forward(&layers[layer_idx].values);
            // $TODO Norming

            previous_values = layers[layer_idx].values;
        }

        var forward_cpu: Linearized = try Linearized.alloc(allocator);
        try forward_cpu.capacityEnsure(allocator, layers[layers.len - 1].values.linearized.op_num);
        forward_cpu.concat(&layers[layers.len - 1].values.linearized);

        // $TODO Is it needed to clear the gradients?
        // $TODO Refactor this reverse stuff
        var layer_idx_reverse: u32 = 0;
        while (layer_idx_reverse < layers.len - 1) : (layer_idx_reverse += 1) {
            const layer_idx: u32 = @as(u32, @truncate(layers.len)) - (layer_idx_reverse + 1);

            // $TODO capacityEnsure needs to happen here

            // $NOTE Just to force the correct order of operations
            layers[layer_idx - 1].values_g.dependOn(&layers[layer_idx].values_g);

            // $TODO Norming
            layers[layer_idx].activation.backward(&layers[layer_idx].values, &layers[layer_idx].values_g);

            switch (layers[layer_idx].tag) {
                .dense => {
                    layers[layer_idx].tag.dense.backward(&layers[layer_idx - 1].values, &layers[layer_idx - 1].values_g, //
                        &layers[layer_idx].values_g);
                },
                .convolution => {
                    layers[layer_idx].tag.convolution.backward(&layers[layer_idx - 1].values, &layers[layer_idx - 1].values_g, //
                        &layers[layer_idx].values, &layers[layer_idx].values_g);
                },
                .reduce => {
                    layers[layer_idx].tag.reduce.backward(&layers[layer_idx - 1].values_g, //
                        &layers[layer_idx].values_g);
                },
                .split => {
                    layers[layer_idx].tag.split.backward(&layers[layer_idx - 1].values, &layers[layer_idx - 1].values_g, //
                        &layers[layer_idx].values_g);
                },
                .residual => {
                    layers[layer_idx].tag.residual.backward(&layers[layers[layer_idx].tag.residual.layer].values_g, //
                        &layers[layer_idx].values_g);
                },
            }
        }

        var backward_cpu: Linearized = try Linearized.alloc(allocator);
        try backward_cpu.capacityEnsure(allocator, layers[0].values.linearized.op_num);
        backward_cpu.concat(&layers[0].values.linearized);

        var learn_cpu: Linearized = try Linearized.alloc(allocator);
        for (0..layers.len) |layer_idx| {
            switch (layers[layer_idx].tag) {
                .dense => {
                    layers[layer_idx].tag.dense.weights.binarySubtract(&layers[layer_idx].tag.dense.weights_g);
                    learn_cpu.concat(&layers[layer_idx].tag.dense.weights.linearized);
                    layers[layer_idx].tag.dense.biases.binarySubtract(&layers[layer_idx].tag.dense.biases_g);
                    learn_cpu.concat(&layers[layer_idx].tag.dense.biases.linearized);
                },
                .convolution => {
                    layers[layer_idx].tag.convolution.weights.binarySubtract(&layers[layer_idx].tag.convolution.weights_g);
                    learn_cpu.concat(&layers[layer_idx].tag.convolution.weights.linearized);
                    layers[layer_idx].tag.convolution.biases.binarySubtract(&layers[layer_idx].tag.convolution.biases_g);
                    learn_cpu.concat(&layers[layer_idx].tag.convolution.biases.linearized);
                },
                .reduce => {},
                .split => {
                    layers[layer_idx].tag.split.weights.binarySubtract(&layers[layer_idx].tag.split.weights_g);
                    learn_cpu.concat(&layers[layer_idx].tag.split.weights.linearized);
                    layers[layer_idx].tag.split.biases.binarySubtract(&layers[layer_idx].tag.split.biases_g);
                    learn_cpu.concat(&layers[layer_idx].tag.split.biases.linearized);
                },
                .residual => {},
            }
        }

        const forward_cl: Program = try Program.alloc(allocator, forward_cpu, size_global, //
            size_local, .O3, device, context, queue);
        const backward_cl: Program = try Program.alloc(allocator, backward_cpu, size_global, //
            size_local, .O3, device, context, queue);
        const learn_cl: Program = try Program.alloc(allocator, learn_cpu, size_global, //
            size_local, .O3, device, context, queue);

        return .{
            .input = input,
            .layers = layers,
            .forward_cpu = forward_cpu,
            .backward_cpu = backward_cpu,
            .learn_cpu = learn_cpu,
            .forward_cl = forward_cl,
            .backward_cl = backward_cl,
            .learn_cl = learn_cl,
        };
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
        for (0..this.layers.len) |layer_idx| {
            this.layers[layer_idx].free(allocator);
        }
        allocator.free(this.layers);
        this.forward_cpu.free(allocator);
        this.backward_cpu.free(allocator);
        this.learn_cpu.free(allocator);
        this.forward_cl.free(allocator);
        this.backward_cl.free(allocator);
        this.learn_cl.free(allocator);
    }
    /// This is a linear congruential generator, which is a PRNG itself, although it is sub par statistically speaking.
    /// But since this only transforms the seed it shouldn't matter at all.
    inline fn seedTransform(num: u64) u64 {
        return num *% 2862933555777941757 +% 1;
    }
    pub fn init(this: *@This(), seed: u32) void {
        var seed_next: u64 = seed;
        for (0..this.layers.len) |layer_idx| {
            switch (this.layers[layer_idx].tag) {
                .dense => {
                    this.layers[layer_idx].tag.dense.weights.unaryRandom(@truncate(seed_next));
                    seed_next = seedTransform(seed_next);
                    this.layers[layer_idx].tag.dense.biases.unaryRandom(@truncate(seed_next));
                    this.layers[layer_idx].tag.dense.weights.realize();
                    this.layers[layer_idx].tag.dense.biases.realize();
                },
                .convolution => {
                    this.layers[layer_idx].tag.convolution.weights.unaryRandom(@truncate(seed_next));
                    seed_next = seedTransform(seed_next);
                    this.layers[layer_idx].tag.convolution.biases.unaryRandom(@truncate(seed_next));
                    this.layers[layer_idx].tag.convolution.weights.realize();
                    this.layers[layer_idx].tag.convolution.biases.realize();
                },
                .reduce => {},
                .split => {
                    this.layers[layer_idx].tag.split.weights.unaryRandom(@truncate(seed_next));
                    seed_next = seedTransform(seed_next);
                    this.layers[layer_idx].tag.split.biases.unaryRandom(@truncate(seed_next));
                    this.layers[layer_idx].tag.split.weights.realize();
                    this.layers[layer_idx].tag.split.biases.realize();
                },
                .residual => {},
            }
            seed_next = seedTransform(seed_next);
        }
    }
    /// Have to put the input values in the dedicated struct field
    pub fn forward(this: *@This(), t: ClDevice.ClDeviceType) !void {
        switch (t) {
            .cpu => {
                // This only copies the data to the cpu if it changed
                for (0..this.layers.len) |layer_idx| {
                    switch (this.layers[layer_idx].tag) {
                        .dense => {
                            try this.layers[layer_idx].tag.dense.weights.buffer.syncToHost(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.dense.biases.buffer.syncToHost(this.forward_cl.queue);
                        },
                        .convolution => {
                            try this.layers[layer_idx].tag.convolution.weights.buffer.syncToHost(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.convolution.biases.buffer.syncToHost(this.forward_cl.queue);
                        },
                        .reduce => {},
                        .split => {
                            try this.layers[layer_idx].tag.split.weights.buffer.syncToHost(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.split.biases.buffer.syncToHost(this.forward_cl.queue);
                        },
                        .residual => {},
                    }
                }
                this.forward_cpu.run();
            },
            .gpu => {
                // This only copies the data to the gpu if it changed
                for (0..this.layers.len) |layer_idx| {
                    switch (this.layers[layer_idx].tag) {
                        .dense => {
                            try this.layers[layer_idx].tag.dense.weights.buffer.syncToDevice(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.dense.biases.buffer.syncToDevice(this.forward_cl.queue);
                        },
                        .convolution => {
                            try this.layers[layer_idx].tag.convolution.weights.buffer.syncToDevice(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.convolution.biases.buffer.syncToDevice(this.forward_cl.queue);
                        },
                        .reduce => {},
                        .split => {
                            try this.layers[layer_idx].tag.split.weights.buffer.syncToDevice(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.split.biases.buffer.syncToDevice(this.forward_cl.queue);
                        },
                        .residual => {},
                    }
                }

                // Maybe have the user do this manually?
                this.input.buffer.syncUpdate(.sync_to_device);
                try this.input.buffer.syncToDevice(this.forward_cl.queue);
                try this.input.buffer.syncWait(this.forward_cl.queue);

                try this.forward_cl.run();

                this.layers[this.layers.len - 1].values.buffer.syncUpdate(.sync_to_host);
                try this.layers[this.layers.len - 1].values.buffer.syncToHost(this.forward_cl.queue);
                try this.layers[this.layers.len - 1].values.buffer.syncWait(this.forward_cl.queue);
            },
        }
    }
    pub fn backward(this: *@This(), t: ClDevice.ClDeviceType) !void {
        switch (t) {
            .cpu => {
                // This only copies the data to the cpu if it changed
                for (0..this.layers.len) |layer_idx| {
                    switch (this.layers[layer_idx].tag) {
                        .dense => {
                            try this.layers[layer_idx].tag.dense.weights.buffer.syncToHost(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.dense.biases.buffer.syncToHost(this.forward_cl.queue);
                        },
                        .convolution => {
                            try this.layers[layer_idx].tag.convolution.weights.buffer.syncToHost(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.convolution.biases.buffer.syncToHost(this.forward_cl.queue);
                        },
                        .reduce => {},
                        .split => {
                            try this.layers[layer_idx].tag.split.weights.buffer.syncToHost(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.split.biases.buffer.syncToHost(this.forward_cl.queue);
                        },
                        .residual => {},
                    }
                }
                this.backward_cpu.run();
            },
            .gpu => {
                // This only copies the data to the gpu if it changed
                for (0..this.layers.len) |layer_idx| {
                    switch (this.layers[layer_idx].tag) {
                        .dense => {
                            try this.layers[layer_idx].tag.dense.weights.buffer.syncToDevice(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.dense.biases.buffer.syncToDevice(this.forward_cl.queue);
                        },
                        .convolution => {
                            try this.layers[layer_idx].tag.convolution.weights.buffer.syncToDevice(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.convolution.biases.buffer.syncToDevice(this.forward_cl.queue);
                        },
                        .reduce => {},
                        .split => {
                            try this.layers[layer_idx].tag.split.weights.buffer.syncToDevice(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.split.biases.buffer.syncToDevice(this.forward_cl.queue);
                        },
                        .residual => {},
                    }
                }

                // Maybe have the user do this manually?
                this.input.buffer.syncUpdate(.sync_to_device);
                try this.input.buffer.syncToDevice(this.forward_cl.queue);
                try this.input.buffer.syncWait(this.forward_cl.queue);

                try this.backward_cl.run();

                this.layers[this.layers.len - 1].values.buffer.syncUpdate(.sync_to_host);
                try this.layers[this.layers.len - 1].values.buffer.syncToHost(this.forward_cl.queue);
                try this.layers[this.layers.len - 1].values.buffer.syncWait(this.forward_cl.queue);
            },
        }
    }
    pub fn learn(this: *@This(), t: ClDevice.ClDeviceType) !void {
        switch (t) {
            .cpu => {
                // This only copies the data to the cpu if it changed
                for (0..this.layers.len) |layer_idx| {
                    switch (this.layers[layer_idx].tag) {
                        .dense => {
                            try this.layers[layer_idx].tag.dense.weights.buffer.syncToHost(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.dense.biases.buffer.syncToHost(this.forward_cl.queue);
                        },
                        .convolution => {
                            try this.layers[layer_idx].tag.convolution.weights.buffer.syncToHost(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.convolution.biases.buffer.syncToHost(this.forward_cl.queue);
                        },
                        .reduce => {},
                        .split => {
                            try this.layers[layer_idx].tag.split.weights.buffer.syncToHost(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.split.biases.buffer.syncToHost(this.forward_cl.queue);
                        },
                        .residual => {},
                    }
                }
                this.learn_cpu.run();
            },
            .gpu => {
                // This only copies the data to the gpu if it changed
                for (0..this.layers.len) |layer_idx| {
                    switch (this.layers[layer_idx].tag) {
                        .dense => {
                            try this.layers[layer_idx].tag.dense.weights.buffer.syncToDevice(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.dense.biases.buffer.syncToDevice(this.forward_cl.queue);
                        },
                        .convolution => {
                            try this.layers[layer_idx].tag.convolution.weights.buffer.syncToDevice(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.convolution.biases.buffer.syncToDevice(this.forward_cl.queue);
                        },
                        .reduce => {},
                        .split => {
                            try this.layers[layer_idx].tag.split.weights.buffer.syncToDevice(this.forward_cl.queue);
                            try this.layers[layer_idx].tag.split.biases.buffer.syncToDevice(this.forward_cl.queue);
                        },
                        .residual => {},
                    }
                }

                // Maybe have the user do this manually?
                this.input.buffer.syncUpdate(.sync_to_device);
                try this.input.buffer.syncToDevice(this.forward_cl.queue);
                try this.input.buffer.syncWait(this.forward_cl.queue);

                try this.learn_cl.run();
                // $TODO Decide if and how to send the data back to the cpu
            },
        }
    }
    /// **__WARN__** Overwrite the files if they already exists without asking for permission
    /// Write the architecture of the net to `filename_arch`, the params to `filename_bin` and a hash of the params in `filename_hash`
    pub fn saveToFiles(this: @This(), filename_arch: []const u8, filename_bin: []const u8, filename_hash: []const u8) !void {
        const file_arch = try std.fs.cwd().createFile(filename_arch, .{ .truncate = true });
        defer file_arch.close();

        try file_arch.seekTo(0);

        const buffer: [4096]u8 = [1]u8{0} ** 4096;
        var buffer_len: usize = 0;

        // $TODO Get rid of these allocations and just have a big buffer
        buffer_len = try std.fmt.bufPrint(buffer[0..], "i {} {} {} {}\n", .{
            this.input.buffer.a_size,
            this.input.buffer.z_size,
            this.input.buffer.y_size,
            this.input.buffer.x_size,
        });
        try file_arch.writeAll(buffer[0..buffer_len]);
        @memset(buffer[0..buffer_len], 0);

        for (0..this.layers.len) |layer_idx| {
            if (this.layers[layer_idx].tag == .residual) {
                assert(this.layers[layer_idx].tag.residual.t == .identity);
            }
            buffer_len = switch (this.layer[layer_idx].tag) {
                .dense => try std.fmt.bufPrint("d {} {}\n", .{
                    this.layers[layer_idx].tag.dense.size_output,
                    this.layers[layer_idx].activation.t,
                }),
                .convolution => buffer_len = try std.fmt.bufPrint("c {} {} {} {} {}\n", .{
                    this.layers[layer_idx].tag.convolution.filters,
                    this.layers[layer_idx].tag.convolution.kernel_size,
                    this.layers[layer_idx].tag.convolution.kernel_stride,
                    this.layers[layer_idx].tag.convolution.kernel_padding,
                    this.layers[layer_idx].activation.t,
                }),
                .reduce => buffer_len = try std.fmt.bufPrint("r {} {}\n", .{
                    this.layers[layer_idx].tag.reduce.kernel_size,
                    this.layers[layer_idx].tag.reduce.kernel_stride,
                }),
                .split => buffer_len = try std.fmt.bufPrint("s {} {}\n", .{
                    this.layers[layer_idx].tag.split.filters,
                    this.layers[layer_idx].activation.t,
                }),
                .residual => buffer_len = try std.fmt.bufPrint("R {} {}\n", .{
                    this.layers[layer_idx].tag.residual.t,
                    this.layers[layer_idx].tag.residual.layer,
                }),
            };
            try file_arch.writeAll(buffer[0..buffer_len]);
            @memset(buffer[0..buffer_len], 0);
        }

        const file_bin = try std.fs.cwd().createFile(filename_bin, .{ .truncate = true });
        defer file_bin.close();

        try file_bin.seekTo(0);
        for (0..this.layers.len) |layer_idx| {
            if (this.layers[layer_idx].tag == .reduce or this.layers[layer_idx].tag == .residual) continue;
            const weights: []f32 = switch (this.layers[layer_idx].tag) {
                .dense => this.layers[layer_idx].tag.dense.weights.buffer.values,
                .convolution => this.layers[layer_idx].tag.convolution.weights.buffer.values,
                .reduce => unreachable,
                .split => this.layers[layer_idx].tag.split.weights.buffer.values,
                .residual => unreachable,
            };
            const biases: []f32 = switch (this.layers[layer_idx].tag) {
                .dense => this.layers[layer_idx].tag.dense.biases.buffer.values,
                .convolution => this.layers[layer_idx].tag.convolution.biases.buffer.values,
                .reduce => unreachable,
                .split => this.layers[layer_idx].tag.split.biases.buffer.values,
                .residual => unreachable,
            };

            const weights_byte: []u8 = @as([*]u8, @ptrCast(weights.ptr))[0..(weights.len * @sizeOf(@TypeOf(weights[0])))];
            const biases_byte: []u8 = @as([*]u8, @ptrCast(biases.ptr))[0..(biases.len * @sizeOf(@TypeOf(biases[0])))];
            try file_bin.writeAll(weights_byte);
            try file_bin.writeAll(biases_byte);
        }

        const file_hash = try std.fs.cwd().createFile(filename_hash, .{ .truncate = true });
        defer file_hash.close();

        var hash = std.hash.Fnv1a_128.init();

        try file_hash.seekTo(0);
        for (0..this.layers.len) |layer_idx| {
            if (this.layers[layer_idx].tag == .reduce or this.layers[layer_idx].tag == .residual) continue;
            const weights: []f32 = switch (this.layers[layer_idx].tag) {
                .dense => this.layers[layer_idx].tag.dense.weights.buffer.values,
                .convolution => this.layers[layer_idx].tag.convolution.weights.buffer.values,
                .reduce => unreachable,
                .split => this.layers[layer_idx].tag.split.weights.buffer.values,
                .residual => unreachable,
            };
            const biases: []f32 = switch (this.layers[layer_idx].tag) {
                .dense => this.layers[layer_idx].tag.dense.biases.buffer.values,
                .convolution => this.layers[layer_idx].tag.convolution.biases.buffer.values,
                .reduce => unreachable,
                .split => this.layers[layer_idx].tag.split.biases.buffer.values,
                .residual => unreachable,
            };

            const weights_byte: []u8 = @as([*]u8, @ptrCast(weights.ptr))[0..(weights.len * @sizeOf(@TypeOf(weights[0])))];
            const biases_byte: []u8 = @as([*]u8, @ptrCast(biases.ptr))[0..(biases.len * @sizeOf(@TypeOf(biases[0])))];
            hash.update(weights_byte);
            hash.update(biases_byte);
        }
        var bytes: [16]u8 = [1]u8{0} ** 16;
        const bytes_ptr: *u128 = @alignCast(@ptrCast(&bytes));
        bytes_ptr.* = hash.final();
        std.debug.print("Wrote: {x:16}\n", .{hash.final()});
        try file_hash.writeAll(&bytes);
    }
    /// Load weights and biases from file to already existing net.
    /// Does not check that the architecture of the values of the file and the provided net match.
    pub fn readFromFile(this: @This(), filename_bin: []const u8, filename_hash: []const u8) !void {
        const file_bin = try std.fs.cwd().openFile(filename_bin, .{});
        defer file_bin.close();

        const file_hash = try std.fs.cwd().openFile(filename_hash, .{});
        defer file_hash.close();

        var hash = std.hash.Fnv1a_128.init();

        try file_bin.seekTo(0);
        for (0..this.layers.len) |layer_idx| {
            if (this.layers[layer_idx].tag == .reduce or this.layers[layer_idx].tag == .residual) continue;
            const weights: []f32 = switch (this.layers[layer_idx].tag) {
                .dense => this.layers[layer_idx].tag.dense.weights.buffer.values,
                .convolution => this.layers[layer_idx].tag.convolution.weights.buffer.values,
                .reduce => unreachable,
                .split => this.layers[layer_idx].tag.split.weights.buffer.values,
                .residual => unreachable,
            };
            const biases: []f32 = switch (this.layers[layer_idx].tag) {
                .dense => this.layers[layer_idx].tag.dense.biases.buffer.values,
                .convolution => this.layers[layer_idx].tag.convolution.biases.buffer.values,
                .reduce => unreachable,
                .split => this.layers[layer_idx].tag.split.biases.buffer.values,
                .residual => unreachable,
            };

            const weights_byte: []u8 = @as([*]u8, @ptrCast(weights.ptr))[0..(weights.len * @sizeOf(@TypeOf(weights[0])))];
            const biases_byte: []u8 = @as([*]u8, @ptrCast(biases.ptr))[0..(biases.len * @sizeOf(@TypeOf(biases[0])))];

            const weights_read = try file_bin.readAll(weights_byte);
            const biases_read = try file_bin.readAll(biases_byte);
            assert(biases_read == biases_byte.len);
            assert(weights_read == weights_byte.len);

            hash.update(weights_byte);
            hash.update(biases_byte);

            switch (this.layers[layer_idx].tag) {
                .dense => {
                    this.layers[layer_idx].tag.dense.biases.buffer.syncUpdate(.sync_to_device);
                    this.layers[layer_idx].tag.dense.weights.buffer.syncUpdate(.sync_to_device);
                },
                .convolution => {
                    this.layers[layer_idx].tag.convolution.biases.buffer.syncUpdate(.sync_to_device);
                    this.layers[layer_idx].tag.convolution.weights.buffer.syncUpdate(.sync_to_device);
                },
                .reduce => unreachable,
                .split => {
                    this.layers[layer_idx].tag.split.biases.buffer.syncUpdate(.sync_to_device);
                    this.layers[layer_idx].tag.split.weights.buffer.syncUpdate(.sync_to_device);
                },
                .residual => unreachable,
            }
        }

        var bytes: [16]u8 align(16) = [1]u8{0} ** 16;
        try file_hash.seekTo(0);
        const bytes_read = try file_hash.readAll(&bytes);
        assert(bytes_read == 16);

        const bytes_ptr: *u128 = @alignCast(@ptrCast(&bytes));
        const hash_read: u128 = bytes_ptr.*;
        const hash_final: u128 = hash.final();
        if (hash_read != hash_final) {
            std.log.warn("Mismatch between hash calculated from {s} = {x:16} and the one found in {s} = {x:16}\n", .{ filename_bin, hash_final, filename_hash, hash_read });
        }
    }
    // $TODO This one
    // pub fn createFromFile(this: @This(), allocator: Allocator, filename_arch: []const u8, filename_bin: []const u8, filename_hash: []const u8) !void {
    // }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Neuralnet {s}\n", .{ [1]u8{' '} ** offset, text });
        } else {
            std.debug.print("{s}Neuralnet\n", .{[1]u8{' '} ** offset});
        }
        this.input.print(padding, offset + padding, "input");
        for (0..this.layers.len) |layer_idx| {
            this.layers[layer_idx].print(padding, padding + offset, null);
        }
    }
    pub fn debug(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Neuralnet {s}\n", .{ [1]u8{' '} ** offset, text });
        } else {
            std.debug.print("{s}Neuralnet\n", .{[1]u8{' '} ** offset});
        }
        this.input.print(padding, offset + padding, "input");
        for (0..this.layers.len) |layer_idx| {
            this.layers[layer_idx].debug(padding, padding + offset, null);
        }
    }
};
