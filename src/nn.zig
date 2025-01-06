const std = @import("std");
const Tensor = @import("./tensor.zig").Tensor;
const Linearized = @import("./tensor.zig").Linearized;
const Op = @import("./tensor.zig").Op;

const ClContext = @import("./runtimes/cl.zig").ClContext;
const ClDevice = @import("./runtimes/cl.zig").ClDevice;
const ClCommandQueue = @import("./runtimes/cl.zig").ClCommandQueue;

const Program = @import("./compiler/program.zig").Program;

const assert = std.debug.assert;

// This backprop is per sample, but I guess if i iterate over the a dimension in the training output then I can do all of this with a singular function call.
// That would remove the need for temp_full

pub const Neuralnet = struct {
    pub const Type = enum(u8) {
        dense,
        convolution,
        reduce,
        split,
        residual,
    };

    // TODO: Make a leaky variant that is max(tanh(x),x). It has a very interesting shape
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
        pub fn alloc(allocator: anytype, t: Activation.Type, a: u32, z: u32, y: u32, x: u32, context: ClContext) !Activation {
            return .{
                .t = t,
                .intermediary = switch (t) {
                    .none => null,
                    .relu => try Tensor.alloc(allocator, a, z, y, x, context),
                    .sigmoid => null,
                    .tanh => null,
                    .silu => try Tensor.alloc(allocator, a, z, y, x, context),
                    .gelu => try Tensor.alloc(allocator, a, z, y, x, context),
                    .leaky => try Tensor.alloc(allocator, a, z, y, x, context),
                },
            };
        }
        pub fn free(this: *@This(), allocator: anytype) void {
            if (this.intermediary) |intermediary| {
                intermediary.free(allocator);
            }
        }
        pub fn forward(this: *@This(), allocator: anytype, input: *Tensor) !void {
            switch (this.t) {
                .none => {},
                .relu => {
                    try input.unaryMax(allocator, 0);
                },
                .sigmoid => {
                    try input.unaryMultiply(allocator, -1);
                    try input.unaryExp(allocator);
                    try input.unaryAdd(allocator, 1);
                    try input.unaryReciprocal(allocator);
                },
                .tanh => {
                    try input.unaryTanh(allocator);
                },
                .silu => {
                    try this.intermediary.?.binarySet(allocator, input);
                    try input.unaryMultiply(allocator, -1);
                    try input.unaryExp(allocator);
                    try input.unaryAdd(allocator, 1);
                    try input.unaryReciprocal(allocator);
                    try input.binaryMultiply(allocator, &this.intermediary.?);
                },
                .gelu => {
                    try this.intermediary.?.binarySet(allocator, input);
                    try input.unaryMultiply(allocator, -1.702);
                    try input.unaryExp(allocator);
                    try input.unaryAdd(allocator, 1);
                    try input.unaryReciprocal(allocator);
                    try input.binaryMultiply(allocator, &this.intermediary.?);
                },
                .leaky => {
                    try this.intermediary.?.binarySet(allocator, input);
                    try this.intermediary.?.unaryMultiply(allocator, Activation.leaky_factor);
                    try input.binaryMax(allocator, &this.intermediary.?);
                },
            }
        }
        pub fn backward(this: *@This(), allocator: anytype, input: *Tensor, input_g: *Tensor) !void {
            switch (this.t) {
                .none => {},
                .relu => {
                    try this.intermediary.?.binarySet(allocator, input);
                    try this.intermediary.?.unarySign(allocator);
                    try input_g.binaryMultiply(allocator, &this.intermediary.?);
                },
                .sigmoid => {
                    // try input.unaryMultiply(allocator, -1);
                    // try input.unaryExp(allocator);
                    // try input.unaryAdd(allocator, 1);
                    // try input.unaryReciprocal(allocator);
                    try this.intermediary.?.binarySet(allocator, input);
                    try this.intermediary.?.unaryMultiply(allocator, -1);
                    try this.intermediary.?.unaryAdd(allocator, 1);
                    try this.intermediary.?.binaryMultiply(allocator, input);
                    try input_g.binaryMultiply(allocator, &this.intermediary.?);
                },
                .tanh => {
                    try input_g.unarySquare(allocator);
                    try input_g.unaryMultiply(allocator, -1);
                    try input_g.unaryAdd(allocator, 1);
                },
                .silu => {
                    unreachable;
                },
                .gelu => {
                    unreachable;
                },
                .leaky => {
                    try this.intermediary.?.binarySet(allocator, input);
                    try this.intermediary.?.unarySign(allocator);
                    try this.intermediary.?.unaryMax(allocator, Activation.leaky_factor);
                    try input_g.binaryMultiply(allocator, &this.intermediary.?);
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
        pub fn alloc(allocator: anytype, t: Norm.Type, a: u32, z: u32, y: u32, x: u32, context: ClContext) !Norm {
            return switch (t) {
                .none => .{
                    .type = t,
                    .mean = null,
                    .variance = null,
                    .max = null,
                },
                .layer => .{
                    .type = t,
                    .mean = try Tensor.alloc(allocator, a, z, y, x, context),
                    .variance = try Tensor.alloc(allocator, a, z, y, x, context),
                    .max = null,
                },
                .batch => .{
                    .type = t,
                    .mean = try Tensor.alloc(allocator, a, z, y, x, context),
                    .variance = try Tensor.alloc(allocator, a, z, y, x, context),
                    .max = null,
                },
                .simple => .{
                    .type = t,
                    .mean = null,
                    .variance = null,
                    .max = try Tensor.alloc(allocator, 1, 1, 1, 1, context),
                },
                .softmax => .{
                    .type = t,
                    // Repurposed to be the place in which to exp the values
                    .mean = null,
                    .variance = null,
                    .max = try Tensor.alloc(allocator, 1, 1, 1, 1, context),
                },
            };
        }
        pub fn free(this: *@This(), allocator: anytype) void {
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
        pub fn forward(this: *@This(), allocator: anytype, input: Tensor) !void {
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
                    input.linaryDivide(allocator, this.max.?);
                },
                .softmax => {
                    input.unaryExp(allocator);
                    this.max.?.reduceSum(allocator, input);
                    input.linaryDivide(allocator, this.max.?);
                },
            }
        }
        pub fn backward(this: *@This(), allocator: anytype, input: Tensor, input_g: Tensor) !void {
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

        pub fn alloc(allocator: anytype, size_input: u32, size_output: u32, context: ClContext) !Dense {
            assert(size_input > 0);
            assert(size_output > 0);

            return .{
                .size_input = size_input,
                .size_output = size_output,
                .weights = try Tensor.alloc(allocator, 1, 1, size_input, size_output, context),
                .biases = try Tensor.alloc(allocator, 1, 1, 1, size_output, context),
                .weights_g = try Tensor.alloc(allocator, 1, 1, size_input, size_output, context),
                .biases_g = try Tensor.alloc(allocator, 1, 1, 1, size_output, context),
                .temp_input = try Tensor.alloc(allocator, 1, 1, size_input, 1, context),
                .temp_output = try Tensor.alloc(allocator, 1, 1, 1, size_output, context),
                .temp_full = try Tensor.alloc(allocator, 1, 1, size_input, size_output, context),
            };
        }
        pub fn free(this: *@This(), allocator: anytype) void {
            this.weights.free(allocator);
            this.biases.free(allocator);
            this.weights_g.free(allocator);
            this.biases_g.free(allocator);
            this.temp_input.free(allocator);
            this.temp_output.free(allocator);
            this.temp_full.free(allocator);
        }
        /// Automagically flattens the input tensor to shape (1, 1, a * z * y * x, 1)
        pub fn forward(this: *@This(), allocator: anytype, input: *Tensor, output: *Tensor) !void {
            input.moveReshape(1, 1, this.size_input, 1);
            this.weights.moveResize(1, 1, this.size_input, 1);
            output.moveResize(1, 1, 1, 1);

            for (0..this.size_output) |row_idx_usize| {
                const row_idx: u32 = @truncate(row_idx_usize);
                this.weights.moveOffset(0, 0, 0, row_idx);
                output.moveOffset(0, 0, 0, row_idx);

                try this.temp_input.binarySet(allocator, &this.weights);
                try this.temp_input.binaryMultiply(allocator, input);
                try output.reduceSum(allocator, &this.temp_input);
            }

            input.moveReshape(input.buffer.a_inherent, input.buffer.z_inherent, input.buffer.y_inherent, input.buffer.x_inherent);
            input.moveOffset(0, 0, 0, 0);
            output.moveResize(output.buffer.a_inherent, output.buffer.z_inherent, output.buffer.y_inherent, output.buffer.x_inherent);
            output.moveOffset(0, 0, 0, 0);
            this.weights.moveResize(1, 1, this.size_input, this.size_output);
            this.weights.moveOffset(0, 0, 0, 0);

            try output.binaryAdd(allocator, &this.biases);
        }
        pub fn backward(this: *@This(), allocator: anytype, input: *Tensor, input_g: *Tensor, output_g: *Tensor) !void {
            // Biases
            try this.biases_g.binaryAdd(allocator, output_g);
            // Weights
            this.temp_full.moveResize(1, 1, 1, this.size_output);
            for (0..this.size_input) |column_idx_usize| {
                const column_idx: u32 = @truncate(column_idx_usize);
                this.temp_full.moveOffset(0, 0, column_idx, 0);
                try this.temp_full.binarySet(allocator, output_g);
            }
            this.temp_full.moveResize(1, 1, this.size_input, 1);
            input.moveReshape(1, 1, this.size_input, 1);
            for (0..this.size_output) |row_idx_usize| {
                const row_idx: u32 = @truncate(row_idx_usize);
                this.temp_full.moveOffset(0, 0, 0, row_idx);
                try this.temp_full.binaryMultiply(allocator, input);
            }
            this.temp_full.moveResize(1, 1, this.size_input, this.size_output);
            this.temp_full.moveOffset(0, 0, 0, 0);
            input.moveReshape(1, input.buffer.z_inherent, input.buffer.y_inherent, input.buffer.x_inherent);
            try this.weights_g.binaryAdd(allocator, &this.temp_full);
            // Previous activation
            input_g.moveReshape(1, 1, this.size_input, 1);
            input_g.moveResize(1, 1, 1, 1);
            this.weights.moveReshape(1, 1, 1, this.size_output);
            for (0..this.size_input) |column_idx_usize| {
                const column_idx: u32 = @truncate(column_idx_usize);
                input_g.moveOffset(0, 0, column_idx, 0);
                this.weights.moveOffset(0, 0, column_idx, 0);
                try this.temp_output.binarySet(allocator, &this.weights);
                try this.temp_output.binaryMultiply(allocator, output_g);
                // Could use sum or avg here, sum it technically more accurate but avg is more stable
                try input_g.reduceSum(allocator, &this.temp_output);
                // input_g.reduceAvg(allocator, this.temp_output);
            }
            input_g.moveReshape(1, input.buffer.z_inherent, input.buffer.y_inherent, input.buffer.x_inherent);
            input_g.moveOffset(0, 0, 0, 0);
            this.weights.moveResize(1, 1, this.size_input, this.size_output);
            this.weights.moveOffset(0, 0, 0, 0);
        }
        pub fn print(this: *@This(), comptime padding: u32, comptime offset: u32, name: ?[]const u8) void {
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
        pub fn debug(this: *@This(), comptime padding: u32, comptime offset: u32, name: ?[]u8) void {
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
            allocator: anytype,
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
                .temp_input_padded = try Tensor.alloc(allocator, 1, z, y + 2 * kernel_padding, //
                    x + 2 * kernel_padding, context),
                .temp_grad_padded = try Tensor.alloc(allocator, 1, z, y + 2 * kernel_padding, //
                    x + 2 * kernel_padding, context),
                .temp_kernel = try Tensor.alloc(allocator, 1, z, kernel_size, kernel_size, context),
                .temp_single = try Tensor.alloc(allocator, 1, 1, 1, 1, context),
            };
        }
        pub fn free(this: *@This(), allocator: anytype) void {
            this.biases.free(allocator);
            this.biases_g.free(allocator);
            this.weights.free(allocator);
            this.weights_g.free(allocator);
            this.temp_input_padded.free(allocator);
            this.temp_grad_padded.free(allocator);
            this.temp_kernel.free(allocator);
            this.temp_single.free(allocator);
        }
        pub fn forward(this: *@This(), allocator: anytype, input: *Tensor, output: *Tensor) !void {
            const x_in_max = input.buffer.x_inherent + this.kernel_padding - 1;
            const y_in_max = input.buffer.y_inherent + this.kernel_padding - 1;
            var x_out_idx: u32 = 0;
            var y_out_idx: u32 = 0;

            // TODO: Figure out a way to remove these padded temporary buffers. Could make the normal buffers padded and just downsize, but that makes things complicated.

            input.moveOffset(0, 0, 0, 0);
            this.biases.moveResize(1, 1, 1, 1);
            this.weights.moveResize(1, input.buffer.z_inherent, this.kernel_size, this.kernel_size);
            output.moveResize(1, 1, 1, 1);
            this.temp_input_padded.moveResize(1, input.buffer.z_inherent, input.buffer.y_inherent, input.buffer.x_inherent);
            this.temp_input_padded.moveOffset(0, 0, this.kernel_padding, this.kernel_padding);
            try this.temp_input_padded.binarySet(allocator, input);
            this.temp_input_padded.moveResize(1, input.buffer.z_inherent, this.kernel_size, this.kernel_size);

            for (0..this.filters) |filter_idx_usize| {
                const filter_idx: u32 = @truncate(filter_idx_usize);
                this.biases.moveOffset(filter_idx, 0, 0, 0);
                this.weights.moveOffset(filter_idx, 0, 0, 0);
                y_out_idx = 0;
                for (0..@divFloor(y_in_max, this.kernel_stride)) |y_in_idx_usize| {
                    const y_in_idx: u32 = @truncate(y_in_idx_usize);
                    x_out_idx = 0;
                    for (0..@divFloor(x_in_max, this.kernel_stride)) |x_in_idx_usize| {
                        const x_in_idx: u32 = @truncate(x_in_idx_usize);
                        output.moveOffset(0, filter_idx, y_out_idx, x_out_idx);
                        this.temp_input_padded.moveOffset(0, 0, y_in_idx * this.kernel_stride, x_in_idx * this.kernel_stride);
                        try this.temp_kernel.binarySet(allocator, &this.temp_input_padded);
                        try this.temp_kernel.binaryMultiply(allocator, &this.weights);
                        try output.reduceSum(allocator, &this.temp_kernel);
                        try output.binaryAdd(allocator, &this.biases);
                        x_out_idx += 1;
                    }
                    y_out_idx += 1;
                }
            }
            this.biases.moveResize(this.filters, 1, 1, 1);
            this.biases.moveOffset(0, 0, 0, 0);
            this.weights.moveResize(this.filters, this.z, this.kernel_size, this.kernel_size);
            this.weights.moveOffset(0, 0, 0, 0);
            output.moveResize(1, output.buffer.z_inherent, output.buffer.y_inherent, output.buffer.x_inherent);
            output.moveOffset(0, 0, 0, 0);
            this.temp_input_padded.moveResize(1, output.buffer.z_inherent, output.buffer.y_inherent + 2 * this.kernel_padding, //
                output.buffer.x_inherent + 2 * this.kernel_padding);
            this.temp_input_padded.moveOffset(0, 0, 0, 0);
        }
        pub fn backward(this: *@This(), allocator: anytype, input: *Tensor, input_g: *Tensor, output: *Tensor, output_g: *Tensor) !void {
            // Biases
            this.biases_g.moveResize(1, 1, 1, 1);
            output_g.moveResize(1, 1, output.buffer.y_inherent, output.buffer.x_inherent);
            for (0..this.filters) |filter_idx_usize| {
                const filter_idx: u32 = @truncate(filter_idx_usize);
                this.biases_g.moveOffset(filter_idx, 0, 0, 0);
                output_g.moveOffset(0, filter_idx, 0, 0);
                // Could do avg here for better numerical stability
                try this.temp_single.reduceSum(allocator, output_g);
                try this.biases_g.binaryAdd(allocator, &this.temp_single);
            }
            this.biases_g.moveResize(this.filters, 1, 1, 1);
            this.biases_g.moveOffset(0, 0, 0, 0);
            output_g.moveResize(1, output.buffer.z_inherent, output.buffer.y_inherent, output.buffer.x_inherent);
            output_g.moveOffset(0, 0, 0, 0);
            // Weights
            var x_in_idx: u32 = 0;
            var y_in_idx: u32 = 0;
            output_g.moveResize(1, 1, 1, 1);
            output_g.moveOffset(0, 0, 0, 0);
            this.weights_g.moveResize(1, input.buffer.z_inherent, this.kernel_size, this.kernel_size);
            this.weights_g.moveOffset(0, 0, 0, 0);
            this.temp_input_padded.moveResize(1, input.buffer.z_inherent, this.kernel_size, this.kernel_size);
            this.temp_input_padded.moveOffset(0, 0, 0, 0);
            for (0..this.filters) |filter_idx_usize| {
                const filter_idx: u32 = @truncate(filter_idx_usize);
                this.weights_g.moveOffset(filter_idx, 0, 0, 0);
                y_in_idx = 0;
                for (0..output.buffer.y_inherent) |y_out_idx_usize| {
                    const y_out_idx: u32 = @truncate(y_out_idx_usize);
                    x_in_idx = 0;
                    for (0..output.buffer.x_inherent) |x_out_idx_usize| {
                        const x_out_idx: u32 = @truncate(x_out_idx_usize);
                        output_g.moveOffset(0, filter_idx, y_out_idx, x_out_idx);
                        this.temp_input_padded.moveOffset(0, 0, y_in_idx, x_in_idx);
                        try this.temp_kernel.binarySet(allocator, &this.temp_input_padded);
                        try this.temp_kernel.linaryMultiply(allocator, output_g);
                        try this.weights_g.binaryAdd(allocator, &this.temp_kernel);
                        // TODO: Why was this kernel_padding in the c version?
                        x_in_idx += this.kernel_stride;
                    }
                    // TODO: Why was this kernel_padding in the c version?
                    y_in_idx += this.kernel_stride;
                }
            }
            output.moveResize(1, output.buffer.z_inherent, output_g.buffer.y_inherent, output_g.buffer.x_inherent);
            output.moveOffset(0, 0, 0, 0);
            this.weights.moveResize(this.filters, input.buffer.z_inherent, this.kernel_size, this.kernel_size);
            this.weights.moveOffset(0, 0, 0, 0);
            this.temp_grad_padded.moveResize(1, input.buffer.z_inherent, input.buffer.y_inherent, input.buffer.x_inherent);
            this.temp_grad_padded.moveOffset(0, 0, this.kernel_padding, this.kernel_padding);

            try input_g.binarySet(allocator, &this.temp_grad_padded);

            this.temp_grad_padded.moveResize(1, input.buffer.z_inherent, //
                input.buffer.y_inherent + 2 * this.kernel_padding, input.buffer.x_inherent + 2 * this.kernel_padding);
        }
        pub fn print(this: *@This(), comptime padding: u32, comptime offset: u32, name: ?[]const u8) void {
            if (name) |text| {
                std.debug.print("{s}Convolution {s}\n", .{ [1]u8{' '} ** offset, text });
            } else {
                std.debug.print("{s}Convolution\n", .{[1]u8{' '} ** offset});
            }
            const out_z: usize = @divFloor(this.z + 2 * this.kernel_padding, this.kernel_stride);
            const out_y: usize = @divFloor(this.y + 2 * this.kernel_padding, this.kernel_stride);
            const out_x: usize = @divFloor(this.x + 2 * this.kernel_padding, this.kernel_stride);
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
        pub fn debug(this: *@This(), comptime padding: u32, comptime offset: u32, name: ?[]u8) void {
            if (name) |text| {
                std.debug.print("{s}Convolution {s}\n", .{ [1]u8{' '} ** offset, text });
            } else {
                std.debug.print("{s}Convolution\n", .{[1]u8{' '} ** offset});
            }
            const out_z: usize = @divFloor(this.z + 2 * this.kernel_padding - 2 * this.kernel_size, this.kernel_stride);
            const out_y: usize = @divFloor(this.y + 2 * this.kernel_padding - 2 * this.kernel_size, this.kernel_stride);
            const out_x: usize = @divFloor(this.x + 2 * this.kernel_padding - 2 * this.kernel_size, this.kernel_stride);

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
        pub const Type = enum {
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
        // The point of the alloc is only to check that the initialisation uses valid values
        pub fn alloc(t: Reduce.Type, z: u32, y: u32, x: u32, kernel_size: u32, kernel_stride: u32) Reduce {
            assert(z > 0);
            assert(y > 0);
            assert(x > 0);
            assert(kernel_size > 0);
            assert(kernel_stride > 0);
            assert(kernel_size + kernel_stride <= y);
            assert(kernel_size + kernel_stride <= x);
            return .{
                .t = t,
                .z = z,
                .y = y,
                .x = x,
                .kernel_size = kernel_size,
                .kernel_stride = kernel_stride,
            };
        }
        pub fn forward(this: *const @This(), allocator: anytype, input: *Tensor, output: *Tensor) !void {
            input.moveResize(1, input.buffer.z_inherent, input.buffer.y_inherent, input.buffer.x_inherent);
            input.moveOffset(0, 0, 0, 0);
            output.moveResize(1, 1, 1, 1);
            output.moveOffset(0, 0, 0, 0);

            var x_out_idx: u32 = 0;
            var y_out_idx: u32 = 0;
            for (0..this.z) |channel_idx_usize| {
                const channel_idx: u32 = @truncate(channel_idx_usize);
                y_out_idx = 0;
                for (0..@divFloor(this.y - this.kernel_size + 1, this.kernel_stride)) |y_in_idx_usize| {
                    const y_in_idx: u32 = @truncate(y_in_idx_usize);
                    x_out_idx = 0;
                    for (0..@divFloor(this.x - this.kernel_size + 1, this.kernel_stride)) |x_in_idx_usize| {
                        const x_in_idx: u32 = @truncate(x_in_idx_usize);
                        input.moveOffset(0, channel_idx, y_in_idx * this.kernel_stride, x_in_idx * this.kernel_stride);
                        output.moveOffset(0, channel_idx, y_out_idx, x_out_idx);
                        // If you really want to you can move this switch outside the loops in case you care about every nanosecond
                        switch (this.t) {
                            .sum => try output.reduceSum(allocator, input),
                            .avg => try output.reduceAvg(allocator, input),
                            .max => try output.reduceMax(allocator, input),
                            .min => try output.reduceMin(allocator, input),
                        }
                        x_out_idx += 1;
                    }
                    y_out_idx += 1;
                }
            }
        }
        /// TODO: This is a mega hack that just is just a loose approximation of the real backprop
        pub fn backward(this: *const @This(), allocator: anytype, input_g: *Tensor, output_g: *Tensor) !void {
            input_g.moveResize(1, 1, this.kernel_size, this.kernel_size);
            output_g.moveResize(1, 1, 1, 1);

            var x_in_idx: u32 = 0;
            var y_in_idx: u32 = 0;
            for (0..this.z) |channel_idx_usize| {
                const channel_idx: u32 = @truncate(channel_idx_usize);
                y_in_idx = 0;
                for (0..this.y) |y_out_idx_usize| {
                    const y_out_idx: u32 = @truncate(y_out_idx_usize);
                    x_in_idx = 0;
                    for (0..this.x) |x_out_idx_usize| {
                        const x_out_idx: u32 = @truncate(x_out_idx_usize);
                        input_g.moveOffset(0, channel_idx, y_in_idx, x_in_idx);
                        output_g.moveOffset(0, channel_idx, y_out_idx, x_out_idx);
                        try input_g.linaryAdd(allocator, output_g);
                        x_in_idx += this.kernel_stride;
                    }
                    y_in_idx += this.kernel_stride;
                }
            }

            input_g.moveResize(1, input_g.buffer.z_inherent, input_g.buffer.y_inherent, input_g.buffer.x_inherent);
            input_g.moveOffset(0, 0, 0, 0);
            output_g.moveResize(1, output_g.buffer.z_inherent, output_g.buffer.y_inherent, output_g.buffer.x_inherent);
            output_g.moveOffset(0, 0, 0, 0);
        }
        pub fn print(this: *const @This(), comptime padding: u32, comptime offset: u32, name: ?[]const u8) void {
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

        pub fn alloc(allocator: anytype, filters: u32, z: u32, y: u32, x: u32, context: ClContext) !Split {
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
                .temp_input = try Tensor.alloc(allocator, 1, z, y, x, context),
            };
        }
        pub fn free(this: *@This(), allocator: anytype) void {
            this.weights.free(allocator);
            this.weights_g.free(allocator);
            this.biases.free(allocator);
            this.biases_g.free(allocator);
            this.temp_input.free(allocator);
        }
        pub fn forward(this: *@This(), allocator: anytype, input: *Tensor, output: *Tensor) !void {
            assert(input.buffer.z_inherent * this.filters == output.buffer.z_inherent);
            assert(input.buffer.y_inherent == output.buffer.y_inherent);
            assert(input.buffer.x_inherent == output.buffer.x_inherent);
            input.moveResize(1, input.buffer.z_inherent, input.buffer.y_inherent, input.buffer.x_inherent);
            input.moveOffset(0, 0, 0, 0);
            output.moveResize(1, output.buffer.z_inherent, output.buffer.y_inherent, output.buffer.x_inherent);
            output.moveOffset(0, 0, 0, 0);
            this.weights.moveResize(1, input.buffer.z_inherent, input.buffer.y_inherent, input.buffer.x_inherent);
            this.weights.moveOffset(0, 0, 0, 0);
            this.biases.moveResize(1, input.buffer.z_inherent, input.buffer.y_inherent, input.buffer.x_inherent);
            this.biases.moveOffset(0, 0, 0, 0);

            for (0..this.filters) |filter_idx_usize| {
                const filter_idx: u32 = @truncate(filter_idx_usize);
                output.moveOffset(0, filter_idx * input.buffer.z_inherent, 0, 0);
                this.weights.moveOffset(filter_idx, 0, 0, 0);
                this.biases.moveOffset(filter_idx, 0, 0, 0);
                try output.binarySet(allocator, input);
                try output.binaryMultiply(allocator, &this.weights);
                try output.binaryAdd(allocator, &this.biases);
            }

            input.moveResize(1, input.buffer.z_inherent, input.buffer.y_inherent, input.buffer.x_inherent);
            input.moveOffset(0, 0, 0, 0);
            output.moveResize(1, output.buffer.z_inherent, output.buffer.y_inherent, output.buffer.x_inherent);
            output.moveOffset(0, 0, 0, 0);
            this.weights.moveResize(this.filters, input.buffer.z_inherent, input.buffer.y_inherent, input.buffer.x_inherent);
            this.weights.moveOffset(0, 0, 0, 0);
            this.biases.moveResize(this.filters, input.buffer.z_inherent, input.buffer.y_inherent, input.buffer.x_inherent);
            this.biases.moveOffset(0, 0, 0, 0);
        }
        pub fn backward(this: *@This(), allocator: anytype, input: *Tensor, input_g: *Tensor, output_g: *Tensor) !void {
            this.biases_g.moveResize(1, this.z * this.filters, this.y, this.x);
            this.biases_g.moveOffset(0, 0, 0, 0);
            try this.biases_g.binarySet(allocator, output_g);
            this.biases_g.moveResize(this.filters, this.z, this.y, this.x);
            this.biases_g.moveOffset(0, 0, 0, 0);

            this.weights_g.moveResize(1, this.z, this.y, this.x);
            output_g.moveResize(1, this.z, this.y, this.x);
            for (0..this.filters) |filter_idx_usize| {
                const filter_idx: u32 = @truncate(filter_idx_usize);
                this.weights_g.moveOffset(filter_idx, 0, 0, 0);
                output_g.moveOffset(0, filter_idx * this.z, 0, 0);
                try this.temp_input.binarySet(allocator, output_g);
                try this.temp_input.binaryMultiply(allocator, input);
                try this.weights_g.binaryAdd(allocator, &this.temp_input);
            }
            this.weights_g.moveResize(this.filters, this.z, this.y, this.x);
            this.weights_g.moveOffset(0, 0, 0, 0);
            output_g.moveResize(1, this.filters * this.z, this.y, this.x);
            output_g.moveOffset(0, 0, 0, 0);

            output_g.moveResize(1, this.z, this.y, this.x);
            this.weights.moveResize(1, this.z, this.y, this.x);
            for (0..this.filters) |filter_idx_usize| {
                const filter_idx: u32 = @truncate(filter_idx_usize);
                this.weights.moveOffset(filter_idx, 0, 0, 0);
                output_g.moveOffset(0, filter_idx * this.z, 0, 0);
                try this.temp_input.binarySet(allocator, output_g);
                try this.temp_input.binaryMultiply(allocator, &this.weights);
                try input_g.binaryAdd(allocator, &this.temp_input);
            }
            this.weights.moveResize(this.filters, this.z, this.y, this.x);
            this.weights.moveOffset(0, 0, 0, 0);
            output_g.moveResize(1, this.filters * this.z, this.y, this.x);
            output_g.moveOffset(0, 0, 0, 0);
        }
        // TODO: Split backward
        pub fn print(this: *@This(), comptime padding: u32, comptime offset: u32, name: ?[]const u8) void {
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
            this.biases.print(padding, offset + padding, "biases"[0..]);
            this.biases_g.print(padding, offset + padding, "biases_g"[0..]);
            this.weights.print(padding, offset + padding, "weights"[0..]);
            this.weights_g.print(padding, offset + padding, "weights_g"[0..]);
            this.temp_input.print(padding, offset + padding, "temp_input"[0..]);
        }
        pub fn debug(this: *@This(), comptime padding: u32, comptime offset: u32, name: ?[]u8) void {
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
            this.biases.debug(padding, offset + padding, "biases");
            this.biases_g.debug(padding, offset + padding, "biases_g");
            this.weights.debug(padding, offset + padding, "weights");
            this.weights_g.debug(padding, offset + padding, "weights_g");
            this.temp_input.debug(padding, offset + padding, "temp_input");
        }
    };
    pub const Residual = struct {
        pub const Type = enum {
            identity,
            convolution,
            dense,
            reduce,
            split,
        };
        const Connection = union(enum) {
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
        pub fn allocDense(layer: u32, allocator: anytype, size_in: u32, size_out: u32, context: ClContext) !Residual {
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
            allocator: anytype,
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
                    .reduce = Reduce.alloc(t, z, y, x, kernel_size, kernel_stride),
                },
            };
        }
        pub fn allocSplit(layer: u32, allocator: anytype, filters: u32, z: u32, y: u32, x: u32, context: ClContext) !Residual {
            return .{
                .layer = layer,
                .t = .split,
                .connection = .{
                    .split = try Split.alloc(allocator, filters, z, y, x, context),
                },
            };
        }
        pub fn free(this: *@This(), allocator: anytype) void {
            switch (this.connection) {
                .identity => {},
                .convolution => this.connection.convolution.free(allocator),
                .dense => this.connection.dense.free(allocator),
                .reduce => {},
                .split => {},
            }
        }
        pub fn forward(this: *@This(), allocator: anytype, input: *Tensor, output: *Tensor) !void {
            assert(input.buffer.a_inherent == output.buffer.a_inherent);
            assert(input.buffer.z_inherent == output.buffer.z_inherent);
            assert(input.buffer.y_inherent == output.buffer.y_inherent);
            assert(input.buffer.x_inherent == output.buffer.x_inherent);

            input.moveResize(input.buffer.a_inherent, input.buffer.z_inherent, input.buffer.y_inherent, input.buffer.x_inherent);
            input.moveOffset(0, 0, 0, 0);
            output.moveResize(input.buffer.a_inherent, input.buffer.z_inherent, input.buffer.y_inherent, input.buffer.x_inherent);
            output.moveOffset(0, 0, 0, 0);

            switch (this.t) {
                .identity => try output.binaryAdd(allocator, input),
                .convolution => unreachable,
                .dense => unreachable,
                .reduce => unreachable,
                .split => unreachable,
            }
        }
        pub fn backward(this: *@This(), allocator: anytype, input_g: *Tensor, output_g: *Tensor) !void {
            assert(input_g.buffer.a_inherent == output_g.buffer.a_inherent);
            assert(input_g.buffer.z_inherent == output_g.buffer.z_inherent);
            assert(input_g.buffer.y_inherent == output_g.buffer.y_inherent);
            assert(input_g.buffer.x_inherent == output_g.buffer.x_inherent);

            input_g.moveResize(input_g.buffer.a_inherent, input_g.buffer.z_inherent, input_g.buffer.y_inherent, input_g.buffer.x_inherent);
            input_g.moveOffset(0, 0, 0, 0);
            output_g.moveResize(input_g.buffer.a_inherent, input_g.buffer.z_inherent, input_g.buffer.y_inherent, input_g.buffer.x_inherent);
            output_g.moveOffset(0, 0, 0, 0);

            switch (this.t) {
                .identity => try input_g.binaryAdd(allocator, output_g),
                .convolution => unreachable,
                .dense => unreachable,
                .reduce => unreachable,
                .split => unreachable,
            }
        }
        pub fn print(this: *@This(), comptime padding: u32, comptime offset: u32, name: ?[]const u8) void {
            if (name) |text| {
                std.debug.print("{s}Residual {s}\n", .{ [1]u8{' '} ** offset, text });
            } else {
                std.debug.print("{s}Residual\n", .{[1]u8{' '} ** offset});
            }

            std.debug.print("{s}Type {}\n", .{ [1]u8{' '} ** (offset + padding), this.t });
            switch (this.connection) {
                .identity => {},
                .convolution => this.connection.convolution.print(padding, offset + padding, "convolution"[0..]),
                .dense => this.connection.dense.print(padding, offset + padding, "dense"[0..]),
                .reduce => this.connection.reduce.print(padding, offset + padding, "reduce"[0..]),
                .split => this.connection.split.print(padding, offset + padding, "split"[0..]),
            }
        }
        pub fn debug(this: *@This(), comptime padding: u32, comptime offset: u32, name: ?[]u8) void {
            if (name) |text| {
                std.debug.print("{s}Residual {s}\n", .{ [1]u8{' '} ** offset, text });
            } else {
                std.debug.print("{s}Residual\n", .{[1]u8{' '} ** offset});
            }

            std.debug.print("{s}Type {}\n", .{ [1]u8{' '} ** (offset + padding), this.t });
            switch (this.connection) {
                .identity => {},
                .convolution => this.connection.convolution.debug(padding, offset + padding, "convolution"[0..]),
                .dense => this.connection.dense.debug(padding, offset + padding, "dense"[0..]),
                .reduce => this.connection.reduce.print(padding, offset + padding, "reduce"[0..]),
                .split => this.connection.split.debug(padding, offset + padding, "split"[0..]),
            }
        }
    };
    pub const Layer = struct {
        /// The config is to only have only the info needed when provided with the previous layer
        pub const Config = union(enum) {
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
                // activation: Activation.Type,
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
                // activation: Activation.Type,
            },
        };
        // TODO: Come up with better name
        compute: union(Type) {
            dense: Dense,
            convolution: Convolution,
            reduce: Reduce,
            split: Split,
            residual: Residual,
        },
        values: Tensor,
        values_g: Tensor,
        activation: Activation,
        pub fn alloc(allocator: anytype, z: u32, y: u32, x: u32, config: Config, context: ClContext) !Layer {
            switch (config) {
                .dense => {
                    return .{
                        .activation = try Activation.alloc(allocator, config.dense.activation, 1, 1, 1, config.dense.size_out, context),
                        .compute = .{ .dense = try Dense.alloc(allocator, z * y * x, config.dense.size_out, context) },
                        .values = try Tensor.alloc(allocator, 1, 1, 1, config.dense.size_out, context),
                        .values_g = try Tensor.alloc(allocator, 1, 1, 1, config.dense.size_out, context),
                    };
                },
                .convolution => {
                    const z_new: u32 = config.convolution.filters;
                    const y_new: u32 = @divFloor(y + 2 * config.convolution.kernel_padding - config.convolution.kernel_size, //
                        config.convolution.kernel_stride + 1);
                    const x_new: u32 = @divFloor(x + 2 * config.convolution.kernel_padding - config.convolution.kernel_size, //
                        config.convolution.kernel_stride + 1);
                    return .{
                        .activation = try Activation.alloc(allocator, config.convolution.activation, 1, z_new, y_new, x_new, context),
                        .compute = .{
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
                    const y_new: u32 = @divFloor(y - config.convolution.kernel_size, config.convolution.kernel_stride + 1);
                    const x_new: u32 = @divFloor(x - config.convolution.kernel_size, config.convolution.kernel_stride + 1);
                    return .{
                        .activation = try Activation.alloc(allocator, .none, 1, z_new, y_new, x_new, context),
                        .compute = .{
                            .reduce = Reduce.alloc(config.reduce.t, z, y, x, config.reduce.kernel_size, //
                                config.reduce.kernel_stride),
                        },
                        .values = try Tensor.alloc(allocator, 1, z_new, y_new, x_new, context),
                        .values_g = try Tensor.alloc(allocator, 1, z_new, y_new, x_new, context),
                    };
                },
                .split => {
                    return .{
                        .activation = try Activation.alloc(allocator, config.split.activation, 1, z * config.split.filters, y, x, context),
                        .compute = .{ .split = try Split.alloc(allocator, config.split.filters, z, y, x, context) },
                        .values = try Tensor.alloc(allocator, 1, z * config.split.filters, y, x, context),
                        .values_g = try Tensor.alloc(allocator, 1, z * config.split.filters, y, x, context),
                    };
                },
                .residual => {
                    return .{
                        .activation = try Activation.alloc(allocator, .none, 1, z, y, x, context),
                        .compute = .{
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
        pub fn free(this: *@This(), allocator: anytype) !void {
            switch (this.compute) {
                .dense => this.compute.dense.free(allocator),
                .convolution => this.compute.convolution.free(allocator),
                .reduce => {},
                .split => this.compute.split.free(allocator),
                .residual => this.compute.residual.free(allocator),
            }
            if (this.activation.intermediary) |*intermediary| {
                intermediary.free(allocator);
            }
            this.values.free(allocator);
            this.values_g.free(allocator);
        }
        pub fn print(this: *@This(), comptime padding: u32, comptime offset: u32, name: ?[]u8) void {
            if (name) |text| {
                std.debug.print("{s}Layer {s}\n", .{ [1]u8{' '} ** offset, text });
            } else {
                std.debug.print("{s}Layer\n", .{[1]u8{' '} ** offset});
            }

            std.debug.print("{s}Type {}\n", .{ [1]u8{' '} ** (padding + offset), this.activation.t });
            switch (this.compute) {
                .dense => this.compute.dense.print(padding, offset + padding, null),
                .convolution => this.compute.convolution.print(padding, offset + padding, null),
                .reduce => this.compute.reduce.print(padding, offset + padding, null),
                .split => this.compute.split.print(padding, offset + padding, null),
                .residual => this.compute.residual.print(padding, offset + padding, null),
            }
        }
        pub fn debug(this: *@This(), comptime padding: u32, comptime offset: u32, name: ?[]u8) void {
            if (name) |text| {
                std.debug.print("{s}Layer {s}\n", .{ [1]u8{' '} ** offset, text });
            } else {
                std.debug.print("{s}Layer\n", .{[1]u8{' '} ** offset});
            }

            std.debug.print("{s}Type {}\n", .{ [1]u8{' '} ** (padding + offset), this.activation.type });
            switch (this.compute) {
                .dense => this.compute.dense.debug(padding, offset + padding, null),
                .convolution => this.compute.convolution.debug(padding, offset + padding, null),
                .reduce => this.compute.reduce.print(padding, offset + padding, null),
                .split => this.compute.split.debug(padding, offset + padding, null),
                .residual => this.compute.residual.debug(padding, offset + padding, null),
            }
        }
    };
    input: Tensor,
    layers: []Layer,
    forward: Linearized,
    backward: Linearized,
    forward_cl: Program,
    backward_cl: Program,
    pub fn alloc(
        allocator: anytype,
        input: Tensor,
        config: []const Layer.Config,
        size_global: u32,
        size_local: u32,
        context: ClContext,
        device: ClDevice,
        queue: ClCommandQueue,
    ) !Neuralnet {
        var layers: []Layer = try allocator.alloc(Layer, config.len);
        var z_previous: u32 = input.buffer.z_inherent;
        var y_previous: u32 = input.buffer.y_inherent;
        var x_previous: u32 = input.buffer.x_inherent;
        for (0..layers.len) |layer_idx| {
            layers[layer_idx] = try Layer.alloc(allocator, z_previous, y_previous, x_previous, config[layer_idx], context);
            z_previous = layers[layer_idx].values.buffer.z_inherent;
            y_previous = layers[layer_idx].values.buffer.y_inherent;
            x_previous = layers[layer_idx].values.buffer.x_inherent;
        }
        var previous_values: Tensor = input;
        for (0..layers.len) |layer_idx| {
            switch (layers[layer_idx].compute) {
                .dense => {
                    try layers[layer_idx].compute.dense.forward(allocator, &previous_values, &layers[layer_idx].values);
                },
                .convolution => {
                    try layers[layer_idx].compute.convolution.forward(allocator, &previous_values, &layers[layer_idx].values);
                },
                .reduce => {
                    try layers[layer_idx].compute.reduce.forward(allocator, &previous_values, &layers[layer_idx].values);
                },
                .split => {
                    try layers[layer_idx].compute.split.forward(allocator, &previous_values, &layers[layer_idx].values);
                },
                .residual => {
                    try layers[layer_idx].compute.residual.forward(allocator, //
                        &layers[layers[layer_idx].compute.residual.layer].values, &layers[layer_idx].values);
                },
            }

            try layers[layer_idx].activation.forward(allocator, &layers[layer_idx].values);
            // TODO: Norming

            previous_values = layers[layer_idx].values;
        }
        var forward: Linearized = try Linearized.alloc(allocator);
        try forward.concat(allocator, &layers[layers.len - 1].values.linearized);

        for (0..layers.len - 1) |layer_idx_reverse| {
            const layer_idx: usize = layers.len - (layer_idx_reverse + 1);

            // TODO: Norming
            try layers[layer_idx].activation.backward(allocator, &layers[layer_idx].values, &layers[layer_idx].values_g);

            switch (layers[layer_idx].compute) {
                .dense => {
                    try layers[layer_idx].compute.dense.backward(allocator, &layers[layer_idx - 1].values, //
                        &layers[layer_idx - 1].values_g, &layers[layer_idx].values_g);
                },
                .convolution => {
                    try layers[layer_idx].compute.convolution.backward(allocator, &layers[layer_idx - 1].values, //
                        &layers[layer_idx - 1].values_g, &layers[layer_idx].values, &layers[layer_idx].values_g);
                },
                .reduce => {
                    try layers[layer_idx].compute.reduce.backward(allocator, &layers[layer_idx - 1].values_g, //
                        &layers[layer_idx].values_g);
                },
                .split => {
                    try layers[layer_idx].compute.split.backward(allocator, &layers[layer_idx - 1].values, //
                        &layers[layer_idx - 1].values_g, &layers[layer_idx].values_g);
                },
                .residual => {
                    try layers[layer_idx].compute.residual.backward(allocator, //
                        &layers[layers[layer_idx].compute.residual.layer].values_g, &layers[layer_idx].values_g);
                },
            }
        }

        var backward: Linearized = try Linearized.alloc(allocator);
        try backward.concat(allocator, &layers[0].values.linearized);

        const forward_cl: Program = try Program.alloc(allocator, forward, size_global, //
            size_local, device, context, queue);
        const backward_cl: Program = try Program.alloc(allocator, backward, size_global, //
            size_local, device, context, queue);
        return .{
            .input = input,
            .layers = layers,
            .forward = forward,
            .backward = backward,
            .forward_cl = forward_cl,
            .backward_cl = backward_cl,
        };
    }
    pub fn free(this: *@This(), allocator: anytype) !void {
        for (0..this.layers.len) |layer_idx| {
            try this.layers[layer_idx].free(allocator);
        }
        allocator.free(this.layers);
        // this.input.free(allocator);
        this.forward.free(allocator);
        this.backward.free(allocator);
        try this.forward_cl.free(allocator);
        try this.backward_cl.free(allocator);
    }
    pub fn print(this: *@This(), comptime padding: u32, comptime offset: u32, name: ?[]const u8) void {
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
};
