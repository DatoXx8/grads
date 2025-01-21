const Kernel = @import("./kernel.zig").Kernel;

const Cl = @import("../runtimes/cl.zig");
const ClDevice = Cl.ClDevice;
const ClContext = Cl.ClContext;
const ClCommandQueue = Cl.ClCommandQueue;
const ClError = Cl.ClError;
const open_cl = Cl.open_cl;

const Linearized = @import("../tensor.zig").Linearized;
const Pir = @import("./pir.zig").Pir;

const assert = std.debug.assert;

const Optimization = @import("./codegen.zig").Optimization;

const std = @import("std");

pub const Program = struct {
    size_global: usize,
    size_local: usize,
    kernel_num: u32,
    kernel: []Kernel,
    // device: *ClDevice,
    // context: *ClContext,
    // TODO: Decide wether to save the queue or not
    // Also decide how to handle the quees in Program.free()
    queue: ClCommandQueue,
    pub fn alloc(
        allocator: anytype,
        linearized: Linearized,
        size_global: u32,
        size_local: u32,
        optimization: Optimization,
        device: ClDevice,
        context: ClContext,
        queue: ClCommandQueue,
    ) !Program {
        const capacity_initial: u32 = 4;
        var op_used: u32 = 0;
        var kernel: []Kernel = try allocator.alloc(Kernel, capacity_initial);
        errdefer allocator.free(kernel);
        var kernel_num: u32 = 0;

        // TODO: Allocate all the pirs at once in steps

        for (0..linearized.op_num) |_| {
            if (linearized.op_num == op_used) {
                break;
            }
            var pir: Pir = try Pir.alloc(allocator, linearized, &op_used);
            defer pir.free(allocator);
            pir.optimize(optimization);
            pir.print(4, 0, null);

            if (kernel_num == kernel.len) {
                kernel = try allocator.realloc(kernel, kernel.len * 2);
            }
            kernel[kernel_num] = try Kernel.alloc(allocator, context, device, pir, size_global, size_local);
            kernel_num += 1;
        }
        assert(op_used == linearized.op_num);

        return .{
            .size_global = size_global,
            .size_local = size_local,
            .kernel_num = kernel_num,
            .kernel = kernel,
            .queue = queue,
        };
    }
    pub fn free(this: @This(), allocator: anytype) !void {
        for (0..this.kernel_num) |kernel_idx| {
            try this.kernel[kernel_idx].free(allocator);
        }
        allocator.free(this.kernel);
    }
    pub fn run(this: @This()) !void {
        for (0..this.kernel_num) |kernel_idx| {
            std.debug.print("{} => {s}\n", .{ kernel_idx, this.kernel[kernel_idx].source });
            if (open_cl.clEnqueueNDRangeKernel(this.queue.queue, this.kernel[kernel_idx].kernel.kernel, //
                1, null, &this.size_global, &this.size_local, 0, null, null) != 0)
            {
                return ClError.ProgramNotRun;
            }
            if (open_cl.clFinish(this.queue.queue) != 0) {
                return ClError.QueueCouldNotWait;
            }
        }
    }
};
