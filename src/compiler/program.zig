const Kernel = @import("./kernel.zig").Kernel;

const Cl = @import("../runtimes/cl.zig");
const ClDevice = Cl.ClDevice;
const ClContext = Cl.ClContext;
const ClCommandQueue = Cl.ClCommandQueue;
const ClError = Cl.ClError;
const OpenCl = Cl.OpenCl;

const Linearized = @import("../tensor.zig").Linearized;
const Pir = @import("./pir.zig").Pir;

const assert = @import("../util.zig").assert;

const Optimisation = @import("./codegen.zig").Optimisation;

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
    command_queue: ClCommandQueue,
    pub fn alloc(
        allocator: anytype,
        linearized: Linearized,
        size_global: u32,
        size_local: u32,
        device: ClDevice,
        context: ClContext,
        command_queue: ClCommandQueue,
    ) !Program {
        const capacity_initial: u32 = 4;
        var op_used: u32 = 0;
        var kernel: []Kernel = try allocator.alloc(Kernel, capacity_initial);
        errdefer allocator.free(kernel);
        var kernel_num: u32 = 0;

        for (0..linearized.op_num) |_| {
            // TODO: Support multiple PIRs per kernel
            var pir: Pir = try Pir.alloc(allocator, linearized, &op_used);
            defer pir.free(allocator);

            if (kernel_num == kernel.len) {
                kernel = try allocator.realloc(kernel, kernel.len * 2);
            }
            kernel[kernel_num] = try Kernel.alloc(allocator, context, device, pir, size_global, size_local, .O0);
            kernel_num += 1;
        }
        assert(op_used == linearized.op_num);

        return .{
            .size_global = size_global,
            .size_local = size_local,
            .kernel_num = kernel_num,
            .kernel = kernel,
            .command_queue = command_queue,
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
            var err: i32 = OpenCl.clEnqueueNDRangeKernel(this.command_queue.queue, this.kernel[kernel_idx].kernel.kernel, //
                1, null, &this.size_global, &this.size_local, 0, null, null);
            if (err != 0) {
                return ClError.ProgramNotRun;
            }
            err = OpenCl.clFinish(this.command_queue.queue);
            if (err != 0) {
                return ClError.QueueCouldNotWait;
            }
        }
    }
};
