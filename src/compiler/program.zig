const Kernel = @import("./kernel.zig").Kernel;

const Cl = @import("../runtimes/cl.zig");
const ClDevice = Cl.ClDevice;
const ClContext = Cl.ClContext;
const ClCommandQueue = Cl.ClCommandQueue;
const ClError = Cl.ClError;
const ClProgram = Cl.ClProgram;
const open_cl = Cl.open_cl;

const Linearized = @import("../tensor.zig").Linearized;

const Ssa = @import("./ssa.zig").Ssa;

const assert = std.debug.assert;

const Optimization = @import("./optimize.zig").Optimization;

const compileKernel = @import("./codegen.zig").compileKernel;
const source_capacity_min = @import("./codegen.zig").capacity_min;
const kernel_base_name = @import("./codegen.zig").kernel_base_name;

const Args = @import("./kernel.zig").Args;

const std = @import("std");

pub const Program = struct {
    size_global: usize,
    size_local: usize,
    kernel: []Kernel,
    program: ClProgram,
    // Convert to [*:0]u8 with source[0.. :0]
    source: []u8,
    queue: ClCommandQueue,
    pub fn alloc(
        allocator: std.mem.Allocator,
        linearized: Linearized,
        size_global: usize,
        size_local: usize,
        optimization: Optimization,
        device: ClDevice,
        context: ClContext,
        queue: ClCommandQueue,
    ) !Program {
        var ssa: Ssa = try Ssa.alloc(allocator, linearized);
        defer ssa.free(allocator);

        try ssa.optimize(allocator, optimization);

        var source: []u8 = try allocator.alloc(u8, source_capacity_min);
        var source_len: usize = 0;
        @memset(source, 0);
        var kernel_args: []Args = try allocator.alloc(Args, ssa.assign_num);
        defer allocator.free(kernel_args);

        const kernel_name: []u8 = try allocator.alloc(u8, (kernel_base_name.len - "{}"[0..].len) +
            if (ssa.assign_num == 0) 0 else std.math.log10_int(ssa.assign_num) + 2);
        errdefer allocator.free(kernel_name);
        defer allocator.free(kernel_name);

        var kernel_num: usize = 0;
        var assign_idx: usize = 0;
        for (0..ssa.assign_num) |_| {
            if (assign_idx == ssa.assign_num) {
                break;
            }
            assert(assign_idx < ssa.assign_num);

            var assign_idx_top: usize = assign_idx + 1;
            for (assign_idx + 1..ssa.assign_num) |assign_search_idx| {
                if (ssa.assign[assign_idx].base.layer() == ssa.assign[assign_search_idx].base.layer()) {
                    assign_idx_top += 1;
                } else {
                    break;
                }
            }
            const layer: []Ssa.Assign = ssa.assign[assign_idx..assign_idx_top];

            // NOTE: This should be enough work to justify storing it in memory
            // TODO: Rethink this when I refactor the args gathering
            kernel_args[kernel_num] = try Args.alloc(allocator, layer);
            // NOTE: The \x00 is to make the string 0-terminated
            @memset(kernel_name, 0);
            const kernel_name_len: usize = (try std.fmt.bufPrint(kernel_name, kernel_base_name, .{kernel_num})).len;

            try compileKernel(allocator, &source, &source_len, layer, 0, 1, kernel_args[kernel_num], //
                kernel_name[0..kernel_name_len], size_global, size_local);

            kernel_num += 1;
            assign_idx = assign_idx_top;
        }

        const program: ClProgram = try ClProgram.alloc(allocator, context, device, source);
        var kernel: []Kernel = try allocator.alloc(Kernel, kernel_num);

        for (0..kernel_num) |kernel_idx| {
            @memset(kernel_name, 0);
            const kernel_name_len: usize = (try std.fmt.bufPrint(kernel_name, kernel_base_name ++ "\x00", .{kernel_idx})).len;
            kernel[kernel_idx] = try Kernel.alloc(program, kernel_name[0..kernel_name_len], kernel_args[kernel_idx]);
        }

        return .{
            .size_global = size_global,
            .size_local = size_local,
            .kernel = kernel,
            .program = program,
            .source = source,
            .queue = queue,
        };
    }
    pub fn free(this: @This(), allocator: std.mem.Allocator) !void {
        for (this.kernel) |*kernel| {
            try kernel.free(allocator);
        }
        allocator.free(this.kernel);
        allocator.free(this.source);
        try this.program.free();
    }
    pub fn run(this: @This()) !void {
        for (this.kernel) |kernel| {
            // TODO: kernel.kernel.kernel is hilarious but should not be a thing
            if (open_cl.clEnqueueNDRangeKernel(this.queue.queue, kernel.kernel.kernel, //
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
