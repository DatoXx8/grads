const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const Cl = @import("../runtimes/cl.zig");
const ClDevice = Cl.ClDevice;
const ClContext = Cl.ClContext;
const ClCommandQueue = Cl.ClCommandQueue;
const ClError = Cl.ClError;
const ClProgram = Cl.ClProgram;
const opencl = Cl.opencl;
const Linearized = @import("../tensor.zig").Linearized;
const source_padding = @import("./codegen.zig").source_padding;
const kernel_base_name = @import("./codegen.zig").kernel_base_name;
const compileKernel = @import("./codegen.zig").compileKernel;
const Kernel = @import("./kernel.zig").Kernel;
const Args = @import("./kernel.zig").Args;
const Optimization = @import("./optimize.zig").Optimization;
const Ssa = @import("./ssa.zig").Ssa;
const Assign = @import("./ssa.zig").Assign;

pub const Program = struct {
    size_global: usize,
    size_local: usize,
    kernel: []Kernel,
    program: ClProgram,
    // $Convert to [*:0]u8 with source[0.. 0]
    source: []const u8,
    queue: ClCommandQueue,
    pub fn alloc(
        allocator: Allocator,
        linearized: Linearized,
        size_global: u32,
        size_local: u32,
        optimization: Optimization,
        device: ClDevice,
        context: ClContext,
        queue: ClCommandQueue,
    ) !Program {
        assert(size_global >= size_local);
        assert(size_global % size_local == 0);

        // Return default program that does nothing
        if (linearized.op_num == 0) {
            const source: []const u8 = try allocator.dupe(u8, "__kernel void unused() {}\n\x00");
            const program: ClProgram = try ClProgram.alloc(allocator, context, device, source);
            var kernel: []Kernel = try allocator.alloc(Kernel, 1);
            kernel[0] = try Kernel.alloc(program, "unused\x00", .{ .arg_mem = &.{}, .arg_id = &.{} });

            return .{
                .size_global = size_global,
                .size_local = size_local,
                .kernel = kernel,
                .program = program,
                .source = source,
                .queue = queue,
            };
        }

        var ssa: Ssa = try Ssa.alloc(allocator, linearized);
        defer ssa.free(allocator);

        try ssa.optimize(allocator, optimization);

        var source: []u8 = try allocator.alloc(u8, source_padding);
        errdefer allocator.free(source);
        @memset(source, 0);
        var source_len: usize = 0;

        var kernel_args: []Args = try allocator.alloc(Args, ssa.assign_num);
        defer allocator.free(kernel_args);

        const kernel_name: []u8 = try allocator.alloc(u8, (kernel_base_name.len - "{}"[0..].len) +
            if (ssa.assign_num == 0) 0 else std.math.log10_int(ssa.assign_num) + 2);
        defer allocator.free(kernel_name);

        var kernel_num: u32 = 0;
        var assign_idx: u32 = 0;

        for (0..ssa.assign_num) |_| {
            if (assign_idx == ssa.assign_num) {
                break;
            }
            assert(assign_idx < ssa.assign_num);

            var assign_idx_top: u32 = assign_idx + 1;
            for (assign_idx + 1..ssa.assign_num) |assign_idx_search| {
                if (ssa.assign[assign_idx].base.layer() == ssa.assign[assign_idx_search].base.layer()) {
                    if (ssa.assign_loop_id[assign_idx] == ssa.assign_loop_id[assign_idx_search]) {
                        assign_idx_top += 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            const layer: []Assign = ssa.assign[assign_idx..assign_idx_top];

            // $NOTE This should be enough work to justify storing it in memory
            // $TODO Rethink this when I refactor the args gathering
            kernel_args[kernel_num] = try Args.alloc(allocator, layer);

            @memset(kernel_name, 0);
            const kernel_name_len: usize = (try std.fmt.bufPrint(kernel_name, kernel_base_name, .{kernel_num})).len;

            try compileKernel(allocator, &source, &source_len, layer, ssa.assign_loop_num[ssa.assign_loop_id[assign_idx]], //
                kernel_name[0..kernel_name_len], kernel_args[kernel_num], size_global, size_local);

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
    pub fn free(this: @This(), allocator: Allocator) void {
        for (this.kernel) |*kernel| {
            kernel.free(allocator);
        }
        allocator.free(this.kernel);
        allocator.free(this.source);
        this.program.free() catch |err| {
            std.log.err("Could not free program because of error {!}\n", .{err});
        };
    }
    pub fn run(this: @This()) !void {
        for (this.kernel) |kernel| {
            // $TODO kernel.kernel.kernel is hilarious but should not be a thing
            if (opencl.clEnqueueNDRangeKernel(this.queue.queue, kernel.kernel.kernel, //
                1, null, &this.size_global, &this.size_local, 0, null, null) != 0)
            {
                return ClError.ProgramNotRun;
            }
        }
        if (opencl.clFinish(this.queue.queue) != 0) {
            return ClError.QueueCouldNotWait;
        }
    }
};
