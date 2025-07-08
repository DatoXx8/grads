const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const Tensor = @import("../Tensor.zig");
const Linearized = Tensor.Linearized;
const opt = @import("./optimize.zig");
const Optimization = opt.Optimization;
const Runtime = @import("./runtimes/Runtime.zig");
const Ssa = @import("./Ssa.zig");
const Assign = Ssa.Assign;

pub const Memory = *anyopaque;
// $TODO Support integer arguments
pub const Args = struct {
    /// arg_num is separate to make kernels with no arguments easier to handle
    arg_num: u64,
    arg_id: []u64,
    arg_mem: []Memory,
    pub fn alloc(allocator: Allocator, assign: Assign) !Args {
        var arg_unique = std.AutoHashMap(u64, Memory).init(allocator);
        errdefer arg_unique.deinit();
        defer arg_unique.deinit();

        try arg_unique.put(assign.base.out.id, assign.base.out.values_runtime);
        if (!assign.base.type.isUnary()) {
            try arg_unique.put(assign.base.in.id, assign.base.in.values_runtime);
        }

        if (assign.inlined) |inlined| {
            // $TODO If the thing is inlined then don't there is no need to pass it to the kernel
            for (0..inlined.inlined_num) |inlined_idx| {
                try arg_unique.put(inlined.base[inlined_idx].out.id, //
                    inlined.base[inlined_idx].out.values_runtime);
                if (!inlined.base[inlined_idx].type.isUnary()) {
                    try arg_unique.put(inlined.base[inlined_idx].in.id, //
                        inlined.base[inlined_idx].in.values_runtime);
                }
            }
        }

        const arg_num: usize = arg_unique.count();
        const arg_id: []u64 = try allocator.alloc(u64, arg_num);
        const arg_mem: []Memory = try allocator.alloc(Memory, arg_num);
        var arg_id_iterator = arg_unique.keyIterator();
        for (0..arg_num) |arg_idx| {
            arg_id[arg_idx] = arg_id_iterator.next().?.*;
            arg_mem[arg_idx] = arg_unique.get(arg_id[arg_idx]) orelse @panic("Severe error in argument gathering."); // This should be doable differently
        }

        return .{
            .arg_num = arg_num,
            .arg_id = arg_id,
            .arg_mem = arg_mem,
        };
    }
    pub fn allocEmpty(allocator: Allocator) !Args {
        return .{
            .arg_num = 0,
            .arg_id = try allocator.alloc(Memory, 1),
            .arg_mem = try allocator.alloc(Memory, 1),
        };
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
        allocator.free(this.arg_id);
        // The arg_mem get's freed with the tensors
        allocator.free(this.arg_mem);
    }
};
pub const Sync = enum(u8) { sync_to_none, sync_to_device, sync_to_host };
pub const KernelPtr = *anyopaque;
// $TODO Support multiple devices
pub const Kernel = struct {
    args: Args,
    ptr: KernelPtr,
};
pub const ProgramPtr = *anyopaque;
pub const Program = @This();
size_global: u32,
size_local: u32,
kernel: []Kernel,
ptr: ProgramPtr,

pub const kernel_base_name = "kern{}";
pub const source_padding = 4096;
pub fn alloc(
    runtime: Runtime,
    allocator: Allocator,
    linearized: Linearized,
    optimization: Optimization,
    size_global: u32,
    size_local: u32,
) !Program {
    assert(size_global >= size_local);
    assert(size_global % size_local == 0);

    var ssa: Ssa = try Ssa.alloc(allocator, linearized, optimization);
    errdefer ssa.free(allocator);
    defer ssa.free(allocator);

    var source: []u8 = try allocator.alloc(u8, source_padding);
    errdefer allocator.free(source);
    defer allocator.free(source);
    @memset(source, 0);
    var source_len: usize = 0;

    var kernel_args: []Args = try allocator.alloc(Args, ssa.assign_num);
    errdefer allocator.free(kernel_args);
    defer allocator.free(kernel_args);

    const kernel_name_len_max = (kernel_base_name.len - "{}"[0..].len) +
        comptime std.math.log10_int(@as(u64, std.math.maxInt(@TypeOf(ssa.assign_num))));
    var kernel_name: [kernel_name_len_max]u8 = @splat(0);

    for (0..ssa.assign_num) |assign_idx| {

        // This should be enough work to justify storing it in memory
        // $TODO Rethink this when I refactor the args gathering
        kernel_args[assign_idx] = try Args.alloc(allocator, ssa.assign[assign_idx]);

        @memset(&kernel_name, 0);
        const kernel_name_written: []const u8 = try std.fmt.bufPrint(&kernel_name, //
            kernel_base_name, .{assign_idx});
        const kernel_name_len: usize = kernel_name_written.len;

        try runtime.assignCompile(allocator, &source, &source_len, ssa.assign[assign_idx], //
            kernel_name[0..kernel_name_len], kernel_args[assign_idx], size_global, size_local);
    }

    const program_ptr: ProgramPtr = try runtime.programAlloc(source);
    var kernel: []Kernel = try allocator.alloc(Kernel, ssa.assign_num);
    errdefer allocator.free(kernel);

    for (0..ssa.assign_num) |kernel_idx| {
        @memset(&kernel_name, 0);
        const kernel_name_len: usize = (try std.fmt.bufPrint(&kernel_name, //
            kernel_base_name ++ "\x00", .{kernel_idx})).len;
        kernel[kernel_idx].ptr = try runtime.kernelAlloc(program_ptr, //
            kernel_name[0..kernel_name_len], kernel_args[kernel_idx]);
        kernel[kernel_idx].args = kernel_args[kernel_idx];
    }

    return .{
        .kernel = kernel,
        .size_global = size_global,
        .size_local = size_local,
        .ptr = program_ptr,
    };
}
pub fn free(this: *@This(), runtime: Runtime, allocator: Allocator) void {
    for (this.kernel) |*kernel| {
        runtime.kernelFree(kernel.ptr);
        kernel.args.free(allocator);
    }
    allocator.free(this.kernel);
}
pub fn run(this: @This(), runtime: Runtime) !void {
    try runtime.programRun(this);
}
