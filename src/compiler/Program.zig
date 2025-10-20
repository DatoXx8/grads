const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const Linearized = @import("../Linearized.zig");
const opt = @import("optimize.zig");
const Optimization = opt.Optimization;
const Runtime = @import("runtimes/Runtime.zig");
const Pir = @import("Pir.zig");
const Assign = Pir.Assign;

pub const Memory = *anyopaque;
pub const Args = struct {
    /// arg_num is separate to make kernels with no arguments easier to handle
    arg_num: u64,
    arg_id: []u64,
    arg_mem: []Memory,
    pub fn alloc(allocator: Allocator, assign: Assign) !Args {
        var arg_unique = std.AutoHashMap(u64, Memory).init(allocator);
        defer arg_unique.deinit();

        const in_root_is_null: bool = if (assign.inlined) |inlined| inlined.in_root == null else true;

        try arg_unique.put(assign.base.out.id, assign.base.out.values_runtime);
        if (!assign.base.kind.isUnary() and in_root_is_null) {
            try arg_unique.put(assign.base.in.id, assign.base.in.values_runtime);
        }

        if (assign.inlined) |inlined| {
            for (0..inlined.inlined_num) |inlined_idx| {
                const out_is_not_inlined: bool = inlined.out[inlined_idx] == null;
                const in_is_not_inlined: bool = inlined.in[inlined_idx] == null;

                if (out_is_not_inlined) {
                    try arg_unique.put(inlined.base[inlined_idx].out.id, //
                        inlined.base[inlined_idx].out.values_runtime);
                }
                if (!inlined.base[inlined_idx].kind.isUnary() and in_is_not_inlined) {
                    try arg_unique.put(inlined.base[inlined_idx].in.id, //
                        inlined.base[inlined_idx].in.values_runtime);
                }
            }
        }

        const arg_num: usize = arg_unique.count();
        const arg_id: []u64 = try allocator.alloc(u64, arg_num);
        errdefer allocator.free(arg_id);
        const arg_mem: []Memory = try allocator.alloc(Memory, arg_num);
        var arg_iterator = arg_unique.iterator();
        for (0..arg_num) |arg_idx| {
            const entry = arg_iterator.next().?;
            arg_id[arg_idx] = entry.key_ptr.*;
            arg_mem[arg_idx] = entry.value_ptr.*;
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
            .arg_id = try allocator.alloc(u64, 1),
            .arg_mem = try allocator.alloc(Memory, 1),
        };
    }
    pub fn free(args: Args, allocator: Allocator) void {
        allocator.free(args.arg_id);
        // The runtime specific arg_mem get's freed with the tensors
        allocator.free(args.arg_mem);
    }
};
// $TODO Support integer arguments
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

/// This is enough for u64 integers.
/// If you try to print larger integers in the codegen then this will break the capacity calculations.
pub const kernel_base_name = "kern{}";
// $TODO I am not happy about having to pass in the gpa here for the source alone. Fix it
pub fn alloc(
    runtime: Runtime,
    gpa: Allocator,
    arena: Allocator,
    arena_temp: Allocator,
    linearized: Linearized,
    depth_max: u32,
    size_global: u32,
    size_local: u32,
) !Program {
    assert(size_global >= size_local);
    assert(size_global % size_local == 0);

    if (linearized.op_num == 0) {
        @branchHint(.unlikely);
        const kernel_empty_name = "empty";
        const source: []const u8 = "kernel void " ++ kernel_empty_name ++ "() {}\n\x00";

        const program_ptr: ProgramPtr = try runtime.programAlloc(source);
        errdefer runtime.programFree(program_ptr);
        var kernel: []Kernel = try arena.alloc(Kernel, 1);
        kernel[0].args = try Args.allocEmpty(arena);
        kernel[0].ptr = try runtime.kernelAlloc(program_ptr, //
            kernel_empty_name ++ "\x00", kernel[0].args);

        return .{
            .kernel = kernel,
            .size_global = size_global,
            .size_local = size_local,
            .ptr = program_ptr,
        };
    } else {
        @branchHint(.likely);
        const pir: Pir = try Pir.alloc(arena_temp, linearized, depth_max, size_global, size_local);

        const kernel_name_len_max = (kernel_base_name.len - "{}"[0..].len) +
            comptime std.math.log10_int(@as(u64, std.math.maxInt(@TypeOf(pir.assign_num))));
        var kernel_name: [kernel_name_len_max]u8 = @splat(0);

        var kernel_num_written: u32 = 0; // Just needed for getting rid of memory leak
        var kernel: []Kernel = try arena.alloc(Kernel, pir.assign_num);

        const source_len_init: u32 = 16 * 1024; // Arbitrary value
        var source: []u8 = try gpa.alloc(u8, source_len_init);
        defer gpa.free(source);
        @memset(source, 0);
        var source_idx: usize = 0;

        var assign_idx: u32 = 0;
        while (assign_idx < pir.assign_num) : (assign_idx += 1) {
            @memset(&kernel_name, 0);
            const kernel_name_written: []const u8 = try std.fmt.bufPrint(&kernel_name, //
                kernel_base_name, .{assign_idx});

            kernel[assign_idx].args = try Args.alloc(arena, pir.assign[assign_idx]);
            kernel_num_written += 1;
            try runtime.assignCompile(gpa, &source, &source_idx, pir.assign[assign_idx], //
                kernel_name_written, kernel[assign_idx].args, size_global, size_local);
        }

        const program_ptr: ProgramPtr = try runtime.programAlloc(source);
        errdefer runtime.programFree(program_ptr);

        for (0..pir.assign_num) |kernel_idx| {
            @memset(&kernel_name, 0);
            const kernel_name_len: usize = (try std.fmt.bufPrint(&kernel_name, //
                kernel_base_name ++ "\x00", .{kernel_idx})).len;
            kernel[kernel_idx].ptr = try runtime.kernelAlloc(program_ptr, //
                kernel_name[0..kernel_name_len], kernel[kernel_idx].args);
        }

        return .{
            .kernel = kernel,
            .size_global = size_global,
            .size_local = size_local,
            .ptr = program_ptr,
        };
    }
}
pub fn free(program: Program, runtime: Runtime) void {
    for (program.kernel) |kernel| {
        runtime.kernelFree(kernel.ptr);
    }
}
pub fn run(program: Program, runtime: Runtime) !void {
    try runtime.programRun(program);
}
