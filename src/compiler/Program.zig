const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

const Linearized = @import("../Linearized.zig");
const opt = @import("optimize.zig");
const Optimization = opt.Optimization;
const Runtime = @import("runtimes/Runtime.zig");
const Pir = @import("Pir.zig");
const Assign = Pir.Assign;
const util = @import("../util.zig");

pub const Memory = *anyopaque;
pub const Args = struct {
    /// arg_num is separate to make kernels with no arguments easier to handle
    arg_num: u64,
    arg_id: []u64,
    arg_mem: []Memory,
    pub fn alloc(gpa: Allocator, arena: Allocator, assign: Assign) !Args {
        var arg_unique = std.AutoHashMap(u64, Memory).init(gpa);
        defer arg_unique.deinit();

        const in_root_is_null: bool = assign.inlined.in_root == null;

        try arg_unique.put(assign.base.out.id, assign.base.out.values_runtime);
        if (!assign.base.kind.isUnary() and in_root_is_null) {
            try arg_unique.put(assign.base.in.id, assign.base.in.values_runtime);
        }

        for (0..assign.inlined.num) |inlined_idx| {
            const out_is_not_inlined: bool = assign.inlined.out[inlined_idx] == null;
            const in_is_not_inlined: bool = assign.inlined.in[inlined_idx] == null;

            if (out_is_not_inlined) {
                try arg_unique.put(assign.inlined.base[inlined_idx].out.id, //
                    assign.inlined.base[inlined_idx].out.values_runtime);
            }
            if (!assign.inlined.base[inlined_idx].kind.isUnary() and in_is_not_inlined) {
                try arg_unique.put(assign.inlined.base[inlined_idx].in.id, //
                    assign.inlined.base[inlined_idx].in.values_runtime);
            }
        }

        const arg_num: usize = arg_unique.count();
        const arg_id: []u64 = try arena.alloc(u64, arg_num);
        const arg_mem: []Memory = try arena.alloc(Memory, arg_num);
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

    if (linearized.num == 0) {
        @branchHint(.unlikely);
        const kernel_empty_name = "empty";
        const source: []const u8 = "kernel void " ++ kernel_empty_name ++ "() {}\n\x00";

        const program_ptr: ProgramPtr = try runtime.programAlloc(source);
        errdefer runtime.programFree(program_ptr);
        var kernel: []Kernel = try arena.alloc(Kernel, 1);
        kernel[0].args = .{
            .arg_num = 0,
            .arg_id = &.{},
            .arg_mem = &.{},
        };
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
        pir.print(4, 0, null);

        const kernel_name_len_max = (kernel_base_name.len - "{}"[0..].len) +
            comptime std.math.log10_int(@as(u64, std.math.maxInt(@TypeOf(pir.assign_num))));
        var kernel_name: [kernel_name_len_max]u8 = @splat(0);

        var kernel_num_written: u32 = 0; // Just needed for getting rid of memory leak
        var kernel: []Kernel = try arena.alloc(Kernel, pir.assign_num);

        const source_len_init: u32 = 16 * 1024; // Arbitrary value
        var source: ArrayList(u8) = try .initCapacity(gpa, source_len_init);
        defer source.deinit(gpa);

        @memset(source.items, 0);

        var assign_idx: u32 = 0;
        while (assign_idx < pir.assign_num) : (assign_idx += 1) {
            @memset(&kernel_name, 0);
            const kernel_name_written: []const u8 = try std.fmt.bufPrint(&kernel_name, //
                kernel_base_name, .{assign_idx});

            kernel[assign_idx].args = try Args.alloc(gpa, arena, pir.assign[assign_idx]);
            kernel_num_written += 1;
            try runtime.assignCompile(gpa, &source, pir.assign[assign_idx], //
                kernel_name_written, kernel[assign_idx].args, size_global, size_local);
        }
        try source.printBounded("\x00", .{});

        const program_ptr: ProgramPtr = try runtime.programAlloc(source.items[0..]);
        errdefer runtime.programFree(program_ptr);

        var kernel_idx: u32 = 0;
        while (kernel_idx < pir.assign_num) : (kernel_idx += 1) {
            @memset(&kernel_name, 0);
            const kernel_name_len: usize = (try std.fmt.bufPrint(&kernel_name, //
                kernel_base_name ++ "\x00", .{kernel_idx})).len;
            kernel[kernel_idx].ptr = try runtime.kernelAlloc(program_ptr, //
                kernel_name[0..kernel_name_len], kernel[kernel_idx].args);
        }

        return .{
            .kernel = kernel,
            .ptr = program_ptr,
            .size_global = size_global,
            .size_local = size_local,
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
