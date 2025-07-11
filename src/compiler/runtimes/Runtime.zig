const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Program = @import("../Program.zig");
const Memory = Program.Memory;
const Sync = Program.Sync;
const Kernel = Program.Kernel;
const ProgramPtr = Program.ProgramPtr;
const KernelPtr = Program.KernelPtr;
const Args = Program.Args;
const Ssa = @import("../Ssa.zig");
const Assign = Ssa.Assign;

pub const RuntimeCl = @import("./RuntimeCl.zig");
pub const RuntimePtx = @import("./RuntimePtx.zig");
pub const RuntimeNoop = @import("./RuntimeNoop.zig");

// $TODO Add a single and multithread CPU runtime for x86_64 avx2 & bmi2 and no extension

pub const Runtime = @This();
pub const kernel_name_base: []const u8 = &[_]u8{'k'};

pub const Error = error{
    ContextInit,
    ContextDeinit,
    MemoryAlloc,
    MemoryFree,
    MemorySync,
    ProgramAlloc,
    ProgramFree,
    ProgramRun,
    KernelAlloc,
    KernelFree,
    QueueWait,
    AssignCompile,
};

state: *anyopaque,
vtable: VTable,

/// A return value of null always means failure here.
pub const VTable = struct {
    /// Init all the relevant context
    init: *const fn (state: *anyopaque) ?void,
    /// Deinint all the relevant context
    deinit: *const fn (state: *anyopaque) ?void,
    memoryAlloc: *const fn (state: *anyopaque, a: u32, z: u32, y: u32, x: u32) ?Memory,
    memoryFree: *const fn (state: *anyopaque, mem: Memory) ?void,
    memorySyncToHost: *const fn (state: *anyopaque, mem: Memory, mem_host: *anyopaque, n_bytes: u32) ?void,
    memorySyncToDevice: *const fn (state: *anyopaque, mem: Memory, mem_host: *anyopaque, n_bytes: u32) ?void,
    programAlloc: *const fn (state: *anyopaque, source: []const u8) ?ProgramPtr,
    programFree: *const fn (state: *anyopaque, program: ProgramPtr) ?void,
    kernelAlloc: *const fn (state: *anyopaque, program: ProgramPtr, name: [*:0]const u8, args: Args) ?KernelPtr,
    kernelFree: *const fn (state: *anyopaque, kernel: KernelPtr) ?void,
    kernelRun: *const fn (
        state: *anyopaque,
        kernel: KernelPtr,
        args: Args,
        size_global: usize,
        size_local: usize,
    ) ?void,
    queueWait: *const fn (state: *anyopaque) ?void,
    assignCompile: *const fn (
        state: *anyopaque,
        allocator: Allocator,
        source: *[]u8,
        offset: *usize,
        assign: Assign,
        name: []const u8,
        args: Args,
        size_global: u32,
        size_local: u32,
    ) ?void,
};

pub fn init(runtime: Runtime) !void {
    if (runtime.vtable.init(runtime.state) == null) {
        @branchHint(.cold);
        return Error.ContextInit;
    }
}
pub fn deinit(runtime: Runtime) void {
    if (runtime.vtable.deinit(runtime.state) == null) {
        @branchHint(.cold);
        std.log.err("Could not free runtime context\n", .{});
    }
}
pub fn memoryAlloc(runtime: Runtime, a: u32, z: u32, y: u32, x: u32) !Memory {
    if (runtime.vtable.memoryAlloc(runtime.state, a, z, y, x)) |memory| {
        @branchHint(.likely);
        return memory;
    } else {
        @branchHint(.cold);
        return Error.MemoryAlloc;
    }
}
pub fn memoryFree(runtime: Runtime, memory: Memory) void {
    if (runtime.vtable.memoryFree(runtime.state, memory) == null) {
        @branchHint(.cold);
        std.log.err("Could not free runtime Memory\n", .{});
    }
}
pub fn memorySyncToHost(runtime: Runtime, memory: Memory, memory_host: *anyopaque, n_bytes: u32) !void {
    if (runtime.vtable.memorySyncToHost(runtime.state, memory, memory_host, n_bytes) == null) {
        @branchHint(.cold);
        return Error.MemorySync;
    }
}
pub fn memorySyncToDevice(runtime: Runtime, memory: Memory, memory_host: *anyopaque, n_bytes: u32) !void {
    if (runtime.vtable.memorySyncToDevice(runtime.state, memory, memory_host, n_bytes) == null) {
        @branchHint(.cold);
        return Error.MemorySync;
    }
}
pub fn programAlloc(runtime: Runtime, source: []const u8) !ProgramPtr {
    if (runtime.vtable.programAlloc(runtime.state, source)) |program_ptr| {
        @branchHint(.likely);
        return program_ptr;
    } else {
        @branchHint(.cold);
        return Error.ProgramAlloc;
    }
}
pub fn programFree(runtime: Runtime, program: ProgramPtr) void {
    if (runtime.vtable.programFree(runtime.state, program) == null) {
        @branchHint(.cold);
        std.log.err("Could not free runtime Program\n", .{});
    }
}
// Kind of stupid that this is basically the only non ProgramPtr in here
pub fn programRun(runtime: Runtime, program: Program) !void {
    for (program.kernel) |kernel| {
        if (runtime.vtable.kernelRun(runtime.state, kernel.ptr, kernel.args, program.size_global, //
            program.size_local) == null)
        {
            @branchHint(.cold);
            return Error.ProgramRun;
        }
    }
    if (runtime.vtable.queueWait(runtime.state) == null) {
        @branchHint(.cold);
        return Error.ProgramRun;
    }
}
pub fn kernelAlloc(runtime: Runtime, program: ProgramPtr, name: []const u8, args: Args) !KernelPtr {
    assert(name[name.len - 1] == '\x00');
    if (runtime.vtable.kernelAlloc(runtime.state, program, @ptrCast(name), args)) |kernel_ptr| {
        @branchHint(.likely);
        return kernel_ptr;
    } else {
        @branchHint(.cold);
        return Error.KernelAlloc;
    }
}
pub fn kernelFree(runtime: Runtime, kernel: KernelPtr) void {
    if (runtime.vtable.kernelFree(runtime.state, kernel) == null) {
        @branchHint(.cold);
        std.log.err("Could not free runtime Kernel\n", .{});
    }
}
pub fn queueWait(runtime: Runtime) !void {
    if (runtime.vtable.queueWait(runtime.state) == null) {
        @branchHint(.cold);
        return Error.QueueWait;
    }
}
pub fn assignCompile(
    runtime: Runtime,
    allocator: Allocator,
    source: *[]u8,
    offset: *usize,
    assign: Assign,
    name: []const u8,
    args: Args,
    size_global: u32,
    size_local: u32,
) !void {
    if (runtime.vtable.assignCompile(runtime.state, allocator, source, offset, assign, name, args, //
        size_global, size_local) == null)
    {
        @branchHint(.cold);
        return Error.AssignCompile;
    }
}
