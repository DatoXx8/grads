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
const Pir = @import("../Pir.zig");
const Assign = Pir.Assign;

pub const RuntimeCl = @import("RuntimeCl.zig");
pub const RuntimePtx = @import("RuntimePtx.zig");
pub const RuntimeNoop = @import("RuntimeNoop.zig");

// This one is gonna take a while
// $TODO Add a single and multithread CPU runtime for x86_64 avx2 & bmi2 and no extension

pub const Runtime = @This();

pub const Error = error{
    ContextInit,
    MemoryAlloc,
    MemorySync,
    KernelRun,
    ProgramAlloc,
    KernelAlloc,
    QueueWait,
};

state: *anyopaque,
vtable: VTable,

/// A return value of null always means failure here.
pub const VTable = struct {
    /// Init all the relevant context
    init: *const fn (state: *anyopaque) Error!void,
    /// Deinint all the relevant context
    deinit: *const fn (state: *anyopaque) void,
    memoryAlloc: *const fn (state: *anyopaque, a: u32, z: u32, y: u32, x: u32) Error!Memory,
    memoryFree: *const fn (state: *anyopaque, mem: Memory) void,
    memorySyncToHost: *const fn (state: *anyopaque, mem: Memory, mem_host: *anyopaque, n_bytes: u32) Error!void,
    memorySyncToDevice: *const fn (state: *anyopaque, mem: Memory, mem_host: *anyopaque, n_bytes: u32) Error!void,
    programAlloc: *const fn (state: *anyopaque, source: []const u8) Error!ProgramPtr,
    programFree: *const fn (state: *anyopaque, program: ProgramPtr) void,
    kernelAlloc: *const fn (state: *anyopaque, program: ProgramPtr, name: [*:0]const u8, args: Args) Error!KernelPtr,
    kernelFree: *const fn (state: *anyopaque, kernel: KernelPtr) void,
    kernelRun: *const fn (
        state: *anyopaque,
        kernel: KernelPtr,
        args: Args,
        size_global: usize,
        size_local: usize,
    ) Error!void,
    queueWait: *const fn (state: *anyopaque) Error!void,
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
    ) Allocator.Error!void,
};

pub fn init(runtime: Runtime) !void {
    try runtime.vtable.init(runtime.state);
}
pub fn deinit(runtime: Runtime) void {
    runtime.vtable.deinit(runtime.state);
}
pub fn memoryAlloc(runtime: Runtime, a: u32, z: u32, y: u32, x: u32) !Memory {
    return try runtime.vtable.memoryAlloc(runtime.state, a, z, y, x);
}
pub fn memoryFree(runtime: Runtime, memory: Memory) void {
    runtime.vtable.memoryFree(runtime.state, memory);
}
pub fn memorySyncToHost(runtime: Runtime, memory: Memory, memory_host: *anyopaque, n_bytes: u32) !void {
    try runtime.vtable.memorySyncToHost(runtime.state, memory, memory_host, n_bytes);
}
pub fn memorySyncToDevice(runtime: Runtime, memory: Memory, memory_host: *anyopaque, n_bytes: u32) !void {
    try runtime.vtable.memorySyncToDevice(runtime.state, memory, memory_host, n_bytes);
}
pub fn programAlloc(runtime: Runtime, source: []const u8) !ProgramPtr {
    return try runtime.vtable.programAlloc(runtime.state, source);
}
pub fn programFree(runtime: Runtime, program: ProgramPtr) void {
    runtime.vtable.programFree(runtime.state, program);
}
// Kind of stupid that this is basically the only non ProgramPtr in here
pub fn programRun(runtime: Runtime, program: Program) !void {
    for (program.kernel) |kernel| {
        try runtime.vtable.kernelRun(runtime.state, kernel.ptr, kernel.args, program.size_global, //
            program.size_local);
    }
    try runtime.vtable.queueWait(runtime.state);
}
pub fn kernelAlloc(runtime: Runtime, program: ProgramPtr, name: []const u8, args: Args) !KernelPtr {
    assert(name[name.len - 1] == '\x00');
    return try runtime.vtable.kernelAlloc(runtime.state, program, @ptrCast(name), args);
}
pub fn kernelFree(runtime: Runtime, kernel: KernelPtr) void {
    runtime.vtable.kernelFree(runtime.state, kernel);
}
pub fn queueWait(runtime: Runtime) !void {
    try runtime.vtable.queueWait(runtime.state);
}
/// Crashes if source doesn't have enough space
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
    try runtime.vtable.assignCompile(runtime.state, allocator, source, offset, assign, name, args, //
        size_global, size_local);
}
