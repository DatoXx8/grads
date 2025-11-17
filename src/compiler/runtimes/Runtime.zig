const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const ArrayList = std.ArrayList;

const Buffer = @import("../../Buffer.zig");
const Vec4 = Buffer.Vec4;
const Id = Buffer.Id;
const Data = Buffer.Data;
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

pub const Pool = struct {
    pub const capacity_intial: u32 = 256;
    data_id_next: Id,
    data_id_free: Id,
    data: []Data,
    pub fn nextId(pool: *Pool, gpa: Allocator) !Id {
        if (pool.data_id_next == pool.data_id_free) {
            defer {
                pool.data_id_next += 1;
                pool.data_id_free += 1;
            }
            if (pool.data_id_next == pool.data.len) {
                pool.data = try gpa.realloc(pool.data, @max(16, pool.data.len * 2));
            }
            return pool.data_id_next;
        } else {
            const buffer: Buffer = .{ .id = pool.data_id_free };
            const buffer_data: Data = buffer.data().*;
            assert(buffer_data.kind == .free);
            if (pool.data_id_free == buffer_data.nextFree()) {
                pool.data_id_free = pool.data_id_next;
            } else {
                pool.data_id_free = buffer_data.nextFree();
            }
            return buffer.id;
        }
    }
    pub fn freeId(pool: *Pool, id: Id) void {
        const buffer: Buffer = .{ .id = id };
        const buffer_data: *Data = buffer.data();
        assert(buffer_data.*.kind != .free);
        // Don't really see a reason the do a special case when id is the last in the pool to decrement buffer_id_next
        if (pool.data_id_free == pool.data_id_next) {
            buffer_data.*.view.offset = id;
        } else {
            buffer_data.*.view.offset = pool.data_id_free;
        }
        buffer_data.*.kind = .free;
        pool.data_id_free = id;
    }
};

// $TODO Make this threadsafe
pub var pool_global: Pool = .{
    .data = &.{},
    .data_id_free = 0,
    .data_id_next = 0,
};

// This one is gonna take a while
// $TODO Add a multithread CPU runtime for x86_64 avx2

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
    memoryAlloc: *const fn (state: *anyopaque, size: Vec4) Error!Memory,
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
        gpa: Allocator,
        source: *ArrayList(u8),
        assign: Assign,
        name: []const u8,
        args: Args,
        size_global: u32,
        size_local: u32,
    ) Allocator.Error!void,
};

pub fn init(runtime: Runtime, gpa: Allocator) !void {
    pool_global.data = try gpa.alloc(Data, Pool.capacity_intial);
    try runtime.vtable.init(runtime.state);
}
pub fn deinit(runtime: Runtime, gpa: Allocator) void {
    gpa.free(pool_global.data);
    runtime.vtable.deinit(runtime.state);
}
pub fn memoryAlloc(runtime: Runtime, size: Vec4) !Memory {
    return try runtime.vtable.memoryAlloc(runtime.state, size);
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
    gpa: Allocator,
    source: *ArrayList(u8),
    assign: Assign,
    name: []const u8,
    args: Args,
    size_global: u32,
    size_local: u32,
) !void {
    try runtime.vtable.assignCompile(runtime.state, gpa, source, assign, name, args, //
        size_global, size_local);
}
