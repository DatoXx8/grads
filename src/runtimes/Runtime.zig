const std = @import("std");
const Allocator = std.mem.Allocator;

const Ssa = @import("../compiler/Ssa.zig");
const Assign = Ssa.Assign;

// Inspired by the stdlib std.mem.Allocator interface
pub const Runtime = @This();

pub const Error = error{
    ContextInit,
    ContextFree,
    MemoryAlloc,
    MemoryFree,
    ProgramAlloc,
    ProgramFree,
    ProgramRun,
    KernelAlloc,
    KernelFree,
};

pub const Memory = *anyopaque;
pub const Args = struct {
    arg_id: []u64, // $TODO arg_id is probably unnecessary
    arg_mem: []Memory,
    pub fn alloc(allocator: Allocator, assign: Assign) !Args {
        var arg_unique = std.AutoHashMap(u64, Memory).init(allocator);
        errdefer arg_unique.deinit();
        defer arg_unique.deinit();

        try arg_unique.put(assign.base.out.id, assign.base.out.values_cl.?);
        if (!assign.base.type.isUnary()) {
            try arg_unique.put(assign.base.in.id, assign.base.in.values_cl.?);
        }

        if (assign.inlined) |inlined| {
            // $TODO If the thing is inlined then don't there is no need to pass it to the kernel
            for (0..inlined.inlined_num) |inlined_idx| {
                try arg_unique.put(inlined.base[inlined_idx].out.id, inlined.base[inlined_idx].out.values_cl.?);
                if (!inlined.base[inlined_idx].type.isUnary()) {
                    try arg_unique.put(inlined.base[inlined_idx].in.id, inlined.base[inlined_idx].in.values_cl.?);
                }
            }
        }

        const arg_num: usize = arg_unique.count();
        const arg_id: []u64 = try allocator.alloc(u64, arg_num);
        const arg_mem: []ClMem = try allocator.alloc(ClMem, arg_num);

        errdefer {
            allocator.free(arg_id);
            allocator.free(arg_mem);
        }

        var arg_id_iterator = arg_unique.keyIterator();
        for (0..arg_num) |arg_idx| {
            const key: u64 = arg_id_iterator.next().?.*;
            arg_id[arg_idx] = key;
            arg_mem[arg_idx] = arg_unique.get(key).?;
        }

        return .{
            .arg_id = arg_id,
            .arg_mem = arg_mem,
        };
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
        // The arg_mem get's freed with the tensors
        allocator.free(this.arg_id);
        allocator.free(this.arg_mem);
    }
};
pub const Kernel = struct {
    args: Args,
    ptr: *anyopaque,
};
pub const Program = struct {
    size_global: u32,
    size_local: u32,
    kernel: []const Kernel,
    // $TODO Questionable wether this is needed beyond debuggging
    source: []const u8,
    ptr: *anyopaque,
};

state: *anyopaque,
vtable: VTable,

/// $WARN A return value of null always means failure here.
pub const VTable = struct {
    /// Init all the relevant context
    init: *const fn (*anyopaque) ?void,
    /// Deinint all the relevant context
    deinit: *const fn (*anyopaque) ?void,
    memoryAlloc: *const fn (*anyopaque, a: u32, z: u32, y: u32, x: u32) ?Memory,
    memoryFree: *const fn (*anyopaque, mem: Memory) ?void,
    programAlloc: *const fn (*anyopaque, source: []const u8) ?Program,
    programFree: *const fn (*anyopaque, program: Program) ?void,
    programRun: *const fn (*anyopaque, program: Program) ?void,
    kernelAlloc: *const fn (*anyopaque, program: Program, name: []const u8) ?Kernel,
    kernelFree: *const fn (*anyopaque, kernel: Kernel) ?void,
};

pub fn init(runtime: *Runtime) !void {
    if (runtime.vtable.init(runtime.state) == null) {
        @branchHint(.cold);
        return Error.ContextInit;
    }
}
pub fn deinit(runtime: *Runtime) !void {
    if (runtime.vtable.deinit(runtime.state) == null) {
        @branchHint(.cold);
        return Error.ContextInit;
    }
}
pub fn memoryAlloc(runtime: *Runtime, a: u32, z: u32, y: u32, x: u32) !Memory {
    if (runtime.vtable.memoryAlloc(runtime.state, a, z, y, x)) |memory| {
        return memory;
    } else {
        return Error.MemoryAlloc;
    }
}
pub fn memoryFree(runtime: *Runtime, memory: Memory) !void {
    if (runtime.vtable.memoryFree(runtime.state, memory) == null) {
        return Error.MemoryFree;
    }
}
pub fn programAlloc(runtime: *Runtime, source: []const u8) !Program {
    //
}
