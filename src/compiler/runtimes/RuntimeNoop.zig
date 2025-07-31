//! Runtime whose functions just do nothing. Return values are `undefined`.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");

const codegen_cl = @import("codegen_cl.zig");
const Runtime = @import("Runtime.zig");
const Program = @import("../Program.zig");
const Kernel = Program.Kernel;
const KernelPtr = Program.KernelPtr;
const ProgramPtr = Program.ProgramPtr;
const Memory = Program.Memory;
const Args = Program.Args;
const Sync = Program.Sync;
const Pir = @import("../Pir.zig");
const Assign = Pir.Assign;

// const opencl_version = @import("opencl_config").opencl_version;

pub const RuntimeNop = @This();

pub fn runtime(_: *@This()) Runtime {
    return Runtime{
        .state = undefined,
        .vtable = .{
            .init = init,
            .deinit = deinit,
            .memoryAlloc = memoryAlloc,
            .memoryFree = memoryFree,
            .memorySyncToDevice = memorySyncToDevice,
            .memorySyncToHost = memorySyncToHost,
            .programAlloc = programAlloc,
            .programFree = programFree,
            .kernelAlloc = kernelAlloc,
            .kernelFree = kernelFree,
            .kernelRun = kernelRun,
            .queueWait = queueWait,
            .assignCompileBytes = assignCompileBytes,
            .assignCompile = assignCompile,
        },
    };
}

pub fn init(_: *anyopaque) ?void {}
pub fn deinit(_: *anyopaque) ?void {}
pub fn memoryAlloc(_: *anyopaque, _: u32, _: u32, _: u32, _: u32) ?Memory {
    return undefined;
}
pub fn memoryFree(_: *anyopaque, _: Memory) ?void {}
pub fn memorySyncToHost(_: *anyopaque, _: Memory, _: *anyopaque, _: u32) ?void {}
pub fn memorySyncToDevice(_: *anyopaque, _: Memory, _: *anyopaque, _: u32) ?void {}
pub fn programAlloc(_: *anyopaque, _: []const u8) ?ProgramPtr {
    return undefined;
}
pub fn programFree(_: *anyopaque, _: ProgramPtr) ?void {}
pub fn kernelAlloc(_: *anyopaque, _: ProgramPtr, _: [*:0]const u8, _: Args) ?KernelPtr {
    return undefined;
}
pub fn kernelFree(_: *anyopaque, _: KernelPtr) ?void {}
pub fn kernelRun(_: *anyopaque, _: KernelPtr, _: Args, _: usize, _: usize) ?void {}
pub fn queueWait(_: *anyopaque) ?void {}
pub fn assignCompileBytes(_: *anyopaque, _: Assign, _: u32, _: Args, _: u32, _: u32) u32 {
    return undefined;
}
pub fn assignCompile(
    _: *anyopaque,
    _: *[]u8,
    _: *usize,
    _: Assign,
    _: []const u8,
    _: Args,
    _: u32,
    _: u32,
) void {}
