//! Runtime based on NVidia PTX.
//! Created because the NVidida OpenCl compiler is dog slow and I want to see if this is faster

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");

const codegen_ptx = @import("codegen_ptx.zig");
const Runtime = @import("Runtime.zig");
const Program = @import("../Program.zig");
const Kernel = Program.Kernel;
const KernelPtr = Program.KernelPtr;
const ProgramPtr = Program.ProgramPtr;
const Memory = Program.Memory;
const Args = Program.Args;
const Sync = Program.Sync;

// PTX is only accessible through the CUDA API from what I've seen
// Man I really want to replace the entire NVIDIA stack or at least every compiler in it
// Compiling OpenCL is unbearably slow, don't know about PTX yet.
// Just look at the naming. of the types and function. They seem so horribly inconsistent
const cuda_header = "cuda.h";
pub const cuda = @cImport({
    @cInclude(cuda_header);
});

const CuDevice = cuda.CUdevice;
const CuContext = cuda.CUcontext;

device: CuDevice,
context: CuContext,
registers_max: u32,
// $MAYBE make a non-default stream?

pub const RuntimePtx = @This();

pub fn runtime(this: *@This()) Runtime {
    return .{
        .state = this,
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
            .assignCompileBytes = codegen_ptx.assignCompileBytes,
            .assignCompile = codegen_ptx.assignCompile,
        },
    };
}

pub fn init(this: *anyopaque) ?void {
    var state: *RuntimePtx = @alignCast(@ptrCast(this));

    if (cuda.cuDeviceGet(&state.device, 0) != cuda.CUDA_SUCCESS) {
        @branchHint(.cold);
        return null;
    }
    if (cuda.cuCtxCreate(&state.context, 0, state.device) != cuda.CUDA_SUCCESS) {
        @branchHint(.cold);
        _ = cuda.cudaFree(state.device);
        return null;
    }
    if (cuda.cuDeviceGetAttribute(state.registers_max, cuda.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, //
        state.device) != cuda.CUDA_SUCCESS)
    {
        @branchHint(.cold);
        _ = cuda.cudaFree(state.device);
        _ = cuda.cuCtxDestrooy(state.context);
        return null;
    }
}
pub fn deinit(this: *anyopaque) ?void {
    const state: *RuntimePtx = @alignCast(@ptrCast(this));

    var failed: bool = false;
    if (cuda.cudaFree(state.device) != cuda.CUDA_SUCCESS) {
        @branchHint(.cold);
        failed = true;
    }
    if (cuda.cuCtxDestrooy(state.context) != cuda.CUDA_SUCCESS) {
        @branchHint(.cold);
        failed = true;
    }
    if (failed) {
        @branchHint(.cold);
        return null;
    }
}
pub fn memoryAlloc(_: *anyopaque, a: u32, z: u32, y: u32, x: u32) ?Memory {
    assert(a > 0);
    assert(z > 0);
    assert(y > 0);
    assert(x > 0);
    var memory: cuda.CUdeviceptr = null;
    if (cuda.cuMemAlloc(&memory, a * z * y * x) == cuda.CUDA_SUCCESS) {
        @branchHint(.likely);
        return @ptrCast(memory);
    } else {
        @branchHint(.cold);
        return null;
    }
}
pub fn memoryFree(_: *anyopaque, memory: Memory) ?void {
    if (cuda.cudaFree(memory) != cuda.CUDA_SUCCESS) {
        @branchHint(.cold);
        return null;
    }
}
pub fn memorySyncToDevice(_: *anyopaque, mem: Memory, mem_host: *anyopaque, n_bytes: u32) ?void {
    if (cuda.cudaMemcpyAsync(@ptrCast(mem), @ptrCast(mem_host), n_bytes, //
        cuda.cudaMemcpyHostToDevice, null) != cuda.CUDA_SUCCESS)
    {
        @branchHint(.cold);
        return null;
    }
}
pub fn memorySyncToHost(_: *anyopaque, mem: Memory, mem_host: *anyopaque, n_bytes: u32) ?void {
    if (cuda.cudaMemcpyAsync(@ptrCast(mem), @ptrCast(mem_host), n_bytes, //
        cuda.cudaMemcpyHostToHost, null) != cuda.CUDA_SUCCESS)
    {
        @branchHint(.cold);
        return null;
    }
}
pub fn programAlloc(_: *anyopaque, source: []const u8) ?ProgramPtr {
    var module: cuda.CUmodule = null;
    if (cuda.cuModuleLoadData(@ptrCast(&module), @ptrCast(source)) != cuda.CUDA_SUCCESS) {
        @branchHint(.cold);
        return null;
    }
    return @ptrCast(module);
}
pub fn programFree(_: *anyopaque, program: ProgramPtr) ?void {
    if (cuda.moduleUnload(@ptrCast(program)) != cuda.CUDA_SUCCESS) {
        @branchHint(.cold);
        return null;
    }
}
pub fn kernelAlloc(_: *anyopaque, program: ProgramPtr, name: [*:0]const u8, _: Args) ?KernelPtr {
    const function: cuda.CUfunction = null;
    if (cuda.cuModuleGetFunction(&function.function, @ptrCast(program), @ptrCast(name)) != //
        cuda.CUDA_SUCCESS)
    {
        @branchHint(.cold);
        return null;
    }
}
pub fn kernelFree(_: *anyopaque, kernel: KernelPtr) ?void {
    if (cuda.cudaFree(@ptrCast(kernel)) != cuda.CUDA_SUCCESS) {
        @branchHint(.cold);
        return null;
    }
}
pub fn kernelRun(_: *anyopaque, kernel: KernelPtr, args: Args, size_global: u32, size_local: u32) ?void {
    const z_dim_grid: u32 = 1;
    const y_dim_grid: u32 = 1;
    const z_dim_block: u32 = 1;
    const y_dim_block: u32 = 1;
    if (cuda.cuLaunchKernel(@ptrCast(kernel), size_global, y_dim_grid, z_dim_grid, //
        size_local, y_dim_block, z_dim_block, 0, 0, @ptrCast(args.arg_mem), null) != cuda.CUDA_SUCCESS)
    {
        @branchHint(.cold);
        return null;
    }
}
pub fn queueWait(_: *anyopaque) ?void {
    if (cuda.cuCtxSynchronize() != cuda.CUDA_SUCCESS) {
        @branchHint(.cold);
        return null;
    }
}
