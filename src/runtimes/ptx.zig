const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");

// const OpenCl_version = @import("OpenCl_config").OpenCl_version;

// It is just this followed by the index of the function
pub const function_name_base: []const u8 = &[_]u8{'k'};

// $NOTE PTX is only accessible through the CUDA API from what I've seen
// Man I really want to replace the entire NVidia stack or at least every compiler in it. Compiling OpenCL is unbearably slow, don't know about PTX yet.
const cuda_header = "cuda.h";
pub const cuda = @cImport({
    @cInclude(cuda_header);
});

pub const CuError = error{
    DeviceNotFound,
    DeviceNotFreed,
    ContextNotFound,
    ContextNotFreed,
    MemNotAlloc,
    MemNotFreed,
    ModuleNotFreed,
    ModulePtxNotAdded,
    FunctionNotBuilt,
    FunctionNotFreed,
};
// $TODO Support multiple devices
pub const CuDevice = struct {
    device: cuda.CUdevice,
    pub fn alloc() !CuDevice {
        var device: cuda.CUdevice = null;
        if (cuda.cuDeviceGet(&device, 0) == cuda.CUDA_SUCCESS) {
            @branchHint(.likely);
            return .{ .device = device };
        } else {
            @branchHint(.cold);
            return CuError.DeviceNotFound;
        }
    }
    pub fn free(this: @This()) !void {
        if (cuda.cudaFree(this) != cuda.CUDA_SUCCESS) {
            @branchHint(.cold);
            return CuError.DeviceNotFreed;
        }
    }
};
pub const CuContext = struct {
    context: cuda.CUcontext,
    pub fn alloc(device: CuDevice) !CuContext {
        var context: cuda.CUcontext = null;
        if (cuda.cuCtxCreate(&context, 0, device.device) == cuda.CUDA_SUCCESS) {
            @branchHint(.likely);
            return .{ .context = context };
        } else {
            @branchHint(.cold);
            return CuError.ContextNotFound;
        }
    }
    pub fn free(this: @This()) !void {
        if (cuda.cudaFree(this) != cuda.CUDA_SUCCESS) {
            @branchHint(.cold);
            return CuError.ContextNotFreed;
        }
    }
};
pub const CuModule = struct {
    module: cuda.CUmodule,
    pub fn addPtx(this: *@This(), source_c: [*:0]const u8) !void {
        if (cuda.cuModuleLoadData(this, @ptrCast(source_c)) != cuda.CUDA_SUCCESS) {
            @branchHint(.cold);
            return CuError.ModulePtxNotAdded;
        }
    }
    pub fn free(this: *@This()) !void {
        if (cuda.moduleUnload(this) != cuda.CUDA_SUCCESS) {
            @branchHint(.cold);
            return CuError.ModuleNotFreed;
        }
    }
};
pub const CuFunction = struct {
    function: cuda.CUfunction,
    pub fn alloc(module: CuModule, source_c: [*:0]const u8, name_c: [*:0]const u8) !CuFunction {
        const function: CuFunction = undefined;
        try module.addPtx(source_c);
        if (cuda.cuModuleGetFunction(&function.function, module, @ptrCast(name_c)) != cuda.CUDA_SUCCESS) {
            @branchHint(.cold);
            return CuError.FunctionNotBuilt;
        }
    }
    pub fn free(this: @This()) !void {
        if (cuda.cudaFree(this) != cuda.CUDA_SUCCESS) {
            @branchHint(.cold);
            return CuError.FunctionNotFreed;
        }
    }
};
pub const CuMem = struct {
    memory: cuda.CUdeviceptr,
    pub fn alloc(a: u32, z: u32, y: u32, x: u32) !CuMem {
        assert(a > 0);
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);
        var memory: cuda.CUdeviceptr = null;
        if (cuda.cuMemAlloc(&memory, a * z * y * x) == cuda.CUDA_SUCCESS) {
            @branchHint(.likely);
            return .{ .memory = memory };
        } else {
            @branchHint(.cold);
            return CuError.MemNotAlloc;
        }
    }
    pub fn free(this: @This()) !void {
        if (cuda.cudaFree(this) != cuda.CUDA_SUCCESS) {
            @branchHint(.cold);
            return CuError.MemNotFreed;
        }
    }
};
