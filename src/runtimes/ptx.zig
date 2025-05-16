const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");

// const OpenCl_version = @import("OpenCl_config").OpenCl_version;

// It is just this followed by the index of the function
pub const function_name_base: []const u8 = &[_]u8{'k'};

// $NOTE PTX is only accessible through the CUDA API from what I've seen
const cuda_header = "cuda.h";
pub const cuda = @cImport({
    @cInclude(cuda_header);
});

pub const CuError = error{
    PlatformNotFound,
    PlatformNotFreed,
    DeviceNotFound,
    DeviceNotFreed,
    DeviceInfoNotFound,
    ContextNotFound,
    ContextNotFreed,
    QueueNotAlloc,
    QueueNotFreed,
    QueueCouldNotWait,
    MemNotAlloc,
    MemNotFreed,
    ProgramNotCreated,
    ProgramNotBuilt,
    ProgramNotFreed,
    ProgramNotRun,
    FunctionNotBuilt,
    FunctionNotFreed,
    ArgNotSet,
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
    program: cuda.CUmodule,
    pub fn alloc(allocator: Allocator, context: CuContext, device: CuDevice, source: []const u8) !CuModule {
        _ = allocator;
        _ = context;
        _ = device;
        _ = source;
    }
    pub fn free(this: @This()) !void {
        if (cuda.cuModuleUnload(this) != cuda.CUDA_SUCCESS) {
            @branchHint(.cold);
            return CuError.ProgramNotFreed;
        }
    }
};
pub const CuFunction = struct {
    function: cuda.CUfunction,
    pub fn alloc(program: CuModule, name_c: [*:0]const u8) !CuFunction {
        _ = program;
        _ = name_c;
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
