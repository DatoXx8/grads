//! Runtime based on OpenCl.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");

const codegen_cl = @import("codegen_cl.zig");
const Runtime = @import("Runtime.zig");
const Error = Runtime.Error;
const Program = @import("../Program.zig");
const Kernel = Program.Kernel;
const KernelPtr = Program.KernelPtr;
const ProgramPtr = Program.ProgramPtr;
const Memory = Program.Memory;
const Args = Program.Args;
const Sync = Program.Sync;

// const opencl_version = @import("opencl_config").opencl_version;

const opencl_header = switch (builtin.target.os.tag) {
    .macos => "OpenCL/cl.h",
    else => "CL/cl.h",
};

pub const opencl = @cImport({
    // cDefine("CL_TARGET_OpenCl_VERSION", opencl_version);
    @cInclude(opencl_header);
});

pub const RuntimeCl = @This();

const ClPlatform = opencl.cl_platform_id;
const ClDevice = opencl.cl_device_id;
const ClContext = opencl.cl_context;
const ClCommandQueue = opencl.cl_command_queue;

device: ClDevice,
context: ClContext,
queue: ClCommandQueue,

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
            .assignCompile = codegen_cl.assignCompile,
        },
    };
}

pub fn init(this: *anyopaque) Error!void {
    var state: *RuntimeCl = @ptrCast(@alignCast(this));

    var platform: ClPlatform = null;

    if (opencl.clGetPlatformIDs(1, &platform, null) != 0) {
        @branchHint(.cold);
        return Error.ContextInit;
    }
    if (opencl.clGetDeviceIDs(platform, opencl.CL_DEVICE_TYPE_GPU, 1, &state.device, null) != 0) {
        @branchHint(.cold);
        return Error.ContextInit;
    }
    var err: i32 = 0;
    state.context = opencl.clCreateContext(null, 1, &state.device, null, null, &err);
    if (err != 0) {
        @branchHint(.cold);
        return Error.ContextInit;
    }
    state.queue = opencl.clCreateCommandQueueWithProperties(state.context, state.device, null, &err);
    if (err != 0) {
        @branchHint(.cold);
        return Error.ContextInit;
    }
}
pub fn deinit(this: *anyopaque) void {
    const state: *RuntimeCl = @ptrCast(@alignCast(this));

    _ = opencl.clReleaseDevice(state.device);
    _ = opencl.clReleaseContext(state.context);
    _ = opencl.clReleaseCommandQueue(state.queue);
}
pub fn memoryAlloc(this: *anyopaque, a: u32, z: u32, y: u32, x: u32) Error!Memory {
    assert(a > 0);
    assert(z > 0);
    assert(y > 0);
    assert(x > 0);

    const state: *RuntimeCl = @ptrCast(@alignCast(this));

    var err: i32 = 0;
    const memory: opencl.cl_mem = opencl.clCreateBuffer(state.context, opencl.CL_MEM_READ_WRITE, //
        a * z * y * x * @sizeOf(f32), null, &err);
    if (err == 0) {
        @branchHint(.likely);
        return @ptrCast(memory);
    } else {
        @branchHint(.cold);
        return Error.MemoryAlloc;
    }
}
pub fn memoryFree(_: *anyopaque, memory: Memory) void {
    _ = opencl.clReleaseMemObject(@ptrCast(memory));
}
pub fn memorySyncToHost(this: *anyopaque, mem: Memory, mem_host: *anyopaque, n_bytes: u32) Error!void {
    const state: *RuntimeCl = @ptrCast(@alignCast(this));
    if (opencl.clEnqueueReadBuffer(state.queue, @ptrCast(mem), opencl.CL_TRUE, 0, n_bytes, mem_host, 0, //
        null, null) != 0)
    {
        @branchHint(.cold);
        return Error.MemorySync;
    }
}
pub fn memorySyncToDevice(this: *anyopaque, mem: Memory, mem_host: *anyopaque, n_bytes: u32) Error!void {
    const state: *RuntimeCl = @ptrCast(@alignCast(this));
    if (opencl.clEnqueueWriteBuffer(state.queue, @ptrCast(mem), opencl.CL_TRUE, 0, n_bytes, mem_host, 0, //
        null, null) != 0)
    {
        @branchHint(.cold);
        return Error.MemorySync;
    }
}
pub fn programAlloc(this: *anyopaque, source: []const u8) Error!ProgramPtr {
    const state: *RuntimeCl = @ptrCast(@alignCast(this));
    var err: i32 = 0;
    var source_c: [*c]const u8 = source[0 .. source.len - 1 :0];
    const program_ptr: opencl.cl_program = opencl.clCreateProgramWithSource(state.context, 1, //
        &source_c, &source.len, &err);
    if (err != 0) {
        @branchHint(.cold);
        return Error.ProgramAlloc;
    }

    if (opencl.clBuildProgram(program_ptr, 0, null, null, null, null) == 0) {
        @branchHint(.likely);
        return @ptrCast(program_ptr);
    } else {
        @branchHint(.cold);
        std.debug.print("{s}\n", .{source});
        return Error.ProgramAlloc;
    }
}
pub fn programFree(_: *anyopaque, program: ProgramPtr) void {
    _ = opencl.clReleaseProgram(@ptrCast(program));
}
pub fn kernelAlloc(_: *anyopaque, program: ProgramPtr, name: [*:0]const u8, args: Args) Error!KernelPtr {
    var err: i32 = 0;
    const kernel: opencl.cl_kernel = opencl.clCreateKernel(@ptrCast(program), name, &err);
    if (err != 0) {
        @branchHint(.cold);
        return Error.KernelAlloc;
    }
    for (0..args.arg_num) |arg_idx| {
        if (opencl.clSetKernelArg(kernel, @intCast(arg_idx), //
            @sizeOf(opencl.cl_mem), @ptrCast(&args.arg_mem[arg_idx])) != 0)
        {
            @branchHint(.cold);
            _ = opencl.clReleaseKernel(kernel);
            return Error.KernelAlloc;
        }
    }
    return @ptrCast(kernel);
}
pub fn kernelFree(_: *anyopaque, kernel: KernelPtr) void {
    _ = opencl.clReleaseKernel(@ptrCast(kernel));
}
pub fn kernelRun(this: *anyopaque, kernel: KernelPtr, _: Args, size_global: usize, size_local: usize) Error!void {
    const state: *RuntimeCl = @ptrCast(@alignCast(this));
    if (opencl.clEnqueueNDRangeKernel(state.queue, @ptrCast(kernel), 1, null, //
        &size_global, &size_local, 0, null, null) != 0)
    {
        @branchHint(.cold);
        return Error.KernelRun;
    }
}
pub fn queueWait(this: *anyopaque) Error!void {
    const state: *RuntimeCl = @ptrCast(@alignCast(this));
    if (opencl.clFinish(state.queue) != 0) {
        @branchHint(.cold);
        return Error.QueueWait;
    }
}
