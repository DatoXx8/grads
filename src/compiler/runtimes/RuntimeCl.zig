const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");

const codegen_cl = @import("./codegen_cl.zig");
const Runtime = @import("./Runtime.zig");
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
    return Runtime{
        .state = &this,
        .vtable = .{
            .init = init,
            .deinit = deinit,
            .memoryAlloc = memoryAlloc,
            .memoryFree = memoryFree,
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

pub fn init(this: *anyopaque) ?void {
    var state: *RuntimeCl = @alignCast(@ptrCast(this));

    var platform: ClPlatform = null;

    if (opencl.clGetPlatformIDs(1, &platform, null) != 0) {
        @branchHint(.cold);
        return null;
    }
    if (opencl.clGetDeviceIDs(platform, opencl.CL_DEVICE_TYPE_GPU, 1, &state.device, null) != 0) {
        @branchHint(.cold);
        return null;
    }

    var err: i32 = 0;
    state.context = opencl.clCreateContext(null, 1, &state.device, null, null, &err);
    if (err != 0) {
        @branchHint(.cold);
        return null;
    }
    state.queue = opencl.clCreateCommandQueueWithProperties(state.context, state.device, null, &err);
    if (err != 0) {
        @branchHint(.cold);
        return null;
    }
}
pub fn deinit(this: *anyopaque) ?void {
    const state: *RuntimeCl = @alignCast(@ptrCast(this));

    var failed: bool = false;
    if (opencl.clReleaseDevice(state.device) != 0) {
        @branchHint(.cold);
        failed = true;
    }
    if (opencl.clReleaseContext(state.context) != 0) {
        @branchHint(.cold);
        failed = true;
    }
    if (opencl.clReleaseCommandQueue(state.queue) != 0) {
        @branchHint(.cold);
        failed = true;
    }
    if (failed) {
        @branchHint(.cold);
        return null;
    }
}
pub fn memoryAlloc(this: *anyopaque, a: u32, z: u32, y: u32, x: u32) ?Memory {
    assert(a > 0);
    assert(z > 0);
    assert(y > 0);
    assert(x > 0);

    const state: *RuntimeCl = @alignCast(@ptrCast(this));

    var err: i32 = 0;
    const memory: opencl.cl_mem = opencl.clCreateBuffer(state.context, opencl.CL_MEM_READ_WRITE, //
        a * z * y * x * @sizeOf(f32), null, &err);
    if (err == 0) {
        @branchHint(.likely);
        return memory;
    } else {
        @branchHint(.cold);
        return null;
    }
}
pub fn memoryFree(this: *anyopaque, memory: Memory) ?void {
    _ = this;
    if (opencl.clReleaseMemObject(@ptrCast(memory)) != 0) {
        @branchHint(.cold);
        return null;
    }
}
pub fn memorySyncToHost(this: *anyopaque, mem: Memory, mem_host: *anyopaque, n_bytes: u32) ?void {
    const state: *RuntimeCl = @alignCast(@ptrCast(this));
    if (opencl.clEnqueueReadBuffer(state.queue, mem, opencl.CL_TRUE, 0, n_bytes, mem_host, 0, //
        null, null) != 0)
    {
        @branchHint(.cold);
        return null;
    }
}
pub fn memorySyncToDevice(this: *anyopaque, mem: Memory, mem_host: *anyopaque, n_bytes: u32) ?void {
    const state: *RuntimeCl = @alignCast(@ptrCast(this));
    if (opencl.clEnqueueWriteBuffer(state.queue, mem, opencl.CL_TRUE, 0, n_bytes, mem_host, 0, //
        null, null) != 0)
    {
        @branchHint(.cold);
        return null;
    }
}
pub fn programAlloc(this: *anyopaque, source: []const u8) ?ProgramPtr {
    const state: *RuntimeCl = @alignCast(@ptrCast(this));
    var err: i32 = 0;
    var source_c: [*c]const u8 = source[0 .. source.len - 1 :0];
    const program_ptr: opencl.cl_program = opencl.clCreateProgramWithSource(state.context, 1, //
        &source_c, &source.len, &err);
    if (err != 0) {
        @branchHint(.cold);
        return null;
    }

    if (opencl.clBuildProgram(program_ptr, 0, null, null, null, null) == 0) {
        @branchHint(.likely);
        return program_ptr;
    } else {
        @branchHint(.cold);
        return null;
    }
}
pub fn programFree(this: *anyopaque, program: ProgramPtr) ?void {
    _ = this;
    if (opencl.clReleaseProgram(@ptrCast(program)) != 0) {
        @branchHint(.cold);
        return null;
    }
}
pub fn programLog(this: *anyopaque, program: ProgramPtr, allocator: Allocator) ?void {
    const state: *RuntimeCl = @alignCast(@ptrCast(this));
    var log_size: usize = 0;
    if (opencl.clGetProgramBuildInfo(program, state.device, opencl.CL_PROGRAM_BUILD_LOG, 0, //
        null, &log_size) != 0)
    {
        @branchHint(.cold);
        return null;
    }
    const log: []u8 = try allocator.alloc(u8, log_size);
    defer allocator.free(log);
    @memset(log[0..], 0);
    if (opencl.clGetProgramBuildInfo(program, state.device, opencl.CL_PROGRAM_BUILD_LOG, //
        log_size + 1, log, null) != 0)
    {
        @branchHint(.cold);
        return null;
    }
    std.debug.print("{s}\n", .{log});
}
pub fn kernelAlloc(this: *anyopaque, program: ProgramPtr, name: [*:0]const u8, args: Args) ?KernelPtr {
    _ = this;
    var err: i32 = 0;
    const kernel: opencl.cl_kernel = opencl.clCreateKernel(@ptrCast(program), name, &err);
    if (err != 0) {
        @branchHint(.cold);
        // std.log.err("Could not build kernel with name {s} because of error {}\n", .{ name, err });
        return null;
    }
    for (0..args.arg_mem.len) |arg_idx| {
        // This pointer cast business is necessary because the function expects a pointer to the cl_mem,
        // but the function signature is just a void *, which confuses the zig compiler because cl_mem is a pointer to _cl_mem
        if (opencl.clSetKernelArg(kernel, @intCast(arg_idx), //
            @sizeOf(opencl.cl_mem), @ptrCast(&args.arg_mem[arg_idx].memory)) != 0)
        {
            @branchHint(.cold);
            if (opencl.clReleaseKernel(kernel.ptr) != 0) {
                @branchHint(.cold);
                return null;
            }
            return null;
        }
    }
}
pub fn kernelFree(this: *anyopaque, kernel: KernelPtr) ?void {
    _ = this;
    if (opencl.clReleaseKernel(@ptrCast(kernel)) != 0) {
        @branchHint(.cold);
        return null;
    }
}
pub fn kernelRun(this: *anyopaque, kernel: KernelPtr, size_global: usize, size_local: usize) ?void {
    const state: *RuntimeCl = @alignCast(@ptrCast(this));
    if (opencl.clEnqueueNDRangeKernel(state.queue, @ptrCast(kernel), 1, null, //
        &size_global, &size_local, 0, null, null) != 0)
    {
        @branchHint(.cold);
        return null;
    }
}
pub fn queueWait(this: *anyopaque) ?void {
    const state: *RuntimeCl = @alignCast(@ptrCast(this));
    if (opencl.clWait(state.queue) != 0) {
        @branchHint(.cold);
        return null;
    }
}
