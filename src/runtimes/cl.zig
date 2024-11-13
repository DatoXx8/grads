const builtin = @import("builtin");
const std = @import("std");
const assert = @import("../util.zig").assert;
// const opencl_version = @import("opencl_config").opencl_version;

const opencl_header = switch (builtin.target.os.tag) {
    .macos => "OpenCL/cl.h",
    else => "CL/cl.h",
};

const opencl = @cImport({
    // @cDefine("CL_TARGET_OPENCL_VERSION", opencl_version);
    @cInclude(opencl_header);
});

pub const ClError = error{
    PlatformNotFound,
    PlatformNotFreed,
    DeviceNotFound,
    DeviceNotFreed,
    ContextNotFound,
    ContextNotFreed,
    QueueNotAlloc,
    QueueNotFreed,
    MemNotAlloc,
    MemNotFreed,
    ProgramNotCreated,
    ProgramNotBuilt,
    ProgramNotFreed,
    KernelNotBuilt,
    KernelNotFreed,
};

// TODO: Support multiple devices
pub const ClDevice = struct {
    pub const ClDeviceType = enum(u8) {
        Cpu,
        Gpu,
    };
    type: ClDeviceType,
    device: opencl.cl_device_id,
    pub fn alloc(device_type: ClDevice.ClDeviceType) !ClDevice {
        var platform: opencl.cl_platform_id = 0;
        var device: opencl.cl_device_id = 0;
        var err: u32 = 0;

        err = opencl.clGetPlatformIDs(1, &platform, null);
        if (err != 0) {
            return ClError.PlatformNotFound;
        }

        err = opencl.clGetDeviceIDs(platform, switch (device_type) {
            .Gpu => opencl.CL_DEVICE_TYPE_GPU,
            .Cpu => opencl.CL_DEVICE_TYPE_CPU,
        }, 1, &device, null);

        if (err == 0) {
            return .{
                .type = device_type,
                .device = device,
            };
        } else {
            return ClError.DeviceNotFound;
        }
    }
    pub fn free(device: ClDevice) !void {
        if (opencl.clReleaseDevice(device.device) == 0) {
            return;
        } else {
            return ClError.DeviceNotFreed;
        }
    }
};

pub const ClContext = struct {
    context: opencl.cl_context,
    pub fn alloc(device: ClDevice) !ClContext {
        var context: opencl.cl_context = 0;
        var err: u32 = 0;
        context = opencl.clCreateContext(null, 1, &device.device, null, null, &err);
        if (err == 0) {
            return .{
                .context = context,
            };
        } else {
            return ClError.ContextNotFound;
        }
    }
    pub fn free(context: ClContext) !void {
        if (opencl.clReleaseContet(context.context) == 0) {
            return;
        } else {
            return ClError.ContextNotFreed;
        }
    }
};

pub const ClCommandQueue = struct {
    queue: opencl.cl_command_queue,
    pub fn alloc(device: ClDevice, context: ClContext) !ClCommandQueue {
        var err: u32 = 0;
        const queue: ClCommandQueue = opencl.clCreateCommandQueueWithProperties(context.context, device.device, null, &err);
        if (err == 0) {
            return queue;
        } else {
            return ClError.QueueNotAlloc;
        }
    }
    pub fn free(queue: ClCommandQueue) !void {
        const err: u32 = opencl.clReleaseCommandQueue(queue.queue);
        if (err == 0) {
            return;
        } else {
            return ClError.QueueNotFreed;
        }
    }
};

pub const ClProgram = struct {
    program: opencl.cl_program,
    pub fn alloc(allocator: anytype, context: ClContext, device: ClDevice, source: [*:0]const u8, source_size: usize) !ClProgram {
        var log_size: u32 = 0;
        var err: u32 = 0;
        var log: ?[*:0]u8 = null;
        const program: opencl.cl_program = opencl.clCreateProgramWithSource(context.context, 1, &source, &source_size, &err);
        if (err != 0) {
            return ClError.ProgramNotCreated;
        }

        if (opencl.clBuildProgram(program, 0, null, null, null, null) == 0) {
            return .{
                .program = program,
            };
        } else {
            std.log.warn("SOURCE: {s}\n", .{source});
            _ = opencl.clGetProgramBuildInfo(program, device.device, opencl.CL_PROGRAM_BUILD_LOG, 0, null, &log_size);
            log = try allocator.alloc(u8, log_size);
            defer allocator.free(log);
            _ = opencl.clGetProgramBuildInfo(program, device.device, opencl.CL_PROGRAM_BUILD_LOG, log_size + 1, log, null);
            std.log.warn("LOG: {s}\n", .{log});
            return ClError.ProgramNotBuilt;
        }
    }
    pub fn free(program: ClProgram) !void {
        if (opencl.clReleaseProgram(program.program) == 0) {
            return;
        } else {
            return ClError.ProgramNotFreed;
        }
    }
};

pub const ClKernel = struct {
    kernel: opencl.cl_kernel,
    pub fn alloc(program: ClProgram, name: [*:0]const u8) !ClKernel {
        var err: u32 = 0;
        const kernel: opencl.cl_kernel = opencl.clCreateKernel(program.program, name, &err);
        if (err == 0) {
            return .{
                .kernel = kernel,
            };
        } else {
            return ClError.KernelNotBuilt;
        }
    }
    pub fn free(kernel: ClKernel) !void {
        if (opencl.clReleaseKernel(kernel.kernel) == 0) {
            return;
        } else {
            return ClError.KernelNotFreed;
        }
    }
};

pub const ClMem = struct {
    memory: opencl.cl_mem,
    pub fn alloc(context: ClContext, a: u32, z: u32, y: u32, x: u32) !ClMem {
        assert(a > 0);
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);

        var err: u32 = 0;
        const memory: opencl.cl_mem = opencl.clCreateBuffer(context, opencl.CL_MEM_READ_WRITE, a * z * y * x * @sizeOf(f32), null, &err);
        if (err == 0) {
            return .{
                .memory = memory,
            };
        } else {
            return ClError.MemNotAlloc;
        }
    }
    pub fn free(this: @This()) !void {
        if (opencl.clReleaseMemObject(this.memory) == 0) {
            return;
        } else {
            return ClError.MemNotFreed;
        }
    }
};
