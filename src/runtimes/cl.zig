const builtin = @import("builtin");
const std = @import("std");
const assert = @import("../util.zig").assert;
// const opencl_version = @import("opencl_config").opencl_version;

pub const kernel_name: []const u8 = &[_]u8{'k'};
pub const kernel_name_c: []const u8 = kernel_name ++ [_]u8{'\x00'};

const opencl_header = switch (builtin.target.os.tag) {
    .macos => "OpenCL/cl.h",
    else => "CL/cl.h",
};

pub const opencl = @cImport({
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
        var platform: opencl.cl_platform_id = null;
        var device: opencl.cl_device_id = null;
        var err: i32 = 0;

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
        var context: opencl.cl_context = null;
        var err: i32 = 0;
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
        if (opencl.clReleaseContext(context.context) == 0) {
            return;
        } else {
            return ClError.ContextNotFreed;
        }
    }
};

pub const ClCommandQueue = struct {
    queue: opencl.cl_command_queue,
    pub fn alloc(device: ClDevice, context: ClContext) !ClCommandQueue {
        var err: i32 = 0;
        const queue: ClCommandQueue = .{ .queue = opencl.clCreateCommandQueueWithProperties(context.context, device.device, null, &err) };
        if (err == 0) {
            return queue;
        } else {
            return ClError.QueueNotAlloc;
        }
    }
    pub fn free(queue: ClCommandQueue) !void {
        const err: i32 = opencl.clReleaseCommandQueue(queue.queue);
        if (err == 0) {
            return;
        } else {
            return ClError.QueueNotFreed;
        }
    }
};

pub const ClProgram = struct {
    program: opencl.cl_program,
    pub fn alloc(allocator: anytype, context: ClContext, device: ClDevice, source: []u8) !ClProgram {
        var log_size: usize = 0;
        var err: i32 = 0;
        // TODO: Get rid of this optional stuff
        var log: ?[]u8 = null;
        var log_c: ?[*:0]u8 = null;
        var source_c: [*c]u8 = source[0 .. source.len - 1 :0];
        const program: opencl.cl_program = opencl.clCreateProgramWithSource(context.context, 1, &source_c, &source.len, &err);
        if (err != 0) {
            return ClError.ProgramNotCreated;
        }

        if (opencl.clBuildProgram(program, 0, null, null, null, null) == 0) {
            return .{
                .program = program,
            };
        } else {
            std.debug.print("{s}\n", .{source});
            _ = opencl.clGetProgramBuildInfo(program, device.device, opencl.CL_PROGRAM_BUILD_LOG, 0, null, &log_size);
            log = try allocator.alloc(u8, log_size);
            defer allocator.free(log.?);
            @memset(log.?[0..], 0);
            log_c = log.?[0 .. log.?.len - 1 :0];
            _ = opencl.clGetProgramBuildInfo(program, device.device, opencl.CL_PROGRAM_BUILD_LOG, log_size + 1, log_c, null);
            std.debug.print("{s}\n", .{log.?});
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
    pub fn alloc(program: ClProgram) !ClKernel {
        var err: i32 = 0;
        const kernel: opencl.cl_kernel = opencl.clCreateKernel(program.program, kernel_name_c[0 .. kernel_name_c.len - 1 :0], &err);
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

        var err: i32 = 0;
        const memory: opencl.cl_mem = opencl.clCreateBuffer(context.context, opencl.CL_MEM_READ_WRITE, a * z * y * x * @sizeOf(f32), null, &err);
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
