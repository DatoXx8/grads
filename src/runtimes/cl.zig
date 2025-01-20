const builtin = @import("builtin");
const std = @import("std");
const assert = std.debug.assert;
// const OpenCl_version = @import("OpenCl_config").OpenCl_version;

pub const kernel_name: []const u8 = &[_]u8{'k'};
/// Null terminated kernel_name
pub const kernel_name_c: []const u8 = kernel_name ++ [_]u8{'\x00'};

const OpenClHeader = switch (builtin.target.os.tag) {
    .macos => "OpenCL/cl.h",
    else => "CL/cl.h",
};

pub const open_cl = @cImport({
    // @cDefine("CL_TARGET_OpenCl_VERSION", OpenCl_version);
    @cInclude(OpenClHeader);
});

pub const ClError = error{
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
    KernelNotBuilt,
    KernelNotFreed,
    ArgNotSet,
};

// TODO: Support multiple devices
pub const ClDevice = struct {
    pub const ClDeviceType = enum(u8) {
        cpu,
        gpu,
    };
    type: ClDeviceType,
    device: open_cl.cl_device_id,
    pub fn alloc(device_type: ClDevice.ClDeviceType) !ClDevice {
        var platform: open_cl.cl_platform_id = null;
        var device: open_cl.cl_device_id = null;
        var err: i32 = 0;

        err = open_cl.clGetPlatformIDs(1, &platform, null);
        if (err != 0) {
            return ClError.PlatformNotFound;
        }

        err = open_cl.clGetDeviceIDs(platform, switch (device_type) {
            .gpu => open_cl.CL_DEVICE_TYPE_GPU,
            .cpu => open_cl.CL_DEVICE_TYPE_CPU,
        }, 1, &device, null);

        if (err == 0) {
            return .{ .type = device_type, .device = device };
        } else {
            return ClError.DeviceNotFound;
        }
    }
    pub fn free(this: *@This()) !void {
        if (open_cl.clReleaseDevice(this.device) == 0) {
            return;
        } else {
            return ClError.DeviceNotFreed;
        }
    }
    pub fn maxSizeLocal(this: *const @This()) !usize {
        var size_local: usize = 0;
        if (open_cl.clGetDeviceInfo(this.device, open_cl.CL_DEVICE_MAX_WORK_GROUP_SIZE, @sizeOf(usize), &size_local, null) == 0) {
            return size_local;
        } else {
            return ClError.DeviceInfoNotFound;
        }
    }
};

pub const ClContext = struct {
    context: open_cl.cl_context,
    pub fn alloc(device: ClDevice) !ClContext {
        var context: open_cl.cl_context = null;
        var err: i32 = 0;
        context = open_cl.clCreateContext(null, 1, &device.device, null, null, &err);
        if (err == 0) {
            return .{ .context = context };
        } else {
            return ClError.ContextNotFound;
        }
    }
    pub fn free(context: ClContext) !void {
        if (open_cl.clReleaseContext(context.context) == 0) {
            return;
        } else {
            return ClError.ContextNotFreed;
        }
    }
};

pub const ClCommandQueue = struct {
    queue: open_cl.cl_command_queue,
    pub fn alloc(device: ClDevice, context: ClContext) !ClCommandQueue {
        var err: i32 = 0;
        const queue: ClCommandQueue = .{ .queue = open_cl.clCreateCommandQueueWithProperties(context.context, device.device, null, &err) };
        if (err == 0) {
            return queue;
        } else {
            return ClError.QueueNotAlloc;
        }
    }
    pub fn free(queue: ClCommandQueue) !void {
        const err: i32 = open_cl.clReleaseCommandQueue(queue.queue);
        if (err == 0) {
            return;
        } else {
            return ClError.QueueNotFreed;
        }
    }
};

pub const ClProgram = struct {
    program: open_cl.cl_program,
    pub fn alloc(allocator: anytype, context: ClContext, device: ClDevice, source: []u8) !ClProgram {
        var log_size: usize = 0;
        var err: i32 = 0;
        // TODO: Get rid of this optional stuff
        var log: ?[]u8 = null;
        var log_c: ?[*:0]u8 = null;
        var source_c: [*c]u8 = source[0 .. source.len - 1 :0];
        const program: open_cl.cl_program = open_cl.clCreateProgramWithSource(context.context, 1, &source_c, &source.len, &err);
        if (err != 0) {
            return ClError.ProgramNotCreated;
        }

        if (open_cl.clBuildProgram(program, 0, null, null, null, null) == 0) {
            return .{ .program = program };
        } else {
            std.debug.print("{s}\n", .{source});
            _ = open_cl.clGetProgramBuildInfo(program, device.device, open_cl.CL_PROGRAM_BUILD_LOG, 0, null, &log_size);
            log = try allocator.alloc(u8, log_size);
            defer allocator.free(log.?);
            @memset(log.?[0..], 0);
            log_c = log.?[0 .. log.?.len - 1 :0];
            _ = open_cl.clGetProgramBuildInfo(program, device.device, open_cl.CL_PROGRAM_BUILD_LOG, log_size + 1, log_c, null);
            std.debug.print("{s}\n", .{log.?});
            return ClError.ProgramNotBuilt;
        }
    }
    pub fn free(program: ClProgram) !void {
        if (open_cl.clReleaseProgram(program.program) == 0) {
            return;
        } else {
            return ClError.ProgramNotFreed;
        }
    }
};

pub const ClKernel = struct {
    kernel: open_cl.cl_kernel,
    pub fn alloc(program: ClProgram) !ClKernel {
        var err: i32 = 0;
        const kernel: open_cl.cl_kernel = open_cl.clCreateKernel(program.program, kernel_name_c[0 .. kernel_name_c.len - 1 :0], &err);
        if (err == 0) {
            return .{ .kernel = kernel };
        } else {
            return ClError.KernelNotBuilt;
        }
    }
    pub fn free(kernel: ClKernel) !void {
        if (open_cl.clReleaseKernel(kernel.kernel) == 0) {
            return;
        } else {
            return ClError.KernelNotFreed;
        }
    }
};

pub const ClMem = struct {
    memory: open_cl.cl_mem,
    pub fn alloc(context: ClContext, a: u32, z: u32, y: u32, x: u32) !ClMem {
        assert(a > 0);
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);

        var err: i32 = 0;
        const memory: open_cl.cl_mem = open_cl.clCreateBuffer(context.context, open_cl.CL_MEM_READ_WRITE, a * z * y * x * @sizeOf(f32), null, &err);
        if (err == 0) {
            return .{ .memory = memory };
        } else {
            return ClError.MemNotAlloc;
        }
    }
    pub fn free(this: @This()) !void {
        if (open_cl.clReleaseMemObject(this.memory) == 0) {
            return;
        } else {
            return ClError.MemNotFreed;
        }
    }
};
