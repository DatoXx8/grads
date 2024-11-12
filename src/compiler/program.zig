const Kernel = @import("./kernel.zig").Kernel;

pub const Program = struct {
    size_global: u32,
    size_local: u32,
    kernel_num: u32,
    kernel: []Kernel,
    // cl_device_id: *ClDeviceId,
    // cl_context: *ClContext,
    // cl_command_queue: *ClCommandQueue,
};
