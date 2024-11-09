// PIR = Parallel intermediate representation

const Op = @import("../tensor.zig").Op;

const DimInfo = struct {
    off_in: u32,
    off_out: u32,
    str_a_out: u32,
    str_z_out: u32,
    str_y_out: u32,
    str_x_out: u32,
    str_a_in: u32,
    str_z_in: u32,
    str_y_in: u32,
    str_x_in: u32,
    wai_a_out: u32,
    wai_z_out: u32,
    wai_y_out: u32,
    wai_x_out: u32,
    wai_a_in: u32,
    wai_z_in: u32,
    wai_y_in: u32,
    wai_x_in: u32,
    res_a_out: u32,
    res_z_out: u32,
    res_y_out: u32,
    res_x_out: u32,
    res_a_in: u32,
    res_z_in: u32,
    res_y_in: u32,
    res_x_in: u32,
};

const Pir = struct {
    repeat_num: u32,
    op_num: u32,
    op: []Op,
    dim_info: []DimInfo,
};

const l = struct {
    k: void,
};

pub const kernel_name: []u8 = [_]u8{"k"};

const Kernel = struct {
    arg_name: []?[]u8,
    // arg_mem: []?ClMem,
    arg_num: u32,
    // arg_cap is arg_name.len
    // source: [*c]u8,
    source: [*]u8,
    // cl_kernel: ClKernel,
    // cl_program: ClProgram,
};

pub const Program = struct {
    size_global: u32,
    size_local: u32,
    kernel_num: u32,
    kernel: []Kernel,
    // cl_device_id: *ClDeviceId,
    // cl_context: *ClContext,
    // cl_command_queue: *ClCommandQueue,
};
