pub const kernel_name: []u8 = [_]u8{"k"};

const buffer_name_size: u32 = @import("../tensor.zig").buffer_name_size;

pub const Kernel = struct {
    arg_name: []?[buffer_name_size]u8,
    // arg_mem: []?ClMem,
    arg_num: u32,
    // arg_cap is arg_name.len
    // source: [*c]u8,
    source: [*:0]u8,
    // cl_kernel: ClKernel,
    // cl_program: ClProgram,
};
