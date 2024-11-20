pub const kernel_name: []u8 = [_]u8{"k"};

const std = @import("std");

const buffer_name_size: u32 = @import("../tensor.zig").buffer_name_size;

const Cl = @import("../runtimes/cl.zig");
const ClMem = Cl.ClMem;
const ClKernel = Cl.ClKernel;
const ClProgram = Cl.ClProgram;
const ClDevice = Cl.ClDevice;
const ClContext = Cl.ClContext;

const Pir = @import("./pir.zig").Pir;

pub const Kernel = struct {
    arg_name: []?[buffer_name_size]u8,
    arg_mem: []?ClMem,
    arg_num: u32,
    // arg_cap is arg_name.len
    // source: [*c]u8,
    source: [*:0]u8,
    kernel: ClKernel,
    program: ClProgram,
    pub fn alloc(
        allocator: anytype,
        context: ClContext,
        device: ClDevice,
        arg_name: []?[buffer_name_size]u8,
        arg_mem: []?ClMem,
        arg_num: u32,
        source: [*:0]const u8,
    ) !Kernel {
        const program: ClProgram = try ClProgram.alloc(allocator, context.context, device.device, source, std.mem.len(source));
        const kernel: ClKernel = try ClKernel.alloc(program, kernel_name);

        return .{
            .arg_name = arg_name,
            .arg_mem = arg_mem,
            .arg_num = arg_num,
            .source = source,
            .kernel = kernel,
            .program = program,
        };
    }
    pub fn free(allocator: anytype, kernel: Kernel) !void {
        try kernel.kernel.free();
        try kernel.program.free();
        allocator.free(kernel.source);
        for (0..kernel.arg_num) |arg_idx| {
            if (kernel.arg_mem[arg_idx]) |mem| {
                mem.free();
            }
            if (kernel.arg_name[arg_idx]) |name| {
                allocator.free(name);
            }
        }
        allocator.free(kernel.arg_name);
        allocator.free(kernel.arg_mem);
    }
};
