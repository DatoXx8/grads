const std = @import("std");

const buffer_name_size: u32 = @import("../tensor.zig").buffer_name_size;

const Cl = @import("../runtimes/cl.zig");
const ClMem = Cl.ClMem;
const ClKernel = Cl.ClKernel;
const ClProgram = Cl.ClProgram;
const ClDevice = Cl.ClDevice;
const ClContext = Cl.ClContext;
const ClError = Cl.ClError;

const Pir = @import("./pir.zig").Pir;

const source_generate = @import("./codegen.zig").generate;
const Optimisation = @import("./codegen.zig").Optimisation;

const OpenCl = @import("../runtimes/cl.zig").OpenCl;

pub const Args = struct {
    arg_name: [][buffer_name_size]u8,
    arg_mem: []ClMem,
    arg_num: u32,
};

pub const Kernel = struct {
    args: Args,
    // arg_cap is arg_name.len
    // source: [*c]u8,
    // Convert to [*:0]u8 with source[0.. :0]
    source: []u8,
    kernel: ClKernel,
    program: ClProgram,
    pub fn args_gather(allocator: anytype, pir: Pir) !Args {
        const arg_initial: u32 = 4;
        var arg_name: [][buffer_name_size]u8 = try allocator.alloc([buffer_name_size]u8, arg_initial);
        var arg_mem: []ClMem = try allocator.alloc(ClMem, arg_initial);
        var arg_num: u32 = 0;

        for (0..pir.op_num) |op_idx| {
            var arg_found_out: bool = false;
            for (0..arg_num) |arg_idx| {
                if (!arg_found_out and
                    std.mem.eql(u8, &arg_name[arg_idx], &pir.op[op_idx].out.name))
                {
                    arg_found_out = true;
                    break;
                }
            }

            if (!arg_found_out) {
                if (arg_num == arg_name.len) {
                    arg_name = try allocator.realloc(arg_name, arg_name.len * 2);
                }
                arg_name[arg_num] = pir.op[op_idx].out.name;
                // TODO: Get rid of this .? stuff
                arg_mem[arg_num] = pir.op[op_idx].out.values_cl.?;
                arg_num += 1;
            }
            // Split because saving on string comparisons saves a lot of computation
            if (!pir.op[op_idx].is_unary()) {
                var arg_found_in: bool = false;
                for (0..arg_num) |arg_idx| {
                    if (!arg_found_in and
                        std.mem.eql(u8, &arg_name[arg_idx], &pir.op[op_idx].in.name))
                    {
                        arg_found_in = true;
                        break;
                    }
                }
                if (!arg_found_in) {
                    if (arg_num == arg_name.len) {
                        arg_name = try allocator.realloc(arg_name, arg_name.len * 2);
                    }
                    arg_name[arg_num] = pir.op[op_idx].in.name;
                    // TODO: Get rid of this .? stuff
                    arg_mem[arg_num] = pir.op[op_idx].out.values_cl.?;
                    arg_num += 1;
                }
            }
        }

        return .{
            .arg_name = arg_name,
            .arg_mem = arg_mem,
            .arg_num = arg_num,
        };
    }
    pub fn alloc(
        allocator: anytype,
        context: ClContext,
        device: ClDevice,
        pir: Pir,
        size_global: u32,
        size_local: u32,
        optimisation: Optimisation,
    ) !Kernel {
        const args: Args = try Kernel.args_gather(allocator, pir);
        errdefer allocator.free(args.arg_name);
        const source: []u8 = try source_generate(allocator, pir, args, size_global, size_local, optimisation);
        errdefer allocator.free(source);

        const program: ClProgram = try ClProgram.alloc(allocator, context, device, source);
        errdefer program.free() catch {};
        const kernel: ClKernel = try ClKernel.alloc(program);
        errdefer kernel.free() catch {};

        for (0..args.arg_num) |arg_idx| {
            const err: i32 = OpenCl.clSetKernelArg(kernel.kernel, @truncate(arg_idx), //
                @sizeOf(OpenCl.cl_mem), args.arg_mem[arg_idx].memory);
            std.debug.print("Error {}\n", .{err});
            if (err != 0) {
                return ClError.ArgNotSet;
            }
        }

        return .{
            .args = args,
            .source = source,
            .kernel = kernel,
            .program = program,
        };
    }
    pub fn free(kernel: @This(), allocator: anytype) !void {
        // The arg_mem get's freed with the tensors

        // for (0..kernel.args.arg_num) |arg_idx| {
        //     try kernel.args.arg_mem[arg_idx].free();
        // }
        allocator.free(kernel.args.arg_name);
        allocator.free(kernel.args.arg_mem);
        allocator.free(kernel.source);
        try kernel.kernel.free();
        try kernel.program.free();
    }
};
