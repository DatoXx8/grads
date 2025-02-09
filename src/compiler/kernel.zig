const std = @import("std");

const buffer_name_size: usize = @import("../tensor.zig").buffer_name_size;

const Cl = @import("../runtimes/cl.zig");
const ClMem = Cl.ClMem;
const ClKernel = Cl.ClKernel;
const ClProgram = Cl.ClProgram;
const ClDevice = Cl.ClDevice;
const ClContext = Cl.ClContext;
const ClError = Cl.ClError;

const Ssa = @import("./ssa.zig").Ssa;

const open_cl = @import("../runtimes/cl.zig").open_cl;

const assert = std.debug.assert;

pub const Args = struct {
    arg_name_offset: []usize,
    arg_mem: []ClMem,
    arg_num: usize,
    pub fn alloc(allocator: std.mem.Allocator, layer: []Ssa.Assign) !Args {
        // TODO: Refactor this to use either hashtables or some other clever thing
        const arg_initial: usize = 4;
        var arg_name_offset: []usize = try allocator.alloc(usize, arg_initial);
        var arg_mem: []ClMem = try allocator.alloc(ClMem, arg_initial);
        var arg_num: usize = 0;

        for (0..layer.len) |assign_idx| {
            var arg_found_out: bool = false;
            for (0..arg_num) |arg_idx| {
                if (!arg_found_out and arg_name_offset[arg_idx] == layer[assign_idx].base.out.name_offset) {
                    arg_found_out = true;
                    break;
                }
            }

            if (!arg_found_out) {
                if (arg_num == arg_name_offset.len) {
                    arg_name_offset = try allocator.realloc(arg_name_offset, arg_name_offset.len * 2);
                    arg_mem = try allocator.realloc(arg_mem, arg_name_offset.len);
                }
                arg_name_offset[arg_num] = layer[assign_idx].base.out.name_offset;
                arg_mem[arg_num] = layer[assign_idx].base.out.values_cl.?;
                arg_num += 1;
            }
            // Split because saving on string comparisons saves a lot of computation
            if (!layer[assign_idx].base.type.isUnary()) {
                var arg_found_in: bool = false;
                for (0..arg_num) |arg_idx| {
                    if (!arg_found_in and arg_name_offset[arg_idx] == layer[assign_idx].base.in.name_offset) {
                        arg_found_in = true;
                        break;
                    }
                }
                if (!arg_found_in) {
                    if (arg_num == arg_name_offset.len) {
                        arg_name_offset = try allocator.realloc(arg_name_offset, arg_name_offset.len * 2);
                        arg_mem = try allocator.realloc(arg_mem, arg_name_offset.len);
                    }
                    arg_name_offset[arg_num] = layer[assign_idx].base.in.name_offset;
                    arg_mem[arg_num] = layer[assign_idx].base.in.values_cl.?;
                    arg_num += 1;
                }
            }
        }

        return .{
            .arg_name_offset = arg_name_offset,
            .arg_mem = arg_mem,
            .arg_num = arg_num,
        };
    }
    pub fn free(this: *@This(), allocator: std.mem.Allocator) void {
        // The arg_mem get's freed with the tensors
        allocator.free(this.arg_name_offset);
        allocator.free(this.arg_mem);
    }
};

pub const Kernel = struct {
    args: Args,
    name_c: []u8,
    kernel: ClKernel,
    pub fn alloc(
        program: ClProgram,
        name_c: []u8,
        args: Args,
    ) !Kernel {
        // This @ptrCast is the ultimate trust me bro
        const kernel: ClKernel = try ClKernel.alloc(program, @ptrCast(name_c));

        for (0..args.arg_num) |arg_idx| {
            // This pointer cast business is necessary because the function expects a pointer to the cl_mem,
            // but the function signature is just a void *, which confuses the zig compiler because cl_mem is a pointer to _cl_mem
            if (open_cl.clSetKernelArg(kernel.kernel, @truncate(arg_idx), //
                @sizeOf(open_cl.cl_mem), @ptrCast(&args.arg_mem[arg_idx].memory)) != 0)
            {
                return ClError.ArgNotSet;
            }
        }

        return .{
            .args = args,
            .kernel = kernel,
            .name_c = name_c,
        };
    }
    pub fn free(this: *@This(), allocator: std.mem.Allocator) !void {
        try this.kernel.free();
        this.args.free(allocator);
        allocator.free(this.name_c);
    }
};
