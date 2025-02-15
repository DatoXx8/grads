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
    pub fn alloc(allocator: std.mem.Allocator, layer: []Ssa.Assign) !Args {
        var arg_unique = std.AutoHashMap(usize, ClMem).init(allocator);
        errdefer arg_unique.deinit();
        defer arg_unique.deinit();

        for (0..layer.len) |assign_idx| {
            try arg_unique.put(layer[assign_idx].base.out.name_offset, layer[assign_idx].base.out.values_cl.?);
            if (!layer[assign_idx].base.type.isUnary()) {
                try arg_unique.put(layer[assign_idx].base.in.name_offset, layer[assign_idx].base.in.values_cl.?);
            }
            if (layer[assign_idx].inlined) |*inlined| {
                for (0..inlined.base.len) |inlined_idx| {
                    try arg_unique.put(inlined.base[inlined_idx].out.name_offset, inlined.base[inlined_idx].out.values_cl.?);
                    if (!inlined.base[inlined_idx].type.isUnary()) {
                        try arg_unique.put(inlined.base[inlined_idx].in.name_offset, inlined.base[inlined_idx].in.values_cl.?);
                    }
                }
            }
        }

        const arg_num: usize = arg_unique.count();
        const arg_name: []usize = try allocator.alloc(usize, arg_num);
        errdefer allocator.free(arg_name);
        const arg_mem: []ClMem = try allocator.alloc(ClMem, arg_num);
        errdefer allocator.free(arg_mem);

        var arg_name_iterator = arg_unique.keyIterator();
        for (0..arg_num) |arg_idx| {
            const key: usize = arg_name_iterator.next().?.*;
            arg_name[arg_idx] = key;
            arg_mem[arg_idx] = arg_unique.get(key).?;
        }

        return .{
            .arg_name_offset = arg_name,
            .arg_mem = arg_mem,
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
    kernel: ClKernel,
    pub fn alloc(
        program: ClProgram,
        name_c: []u8,
        args: Args,
    ) !Kernel {
        // This @ptrCast is the ultimate trust me bro
        const kernel: ClKernel = try ClKernel.alloc(program, @ptrCast(name_c));

        for (0..args.arg_mem.len) |arg_idx| {
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
        };
    }
    pub fn free(this: *@This(), allocator: std.mem.Allocator) !void {
        try this.kernel.free();
        this.args.free(allocator);
    }
};
