const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const Cl = @import("../runtimes/cl.zig");
const ClMem = Cl.ClMem;
const ClKernel = Cl.ClKernel;
const ClProgram = Cl.ClProgram;
const ClDevice = Cl.ClDevice;
const ClContext = Cl.ClContext;
const ClError = Cl.ClError;
const opencl = @import("../runtimes/cl.zig").opencl;
const buffer_name_size = @import("../tensor.zig").buffer_name_size;
const Ssa = @import("./ssa.zig").Ssa;
const Assign = @import("./ssa.zig").Assign;

pub const Args = struct {
    arg_name_offset: []u64,
    arg_mem: []ClMem,
    pub fn alloc(allocator: Allocator, layer: []Assign) !Args {
        var arg_unique = std.AutoHashMap(u64, ClMem).init(allocator);
        errdefer arg_unique.deinit();
        defer arg_unique.deinit();

        for (0..layer.len) |assign_idx| {
            try arg_unique.put(layer[assign_idx].base.out.name_offset, layer[assign_idx].base.out.values_cl.?);
            if (!layer[assign_idx].base.type.isUnary()) {
                try arg_unique.put(layer[assign_idx].base.in.name_offset, layer[assign_idx].base.in.values_cl.?);
            }

            if (layer[assign_idx].inlined) |inlined| {
                // $TODO If the thing is inlined then don't there is no need to pass it to the kernel
                for (0..inlined.inlined_num) |inlined_idx| {
                    try arg_unique.put(inlined.base[inlined_idx].out.name_offset, inlined.base[inlined_idx].out.values_cl.?);
                    if (!inlined.base[inlined_idx].type.isUnary()) {
                        try arg_unique.put(inlined.base[inlined_idx].in.name_offset, inlined.base[inlined_idx].in.values_cl.?);
                    }
                }
            }
        }

        const arg_num: usize = arg_unique.count();
        const arg_name: []u64 = try allocator.alloc(u64, arg_num);
        const arg_mem: []ClMem = try allocator.alloc(ClMem, arg_num);

        errdefer {
            allocator.free(arg_name);
            allocator.free(arg_mem);
        }

        var arg_name_iterator = arg_unique.keyIterator();
        for (0..arg_num) |arg_idx| {
            const key: u64 = arg_name_iterator.next().?.*;
            arg_name[arg_idx] = key;
            arg_mem[arg_idx] = arg_unique.get(key).?;
        }

        return .{
            .arg_name_offset = arg_name,
            .arg_mem = arg_mem,
        };
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
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
        name_c: []const u8,
        args: Args,
    ) !Kernel {
        // This @ptrCast is the ultimate trust me bro
        const kernel: ClKernel = try ClKernel.alloc(program, @ptrCast(name_c));

        for (0..args.arg_mem.len) |arg_idx| {
            // This pointer cast business is necessary because the function expects a pointer to the cl_mem,
            // but the function signature is just a void *, which confuses the zig compiler because cl_mem is a pointer to _cl_mem
            if (opencl.clSetKernelArg(kernel.kernel, @truncate(arg_idx), //
                @sizeOf(opencl.cl_mem), @ptrCast(&args.arg_mem[arg_idx].memory)) != 0)
            {
                return ClError.ArgNotSet;
            }
        }

        return .{
            .args = args,
            .kernel = kernel,
        };
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
        this.kernel.free() catch |err| {
            std.log.err("Could not free kernel because of error {!}\n", .{err});
        };
        this.args.free(allocator);
    }
};
