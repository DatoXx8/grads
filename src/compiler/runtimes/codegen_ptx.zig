const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const bufPrint = std.fmt.bufPrint;

const Tensor = @import("../../Tensor.zig");
const buffer_name_size = Tensor.buffer_name_size;
const Op = Tensor.Op;
const nameFromId = Tensor.Buffer.nameFromId;
const todo = @import("../../util.zig").todo;
const Pir = @import("../Pir.zig");
const Assign = Pir.Assign;
const Base = Pir.Base;
const DimInfo = Pir.DimInfo;
const Inlined = Pir.Inlined;
const Program = @import("../Program.zig");
const kernel_base_name = Program.kernel_base_name;
const length_int_max = Program.length_int_max;
const Args = Program.Args;
const Runtime = @import("Runtime.zig");
const RuntimePtx = Runtime.RuntimePtx;

// Register scheme:
// %r___ .b32
//      %r1 always holds the global id, %r7 hold the unincremented global id, the other %r___ are only used to compute %r1.
//      6 in total? 7 with envreg
//
// %rd___ .b64
//      %rd0 holds the offset calculated for assign.base.out
//      %rd1 holds the offset calculated for assign.base.in
//      %rd2(i+1)   holds the offset for assign.inlined.base[i].out
//      %rd2(i+1)+1 holds the offset for assign.inlined.base[i].in
//      (2 * (inlined_num + 1) in total, could recycle unused ones from further up in the tree)
//
// %f___ .f32
//      Trickiest one. For now I am tribking to reset these indices after every a, z, y, x
//      (Then there should be at most (exactly?) 2 + pir.inlined.inlined_num %f registers
//      Alternatively 3 + ... if the output wants to be in a different register
//      Could recycle old registers from further up the tree.)
//
// %p___ .pred
//      %p1 always hold the early exit condition. i.e if global_id is less than graeter (equal?) some value then exit
//      (Always 1 in total)
//
// In case of assign loop reuse the registers from the previous one

// Comment from https://forums.developer.nvidia.com/t/registers-per-thread-limit-and-occupancy/486 regarding register limits:
// "
// This is a little confusing in the programming guide (fixed in next version), thanks for pointing it out. It’s not that registered are allocated in multiples of 64…
//
// Here’s the new info that will be in the programming guide:
//
// Several blocks can be processed by the same multiprocessor concurrently by allocating the multiprocessor’s registers and shared memory among the blocks. More precisely, the number of registers available per thread is equal to:
//
// N_registersPerMultiprocessor / CEIL(N_concurrentBlocks*N_threadsPerBlock, 64)
//
// where N_registersPerMultiprocessor is the total number of registers per multiprocessor, N_concurrentBlocks is the number of concurrent blocks, N_threadsPerBlock is the number of threads per block, and CEIL(X, 64) means rounded up to the nearest multiple of 64.
//
// (So the 64 is not referring to registers, but to threads)
//
// Mark
// "

// Find address_size in here?
// kinddef enum CUpointer_attribute_enum {
//     CU_POINTER_ATTRIBUTE_CONTEXT = 1,                     /**< The ::CUcontext on which a pointer was allocated or registered */
//     CU_POINTER_ATTRIBUTE_MEMORY_kind = 2,                 /**< The ::CUmemorykind describing the physical location of a pointer */
//     CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,              /**< The address at which a pointer's memory may be accessed on the device */
//     CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,                /**< The address at which a pointer's memory may be accessed on the host */
//     CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,                  /**< A pair of tokens for use with the nv-p2p.h Linux kernel interface */
//     CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,                 /**< Synchronize every synchronous memory operation initiated on this region */
//     CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,                   /**< A process-wide unique ID for an allocated memory region*/
//     CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,                  /**< Indicates if the pointer points to managed memory */
//     CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,              /**< A device ordinal of a device on which a pointer was allocated or registered */
//     CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10, /**< 1 if this pointer maps to an allocation that is suitable for ::cudaIpcGetMemHandle, 0 otherwise **/
//     CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,           /**< Starting address for this requested pointer */
//     CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,                 /**< Size of the address range for this requested pointer */
//     CU_POINTER_ATTRIBUTE_MAPPED = 13,                     /**< 1 if this pointer is in a valid address range that is mapped to a backing allocation, 0 otherwise **/
//     CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_kindS = 14,       /**< Bitmask of allowed ::CUmemAllocationHandleKind for this allocation **/
//     CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15, /**< 1 if the memory this pointer is referencing can be used with the GPUDirect RDMA API **/
//     CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,               /**< Returns the access flags the device associated with the current context has on the corresponding memory referenced by the pointer given */
//     CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17,             /**< Returns the mempool handle for the allocation if it was allocated from a mempool. Otherwise returns NULL. **/
//     CU_POINTER_ATTRIBUTE_MAPPING_SIZE = 18,               /**< Size of the actual underlying mapping that the pointer belongs to **/ <-- This one?
//     CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR = 19,          /**< The start address of the mapping that the pointer belongs to **/
//     CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID = 20             /**< A process-wide unique id corresponding to the physical allocation the pointer belongs to **/
//   , CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE = 21    /**< Returns in \p *data a boolean that indicates whether the pointer points to memory that is capable to be used for hardware accelerated decompression. */
// } CUpointer_attribute;
//
// here?
// cuMemGetAddressRange
//
// Kind of seems like it's just @bitSizeOf(usize)?

// $TODO Get SIMD width

// Float format 0f[hex] instead of 0x[hex]. Doable with 0f{X}

const tab: []const u8 = "    ";
const register_global_id: []const u8 = "%r1";

/// Write format string to buffer and ensure there is at least `padding` bytes left
fn writeSource(source: *[]u8, offset: *usize, comptime fmt: []const u8, args: anytype) void {
    const written = bufPrint(source.*[offset.*..], fmt, args) catch unreachable;
    offset.* += written.len;
}

/// Store the offset of the buffer at base_idx in %rd(base_idx + @intFromBool(is_in))
fn writeIndices(source: *[]u8, offset: *usize, assign: Assign) void {
    const base_num: u32 = 1 + if (assign.inlined) |inlined| inlined.inlined_num else 0;
    var base_idx: u32 = 0;

    while (base_idx < base_num) : (base_idx += 1) {
        for (0..2) |side| {
            const is_out: bool = side == 0;
            const dim_info: DimInfo = if (base_idx == 0)
                (if (is_out) assign.base.out_dim else assign.base.in_dim)
            else
                (if (is_out) assign.inlined.?.base[base_idx - 1].out_dim else assign.inlined.?.base[base_idx - 1].in_dim);
            writeSource(source, offset, tab ++ "mov.u64 %rd{}, {};\n", //
                .{ 2 * base_idx + side, dim_info.off });
            for (0..4) |dim| {
                const stride: u32 = switch (dim) {
                    0 => dim_info.a_stride,
                    1 => dim_info.x_stride,
                    2 => dim_info.y_stride,
                    3 => dim_info.z_stride,
                    else => unreachable,
                };
                const wait: u32 = switch (dim) {
                    0 => dim_info.a_wait,
                    1 => dim_info.x_wait,
                    2 => dim_info.y_wait,
                    3 => dim_info.z_wait,
                    else => unreachable,
                };
                const reset: u32 = switch (dim) {
                    0 => dim_info.a_reset,
                    1 => dim_info.x_reset,
                    2 => dim_info.y_reset,
                    3 => dim_info.z_reset,
                    else => unreachable,
                };
                if (stride != 0) {
                    writeSource(source, offset, tab ++ "mov.u64 %rd{}, {s};\n", //
                        .{ 2 * base_num, register_global_id });
                    if (reset != DimInfo.reset_default) {
                        assert(reset != 0);
                        assert(reset != 1);
                        // const power_of_two: bool = std.math.isPowerOfTwo(reset);
                        const power_of_two: bool = @popCount(reset) == 1;
                        if (power_of_two) {
                            writeSource(source, offset, tab ++ "and.u64 %rd{}, {d};\n", //
                                .{ 2 * base_num, reset - 1 });
                        } else {
                            writeSource(source, offset, tab ++ "rem.u64 %rd{}, {d};\n", //
                                .{ 2 * base_num, reset });
                        }
                    }
                    if (wait != 1) {
                        assert(wait != 0);
                        writeSource(source, offset, tab ++ "div.b64 %rd{}, {d};\n", //
                            .{ 2 * base_num, wait });
                    }
                    if (stride == 1) {
                        assert(stride != 0);
                        writeSource(source, offset, tab ++ "add.u64 %rd{}, %rd{}, %rd{};\n", //
                            .{ 2 * base_idx + side, 2 * base_num, 2 * base_idx + side });
                    } else {
                        writeSource(source, offset, tab ++ "mad.lo.u64 %rd{}, %rd{}, {d}, %rd{};\n", //
                            .{ 2 * base_idx + side, 2 * base_num, stride, 2 * base_idx + side });
                    }
                }
            }
        }
    }
}
fn writeBase(source: *[]u8, offset: *usize, assign: Assign) void {
    _ = source;
    _ = offset;
    _ = assign;
}
fn writeAssign(source: *[]u8, offset: *usize, assign: Assign) void {
    _ = source;
    _ = offset;
    const a_size: u32 = if (assign.base.kind.isReduce()) assign.base.in.a_size else assign.base.out.a_size;
    const z_size: u32 = if (assign.base.kind.isReduce()) assign.base.in.z_size else assign.base.out.z_size;
    const y_size: u32 = if (assign.base.kind.isReduce()) assign.base.in.y_size else assign.base.out.y_size;
    const x_size: u32 = if (assign.base.kind.isReduce()) assign.base.in.x_size else assign.base.out.x_size;
    var a: u32 = 0;
    while (a < a_size) : (a += 1) {
        var z: u32 = 0;
        while (z < z_size) : (z += 1) {
            var y: u32 = 0;
            while (y < y_size) : (y += 1) {
                var x: u32 = 0;
                while (x < x_size) : (x += 1) {
                    const off_out: u32 = if (assign.base.kind.isReduce())
                        0
                    else
                        assign.base.out.at(a, z, y, x);
                    const off_in: u32 = if (assign.base.kind.isExpand())
                        assign.base.out.at(a, z, y, x)
                    else
                        0;
                    _ = off_out;
                    _ = off_in;
                }
            }
        }
    }
}
// If this breaks it might be some %envreg3 stuff. It's there in the compiled OpenCl but I don't understand it
const register_global_id_calculation: []const u8 =
    tab ++ "mov.u32 %r2, %ctaid.x;\n" ++
    tab ++ "mov.u32 %r3, %ntid.x;\n" ++
    tab ++ "mov.u32 %r4, %tid.x;\n" ++
    tab ++ "mad.lo.s32 %r1, %r3, %r2, %r4;\n" ++
    tab ++ "mov.u32 %r7, %r1;\n" ++
    "\n";
pub fn assignCompileBytes(_: *anyopaque, assign: Assign, name_len_max: u32, args: Args, size_global: u32, size_local: u32) u32 {
    const boilerplate_kernel: []const u8 =
        ".entry (\n" ++
        ")\n" ++
        "{\n" ++
        tab ++ ".reg .pred %p<>;\n" ++
        tab ++ ".reg .f32 %f<>;\n" ++
        tab ++ ".reg .b32 %r<>;\n" ++
        tab ++ ".reg .b64 %rd<>;\n" ++
        "}\n";
    const boilerplate_argument: []const u8 = tab ++ ".param .u64 .ptr .global .align 4 _param_,\n" ++
        "ld.param.u64 %rd, [_param_];\n";
    const length_header: u32 = @intCast(boilerplate_kernel.len + name_len_max + 4 * length_int_max +
        args.arg_num * (boilerplate_argument.len + 2 * name_len_max + 2 * length_int_max));

    const length_register_id_calculation: u32 = @intCast(register_global_id_calculation.len);

    _ = assign;
    _ = size_global;
    _ = size_local;

    return length_register_id_calculation + length_header;
}
pub fn assignCompile(
    this: *anyopaque,
    source: *[]u8,
    offset: *usize,
    assign: Assign,
    name: []const u8,
    args: Args,
    size_global: u32,
    size_local: u32,
) void {
    assert(assign.base.repeats > 0);
    assert(size_global > 0);
    assert(size_local > 0);
    assert(size_global % size_local == 0);
    assert(std.mem.startsWith(u8, name, kernel_base_name));

    const state: *RuntimePtx = @ptrCast(@alignCast(this));
    std.debug.print("offset {} + bytes {} < len {}", .{ offset.*, assignCompileBytes(state, assign, @intCast(name.len), args, size_global, size_local), source.len });
    assert(offset.* + assignCompileBytes(state, assign, @intCast(name.len), args, size_global, size_local) < source.len);

    const registers_max: u32 = state.registers_max;
    // No support for cards with less than this many registers planned.
    // You can try disabling this assertion, but this is not designed for such a case.
    // $FIXME For the codegen right now we just ignore the register limit and keep on allocating
    assert(registers_max >= 32);

    // $FIXME This is really just a bug. This case needs to be handled
    assert(registers_max >= 2 + if (assign.inlined) |i| i.inlined_num else 0);

    // $FIXME Recognize actual address size. Don't know how to do that yet
    if (std.mem.eql(u8, name, kernel_base_name ++ "0")) {
        assert(source.*[0] == '\x00');
        assert(offset.* == 0);
        const header_global: []const u8 =
            "// Non official PTX generated by the Grads cmopiler for PTX " ++
            "// Based on NVVM 7.0.1" ++
            "\n" ++
            ".version 8.7\n" ++
            ".target sm_89, texmode_independent\n" ++
            ".address_size 64\n";
        writeSource(source, offset, header_global, .{});
    }
    writeSource(source, offset, ".entry {s}(\n", .{name});
    for (0..args.arg_num) |arg_idx| {
        if (arg_idx == args.arg_num - 1) {
            writeSource(source, offset, //
                tab ++ ".param .u64 .ptr .global .align 4 {s}_param_{}\n", .{ name, arg_idx });
        } else {
            writeSource(source, offset, //
                tab ++ ".param .u64 .ptr .global .align 4 {s}_param_{},\n", .{ name, arg_idx });
        }
    }
    writeSource(source, offset, ")\n{{\n", .{});

    // $FIXME reserve registers
    todo(@src());

    writeSource(source, offset, register_global_id_calculation, .{});

    const kernel_repeats_leftover: bool = (assign.base.repeats % size_global) != 0;
    const kernel_repeats: u32 = @divFloor(assign.base.repeats, size_global) +
        @intFromBool(kernel_repeats_leftover);
    for (0..kernel_repeats) |kernel_idx| {
        if (kernel_idx != 0) {
            writeSource(source, offset, tab ++ "add.u32 %r1, %r1, {};\n", .{size_global});
        }
        if (kernel_repeats_leftover and kernel_idx == kernel_repeats - 1) {
            const early_exit_condition: []const u8 =
                tab ++ "setp.gt.u32 %p1, %r7, {};\n" ++
                tab ++ "@%p1 bra $EXIT;\n";
            writeSource(source, offset, early_exit_condition, .{kernel_repeats_leftover - 1});
        }

        writeIndices(source, offset, assign);
        writeAssign(source, offset, assign);
    }

    const early_exit_and_ret: []const u8 =
        "$EXIT:\n" ++
        tab ++ "ret;\n" ++
        "}}\n";
    writeSource(source, offset, early_exit_and_ret, .{});
}
