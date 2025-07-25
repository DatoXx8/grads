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
const source_padding = Program.source_padding;
const kernel_base_name = Program.kernel_base_name;
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
// typedef enum CUpointer_attribute_enum {
//     CU_POINTER_ATTRIBUTE_CONTEXT = 1,                     /**< The ::CUcontext on which a pointer was allocated or registered */
//     CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,                 /**< The ::CUmemorytype describing the physical location of a pointer */
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
//     CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,       /**< Bitmask of allowed ::CUmemAllocationHandleType for this allocation **/
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

/// Expand buffer if necessary and set new bytes to 0
fn capacityEnsure(allocator: Allocator, source: *[]u8, offset: usize) Allocator.Error!void {
    if (source.len - offset < source_padding) {
        const len_old: usize = source.len;
        source.* = try allocator.realloc(source.*, len_old * 2);
        @memset(source.*[len_old..], 0);
    }
}

const WriteSourceError = Allocator.Error || std.fmt.BufPrintError;
/// Write format string to buffer and ensure there is at least `padding` bytes left
fn writeSource(allocator: Allocator, source: *[]u8, offset: *usize, comptime fmt: []const u8, args: anytype) WriteSourceError!void {
    // $TODO Validate that there is enough space for this and expand if there isn't
    const written = try bufPrint(source.*[offset.*..], fmt, args);
    offset.* += written.len;
    try capacityEnsure(allocator, source, offset.*);
}

/// Store the offset of the buffer at base_idx in %rd(base_idx + @intFromBool(is_in))
fn writeIndices(allocator: Allocator, source: *[]u8, offset: *usize, assign: Assign) WriteSourceError!void {
    const base_num: u32 = 1 + if (assign.inlined) |inlined| inlined.inlined_num else 0;
    var base_idx: u32 = 0;

    while (base_idx < base_num) : (base_idx += 1) {
        // const in_dim: DimInfo = if (base_idx == 0)
        //     assign.base.in_dim
        // else
        //     assign.inlined.?.base[base_idx - 1].in_dim;
        const out_dim: DimInfo = if (base_idx == 0)
            assign.base.out_dim
        else
            assign.inlined.?.base[base_idx - 1].out_dim;
        try writeSource(allocator, source, offset, tab ++ "mov.u64 %rd{}, {};\n", .{ 2 * base_idx, out_dim.off });
        if (out_dim.a_stride != 0) {
            try writeSource(allocator, source, offset, tab ++ "mov.u64 %rd{}, {s};\n", .{ 2 * base_num, register_global_id });
            // $TODO For power of two (detemined with @popCount = 1) this can be bitwise and by power of two - 1
            if (out_dim.a_reset != DimInfo.reset_default) {
                try writeSource(allocator, source, offset, tab ++ "rem.u64 %rd{}, {d};\n", .{ 2 * base_num, out_dim.a_reset });
            }
            if (out_dim.a_wait != 1) {
                try writeSource(allocator, source, offset, tab ++ "div.u64 %rd{}, {d};\n", .{ 2 * base_num, out_dim.a_wait });
            }
            if (out_dim.a_stride != 1) {
                try writeSource(allocator, source, offset, tab ++ "div.u64 %rd{}, {d};\n", .{ 2 * base_num, out_dim.a_stride });
            }
            try writeSource(allocator, source, offset, tab ++ "add.u64 %rd{}, %rd{}, %rd{};\n", //
                .{ 2 * base_idx, 2 * base_idx, 2 * base_num });
        }
    }
}

pub fn assignCompile(
    this: *anyopaque,
    allocator: Allocator,
    source: *[]u8,
    offset: *usize,
    assign: Assign,
    name: []const u8,
    args: Args,
    size_global: u32,
    size_local: u32,
) ?void {
    assert(assign.base.repeats > 0);
    assert(size_global > 0);
    assert(size_local > 0);
    assert(size_global % size_local == 0);
    assert(std.mem.startsWith(u8, name, kernel_base_name));

    const state: *RuntimePtx = @alignCast(@ptrCast(this));
    const registers_max: u32 = state.registers_max;
    // No support for cards with less than this many registers planned.
    // You can try disabling this assertion, but this is not designed for such a case.
    assert(registers_max >= 32);

    // $FIXME Recognize actual address size. Don't know how to do that yet
    if (std.mem.eql(u8, name, kernel_base_name ++ "0")) {
        assert(source.*[0] == '\x00');
        assert(offset.* == 0);
        writeSource(allocator, source, offset,
            \\// Non official PTX generated by the Grads cmopiler for PTX 
            \\// Based on NVVM 7.0.1
            \\
            \\.version 8.7
            \\.target sm_89, texmode_independent
            \\.address_size 64
        , .{}) catch return null;
    }
    writeSource(allocator, source, offset, ".entry {s}(\n", .{name}) catch return null;
    for (0..args.arg_num) |arg_idx| {
        if (arg_idx == args.arg_num - 1) {
            writeSource(allocator, source, offset, //
                tab ++ ".param .u64 .ptr .global .align 4 {s}_param_{}\n", .{ name, arg_idx }) catch return null;
        } else {
            writeSource(allocator, source, offset, //
                tab ++ ".param .u64 .ptr .global .align 4 {s}_param_{},\n", .{ name, arg_idx }) catch return null;
        }
    }
    writeSource(allocator, source, offset, ")\n{{\n", .{}) catch return null;

    // $FIXME reserve registers

    // $TODO What about %envreg3? It's there in the compiled OpenCl but I don't understand it
    writeSource(allocator, source, offset, tab ++ "mov.u32 %r2, %ctaid.x;\n", .{}) catch return null;
    writeSource(allocator, source, offset, tab ++ "mov.u32 %r3, %ntid.x;\n", .{}) catch return null;
    writeSource(allocator, source, offset, tab ++ "mov.u32 %r4, %tid.x;\n", .{}) catch return null;
    writeSource(allocator, source, offset, tab ++ "mad.lo.s32 %r1, %r3, %r2, %r4;\n", .{}) catch return null;
    writeSource(allocator, source, offset, tab ++ "mov.u32 %r7, %r1;\n", .{}) catch return null;

    const kernel_repeats_leftover: bool = (assign.base.repeats % size_global) != 0;
    const kernel_repeats: u32 = @divFloor(assign.base.repeats, size_global) +
        @intFromBool(kernel_repeats_leftover);
    for (0..kernel_repeats) |kernel_idx| {
        if (kernel_idx != 0) {
            writeSource(allocator, source, offset, tab ++ "add.u32 %r1, %r1, {};\n", .{size_global}) catch return null;
        }
        if (kernel_repeats_leftover and kernel_idx == kernel_repeats - 1) {
            writeSource(allocator, source, offset, tab ++ "setp.gt.u32 %p1, %r7, {};\n", .{kernel_repeats_leftover - 1}) catch return null;
            writeSource(allocator, source, offset, tab ++ "@%p1 bra $EXIT;\n", .{}) catch return null;
        }

        writeIndices(allocator, source, offset, assign);
        writeValue(allocator, source, offset, assign, a, z, y, x);
    }

    writeSource(allocator, source, offset,
        \\$EXIT:
        \\    ret;
        \\}}\n
    , .{}) catch return null;
}
