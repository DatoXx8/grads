const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const bufPrint = std.fmt.bufPrint;

const Ssa = @import("./ssa.zig").Ssa;
const Assign = @import("./ssa.zig").Assign;

const buffer_name_size = @import("../tensor.zig").buffer_name_size;
const Op = @import("../tensor.zig").Op;
const nameFromOffset = @import("../tensor.zig").Buffer.nameFromOffset;

const ClMem = @import("../runtimes/cl.zig").ClMem;

const Args = @import("./kernel.zig").Args;

pub const kernel_base_name = "kern{}";
pub const source_padding = 4096;

// TODO: Make my own format string implementation, can't really get faster trivially without changing behaviour, which I don't really mind

/// Expand buffer if necessary and set new bytes to 0
fn capacityEnsure(allocator: Allocator, source: *[]u8, offset: usize) !void {
    if (source.len - offset < source_padding) {
        const len_old: usize = source.len;
        source.* = try allocator.realloc(source.*, len_old * 2);
        @memset(source.*[len_old..], 0);
    }
}

/// Write format string to buffer and ensure there is at least `padding` bytes left
fn writeSource(allocator: Allocator, source: *[]u8, offset: *usize, comptime fmt: []const u8, args: anytype) !void {
    // TODO: Validate that there is enough space for this and expand if there isn't
    const written = try bufPrint(source.*[offset.*..], fmt, args);
    offset.* += written.len;
    try capacityEnsure(allocator, source, offset.*);
}

pub fn compileKernel(
    allocator: Allocator,
    source: *[]u8,
    offset: *usize,
    assign: []const Assign,
    kernel_name: []const u8,
    kernel_args: Args,
    size_global: usize,
    size_local: usize,
) !void {
    _ = assign;
    _ = size_global;
    _ = size_local;

    try writeSource(allocator, source, offset, "__kernel void {s}(", .{kernel_name});
    assert(kernel_args.arg_mem.len == kernel_args.arg_name_offset.len);
    for (0..kernel_args.arg_name_offset.len) |arg_idx| {
        if (arg_idx == 0) {
            try writeSource(allocator, source, offset, "__global float *{s}", .{nameFromOffset(kernel_args.arg_name_offset[arg_idx])});
        } else {
            try writeSource(allocator, source, offset, ", __global float *{s}", .{nameFromOffset(kernel_args.arg_name_offset[arg_idx])});
        }
    }
    try writeSource(allocator, source, offset, ") {{\n", .{});

    try writeSource(allocator, source, offset, "}}\n", .{});
    try capacityEnsure(allocator, source, offset.*);
}
