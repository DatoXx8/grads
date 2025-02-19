const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Ssa = @import("./ssa.zig").Ssa;
const DimInfo = Ssa.DimInfo;
const Assign = Ssa.Assign;

const buffer_name_size = @import("../tensor.zig").buffer_name_size;
const Op = @import("../tensor.zig").Op;
const nameFromOffset = @import("../tensor.zig").Buffer.nameFromOffset;

const bufPrint = std.fmt.bufPrint;

const ClMem = @import("../runtimes/cl.zig").ClMem;

const Args = @import("./kernel.zig").Args;

pub const kernel_base_name = "kern{}";
pub const capacity_min: usize = 2048;
const padding: usize = 1024;

/// Expand buffer if necessary and set new bytes to 0
fn capacityEnsure(allocator: Allocator, source: []u8, offset: usize) ![]u8 {
    if (source.len - offset < padding) {
        const len_old: usize = source.len;
        const new: []u8 = try allocator.realloc(source, len_old * 2);
        @memset(new[len_old..], 0);
        return new;
    } else {
        return source;
    }
}

/// Write format string to buffer and ensure there is at least `padding` bytes left
fn writeBuffer(allocator: Allocator, source: *[]u8, offset: *usize, comptime fmt: []const u8, args: anytype) !void {
    // TODO: Validate that there is enough space for this and expand if there isn't
    const written = try bufPrint(source.*[offset.*..], fmt, args);
    offset.* += written.len;
    source.* = try capacityEnsure(allocator, source.*, offset.*);
}
