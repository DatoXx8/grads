const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Tensor = @import("../Tensor.zig");
const Op = Tensor.Op;
const Linearized = Tensor.Linearized;
const Buffer = Tensor.Buffer;
const opt = @import("optimize.zig");
const Optimization = opt.Optimization;

const Program = @import("Program.zig");
const Memory = Program.Memory;

pub const OffsetGen = struct {
    pub const value_none: u32 = 0; // All values except offset can never be set to 0 after initialization
    pub const stride_none: u32 = 0;
    pub const reset_none: u32 = 1 << 31;
    pub const wait_none: u32 = 1;
    offset: u32,
    a_stride: u32,
    z_stride: u32,
    y_stride: u32,
    x_stride: u32,
    a_reset: u32,
    z_reset: u32,
    y_reset: u32,
    x_reset: u32,
    a_wait: u32,
    z_wait: u32,
    y_wait: u32,
    x_wait: u32,
    pub fn init(offset: u32) OffsetGen {
        return .{
            .offset = offset,
            .a_stride = value_none,
            .z_stride = value_none,
            .y_stride = value_none,
            .x_stride = value_none,
            .a_reset = value_none,
            .z_reset = value_none,
            .y_reset = value_none,
            .x_reset = value_none,
            .a_wait = value_none,
            .z_wait = value_none,
            .y_wait = value_none,
            .x_wait = value_none,
        };
    }
    pub fn postprocess(this: *OffsetGen) void {
        this.a_stride = if (this.a_stride == value_none) stride_none else this.a_stride;
        this.z_stride = if (this.z_stride == value_none) stride_none else this.z_stride;
        this.y_stride = if (this.y_stride == value_none) stride_none else this.y_stride;
        this.x_stride = if (this.x_stride == value_none) stride_none else this.x_stride;
        this.a_reset = if (this.a_reset == value_none) reset_none else this.a_reset;
        this.z_reset = if (this.z_reset == value_none) reset_none else this.z_reset;
        this.y_reset = if (this.y_reset == value_none) reset_none else this.y_reset;
        this.x_reset = if (this.x_reset == value_none) reset_none else this.x_reset;
        this.a_wait = if (this.a_wait == value_none) wait_none else this.a_wait;
        this.z_wait = if (this.z_wait == value_none) wait_none else this.z_wait;
        this.y_wait = if (this.y_wait == value_none) wait_none else this.y_wait;
        this.x_wait = if (this.x_wait == value_none) wait_none else this.x_wait;
    }
    pub fn print(this: OffsetGen, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}{s}\n", .{ " " ** offset, text });
        }
        const indent: []const u8 = " " ** (offset * padding);
        std.debug.print("{s}offset: {d:10}\n", .{ indent, this.offset_base });
        std.debug.print("{s}stride ({d:10}, {d:10}, {d:10}, {d:10})\n", //
            .{ indent, this.a_stride, this.z_stride, this.y_stride, this.x_stride });
        std.debug.print("{s}reset  ({d:10}, {d:10}, {d:10}, {d:10})\n", //
            .{ indent, this.a_reset, this.z_reset, this.y_reset, this.x_reset });
        std.debug.print("{s}wait   ({d:10}, {d:10}, {d:10}, {d:10})\n", //
            .{ indent, this.a_wait, this.z_wait, this.y_wait, this.x_wait });
    }
};

// Not really that small, but removes strides, sync status and values.len, saves 4*4+1+8 = 41 Bytes
pub const BufferSmall = struct {
    a_size: u32,
    z_size: u32,
    y_size: u32,
    x_size: u32,
    offset: u32,
    intermediary: bool,
    id: u64,
    values_ptr_host: [*]f32,
    values_ptr_device: Memory,
    pub fn init(buffer: Buffer) BufferSmall {
        return .{
            .id = buffer.id,
            .intermediary = buffer.intermediary,
            .values_ptr_host = buffer.values.ptr,
            .values_ptr_device = buffer.values_runtime,
            .offset = buffer.offset,
            .a_size = buffer.a_size,
            .z_size = buffer.z_size,
            .y_size = buffer.y_size,
            .x_size = buffer.x_size,
        };
    }
    pub fn equal(this: BufferSmall, target: BufferSmall) bool {
        return this.id == target.id and this.offset == target.offset and
            this.a_size == target.a_size and this.z_size == target.z_size and
            this.y_size == target.y_size and this.x_size == target.x_size;
    }
    pub fn equalNoOffset(this: BufferSmall, target: BufferSmall) bool {
        return this.id == target.id and
            this.a_size == target.a_size and this.z_size == target.z_size and
            this.y_size == target.y_size and this.x_size == target.x_size;
    }
    pub inline fn aOffset(this: BufferSmall) u32 {
        return @divFloor(this.offset, this.a_stride);
    }
    pub inline fn zOffset(this: BufferSmall) u32 {
        return @divFloor(this.offset % this.a_stride, this.z_stride);
    }
    pub inline fn yOffset(this: BufferSmall) u32 {
        return @divFloor(this.offset % this.z_stride, this.y_stride);
    }
    pub inline fn xOffset(this: BufferSmall) u32 {
        return @divFloor(this.offset % this.y_stride, this.x_stride);
    }
    pub inline fn at(this: BufferSmall, a: u32, z: u32, y: u32, x: u32) u32 {
        return this.offset + a * this.a_stride + z * this.z_stride +
            y * this.y_stride + x * this.x_stride;
    }
    /// Return wether the two buffers overlap in any place
    pub inline fn overlaps(this: BufferSmall, target: BufferSmall) bool {
        const a_1: u32 = this.aOffset();
        const z_1: u32 = this.zOffset();
        const y_1: u32 = this.yOffset();
        const x_1: u32 = this.xOffset();

        const a_2: u32 = target.aOffset();
        const z_2: u32 = target.zOffset();
        const y_2: u32 = target.yOffset();
        const x_2: u32 = target.xOffset();

        return @max(a_1, a_2) < @min(a_1 + this.a_size, a_2 + target.a_size) and
            @max(z_1, z_2) < @min(z_1 + this.z_size, z_2 + target.z_size) and
            @max(y_1, y_2) < @min(y_1 + this.y_size, y_2 + target.y_size) and
            @max(x_1, x_2) < @min(x_1 + this.x_size, x_2 + target.x_size);
    }
    /// Return wether the two buffers overlap in all places
    pub inline fn overlapsAll(this: BufferSmall, target: BufferSmall) bool {
        return this.a_size == target.a_size and
            this.z_size == target.z_size and
            this.y_size == target.y_size and
            this.x_size == target.x_size and
            this.aOffset() == target.aOffset() and
            this.zOffset() == target.zOffset() and
            this.yOffset() == target.yOffset() and
            this.xOffset() == target.xOffset();
    }
    /// Return wether the two buffers overlap in some, but not all places
    pub inline fn overlapsPartial(this: BufferSmall, target: BufferSmall) bool {
        return this.overlaps(target) and !this.overlapsAll(target);
    }
    pub fn print(this: BufferSmall, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}BufferSmall {s}\n", .{ " " ** offset, text });
        }
        std.debug.print("{s}({d} {d} {d} {d}) [{d} {d} {d} {d} = {d}] \"{s}\"\n", .{
            " " ** (offset + padding),
            this.out.a_size,
            this.out.z_size,
            this.out.y_size,
            this.out.x_size,
            this.out.aOffset(),
            this.out.zOffset(),
            this.out.yOffset(),
            this.out.xOffset(),
            this.out.offset,
            this.out.name(),
        });
    }
};
pub const Base = struct {
    kind: Op.Kind,
    u_var: f32,
    repeats: u32,
    out_offset_gen: OffsetGen,
    in_offset_gen: OffsetGen,
    out: BufferSmall,
    in: BufferSmall,
    pub inline fn equal(this: Base, target: Base) bool {
        return this.out.equal(target.out) and this.in.equal(target.in) and
            this.kind == target.kind and this.u_var == target.u_var;
    }
    pub inline fn equalNoOffset(this: Base, target: Base) bool {
        return this.out.equalNoOffset(target.out) and this.in.equalNoOffset(target.in) and
            this.kind == target.kind and this.u_var == target.u_var;
    }
    pub fn print(this: Base, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Base repeats {d:9}, u_var {d:9}, kind {t}, name {s}\n", //
                .{ " " ** offset, this.repeats, this.u_var, this.kind, text });
        } else {
            std.debug.print("{s}Base repeats {d:9}, u_var {d:9}, kind {t}\n", //
                .{ " " ** offset, this.repeats, this.u_var, this.kind });
        }
        std.debug.print("{s}out\n", .{" " ** offset});
        this.out.print(padding, offset + padding, null);
        this.out_offset_gen.print(padding, offset + padding, null);
        std.debug.print("{s}in\n", .{" " ** offset});
        this.in.print(padding, offset + padding, null);
        this.in_offset_gen.print(padding, offset + padding, null);
    }
};
pub const Inlined = struct {
    /// Indices equal to the "parent" assign_idx
    out_idx: u32,
    in_idx: u32,
};

// Since I want to make optimization in any order possible it might not be the smartest
// idea ever to just plainly store the indices of the inlined ops, because you need to make sure
// that none of the removed ops are referenced by others
