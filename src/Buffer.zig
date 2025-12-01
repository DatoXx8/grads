const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const Program = @import("compiler/Program.zig");
const Memory = Program.Memory;
const Runtime = @import("compiler/runtimes/Runtime.zig");
const util = @import("util.zig");

/// 4 is probably already enough. 26 ^ 4 = 456.976
/// 8 is absolute overkill. 26 ^ 8 = 208.827.064.576
/// No real downside to making this a larger value except making codegen slightly slower
pub const buffer_name_size: u32 = 8;
/// Valid increment of the char values in the buffer name
pub const buffer_name_char_options: u32 = 'z' - 'a' + 1;

pub const Buffer = @This();
pub const Vec4 = struct {
    a: u32,
    z: u32,
    y: u32,
    x: u32,
    pub fn splat(value: u32) Vec4 {
        return .{ .a = value, .z = value, .y = value, .x = value };
    }
    pub fn setA(vec4: Vec4, value: u32) Vec4 {
        return .{ .a = value, .z = vec4.z, .y = vec4.y, .x = vec4.x };
    }
    pub fn setZ(vec4: Vec4, value: u32) Vec4 {
        return .{ .a = vec4.a, .z = value, .y = vec4.y, .x = vec4.x };
    }
    pub fn setY(vec4: Vec4, value: u32) Vec4 {
        return .{ .a = vec4.a, .z = vec4.z, .y = value, .x = vec4.x };
    }
    pub fn setX(vec4: Vec4, value: u32) Vec4 {
        return .{ .a = vec4.a, .z = vec4.z, .y = vec4.y, .x = value };
    }
    pub fn equal(vec4_1: Vec4, vec4_2: Vec4) bool {
        return vec4_1.a == vec4_2.a and vec4_1.z == vec4_2.z and
            vec4_1.y == vec4_2.y and vec4_1.x == vec4_2.x;
    }
    pub fn productOfElements(vec4: Vec4) u32 {
        return vec4.a * vec4.z * vec4.y * vec4.x;
    }
};
pub const Stride = struct {
    a: u32,
    z: u32,
    y: u32,
    pub fn equal(stride_1: Stride, stride_2: Stride) bool {
        return stride_1.a == stride_2.a and stride_1.z == stride_2.z and
            stride_1.y == stride_2.y;
    }
    pub fn x(_: Stride) u32 {
        return 1;
    }
    pub fn toVec4(stride: Stride) Vec4 {
        return .{ .a = stride.a, .z = stride.z, .y = stride.y, .x = 1 };
    }
};
// $TODO Stride can be a Vec3! This means View is 32 bytes instead of 36!
pub const SyncStatus = enum(u8) {
    sync_to_host,
    sync_to_device,
    sync_to_none,
};
pub const Id = u32;
pub const Kind = enum(u8) {
    /// This is just to make debugging easier, if you see this anywhere something has gone wrong.
    free,
    /// Buffer that is supposed to hold the end result of some computation.
    /// Values in this should not get changed by the optimizer.
    normal,
    /// Intermediary buffers are *not* expected to hold the same values after the compilers optimizations.
    intermediary,
};
/// Current view into the underlying buffer
pub const View = struct {
    size: Vec4,
    stride: Stride,
    offset: u32, // Gets repurposed as the next free id in case this is an entry in the free list
    pub fn at(view_1: View, offset: Vec4) u32 {
        return view_1.offset + offset.a * view_1.stride.a + offset.z * view_1.stride.z +
            offset.y * view_1.stride.y + offset.x;
    }
    pub fn atNoOffset(view_1: View, offset: Vec4) u32 {
        return offset.a * view_1.stride.a + offset.z * view_1.stride.z +
            offset.y * view_1.stride.y + offset.x;
    }
    pub inline fn aOffset(view_1: View) u32 {
        return @divFloor(view_1.offset, view_1.stride.a);
    }
    pub inline fn zOffset(view_1: View) u32 {
        return @divFloor(view_1.offset % view_1.stride.a, view_1.stride.z);
    }
    pub inline fn yOffset(view_1: View) u32 {
        return @divFloor(view_1.offset % view_1.stride.z, view_1.stride.y);
    }
    pub inline fn xOffset(view_1: View) u32 {
        return view_1.offset % view_1.stride.y;
    }
    pub inline fn overlaps(view_1: View, view_2: View) bool {
        const a_1: u32 = view_1.aOffset();
        const z_1: u32 = view_1.zOffset();
        const y_1: u32 = view_1.yOffset();
        const x_1: u32 = view_1.xOffset();

        const a_2: u32 = view_2.aOffset();
        const z_2: u32 = view_2.zOffset();
        const y_2: u32 = view_2.yOffset();
        const x_2: u32 = view_2.xOffset();

        return @max(a_1, a_2) < @min(a_1 + view_1.size.a, a_2 + view_2.size.a) and
            @max(z_1, z_2) < @min(z_1 + view_1.size.z, z_2 + view_2.size.z) and
            @max(y_1, y_2) < @min(y_1 + view_1.size.y, y_2 + view_2.size.y) and
            @max(x_1, x_2) < @min(x_1 + view_1.size.x, x_2 + view_2.size.x);
    }
    pub inline fn overlapsAll(view_1: View, view_2: View) bool {
        return view_1.size.a == view_2.size.a and
            view_1.size.z == view_2.size.z and
            view_1.size.y == view_2.size.y and
            view_1.size.x == view_2.size.x and
            view_1.aOffset() == view_2.aOffset() and
            view_1.zOffset() == view_2.zOffset() and
            view_1.yOffset() == view_2.yOffset() and
            view_1.xOffset() == view_2.xOffset();
    }
    pub inline fn overlapsPartial(view_1: View, view_2: View) bool {
        return view_1.overlaps(view_2) and !view_1.overlapsAll(view_2);
    }
};
pub const Data = struct {
    view: View,
    values: []f32,
    values_runtime: Memory,
    sync: SyncStatus,
    kind: Kind,
    pub fn nextFree(data_1: Data) Id {
        assert(data_1.kind == .free);
        return data_1.view.offset;
    }
};

id: Id,

pub fn alloc(runtime: Runtime, gpa: Allocator, arena: Allocator, size: Vec4, kind: Kind) !Buffer {
    assert(switch (kind) {
        .free => false,
        .normal => true,
        .intermediary => true,
    });
    assert(size.a > 0);
    assert(size.z > 0);
    assert(size.y > 0);
    assert(size.x > 0);

    const buffer: Buffer = .{ .id = try Runtime.pool_global.nextId(gpa) };
    const buffer_data: *Data = buffer.data();
    buffer_data.* = .{
        .view = .{
            .offset = 0,
            .size = size,
            .stride = .{ .a = size.z * size.y * size.x, .z = size.y * size.x, .y = size.x },
        },
        .sync = SyncStatus.sync_to_none,
        .values = try arena.alloc(f32, size.productOfElements()),
        .values_runtime = try runtime.memoryAlloc(size),
        .kind = kind,
    };

    return buffer;
}
pub fn name(buffer: Buffer) [buffer_name_size]u8 {
    var name_result: [buffer_name_size]u8 = [_]u8{'a'} ** buffer_name_size;
    const divisor: u64 = buffer_name_char_options;
    var left: u64 = buffer.id;
    var char_idx: u32 = 0;
    while (char_idx < buffer_name_size) : (char_idx += 1) {
        name_result[char_idx] += @intCast(left % divisor);
        left /= divisor;
    }
    assert(left == 0); // Enforce that you don't generate new buffers beyond 'zzzz...zzz'

    return name_result;
}
pub fn free(buffer: Buffer, runtime: Runtime) void {
    runtime.memoryFree(buffer.data().*.values_runtime);
    Runtime.pool_global.freeId(buffer.id);
}
pub fn data(buffer: Buffer) *Data {
    assert(buffer.id < Runtime.pool_global.data_id_next);
    return &Runtime.pool_global.data[buffer.id];
}
pub fn view(buffer: Buffer) *View {
    assert(buffer.id < Runtime.pool_global.data_id_next);
    return &Runtime.pool_global.data[buffer.id].view;
}
pub fn syncToHost(buffer: Buffer, runtime: Runtime) !void {
    const buffer_data: *Data = buffer.data();
    if (buffer_data.*.sync == .sync_to_host) {
        const n_bytes: u32 = @intCast(buffer_data.*.values.len * @sizeOf(@TypeOf(buffer_data.*.values[0])));
        try runtime.memorySyncToHost(buffer_data.*.values_runtime, buffer_data.*.values.ptr, n_bytes);
        buffer_data.*.sync = .sync_to_none;
    }
}
pub fn syncToDevice(buffer: Buffer, runtime: Runtime) !void {
    const buffer_data: *Data = buffer.data();
    if (buffer_data.*.sync == .sync_to_device) {
        const n_bytes: u32 = @intCast(buffer_data.*.values.len * @sizeOf(@TypeOf(buffer_data.*.values[0])));
        try runtime.memorySyncToDevice(buffer_data.*.values_runtime, buffer_data.*.values.ptr, n_bytes);
        buffer_data.*.sync = .sync_to_none;
    }
}
pub fn syncUpdate(buffer: Buffer, sync: SyncStatus) void {
    const buffer_data: *Data = buffer.data();
    assert(buffer_data.*.sync == .sync_to_none or buffer_data.*.sync == sync);
    assert(sync != .sync_to_none);
    buffer_data.*.sync = sync;
}
// /// Checks for equal size, offset and name.
// pub inline fn equal(buffer: Buffer, target: Buffer) bool {
// }
// /// Checks for equal size and name.
// /// Does *not* check for inherent buffer size or offsets.
// pub inline fn equalNoOffset(buffer: Buffer, target: Buffer) bool {
// }
// $TODO This could also just be a function that returns a View... Decide if that is a good idea
pub fn moveReshape(buffer: *Buffer, size: Vec4) void {
    const buffer_data: *Data = buffer.data();
    assert(size.a > 0);
    assert(size.z > 0);
    assert(size.y > 0);
    assert(size.x > 0);
    assert(size.a <= buffer_data.*.values.len);
    assert(size.z <= buffer_data.*.values.len);
    assert(size.y <= buffer_data.*.values.len);
    assert(size.x <= buffer_data.*.values.len);
    assert(size.productOfElements() <= buffer_data.*.values.len);
    buffer_data.*.view.size = size;
    buffer_data.*.view.stride = .{ .a = size.z * size.y * size.x, .z = size.y * size.x, .y = size.x };
}
pub fn moveResize(buffer: Buffer, size: Vec4) void {
    const buffer_data: *Data = buffer.data();
    assert(size.a > 0);
    assert(size.z > 0);
    assert(size.y > 0);
    assert(size.x > 0);
    assert(size.a <= buffer_data.*.values.len);
    assert(size.z <= buffer_data.*.values.len);
    assert(size.y <= buffer_data.*.values.len);
    assert(size.x <= buffer_data.*.values.len);
    assert(size.productOfElements() <= buffer_data.*.values.len);
    buffer_data.*.view.size = size;
}
pub fn moveOffset(buffer: Buffer, offset: Vec4) void {
    const buffer_data: *Data = buffer.data();
    assert(offset.a < buffer_data.*.values.len);
    assert(offset.z < buffer_data.*.values.len);
    assert(offset.y < buffer_data.*.values.len);
    assert(offset.x < buffer_data.*.values.len);
    const offset_idx: u32 = offset.a * buffer_data.*.view.stride.a +
        offset.z * buffer_data.*.view.stride.z +
        offset.y * buffer_data.*.view.stride.y +
        offset.x;
    assert(offset_idx < buffer_data.*.values.len);
    buffer_data.*.view.offset = offset_idx;
}
pub fn print(buffer: Buffer, padding: comptime_int, offset: comptime_int, desc: ?[]const u8) void {
    if (desc) |text| {
        util.log.print("{s}Buffer {s} = {s}\n", .{ " " ** offset, buffer.name(), text });
    } else {
        util.log.print("{s}Buffer {s}\n", .{ " " ** offset, buffer.name() });
    }

    const buffer_data: Data = buffer.data().*;
    const buffer_view: View = buffer.view().*;

    var a: u32 = 0;
    while (a < buffer_view.size.a) : (a += 1) {
        var z: u32 = 0;
        while (z < buffer_view.size.z) : (z += 1) {
            var y: u32 = 0;
            while (y < buffer_view.size.y) : (y += 1) {
                util.log.print("{s}[", .{" " ** (offset + padding)});
                var x: u32 = 0;
                while (x < buffer_view.size.x) : (x += 1) {
                    util.log.print(" {d:8.4}", .{buffer_data.values[buffer_view.at(.{ .a = a, .z = z, .y = y, .x = x })]});
                }
                util.log.print("]\n", .{});
            }
            if (z != buffer_view.size.z - 1) {
                util.log.print("\n", .{});
            }
        }
        if (a != buffer_view.size.a - 1) {
            util.log.print("\n", .{});
        }
    }
}
