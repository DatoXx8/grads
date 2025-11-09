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
pub const SyncStatus = enum(u8) {
    sync_to_host,
    sync_to_device,
    sync_to_none,
};
pub const Id = enum(u32) {
    _,
    pub fn name(id: Id) [buffer_name_size]u8 {
        var name_result: [buffer_name_size]u8 = [_]u8{'a'} ** buffer_name_size;
        const divisor: u64 = buffer_name_char_options;
        var left: u64 = id;
        var char_idx: u32 = 0;
        while (char_idx < buffer_name_size) : (char_idx += 1) {
            name_result[char_idx] += @intCast(left % divisor);
            left /= divisor;
        }
        assert(left == 0); // Enforce that you don't generate new buffers beyond 'zzzz...zzz'

        return name_result;
    }
    pub fn from(val: u32) Id {
        return @enumFromInt(val);
    }
    pub fn value(id: Id) u32 {
        return @intFromEnum(id);
    }
    pub fn fetch(id: Id) *Buffer {
        const id_value: u32 = id.value();
        assert(id_value < pool_global.buffer_id_next);
        return &pool_global.buffer[id_value];
    }
};
pub const Kind = enum(u8) {
    /// This is just to make debugging easier, if you see this anywhere something has gone wrong.
    free,
    /// Buffer that is supposed to hold the end result of some computation.
    /// Values in this should not get changed by the optimizer.
    normal,
    /// Intermediary buffers are *not* expected to hold the same values after the compilers optimizations.
    intermediary,
};
pub const Pool = struct {
    buffer_id_next: Id,
    buffer_id_free: Id,
    buffer: []Buffer,
    pub fn nextId(gpa: Allocator, pool: *Pool) !Id {
        if (pool.buffer_id_next == pool.buffer_id_free) {
            defer {
                pool.buffer_id_next = Id.from(pool.buffer_id_next.value() + 1); // $TODO Maybe just make a .incr function
                pool.buffer_id_free = Id.from(pool.buffer_id_free.value() + 1);
            }
            if (pool.buffer_id_next.value() == pool.buffer.len) {
                pool.buffer = try gpa.realloc(pool.buffer, @min(16, pool.buffer.len * 2));
            }
            return pool.buffer_id_next;
        } else {
            const id: Id = pool.buffer_id_free;
            const buffer_id: *Buffer = id.fetch();
            if (id == buffer_id.nextFree()) {
                pool.buffer_id_free = pool.buffer_id_next;
            } else {
                pool.buffer_id_free = buffer_id.nextFree();
            }
            assert(buffer_id.kind == .free);
            return id;
        }
    }
    pub fn freeId(pool: *Pool, id: Id) void {
        const buffer_id: *Buffer = id.fetch();
        assert(buffer_id.kind != .free);
        // Don't really see a reason the do a special case when id is the last in the pool to decrement buffer_id_next
        if (pool.buffer_id_free == pool.buffer_id_next) {
            buffer_id.*.offset = id.value();
        } else {
            buffer_id.*.offset = pool.buffer_id_free;
        }
        buffer_id.*.kind = .free;
        pool.buffer_id_free = id;
    }
};

var pool_global: Pool = .{
    .buffer = &.{},
    .buffer_id_next = 0,
    .buffer_id_free = 0,
};

a_size: u32,
z_size: u32,
y_size: u32,
x_size: u32,
a_stride: u32,
z_stride: u32,
y_stride: u32,
x_stride: u32,
offset: u32, // Gets repurposed as the next free id in case this is an entry in the free list
values: []f32,
values_runtime: Memory,
sync: SyncStatus,
kind: Kind,
pub fn alloc(runtime: Runtime, arena: Allocator, a: u32, z: u32, y: u32, x: u32, kind: Kind) !Id {
    util.todo(@src()); // Don't yet know how I feel about storing the id back in here again. One should only ever get access to a buffer through its id so that feels redundant
    assert(switch (kind) {
        .free => false,
        .normal => true,
        .intermediary => true,
    });
    assert(a > 0);
    assert(z > 0);
    assert(y > 0);
    assert(x > 0);

    const buffer_id: Id = pool_global.nextId();
    const buffer: *Buffer = buffer_id.fetch();
    buffer.* = .{
        .sync = SyncStatus.sync_to_none,
        .a_size = a,
        .z_size = z,
        .y_size = y,
        .x_size = x,
        .a_stride = z * y * x,
        .z_stride = y * x,
        .y_stride = x,
        .x_stride = 1,
        .offset = 0,
        .values = try arena.alloc(f32, a * z * y * x),
        .values_runtime = try runtime.memoryAlloc(a, z, y, x),
        .kind = kind,
    };

    return buffer_id;
}
pub fn free(buffer: Buffer, runtime: Runtime) void {
    runtime.memoryFree(buffer.values_runtime);
}
pub fn nextFree(buffer: Buffer) Id {
    assert(buffer.kind == .free);
    return Id.from(buffer.offset);
}
pub inline fn aOffset(buffer: Buffer) u32 {
    return @divFloor(buffer.offset, buffer.a_stride);
}
pub inline fn zOffset(buffer: Buffer) u32 {
    return @divFloor(buffer.offset % buffer.a_stride, buffer.z_stride);
}
pub inline fn yOffset(buffer: Buffer) u32 {
    return @divFloor(buffer.offset % buffer.z_stride, buffer.y_stride);
}
pub inline fn xOffset(buffer: Buffer) u32 {
    return @divFloor(buffer.offset % buffer.y_stride, buffer.x_stride);
}
pub inline fn at(buffer: Buffer, a: u32, z: u32, y: u32, x: u32) u32 {
    const offset: u32 = buffer.offset + a * buffer.a_stride + z * buffer.z_stride +
        y * buffer.y_stride + x * buffer.x_stride;
    assert(offset < buffer.values.len);
    return offset;
}
pub fn syncToHost(buffer: *Buffer, runtime: Runtime) !void {
    if (buffer.sync == .sync_to_host) {
        const n_bytes: u32 = @intCast(buffer.values.len * @sizeOf(@TypeOf(buffer.values[0])));
        try runtime.memorySyncToHost(buffer.values_runtime, buffer.values.ptr, n_bytes);
        buffer.sync = .sync_to_none;
    }
}
pub fn syncToDevice(buffer: *Buffer, runtime: Runtime) !void {
    if (buffer.sync == .sync_to_device) {
        const n_bytes: u32 = @intCast(buffer.values.len * @sizeOf(@TypeOf(buffer.values[0])));
        try runtime.memorySyncToDevice(buffer.values_runtime, buffer.values.ptr, n_bytes);
        buffer.sync = .sync_to_none;
    }
}
pub fn syncUpdate(buffer: *Buffer, sync: SyncStatus) void {
    assert(buffer.sync == .sync_to_none or buffer.sync == sync);
    assert(sync != .sync_to_none);
    buffer.sync = sync;
}
/// Checks for equal size, offset and name.
pub inline fn equal(buffer: Buffer, target: Buffer) bool {
    return buffer.id == target.id and
        buffer.a_size == target.a_size and buffer.z_size == target.z_size and
        buffer.y_size == target.y_size and buffer.x_size == target.x_size and
        buffer.offset == target.offset;
}
/// Checks for equal size and name.
/// Does *not* check for inherent buffer size or offsets.
pub inline fn equalNoOffset(buffer: Buffer, target: Buffer) bool {
    return buffer.id == target.id and
        buffer.a_size == target.a_size and buffer.z_size == target.z_size and
        buffer.y_size == target.y_size and buffer.x_size == target.x_size;
}
/// Return wether the two buffers overlap in any place
pub inline fn overlaps(buffer: Buffer, target: Buffer) bool {
    const a_1: u32 = buffer.aOffset();
    const z_1: u32 = buffer.zOffset();
    const y_1: u32 = buffer.yOffset();
    const x_1: u32 = buffer.xOffset();

    const a_2: u32 = target.aOffset();
    const z_2: u32 = target.zOffset();
    const y_2: u32 = target.yOffset();
    const x_2: u32 = target.xOffset();

    return @max(a_1, a_2) < @min(a_1 + buffer.a_size, a_2 + target.a_size) and
        @max(z_1, z_2) < @min(z_1 + buffer.z_size, z_2 + target.z_size) and
        @max(y_1, y_2) < @min(y_1 + buffer.y_size, y_2 + target.y_size) and
        @max(x_1, x_2) < @min(x_1 + buffer.x_size, x_2 + target.x_size);
}
/// Return wether the two buffers overlap in all places
pub inline fn overlapsAll(buffer: Buffer, target: Buffer) bool {
    return buffer.a_size == target.a_size and
        buffer.z_size == target.z_size and
        buffer.y_size == target.y_size and
        buffer.x_size == target.x_size and
        buffer.aOffset() == target.aOffset() and
        buffer.zOffset() == target.zOffset() and
        buffer.yOffset() == target.yOffset() and
        buffer.xOffset() == target.xOffset();
}
/// Return wether the two buffers overlap in some, but not all places
pub inline fn overlapsPartial(buffer: Buffer, target: Buffer) bool {
    return buffer.overlaps(target) and !buffer.overlapsAll(target);
}
pub fn moveReshape(buffer: *Buffer, a: u32, z: u32, y: u32, x: u32) void {
    assert(a > 0);
    assert(z > 0);
    assert(y > 0);
    assert(x > 0);
    assert(a <= buffer.values.len);
    assert(z <= buffer.values.len);
    assert(y <= buffer.values.len);
    assert(x <= buffer.values.len);
    assert(a * z * y * x <= buffer.values.len);
    buffer.a_size = a;
    buffer.z_size = z;
    buffer.y_size = y;
    buffer.x_size = x;
    buffer.a_stride = z * y * x;
    buffer.z_stride = y * x;
    buffer.y_stride = x;
    buffer.x_stride = 1;
}
pub fn moveResize(buffer: *Buffer, a: u32, z: u32, y: u32, x: u32) void {
    assert(a > 0);
    assert(z > 0);
    assert(y > 0);
    assert(x > 0);
    assert(a <= buffer.values.len);
    assert(z <= buffer.values.len);
    assert(y <= buffer.values.len);
    assert(x <= buffer.values.len);
    assert(a * z * y * x <= buffer.values.len);
    buffer.a_size = a;
    buffer.z_size = z;
    buffer.y_size = y;
    buffer.x_size = x;
}
pub fn moveOffset(buffer: *Buffer, a: u32, z: u32, y: u32, x: u32) void {
    assert(a < buffer.values.len);
    assert(z < buffer.values.len);
    assert(y < buffer.values.len);
    assert(x < buffer.values.len);
    const offset: u32 = a * buffer.a_stride + z * buffer.z_stride + y * buffer.y_stride +
        x * buffer.x_stride;
    assert(offset < buffer.values.len);
    buffer.offset = offset;
}
pub fn print(buffer: Buffer, padding: comptime_int, offset: comptime_int, desc: ?[]const u8) void {
    if (desc) |text| {
        util.log.print("{s}Buffer {s} = {s}\n", .{ " " ** offset, buffer.name(), text });
    } else {
        util.log.print("{s}Buffer {s}\n", .{ " " ** offset, buffer.name() });
    }
    var a: u32 = 0;
    while (a < buffer.a_size) : (a += 1) {
        var z: u32 = 0;
        while (z < buffer.z_size) : (z += 1) {
            var y: u32 = 0;
            while (y < buffer.y_size) : (y += 1) {
                util.log.print("{s}[", .{" " ** (offset + padding)});
                var x: u32 = 0;
                while (x < buffer.x_size) : (x += 1) {
                    util.log.print(" {d:8.4}", .{buffer.values[buffer.at(a, z, y, x)]});
                }
                util.log.print("]\n", .{});
            }
            if (z != buffer.z_size - 1) {
                util.log.print("\n", .{});
            }
        }
        if (a != buffer.a_size - 1) {
            util.log.print("\n", .{});
        }
    }
}
