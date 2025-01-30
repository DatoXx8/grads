const std = @import("std");
const math = std.math;

const pcg = @import("./prng.zig").pcg;

const assert = std.debug.assert;

const ClMem = @import("./runtimes/cl.zig").ClMem;
const ClContext = @import("./runtimes/cl.zig").ClContext;
const ClCommandQueue = @import("./runtimes/cl.zig").ClCommandQueue;
const OpenCl = @import("./runtimes/cl.zig").open_cl;

// TODO: Get rid of this anytype bs. That is downright horrible imo.
// TODO: Split the file up more?

/// 4 is probably already enough. 26 ^ 4 = 456.976
/// 8 is absolute overkill. 26 ^ 8 = 208.827.064.576
pub const buffer_name_size: usize = 8;
var buffer_name_offset: usize = 0;

pub const Buffer = struct {
    pub const SyncStatus = enum(u8) {
        sync_to_host,
        sync_to_device,
        sync_to_none,
    };
    pub const SyncError = error{
        FailedToHost,
        FailedToDevice,
        FailedWait,
    };
    a_inherent: usize,
    z_inherent: usize,
    y_inherent: usize,
    x_inherent: usize,
    a_size: usize,
    z_size: usize,
    y_size: usize,
    x_size: usize,
    a_stride: usize,
    z_stride: usize,
    y_stride: usize,
    x_stride: usize,
    a_offset: usize,
    z_offset: usize,
    y_offset: usize,
    x_offset: usize,
    offset: usize,
    values: []f32,
    values_cl: ?ClMem,
    sync: SyncStatus,
    name: [buffer_name_size]u8,
    name_offset: usize,
    pub fn alloc(allocator: anytype, a: usize, z: usize, y: usize, x: usize, context: ?ClContext) !Buffer {
        assert(a > 0);
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);

        var name: [buffer_name_size]u8 = [_]u8{'a'} ** buffer_name_size;
        const divisor: usize = 26;
        var left: usize = buffer_name_offset;
        for (0..buffer_name_size) |char_idx| {
            name[char_idx] += @truncate(left % divisor);
            left = left / divisor;
        }
        // Enforce that you don't generate new tensors beyond 'zzzz...zzz'
        assert(left == 0);

        // Have to do it this way because there is no such thing as ++ in Zig.
        buffer_name_offset += 1;
        // TODO: Get rid of this case destinction. It's really ugly.
        if (context) |ctx| {
            return .{
                .name_offset = buffer_name_offset - 1,
                .name = name,
                .sync = SyncStatus.sync_to_none,
                .a_size = a,
                .z_size = z,
                .y_size = y,
                .x_size = x,
                .a_inherent = a,
                .z_inherent = z,
                .y_inherent = y,
                .x_inherent = x,
                .a_stride = z * y * x,
                .z_stride = y * x,
                .y_stride = x,
                .x_stride = 1,
                .offset = 0,
                .a_offset = 0,
                .z_offset = 0,
                .y_offset = 0,
                .x_offset = 0,
                .values = try allocator.alloc(f32, a * z * y * x),
                .values_cl = try ClMem.alloc(ctx, a, z, y, x),
            };
        } else {
            return .{
                .name_offset = buffer_name_offset - 1,
                .name = name,
                .sync = SyncStatus.sync_to_none,
                .a_size = a,
                .z_size = z,
                .y_size = y,
                .x_size = x,
                .a_inherent = a,
                .z_inherent = z,
                .y_inherent = y,
                .x_inherent = x,
                .a_stride = z * y * x,
                .z_stride = y * x,
                .y_stride = x,
                .x_stride = 1,
                .offset = 0,
                .a_offset = 0,
                .z_offset = 0,
                .y_offset = 0,
                .x_offset = 0,
                .values = try allocator.alloc(f32, a * z * y * x),
                .values_cl = null,
            };
        }
    }
    pub fn free(this: *const @This(), allocator: anytype) void {
        allocator.free(this.values);
    }
    pub fn at(this: *const @This(), a: usize, z: usize, y: usize, x: usize) usize {
        assert(a < this.a_size);
        assert(z < this.z_size);
        assert(y < this.y_size);
        assert(x < this.x_size);
        return this.offset + a * this.a_stride + z * this.z_stride + y * this.y_stride + x * this.x_stride;
    }
    pub fn syncToHost(this: *@This(), queue: ClCommandQueue) !void {
        if (this.sync == .sync_to_host) {
            const size: usize = this.a_inherent * this.z_inherent * this.y_inherent * this.x_inherent * @sizeOf(f32);
            if (OpenCl.clEnqueueReadBuffer(queue.queue, this.values_cl.?.memory, //
                OpenCl.CL_TRUE, 0, size, this.values.ptr, 0, null, null) != 0)
            {
                return SyncError.FailedToHost;
            }
            this.sync = .sync_to_none;
        }
    }
    pub fn syncToDevice(this: *@This(), queue: ClCommandQueue) !void {
        if (this.sync == .sync_to_device) {
            const size: usize = this.a_inherent * this.z_inherent * this.y_inherent * this.x_inherent * @sizeOf(f32);
            if (OpenCl.clEnqueueWriteBuffer(queue.queue, this.values_cl.?.memory, //
                OpenCl.CL_TRUE, 0, size, this.values.ptr, 0, null, null) != 0)
            {
                return SyncError.FailedToDevice;
            }
            this.sync = .sync_to_none;
        }
    }
    pub fn syncUpdate(this: *@This(), sync: SyncStatus) void {
        assert(this.sync == .sync_to_none or this.sync == sync);
        assert(sync != .sync_to_none);
        this.sync = sync;
    }
    pub fn syncWait(this: *@This(), queue: ClCommandQueue) !void {
        _ = this;
        if (OpenCl.clFinish(queue.queue) == 0) {
            return;
        } else {
            return SyncError.FailedWait;
        }
    }
};
// TODO: Maybe truncate the names to 3 letters each
// TODO: Put this in a different file
pub const Op = struct {
    // Linary is like binary but the in buffer has size [1, 1, 1, 1]
    pub const Type = enum(u8) {
        unary_add,
        unary_subtract,
        unary_multiply,
        unary_divide,
        unary_exp,
        unary_log,
        unary_square,
        unary_sqrt,
        unary_reciprocal,
        unary_max,
        unary_min,
        unary_set,
        unary_random,
        unary_tanh,
        unary_absolute,
        unary_sign,
        binary_add,
        binary_subtract,
        binary_multiply,
        binary_divide,
        binary_max,
        binary_min,
        binary_set,
        linary_add,
        linary_subtract,
        linary_multiply,
        linary_divide,
        linary_max,
        linary_min,
        linary_set,
        reduce_sum,
        reduce_max,
        reduce_avg,
        reduce_min,
        pub inline fn isUnary(this: @This()) bool {
            return this == .unary_add or this == .unary_subtract or
                this == .unary_multiply or this == .unary_divide or
                this == .unary_exp or this == .unary_log or
                this == .unary_square or this == .unary_sqrt or
                this == .unary_reciprocal or this == .unary_max or
                this == .unary_min or this == .unary_set or
                this == .unary_random or this == .unary_tanh or
                this == .unary_absolute or this == .unary_sign;
        }
        pub inline fn isBinary(this: @This()) bool {
            return this == .binary_add or this == .binary_subtract or
                this == .binary_multiply or this == .binary_divide or
                this == .binary_max or this == .binary_min or
                this == .binary_set;
        }
        pub inline fn isLinary(this: @This()) bool {
            return this == .linary_add or this == .linary_subtract or
                this == .linary_multiply or this == .linary_divide or
                this == .linary_max or this == .linary_min or
                this == .linary_set;
        }
        pub inline fn isReduce(this: @This()) bool {
            return this == .reduce_sum or this == .reduce_max or
                this == .reduce_avg or this == .reduce_min;
        }
    };
    type: Type,
    u_var: f32,
    out: Buffer,
    in: Buffer,
    pub fn equal(this: *const @This(), target: Op) bool {
        return this.type == target.type and this.u_var == this.u_var and
            this.out.name_offset == target.out.name_offset and this.out.a_size == target.out.a_size and
            this.out.z_size == target.out.z_size and this.out.y_size == target.out.y_size and
            this.out.x_size == target.out.x_size and this.in.name_offset == target.in.name_offset and
            this.in.a_size == target.in.a_size and this.in.z_size == target.in.z_size and
            this.in.y_size == target.in.y_size and this.in.x_size == target.in.x_size;
    }
    pub fn overlaps(this: *const @This(), target: Op) bool {
        // TODO: Implement this for non same-size buffers
        assert(this.out.a_size == target.out.a_size);
        assert(this.out.z_size == target.out.z_size);
        assert(this.out.y_size == target.out.y_size);
        assert(this.out.x_size == target.out.x_size);

        const a_1: usize = this.out.a_offset;
        const z_1: usize = this.out.z_offset;
        const y_1: usize = this.out.y_offset;
        const x_1: usize = this.out.x_offset;

        const a_2: usize = target.out.a_offset;
        const z_2: usize = target.out.z_offset;
        const y_2: usize = target.out.y_offset;
        const x_2: usize = target.out.x_offset;

        return @max(a_1, a_2) - @min(a_1, a_2) < this.out.a_size and
            @max(z_1, z_2) - @min(z_1, z_2) < this.out.z_size and
            @max(y_1, y_2) - @min(y_1, y_2) < this.out.y_size and
            @max(x_1, x_2) - @min(x_1, x_2) < this.out.x_size;
    }
    pub inline fn isUnary(this: *const @This()) bool {
        return this.type.isUnary();
    }
    pub inline fn isBinary(this: *const @This()) bool {
        return this.type.isBinary();
    }
    pub inline fn isLinary(this: *const @This()) bool {
        return this.type.isLinary();
    }
    pub inline fn isReduce(this: *const @This()) bool {
        return this.type.isReduce();
    }
    pub inline fn isOutInlinable(this: *const @This(), target: *const @This()) bool {
        // TODO: Need to check that there is not a way that changes, that would be done by target get lost in case they get used somewhere else
        return (!target.type.isReduce() and target.type != .linary_set and target.type != .binary_set and target.type != .unary_set) and
            this.out.name_offset == target.out.name_offset and
            this.out.a_size == target.out.a_size and
            this.out.z_size == target.out.z_size and
            this.out.y_size == target.out.y_size and
            this.out.x_size == target.out.x_size and
            this.out.a_offset == target.out.a_offset and
            this.out.z_offset == target.out.z_offset and
            this.out.y_offset == target.out.y_offset and
            this.out.x_offset == target.out.x_offset;
    }
    pub inline fn isInInlinable(this: *const @This(), target: *const @This()) bool {
        // TODO: Need to check that there is not a way that changes, that would be done by target get lost in case they get used somewhere else
        return (!target.type.isReduce() and target.type != .linary_set and target.type != .binary_set and target.type != .unary_set) and
            this.out.name_offset == target.in.name_offset and
            this.out.a_size == target.in.a_size and
            this.out.z_size == target.in.z_size and
            this.out.y_size == target.in.y_size and
            this.out.x_size == target.in.x_size and
            this.out.a_offset == target.in.a_offset and
            this.out.z_offset == target.in.z_offset and
            this.out.y_offset == target.in.y_offset and
            this.out.x_offset == target.in.x_offset;
    }
    // TODO: Optimize this with simd, see @Vector
    pub fn realize(this: *const @This()) void {
        if (this.isUnary()) {
            // In buffer is just a copy of out buffer, basically just a sanity check.
            assert(this.out.a_size == this.in.a_size);
            assert(this.out.z_size == this.in.z_size);
            assert(this.out.y_size == this.in.y_size);
            assert(this.out.x_size == this.in.x_size);
        } else if (this.isBinary()) {
            assert(this.out.a_size == this.in.a_size);
            assert(this.out.z_size == this.in.z_size);
            assert(this.out.y_size == this.in.y_size);
            assert(this.out.x_size == this.in.x_size);
        } else if (this.isLinary()) {
            assert(this.in.a_size == 1);
            assert(this.in.z_size == 1);
            assert(this.in.y_size == 1);
            assert(this.in.x_size == 1);
        } else if (this.isReduce()) {
            assert(this.out.a_size == 1);
            assert(this.out.z_size == 1);
            assert(this.out.y_size == 1);
            assert(this.out.x_size == 1);
        } else {
            unreachable;
        }
        switch (this.type) {
            .unary_add => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] += this.u_var;
                            }
                        }
                    }
                }
            },
            .unary_subtract => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] -= this.u_var;
                            }
                        }
                    }
                }
            },
            .unary_multiply => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] *= this.u_var;
                            }
                        }
                    }
                }
            },
            .unary_divide => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] /= this.u_var;
                            }
                        }
                    }
                }
            },
            .unary_exp => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] = math.exp(this.out.values[this.out.at(a, z, y, x)]);
                            }
                        }
                    }
                }
            },
            .unary_log => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] = math.log(f32, math.e, this.out.values[this.out.at(a, z, y, x)]);
                            }
                        }
                    }
                }
            },
            .unary_square => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] *= this.out.values[this.out.at(a, z, y, x)];
                            }
                        }
                    }
                }
            },
            .unary_sqrt => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] = math.sqrt(this.out.values[this.out.at(a, z, y, x)]);
                            }
                        }
                    }
                }
            },
            .unary_reciprocal => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] = 1 / this.out.values[this.out.at(a, z, y, x)];
                            }
                        }
                    }
                }
            },
            .unary_max => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] = @max(this.out.values[this.out.at(a, z, y, x)], this.u_var);
                            }
                        }
                    }
                }
            },
            .unary_min => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] = @min(this.out.values[this.out.at(a, z, y, x)], this.u_var);
                            }
                        }
                    }
                }
            },
            .unary_set => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] = this.u_var;
                            }
                        }
                    }
                }
            },
            .unary_random => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] = pcg.randF32();
                            }
                        }
                    }
                }
            },
            .unary_tanh => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] = std.math.tanh(this.out.values[this.out.at(a, z, y, x)]);
                            }
                        }
                    }
                }
            },
            .unary_absolute => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                if (this.out.values[this.out.at(a, z, y, x)] < 0) {
                                    this.out.values[this.out.at(a, z, y, x)] = -this.out.values[this.out.at(a, z, y, x)];
                                } else {
                                    this.out.values[this.out.at(a, z, y, x)] = this.out.values[this.out.at(a, z, y, x)];
                                }
                            }
                        }
                    }
                }
            },
            .unary_sign => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                if (this.out.values[this.out.at(a, z, y, x)] > 0) {
                                    this.out.values[this.out.at(a, z, y, x)] = 1;
                                } else if (this.out.values[this.out.at(a, z, y, x)] < 0) {
                                    this.out.values[this.out.at(a, z, y, x)] = -1;
                                } else {
                                    this.out.values[this.out.at(a, z, y, x)] = 0;
                                }
                            }
                        }
                    }
                }
            },
            .binary_add => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] += this.in.values[this.in.at(a, z, y, x)];
                            }
                        }
                    }
                }
            },
            .binary_subtract => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] -= this.in.values[this.in.at(a, z, y, x)];
                            }
                        }
                    }
                }
            },
            .binary_multiply => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] *= this.in.values[this.in.at(a, z, y, x)];
                            }
                        }
                    }
                }
            },
            .binary_divide => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] /= this.in.values[this.in.at(a, z, y, x)];
                            }
                        }
                    }
                }
            },
            .binary_max => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] = @max(this.out.values[this.out.at(a, z, y, x)], this.in.values[this.in.at(a, z, y, x)]);
                            }
                        }
                    }
                }
            },
            .binary_min => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] = @min(this.out.values[this.out.at(a, z, y, x)], this.in.values[this.in.at(a, z, y, x)]);
                            }
                        }
                    }
                }
            },
            .binary_set => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] = this.in.values[this.in.at(a, z, y, x)];
                            }
                        }
                    }
                }
            },
            .linary_add => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] += this.in.values[this.in.at(0, 0, 0, 0)];
                            }
                        }
                    }
                }
            },
            .linary_subtract => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] -= this.in.values[this.in.at(0, 0, 0, 0)];
                            }
                        }
                    }
                }
            },
            .linary_multiply => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] *= this.in.values[this.in.at(0, 0, 0, 0)];
                            }
                        }
                    }
                }
            },
            .linary_divide => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] /= this.in.values[this.in.at(0, 0, 0, 0)];
                            }
                        }
                    }
                }
            },
            .linary_max => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] = @max(this.out.values[this.out.at(a, z, y, x)], this.in.values[this.in.at(0, 0, 0, 0)]);
                            }
                        }
                    }
                }
            },
            .linary_min => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] = @min(this.out.values[this.out.at(a, z, y, x)], this.in.values[this.in.at(0, 0, 0, 0)]);
                            }
                        }
                    }
                }
            },
            .linary_set => {
                for (0..this.out.a_size) |a| {
                    for (0..this.out.z_size) |z| {
                        for (0..this.out.y_size) |y| {
                            for (0..this.out.x_size) |x| {
                                this.out.values[this.out.at(a, z, y, x)] = this.in.values[this.in.at(0, 0, 0, 0)];
                            }
                        }
                    }
                }
            },
            .reduce_sum => {
                this.out.values[this.out.at(0, 0, 0, 0)] = 0;
                for (0..this.in.a_size) |a| {
                    for (0..this.in.z_size) |z| {
                        for (0..this.in.y_size) |y| {
                            for (0..this.in.x_size) |x| {
                                this.out.values[this.out.at(0, 0, 0, 0)] += this.in.values[this.in.at(a, z, y, x)];
                            }
                        }
                    }
                }
            },
            .reduce_max => {
                this.out.values[this.out.at(0, 0, 0, 0)] = -std.math.inf(f32);
                for (0..this.in.a_size) |a| {
                    for (0..this.in.z_size) |z| {
                        for (0..this.in.y_size) |y| {
                            for (0..this.in.x_size) |x| {
                                this.out.values[this.out.at(0, 0, 0, 0)] = @max(this.out.values[this.out.at(0, 0, 0, 0)], this.in.values[this.in.at(a, z, y, x)]);
                            }
                        }
                    }
                }
            },
            .reduce_min => {
                this.out.values[this.out.at(0, 0, 0, 0)] = std.math.inf(f32);
                for (0..this.in.a_size) |a| {
                    for (0..this.in.z_size) |z| {
                        for (0..this.in.y_size) |y| {
                            for (0..this.in.x_size) |x| {
                                this.out.values[this.out.at(0, 0, 0, 0)] = @min(this.out.values[this.out.at(0, 0, 0, 0)], this.in.values[this.in.at(a, z, y, x)]);
                            }
                        }
                    }
                }
            },
            .reduce_avg => {
                this.out.values[this.out.at(0, 0, 0, 0)] = 0;
                for (0..this.in.a_size) |a| {
                    for (0..this.in.z_size) |z| {
                        for (0..this.in.y_size) |y| {
                            for (0..this.in.x_size) |x| {
                                this.out.values[this.out.at(0, 0, 0, 0)] += this.in.values[this.in.at(a, z, y, x)];
                            }
                        }
                    }
                }
                this.out.values[this.out.at(0, 0, 0, 0)] /= @as(f32, @floatFromInt(this.in.a_size * this.in.z_size * this.in.y_size * this.in.x_size));
            },
        }
    }
    pub fn print(this: *const @This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}{s} ", .{ " " ** (padding + offset), text });
        } else {
            std.debug.print("{s}", .{" " ** (padding + offset)});
        }
        if (this.isUnary()) {
            std.debug.print("U {c} ({d} {d} {d} {d}) [{d}] \"{s}\" {d}\n", .{
                switch (this.type) {
                    .unary_add => "add",
                    .unary_subtract => "sub",
                    .unary_multiply => "mul",
                    .unary_divide => "div",
                    .unary_exp => "exp",
                    .unary_log => "log",
                    .unary_square => "sqr",
                    .unary_sqrt => "sqt",
                    .unary_reciprocal => "rcp",
                    .unary_max => "max",
                    .unary_min => "min",
                    .unary_set => "set",
                    .unary_random => "rng",
                    .unary_tanh => "tanh",
                    .unary_absolute => "abs",
                    .unary_sign => "sgn",
                    else => unreachable,
                },
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
                this.u_var,
            });
        } else {
            const op_kind: u8 = if (this.isBinary()) 'B' else (if (this.isLinary()) 'L' else 'R');
            std.debug.print("{c} {s} ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
                op_kind,
                switch (this.type) {
                    .binary_add => "add",
                    .binary_subtract => "sub",
                    .binary_multiply => "mul",
                    .binary_divide => "div",
                    .binary_max => "max",
                    .binary_min => "min",
                    .binary_set => "set",
                    .linary_add => "add",
                    .linary_subtract => "sub",
                    .linary_multiply => "mul",
                    .linary_divide => "div",
                    .linary_max => "max",
                    .linary_min => "min",
                    .linary_set => "set",
                    .reduce_sum => "sum",
                    .reduce_max => "max",
                    .reduce_min => "min",
                    .reduce_avg => "avg",
                    else => unreachable,
                },
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
                this.in.name,
            });
        }
    }
    pub fn debug(this: *const @This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}{s} ", .{ " " ** (padding + offset), text });
        } else {
            std.debug.print("{s}", .{" " ** (padding + offset)});
        }
        if (this.isUnary()) {
            std.debug.print("U {c} ({d} {d} {d} {d}) [{d} {d} {d} {d} = {d}] \"{s}\" {d}\n", .{
                switch (this.type) {
                    .unary_add => "add",
                    .unary_subtract => "sub",
                    .unary_multiply => "mul",
                    .unary_divide => "div",
                    .unary_exp => "exp",
                    .unary_log => "log",
                    .unary_square => "sqr",
                    .unary_sqrt => "sqt",
                    .unary_reciprocal => "rcp",
                    .unary_max => "max",
                    .unary_min => "min",
                    .unary_set => "set",
                    .unary_random => "rng",
                    .unary_tanh => "tanh",
                    .unary_absolute => "abs",
                    .unary_sign => "sgn",
                    else => unreachable,
                },
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.a_offset,
                this.out.z_offset,
                this.out.y_offset,
                this.out.x_offset,
                this.out.offset,
                this.out.name,
                this.u_var,
            });
        } else {
            const op_kind: u8 = if (this.isBinary()) 'B' else (if (this.isLinary()) 'L' else 'R');
            std.debug.print("{c} {s} ({d} {d} {d} {d}) [{d} {d} {d} {d} = {d}] \"{s}\" ({d} {d} {d} {d}) [{d} {d} {d} {d} = {d}] \"{s}\"\n", .{
                op_kind,
                switch (this.type) {
                    .binary_add => "add",
                    .binary_subtract => "sub",
                    .binary_multiply => "mul",
                    .binary_divide => "div",
                    .binary_max => "max",
                    .binary_min => "min",
                    .binary_set => "set",
                    .linary_add => "add",
                    .linary_subtract => "sub",
                    .linary_multiply => "mul",
                    .linary_divide => "div",
                    .linary_max => "max",
                    .linary_min => "min",
                    .linary_set => "set",
                    .reduce_sum => "sum",
                    .reduce_max => "max",
                    .reduce_min => "min",
                    .reduce_avg => "avg",
                    else => unreachable,
                },
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.a_offset,
                this.out.z_offset,
                this.out.y_offset,
                this.out.x_offset,
                this.out.offset,
                this.out.name,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.a_offset,
                this.in.z_offset,
                this.in.y_offset,
                this.in.x_offset,
                this.in.offset,
                this.in.name,
            });
        }
    }
};

// TODO: There is probably a built-in way to do this
const op_cap_base: usize = 4;
pub const Linearized = struct {
    op: []Op,
    op_num: usize,
    pub fn alloc(allocator: anytype) !Linearized {
        return .{
            .op_num = 0,
            .op = try allocator.alloc(Op, op_cap_base),
        };
    }
    pub fn capacityEnsure(this: *@This(), allocator: anytype, capacity: usize) !void {
        if (this.op.len - this.op_num < capacity) {
            this.op = try allocator.realloc(this.op, this.op_num + capacity);
        }
    }
    pub fn free(this: *@This(), allocator: anytype) void {
        this.op_num = 0;
        allocator.free(this.op);
    }
    pub fn clear(this: *@This()) void {
        this.op_num = 0;
    }
    pub fn run(this: *@This()) void {
        for (0..this.op_num) |op_idx| {
            this.op[op_idx].realize();
        }
    }
    // TODO: Make a function that expands to the least power of 2 above some value
    // Double the capacity of this.op
    // fn expand(this: *@This(), allocator: anytype) !void {
    //     this.op = try allocator.realloc(this.op, this.op.len * 2);
    // }
    pub fn append(this: *@This(), op: *const Op) void {
        assert(this.op_num < this.op.len);
        this.op[this.op_num] = op.*;
        this.op_num += 1;
    }
    pub fn concat(this: *@This(), source: *Linearized) void {
        assert(this.op_num + source.op_num <= this.op.len);
        for (0..source.op_num) |op_idx| {
            this.op[this.op_num + op_idx] = source.op[op_idx];
        }
        this.op_num += source.op_num;
        source.clear();
    }
    pub fn print(this: *const @This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Linearized = {s}\n", .{ " " ** offset, text });
        } else {
            std.debug.print("{s}Linearized\n", .{" " ** offset});
        }
        if (this.op_num == 0) {
            std.debug.print("{s}[] => empty\n", .{" " ** (offset + padding)});
        } else {
            for (0..this.op_num) |op_idx| {
                std.debug.print("{s}[{}] => ", .{ " " ** (offset + padding), op_idx });
                this.op[op_idx].print(0, 0, null);
            }
        }
    }
    pub fn debug(this: *const @This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Linearized = {s}\n", .{ " " ** offset, text });
        } else {
            std.debug.print("{s}Linearized\n", .{" " ** offset});
        }
        if (this.op_num == 0) {
            std.debug.print("{s}[] => empty\n", .{" " ** (offset + padding)});
        } else {
            for (0..this.op_num) |op_idx| {
                std.debug.print("{s}[{}] => ", .{ " " ** (offset + padding), op_idx });
                this.op[op_idx].debug(0, 0, null);
            }
        }
    }
};

pub const Tensor = struct {
    buffer: Buffer,
    linearized: Linearized,
    pub fn alloc(allocator: anytype, a: usize, z: usize, y: usize, x: usize, context: ?ClContext) !Tensor {
        assert(a > 0);
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);

        return .{
            .buffer = try Buffer.alloc(allocator, a, z, y, x, context),
            .linearized = try Linearized.alloc(allocator),
        };
    }
    pub fn free(this: *@This(), allocator: anytype) void {
        this.buffer.free(allocator);
        this.linearized.free(allocator);
    }
    /// TODO: Decide if this should clear the linearized. On one hand it makes it so that you don't need to rebuild the linearized if you want to run it again
    /// However it is more intuitive that if you reailze a tensor that it should clear the linearized used to generate it
    /// Also if you run something more than once you should compile it I guess
    pub fn realize(this: *@This()) void {
        if (this.linearized.op_num != 0) {
            this.linearized.run();
            this.linearized.clear();
            this.buffer.syncUpdate(.sync_to_device);
        }
    }
    pub fn print(this: *const @This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Tensor {s} = {s}\n", .{ " " ** offset, this.buffer.name, text });
        } else {
            std.debug.print("{s}Tensor {s}\n", .{ " " ** offset, this.buffer.name });
        }
        for (0..this.buffer.a_size) |a| {
            for (0..this.buffer.z_size) |z| {
                for (0..this.buffer.y_size) |y| {
                    std.debug.print("{s}[", .{" " ** (offset + padding)});
                    for (0..this.buffer.x_size) |x| {
                        std.debug.print(" {d:8.4}", .{this.buffer.values[this.buffer.at(a, z, y, x)]});
                    }
                    std.debug.print("]\n", .{});
                }
                std.debug.print("\n", .{});
            }
            std.debug.print("\n", .{});
        }
    }
    pub fn debug(this: *const @This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Tensor {s} = {s}\n", .{ " " ** offset, this.buffer.name, text });
        } else {
            std.debug.print("{s}Tensor {s}\n", .{ " " ** offset, this.buffer.name });
        }
        this.linearized.debug(0, 0, null);
        for (0..this.buffer.a_size) |a| {
            for (0..this.buffer.z_size) |z| {
                for (0..this.buffer.y_size) |y| {
                    std.debug.print("{s}[", .{" " ** (offset + padding)});
                    for (0..this.buffer.x_size) |x| {
                        std.debug.print(" {d:8.4}", .{this.buffer.values[this.buffer.at(a, z, y, x)]});
                    }
                    std.debug.print("]\n", .{});
                }
                std.debug.print("\n", .{});
            }
            std.debug.print("\n", .{});
        }
    }
    pub fn unaryAdd(this: *@This(), u_var: f32) void {
        assert(!math.isNan(u_var));
        assert(!math.isInf(u_var));
        this.linearized.append(&.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_add,
            .u_var = u_var,
        });
    }
    pub fn unarySubtract(this: *@This(), u_var: f32) void {
        assert(!math.isNan(u_var));
        assert(!math.isInf(u_var));
        this.linearized.append(&.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_subtract,
            .u_var = u_var,
        });
    }
    pub fn unaryMultiply(this: *@This(), u_var: f32) void {
        assert(!math.isNan(u_var));
        assert(!math.isInf(u_var));
        this.linearized.append(&.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_multiply,
            .u_var = u_var,
        });
    }
    pub fn unaryDivide(this: *@This(), u_var: f32) void {
        assert(!math.isNan(u_var));
        assert(!math.isInf(u_var));
        this.linearized.append(&.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_divide,
            .u_var = u_var,
        });
    }
    pub fn unaryExp(this: *@This()) void {
        this.linearized.append(&.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_exp,
            .u_var = 0,
        });
    }
    pub fn unaryLog(this: *@This()) void {
        this.linearized.append(&.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_log,
            .u_var = 0,
        });
    }
    pub fn unarySquare(this: *@This()) void {
        this.linearized.append(&.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_square,
            .u_var = 0,
        });
    }
    pub fn unarySqrt(this: *@This()) void {
        this.linearized.append(&.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_sqrt,
            .u_var = 0,
        });
    }
    pub fn unaryReciprocal(this: *@This()) void {
        this.linearized.append(&.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_reciprocal,
            .u_var = 0,
        });
    }
    pub fn unaryMax(this: *@This(), u_var: f32) void {
        assert(!math.isNan(u_var));
        assert(!math.isInf(u_var));
        this.linearized.append(&.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_max,
            .u_var = u_var,
        });
    }
    pub fn unaryMin(this: *@This(), u_var: f32) void {
        assert(!math.isNan(u_var));
        assert(!math.isInf(u_var));
        this.linearized.append(&.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_min,
            .u_var = u_var,
        });
    }
    pub fn unarySet(this: *@This(), u_var: f32) void {
        assert(!math.isNan(u_var));
        assert(!math.isInf(u_var));
        this.linearized.append(&.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_set,
            .u_var = u_var,
        });
    }
    pub fn unaryRandom(this: *@This()) void {
        this.linearized.append(&.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_random,
            .u_var = 0,
        });
    }
    pub fn unaryTanh(this: *@This()) void {
        this.linearized.append(&.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_tanh,
            .u_var = 0,
        });
    }
    pub fn unaryAbsolute(this: *@This()) void {
        this.linearized.append(&.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_absolute,
            .u_var = 0,
        });
    }
    pub fn unarySign(this: *@This()) void {
        this.linearized.append(&.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_sign,
            .u_var = 0,
        });
    }
    pub fn binaryAdd(this: *@This(), source: *@This()) void {
        assert(this.buffer.a_size == source.buffer.a_size);
        assert(this.buffer.z_size == source.buffer.z_size);
        assert(this.buffer.y_size == source.buffer.y_size);
        assert(this.buffer.x_size == source.buffer.x_size);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .binary_add,
            .u_var = 0,
        });
    }
    pub fn binarySubtract(this: *@This(), source: *@This()) void {
        assert(this.buffer.a_size == source.buffer.a_size);
        assert(this.buffer.z_size == source.buffer.z_size);
        assert(this.buffer.y_size == source.buffer.y_size);
        assert(this.buffer.x_size == source.buffer.x_size);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .binary_subtract,
            .u_var = 0,
        });
    }
    pub fn binaryMultiply(this: *@This(), source: *@This()) void {
        assert(this.buffer.a_size == source.buffer.a_size);
        assert(this.buffer.z_size == source.buffer.z_size);
        assert(this.buffer.y_size == source.buffer.y_size);
        assert(this.buffer.x_size == source.buffer.x_size);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .binary_multiply,
            .u_var = 0,
        });
    }
    pub fn binaryDivide(this: *@This(), source: *@This()) void {
        assert(this.buffer.a_size == source.buffer.a_size);
        assert(this.buffer.z_size == source.buffer.z_size);
        assert(this.buffer.y_size == source.buffer.y_size);
        assert(this.buffer.x_size == source.buffer.x_size);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .binary_divide,
            .u_var = 0,
        });
    }
    pub fn binaryMax(this: *@This(), source: *@This()) void {
        assert(this.buffer.a_size == source.buffer.a_size);
        assert(this.buffer.z_size == source.buffer.z_size);
        assert(this.buffer.y_size == source.buffer.y_size);
        assert(this.buffer.x_size == source.buffer.x_size);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .binary_max,
            .u_var = 0,
        });
    }
    pub fn binaryMin(this: *@This(), source: *@This()) void {
        assert(this.buffer.a_size == source.buffer.a_size);
        assert(this.buffer.z_size == source.buffer.z_size);
        assert(this.buffer.y_size == source.buffer.y_size);
        assert(this.buffer.x_size == source.buffer.x_size);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .binary_min,
            .u_var = 0,
        });
    }
    pub fn binarySet(this: *@This(), source: *@This()) void {
        assert(this.buffer.a_size == source.buffer.a_size);
        assert(this.buffer.z_size == source.buffer.z_size);
        assert(this.buffer.y_size == source.buffer.y_size);
        assert(this.buffer.x_size == source.buffer.x_size);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .binary_set,
            .u_var = 0,
        });
    }
    pub fn linaryAdd(this: *@This(), source: *@This()) void {
        assert(source.buffer.a_size == 1);
        assert(source.buffer.z_size == 1);
        assert(source.buffer.y_size == 1);
        assert(source.buffer.x_size == 1);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .linary_add,
            .u_var = 0,
        });
    }
    pub fn linarySubtract(this: *@This(), source: *@This()) void {
        assert(source.buffer.a_size == 1);
        assert(source.buffer.z_size == 1);
        assert(source.buffer.y_size == 1);
        assert(source.buffer.x_size == 1);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .linary_subtract,
            .u_var = 0,
        });
    }
    pub fn linaryMultiply(this: *@This(), source: *@This()) void {
        assert(source.buffer.a_size == 1);
        assert(source.buffer.z_size == 1);
        assert(source.buffer.y_size == 1);
        assert(source.buffer.x_size == 1);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .linary_multiply,
            .u_var = 0,
        });
    }
    pub fn linaryDivide(this: *@This(), source: *@This()) void {
        assert(source.buffer.a_size == 1);
        assert(source.buffer.z_size == 1);
        assert(source.buffer.y_size == 1);
        assert(source.buffer.x_size == 1);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .linary_divide,
            .u_var = 0,
        });
    }
    pub fn linaryMax(this: *@This(), source: *@This()) void {
        assert(source.buffer.a_size == 1);
        assert(source.buffer.z_size == 1);
        assert(source.buffer.y_size == 1);
        assert(source.buffer.x_size == 1);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .linary_max,
            .u_var = 0,
        });
    }
    pub fn linaryMin(this: *@This(), source: *@This()) void {
        assert(source.buffer.a_size == 1);
        assert(source.buffer.z_size == 1);
        assert(source.buffer.y_size == 1);
        assert(source.buffer.x_size == 1);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .linary_min,
            .u_var = 0,
        });
    }
    pub fn linarySet(this: *@This(), source: *@This()) void {
        assert(source.buffer.a_size == 1);
        assert(source.buffer.z_size == 1);
        assert(source.buffer.y_size == 1);
        assert(source.buffer.x_size == 1);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .linary_set,
            .u_var = 0,
        });
    }
    pub fn reduceSum(this: *@This(), source: *@This()) void {
        assert(this.buffer.a_size == 1);
        assert(this.buffer.z_size == 1);
        assert(this.buffer.y_size == 1);
        assert(this.buffer.x_size == 1);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .reduce_sum,
            .u_var = 0,
        });
    }
    pub fn reduceMax(this: *@This(), source: *@This()) void {
        assert(this.buffer.a_size == 1);
        assert(this.buffer.z_size == 1);
        assert(this.buffer.y_size == 1);
        assert(this.buffer.x_size == 1);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .reduce_max,
            .u_var = 0,
        });
    }
    pub fn reduceMin(this: *@This(), source: *@This()) void {
        assert(this.buffer.a_size == 1);
        assert(this.buffer.z_size == 1);
        assert(this.buffer.y_size == 1);
        assert(this.buffer.x_size == 1);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .reduce_min,
            .u_var = 0,
        });
    }
    pub fn reduceAvg(this: *@This(), source: *@This()) void {
        assert(this.buffer.a_size == 1);
        assert(this.buffer.z_size == 1);
        assert(this.buffer.y_size == 1);
        assert(this.buffer.x_size == 1);
        this.linearized.concat(&source.linearized);
        this.linearized.append(&.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .reduce_avg,
            .u_var = 0,
        });
    }
    pub fn moveReshape(this: *@This(), a: usize, z: usize, y: usize, x: usize) void {
        assert(a > 0);
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);
        assert(a <= this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        assert(z <= this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        assert(y <= this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        assert(x <= this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        this.buffer.a_size = a;
        this.buffer.z_size = z;
        this.buffer.y_size = y;
        this.buffer.x_size = x;
        this.buffer.a_stride = z * y * x;
        this.buffer.z_stride = y * x;
        this.buffer.y_stride = x;
        this.buffer.x_stride = 1;
    }
    pub fn moveResize(this: *@This(), a: usize, z: usize, y: usize, x: usize) void {
        assert(a > 0);
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);
        assert(a <= this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        assert(z <= this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        assert(y <= this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        assert(x <= this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        this.buffer.a_size = a;
        this.buffer.z_size = z;
        this.buffer.y_size = y;
        this.buffer.x_size = x;
    }
    pub fn moveOffset(this: *@This(), a: usize, z: usize, y: usize, x: usize) void {
        assert(a < this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        assert(z < this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        assert(y < this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        assert(x < this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        this.buffer.offset = a * this.buffer.a_stride + z * this.buffer.z_stride + y * this.buffer.y_stride + x * this.buffer.x_stride;
        this.buffer.a_offset = a;
        this.buffer.z_offset = z;
        this.buffer.y_offset = y;
        this.buffer.x_offset = x;
    }
    pub fn dependOn(this: *@This(), prerequisite: *@This()) void {
        this.linearized.concat(&prerequisite.linearized);
    }
};
