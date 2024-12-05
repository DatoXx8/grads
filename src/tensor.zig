const std = @import("std");
const math = std.math;

const Pcg = @import("./prng.zig").Pcg;

const assert = @import("./util.zig").assert;

const ClMem = @import("./runtimes/cl.zig").ClMem;
const ClContext = @import("./runtimes/cl.zig").ClContext;
const ClCommandQueue = @import("./runtimes/cl.zig").ClCommandQueue;
const OpenCl = @import("./runtimes/cl.zig").opencl;

// TODO: Get rid of this anytype bs. That is downright horrible imo.
// TODO: Split the file up more?

/// 4 is probably already enough. 26 ^ 4 = 456.976
/// 8 is absolute overkill. 26 ^ 8 = 208.827.064.576
pub const buffer_name_size: u32 = 8;
var buffer_name_offset: u32 = 0;

const Buffer = struct {
    const SyncStatus = enum(u8) {
        sync_to_host,
        sync_to_device,
        sync_to_none,
    };
    const SyncError = error{
        FailedToHost,
        FailedToDevice,
    };
    a_inherent: u32,
    z_inherent: u32,
    y_inherent: u32,
    x_inherent: u32,
    a_size: u32,
    z_size: u32,
    y_size: u32,
    x_size: u32,
    a_stride: u32,
    z_stride: u32,
    y_stride: u32,
    x_stride: u32,
    a_offset: u32,
    z_offset: u32,
    y_offset: u32,
    x_offset: u32,
    offset: u32,
    values: []f32,
    values_cl: ?ClMem,
    sync: SyncStatus,
    name: [buffer_name_size]u8,
    name_offset: u32,
    pub fn alloc(allocator: anytype, a: u32, z: u32, y: u32, x: u32, context: ?ClContext) !Buffer {
        assert(a > 0);
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);

        var name: [buffer_name_size]u8 = [_]u8{'a'} ** buffer_name_size;
        const divisor: u64 = 26;
        var left: u32 = buffer_name_offset;
        for (0..buffer_name_size) |char_idx| {
            name[char_idx] += @truncate(left % divisor);
            left = @truncate(left / divisor);
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
                .z_offset = z,
                .y_offset = y,
                .x_offset = x,
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
                .z_offset = z,
                .y_offset = y,
                .x_offset = x,
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
    pub fn sync_to_host(this: *@This(), command_queue: ClCommandQueue) !void {
        assert(this.sync == .sync_to_host);
        const size: u32 = this.a_inherent * this.z_inherent * this.y_inherent * this.x_inherent * @sizeOf(f32);
        const err: i32 = OpenCl.clEnqueueReadBuffer(command_queue.queue, this.values_cl.?.memory, //
            OpenCl.CL_TRUE, 0, size, this.values.ptr, 0, null, null);
        if (err != 0) {
            return SyncError.FailedToHost;
        }
        this.sync = .sync_to_none;
    }
    pub fn sync_to_device(this: *@This(), command_queue: ClCommandQueue) !void {
        assert(this.sync == .sync_to_device);
        const size: u32 = this.a_inherent * this.z_inherent * this.y_inherent * this.x_inherent * @sizeOf(f32);
        const err: i32 = OpenCl.clEnqueueWriteBuffer(command_queue.queue, this.values_cl.?.memory, //
            OpenCl.CL_TRUE, 0, size, this.values.ptr, 0, null, null);
        if (err != 0) {
            return SyncError.FailedToDevice;
        }
        this.sync = .sync_to_none;
    }
    pub fn sync_update(this: *@This(), sync: SyncStatus) void {
        assert(this.sync == .sync_to_none);
        assert(sync != .sync_to_none);
        this.sync = sync;
    }
    pub fn sync_wait(command_queue: ClCommandQueue) void {
        OpenCl.clFinish(command_queue.queue);
    }
};
// TODO: Maybe truncate the names to 3 letters each
pub const Op = struct {
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
        linary_add, // This is like binary but the in buffer has size [1, 1, 1, 1]
        linary_subtract, // This is like binary but the in buffer has size [1, 1, 1, 1]
        linary_multiply, // This is like binary but the in buffer has size [1, 1, 1, 1]
        linary_divide, // This is like binary but the in buffer has size [1, 1, 1, 1]
        linary_max, // This is like binary but the in buffer has size [1, 1, 1, 1]
        linary_min, // This is like binary but the in buffer has size [1, 1, 1, 1]
        linary_set, // This is like binary but the in buffer has size [1, 1, 1, 1]
        reduce_sum,
        reduce_max,
        reduce_avg,
        reduce_min,
    };
    type: Type,
    u_var: f32,
    // TODO: Probably don't need to save the whole Buffer struct here
    // Save the pointers to the values and just save the offset and strides?
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

        const a_1: u32 = this.out.a_size;
        const z_1: u32 = this.out.z_size;
        const y_1: u32 = this.out.y_size;
        const x_1: u32 = this.out.x_size;

        const a_2: u32 = target.out.a_size;
        const z_2: u32 = target.out.z_size;
        const y_2: u32 = target.out.y_size;
        const x_2: u32 = target.out.x_size;

        return @max(a_1, a_2) - @min(a_1, a_2) < this.out.a_size and
            @max(z_1, z_2) - @min(z_1, z_2) < this.out.z_size and
            @max(y_1, y_2) - @min(y_1, y_2) < this.out.y_size and
            @max(x_1, x_2) - @min(x_1, x_2) < this.out.x_size;
    }
    pub inline fn is_unary(this: *const @This()) bool {
        return this.type == .unary_add or this.type == .unary_subtract or
            this.type == .unary_multiply or this.type == .unary_divide or
            this.type == .unary_exp or this.type == .unary_log or
            this.type == .unary_square or this.type == .unary_sqrt or
            this.type == .unary_reciprocal or this.type == .unary_max or
            this.type == .unary_min or this.type == .unary_set or
            this.type == .unary_random or this.type == .unary_tanh or
            this.type == .unary_absolute or this.type == .unary_sign;
    }
    pub inline fn is_binary(this: *const @This()) bool {
        return this.type == .binary_add or this.type == .binary_subtract or
            this.type == .binary_multiply or this.type == .binary_divide or
            this.type == .binary_max or this.type == .binary_min or
            this.type == .binary_set;
    }
    pub inline fn is_linary(this: *const @This()) bool {
        return this.type == .linary_add or this.type == .linary_subtract or
            this.type == .linary_multiply or this.type == .linary_divide or
            this.type == .linary_max or this.type == .linary_min or
            this.type == .linary_set;
    }
    pub inline fn is_reduce(this: *const @This()) bool {
        return this.type == .reduce_sum or this.type == .reduce_max or
            this.type == .reduce_avg or this.type == .reduce_min;
    }
    // TODO: Optimise this with simd, see @Vector
    pub fn realize(this: *const @This()) void {
        if (this.is_unary()) {
            // In buffer is just a copy of out buffer, basically just a sanity check.
            assert(this.out.a_size == this.in.a_size);
            assert(this.out.z_size == this.in.z_size);
            assert(this.out.y_size == this.in.y_size);
            assert(this.out.x_size == this.in.x_size);
        } else if (this.is_binary()) {
            assert(this.out.a_size == this.in.a_size);
            assert(this.out.z_size == this.in.z_size);
            assert(this.out.y_size == this.in.y_size);
            assert(this.out.x_size == this.in.x_size);
        } else if (this.is_linary()) {
            assert(this.in.a_size == 1);
            assert(this.in.z_size == 1);
            assert(this.in.y_size == 1);
            assert(this.in.x_size == 1);
        } else if (this.is_reduce()) {
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
                                this.out.values[this.out.at(a, z, y, x)] = Pcg.rand_f32();
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
    // Really unhappy about this anytype thing...
    pub fn print(this: *const @This(), writer: anytype, comptime padding: u32, comptime offset: u32, name: ?[]u8) !void {
        if (name) |text| {
            try writer.print("{s}{s} ", .{ " " ** (padding + offset), text });
        } else {
            try writer.print("{s}", .{" " ** (padding + offset)});
        }
        switch (this.type) {
            .unary_add => try writer.print("U add ({d} {d} {d} {d}) [{d}] \"{s}\" {d}\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
                this.u_var,
            }),
            .unary_subtract => try writer.print("U sub ({d} {d} {d} {d}) [{d}] \"{s}\" {d}\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
                this.u_var,
            }),
            .unary_multiply => try writer.print("U mul ({d} {d} {d} {d}) [{d}] \"{s}\" {d}\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
                this.u_var,
            }),
            .unary_divide => try writer.print("U div ({d} {d} {d} {d}) [{d}] \"{s}\" {d}\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
                this.u_var,
            }),
            .unary_exp => try writer.print("U exp ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
            }),
            .unary_log => try writer.print("U log ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
            }),
            .unary_square => try writer.print("U sqr ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
            }),
            .unary_sqrt => try writer.print("U sqt ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
            }),
            .unary_reciprocal => try writer.print("U rec ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
            }),
            .unary_max => try writer.print("U max ({d} {d} {d} {d}) [{d}] \"{s}\" {d}\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
                this.u_var,
            }),
            .unary_min => try writer.print("U min ({d} {d} {d} {d}) [{d}] \"{s}\" {d}\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
                this.u_var,
            }),
            .unary_set => try writer.print("U set ({d} {d} {d} {d}) [{d}] \"{s}\" {d}\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
                this.u_var,
            }),
            .unary_random => try writer.print("U rng ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
            }),
            .unary_tanh => try writer.print("U tan ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
            }),
            .unary_absolute => try writer.print("U abs ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
            }),
            .unary_sign => try writer.print("U sgn ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.out.name,
            }),
            .binary_add => try writer.print("B add ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
            .binary_subtract => try writer.print("B sub ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
            .binary_multiply => try writer.print("B mul ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
            .binary_divide => try writer.print("B div ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
            .binary_max => try writer.print("B max ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
            .binary_min => try writer.print("B min ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
            .binary_set => try writer.print("B set ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
            .linary_add => try writer.print("L add ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
            .linary_subtract => try writer.print("L sub ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
            .linary_multiply => try writer.print("L mul ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
            .linary_divide => try writer.print("L div ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
            .linary_max => try writer.print("L max ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
            .linary_min => try writer.print("L min ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
            .linary_set => try writer.print("L set ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
            .reduce_sum => try writer.print("R sum ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
            .reduce_max => try writer.print("R max ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
            .reduce_min => try writer.print("R min ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
            .reduce_avg => try writer.print("R avg ({d} {d} {d} {d}) [{d}] \"{s}\" ({d} {d} {d} {d}) [{d}] \"{s}\"\n", .{
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
            }),
        }
    }
};

// TODO: There is probably a built-in way to do this
const op_cap_base: u32 = 4;
pub const Linearized = struct {
    /// Capacity is op.len
    op: []Op,
    op_num: usize,
    pub fn alloc(allocator: anytype) !Linearized {
        const linearized: Linearized = .{
            .op_num = 0,
            .op = try allocator.alloc(Op, op_cap_base),
        };
        return linearized;
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
    fn expand(this: *@This(), allocator: anytype) !void {
        this.op = try allocator.realloc(this.op, this.op.len * 2);
    }
    pub fn append(this: *@This(), allocator: anytype, op: *const Op) !void {
        if (this.op_num == this.op.len - 1) {
            try this.expand(allocator);
        }
        this.op[this.op_num] = op.*;
        this.op_num += 1;
    }
    pub fn concat(this: *@This(), allocator: anytype, source: *Linearized) !void {
        while (this.op_num + source.op_num >= this.op.len) {
            try this.expand(allocator);
        }
        for (0..source.op_num) |op_idx| {
            this.op[this.op_num + op_idx] = source.op[op_idx];
        }
        this.op_num += source.op_num;
        source.clear();
    }
    pub fn print(this: *const @This(), writer: anytype, comptime padding: u32, comptime offset: u32, name: ?[]u8) !void {
        if (name) |text| {
            try writer.print("{s}Linearized = {s}\n", .{ " " ** offset, text });
        } else {
            try writer.print("{s}Linearized\n", .{" " ** offset});
        }
        if (this.op_num == 0) {
            try writer.print("{s}[] => empty\n", .{" " ** (offset + padding)});
        } else {
            for (0..this.op_num) |op_idx| {
                try writer.print("{s}[{}] => ", .{ " " ** (offset + padding), op_idx });
                try this.op[op_idx].print(writer, 0, 0, null);
            }
        }
    }
};

pub const Tensor = struct {
    buffer: Buffer,
    linearized: Linearized,
    pub fn alloc(allocator: anytype, a: u32, z: u32, y: u32, x: u32, context: ?ClContext) !Tensor {
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
        // this.buffer = null;
        // this.linearized = null;
    }
    /// TODO: Decide if this should clear the linearized. On one hand it makes it so that you don't need to rebuild the linearized if you want to run it again
    /// However it is more intuitive that if you reailze a tensor that it should clear the linearized used to generate it
    /// Also if you run something more than once you should compile it I guess
    pub fn realize(this: *@This()) void {
        if (this.linearized.op_num != 0) {
            this.linearized.run();
            this.linearized.clear();
            // TODO: This should update sync to `snyc_to_host`
        }
    }
    pub fn print(this: *const @This(), writer: anytype, comptime padding: u32, comptime offset: u32, name: ?[]u8) !void {
        if (name) |text| {
            try writer.print("{s}Tensor {s} = {s}\n", .{ " " ** offset, this.buffer.name, text });
        } else {
            try writer.print("{s}Tensor {s}\n", .{ " " ** offset, this.buffer.name });
        }
        for (0..this.buffer.a_size) |a| {
            for (0..this.buffer.z_size) |z| {
                for (0..this.buffer.y_size) |y| {
                    try writer.print("{s}[", .{" " ** (offset + padding)});
                    for (0..this.buffer.x_size) |x| {
                        try writer.print(" {d:8.4}", .{this.buffer.values[this.buffer.at(a, z, y, x)]});
                    }
                    try writer.print("]\n", .{});
                }
                try writer.print("\n", .{});
            }
            try writer.print("\n", .{});
        }
    }
    pub fn unary_add(this: *@This(), allocator: anytype, u_var: f32) !void {
        assert(!math.isNan(u_var));
        assert(!math.isInf(u_var));
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_add,
            .u_var = u_var,
        });
    }
    pub fn unary_subtract(this: *@This(), allocator: anytype, u_var: f32) !void {
        assert(!math.isNan(u_var));
        assert(!math.isInf(u_var));
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_subtract,
            .u_var = u_var,
        });
    }
    pub fn unary_multiply(this: *@This(), allocator: anytype, u_var: f32) !void {
        assert(!math.isNan(u_var));
        assert(!math.isInf(u_var));
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_multiply,
            .u_var = u_var,
        });
    }
    pub fn unary_divide(this: *@This(), allocator: anytype, u_var: f32) !void {
        assert(!math.isNan(u_var));
        assert(!math.isInf(u_var));
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_divide,
            .u_var = u_var,
        });
    }
    pub fn unary_exp(this: *@This(), allocator: anytype) !void {
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_exp,
            .u_var = 0,
        });
    }
    pub fn unary_log(this: *@This(), allocator: anytype) !void {
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_log,
            .u_var = 0,
        });
    }
    pub fn unary_square(this: *@This(), allocator: anytype) !void {
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_square,
            .u_var = 0,
        });
    }
    pub fn unary_sqrt(this: *@This(), allocator: anytype) !void {
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_sqrt,
            .u_var = 0,
        });
    }
    pub fn unary_reciprocal(this: *@This(), allocator: anytype) !void {
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_reciprocal,
            .u_var = 0,
        });
    }
    pub fn unary_max(this: *@This(), allocator: anytype, u_var: f32) !void {
        assert(!math.isNan(u_var));
        assert(!math.isInf(u_var));
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_max,
            .u_var = u_var,
        });
    }
    pub fn unary_min(this: *@This(), allocator: anytype, u_var: f32) !void {
        assert(!math.isNan(u_var));
        assert(!math.isInf(u_var));
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_min,
            .u_var = u_var,
        });
    }
    pub fn unary_set(this: *@This(), allocator: anytype, u_var: f32) !void {
        assert(!math.isNan(u_var));
        assert(!math.isInf(u_var));
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_set,
            .u_var = u_var,
        });
    }
    pub fn unary_random(this: *@This(), allocator: anytype) !void {
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_random,
            .u_var = 0,
        });
    }
    pub fn unary_tanh(this: *@This(), allocator: anytype) !void {
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_tanh,
            .u_var = 0,
        });
    }
    pub fn unary_absolute(this: *@This(), allocator: anytype) !void {
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_absolute,
            .u_var = 0,
        });
    }
    pub fn unary_sign(this: *@This(), allocator: anytype) !void {
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_sign,
            .u_var = 0,
        });
    }
    pub fn binary_add(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(this.buffer.a_size == source.buffer.a_size);
        assert(this.buffer.z_size == source.buffer.z_size);
        assert(this.buffer.y_size == source.buffer.y_size);
        assert(this.buffer.x_size == source.buffer.x_size);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .binary_add,
            .u_var = 0,
        });
    }
    pub fn binary_subtract(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(this.buffer.a_size == source.buffer.a_size);
        assert(this.buffer.z_size == source.buffer.z_size);
        assert(this.buffer.y_size == source.buffer.y_size);
        assert(this.buffer.x_size == source.buffer.x_size);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .binary_subtract,
            .u_var = 0,
        });
    }
    pub fn binary_multiply(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(this.buffer.a_size == source.buffer.a_size);
        assert(this.buffer.z_size == source.buffer.z_size);
        assert(this.buffer.y_size == source.buffer.y_size);
        assert(this.buffer.x_size == source.buffer.x_size);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .binary_multiply,
            .u_var = 0,
        });
    }
    pub fn binary_divide(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(this.buffer.a_size == source.buffer.a_size);
        assert(this.buffer.z_size == source.buffer.z_size);
        assert(this.buffer.y_size == source.buffer.y_size);
        assert(this.buffer.x_size == source.buffer.x_size);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .binary_divide,
            .u_var = 0,
        });
    }
    pub fn binary_max(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(this.buffer.a_size == source.buffer.a_size);
        assert(this.buffer.z_size == source.buffer.z_size);
        assert(this.buffer.y_size == source.buffer.y_size);
        assert(this.buffer.x_size == source.buffer.x_size);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .binary_max,
            .u_var = 0,
        });
    }
    pub fn binary_min(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(this.buffer.a_size == source.buffer.a_size);
        assert(this.buffer.z_size == source.buffer.z_size);
        assert(this.buffer.y_size == source.buffer.y_size);
        assert(this.buffer.x_size == source.buffer.x_size);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .binary_min,
            .u_var = 0,
        });
    }
    pub fn binary_set(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(this.buffer.a_size == source.buffer.a_size);
        assert(this.buffer.z_size == source.buffer.z_size);
        assert(this.buffer.y_size == source.buffer.y_size);
        assert(this.buffer.x_size == source.buffer.x_size);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .binary_set,
            .u_var = 0,
        });
    }
    pub fn linary_add(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(source.buffer.a_size == 1);
        assert(source.buffer.z_size == 1);
        assert(source.buffer.y_size == 1);
        assert(source.buffer.x_size == 1);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .linary_add,
            .u_var = 0,
        });
    }
    pub fn linary_subtract(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(source.buffer.a_size == 1);
        assert(source.buffer.z_size == 1);
        assert(source.buffer.y_size == 1);
        assert(source.buffer.x_size == 1);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .linary_subtract,
            .u_var = 0,
        });
    }
    pub fn linary_multiply(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(source.buffer.a_size == 1);
        assert(source.buffer.z_size == 1);
        assert(source.buffer.y_size == 1);
        assert(source.buffer.x_size == 1);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .linary_multiply,
            .u_var = 0,
        });
    }
    pub fn linary_divide(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(source.buffer.a_size == 1);
        assert(source.buffer.z_size == 1);
        assert(source.buffer.y_size == 1);
        assert(source.buffer.x_size == 1);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .linary_divide,
            .u_var = 0,
        });
    }
    pub fn linary_max(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(source.buffer.a_size == 1);
        assert(source.buffer.z_size == 1);
        assert(source.buffer.y_size == 1);
        assert(source.buffer.x_size == 1);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .linary_max,
            .u_var = 0,
        });
    }
    pub fn linary_min(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(source.buffer.a_size == 1);
        assert(source.buffer.z_size == 1);
        assert(source.buffer.y_size == 1);
        assert(source.buffer.x_size == 1);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .linary_min,
            .u_var = 0,
        });
    }
    pub fn linary_set(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(source.buffer.a_size == 1);
        assert(source.buffer.z_size == 1);
        assert(source.buffer.y_size == 1);
        assert(source.buffer.x_size == 1);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .linary_set,
            .u_var = 0,
        });
    }
    pub fn reduce_sum(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(this.buffer.a_size == 1);
        assert(this.buffer.z_size == 1);
        assert(this.buffer.y_size == 1);
        assert(this.buffer.x_size == 1);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .reduce_sum,
            .u_var = 0,
        });
    }
    pub fn reduce_max(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(this.buffer.a_size == 1);
        assert(this.buffer.z_size == 1);
        assert(this.buffer.y_size == 1);
        assert(this.buffer.x_size == 1);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .reduce_max,
            .u_var = 0,
        });
    }
    pub fn reduce_min(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(this.buffer.a_size == 1);
        assert(this.buffer.z_size == 1);
        assert(this.buffer.y_size == 1);
        assert(this.buffer.x_size == 1);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .reduce_min,
            .u_var = 0,
        });
    }
    pub fn reduce_avg(this: *@This(), allocator: anytype, source: *@This()) !void {
        assert(this.buffer.a_size == 1);
        assert(this.buffer.z_size == 1);
        assert(this.buffer.y_size == 1);
        assert(this.buffer.x_size == 1);
        try this.linearized.concat(allocator, &source.linearized);
        try this.linearized.append(allocator, &.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .reduce_avg,
            .u_var = 0,
        });
    }
    pub fn move_reshape(this: *@This(), a: u32, z: u32, y: u32, x: u32) void {
        assert(a > 0);
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);
        assert(a < this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        assert(z < this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        assert(y < this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        assert(x < this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        this.buffer.a_size = a;
        this.buffer.z_size = z;
        this.buffer.y_size = y;
        this.buffer.x_size = x;
        this.buffer.a_stride = z * y * x;
        this.buffer.z_stride = y * x;
        this.buffer.y_stride = x;
        this.buffer.x_stride = 1;
    }
    pub fn move_resize(this: *@This(), a: u32, z: u32, y: u32, x: u32) void {
        assert(a > 0);
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);
        assert(a < this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        assert(z < this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        assert(y < this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        assert(x < this.buffer.a_inherent * this.buffer.z_inherent * this.buffer.y_inherent * this.buffer.x_inherent);
        this.buffer.a_size = a;
        this.buffer.z_size = z;
        this.buffer.y_size = y;
        this.buffer.x_size = x;
    }
    pub fn move_offset(this: *@This(), a: u32, z: u32, y: u32, x: u32) void {
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
};
