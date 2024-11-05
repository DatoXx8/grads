const std = @import("std");
const math = std.math;
const assert = std.debug.assert;

// TODO: Init and free functions

const SyncStatus = enum(u8) {
    sync_to_host,
    snyc_to_device,
    sync_to_none,
};

const buffer_name_size: u32 = 16;
var buffer_name_offset: u32 = 0;

const Buffer = struct {
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
    offset: u32,
    values: ?[]f32,
    // values_cl: ClMem,
    sync: SyncStatus,
    name: [buffer_name_size]u8,
    name_offset: u32,
    pub fn alloc(allocator: std.mem.Allocator, a: u32, z: u32, y: u32, x: u32) Buffer {
        assert(a);
        assert(z);
        assert(y);
        assert(x);

        var name: [buffer_name_size]u8 = ['a']**buffer_name_size;
        var mod: u32 = 26;
        for (0..buffer_name_size) |char_idx| {
            name[char_idx] += buffer_name_offset % mod;
            mod *= 26;
        }

        const buffer: Buffer = .{
            .name_offset = buffer_name_offset,
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
            .values = allocator.alloc(f32, a * z * y * x) orelse null,
        };

        buffer_name_offset += 1;

        return buffer;
    }
    pub fn free(this: *@This(), allocator: std.mem.Allocator) void {
        assert(this);
        assert(this.values);

        allocator.free(this.values);
        this.values = null;
    }
    pub fn at(this: *@This(), a: u32, z: u32, y: u32, x: u32) u32 {
        return this.offset + a * this.a_stride + z * this.z_stride + y * this.y_stride + x * this.x_stride;
    }
    // pub fn sync_start
    // pub fn sync_wait
};
const Op = struct {
    type: enum(u8) {
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
    },
    u_var: f32,
    // TODO: Probably don't need to save the whole Buffer struct here
    out: Buffer,
    in: Buffer,
    pub fn realize(this: *@This()) void {
        assert(this);
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
                                // TODO: I don't think this is ln
                                this.out.values[this.out.at(a, z, y, x)] = math.log(this.out.values[this.out.at(a, z, y, x)]);
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
                                // Float with mean = 0 and std deviation = 1
                                this.out.values[this.out.at(a, z, y, x)] = std.crypto.random.floatNorm(f32);
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
                this.out.values[this.out.at(0, 0, 0, 0)] /= @as(f32, this.in.a_size * this.in.z_size * this.in.y_size * this.in.x_size);
            },
        }
    }
    // Really unhappy about this anytype thing...
    pub fn print(this: *@This(), writer: anytype, comptime padding: u32, comptime offset: u32, name: ?[]u8) void {
        if (name) |text| {
            try writer.print("{s}{s} ", .{ " " ** (padding + offset), text });
        } else {
            try writer.print("{s}", .{" " ** (padding + offset)});
        }
        switch (this.type) {
            .unary_add => try writer.print("U add ({d} {d} {d} {d}) [{d}] {d}\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.u_var,
            }),
            .unary_subtract => try writer.print("U sub ({d} {d} {d} {d}) [{d}] {d}\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.u_var,
            }),
            .unary_multiply => try writer.print("U mul ({d} {d} {d} {d}) [{d}] {d}\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.u_var,
            }),
            .unary_divide => try writer.print("U div ({d} {d} {d} {d}) [{d}] {d}\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.u_var,
            }),
            .unary_exp => try writer.print("U exp ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
            }),
            .unary_log => try writer.print("U log ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
            }),
            .unary_square => try writer.print("U sqr ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
            }),
            .unary_sqrt => try writer.print("U sqt ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
            }),
            .unary_reciprocal => try writer.print("U rec ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
            }),
            .unary_max => try writer.print("U max ({d} {d} {d} {d}) [{d}] {d}\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.u_var,
            }),
            .unary_min => try writer.print("U min ({d} {d} {d} {d}) [{d}] {d}\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.u_var,
            }),
            .unary_set => try writer.print("U set ({d} {d} {d} {d}) [{d}] {d}\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.u_var,
            }),
            .unary_random => try writer.print("U rng ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
            }),
            .unary_tanh => try writer.print("U tan ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
            }),
            .unary_absolute => try writer.print("U abs ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
            }),
            .unary_sign => try writer.print("U sgn ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
            }),
            .binary_add => try writer.print("B add ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
            .binary_subtract => try writer.print("B sub ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
            .binary_multiply => try writer.print("B mul ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
            .binary_divide => try writer.print("B div ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
            .binary_max => try writer.print("B max ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
            .binary_min => try writer.print("B min ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
            .binary_set => try writer.print("B set ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
            .linary_add => try writer.print("L add ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
            .linary_subtract => try writer.print("L sub ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
            .linary_multiply => try writer.print("L mul ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
            .linary_divide => try writer.print("L div ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
            .linary_max => try writer.print("L max ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
            .linary_min => try writer.print("L min ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
            .linary_set => try writer.print("L set ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
            .reduce_sum => try writer.print("R sum ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
            .reduce_max => try writer.print("R max ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
            .reduce_min => try writer.print("R min ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
            .reduce_avg => try writer.print("R avg ({d} {d} {d} {d}) [{d}] ({d} {d} {d} {d}) [{d}]\n", .{
                this.out.a_size,
                this.out.z_size,
                this.out.y_size,
                this.out.x_size,
                this.out.offset,
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.offset,
            }),
        }
    }
};

// TODO: There is probably a built-in way to do this
const op_cap_base: u32 = 4;
const Linearized = struct {
    op: ?[]Op,
    op_num: u32,
    // Equivalent to op.len
    // TODO: Remove this eventually
    op_cap: u32,
    pub fn alloc(allocator: std.mem.Allocator) Linearized {
        const linearized: Linearized = .{
            .op_num = 0,
            .op_cap = 0,
            .op = allocator.alloc(f32, op_cap_base),
        };
        return linearized;
    }
    pub fn free(this: *@This(), allocator: std.mem.Allocator) void {
        assert(this);
        assert(this.op);
        assert(this.op_cap);

        this.op_num = 0;
        this.op_cap = 0;
        allocator.free(this.op);
        this.op = null;
    }
    pub fn clear(this: *@This()) void {
        assert(this);
        assert(this.op);
        assert(this.op_cap);

        this.op_num = 0;
    }
    pub fn run(this: *@This()) void {
        assert(this);
        assert(this.op);
        for (0..this.op_num) |op_idx| {
            this.op[op_idx].realize();
        }
    }
    // TODO: Make a function that expands to the least power of 2 above some value
    // Double the capacity of this.op
    fn expand(this: *@This(), allocator: std.mem.Allocator) void {
        assert(this);
        assert(this.op);
        assert(this.op_cap);

        this.op_cap *= 2;
        this.op = allocator.realloc(this.op, this.op_cap);
    }
    pub fn append(this: *@This(), allocator: std.mem.Allocator, op: *Op) void {
        assert(this);
        assert(op);
        assert(this.op);

        if (this.op_num == this.op_cap - 1) {
            this.expand(allocator);
        }
        this.op[this.op_num] = op;
        this.op_num += 1;
    }
    pub fn concat(this: *@This(), allocator: std.mem.Allocator, source: *Linearized) void {
        assert(this);
        assert(this.op);
        assert(source);
        assert(source.op);

        // This effectively means that the max growth factor is 2^20 = 1_048_576
        const max_expand_tries = 20;
        for (0..max_expand_tries) |_| {
            if (this.op_num + source.op_num < this.op_cap) {
                break;
            } else {
                this.expand(allocator);
            }
        }
        assert(this.op_num + source.op_num < this.op_cap);
        for (0..source.op_num) |op_idx| {
            this.op[this.op_num + op_idx] = source.op[op_idx];
        }
        this.op_num += source.op_num;
    }
};

pub const Tensor = struct {
    buffer: Buffer,
    linearized: Linearized,
    pub fn alloc(allocator: std.mem.Allocator, a: u32, z: u32, y: u32, x: u32) Tensor {
        assert(a);
        assert(z);
        assert(y);
        assert(x);

        const tensor: Tensor = .{
            .buffer = Buffer.alloc(allocator, a, z, y, x),
            .linearized = Linearized.alloc(allocator),
        };

        return tensor;
    }
    pub fn free(this: @This(), allocator: std.mem.Allocator) void {
        assert(this);
        assert(this.linearized);
        assert(this.buffer);

        this.buffer.free(allocator);
        this.linearized.free(allocator);
        this.buffer = null;
        this.linearized = null;
    }
    /// TODO: Decide if this should clear the linearized. On one hand it makes it so that you don't need to rebuild the linearized if you want to run it again
    /// However it is more intuitive that if you reailze a tensor that it should clear the linearized used to generate it
    pub fn realize(this: @This()) void {
        assert(this);
        this.linearized.run();
        this.linearized.clear();
    }
    // TODO: All the op add functions
};
