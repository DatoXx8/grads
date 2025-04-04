const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const ClMem = @import("./runtimes/cl.zig").ClMem;
const ClContext = @import("./runtimes/cl.zig").ClContext;
const ClCommandQueue = @import("./runtimes/cl.zig").ClCommandQueue;
const opencl = @import("./runtimes/cl.zig").opencl;

/// 4 is probably already enough. 26 ^ 4 = 456.976
/// 8 is absolute overkill. 26 ^ 8 = 208.827.064.576
pub const buffer_name_size: u64 = 8;
/// Valid increment of the char values in the buffer name
pub const buffer_name_char_options: u64 = 'z' - 'a' + 1;
/// For keeping track of the current number of buffers allocated
var buffer_name_offset: u64 = 0;

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

    // $NOTE I used to save the initial sizes of each dimension but based on usage I noticed only the total size was
    //  necessary and that is already saved in values.len

    a_size: u32,
    z_size: u32,
    y_size: u32,
    x_size: u32,
    a_stride: u32,
    z_stride: u32,
    y_stride: u32,
    x_stride: u32,
    offset: u32,
    values: []f32,
    values_cl: ?ClMem,
    sync: SyncStatus,
    name_offset: u64,
    intermediary: bool,
    pub fn alloc(allocator: Allocator, a: u32, z: u32, y: u32, x: u32, context: ?ClContext) !Buffer {
        assert(a > 0);
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);

        defer buffer_name_offset += 1;
        return .{
            .name_offset = buffer_name_offset,
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
            .values = try allocator.alloc(f32, a * z * y * x),
            .values_cl = if (context) |ctx| try ClMem.alloc(ctx, a, z, y, x) else null,
            .intermediary = false,
        };
    }
    pub fn allocIntermediary(allocator: Allocator, a: u32, z: u32, y: u32, x: u32, context: ?ClContext) !Buffer {
        assert(a > 0);
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);

        defer buffer_name_offset += 1;
        return .{
            .name_offset = buffer_name_offset,
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
            .values = try allocator.alloc(f32, a * z * y * x),
            .values_cl = if (context) |ctx| try ClMem.alloc(ctx, a, z, y, x) else null,
            .intermediary = true,
        };
    }
    pub fn free(this: @This(), allocator: Allocator) void {
        allocator.free(this.values);
        if (this.values_cl) |values_cl| {
            // $NOTE I am not sure if this is the right approach or to just return the error, but I hate that the free function could fail then
            values_cl.free() catch |err| {
                std.log.err("Could not free values_cl in buffer {} because of error {!}\n", .{ this.name_offset, err });
            };
        }
    }
    // $PERF these function below are used to calculate fields that I removed from the struct because they can just be calculated relatively quickly
    //  to reduce memory usage. Like the nameFromOffset function takes about 20ns on my machine (Ryzen 5 5600x) using a debug build to calculate a name for buffer_name_size = 8
    //  which is ~5x faster than a read from main memory.
    pub inline fn name(this: @This()) [buffer_name_size]u8 {
        return nameFromOffset(this.name_offset);
    }
    pub inline fn nameFromOffset(name_offset: u64) [buffer_name_size]u8 {
        var name_result: [buffer_name_size]u8 = [_]u8{'a'} ** buffer_name_size;
        const divisor: u64 = buffer_name_char_options;
        var left: u64 = name_offset;
        for (0..buffer_name_size) |char_idx| {
            name_result[char_idx] += @truncate(left % divisor);
            left = left / divisor;
        }
        // $NOTE Enforce that you don't generate new tensors beyond 'zzzz...zzz'
        assert(left == 0);

        return name_result;
    }
    pub inline fn aOffset(this: @This()) u32 {
        return @divFloor(this.offset, this.a_stride);
    }
    pub inline fn zOffset(this: @This()) u32 {
        return @divFloor(this.offset % this.a_stride, this.z_stride);
    }
    pub inline fn yOffset(this: @This()) u32 {
        return @divFloor(this.offset % this.z_stride, this.y_stride);
    }
    pub inline fn xOffset(this: @This()) u32 {
        return @divFloor(this.offset % this.y_stride, this.x_stride);
    }
    pub inline fn at(this: @This(), a: u32, z: u32, y: u32, x: u32) u32 {
        // $NOTE Could add the per dimension asserts back in
        const offset: u32 = this.offset + a * this.a_stride + z * this.z_stride + y * this.y_stride + x * this.x_stride;
        assert(offset < this.values.len);
        return offset;
    }
    pub fn syncToHost(this: *@This(), queue: ClCommandQueue) !void {
        if (this.sync == .sync_to_host) {
            const size: usize = this.values.len * @sizeOf(f32);
            if (opencl.clEnqueueReadBuffer(queue.queue, this.values_cl.?.memory, //
                opencl.CL_TRUE, 0, size, this.values.ptr, 0, null, null) != 0)
            {
                return SyncError.FailedToHost;
            }
            this.sync = .sync_to_none;
        }
    }
    pub fn syncToDevice(this: *@This(), queue: ClCommandQueue) !void {
        if (this.sync == .sync_to_device) {
            const size: usize = this.values.len * @sizeOf(f32);
            if (opencl.clEnqueueWriteBuffer(queue.queue, this.values_cl.?.memory, //
                opencl.CL_TRUE, 0, size, this.values.ptr, 0, null, null) != 0)
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
    pub fn syncWait(_: *@This(), queue: ClCommandQueue) !void {
        if (opencl.clFinish(queue.queue) == 0) {
            return;
        } else {
            return SyncError.FailedWait;
        }
    }
    /// Checks for equal size, offset and name.
    /// Does *not* check for inherent buffer size.
    pub inline fn equal(this: @This(), target: @This()) bool {
        return this.name_offset == target.name_offset and
            this.a_size == target.a_size and this.z_size == target.z_size and
            this.y_size == target.y_size and this.x_size == target.x_size and
            this.aOffset() == target.aOffset() and this.zOffset() == target.zOffset() and
            this.yOffset() == target.yOffset() and this.xOffset() == target.xOffset();
    }
    /// Checks for equal size and name.
    /// Does *not* check for inherent buffer size or offsets.
    pub inline fn equalNoOffset(this: @This(), target: @This()) bool {
        return this.name_offset == target.name_offset and
            this.a_size == target.a_size and this.z_size == target.z_size and
            this.y_size == target.y_size and this.x_size == target.x_size;
    }
    /// Return wether the two buffers overlap in any place
    pub inline fn overlaps(this: @This(), target: @This()) bool {
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
    pub inline fn overlapsAll(this: @This(), target: @This()) bool {
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
    pub inline fn overlapsPartial(this: @This(), target: @This()) bool {
        const a_1: u32 = target.aOffset();
        const z_1: u32 = target.zOffset();
        const y_1: u32 = target.yOffset();
        const x_1: u32 = target.xOffset();

        const a_2: u32 = target.aOffset();
        const z_2: u32 = target.zOffset();
        const y_2: u32 = target.yOffset();
        const x_2: u32 = target.xOffset();

        return @max(a_1, a_2) < @min(a_1 + this.a_size, a_2 + target.a_size) and a_1 != a_2 and
            @max(z_1, z_2) < @min(z_1 + this.z_size, z_2 + target.z_size) and z_1 != z_2 and
            @max(y_1, y_2) < @min(y_1 + this.y_size, y_2 + target.y_size) and y_1 != y_2 and
            @max(x_1, x_2) < @min(x_1 + this.x_size, x_2 + target.x_size) and x_1 != x_2;
    }
};
pub const Op = struct {
    // $TODO Linary is a truly terrible name
    /// Linary is like binary but the in buffer has size [1, 1, 1, 1]
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
            // $NOTE I did this with a switch statement so that you are forced to handle this in case you add a new op
            return switch (this) {
                .unary_add => true,
                .unary_subtract => true,
                .unary_multiply => true,
                .unary_divide => true,
                .unary_exp => true,
                .unary_log => true,
                .unary_square => true,
                .unary_sqrt => true,
                .unary_reciprocal => true,
                .unary_max => true,
                .unary_min => true,
                .unary_set => true,
                .unary_random => true,
                .unary_tanh => true,
                .unary_absolute => true,
                .unary_sign => true,
                .binary_add => false,
                .binary_subtract => false,
                .binary_multiply => false,
                .binary_divide => false,
                .binary_max => false,
                .binary_min => false,
                .binary_set => false,
                .linary_add => false,
                .linary_subtract => false,
                .linary_multiply => false,
                .linary_divide => false,
                .linary_max => false,
                .linary_min => false,
                .linary_set => false,
                .reduce_sum => false,
                .reduce_max => false,
                .reduce_avg => false,
                .reduce_min => false,
            };
        }
        pub inline fn isBinary(this: @This()) bool {
            // $NOTE I did this with a switch statement so that you are forced to handle this in case you add a new op
            return switch (this) {
                .unary_add => false,
                .unary_subtract => false,
                .unary_multiply => false,
                .unary_divide => false,
                .unary_exp => false,
                .unary_log => false,
                .unary_square => false,
                .unary_sqrt => false,
                .unary_reciprocal => false,
                .unary_max => false,
                .unary_min => false,
                .unary_set => false,
                .unary_random => false,
                .unary_tanh => false,
                .unary_absolute => false,
                .unary_sign => false,
                .binary_add => true,
                .binary_subtract => true,
                .binary_multiply => true,
                .binary_divide => true,
                .binary_max => true,
                .binary_min => true,
                .binary_set => true,
                .linary_add => false,
                .linary_subtract => false,
                .linary_multiply => false,
                .linary_divide => false,
                .linary_max => false,
                .linary_min => false,
                .linary_set => false,
                .reduce_sum => false,
                .reduce_max => false,
                .reduce_avg => false,
                .reduce_min => false,
            };
        }
        pub inline fn isLinary(this: @This()) bool {
            // $NOTE I did this with a switch statement so that you are forced to handle this in case you add a new op
            return switch (this) {
                .unary_add => false,
                .unary_subtract => false,
                .unary_multiply => false,
                .unary_divide => false,
                .unary_exp => false,
                .unary_log => false,
                .unary_square => false,
                .unary_sqrt => false,
                .unary_reciprocal => false,
                .unary_max => false,
                .unary_min => false,
                .unary_set => false,
                .unary_random => false,
                .unary_tanh => false,
                .unary_absolute => false,
                .unary_sign => false,
                .binary_add => false,
                .binary_subtract => false,
                .binary_multiply => false,
                .binary_divide => false,
                .binary_max => false,
                .binary_min => false,
                .binary_set => false,
                .linary_add => true,
                .linary_subtract => true,
                .linary_multiply => true,
                .linary_divide => true,
                .linary_max => true,
                .linary_min => true,
                .linary_set => true,
                .reduce_sum => false,
                .reduce_max => false,
                .reduce_avg => false,
                .reduce_min => false,
            };
        }
        pub inline fn isReduce(this: @This()) bool {
            // $NOTE I did this with a switch statement so that you are forced to handle this in case you add a new op
            return switch (this) {
                .unary_add => false,
                .unary_subtract => false,
                .unary_multiply => false,
                .unary_divide => false,
                .unary_exp => false,
                .unary_log => false,
                .unary_square => false,
                .unary_sqrt => false,
                .unary_reciprocal => false,
                .unary_max => false,
                .unary_min => false,
                .unary_set => false,
                .unary_random => false,
                .unary_tanh => false,
                .unary_absolute => false,
                .unary_sign => false,
                .binary_add => false,
                .binary_subtract => false,
                .binary_multiply => false,
                .binary_divide => false,
                .binary_max => false,
                .binary_min => false,
                .binary_set => false,
                .linary_add => false,
                .linary_subtract => false,
                .linary_multiply => false,
                .linary_divide => false,
                .linary_max => false,
                .linary_min => false,
                .linary_set => false,
                .reduce_sum => true,
                .reduce_max => true,
                .reduce_avg => true,
                .reduce_min => true,
            };
        }
    };
    type: Type,
    // $NOTE When using this as a seed for the unary_random op, the integer value will be cast back and forth with @bitCast,
    //  here's hoping some NaN floating point magic doesn't ruin that
    u_var: f32,
    out: Buffer,
    in: Buffer,
    pub inline fn isUnary(this: @This()) bool {
        return this.type.isUnary();
    }
    pub inline fn isBinary(this: @This()) bool {
        return this.type.isBinary();
    }
    pub inline fn isLinary(this: @This()) bool {
        return this.type.isLinary();
    }
    pub inline fn isReduce(this: @This()) bool {
        return this.type.isReduce();
    }
    pub inline fn isOutInlinable(this: @This(), target: @This()) bool {
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
    pub inline fn isInInlinable(this: @This(), target: @This()) bool {
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
    // $TODO Make this sucker SIMD-able
    pub fn realize(this: @This()) void {
        if (this.isUnary()) {
            // $NOTE In buffer is just a copy of out buffer, basically just a sanity check.
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
            .reduce_sum => {
                this.out.values[this.out.at(0, 0, 0, 0)] = 0;
            },
            .reduce_max => {
                this.out.values[this.out.at(0, 0, 0, 0)] = -std.math.inf(f32);
            },
            .reduce_min => {
                this.out.values[this.out.at(0, 0, 0, 0)] = std.math.inf(f32);
            },
            .reduce_avg => {
                this.out.values[this.out.at(0, 0, 0, 0)] = 0;
            },
            else => {},
        }

        var rng: ?std.Random.Pcg = if (this.type == .unary_random) std.Random.Pcg.init(@as(u32, @bitCast(this.u_var))) else null;

        // $TODO Add SIMD comptime width using @Vector

        // Just to be clear I know that putting the loop outside might make it slower because you have to go through the switch statement every time, but
        // the branch predictor is likely to have an extremely easy time predicting the branches since it's the same every single time.
        // Which should mean that as long as your CPU even has a branch predictor it should cause very little to no performance impact.
        // I measured it by running some arbitrary ops and there was no measurable difference
        const a_size: u32 = if (this.isReduce()) this.in.a_size else this.out.a_size;
        const z_size: u32 = if (this.isReduce()) this.in.z_size else this.out.z_size;
        const y_size: u32 = if (this.isReduce()) this.in.y_size else this.out.y_size;
        const x_size: u32 = if (this.isReduce()) this.in.x_size else this.out.x_size;
        var a: u32 = 0;
        while (a < a_size) : (a += 1) {
            var z: u32 = 0;
            while (z < z_size) : (z += 1) {
                var y: u32 = 0;
                while (y < y_size) : (y += 1) {
                    var x: u32 = 0;
                    while (x < x_size) : (x += 1) {
                        switch (this.type) {
                            .unary_add => {
                                this.out.values[this.out.at(a, z, y, x)] += this.u_var;
                            },
                            .unary_subtract => {
                                this.out.values[this.out.at(a, z, y, x)] -= this.u_var;
                            },
                            .unary_multiply => {
                                this.out.values[this.out.at(a, z, y, x)] *= this.u_var;
                            },
                            .unary_divide => {
                                this.out.values[this.out.at(a, z, y, x)] /= this.u_var;
                            },
                            .unary_exp => {
                                this.out.values[this.out.at(a, z, y, x)] = @exp(this.out.values[this.out.at(a, z, y, x)]);
                            },
                            .unary_log => {
                                this.out.values[this.out.at(a, z, y, x)] = @log(this.out.values[this.out.at(a, z, y, x)]);
                            },
                            .unary_square => {
                                this.out.values[this.out.at(a, z, y, x)] *= this.out.values[this.out.at(a, z, y, x)];
                            },
                            .unary_sqrt => {
                                this.out.values[this.out.at(a, z, y, x)] = @sqrt(this.out.values[this.out.at(a, z, y, x)]);
                            },
                            .unary_reciprocal => {
                                this.out.values[this.out.at(a, z, y, x)] = 1 / this.out.values[this.out.at(a, z, y, x)];
                            },
                            .unary_max => {
                                this.out.values[this.out.at(a, z, y, x)] = @max(this.out.values[this.out.at(a, z, y, x)], this.u_var);
                            },
                            .unary_min => {
                                this.out.values[this.out.at(a, z, y, x)] = @min(this.out.values[this.out.at(a, z, y, x)], this.u_var);
                            },
                            .unary_set => {
                                this.out.values[this.out.at(a, z, y, x)] = this.u_var;
                            },
                            .unary_random => {
                                // $TODO Make my own PCG implementation that can do SIMD
                                this.out.values[this.out.at(a, z, y, x)] = rng.?.random().floatNorm(f32);
                            },
                            .unary_tanh => {
                                this.out.values[this.out.at(a, z, y, x)] = std.math.tanh(this.out.values[this.out.at(a, z, y, x)]);
                            },
                            .unary_absolute => {
                                if (this.out.values[this.out.at(a, z, y, x)] < 0) {
                                    this.out.values[this.out.at(a, z, y, x)] = -this.out.values[this.out.at(a, z, y, x)];
                                } else {
                                    this.out.values[this.out.at(a, z, y, x)] = this.out.values[this.out.at(a, z, y, x)];
                                }
                            },
                            .unary_sign => {
                                if (this.out.values[this.out.at(a, z, y, x)] > 0) {
                                    this.out.values[this.out.at(a, z, y, x)] = 1;
                                } else if (this.out.values[this.out.at(a, z, y, x)] < 0) {
                                    this.out.values[this.out.at(a, z, y, x)] = -1;
                                } else {
                                    this.out.values[this.out.at(a, z, y, x)] = 0;
                                }
                                // $fn signVector(comptime T: type, vector @Vector(4, T)) @Vector(4, i32) {
                                //     const zero = @splat(4, @as(T, 0));
                                //     const one = @splat(4, @as(T, 1));
                                //     const neg_one = @splat(4, @as(T, -1));
                                //
                                //     const cmp_lt = @Vector(4, bool)(vector < zero);
                                //     const cmp_gt = @Vector(4, bool)(vector > zero);
                                //
                                //     return @select(cmp_lt, neg_one, @select(cmp_gt, one, @splat(4, @as(i32, 0))));
                                // }
                            },
                            .binary_add => {
                                this.out.values[this.out.at(a, z, y, x)] += this.in.values[this.in.at(a, z, y, x)];
                            },
                            .binary_subtract => {
                                this.out.values[this.out.at(a, z, y, x)] -= this.in.values[this.in.at(a, z, y, x)];
                            },
                            .binary_multiply => {
                                this.out.values[this.out.at(a, z, y, x)] *= this.in.values[this.in.at(a, z, y, x)];
                            },
                            .binary_divide => {
                                this.out.values[this.out.at(a, z, y, x)] /= this.in.values[this.in.at(a, z, y, x)];
                            },
                            .binary_max => {
                                this.out.values[this.out.at(a, z, y, x)] = @max(this.out.values[this.out.at(a, z, y, x)], this.in.values[this.in.at(a, z, y, x)]);
                            },
                            .binary_min => {
                                this.out.values[this.out.at(a, z, y, x)] = @min(this.out.values[this.out.at(a, z, y, x)], this.in.values[this.in.at(a, z, y, x)]);
                            },
                            .binary_set => {
                                this.out.values[this.out.at(a, z, y, x)] = this.in.values[this.in.at(a, z, y, x)];
                            },
                            .linary_add => {
                                this.out.values[this.out.at(a, z, y, x)] += this.in.values[this.in.at(0, 0, 0, 0)];
                            },
                            .linary_subtract => {
                                this.out.values[this.out.at(a, z, y, x)] -= this.in.values[this.in.at(0, 0, 0, 0)];
                            },
                            .linary_multiply => {
                                this.out.values[this.out.at(a, z, y, x)] *= this.in.values[this.in.at(0, 0, 0, 0)];
                            },
                            .linary_divide => {
                                this.out.values[this.out.at(a, z, y, x)] /= this.in.values[this.in.at(0, 0, 0, 0)];
                            },
                            .linary_max => {
                                this.out.values[this.out.at(a, z, y, x)] = @max(this.out.values[this.out.at(a, z, y, x)], this.in.values[this.in.at(0, 0, 0, 0)]);
                            },
                            .linary_min => {
                                this.out.values[this.out.at(a, z, y, x)] = @min(this.out.values[this.out.at(a, z, y, x)], this.in.values[this.in.at(0, 0, 0, 0)]);
                            },
                            .linary_set => {
                                this.out.values[this.out.at(a, z, y, x)] = this.in.values[this.in.at(0, 0, 0, 0)];
                            },
                            .reduce_sum => {
                                this.out.values[this.out.at(0, 0, 0, 0)] += this.in.values[this.in.at(a, z, y, x)];
                            },
                            .reduce_max => {
                                this.out.values[this.out.at(0, 0, 0, 0)] = @max(this.out.values[this.out.at(0, 0, 0, 0)], this.in.values[this.in.at(a, z, y, x)]);
                            },
                            .reduce_min => {
                                this.out.values[this.out.at(0, 0, 0, 0)] = @min(this.out.values[this.out.at(0, 0, 0, 0)], this.in.values[this.in.at(a, z, y, x)]);
                            },
                            .reduce_avg => {
                                this.out.values[this.out.at(0, 0, 0, 0)] += this.in.values[this.in.at(a, z, y, x)];
                            },
                        }
                    }
                }
            }
        }
        if (this.type == .reduce_avg) {
            this.out.values[this.out.at(0, 0, 0, 0)] /= @as(f32, @floatFromInt(this.in.a_size * this.in.z_size * this.in.y_size * this.in.x_size));
        }
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}{s} ", .{ " " ** (padding + offset), text });
        } else {
            std.debug.print("{s}", .{" " ** (padding + offset)});
        }
        if (this.isUnary()) {
            std.debug.print("U {s} ({d} {d} {d} {d}) [{d} {d} {d} {d} = {d}] \"{s}\" {d}\n", .{
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
                this.out.aOffset(),
                this.out.zOffset(),
                this.out.yOffset(),
                this.out.xOffset(),
                this.out.offset,
                this.out.name(),
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
                this.out.aOffset(),
                this.out.zOffset(),
                this.out.yOffset(),
                this.out.xOffset(),
                this.out.offset,
                this.out.name(),
                this.in.a_size,
                this.in.z_size,
                this.in.y_size,
                this.in.x_size,
                this.in.aOffset(),
                this.in.zOffset(),
                this.in.yOffset(),
                this.in.xOffset(),
                this.in.offset,
                this.in.name(),
            });
        }
    }
};

const op_cap_base = 4;
pub const Linearized = struct {
    op: []Op,
    op_num: u32,
    pub fn alloc(allocator: Allocator) !Linearized {
        return .{
            .op_num = 0,
            .op = try allocator.alloc(Op, op_cap_base),
        };
    }
    pub fn capacityEnsure(this: *@This(), allocator: Allocator, capacity: u32) !void {
        if (this.op.len - this.op_num < capacity) {
            this.op = try allocator.realloc(this.op, this.op_num + capacity);
        }
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
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
    pub fn append(this: *@This(), op: Op) void {
        assert(this.op_num < this.op.len);
        this.op[this.op_num] = op;
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
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
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
};

pub const Tensor = struct {
    buffer: Buffer,
    linearized: Linearized,
    pub fn alloc(allocator: Allocator, a: u32, z: u32, y: u32, x: u32, context: ?ClContext) !Tensor {
        assert(a > 0);
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);

        return .{
            .buffer = try Buffer.alloc(allocator, a, z, y, x, context),
            .linearized = try Linearized.alloc(allocator),
        };
    }
    pub fn allocIntermediary(allocator: Allocator, a: u32, z: u32, y: u32, x: u32, context: ?ClContext) !Tensor {
        assert(a > 0);
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);

        return .{
            .buffer = try Buffer.allocIntermediary(allocator, a, z, y, x, context),
            .linearized = try Linearized.alloc(allocator),
        };
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
        this.buffer.free(allocator);
        this.linearized.free(allocator);
    }
    /// Clears the linearized ops in the tensor, meaning that if you want to run it a few times then use tensor.linearized.run() or compile it to a program if you run it often.
    pub fn realize(this: *@This()) void {
        if (this.linearized.op_num != 0) {
            this.linearized.run();
            this.linearized.clear();
            this.buffer.syncUpdate(.sync_to_device);
        }
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Tensor {s} = {s}\n", .{ " " ** offset, this.buffer.name(), text });
        } else {
            std.debug.print("{s}Tensor {s}\n", .{ " " ** offset, this.buffer.name() });
        }
        var a: u32 = 0;
        while (a < this.buffer.a_size) : (a += 1) {
            var z: u32 = 0;
            while (z < this.buffer.z_size) : (z += 1) {
                var y: u32 = 0;
                while (y < this.buffer.y_size) : (y += 1) {
                    std.debug.print("{s}[", .{" " ** (offset + padding)});
                    var x: u32 = 0;
                    while (x < this.buffer.x_size) : (x += 1) {
                        std.debug.print(" {d:8.4}", .{this.buffer.values[this.buffer.at(a, z, y, x)]});
                    }
                    std.debug.print("]\n", .{});
                }
                std.debug.print("\n", .{});
            }
            std.debug.print("\n", .{});
        }
    }
    pub fn debug(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Tensor {s} = {s}\n", .{ " " ** offset, this.buffer.name(), text });
        } else {
            std.debug.print("{s}Tensor {s}\n", .{ " " ** offset, this.buffer.name() });
        }
        this.linearized.debug(0, 0, null);
        var a: u32 = 0;
        while (a < this.buffer.a_size) : (a += 1) {
            var z: u32 = 0;
            while (z < this.buffer.z_size) : (z += 1) {
                var y: u32 = 0;
                while (y < this.buffer.y_size) : (y += 1) {
                    std.debug.print("{s}[", .{" " ** (offset + padding)});
                    var x: u32 = 0;
                    while (x < this.buffer.x_size) : (x += 1) {
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
        assert(!std.math.isNan(u_var));
        assert(!std.math.isInf(u_var));
        this.linearized.append(.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_add,
            .u_var = u_var,
        });
    }
    pub fn unarySubtract(this: *@This(), u_var: f32) void {
        assert(!std.math.isNan(u_var));
        assert(!std.math.isInf(u_var));
        this.linearized.append(.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_subtract,
            .u_var = u_var,
        });
    }
    pub fn unaryMultiply(this: *@This(), u_var: f32) void {
        assert(!std.math.isNan(u_var));
        assert(!std.math.isInf(u_var));
        this.linearized.append(.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_multiply,
            .u_var = u_var,
        });
    }
    pub fn unaryDivide(this: *@This(), u_var: f32) void {
        assert(!std.math.isNan(u_var));
        assert(!std.math.isInf(u_var));
        this.linearized.append(.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_divide,
            .u_var = u_var,
        });
    }
    pub fn unaryExp(this: *@This()) void {
        this.linearized.append(.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_exp,
            .u_var = 0,
        });
    }
    pub fn unaryLog(this: *@This()) void {
        this.linearized.append(.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_log,
            .u_var = 0,
        });
    }
    pub fn unarySquare(this: *@This()) void {
        this.linearized.append(.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_square,
            .u_var = 0,
        });
    }
    pub fn unarySqrt(this: *@This()) void {
        this.linearized.append(.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_sqrt,
            .u_var = 0,
        });
    }
    pub fn unaryReciprocal(this: *@This()) void {
        this.linearized.append(.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_reciprocal,
            .u_var = 0,
        });
    }
    pub fn unaryMax(this: *@This(), u_var: f32) void {
        assert(!std.math.isNan(u_var));
        assert(!std.math.isInf(u_var));
        this.linearized.append(.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_max,
            .u_var = u_var,
        });
    }
    pub fn unaryMin(this: *@This(), u_var: f32) void {
        assert(!std.math.isNan(u_var));
        assert(!std.math.isInf(u_var));
        this.linearized.append(.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_min,
            .u_var = u_var,
        });
    }
    pub fn unarySet(this: *@This(), u_var: f32) void {
        assert(!std.math.isNan(u_var));
        assert(!std.math.isInf(u_var));
        this.linearized.append(.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_set,
            .u_var = u_var,
        });
    }
    // $TODO Decide if this explicit seed thing is actually any good at all.
    //  I don't really want to make the user think about it, but the explicit-nes is also nice
    /// Here u_var is the seed of the prng
    pub fn unaryRandom(this: *@This(), u_var: u32) void {
        this.linearized.append(.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_random,
            .u_var = @bitCast(u_var),
        });
    }
    pub fn unaryTanh(this: *@This()) void {
        this.linearized.append(.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_tanh,
            .u_var = 0,
        });
    }
    pub fn unaryAbsolute(this: *@This()) void {
        this.linearized.append(.{
            .out = this.buffer,
            .in = this.buffer,
            .type = .unary_absolute,
            .u_var = 0,
        });
    }
    pub fn unarySign(this: *@This()) void {
        this.linearized.append(.{
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
        this.linearized.append(.{
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
        this.linearized.append(.{
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
        this.linearized.append(.{
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
        this.linearized.append(.{
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
        this.linearized.append(.{
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
        this.linearized.append(.{
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
        this.linearized.append(.{
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
        this.linearized.append(.{
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
        this.linearized.append(.{
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
        this.linearized.append(.{
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
        this.linearized.append(.{
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
        this.linearized.append(.{
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
        this.linearized.append(.{
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
        this.linearized.append(.{
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
        this.linearized.append(.{
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
        this.linearized.append(.{
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
        this.linearized.append(.{
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
        this.linearized.append(.{
            .out = this.buffer,
            .in = source.buffer,
            .type = .reduce_avg,
            .u_var = 0,
        });
    }
    pub fn moveReshape(this: *@This(), a: u32, z: u32, y: u32, x: u32) void {
        assert(a > 0);
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);
        // $NOTE These are here so that it is easier to identify if there is a single run-away dimension
        assert(a <= this.buffer.values.len);
        assert(z <= this.buffer.values.len);
        assert(y <= this.buffer.values.len);
        assert(x <= this.buffer.values.len);
        assert(a * z * y * x <= this.buffer.values.len);
        this.buffer.a_size = a;
        this.buffer.z_size = z;
        this.buffer.y_size = y;
        this.buffer.x_size = x;
        this.buffer.a_stride = z * y * x;
        this.buffer.z_stride = y * x;
        this.buffer.y_stride = x;
        this.buffer.x_stride = 1;
    }
    pub fn moveResize(this: *@This(), a: u32, z: u32, y: u32, x: u32) void {
        assert(a > 0);
        assert(z > 0);
        assert(y > 0);
        assert(x > 0);
        // $NOTE These are here so that it is easier to identify if there is a single run-away dimension
        assert(a <= this.buffer.values.len);
        assert(z <= this.buffer.values.len);
        assert(y <= this.buffer.values.len);
        assert(x <= this.buffer.values.len);
        assert(a * z * y * x <= this.buffer.values.len);
        this.buffer.a_size = a;
        this.buffer.z_size = z;
        this.buffer.y_size = y;
        this.buffer.x_size = x;
    }
    pub fn moveOffset(this: *@This(), a: u32, z: u32, y: u32, x: u32) void {
        assert(a < this.buffer.values.len);
        assert(z < this.buffer.values.len);
        assert(y < this.buffer.values.len);
        assert(x < this.buffer.values.len);
        // $NOTE These are here so that it is easier to identify if there is a single run-away dimension
        assert(a * this.buffer.a_stride + z * this.buffer.z_stride + y * this.buffer.y_stride +
            x * this.buffer.x_stride < this.buffer.values.len);
        this.buffer.offset = a * this.buffer.a_stride + z * this.buffer.z_stride + y * this.buffer.y_stride + x * this.buffer.x_stride;
    }
    pub fn dependOn(this: *@This(), prerequisite: *@This()) void {
        this.linearized.concat(&prerequisite.linearized);
    }
};
