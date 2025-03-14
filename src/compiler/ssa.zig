const std = @import("std");

const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Op = @import("../tensor.zig").Op;
const Linearized = @import("../tensor.zig").Linearized;
const Buffer = @import("../tensor.zig").Buffer;

const Optimization = @import("./optimize.zig").Optimization;

// $FIXME
//
// B set (1 6 2 4) [0, 0, 0, 0 = 0] false "khaaaaaa" (1 6 2 4) [0, 0, 0, 0 = 0] false "ygaaaaaa" 0
// B set (1 6 2 4) [0, 0, 2, 0 = 8] false "khaaaaaa" (1 6 2 4) [0, 0, 2, 0 = 8] false "ygaaaaaa" 2
// B set (1 6 2 4) [1, 0, 1, 0 = 124] false "khaaaaaa" (1 6 2 4) [1, 0, 1, 0 = 124] false "ygaaaaaa" 4
// B set (1 6 2 4) [2, 0, 0, 0 = 240] false "khaaaaaa" (1 6 2 4) [2, 0, 0, 0 = 240] false "ygaaaaaa" 6
// B set (1 6 2 4) [2, 0, 2, 0 = 248] false "khaaaaaa" (1 6 2 4) [2, 0, 2, 0 = 248] false "ygaaaaaa" 8
// B set (1 6 2 4) [3, 0, 1, 0 = 364] false "khaaaaaa" (1 6 2 4) [3, 0, 1, 0 = 364] false "ygaaaaaa" 10
// B set (1 6 2 4) [4, 0, 0, 0 = 480] false "khaaaaaa" (1 6 2 4) [4, 0, 0, 0 = 480] false "ygaaaaaa" 12
// B set (1 6 2 4) [4, 0, 2, 0 = 488] false "khaaaaaa" (1 6 2 4) [4, 0, 2, 0 = 488] false "ygaaaaaa" 14
// B set (1 6 2 4) [5, 0, 1, 0 = 604] false "khaaaaaa" (1 6 2 4) [5, 0, 1, 0 = 604] false "ygaaaaaa" 16
//
// Gets transformed into
//
// 0 => 0 0 0 0
// 1 => 0 0 1 0
// 2 => 1 0 2 0
// 3 => 1 0 0 0
// 4 => 2 0 1 0
// 5 => 2 0 2 0
// 6 => 3 0 0 0
// 7 => 3 0 1 0
// 8 => 4 0 2 0
//
// Because the per dimension offsets aren't linear... This is a huge issue.
pub const DimInfo = struct {
    // $MAYBE Could make the index table smaller by having a reset variable again, but now finding resets is way harder because actual loop checks are necessary
    // $NOTE These are essentialy the gcd of the different lengths between the offset changes
    a_wait_out: u32,
    z_wait_out: u32,
    y_wait_out: u32,
    x_wait_out: u32,
    a_wait_in: u32,
    z_wait_in: u32,
    y_wait_in: u32,
    x_wait_in: u32,
    // $TODO Is there a way to not make this a dynamic memory allocation?
    a_off_out: []u32,
    z_off_out: []u32,
    y_off_out: []u32,
    x_off_out: []u32,
    a_off_in: []u32,
    z_off_in: []u32,
    y_off_in: []u32,
    x_off_in: []u32,
    off: []u32,
    pub fn alloc(allocator: Allocator, base: []const Base) !DimInfo {
        assert(base.len > 0);

        var a_idx_out: u32 = 0;
        var z_idx_out: u32 = 0;
        var y_idx_out: u32 = 0;
        var x_idx_out: u32 = 0;
        var a_idx_in: u32 = 0;
        var z_idx_in: u32 = 0;
        var y_idx_in: u32 = 0;
        var x_idx_in: u32 = 0;

        var a_wait_out: u32 = @truncate(base.len);
        var z_wait_out: u32 = @truncate(base.len);
        var y_wait_out: u32 = @truncate(base.len);
        var x_wait_out: u32 = @truncate(base.len);
        var a_wait_in: u32 = @truncate(base.len);
        var z_wait_in: u32 = @truncate(base.len);
        var y_wait_in: u32 = @truncate(base.len);
        var x_wait_in: u32 = @truncate(base.len);

        var base_idx: u32 = 0;
        while (base_idx < base.len) : (base_idx += 1) {
            if (base[a_idx_out].out.aOffset() != base[base_idx].out.aOffset()) {
                a_wait_out = std.math.gcd(a_wait_out, base_idx - a_idx_out);
                a_idx_out = base_idx;
            }
            if (base[z_idx_out].out.zOffset() != base[base_idx].out.zOffset()) {
                z_wait_out = std.math.gcd(z_wait_out, base_idx - z_idx_out);
                z_idx_out = base_idx;
            }
            if (base[y_idx_out].out.yOffset() != base[base_idx].out.yOffset()) {
                y_wait_out = std.math.gcd(y_wait_out, base_idx - y_idx_out);
                y_idx_out = base_idx;
            }
            if (base[x_idx_out].out.xOffset() != base[base_idx].out.xOffset()) {
                x_wait_out = std.math.gcd(x_wait_out, base_idx - x_idx_out);
                x_idx_out = base_idx;
            }
            // $TODO Maybe I should assert that all these are of the same type and things like that
            if (base[base_idx].type.isUnary()) {
                if (base[a_idx_in].in.aOffset() != base[base_idx].in.aOffset()) {
                    a_wait_in = std.math.gcd(a_wait_in, base_idx - a_idx_in);
                    a_idx_in = base_idx;
                }
                if (base[z_idx_in].in.zOffset() != base[base_idx].in.zOffset()) {
                    z_wait_in = std.math.gcd(z_wait_in, base_idx - z_idx_in);
                    z_idx_in = base_idx;
                }
                if (base[y_idx_in].in.yOffset() != base[base_idx].in.yOffset()) {
                    y_wait_in = std.math.gcd(y_wait_in, base_idx - y_idx_in);
                    y_idx_in = base_idx;
                }
                if (base[x_idx_in].in.xOffset() != base[base_idx].in.xOffset()) {
                    x_wait_in = std.math.gcd(x_wait_in, base_idx - x_idx_in);
                    x_idx_in = base_idx;
                }
            }
        }

        const a_len_out = std.math.divCeil(u32, base.len, a_wait_out);
        const z_len_out = std.math.divCeil(u32, base.len, z_wait_out);
        const y_len_out = std.math.divCeil(u32, base.len, y_wait_out);
        const x_len_out = std.math.divCeil(u32, base.len, x_wait_out);
        const a_len_in = std.math.divCeil(u32, base.len, a_wait_in);
        const z_len_in = std.math.divCeil(u32, base.len, z_wait_in);
        const y_len_in = std.math.divCeil(u32, base.len, y_wait_in);
        const x_len_in = std.math.divCeil(u32, base.len, x_wait_in);

        const off = try allocator.alloc(u32, a_len_out + z_len_out + y_len_out + x_len_out + a_len_in + z_len_in + y_len_in + x_len_in);
        


    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}DimInfo {s}\n", .{ [1]u8{' '} ** offset, text });
        }
        std.debug.print("{s}str => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
            " " ** (offset + padding), //
            this.a_stride_out, this.z_stride_out, this.y_stride_out, this.x_stride_out, //
            this.a_stride_in,  this.z_stride_in,  this.y_stride_in,  this.x_stride_in,
        });
        std.debug.print("{s}res => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
            " " ** (offset + padding), //
            this.a_reset_out, this.z_reset_out, this.y_reset_out, this.x_reset_out, //
            this.a_reset_in,  this.z_reset_in,  this.y_reset_in,  this.x_reset_in,
        });
        std.debug.print("{s}wai => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
            " " ** (offset + padding), //
            this.a_wait_out, this.z_wait_out, this.y_wait_out, this.x_wait_out, //
            this.a_wait_in,  this.z_wait_in,  this.y_wait_in,  this.x_wait_in,
        });
        std.debug.print("{s}idx => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
            " " ** (offset + padding), //
            this.a_idx_out, this.z_idx_out, this.y_idx_out, this.x_idx_out, //
            this.a_idx_in,  this.z_idx_in,  this.y_idx_in,  this.x_idx_in,
        });
    }
};

/// The basic thing the Assignment does without any funny business
pub const Base = struct {
    out: Buffer,
    in: Buffer,
    type: Op.Type,
    u_var: f32,
    dim_info: DimInfo,
    // $NOTE If you need more dependency layers than this there's some funky stuff going on
    layer_out: u32,
    layer_in: u32,
    pub inline fn layer(this: @This()) u32 {
        return @max(this.layer_out, this.layer_in);
    }
    // $TODO Unsure how to check the layer things
    pub inline fn equals(this: @This(), target: @This()) bool {
        return this.out.equalNoOffset(target.out) and this.in.equalNoOffset(target.in) and
            this.type == target.type and this.u_var == target.u_var;
    }
    /// Not a great name. Essentialy this returns wether the op has a different result depending on what was in the buffer before.
    pub inline fn overwrites(this: @This()) bool {
        // $NOTE I did this with a switch statement so that you are forced to handle this in case you add a new op
        return switch (this.type) {
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
            .unary_set => true,
            .unary_random => true,
            .unary_tanh => false,
            .unary_absolute => false,
            .unary_sign => false,
            .binary_add => false,
            .binary_subtract => false,
            .binary_multiply => false,
            .binary_divide => false,
            .binary_max => false,
            .binary_min => false,
            .binary_set => true,
            .linary_add => false,
            .linary_subtract => false,
            .linary_multiply => false,
            .linary_divide => false,
            .linary_max => false,
            .linary_min => false,
            .linary_set => true,
            .reduce_sum => true,
            .reduce_max => true,
            .reduce_avg => true,
            .reduce_min => true,
        };
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Base {s}\n", .{ " " ** offset, text });
        }
        if (this.type.isUnary()) {
            std.debug.print("{s}U {s} ({d} {d} {d} {d}) [{d}, {d}, {d}, {d} = {d}] {} \"{s}\" {d} {}\n", .{
                " " ** (offset + padding),
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
                this.out.intermediary,
                this.out.name(),
                this.u_var,
                this.layer(),
            });
        } else {
            const op_kind: u8 = if (this.type.isBinary()) 'B' else (if (this.type.isLinary()) 'L' else (if (this.type.isReduce()) 'R' else unreachable));
            std.debug.print("{s}{c} {s} ({d} {d} {d} {d}) [{d}, {d}, {d}, {d} = {d}] {} \"{s}\" ({d} {d} {d} {d}) [{d}, {d}, {d}, {d} = {d}] {} \"{s}\" {}\n", .{
                " " ** (offset + padding),
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
                this.out.intermediary,
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
                this.in.intermediary,
                this.in.name(),
                this.layer(),
            });
        }
        this.dim_info.print(padding, padding + offset, null);
    }
};

// $TODO Maybe make all these slices from a global / per ssa buffer
/// Tree representation of inlined ops
pub const Inlined = struct {
    base: []Base,
    out: []?u32,
    in: []?u32,
    out_root: ?u32,
    in_root: ?u32,
    inlined_num: u32,
};

/// Wether or not to split a single operation across kernels
pub const Split = struct {
    //
};

/// Describes how to utilize blocks for better caching
pub const Block = struct {
    //
};

/// Describes the SIMD width and tells to codegen to actually use SIMD
pub const Simd = struct {
    //
};

/// Essentialy this is one unit of work
pub const Assign = struct {
    base: Base,
    inlined: ?Inlined,
    split: ?Split,
    block: ?Block,
    simd: ?Simd,
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}Assign {s}\n", .{ " " ** offset, text });
        }
        this.base.print(padding, offset, null);
        if (this.inlined) |inlined| {
            std.debug.print("{s}Inlined out_base {?} in_base {?}\n", .{ " " ** (offset + padding), inlined.out_root, inlined.in_root });
            for (0..inlined.inlined_num) |inlined_idx| {
                inlined.base[inlined_idx].print(padding, padding + offset, null);
            }
        }
        if (this.split) |split| {
            _ = split;
            unreachable;
        }
        if (this.block) |block| {
            _ = block;
            unreachable;
        }
        if (this.simd) |simd| {
            _ = simd;
            unreachable;
        }
    }
};

pub const Ssa = struct {
    assign: []Assign,
    assign_num: u32,
    assign_loop_id: []u32,
    /// Indexed by the id from above
    assign_loop_num: []u32,
    // fn layerLessThan(_: void, lhs: Assign, rhs Assign) bool {
    //     return lhs.base.layer() < rhs.base.layer();
    // }
    pub fn alloc(allocator: Allocator, linearized: Linearized) !Ssa {
        assert(linearized.op_num > 0);

        var layer_read = std.AutoArrayHashMap(u64, u32).init(allocator);
        var layer_write = std.AutoArrayHashMap(u64, u32).init(allocator);
        errdefer {
            layer_read.deinit();
            layer_write.deinit();
        }
        defer {
            layer_read.deinit();
            layer_write.deinit();
        }

        const assign: []Assign = try allocator.alloc(Assign, linearized.op_num);
        const assign_loop_id: []u32 = try allocator.alloc(u32, linearized.op_num);
        const assign_loop_num: []u32 = try allocator.alloc(u32, linearized.op_num);
        errdefer {
            allocator.free(assign);
            allocator.free(assign_loop_id);
            allocator.free(assign_loop_num);
        }
        @memset(assign_loop_id, 0);
        // $NOTE I don't think the other values should be initialised to prevent strange reads
        assign_loop_num[0] = 1;

        for (0..linearized.op_num) |op_idx| {
            const layer_out: u32 = @max(
                layer_write.get(linearized.op[op_idx].out.name_offset) orelse 0,
                layer_read.get(linearized.op[op_idx].out.name_offset) orelse 0,
            );
            const layer_in: u32 = layer_write.get(linearized.op[op_idx].in.name_offset) orelse 0;
            const layer_idx: u32 = @max(layer_out, layer_in);

            assign[op_idx] = .{
                .base = .{
                    .type = linearized.op[op_idx].type,
                    .u_var = linearized.op[op_idx].u_var,
                    .out = linearized.op[op_idx].out,
                    .in = linearized.op[op_idx].in,
                    .layer_out = layer_out,
                    .layer_in = layer_in,
                    .dim_info = undefined,
                },
                .inlined = null,
                .split = null,
                .simd = null,
                .block = null,
            };

            assign[op_idx].base.dim_info = DimInfo.init(allocator, &[1]Base{assign[op_idx].base});

            // $NOTE This overwrites the data if it already existed
            try layer_write.put(assign[op_idx].base.out.name_offset, layer_idx + 1);
            try layer_read.put(assign[op_idx].base.in.name_offset, layer_idx + 1);
        }

        // $NOTE Why was this ever here? Just to group assignments that could be on the same layer?
        // std.mem.sort(Assign, assign, {}, layerLessThan);

        return .{
            .assign = assign,
            .assign_num = linearized.op_num,
            .assign_loop_id = assign_loop_id,
            .assign_loop_num = assign_loop_num,
        };
    }
    pub fn free(this: *@This(), allocator: Allocator) void {
        for (0..this.assign_num) |assign_idx| {
            if (this.assign[assign_idx].inlined) |*inlined| {
                allocator.free(inlined.base);
                allocator.free(inlined.in);
                allocator.free(inlined.out);
            }
            if (this.assign[assign_idx].split) |split| {
                _ = split;
                unreachable;
            }
            if (this.assign[assign_idx].block) |block| {
                _ = block;
                unreachable;
            }
            if (this.assign[assign_idx].simd) |simd| {
                _ = simd;
                unreachable;
            }
        }
        allocator.free(this.assign);
        allocator.free(this.assign_loop_id);
        allocator.free(this.assign_loop_num);
    }
    pub fn optimize(this: *@This(), allocator: Allocator, optimization: Optimization) !void {
        if (optimization == .O0) {
            return;
        }

        try optimization.inlineOp(allocator, this);
        try optimization.parallelize(allocator, this);
        try optimization.splitKernel(allocator, this);

        if (optimization == .O1) {
            return;
        }

        try optimization.simd(allocator, this);

        if (optimization == .O2) {
            return;
        }

        try optimization.memoryLayout(allocator, this);
    }
    pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}SSA {s}\n", .{ " " ** offset, text });
        } else {
            std.debug.print("{s}SSA\n", .{" " ** offset});
        }
        for (0..this.assign_num) |assign_idx| {
            const id: u32 = this.assign_loop_id[assign_idx];
            std.debug.print("{s}[{}] => Loop id {} num {}\n", .{ " " ** offset, assign_idx, id, this.assign_loop_num[id] });
            this.assign[assign_idx].print(padding, offset + padding, null);
        }
    }
};
