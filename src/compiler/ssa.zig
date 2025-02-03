const std = @import("std");

const assert = std.debug.assert;

const Op = @import("../tensor.zig").Op;
const Linearized = @import("../tensor.zig").Linearized;
const Buffer = @import("../tensor.zig").Buffer;

pub const Ssa = struct {
    pub const DimInfo = struct {
        // Offsets for in and out buffer can just be calculated with the idx information
        a_idx_out: usize,
        z_idx_out: usize,
        y_idx_out: usize,
        x_idx_out: usize,
        a_idx_in: usize,
        z_idx_in: usize,
        y_idx_in: usize,
        x_idx_in: usize,
        a_stride_out: usize,
        z_stride_out: usize,
        y_stride_out: usize,
        x_stride_out: usize,
        a_stride_in: usize,
        z_stride_in: usize,
        y_stride_in: usize,
        x_stride_in: usize,
        a_reset_out: usize,
        z_reset_out: usize,
        y_reset_out: usize,
        x_reset_out: usize,
        a_reset_in: usize,
        z_reset_in: usize,
        y_reset_in: usize,
        x_reset_in: usize,
        a_wait_out: usize,
        z_wait_out: usize,
        y_wait_out: usize,
        x_wait_out: usize,
        a_wait_in: usize,
        z_wait_in: usize,
        y_wait_in: usize,
        x_wait_in: usize,
        pub fn init(op: []Op, loop_num: usize) DimInfo {
            assert(loop_num > 0);
            assert(op.len == loop_num);

            // TODO: Could probably make this initializer smaller by making it undefined and then having multiple things in a line like:
            //  dim_info.a_wait_out, dim_info.z_wait_out... = .{0,0,0,0}t
            var dim_info: DimInfo = .{
                .a_idx_out = 0,
                .z_idx_out = 0,
                .y_idx_out = 0,
                .x_idx_out = 0,
                .a_idx_in = 0,
                .z_idx_in = 0,
                .y_idx_in = 0,
                .x_idx_in = 0,
                .a_stride_out = 0,
                .z_stride_out = 0,
                .y_stride_out = 0,
                .x_stride_out = 0,
                .a_stride_in = 0,
                .z_stride_in = 0,
                .y_stride_in = 0,
                .x_stride_in = 0,
                .a_reset_out = loop_num,
                .z_reset_out = loop_num,
                .y_reset_out = loop_num,
                .x_reset_out = loop_num,
                .a_reset_in = loop_num,
                .z_reset_in = loop_num,
                .y_reset_in = loop_num,
                .x_reset_in = loop_num,
                .a_wait_out = 1,
                .z_wait_out = 1,
                .y_wait_out = 1,
                .x_wait_out = 1,
                .a_wait_in = 1,
                .z_wait_in = 1,
                .y_wait_in = 1,
                .x_wait_in = 1,
            };

            for (1..loop_num) |loop_idx| {
                if (op[dim_info.a_idx_out].out.a_offset > op[loop_idx].out.a_offset) {
                    dim_info.a_idx_out = loop_idx;
                }
                if (op[dim_info.z_idx_out].out.z_offset > op[loop_idx].out.z_offset) {
                    dim_info.z_idx_out = loop_idx;
                }
                if (op[dim_info.y_idx_out].out.y_offset > op[loop_idx].out.y_offset) {
                    dim_info.y_idx_out = loop_idx;
                }
                if (op[dim_info.x_idx_out].out.x_offset > op[loop_idx].out.x_offset) {
                    dim_info.x_idx_out = loop_idx;
                }
                if (op[dim_info.a_idx_in].in.a_offset > op[loop_idx].in.a_offset) {
                    dim_info.a_idx_in = loop_idx;
                }
                if (op[dim_info.z_idx_in].in.z_offset > op[loop_idx].in.z_offset) {
                    dim_info.z_idx_in = loop_idx;
                }
                if (op[dim_info.y_idx_in].in.y_offset > op[loop_idx].in.y_offset) {
                    dim_info.y_idx_in = loop_idx;
                }
                if (op[dim_info.x_idx_in].in.x_offset > op[loop_idx].in.x_offset) {
                    dim_info.x_idx_in = loop_idx;
                }
            }

            var a_left_out, var z_left_out, var y_left_out, var x_left_out = .{ false, false, false, false };
            var a_left_in, var z_left_in, var y_left_in, var x_left_in = .{ false, false, false, false };
            var a_enter_out, var z_enter_out, var y_enter_out, var x_enter_out = .{ false, false, false, false };
            var a_enter_in, var z_enter_in, var y_enter_in, var x_enter_in = .{ false, false, false, false };

            for (1..loop_num) |loop_idx| {
                if (!a_left_out and op[dim_info.a_idx_out].out.a_offset !=
                    op[(loop_idx + dim_info.a_idx_out) % loop_num].out.a_offset)
                {
                    a_left_out = true;
                    dim_info.a_wait_out = loop_idx;
                    dim_info.a_stride_out = op[(loop_idx + dim_info.a_idx_out) % loop_num].out.a_offset -
                        op[dim_info.a_idx_out].out.a_offset;
                } else if (a_left_out and !a_enter_out and op[dim_info.a_idx_out].out.a_offset ==
                    op[(loop_idx + dim_info.a_idx_out) % loop_num].out.a_offset)
                {
                    a_enter_out = true;
                    dim_info.a_reset_out = loop_idx;
                }
                if (!z_left_out and op[dim_info.z_idx_out].out.z_offset !=
                    op[(loop_idx + dim_info.z_idx_out) % loop_num].out.z_offset)
                {
                    z_left_out = true;
                    dim_info.z_wait_out = loop_idx;
                    dim_info.z_stride_out = op[(loop_idx + dim_info.z_idx_out) % loop_num].out.z_offset -
                        op[dim_info.z_idx_out].out.z_offset;
                } else if (z_left_out and !z_enter_out and op[dim_info.z_idx_out].out.z_offset ==
                    op[(loop_idx + dim_info.z_idx_out) % loop_num].out.z_offset)
                {
                    z_enter_out = true;
                    dim_info.z_reset_out = loop_idx;
                }
                if (!y_left_out and op[dim_info.y_idx_out].out.y_offset !=
                    op[(loop_idx + dim_info.y_idx_out) % loop_num].out.y_offset)
                {
                    y_left_out = true;
                    dim_info.y_wait_out = loop_idx;
                    dim_info.y_stride_out = op[(loop_idx + dim_info.y_idx_out) % loop_num].out.y_offset -
                        op[dim_info.y_idx_out].out.y_offset;
                } else if (y_left_out and !y_enter_out and op[dim_info.y_idx_out].out.y_offset ==
                    op[(loop_idx + dim_info.y_idx_out) % loop_num].out.y_offset)
                {
                    y_enter_out = true;
                    dim_info.y_reset_out = loop_idx;
                }
                if (!x_left_out and op[dim_info.x_idx_out].out.x_offset !=
                    op[(loop_idx + dim_info.x_idx_out) % loop_num].out.x_offset)
                {
                    x_left_out = true;
                    dim_info.x_wait_out = loop_idx;
                    dim_info.x_stride_out = op[(loop_idx + dim_info.x_idx_out) % loop_num].out.x_offset -
                        op[dim_info.x_idx_out].out.x_offset;
                } else if (x_left_out and !x_enter_out and op[dim_info.x_idx_out].out.x_offset ==
                    op[(loop_idx + dim_info.x_idx_out) % loop_num].out.x_offset)
                {
                    x_enter_out = true;
                    dim_info.x_reset_out = loop_idx;
                }
                if (!a_left_in and op[dim_info.a_idx_in].in.a_offset !=
                    op[(loop_idx + dim_info.a_idx_in) % loop_num].in.a_offset)
                {
                    a_left_in = true;
                    dim_info.a_wait_in = loop_idx;
                    dim_info.a_stride_in = op[(loop_idx + dim_info.a_idx_in) % loop_num].in.a_offset -
                        op[dim_info.a_idx_in].in.a_offset;
                } else if (a_left_in and !a_enter_in and op[dim_info.a_idx_in].in.a_offset ==
                    op[(loop_idx + dim_info.a_idx_in) % loop_num].in.a_offset)
                {
                    a_enter_in = true;
                    dim_info.a_reset_in = loop_idx;
                }
                if (!z_left_in and op[dim_info.z_idx_in].in.z_offset !=
                    op[(loop_idx + dim_info.z_idx_in) % loop_num].in.z_offset)
                {
                    z_left_in = true;
                    dim_info.z_wait_in = loop_idx;
                    dim_info.z_stride_in = op[(loop_idx + dim_info.z_idx_in) % loop_num].in.z_offset -
                        op[dim_info.z_idx_in].in.z_offset;
                } else if (z_left_in and !z_enter_in and op[dim_info.z_idx_in].in.z_offset ==
                    op[(loop_idx + dim_info.z_idx_in) % loop_num].in.z_offset)
                {
                    z_enter_in = true;
                    dim_info.z_reset_in = loop_idx;
                }
                if (!y_left_in and op[dim_info.y_idx_in].in.y_offset !=
                    op[(loop_idx + dim_info.y_idx_in) % loop_num].in.y_offset)
                {
                    y_left_in = true;
                    dim_info.y_wait_in = loop_idx;
                    dim_info.y_stride_in = op[(loop_idx + dim_info.y_idx_in) % loop_num].in.y_offset -
                        op[dim_info.y_idx_in].in.y_offset;
                } else if (y_left_in and !y_enter_in and op[dim_info.y_idx_in].in.y_offset ==
                    op[(loop_idx + dim_info.y_idx_in) % loop_num].in.y_offset)
                {
                    y_enter_in = true;
                    dim_info.y_reset_in = loop_idx;
                }
                if (!x_left_in and op[dim_info.x_idx_in].in.x_offset !=
                    op[(loop_idx + dim_info.x_idx_in) % loop_num].in.x_offset)
                {
                    x_left_in = true;
                    dim_info.x_wait_in = loop_idx;
                    dim_info.x_stride_in = op[(loop_idx + dim_info.x_idx_in) % loop_num].in.x_offset -
                        op[dim_info.x_idx_in].in.x_offset;
                } else if (x_left_in and !x_enter_in and op[dim_info.x_idx_in].in.x_offset ==
                    op[(loop_idx + dim_info.x_idx_in) % loop_num].in.x_offset)
                {
                    x_enter_in = true;
                    dim_info.x_reset_in = loop_idx;
                }
            }

            return dim_info;
        }
        pub fn print(this: *@This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
            if (name) |text| {
                std.debug.print("{s}DimInfo {s}\n", .{ [1]u8{' '} ** offset, text });
            }
            std.debug.print("{s}str => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
                " " ** (offset + padding),
                this.a_stride_out,
                this.z_stride_out,
                this.y_stride_out,
                this.x_stride_out,
                this.a_stride_out,
                this.z_stride_out,
                this.y_stride_out,
                this.x_stride_out,
            });
            std.debug.print("{s}wai => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
                " " ** (offset + padding),
                this.a_wait_out,
                this.z_wait_out,
                this.y_wait_out,
                this.x_wait_out,
                this.a_wait_out,
                this.z_wait_out,
                this.y_wait_out,
                this.x_wait_out,
            });
            std.debug.print("{s}res => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
                " " ** (offset + padding),
                this.a_reset_out,
                this.z_reset_out,
                this.y_reset_out,
                this.x_reset_out,
                this.a_reset_out,
                this.z_reset_out,
                this.y_reset_out,
                this.x_reset_out,
            });
            std.debug.print("{s}idx => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
                " " ** (offset + padding),
                this.a_idx_out,
                this.z_idx_out,
                this.y_idx_out,
                this.x_idx_out,
                this.a_idx_out,
                this.z_idx_out,
                this.y_idx_out,
                this.x_idx_out,
            });
            //     this.wai_x_in,
            //     this.res_x_in,
            //     this.idx_x_in,
        }
    };
    pub const Assignment = struct {
        type: Op.Type,
        u_var: f32,
        out: Buffer,
        in: Buffer,
        // This is the number of times the out / in buffer has been written too
        layer_out: usize,
        layer_in: usize,
        pub fn print(this: *const @This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
            const op: Op = .{
                .out = this.out,
                .in = this.in,
                .u_var = this.u_var,
                .type = this.type,
            };
            if (name) |text| {
                std.debug.print("{s}Assignment {s}\n", .{ [1]u8{' '} ** offset, text });
            }
            std.debug.print("{s}{} {} ", .{ [1]u8{' '} ** (offset + padding), this.layer_out, this.layer_in });
            op.print(0, 0, null);
        }
        pub fn debug(this: *const @This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
            const op: Op = .{
                .out = this.out,
                .in = this.in,
                .u_var = this.u_var,
                .type = this.type,
            };
            if (name) |text| {
                std.debug.print("{s}Assignment {s}\n", .{ [1]u8{' '} ** offset, text });
            }
            std.debug.print("{s}{} {} ", .{ [1]u8{' '} ** (offset + padding), this.layer_out, this.layer_in });
            op.debug(0, 0, null);
        }
    };
    pub const DepLayer = struct {
        assignment: []Assignment,
        assignment_num: usize,
        assignment_dim: []DimInfo,
    };
    // TODO: This is a hack. I should make some smart way to group loop layers together
    layer: []DepLayer,
    layer_num: usize,
    // 0 in case there is no loop
    layer_loop_id: []usize,
    layer_loop_num: []usize,
    pub fn alloc(allocator: std.mem.Allocator, linearized: Linearized) !Ssa {
        // Don't think hashmaps are avoidable sadly :^(
        var assignment_layer_write = std.AutoHashMap(usize, usize).init(allocator);
        defer assignment_layer_write.deinit();

        var assignment_layer_read = std.AutoHashMap(usize, usize).init(allocator);
        defer assignment_layer_read.deinit();

        const layer_count_op: []usize = try allocator.alloc(usize, linearized.op_num);
        defer allocator.free(layer_count_op);
        @memset(layer_count_op, 0);

        const op_slice: []Op = try allocator.alloc(Op, linearized.op_num);
        defer allocator.free(op_slice);

        var assignment: []?Assignment = try allocator.alloc(?Assignment, linearized.op_num);
        defer allocator.free(assignment);
        @memset(assignment, null);

        var dim_info: []DimInfo = try allocator.alloc(DimInfo, linearized.op_num);
        defer allocator.free(dim_info);

        var layer_loop_id_tracker: usize = 1;
        const layer_loop_num: []usize = try allocator.alloc(usize, linearized.op_num);
        const layer_loop_id: []usize = try allocator.alloc(usize, linearized.op_num);
        @memset(layer_loop_num, 0);
        @memset(layer_loop_id, linearized.op_num);

        var op_search_idx: usize = 0;
        for (0..linearized.op_num) |_| {
            if (op_search_idx == linearized.op_num) {
                break;
            }
            assert(op_search_idx < linearized.op_num);

            var loop_len: usize = 0;
            // TODO: Maybe make this a constant like 4096 or something.
            const loop_len_max: usize = @divFloor(linearized.op_num - op_search_idx, 2);
            if (loop_len_max != 0) {
                for (1..loop_len_max) |loop_search| {
                    const match_potential: bool = linearized.op[op_search_idx].equal(linearized.op[op_search_idx + loop_search]);
                    var match: bool = true;
                    if (match_potential) {
                        for (1..loop_search) |op_idx| {
                            if (linearized.op[op_search_idx + op_idx].equal(linearized.op[op_search_idx + loop_search + op_idx])) {
                                continue;
                            } else {
                                match = false;
                                break;
                            }
                        }
                        if (match) {
                            loop_len = loop_search;
                            break;
                        }
                    }
                }
            }

            var layer_loop_id_curr: usize = 0;

            var loop_num: usize = 1;
            if (loop_len == 0) {
                loop_len = 1;
                layer_loop_id_curr = 0;
            } else {
                const loop_num_max: usize = @divFloor(linearized.op_num - op_search_idx, loop_len);
                for (1..loop_num_max) |loop_idx| {
                    var same: bool = true;
                    for (0..loop_len) |op_idx| {
                        if (!linearized.op[op_search_idx + op_idx].equal(linearized.op[op_search_idx + op_idx + loop_idx * loop_len])) {
                            same = false;
                            break;
                        }
                    }
                    if (same) {
                        loop_num += 1;
                    } else {
                        break;
                    }
                }
                // If this isn't the case some *really* weird things are going on
                assert(loop_num <= loop_num_max);

                layer_loop_id_curr = layer_loop_id_tracker;
                layer_loop_id_tracker += 1;
            }

            for (0..loop_len) |op_inner_idx| {
                const op_idx = op_search_idx + op_inner_idx;

                var op_slice_info: []Op = op_slice[0..loop_num];
                @memset(op_slice_info, undefined);
                for (0..loop_num) |loop_idx| {
                    op_slice_info[loop_idx] = linearized.op[op_idx + loop_idx * loop_len];
                }
                dim_info[op_idx] = DimInfo.init(op_slice_info, loop_num);

                const layer_out: usize = @max(
                    assignment_layer_write.get(linearized.op[op_idx].out.name_offset) orelse 0,
                    assignment_layer_read.get(linearized.op[op_idx].out.name_offset) orelse 0,
                );
                const layer_in: usize = assignment_layer_write.get(linearized.op[op_idx].in.name_offset) orelse 0;
                const layer_idx: usize = @max(layer_out, layer_in);

                assignment[op_idx] = .{
                    .type = linearized.op[op_idx].type,
                    .u_var = linearized.op[op_idx].u_var,
                    .out = linearized.op[op_idx].out,
                    .in = linearized.op[op_idx].in,
                    .layer_out = layer_out,
                    .layer_in = layer_in,
                };

                // This overwrites the data if it already existed
                try assignment_layer_write.put(assignment[op_idx].?.out.name_offset, layer_idx + 1);
                try assignment_layer_read.put(assignment[op_idx].?.in.name_offset, layer_idx + 1);

                assert(layer_loop_id[layer_idx] == linearized.op_num or layer_loop_id[layer_idx] == layer_loop_id_curr);

                layer_count_op[layer_idx] += 1;
                layer_loop_id[layer_idx] = layer_loop_id_curr;
                layer_loop_num[layer_idx] = loop_num;
            }

            op_search_idx += loop_len * loop_num;
        }

        // TODO: Refactor this. This just has to be doable in one loop
        // TODO: When debugging is done uncomment this break and remove the zero_found things.
        var zero_found: bool = false;
        var layer_num: usize = 0;
        for (0..linearized.op_num) |op_idx| {
            if (layer_count_op[op_idx] == 0) {
                // break;
                zero_found = true;
            } else {
                assert(!zero_found);
                layer_num = op_idx + 1;
            }
        }

        zero_found = false;
        const layer = try allocator.alloc(DepLayer, layer_num);
        for (0..layer_num) |layer_idx| {
            if (layer_count_op[layer_idx] == 0) {
                // break;
                zero_found = true;
            } else {
                assert(!zero_found);
                layer[layer_idx].assignment_num = 0;
                layer[layer_idx].assignment = try allocator.alloc(Assignment, layer_count_op[layer_idx]);
                layer[layer_idx].assignment_dim = try allocator.alloc(DimInfo, layer_count_op[layer_idx]);
            }
        }

        for (0..linearized.op_num) |op_idx| {
            if (assignment[op_idx]) |assignment_curr| {
                const layer_idx: usize = @max(assignment_curr.layer_out, assignment_curr.layer_in);
                if (layer_count_op[layer_idx] == 0) {
                    break;
                } else {
                    layer[layer_idx].assignment[layer[layer_idx].assignment_num] = assignment_curr;
                    layer[layer_idx].assignment_dim[layer[layer_idx].assignment_num] = dim_info[op_idx];
                    layer[layer_idx].assignment_num += 1;
                }
            }
        }

        return .{
            .layer = layer,
            .layer_num = layer.len,
            .layer_loop_id = layer_loop_id,
            .layer_loop_num = layer_loop_num,
        };
    }
    pub fn free(this: *@This(), allocator: std.mem.Allocator) void {
        for (0..this.layer_num) |layer_idx| {
            allocator.free(this.layer[layer_idx].assignment);
            allocator.free(this.layer[layer_idx].assignment_dim);
        }
        allocator.free(this.layer);
        allocator.free(this.layer_loop_num);
        allocator.free(this.layer_loop_id);
    }
    pub fn print(this: *const @This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}SSA {s}\n", .{ [1]u8{' '} ** offset, text });
        } else {
            std.debug.print("{s}SSA\n", .{[1]u8{' '} ** offset});
        }
        for (0..this.layer_num) |layer_idx| {
            std.debug.print("{s}Layer idx={} of {}: id={} repeat_num={}\n", .{
                [1]u8{' '} ** (offset + padding),
                layer_idx,
                this.layer_num,
                this.layer_loop_id[layer_idx],
                this.layer_loop_num[layer_idx],
            });
            for (0..this.layer[layer_idx].assignment_num) |assignment_idx| {
                std.debug.print("{s}[{}] ", .{ [1]u8{' '} ** (offset + padding), assignment_idx });
                this.layer[layer_idx].assignment[assignment_idx].print(padding, offset + padding, null);
                this.layer[layer_idx].assignment_dim[assignment_idx].print(padding, offset + padding, null);
            }
        }
    }
    pub fn debug(this: *const @This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}SSA {s}\n", .{ [1]u8{' '} ** offset, text });
        } else {
            std.debug.print("{s}SSA\n", .{[1]u8{' '} ** offset});
        }
        for (0..this.layer_num) |layer_idx| {
            std.debug.print("{s}Layer idx={} of {}: id={} repeat_num={}\n", .{
                [1]u8{' '} ** (offset + padding),
                layer_idx,
                this.layer_num,
                this.layer_loop_id[layer_idx],
                this.layer_loop_num[layer_idx],
            });
            for (0..this.layer[layer_idx].assignment_num) |assignment_idx| {
                std.debug.print("{s}[{}] ", .{ [1]u8{' '} ** (offset + padding), assignment_idx });
                this.layer[layer_idx].assignment[assignment_idx].debug(0, 0, null);
                this.layer[layer_idx].assignment_dim[assignment_idx].print(padding, offset + padding, null);
            }
        }
    }
};
