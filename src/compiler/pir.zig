// PIR = Parallel intermediate representation

const Op = @import("../tensor.zig").Op;
const Linearized = @import("../tensor.zig").Linearized;
const buffer_name_size: u32 = @import("../tensor.zig").buffer_name_size;
const assert = @import("../util.zig").assert;

const DimInfo = struct {
    off_in: u32,
    off_out: u32,
    str_a_out: u32,
    str_z_out: u32,
    str_y_out: u32,
    str_x_out: u32,
    str_a_in: u32,
    str_z_in: u32,
    str_y_in: u32,
    str_x_in: u32,
    wai_a_out: u32,
    wai_z_out: u32,
    wai_y_out: u32,
    wai_x_out: u32,
    wai_a_in: u32,
    wai_z_in: u32,
    wai_y_in: u32,
    wai_x_in: u32,
    res_a_out: u32,
    res_z_out: u32,
    res_y_out: u32,
    res_x_out: u32,
    res_a_in: u32,
    res_z_in: u32,
    res_y_in: u32,
    res_x_in: u32,
};

pub const Pir = struct {
    repeat_num: u32,
    op_num: u32,
    op: []Op,
    dim_info: []DimInfo,
    pub fn alloc(allocator: anytype, linearized: Linearized, op_used: *u32) !Pir {
        assert(op_used.* < linearized.op_num - 1);
        var op_num: u32 = 0;
        var group_num: u32 = 1;
        const op_start: u32 = op_used.*;

        for (1..linearized.op_num - op_start) |op_off| {
            if (linearized.op[op_start].equal(linearized.op[op_start + op_off]) and
                !linearized.op[op_start].overlaps(linearized.op[op_start + op_off]))
            {
                var all_same: bool = true;
                for (1..op_off) |inner_off| {
                    if (!linearized.op[op_start + inner_off].equal(linearized.op[op_start + op_off + inner_off]) or
                        linearized.op[op_start + inner_off].overlaps(linearized.op[op_start + op_off + inner_off]))
                    {
                        all_same = false;
                        break;
                    }
                }
                if (all_same) {
                    op_num = @as(u32, @intCast(op_off));
                    break;
                } else {
                    continue;
                }
            }
        }
        if (op_num == 0) {
            op_num = 1;
        } else {
            const group_num_max: u32 = @divFloor(linearized.op_num - op_num - op_start + 1, op_num);
            for (1..group_num_max) |group_idx| {
                var all_equal: bool = true;
                for (0..op_num) |inner_off| {
                    // TODO: Need to check for no overlap here probably
                    if (linearized.op[op_start + inner_off].equal(linearized.op[op_start + inner_off + group_idx * op_num])) {
                        continue;
                    } else {
                        all_equal = false;
                        break;
                    }
                }
                if (all_equal) {
                    group_num += 1;
                } else {
                    break;
                }
            }
        }
        const op: []Op = try allocator.alloc(Op, op_num * group_num);
        const dim_info: []DimInfo = try allocator.alloc(DimInfo, op_num * group_num);
        for (0..op_num) |op_idx| {
            op[op_idx] = linearized.op[op_start + op_idx];
        }

        for (0..op_num) |op_idx| {
            dim_info[op_idx].off_out = linearized.op[op_start + op_idx].out.offset;
            const a_out_initial = linearized.op[op_start + op_idx].out.a_offset;
            const z_out_initial = linearized.op[op_start + op_idx].out.z_offset;
            const y_out_initial = linearized.op[op_start + op_idx].out.y_offset;
            const x_out_initial = linearized.op[op_start + op_idx].out.x_offset;
            var a_out_left: bool = false;
            var z_out_left: bool = false;
            var y_out_left: bool = false;
            var x_out_left: bool = false;
            var a_out_reenter: bool = false;
            var z_out_reenter: bool = false;
            var y_out_reenter: bool = false;
            var x_out_reenter: bool = false;

            const a_in_initial = linearized.op[op_start + op_idx].in.a_offset;
            const z_in_initial = linearized.op[op_start + op_idx].in.z_offset;
            const y_in_initial = linearized.op[op_start + op_idx].in.y_offset;
            const x_in_initial = linearized.op[op_start + op_idx].in.x_offset;
            var a_in_left: bool = false;
            var z_in_left: bool = false;
            var y_in_left: bool = false;
            var x_in_left: bool = false;
            var a_in_reenter: bool = false;
            var z_in_reenter: bool = false;
            var y_in_reenter: bool = false;
            var x_in_reenter: bool = false;

            for (0..group_num) |group_idx| {
                if (a_out_left) {
                    if (!a_out_reenter) {
                        if (linearized.op[op_start + op_idx + group_idx * op_num].out.a_offset == a_out_initial) {
                            dim_info[op_idx].res_a_out = group_idx;
                            a_out_reenter = true;
                        }
                    }
                } else {
                    if (linearized.op[op_start + op_idx + group_idx * op_num].out.a_offset != a_out_initial) {
                        dim_info[op_idx].wai_a_out = group_idx;
                        dim_info[op_idx].str_a_out = linearized.op[op_start + op_idx + group_idx * op_num].out.a_offset - a_out_initial;
                        a_out_left = true;
                    }
                }
                if (z_out_left) {
                    if (!z_out_reenter) {
                        if (linearized.op[op_start + op_idx + group_idx * op_num].out.z_offset == z_out_initial) {
                            dim_info[op_idx].res_z_out = group_idx;
                            z_out_reenter = true;
                        }
                    }
                } else {
                    if (linearized.op[op_start + op_idx + group_idx * op_num].out.z_offset != z_out_initial) {
                        dim_info[op_idx].wai_z_out = group_idx;
                        dim_info[op_idx].str_z_out = linearized.op[op_start + op_idx + group_idx * op_num].out.z_offset - z_out_initial;
                        z_out_left = true;
                    }
                }
                if (y_out_left) {
                    if (!y_out_reenter) {
                        if (linearized.op[op_start + op_idx + group_idx * op_num].out.y_offset == y_out_initial) {
                            dim_info[op_idx].res_y_out = group_idx;
                            y_out_reenter = true;
                        }
                    }
                } else {
                    if (linearized.op[op_start + op_idx + group_idx * op_num].out.y_offset != y_out_initial) {
                        dim_info[op_idx].wai_y_out = group_idx;
                        dim_info[op_idx].str_y_out = linearized.op[op_start + op_idx + group_idx * op_num].out.y_offset - y_out_initial;
                        y_out_left = true;
                    }
                }
                if (x_out_left) {
                    if (!x_out_reenter) {
                        if (linearized.op[op_start + op_idx + group_idx * op_num].out.x_offset == x_out_initial) {
                            dim_info[op_idx].res_x_out = group_idx;
                            x_out_reenter = true;
                        }
                    }
                } else {
                    if (linearized.op[op_start + op_idx + group_idx * op_num].out.x_offset != x_out_initial) {
                        dim_info[op_idx].wai_x_out = group_idx;
                        dim_info[op_idx].str_x_out = linearized.op[op_start + op_idx + group_idx * op_num].out.x_offset - x_out_initial;
                        x_out_left = true;
                    }
                }
                if (a_in_left) {
                    if (!a_in_reenter) {
                        if (linearized.op[op_start + op_idx + group_idx * op_num].in.a_offset == a_in_initial) {
                            dim_info[op_idx].res_a_in = group_idx;
                            a_in_reenter = true;
                        }
                    }
                } else {
                    if (linearized.op[op_start + op_idx + group_idx * op_num].in.a_offset != a_in_initial) {
                        dim_info[op_idx].wai_a_in = group_idx;
                        dim_info[op_idx].str_a_in = linearized.op[op_start + op_idx + group_idx * op_num].in.a_offset - a_in_initial;
                        a_in_left = true;
                    }
                }
                if (z_in_left) {
                    if (!z_in_reenter) {
                        if (linearized.op[op_start + op_idx + group_idx * op_num].in.z_offset == z_in_initial) {
                            dim_info[op_idx].res_z_in = group_idx;
                            z_in_reenter = true;
                        }
                    }
                } else {
                    if (linearized.op[op_start + op_idx + group_idx * op_num].in.z_offset != z_in_initial) {
                        dim_info[op_idx].wai_z_in = group_idx;
                        dim_info[op_idx].str_z_in = linearized.op[op_start + op_idx + group_idx * op_num].in.z_offset - z_in_initial;
                        z_in_left = true;
                    }
                }
                if (y_in_left) {
                    if (!y_in_reenter) {
                        if (linearized.op[op_start + op_idx + group_idx * op_num].in.y_offset == y_in_initial) {
                            dim_info[op_idx].res_y_in = group_idx;
                            y_in_reenter = true;
                        }
                    }
                } else {
                    if (linearized.op[op_start + op_idx + group_idx * op_num].in.y_offset != y_in_initial) {
                        dim_info[op_idx].wai_y_in = group_idx;
                        dim_info[op_idx].str_y_in = linearized.op[op_start + op_idx + group_idx * op_num].in.y_offset - y_in_initial;
                        y_in_left = true;
                    }
                }
                if (x_in_left) {
                    if (!x_in_reenter) {
                        if (linearized.op[op_start + op_idx + group_idx * op_num].in.x_offset == x_in_initial) {
                            dim_info[op_idx].res_x_in = group_idx;
                            x_in_reenter = true;
                        }
                    }
                } else {
                    if (linearized.op[op_start + op_idx + group_idx * op_num].in.x_offset != x_in_initial) {
                        dim_info[op_idx].wai_x_in = group_idx;
                        dim_info[op_idx].str_x_in = linearized.op[op_start + op_idx + group_idx * op_num].in.x_offset - x_in_initial;
                        x_in_left = true;
                    }
                }
            }
        }

        op_used.* += op_num * group_num;

        return .{
            .op_num = op_num,
            .op = op,
            .repeat_num = group_num,
            .dim_info = dim_info,
        };
    }
    pub fn free(this: *@This(), allocator: anytype) void {
        allocator.free(this.op);
        allocator.free(this.dim_info);
    }
    pub fn print(this: *@This(), writer: anytype, comptime padding: u32, comptime offset: u32, name: ?[]u8) !void {
        if (name) |text| {
            try writer.print("{s}PIR = {s}\n", .{ " " ** offset, text });
        }
        for (0..this.op_num) |op_idx| {
            try writer.print("{s}[{}] => ", .{ " " ** (offset + padding), op_idx });
            try this.op[op_idx].print(writer, 0, 0, null);
            try writer.print("{s}off => out({d:4}) in({d:4})\n", .{
                " " ** (offset + 2 * padding),
                this.dim_info[op_idx].off_out,
                this.dim_info[op_idx].off_in,
            });
            try writer.print("{s}str => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
                " " ** (offset + 2 * padding),
                this.dim_info[op_idx].str_a_out,
                this.dim_info[op_idx].str_z_out,
                this.dim_info[op_idx].str_y_out,
                this.dim_info[op_idx].str_x_out,
                this.dim_info[op_idx].str_a_in,
                this.dim_info[op_idx].str_z_in,
                this.dim_info[op_idx].str_y_in,
                this.dim_info[op_idx].str_x_in,
            });
            try writer.print("{s}wai => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
                " " ** (offset + 2 * padding),
                this.dim_info[op_idx].wai_a_out,
                this.dim_info[op_idx].wai_z_out,
                this.dim_info[op_idx].wai_y_out,
                this.dim_info[op_idx].wai_x_out,
                this.dim_info[op_idx].wai_a_in,
                this.dim_info[op_idx].wai_z_in,
                this.dim_info[op_idx].wai_y_in,
                this.dim_info[op_idx].wai_x_in,
            });
            try writer.print("{s}res => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
                " " ** (offset + 2 * padding),
                this.dim_info[op_idx].res_a_out,
                this.dim_info[op_idx].res_z_out,
                this.dim_info[op_idx].res_y_out,
                this.dim_info[op_idx].res_x_out,
                this.dim_info[op_idx].res_a_in,
                this.dim_info[op_idx].res_z_in,
                this.dim_info[op_idx].res_y_in,
                this.dim_info[op_idx].res_x_in,
            });
        }
    }
};
