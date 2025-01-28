// PIR = Parallel intermediate representation

const std = @import("std");

const Op = @import("../tensor.zig").Op;
const Linearized = @import("../tensor.zig").Linearized;
const Optimization = @import("./codegen.zig").Optimization;
const buffer_name_size: usize = @import("../tensor.zig").buffer_name_size;
const assert = std.debug.assert;

// To fix that I actually need to do the dreaded sorting... that won't be fun...
// Also just put the DimInfo gathering in this struct
pub const DimInfo = struct {
    str_a_out: usize,
    str_z_out: usize,
    str_y_out: usize,
    str_x_out: usize,
    str_a_in: usize,
    str_z_in: usize,
    str_y_in: usize,
    str_x_in: usize,
    wai_a_out: usize,
    wai_z_out: usize,
    wai_y_out: usize,
    wai_x_out: usize,
    wai_a_in: usize,
    wai_z_in: usize,
    wai_y_in: usize,
    wai_x_in: usize,
    res_a_out: usize,
    res_z_out: usize,
    res_y_out: usize,
    res_x_out: usize,
    res_a_in: usize,
    res_z_in: usize,
    res_y_in: usize,
    res_x_in: usize,
    off_a_out: usize,
    off_z_out: usize,
    off_y_out: usize,
    off_x_out: usize,
    off_a_in: usize,
    off_z_in: usize,
    off_y_in: usize,
    off_x_in: usize,
    // Effectively the offset of the global_id when compiling (offsetting happens per dimension)
    // TODO: This is a terrible name
    idx_a_out: usize,
    idx_z_out: usize,
    idx_y_out: usize,
    idx_x_out: usize,
    idx_a_in: usize,
    idx_z_in: usize,
    idx_y_in: usize,
    idx_x_in: usize,
    pub fn alloc(linearized: Linearized, op_start: usize, op_num: usize, repeat_num: usize) DimInfo {
        var idx_a_out: usize = 0;
        var idx_z_out: usize = 0;
        var idx_y_out: usize = 0;
        var idx_x_out: usize = 0;
        var idx_a_in: usize = 0;
        var idx_z_in: usize = 0;
        var idx_y_in: usize = 0;
        var idx_x_in: usize = 0;
        for (1..repeat_num) |repeat_idx| {
            if (linearized.op[op_start + idx_a_out * op_num].out.a_offset > linearized.op[op_start + repeat_idx * op_num].out.a_offset) {
                idx_a_out = repeat_idx;
            }
            if (linearized.op[op_start + idx_z_out * op_num].out.z_offset > linearized.op[op_start + repeat_idx * op_num].out.z_offset) {
                idx_z_out = repeat_idx;
            }
            if (linearized.op[op_start + idx_y_out * op_num].out.y_offset > linearized.op[op_start + repeat_idx * op_num].out.y_offset) {
                idx_y_out = repeat_idx;
            }
            if (linearized.op[op_start + idx_x_out * op_num].out.x_offset > linearized.op[op_start + repeat_idx * op_num].out.x_offset) {
                idx_x_out = repeat_idx;
            }
            if (linearized.op[op_start + idx_a_in * op_num].in.a_offset > linearized.op[op_start + repeat_idx * op_num].in.a_offset) {
                idx_a_in = repeat_idx;
            }
            if (linearized.op[op_start + idx_z_in * op_num].in.z_offset > linearized.op[op_start + repeat_idx * op_num].in.z_offset) {
                idx_z_in = repeat_idx;
            }
            if (linearized.op[op_start + idx_y_in * op_num].in.y_offset > linearized.op[op_start + repeat_idx * op_num].in.y_offset) {
                idx_y_in = repeat_idx;
            }
            if (linearized.op[op_start + idx_x_in * op_num].in.x_offset > linearized.op[op_start + repeat_idx * op_num].in.x_offset) {
                idx_x_in = repeat_idx;
            }
        }
        var str_a_out: usize = 0;
        var str_z_out: usize = 0;
        var str_y_out: usize = 0;
        var str_x_out: usize = 0;
        var str_a_in: usize = 0;
        var str_z_in: usize = 0;
        var str_y_in: usize = 0;
        var str_x_in: usize = 0;
        var wai_a_out: usize = 0;
        var wai_z_out: usize = 0;
        var wai_y_out: usize = 0;
        var wai_x_out: usize = 0;
        var wai_a_in: usize = 0;
        var wai_z_in: usize = 0;
        var wai_y_in: usize = 0;
        var wai_x_in: usize = 0;
        var res_a_out: usize = 0;
        var res_z_out: usize = 0;
        var res_y_out: usize = 0;
        var res_x_out: usize = 0;
        var res_a_in: usize = 0;
        var res_z_in: usize = 0;
        var res_y_in: usize = 0;
        var res_x_in: usize = 0;

        var a_out_left: bool = false;
        var z_out_left: bool = false;
        var y_out_left: bool = false;
        var x_out_left: bool = false;
        var a_in_left: bool = false;
        var z_in_left: bool = false;
        var y_in_left: bool = false;
        var x_in_left: bool = false;
        var a_out_reenter: bool = false;
        var z_out_reenter: bool = false;
        var y_out_reenter: bool = false;
        var x_out_reenter: bool = false;
        var a_in_reenter: bool = false;
        var z_in_reenter: bool = false;
        var y_in_reenter: bool = false;
        var x_in_reenter: bool = false;

        for (1..repeat_num) |repeat_idx| {
            if (a_out_left and !a_out_reenter and linearized.op[op_start + ((idx_a_out + repeat_idx) % repeat_num) * op_num].out.a_offset ==
                linearized.op[op_start + idx_a_out * op_num].out.a_offset)
            {
                a_out_reenter = true;
                res_a_out = repeat_idx;
            } else if (!a_out_left and !a_out_reenter and linearized.op[op_start + ((idx_a_out + repeat_idx) % repeat_num) * op_num].out.a_offset !=
                linearized.op[op_start + idx_a_out * op_num].out.a_offset)
            {
                a_out_left = true;
                str_a_out = linearized.op[op_start + ((idx_a_out + repeat_idx) % repeat_num) * op_num].out.a_offset -
                    linearized.op[op_start + idx_a_out * op_num].out.a_offset;
                wai_a_out = repeat_idx;
            }
            if (z_out_left and !z_out_reenter and linearized.op[op_start + ((idx_z_out + repeat_idx) % repeat_num) * op_num].out.z_offset ==
                linearized.op[op_start + idx_z_out * op_num].out.z_offset)
            {
                z_out_reenter = true;
                res_z_out = repeat_idx;
            } else if (!z_out_left and !z_out_reenter and linearized.op[op_start + ((idx_z_out + repeat_idx) % repeat_num) * op_num].out.z_offset !=
                linearized.op[op_start + idx_z_out * op_num].out.z_offset)
            {
                z_out_left = true;
                str_z_out = linearized.op[op_start + ((idx_z_out + repeat_idx) % repeat_num) * op_num].out.z_offset -
                    linearized.op[op_start + idx_z_out * op_num].out.z_offset;
                wai_z_out = repeat_idx;
            }
            if (y_out_left and !y_out_reenter and linearized.op[op_start + ((idx_y_out + repeat_idx) % repeat_num) * op_num].out.y_offset ==
                linearized.op[op_start + idx_y_out * op_num].out.y_offset)
            {
                y_out_reenter = true;
                res_y_out = repeat_idx;
            } else if (!y_out_left and !y_out_reenter and linearized.op[op_start + ((idx_y_out + repeat_idx) % repeat_num) * op_num].out.y_offset !=
                linearized.op[op_start + idx_y_out * op_num].out.y_offset)
            {
                y_out_left = true;
                str_y_out = linearized.op[op_start + ((idx_y_out + repeat_idx) % repeat_num) * op_num].out.y_offset -
                    linearized.op[op_start + idx_y_out * op_num].out.y_offset;
                wai_y_out = repeat_idx;
            }
            if (x_out_left and !x_out_reenter and linearized.op[op_start + ((idx_x_out + repeat_idx) % repeat_num) * op_num].out.x_offset ==
                linearized.op[op_start + idx_x_out * op_num].out.x_offset)
            {
                x_out_reenter = true;
                res_x_out = repeat_idx;
            } else if (!x_out_left and !x_out_reenter and linearized.op[op_start + ((idx_x_out + repeat_idx) % repeat_num) * op_num].out.x_offset !=
                linearized.op[op_start + idx_x_out * op_num].out.x_offset)
            {
                x_out_left = true;
                str_x_out = linearized.op[op_start + ((idx_x_out + repeat_idx) % repeat_num) * op_num].out.x_offset -
                    linearized.op[op_start + idx_x_out * op_num].out.x_offset;
                wai_x_out = repeat_idx;
            }
            if (a_in_left and !a_in_reenter and linearized.op[op_start + ((idx_a_in + repeat_idx) % repeat_num) * op_num].in.a_offset ==
                linearized.op[op_start + idx_a_in * op_num].in.a_offset)
            {
                a_in_reenter = true;
                res_a_in = repeat_idx;
            } else if (!a_in_left and !a_in_reenter and linearized.op[op_start + ((idx_a_in + repeat_idx) % repeat_num) * op_num].in.a_offset !=
                linearized.op[op_start + idx_a_in * op_num].in.a_offset)
            {
                a_in_left = true;
                str_a_in = linearized.op[op_start + ((idx_a_in + repeat_idx) % repeat_num) * op_num].in.a_offset -
                    linearized.op[op_start + idx_a_in * op_num].in.a_offset;
                wai_a_in = repeat_idx;
            }
            if (z_in_left and !z_in_reenter and linearized.op[op_start + ((idx_z_in + repeat_idx) % repeat_num) * op_num].in.z_offset ==
                linearized.op[op_start + idx_z_in * op_num].in.z_offset)
            {
                z_in_reenter = true;
                res_z_in = repeat_idx;
            } else if (!z_in_left and !z_in_reenter and linearized.op[op_start + ((idx_z_in + repeat_idx) % repeat_num) * op_num].in.z_offset !=
                linearized.op[op_start + idx_z_in * op_num].in.z_offset)
            {
                z_in_left = true;
                str_z_in = linearized.op[op_start + ((idx_z_in + repeat_idx) % repeat_num) * op_num].in.z_offset -
                    linearized.op[op_start + idx_z_in * op_num].in.z_offset;
                wai_z_in = repeat_idx;
            }
            if (y_in_left and !y_in_reenter and linearized.op[op_start + ((idx_y_in + repeat_idx) % repeat_num) * op_num].in.y_offset ==
                linearized.op[op_start + idx_y_in * op_num].in.y_offset)
            {
                y_in_reenter = true;
                res_y_in = repeat_idx;
            } else if (!y_in_left and !y_in_reenter and linearized.op[op_start + ((idx_y_in + repeat_idx) % repeat_num) * op_num].in.y_offset !=
                linearized.op[op_start + idx_y_in * op_num].in.y_offset)
            {
                y_in_left = true;
                str_y_in = linearized.op[op_start + ((idx_y_in + repeat_idx) % repeat_num) * op_num].in.y_offset -
                    linearized.op[op_start + idx_y_in * op_num].in.y_offset;
                wai_y_in = repeat_idx;
            }
            if (x_in_left and !x_in_reenter and linearized.op[op_start + ((idx_x_in + repeat_idx) % repeat_num) * op_num].in.x_offset ==
                linearized.op[op_start + idx_x_in * op_num].in.x_offset)
            {
                x_in_reenter = true;
                res_x_in = repeat_idx;
            } else if (!x_in_left and !x_in_reenter and linearized.op[op_start + ((idx_x_in + repeat_idx) % repeat_num) * op_num].in.x_offset !=
                linearized.op[op_start + idx_x_in * op_num].in.x_offset)
            {
                x_in_left = true;
                str_x_in = linearized.op[op_start + ((idx_x_in + repeat_idx) % repeat_num) * op_num].in.x_offset -
                    linearized.op[op_start + idx_x_in * op_num].in.x_offset;
                wai_x_in = repeat_idx;
            }
        }

        if (!a_out_left) {
            res_a_out = 1;
            wai_a_out = 1;
            str_a_out = 0;
        } else {
            if (!a_out_reenter) {
                res_a_out = repeat_num;
            }
        }
        if (!z_out_left) {
            res_z_out = 1;
            wai_z_out = 1;
            str_z_out = 0;
        } else {
            if (!z_out_reenter) {
                res_z_out = repeat_num;
            }
        }
        if (!y_out_left) {
            res_y_out = 1;
            wai_y_out = 1;
            str_y_out = 0;
        } else {
            if (!y_out_reenter) {
                res_y_out = repeat_num;
            }
        }
        if (!x_out_left) {
            res_x_out = 1;
            wai_x_out = 1;
            str_x_out = 0;
        } else {
            if (!x_out_reenter) {
                res_x_out = repeat_num;
            }
        }
        if (!a_in_left) {
            res_a_in = 1;
            wai_a_in = 1;
            str_a_in = 0;
        } else {
            if (!a_in_reenter) {
                res_a_in = repeat_num;
            }
        }
        if (!z_in_left) {
            res_z_in = 1;
            wai_z_in = 1;
            str_z_in = 0;
        } else {
            if (!z_in_reenter) {
                res_z_in = repeat_num;
            }
        }
        if (!y_in_left) {
            res_y_in = 1;
            wai_y_in = 1;
            str_y_in = 0;
        } else {
            if (!y_in_reenter) {
                res_y_in = repeat_num;
            }
        }
        if (!x_in_left) {
            res_x_in = 1;
            wai_x_in = 1;
            str_x_in = 0;
        } else {
            if (!x_in_reenter) {
                res_x_in = repeat_num;
            }
        }

        return .{
            .str_a_out = str_a_out,
            .str_z_out = str_z_out,
            .str_y_out = str_y_out,
            .str_x_out = str_x_out,
            .str_a_in = str_a_in,
            .str_z_in = str_z_in,
            .str_y_in = str_y_in,
            .str_x_in = str_x_in,
            .wai_a_out = wai_a_out,
            .wai_z_out = wai_z_out,
            .wai_y_out = wai_y_out,
            .wai_x_out = wai_x_out,
            .wai_a_in = wai_a_in,
            .wai_z_in = wai_z_in,
            .wai_y_in = wai_y_in,
            .wai_x_in = wai_x_in,
            .res_a_out = res_a_out,
            .res_z_out = res_z_out,
            .res_y_out = res_y_out,
            .res_x_out = res_x_out,
            .res_a_in = res_a_in,
            .res_z_in = res_z_in,
            .res_y_in = res_y_in,
            .res_x_in = res_x_in,
            .off_a_out = linearized.op[op_start + idx_a_out * op_num].out.a_offset,
            .off_z_out = linearized.op[op_start + idx_z_out * op_num].out.z_offset,
            .off_y_out = linearized.op[op_start + idx_y_out * op_num].out.y_offset,
            .off_x_out = linearized.op[op_start + idx_x_out * op_num].out.x_offset,
            .off_a_in = linearized.op[op_start + idx_a_in * op_num].in.a_offset,
            .off_z_in = linearized.op[op_start + idx_z_in * op_num].in.z_offset,
            .off_y_in = linearized.op[op_start + idx_y_in * op_num].in.y_offset,
            .off_x_in = linearized.op[op_start + idx_x_in * op_num].in.x_offset,
            .idx_a_out = idx_a_out,
            .idx_z_out = idx_z_out,
            .idx_y_out = idx_y_out,
            .idx_x_out = idx_x_out,
            .idx_a_in = idx_a_in,
            .idx_z_in = idx_z_in,
            .idx_y_in = idx_y_in,
            .idx_x_in = idx_x_in,
        };
    }
};

pub const Pir = struct {
    pub const Inline = enum(u8) {
        none,
        in,
        out,
    };
    repeat_num: usize,
    op_num: usize,
    op: []Op,
    dim_info: []DimInfo,
    inline_type: []Pir.Inline,
    pub fn alloc(allocator: anytype, linearized: Linearized, op_used: *usize) !Pir {
        assert(op_used.* < linearized.op_num);
        var op_num: usize = 1;
        var repeat_num: usize = 1;
        const op_start: usize = op_used.*;
        var inline_type: []Pir.Inline = try allocator.alloc(Pir.Inline, linearized.op_num - op_start);
        errdefer allocator.free(inline_type);
        for (0..linearized.op_num - op_start) |op_idx| {
            inline_type[op_idx] = .none;
        }

        // TODO: Put all inlineable ops in the kernel even in case no loop is found

        for (op_start + 1..linearized.op_num) |op_off| {
            if (linearized.op[op_start].equal(linearized.op[op_off]) and !linearized.op[op_start].overlaps(linearized.op[op_off])) {
                var all_same: bool = true;
                for (0..op_off - op_start) |inner_off| {
                    // TODO: To fix the current convolution thing with overlaps I also need to check that it only cancels if the significant tensor (last out tensor in group) overlaps
                    // TODO: If I do this significant tensor thing then ther probably should be a way to mark a tensor as significant
                    if (!linearized.op[op_start + inner_off].equal(linearized.op[op_off + inner_off]) or
                        linearized.op[op_start + inner_off].overlaps(linearized.op[op_off + inner_off]))
                    {
                        all_same = false;
                        break;
                    }
                }
                if (all_same) {
                    op_num = op_off - op_start;
                    break;
                } else {
                    continue;
                }
            }
        }
        const repeat_num_max: usize = @divFloor(linearized.op_num - (op_num + op_start), op_num) + 1;
        for (1..repeat_num_max) |repeat_idx| {
            var all_equal: bool = true;
            for (0..op_num) |inner_off| {
                if (linearized.op[op_start + inner_off].equal(linearized.op[op_start + inner_off + repeat_idx * op_num]) and
                    !linearized.op[op_start + inner_off].overlaps(linearized.op[op_start + inner_off + repeat_idx * op_num]))
                {
                    continue;
                } else {
                    all_equal = false;
                    break;
                }
            }
            if (all_equal) {
                // TODO: This really needs to be optimized. Four dim loops should really be avoided
                var overlap: bool = false;
                for (0..repeat_num + 1) |outer_idx| outer: {
                    for (outer_idx + 1..repeat_num + 1) |inner_idx| {
                        if (outer_idx == inner_idx) {
                            continue;
                        }
                        for (0..op_num) |outer_off| {
                            for (0..op_num) |inner_off| {
                                const overlap_valid: bool = linearized.op[op_start + outer_off + outer_idx * op_num].out.name_offset ==
                                    linearized.op[op_start + inner_off + inner_idx * op_num].out.name_offset and
                                    linearized.op[op_start + outer_off + outer_idx * op_num].out.a_size ==
                                    linearized.op[op_start + inner_off + inner_idx * op_num].out.a_size and
                                    linearized.op[op_start + outer_off + outer_idx * op_num].out.z_size ==
                                    linearized.op[op_start + inner_off + inner_idx * op_num].out.z_size and
                                    linearized.op[op_start + outer_off + outer_idx * op_num].out.y_size ==
                                    linearized.op[op_start + inner_off + inner_idx * op_num].out.y_size and
                                    linearized.op[op_start + outer_off + outer_idx * op_num].out.x_size ==
                                    linearized.op[op_start + inner_off + inner_idx * op_num].out.x_size;
                                if (overlap_valid and
                                    linearized.op[op_start + outer_off + outer_idx * op_num].overlaps(linearized.op[op_start + inner_off + inner_idx * op_num]))
                                {
                                    overlap = true;
                                    break :outer;
                                }
                            }
                        }
                    }
                }
                overlap = overlap;

                if (overlap) {
                    break;
                } else {
                    repeat_num += 1;
                }
            } else {
                break;
            }
        }
        const op: []Op = try allocator.alloc(Op, op_num);
        const dim_info: []DimInfo = try allocator.alloc(DimInfo, op_num);
        for (0..op_num) |op_idx| {
            op[op_idx] = linearized.op[op_start + op_idx];
        }

        for (0..op_num) |op_idx| {
            dim_info[op_idx] = DimInfo.alloc(linearized, op_start + op_idx, op_num, repeat_num);
        }

        op_used.* += op_num * repeat_num;

        return .{
            .op_num = op_num,
            .op = op,
            .repeat_num = repeat_num,
            .dim_info = dim_info,
            .inline_type = inline_type,
        };
    }
    pub fn free(this: *@This(), allocator: anytype) void {
        allocator.free(this.op);
        allocator.free(this.dim_info);
        allocator.free(this.inline_type);
    }
    pub fn optimize(this: *@This(), optimization: Optimization) void {
        // Structured like this to make debugging easier
        switch (optimization) {
            .O0 => {},
            .O1 => {
                this.optimizeInline();
            },
            .O2 => {
                this.optimizeInline();
            },
            .O3 => {
                this.optimizeInline();
            },
        }
    }
    pub fn optimizeInline(this: *@This()) void {
        for (1..this.op_num) |op_idx| {
            if (this.op[op_idx - 1].isOutInlinable(&this.op[op_idx])) {
                this.inline_type[op_idx] = .out;
            } else if (this.op[op_idx - 1].isInInlinable(&this.op[op_idx])) {
                this.inline_type[op_idx] = .in;
            } else {
                this.inline_type[op_idx] = .none;
            }
        }
    }
    pub fn print(this: *const @This(), comptime padding: usize, comptime offset: usize, name: ?[]const u8) void {
        if (name) |text| {
            std.debug.print("{s}PIR = {s}\n", .{ " " ** offset, text });
        }
        for (0..this.op_num) |op_idx| {
            std.debug.print("{s}[{}] => ", .{ " " ** (offset + padding), op_idx });
            this.op[op_idx].print(0, 0, null);
            // std.debug.print("{s}off => out({d:4}) in({d:4})\n", .{
            //     " " ** (offset + 2 * padding),
            //     this.dim_info[op_idx].off_out,
            //     this.dim_info[op_idx].off_in,
            // });
            std.debug.print("{s}Repeats {}\n", .{ " " ** (offset + 2 * padding), this.repeat_num });
            std.debug.print("{s}str => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
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
            std.debug.print("{s}wai => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
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
            std.debug.print("{s}res => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
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
            std.debug.print("{s}off => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
                " " ** (offset + 2 * padding),
                this.dim_info[op_idx].off_a_out,
                this.dim_info[op_idx].off_z_out,
                this.dim_info[op_idx].off_y_out,
                this.dim_info[op_idx].off_x_out,
                this.dim_info[op_idx].off_a_in,
                this.dim_info[op_idx].off_z_in,
                this.dim_info[op_idx].off_y_in,
                this.dim_info[op_idx].off_x_in,
            });
            std.debug.print("{s}idx => out({d:4}, {d:4}, {d:4}, {d:4}) in({d:4}, {d:4}, {d:4}, {d:4})\n", .{
                " " ** (offset + 2 * padding),
                this.dim_info[op_idx].idx_a_out,
                this.dim_info[op_idx].idx_z_out,
                this.dim_info[op_idx].idx_y_out,
                this.dim_info[op_idx].idx_x_out,
                this.dim_info[op_idx].idx_a_in,
                this.dim_info[op_idx].idx_z_in,
                this.dim_info[op_idx].idx_y_in,
                this.dim_info[op_idx].idx_x_in,
            });
            std.debug.print("{s}inline => {}\n", .{
                " " ** (offset + 2 * padding),
                this.inline_type[op_idx],
            });
        }
    }
};
