// Helper file that implements a function that accepts linearized ops and prints text that reproduces those operations
const std = @import("std");
const Pcg = std.Random.Pcg;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const grads = @import("grads");
const Tensor = grads.Tensor;
const Buffer = Tensor.Buffer;
const Op = Tensor.Op;
const Linearized = grads.Linearized;
const OpType = grads.Op.Type;
const buffer_name_size = grads.Tensor.buffer_name_size;

// $FIXME This can't handle reshapes right now, only resizes
pub fn textifyLinearized(allocator: Allocator, linearized: Linearized, a: u32, z: u32, y: u32, x: u32) ![]const u8 {
    const max_chars_per_op: u32 = 500; // Random number
    var text: []u8 = try allocator.alloc(u8, linearized.op_num * max_chars_per_op);
    errdefer allocator.free(text);
    var text_idx: u32 = 0;

    var unique_buffer_num: u32 = 0;
    var unique_buffer_ids: []u64 = try allocator.alloc(u64, linearized.op_num * 2);
    defer allocator.free(unique_buffer_ids);

    var op_idx: u32 = 0;
    while (op_idx < linearized.op_num) : (op_idx += 1) {
        var found: bool = false;
        var op_idx_search: u32 = 0;
        while (op_idx_search < unique_buffer_num) : (op_idx_search += 1) {
            if (unique_buffer_ids[op_idx_search] == linearized.op[op_idx].out.id) {
                found = true;
            }
        }
        if (!found) {
            unique_buffer_ids[unique_buffer_num] = linearized.op[op_idx].out.id;
            unique_buffer_num += 1;
        }
        if (!linearized.op[op_idx].type.isUnary()) {
            found = false;
            op_idx_search = 0;
            while (op_idx_search < unique_buffer_num) : (op_idx_search += 1) {
                if (unique_buffer_ids[op_idx_search] == linearized.op[op_idx].in.id) {
                    found = true;
                }
            }
            if (!found) {
                unique_buffer_ids[unique_buffer_num] = linearized.op[op_idx].in.id;
                unique_buffer_num += 1;
            }
        }
    }

    var unique_buffer_idx: u32 = 0;
    while (unique_buffer_idx < unique_buffer_num) : (unique_buffer_idx += 1) {
        const name: [buffer_name_size]u8 = Tensor.Buffer.nameFromId(unique_buffer_ids[unique_buffer_idx]);
        const written = try std.fmt.bufPrint(text[text_idx..], //
            "var {s}: Tensor = .alloc(runtime, allocator, {}, {}, {}, {}, {});\n" ++
                "defer {s}.free(runtime, allocator);\n", .{ name, a, z, y, x, linearized.op_num, name });
        text_idx += @intCast(written.len);
    }

    op_idx = 0;
    while (op_idx < linearized.op_num) : (op_idx += 1) {
        const op: Op = linearized.op[op_idx];
        const out: Buffer = op.out;
        const name_out: [buffer_name_size]u8 = out.name();
        const in: Buffer = op.in;
        const name_in: [buffer_name_size]u8 = in.name();
        var written = try std.fmt.bufPrint(text[text_idx..], "{s}.moveResize({}, {}, {}, {});\n" ++
            "{s}.moveOffset({}, {}, {}, {});\n", .{
            name_out, out.a_size,    out.z_size,    out.y_size,    out.x_size, //
            name_out, out.aOffset(), out.zOffset(), out.yOffset(), out.xOffset(),
        });
        text_idx += @intCast(written.len);
        if (!op.type.isUnary()) {
            written = try std.fmt.bufPrint(text[text_idx..], "{s}.moveResize({}, {}, {}, {});\n" ++
                "{s}.moveOffset({}, {}, {}, {});\n", .{
                name_in, in.a_size,    in.z_size,    in.y_size,    in.x_size, //
                name_in, in.aOffset(), in.zOffset(), in.yOffset(), in.xOffset(),
            });
            text_idx += @intCast(written.len);
        }

        written = try switch (op.type) {
            .unary_add => std.fmt.bufPrint(text[text_idx..], "{s}.unaryAdd({d});\n", .{ name_out, op.u_var }),
            .unary_subtract => std.fmt.bufPrint(text[text_idx..], "{s}.unarySubtract({d});\n", .{ name_out, op.u_var }),
            .unary_multiply => std.fmt.bufPrint(text[text_idx..], "{s}.unaryMultiply({d});\n", .{ name_out, op.u_var }),
            .unary_divide => std.fmt.bufPrint(text[text_idx..], "{s}.unaryDivide({d});\n", .{ name_out, op.u_var }),
            .unary_exp => std.fmt.bufPrint(text[text_idx..], "{s}.unaryExp();\n", .{name_out}),
            .unary_log => std.fmt.bufPrint(text[text_idx..], "{s}.unaryLog();\n", .{name_out}),
            .unary_square => std.fmt.bufPrint(text[text_idx..], "{s}.unarySquare();\n", .{name_out}),
            .unary_sqrt => std.fmt.bufPrint(text[text_idx..], "{s}.unarySqrt();\n", .{name_out}),
            .unary_reciprocal => std.fmt.bufPrint(text[text_idx..], "{s}.unaryReciprocal();\n", .{name_out}),
            .unary_max => std.fmt.bufPrint(text[text_idx..], "{s}.unaryMax({d});\n", .{ name_out, op.u_var }),
            .unary_min => std.fmt.bufPrint(text[text_idx..], "{s}.unaryMin({d});\n", .{ name_out, op.u_var }),
            .unary_set => std.fmt.bufPrint(text[text_idx..], "{s}.unarySet({d});\n", .{ name_out, op.u_var }),
            .unary_random => std.fmt.bufPrint(text[text_idx..], "{s}.unaryRandom({});\n", .{ name_out, @as(u32, @bitCast(op.u_var)) }),
            .unary_tanh => std.fmt.bufPrint(text[text_idx..], "{s}.unaryTanh();\n", .{name_out}),
            .unary_absolute => std.fmt.bufPrint(text[text_idx..], "{s}.unaryAbsolute();\n", .{name_out}),
            .unary_sign => std.fmt.bufPrint(text[text_idx..], "{s}.unarySign();\n", .{name_out}),
            .binary_add => std.fmt.bufPrint(text[text_idx..], "{s}.binaryAdd({s});\n", .{ name_out, name_in }),
            .binary_subtract => std.fmt.bufPrint(text[text_idx..], "{s}.binarySubtract({s});\n", .{ name_out, name_in }),
            .binary_multiply => std.fmt.bufPrint(text[text_idx..], "{s}.binaryMultiply({s});\n", .{ name_out, name_in }),
            .binary_divide => std.fmt.bufPrint(text[text_idx..], "{s}.binaryDivide({s});\n", .{ name_out, name_in }),
            .binary_max => std.fmt.bufPrint(text[text_idx..], "{s}.binaryMax({s});\n", .{ name_out, name_in }),
            .binary_min => std.fmt.bufPrint(text[text_idx..], "{s}.binaryMin({s});\n", .{ name_out, name_in }),
            .binary_set => std.fmt.bufPrint(text[text_idx..], "{s}.binarySet({s});\n", .{ name_out, name_in }),
            .expand_add => std.fmt.bufPrint(text[text_idx..], "{s}.expandAdd({s});\n", .{ name_out, name_in }),
            .expand_subtract => std.fmt.bufPrint(text[text_idx..], "{s}.expandSubtract({s});\n", .{ name_out, name_in }),
            .expand_multiply => std.fmt.bufPrint(text[text_idx..], "{s}.expandMultiply({s});\n", .{ name_out, name_in }),
            .expand_divide => std.fmt.bufPrint(text[text_idx..], "{s}.expandDivide({s});\n", .{ name_out, name_in }),
            .expand_max => std.fmt.bufPrint(text[text_idx..], "{s}.expandMax({s});\n", .{ name_out, name_in }),
            .expand_min => std.fmt.bufPrint(text[text_idx..], "{s}.expandMin({s});\n", .{ name_out, name_in }),
            .expand_set => std.fmt.bufPrint(text[text_idx..], "{s}.expandSet({s});\n", .{ name_out, name_in }),
            .reduce_sum => std.fmt.bufPrint(text[text_idx..], "{s}.reduceSum({s});\n", .{ name_out, name_in }),
            .reduce_max => std.fmt.bufPrint(text[text_idx..], "{s}.reduceMax({s});\n", .{ name_out, name_in }),
            .reduce_avg => std.fmt.bufPrint(text[text_idx..], "{s}.reduceAvg({s});\n", .{ name_out, name_in }),
            .reduce_min => std.fmt.bufPrint(text[text_idx..], "{s}.reduceMin({s});\n", .{ name_out, name_in }),
        };
        text_idx += @intCast(written.len);
    }

    return text;
}
