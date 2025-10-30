const std = @import("std");
const Pcg = std.Random.Pcg;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const grads = @import("grads");
const Buffer = grads.Buffer;
const buffer_name_size = Buffer.buffer_name_size;
const Linearized = grads.Linearized;
const Op = Linearized.Op;
const OpKind = grads.Op.Kind;

// $FIXME This can't handle reshapes right now, only resizes
// $FIXME This currently only handles OpenCl runtime
pub fn textifyLinearized(
    gpa: Allocator,
    linearized: Linearized,
    a: u32,
    z: u32,
    y: u32,
    x: u32,
    linearized_name: []const u8,
) !struct { slice: []const u8, len: u32 } {
    const max_chars_per_op: u32 = 500; // Random number
    var text: []u8 = try gpa.alloc(u8, linearized.op_num * max_chars_per_op);
    errdefer gpa.free(text);
    var text_idx: u32 = 0;

    var unique_buffer_num: u32 = 0;
    var unique_buffer_id: []u64 = try gpa.alloc(u64, linearized.op_num * 2);
    defer gpa.free(unique_buffer_id);

    var unique_buffer_kind: []Buffer.Kind = try gpa.alloc(Buffer.Kind, linearized.op_num * 2);
    defer gpa.free(unique_buffer_kind);

    var op_idx: u32 = 0;
    while (op_idx < linearized.op_num) : (op_idx += 1) {
        var found: bool = false;
        var op_idx_search: u32 = 0;
        while (op_idx_search < unique_buffer_num) : (op_idx_search += 1) {
            if (unique_buffer_id[op_idx_search] == linearized.op[op_idx].out.id) {
                found = true;
            }
        }
        if (!found) {
            unique_buffer_id[unique_buffer_num] = linearized.op[op_idx].out.id;
            unique_buffer_kind[unique_buffer_num] = linearized.op[op_idx].out.kind;
            unique_buffer_num += 1;
        }
        if (!linearized.op[op_idx].kind.isUnary()) {
            found = false;
            op_idx_search = 0;
            while (op_idx_search < unique_buffer_num) : (op_idx_search += 1) {
                if (unique_buffer_id[op_idx_search] == linearized.op[op_idx].in.id) {
                    found = true;
                }
            }
            if (!found) {
                unique_buffer_id[unique_buffer_num] = linearized.op[op_idx].in.id;
                unique_buffer_kind[unique_buffer_num] = linearized.op[op_idx].in.kind;
                unique_buffer_num += 1;
            }
        }
    }

    const header_written: []const u8 =
        try std.fmt.bufPrint(text[text_idx..], "    var {s}: Linearized = try .alloc(arena, {});\n\n", .{ linearized_name, linearized.op_num });
    text_idx += @intCast(header_written.len);

    var unique_buffer_idx: u32 = 0;
    while (unique_buffer_idx < unique_buffer_num) : (unique_buffer_idx += 1) {
        const name: [buffer_name_size]u8 = Buffer.nameFromId(unique_buffer_id[unique_buffer_idx]);
        const written = try std.fmt.bufPrint(text[text_idx..], //
            "    var {s}: Buffer = try .alloc(runtime, arena, {}, {}, {}, {}, .{s});\n" ++
                "    defer {s}.free(runtime);\n", .{ name, a, z, y, x, @tagName(unique_buffer_kind[unique_buffer_idx]), name });
        text_idx += @intCast(written.len);
    }

    op_idx = 0;
    while (op_idx < linearized.op_num) : (op_idx += 1) {
        const op: Op = linearized.op[op_idx];
        const out: Buffer = op.out;
        const name_out: [buffer_name_size]u8 = out.name();
        const in: Buffer = op.in;
        const name_in: [buffer_name_size]u8 = in.name();
        var written = try std.fmt.bufPrint(text[text_idx..], "    {s}.moveResize({}, {}, {}, {});\n" ++
            "    {s}.moveOffset({}, {}, {}, {});\n", .{
            name_out, out.a_size,    out.z_size,    out.y_size,    out.x_size, //
            name_out, out.aOffset(), out.zOffset(), out.yOffset(), out.xOffset(),
        });
        text_idx += @intCast(written.len);
        if (!op.kind.isUnary()) {
            written = try std.fmt.bufPrint(text[text_idx..], "    {s}.moveResize({}, {}, {}, {});\n" ++
                "    {s}.moveOffset({}, {}, {}, {});\n", .{
                name_in, in.a_size,    in.z_size,    in.y_size,    in.x_size, //
                name_in, in.aOffset(), in.zOffset(), in.yOffset(), in.xOffset(),
            });
            text_idx += @intCast(written.len);
        }

        written = try switch (op.kind) {
            .unary_add => std.fmt.bufPrint(text[text_idx..], "    {s}.unaryAdd({s}, {d});\n", .{ linearized_name, name_out, op.u_var }),
            .unary_subtract => std.fmt.bufPrint(text[text_idx..], "    {s}.unarySubtract({s}, {d});\n", .{ linearized_name, name_out, op.u_var }),
            .unary_multiply => std.fmt.bufPrint(text[text_idx..], "    {s}.unaryMultiply({s}, {d});\n", .{ linearized_name, name_out, op.u_var }),
            .unary_divide => std.fmt.bufPrint(text[text_idx..], "    {s}.unaryDivide({s}, {d});\n", .{ linearized_name, name_out, op.u_var }),
            .unary_exp => std.fmt.bufPrint(text[text_idx..], "    {s}.unaryExp({s});\n", .{ linearized_name, name_out }),
            .unary_log => std.fmt.bufPrint(text[text_idx..], "    {s}.unaryLog({s});\n", .{ linearized_name, name_out }),
            .unary_square => std.fmt.bufPrint(text[text_idx..], "    {s}.unarySquare({s});\n", .{ linearized_name, name_out }),
            .unary_sqrt => std.fmt.bufPrint(text[text_idx..], "    {s}.unarySqrt({s});\n", .{ linearized_name, name_out }),
            .unary_reciprocal => std.fmt.bufPrint(text[text_idx..], "    {s}.unaryReciprocal({s});\n", .{ linearized_name, name_out }),
            .unary_max => std.fmt.bufPrint(text[text_idx..], "    {s}.unaryMax({s}, {d});\n", .{ linearized_name, name_out, op.u_var }),
            .unary_min => std.fmt.bufPrint(text[text_idx..], "    {s}.unaryMin({s}, {d});\n", .{ linearized_name, name_out, op.u_var }),
            .unary_set => std.fmt.bufPrint(text[text_idx..], "    {s}.unarySet({s}, {d});\n", .{ linearized_name, name_out, op.u_var }),
            .unary_random => std.fmt.bufPrint(text[text_idx..], "    {s}.unaryRandom({s}, {});\n", .{ linearized_name, name_out, @as(u32, @bitCast(op.u_var)) }),
            .unary_tanh => std.fmt.bufPrint(text[text_idx..], "    {s}.unaryTanh({s});\n", .{ linearized_name, name_out }),
            .unary_absolute => std.fmt.bufPrint(text[text_idx..], "    {s}.unaryAbsolute({s});\n", .{ linearized_name, name_out }),
            .unary_sign => std.fmt.bufPrint(text[text_idx..], "    {s}.unarySign({s});\n", .{ linearized_name, name_out }),
            .binary_add => std.fmt.bufPrint(text[text_idx..], "    {s}.binaryAdd({s}, {s});\n", .{ linearized_name, name_out, name_in }),
            .binary_subtract => std.fmt.bufPrint(text[text_idx..], "    {s}.binarySubtract({s}, {s});\n", .{ linearized_name, name_out, name_in }),
            .binary_multiply => std.fmt.bufPrint(text[text_idx..], "    {s}.binaryMultiply({s}, {s});\n", .{ linearized_name, name_out, name_in }),
            .binary_divide => std.fmt.bufPrint(text[text_idx..], "    {s}.binaryDivide({s}, {s});\n", .{ linearized_name, name_out, name_in }),
            .binary_max => std.fmt.bufPrint(text[text_idx..], "    {s}.binaryMax({s}, {s});\n", .{ linearized_name, name_out, name_in }),
            .binary_min => std.fmt.bufPrint(text[text_idx..], "    {s}.binaryMin({s}, {s});\n", .{ linearized_name, name_out, name_in }),
            .binary_set => std.fmt.bufPrint(text[text_idx..], "    {s}.binarySet({s}, {s});\n", .{ linearized_name, name_out, name_in }),
            .expand_add => std.fmt.bufPrint(text[text_idx..], "    {s}.expandAdd({s}, {s});\n", .{ linearized_name, name_out, name_in }),
            .expand_subtract => std.fmt.bufPrint(text[text_idx..], "    {s}.expandSubtract({s}, {s});\n", .{ linearized_name, name_out, name_in }),
            .expand_multiply => std.fmt.bufPrint(text[text_idx..], "    {s}.expandMultiply({s}, {s});\n", .{ linearized_name, name_out, name_in }),
            .expand_divide => std.fmt.bufPrint(text[text_idx..], "    {s}.expandDivide({s}, {s});\n", .{ linearized_name, name_out, name_in }),
            .expand_max => std.fmt.bufPrint(text[text_idx..], "    {s}.expandMax({s}, {s});\n", .{ linearized_name, name_out, name_in }),
            .expand_min => std.fmt.bufPrint(text[text_idx..], "    {s}.expandMin({s}, {s});\n", .{ linearized_name, name_out, name_in }),
            .expand_set => std.fmt.bufPrint(text[text_idx..], "    {s}.expandSet({s}, {s});\n", .{ linearized_name, name_out, name_in }),
            .reduce_sum => std.fmt.bufPrint(text[text_idx..], "    {s}.reduceSum({s}, {s});\n", .{ linearized_name, name_out, name_in }),
            .reduce_max => std.fmt.bufPrint(text[text_idx..], "    {s}.reduceMax({s}, {s});\n", .{ linearized_name, name_out, name_in }),
            .reduce_avg => std.fmt.bufPrint(text[text_idx..], "    {s}.reduceAvg({s}, {s});\n", .{ linearized_name, name_out, name_in }),
            .reduce_min => std.fmt.bufPrint(text[text_idx..], "    {s}.reduceMin({s}, {s});\n", .{ linearized_name, name_out, name_in }),
        };
        text_idx += @intCast(written.len);
    }

    return .{ .slice = text, .len = text_idx };
}
