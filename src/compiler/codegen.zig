// TODO: Optimisation
// Optimisation levels
// O0 - none
// O1 - inline, split, merge kernels
// O2 - fuse along all axis
// O3 - memory optimizer (SLOW!!!)

const std = @import("std");

const Pir = @import("./pir.zig").Pir;

const assert = @import("../util.zig").assert;

const buffer_name_size = @import("../tensor.zig").buffer_name_size;

pub const Optimisation = enum(u8) {
    O0,
    O1,
    O2,
    O3,
};

fn args_gather(allocator: anytype, pir: Pir) ![][buffer_name_size]u8 {
    const arg_initial: u32 = 4;
    var arg_name: [][buffer_name_size]u8 = try allocator.alloc([buffer_name_size]u8, arg_initial);
    var arg_count: u32 = 0;

    for (0..pir.op_num) |op_idx| {
        // TODO: Split cases by is_unary because halfing the string comparisons is faster
        var arg_found_out: bool = false;
        var arg_found_in: bool = false;
        for (0..arg_count) |arg_idx| {
            if (!arg_found_out and
                std.mem.eql(u8, &arg_name[arg_idx], &pir.op[op_idx].out.name))
            {
                arg_found_out = true;
            }
            if (!arg_found_in and
                std.mem.eql(u8, &arg_name[arg_idx], &pir.op[op_idx].in.name))
            {
                arg_found_in = true;
            }
            if (arg_found_out and arg_found_in) {
                break;
            }
        }

        if (!arg_found_out) {
            if (arg_count == arg_name.len) {
                arg_name = try allocator.realloc(arg_name, arg_name.len * 2);
            }
            arg_name[arg_count] = pir.op[op_idx].out.name;
            std.debug.print("{s}\n", .{arg_name[arg_count]});
            arg_count += 1;
        }
        if (!arg_found_in) {
            if (arg_count == arg_name.len) {
                arg_name = try allocator.realloc(arg_name, arg_name.len * 2);
            }
            arg_name[arg_count] = pir.op[op_idx].in.name;
            std.debug.print("{s}\n", .{arg_name[arg_count]});
            arg_count += 1;
        }
    }
    return arg_name;
}

// pub fn generate(allocator: anytype, pir: Pir, size_global: u32, size_local: u32, optimisation: Optimisation) ![*:0]u8 {
pub fn generate(allocator: anytype, pir: Pir, size_global: u32, size_local: u32, optimisation: Optimisation) !void {
    assert(optimisation == .O0);
    assert(size_global % size_local == 0);
    assert(size_global > 0);
    assert(size_local > 0);
    // This might not be necessary because I think it is implied by the 3 above
    assert(size_global >= size_local);

    const args: [][buffer_name_size]u8 = try args_gather(allocator, pir);
    _ = args;
}
