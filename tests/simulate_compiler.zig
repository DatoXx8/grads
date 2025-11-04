const std = @import("std");
const Pcg = std.Random.Pcg;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const grads = @import("grads");
const Linearized = grads.Linearized;
const OpKind = grads.Op.Kind;
const Program = grads.Program;
const Runtime = grads.Runtime;
const RuntimeCl = grads.RuntimeCl;
const Optimization = grads.Optimization;
const util = grads.util;

const randomLinearized = @import("random_linearized.zig").randomLinearized;
const RegressionTest = @import("regression_compiler.zig").RegressionTest;
const a_size_max = @import("random_linearized.zig").a_size_max;
const z_size_max = @import("random_linearized.zig").z_size_max;
const y_size_max = @import("random_linearized.zig").y_size_max;
const x_size_max = @import("random_linearized.zig").x_size_max;
const op_num = @import("random_linearized.zig").op_num;
const buffer_num = @import("random_linearized.zig").buffer_num;

const AssertError = error{
    nan,
    inf,
    difference,
};
/// Margin of error
const epsilon: f32 = 1e-6;
const epsilon_relative: f32 = 1e-4;
/// Check for equality between the two floats within the margin of error of `epsilon`
fn assertEq(val1: f32, val2: f32) !void {
    if (std.math.isNan(val1) or std.math.isNan(val2)) {
        // For nicer output formatting
        util.log.print("\n", .{});
        std.log.err("Found NaN in equality comparison.\n", .{});
        return AssertError.nan;
    } else if (std.math.isInf(val1) or std.math.isInf(val2)) {
        util.log.print("\n", .{});
        std.log.err("Found Inf in equality comparison.\n", .{});
        return AssertError.nan;
    } else if (std.math.approxEqAbs(f32, val1, val2, epsilon) or std.math.approxEqRel(f32, val1, val2, epsilon_relative)) {
        return;
    } else {
        // For nicer output formatting
        util.log.print("\n", .{});
        std.log.err("Difference between {d} and {d} is too large.\n", .{ val1, val2 });
        return AssertError.difference;
    }
}

fn simulateCompiler(
    runtime: Runtime,
    gpa: Allocator,
    op_included: [op_num]bool,
    rng: u64,
    depth_max: u32,
) !void {
    var pcg = Pcg.init(rng);

    var arena_allocator: ArenaAllocator = .init(gpa);
    defer arena_allocator.deinit();
    const arena: Allocator = arena_allocator.allocator();

    var arena_temp_allocator: ArenaAllocator = .init(gpa);
    defer arena_temp_allocator.deinit();
    const arena_temp: Allocator = arena_temp_allocator.allocator();

    var buffer1 = try randomLinearized(runtime, arena, op_included, rng);
    defer {
        for (&buffer1.buffer) |buffer| {
            buffer.free(runtime);
        }
    }
    var buffer2 = try randomLinearized(runtime, arena, op_included, rng);
    defer {
        for (&buffer2.buffer) |buffer| {
            buffer.free(runtime);
        }
    }

    buffer2.linearized.realize();

    const size_local: u32 = pcg.random().uintLessThan(u32, 10) + 1;
    const size_global: u32 = size_local * (pcg.random().uintLessThan(u32, 10) + 1);

    for (0..buffer_num) |buffer_idx| {
        buffer1.buffer[buffer_idx].syncUpdate(.sync_to_device);
        try buffer1.buffer[buffer_idx].syncToDevice(runtime);
    }

    var program: Program = try Program.alloc(runtime, gpa, arena, arena_temp, //
        buffer1.linearized, depth_max, size_global, size_local);
    defer program.free(runtime);

    try program.run(runtime);

    buffer1.buffer[buffer1.out_idx].syncUpdate(.sync_to_host);
    try buffer1.buffer[buffer1.out_idx].syncToHost(runtime);

    for (0..buffer1.buffer[buffer1.out_idx].values.len) |arg_idx| {
        assertEq(buffer1.buffer[buffer1.out_idx].values[arg_idx], buffer2.buffer[buffer2.out_idx].values[arg_idx]) catch |err| {
            std.log.err("Difference at index {} = [{}, {}, {}, {}] with {any} between compiled {d} \"{s}\" and linearized {d} \"{s}\" with rng={} and opt={}\n", .{
                arg_idx,
                arg_idx / (z_size_max * y_size_max * x_size_max),
                arg_idx / (y_size_max * x_size_max) % z_size_max,
                arg_idx / x_size_max % y_size_max,
                arg_idx % x_size_max,
                op_included,
                buffer1.buffer[buffer1.out_idx].values[arg_idx],
                buffer1.buffer[buffer1.out_idx].name(),
                buffer2.buffer[buffer2.out_idx].values[arg_idx],
                buffer2.buffer[buffer2.out_idx].name(),
                rng,
                depth_max,
            });
            return err;
        };
    }
}

fn minifyCompiler(
    runtime: Runtime,
    gpa: Allocator,
    rng: u64,
    depth_max: u32,
    err: anyerror,
) !void {
    util.log.disable();
    assert(buffer_num > 1);
    assert(op_num > 0);
    var op_included: [op_num]bool = @splat(true);
    for (0..op_num) |op_idx| {
        var failed: bool = false;
        op_included[op_idx] = false;
        var erro: anyerror = undefined;
        simulateCompiler(runtime, gpa, op_included, rng, depth_max) catch |e| {
            erro = e;
            failed = true;
        };
        if (!failed) {
            op_included[op_idx] = true;
        }
    }
    var depth_max_first_fail: u32 = 0;
    while (depth_max_first_fail < depth_max) : (depth_max_first_fail += 1) {
        var failed: bool = false;
        simulateCompiler(runtime, gpa, op_included, rng, depth_max_first_fail) catch {
            failed = true;
        };
        if (failed) {
            break;
        }
    }
    util.log.enable();
    util.log.print("\n\nMinimal: {any} with depth: {}\n", .{ op_included, depth_max_first_fail });
    var failed: bool = false;
    simulateCompiler(runtime, gpa, op_included, rng, depth_max_first_fail) catch {
        failed = true;
    };
    if (!failed) {
        std.log.err("Strange error. Probably from some race condition.\n", .{});
    }
    return err;
}

pub fn main() !void {
    var geneal_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = geneal_purpose_allocator.allocator();
    defer _ = geneal_purpose_allocator.detectLeaks();

    var args = try std.process.argsWithAllocator(gpa);
    defer args.deinit();

    var rng_saved: ?u64 = null;
    var loop_infinite: bool = false;
    var loop_count: u64 = 1;
    var opt_saved: ?u32 = null;
    // Skip the executable call
    _ = args.next();
    while (args.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "rng=")) {
            const offset = "rng="[0..].len;
            rng_saved = try std.fmt.parseInt(u64, arg[offset..], 10);
        } else if (std.mem.startsWith(u8, arg, "loop=")) {
            const offset = "loop="[0..].len;
            loop_count = std.fmt.parseInt(u64, arg[offset..], 10) catch 0;
            if (loop_count == 0) {
                loop_infinite = true;
            }
            // Iff the loop is infinite then the loop count has to be 0
            assert(loop_infinite == (loop_count == 0));
        } else if (std.mem.startsWith(u8, arg, "opt=")) {
            const offset = "opt="[0..].len;
            const parse: []const u8 = arg[offset..];
            opt_saved = std.fmt.parseInt(u32, arg[offset..], 10) catch null;

            if (opt_saved == null) {
                util.log.print("Found unrecognized optimization {s}, expected opt=[number]\n", .{parse});
                @panic("See above error message");
            }
        } else {
            util.log.print("error: Found unrecognised option `{s}`, expected `rng=<number>`, `loop=[number] or opt=[number]\n", .{arg});
            @panic("See above error message");
        }
    }

    const rng: u64 = switch (rng_saved == null) {
        true => std.crypto.random.int(u64),
        false => rng_saved.?,
    };

    var runtime_cl: RuntimeCl = undefined;
    var runtime: Runtime = runtime_cl.runtime();
    try runtime.init();
    defer runtime.deinit();

    // Not just the max because other optimizations might remove broken states by pure luck
    const depth_max: []const u32 = &.{ 0, 1, 10, 100, 1000 };

    if (loop_infinite) {
        var loop_idx: u64 = 0;
        while (true) {
            util.log.print("{} => simulate_compiler: rng={}... ", .{ loop_idx, rng +% loop_idx });
            if (opt_saved) |opt| {
                simulateCompiler(runtime, gpa, @splat(true), rng +% loop_idx, opt) catch |err| {
                    try minifyCompiler(runtime, gpa, rng +% loop_idx, opt, err);
                };
                util.log.print("{d} ", .{opt});
            } else {
                for (depth_max) |depth| {
                    simulateCompiler(runtime, gpa, @splat(true), rng +% loop_idx, depth) catch |err| {
                        try minifyCompiler(runtime, gpa, rng +% loop_idx, depth, err);
                    };
                    util.log.print("{d} ", .{depth});
                }
            }
            loop_idx += 1;
            util.log.print("passed!\n", .{});
        }
    } else {
        for (0..loop_count) |loop_idx| {
            util.log.print("{} => simulate_compiler: rng={}... ", .{ loop_idx, rng +% loop_idx });
            if (opt_saved) |opt| {
                simulateCompiler(runtime, gpa, @splat(true), rng +% loop_idx, opt) catch |err| {
                    try minifyCompiler(runtime, gpa, rng +% loop_idx, opt, err);
                };
                util.log.print("{d} ", .{opt});
            } else {
                for (depth_max) |depth| {
                    simulateCompiler(runtime, gpa, @splat(true), rng +% loop_idx, depth) catch |err| {
                        try minifyCompiler(runtime, gpa, rng +% loop_idx, depth, err);
                    };
                    util.log.print("{d} ", .{depth});
                }
            }
            util.log.print("passed!\n", .{});
        }
    }
}
