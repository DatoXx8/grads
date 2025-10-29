const std = @import("std");
const Pcg = std.Random.Pcg;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const grads = @import("grads");
const Buffer = grads.Buffer;
const Linearized = grads.Linearized;
const Op = grads.Op;
const Program = grads.Program;
const Runtime = grads.Runtime;
const RuntimeCl = grads.RuntimeCl;
const Optimization = grads.Optimization;

/// Margin of error
const epsilon: f32 = 1e-6;
const epsilon_relative: f32 = 1e-4;
/// Check for equality between the two floats within the margin of error of `epsilon`
fn checkEq(val1: f32, val2: f32) bool {
    if (std.math.isNan(val1) or std.math.isNan(val2)) {
        // For nicer output formatting
        std.debug.print("\n", .{});
        std.log.err("Found NaN in equality comparison.\n", .{});
        return false;
    } else if (std.math.isInf(val1) or std.math.isInf(val2)) {
        std.debug.print("\n", .{});
        std.log.err("Found Inf in equality comparison.\n", .{});
        return false;
    } else if (std.math.approxEqAbs(f32, val1, val2, epsilon) or std.math.approxEqRel(f32, val1, val2, epsilon_relative)) {
        return true;
    } else {
        // For nicer output formatting
        std.debug.print("\n", .{});
        std.log.err("Difference between {d} and {d} is too large.\n", .{ val1, val2 });
        return false;
    }
}

pub const SimpleLinearized = struct {
    pub const SimpleBuffer = struct {
        id: u32,
        kind: Buffer.Kind,
        a_size: u32,
        z_size: u32,
        y_size: u32,
        x_size: u32,
        a_stride: u32,
        z_stride: u32,
        y_stride: u32,
        x_stride: u32,
        offset: u32,
        pub fn fromBuffer(buffer: Buffer) SimpleBuffer {
            return .{
                .id = buffer.id,
                .kind = buffer.kind,
                .a_size = buffer.a_size,
                .z_size = buffer.z_size,
                .y_size = buffer.y_size,
                .x_size = buffer.x_size,
                .a_stride = buffer.a_stride,
                .z_stride = buffer.z_stride,
                .y_stride = buffer.y_stride,
                .x_stride = buffer.x_stride,
                .offset = buffer.offset,
            };
        }
    };
    pub const SimpleOp = struct {
        in: SimpleBuffer,
        out: SimpleBuffer,
        kind: Op.Kind,
        u_var: f32,
        pub fn fromOp(op: Op) SimpleOp {
            return .{
                .out = SimpleBuffer.fromBuffer(op.out),
                .in = SimpleBuffer.fromBuffer(op.in),
                .kind = op.kind,
                .u_var = op.u_var,
            };
        }
    };
    simple_op: []SimpleOp,
    num: u32,
    pub fn populate(
        simple_linearized: SimpleLinearized,
        runtime: Runtime,
        arena: Allocator,
        a: u32,
        z: u32,
        y: u32,
        x: u32,
    ) !struct {
        buffer: []Buffer,
        linearized: Linearized,
    } {
        var unique_buffer_num: u32 = 0;
        var unique_buffer_id: []u64 = try arena.alloc(u64, simple_linearized.op_num * 2);
        var unique_buffer_kind: []Buffer.Kind = try arena.alloc(Buffer.Kind, simple_linearized.op_num * 2);

        var op_idx: u32 = 0;
        while (op_idx < simple_linearized.num) : (op_idx += 1) {
            var found: bool = false;
            var op_idx_search: u32 = 0;
            while (op_idx_search < unique_buffer_num) : (op_idx_search += 1) {
                if (unique_buffer_id[op_idx_search] == simple_linearized.simple_op[op_idx].out.id) {
                    found = true;
                }
            }
            if (!found) {
                unique_buffer_id[unique_buffer_num] = simple_linearized.simple_op[op_idx].out.id;
                unique_buffer_kind[unique_buffer_num] = simple_linearized.simple_op[op_idx].out.kind;
                unique_buffer_num += 1;
            }
            if (!simple_linearized.simple_op[op_idx].kind.isUnary()) {
                found = false;
                op_idx_search = 0;
                while (op_idx_search < unique_buffer_num) : (op_idx_search += 1) {
                    if (unique_buffer_id[op_idx_search] == simple_linearized.simple_op[op_idx].in.id) {
                        found = true;
                    }
                }
                if (!found) {
                    unique_buffer_id[unique_buffer_num] = simple_linearized.simple_op[op_idx].in.id;
                    unique_buffer_kind[unique_buffer_num] = simple_linearized.simple_op[op_idx].in.kind;
                    unique_buffer_num += 1;
                }
            }
        }

        var buffer: []Buffer = try arena.alloc(Buffer, unique_buffer_num);
        var unique_buffer_idx: u32 = 0;
        while (unique_buffer_idx < unique_buffer_num) : (unique_buffer_idx += 1) {
            buffer[unique_buffer_idx] = Buffer.alloc(runtime, arena, a, z, y, x, unique_buffer_kind[unique_buffer_idx]) catch |err| {
                var free_idx: u32 = 0;
                while (free_idx < unique_buffer_idx) : (free_idx += 1) {
                    buffer[free_idx].free(runtime);
                }
                return err;
            };
        }
        errdefer {
            var free_idx: u32 = 0;
            while (free_idx < unique_buffer_num) : (free_idx += 1) {
                buffer[free_idx].free(runtime);
            }
        }

        var linearized: Linearized = try .alloc(arena, simple_linearized.num);
        op_idx = 0;
        while (op_idx < simple_linearized.num) : (op_idx += 1) {
            const out_id: u64 = simple_linearized.simple_op[op_idx].out.id;
            var buffer_out_idx: u32 = 0;
            while (buffer_out_idx < unique_buffer_num) : (buffer_out_idx += 1) {
                if (out_id == unique_buffer_id[buffer_out_idx]) {
                    break;
                }
            }
            const in_id: u64 = simple_linearized.simple_op[op_idx].in.id;
            var buffer_in_idx: u32 = 0;
            while (buffer_in_idx < unique_buffer_num) : (buffer_in_idx += 1) {
                if (in_id == unique_buffer_id[buffer_in_idx]) {
                    break;
                }
            }

            linearized.op[op_idx] = .{
                .out = .{
                    .sync = .sync_to_none,
                    .id = simple_linearized.simple_op[op_idx].out.id,
                    .a_size = simple_linearized.simple_op[op_idx].out.a_size,
                    .z_size = simple_linearized.simple_op[op_idx].out.z_size,
                    .y_size = simple_linearized.simple_op[op_idx].out.y_size,
                    .x_size = simple_linearized.simple_op[op_idx].out.x_size,
                    .a_stride = simple_linearized.simple_op[op_idx].out.a_stride,
                    .z_stride = simple_linearized.simple_op[op_idx].out.z_stride,
                    .y_stride = simple_linearized.simple_op[op_idx].out.y_stride,
                    .x_stride = simple_linearized.simple_op[op_idx].out.x_stride,
                    .offset = simple_linearized.simple_op[op_idx].out.offset,
                    .kind = simple_linearized.simple_op[op_idx].out.kind,
                    .values = buffer[buffer_out_idx].values,
                    .values_runtime = buffer[buffer_out_idx].values_runtime,
                },
                .in = .{
                    .sync = .sync_to_none,
                    .id = simple_linearized.simple_op[op_idx].in.id,
                    .a_size = simple_linearized.simple_op[op_idx].in.a_size,
                    .z_size = simple_linearized.simple_op[op_idx].in.z_size,
                    .y_size = simple_linearized.simple_op[op_idx].in.y_size,
                    .x_size = simple_linearized.simple_op[op_idx].in.x_size,
                    .a_stride = simple_linearized.simple_op[op_idx].in.a_stride,
                    .z_stride = simple_linearized.simple_op[op_idx].in.z_stride,
                    .y_stride = simple_linearized.simple_op[op_idx].in.y_stride,
                    .x_stride = simple_linearized.simple_op[op_idx].in.x_stride,
                    .offset = simple_linearized.simple_op[op_idx].in.offset,
                    .kind = simple_linearized.simple_op[op_idx].in.kind,
                    .values = buffer[buffer_in_idx].values,
                    .values_runtime = buffer[buffer_in_idx].values_runtime,
                },
                .kind = simple_linearized.simple_op[op_idx].kind,
                .u_var = simple_linearized.simple_op[op_idx].u_var,
            };
        }

        return .{
            .linearized = linearized,
            .buffer = buffer,
        };
    }
};

const RegressionTest = struct {
    simple_linearized: SimpleLinearized,
    depth_max: u32,
    a: u32,
    z: u32,
    y: u32,
    x: u32,
    size_global: u32,
    size_local: u32,
    pub fn run(reg_test: RegressionTest, runtime: Runtime, gpa: Allocator) bool {
        if (true) {
            @panic("Need to init the values in the tensors somehow."); // <-- $FIXME
        }
        const arena_allocator: ArenaAllocator = .init(gpa);
        defer arena_allocator.deinit();
        const arena: Allocator = arena_allocator.allocator();

        const arena_temp_allocator: ArenaAllocator = .init(gpa);
        defer arena_temp_allocator.deinit();
        const arena_temp: Allocator = arena_temp_allocator.allocator();

        const result1 = try reg_test.simple_linearized.populate(runtime, arena, //
            reg_test.a, reg_test.z, reg_test.y, reg_test.x);
        const linearized1: Linearized = result1.linearized;
        const buffer1: []Buffer = result1.buffer;
        defer {
            for (buffer1) |buffer| {
                buffer.free(runtime);
            }
        }

        const result2 = try reg_test.simple_linearized.populate(runtime, arena, //
            reg_test.a, reg_test.z, reg_test.y, reg_test.x);
        const linearized2: Linearized = result2.linearized;
        const buffer2: []Buffer = result2.buffer;
        defer {
            for (buffer2) |buffer| {
                buffer.free(runtime);
            }
        }

        linearized2.realize();

        for (buffer1) |*buffer| {
            buffer.syncUpdate(.sync_to_device);
            try buffer.syncToDevice(runtime);
        }

        var program: Program = try Program.alloc(runtime, gpa, arena, arena_temp, //
            linearized1, reg_test.depth_max, reg_test.size_global, reg_test.size_local);
        defer program.free(runtime);
        try program.run(runtime);

        linearized1.op[linearized1.num - 1].out.syncUpdate(.sync_to_host);
        try linearized1.op[linearized1.num - 1].out.syncToHost(runtime);

        var arg_idx: u32 = 0;
        while (arg_idx < reg_test.a * reg_test.z * reg_test.y * reg_test.x) : (arg_idx += 1) {
            if (!checkEq(buffer1.buffer[buffer1.out_idx].values[arg_idx], //
                buffer2.buffer[buffer2.out_idx].values[arg_idx]))
            {
                return false;
            }
        }
    }
};
