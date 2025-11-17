const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const DefaultPrng = std.Random.DefaultPrng;
const ArenaAllocator = std.heap.ArenaAllocator;

const Layer = @import("Layer.zig");
const Activation = Layer.Activation;
const Dense = Layer.Dense;
const Convolution = Layer.Convolution;
const Reduce = Layer.Reduce;
const Split = Layer.Split;
const Residual = Layer.Residual;
const Runtime = @import("compiler/runtimes/Runtime.zig");
const Program = @import("compiler/Program.zig");
const Linearized = @import("Linearized.zig");
const Op = Linearized.Op;
const Buffer = @import("Buffer.zig");
const Vec4 = Buffer.Vec4;
const util = @import("util.zig");

pub const Neuralnet = @This();
arena: ArenaAllocator,
layer: []Layer,
in: Buffer,
in_g: Buffer,
forward_compiled: Program,
backward_compiled: Program,
learn_compiled: Program,
runtime: Runtime,
pub fn alloc(
    runtime: Runtime,
    gpa: Allocator,
    size_initial: Vec4,
    config: []const Layer.Config,
    size_global: u32,
    size_local: u32,
) !Neuralnet {
    assert(size_global > 0);
    assert(size_local > 0);
    assert(size_global % size_local == 0);
    assert(size_initial.a == 1);
    assert(size_initial.z > 0);
    assert(size_initial.y > 0);
    assert(size_initial.x > 0);

    var arena_nn_allocator: ArenaAllocator = .init(gpa);
    errdefer arena_nn_allocator.deinit();
    const arena_nn: Allocator = arena_nn_allocator.allocator();

    var arena_temp_allocator: ArenaAllocator = .init(gpa);
    defer arena_temp_allocator.deinit();
    const arena_temp: Allocator = arena_nn_allocator.allocator();

    var layer: []Layer = try arena_nn.alloc(Layer, config.len);

    var capacity_forward: u32 = 0;
    var capacity_backward: u32 = 0;
    var capacity_learn: u32 = 0;

    var size_in: Vec4 = size_initial;

    const in: Buffer = try Buffer.alloc(runtime, gpa, arena_nn, size_in, .normal);
    const in_g: Buffer = try Buffer.alloc(runtime, gpa, arena_nn, size_in, .normal);

    var layer_idx: u32 = 0;
    while (layer_idx < config.len) : (layer_idx += 1) {
        // $TODO Activation and norming
        const size_out: Vec4 = .{
            .a = 1,
            .z = switch (config[layer_idx]) {
                .dense => 1,
                .convolution => |c| c.filters,
                .reduce => size_in.z,
                .split => |s| size_in.z * s.filters,
                .residual => size_in.z,
            },
            .y = switch (config[layer_idx]) {
                .dense => 1,
                .convolution => |c| Convolution.sizeNew(size_in.y, c.kernel_size, c.kernel_stride, c.kernel_padding),
                .reduce => |r| Reduce.sizeNew(size_in.y, r.kernel_size, r.kernel_stride),
                .split => size_in.y,
                .residual => size_in.y,
            },
            .x = switch (config[layer_idx]) {
                .dense => |d| d.size_out,
                .convolution => |c| Convolution.sizeNew(size_in.x, c.kernel_size, c.kernel_stride, c.kernel_padding),
                .reduce => |r| Reduce.sizeNew(size_in.x, r.kernel_size, r.kernel_stride),
                .split => size_in.x,
                .residual => size_in.x,
            },
        };
        const forward_cap: u32 = switch (config[layer_idx]) {
            .dense => |d| d.size_out * 3 + 1,
            .convolution => |c| 4 * c.filters * size_out.y * size_out.x + 1,
            .reduce => size_in.z * size_out.y * size_out.x,
            .split => |s| 3 * s.filters,
            .residual => 1,
        };
        const backward_cap: u32 = switch (config[layer_idx]) {
            .dense => |d| 2 + d.size_out + 4 * size_in.productOfElements(),
            .convolution => |c| 2 * c.filters + 6 * c.filters * size_out.y * size_out.x + 1,
            .reduce => size_out.y * size_out.x,
            .split => |s| 1 + 6 * s.filters,
            .residual => 1,
        };
        const learn_cap: u32 = switch (config[layer_idx]) {
            .dense => 4,
            .convolution => 4,
            .reduce => 0,
            .split => 4,
            .residual => 0,
        };
        layer[layer_idx].activation = try Activation.alloc(runtime, gpa, arena_nn, switch (config[layer_idx]) {
            .dense => |d| d.activation_kind,
            .convolution => |c| c.activation_kind,
            .reduce => .none,
            .split => |s| s.activation_kind,
            .residual => .none,
        }, size_out);
        layer[layer_idx].tag = switch (config[layer_idx]) {
            .dense => |d| .{ .dense = try Dense.alloc(runtime, gpa, arena_nn, size_in.productOfElements(), d.size_out) },
            .convolution => |c| .{
                .convolution = try Convolution.alloc(runtime, gpa, arena_nn, size_in, c.filters, //
                    c.kernel_size, c.kernel_stride, c.kernel_padding),
            },
            .reduce => |r| .{ .reduce = Reduce.init(size_in, r.kernel_size, r.kernel_stride, r.t) },
            .split => |s| .{ .split = try Split.alloc(runtime, gpa, arena_nn, size_in, s.filters) },
            .residual => |r| .{ .residual = .{ .t = .identity, .in_layer = r.in_layer } },
        };

        layer[layer_idx].values = try Buffer.alloc(runtime, gpa, arena_nn, size_out, .normal);
        layer[layer_idx].values_g = try Buffer.alloc(runtime, gpa, arena_nn, size_out, .normal);
        capacity_forward += forward_cap;
        capacity_backward += backward_cap;
        capacity_learn += learn_cap;
        size_in = size_out;
    }

    var forward_cpu: Linearized = try Linearized.alloc(arena_temp, capacity_forward);
    var backward_cpu: Linearized = try Linearized.alloc(arena_temp, capacity_backward);
    var learn_cpu: Linearized = try Linearized.alloc(arena_temp, capacity_learn);

    var values_prev: Buffer = in;
    layer_idx = 0;
    while (layer_idx < config.len) : (layer_idx += 1) {
        switch (layer[layer_idx].tag) {
            .dense => |*d| {
                size_in = values_prev.view().size;
                values_prev.moveReshape(.{ .a = 1, .z = 1, .y = size_in.productOfElements(), .x = 1 });
                d.forward(&forward_cpu, values_prev, &layer[layer_idx].values);
                values_prev.moveReshape(size_in);
            },
            .convolution => |*c| c.forward(&forward_cpu, values_prev, &layer[layer_idx].values),
            .reduce => |*r| r.forward(&forward_cpu, &values_prev, &layer[layer_idx].values),
            .split => |*s| s.forward(&forward_cpu, values_prev, &layer[layer_idx].values),
            // .residual => |*r| r.forward(&layer[r.in_layer].values, Buffer.fetch(layer[layer_idx].values_id)),
            .residual => util.todo(@src()),
        }
        layer[layer_idx].activation.forward(&forward_cpu, layer[layer_idx].values);
        values_prev = layer[layer_idx].values;
    }

    var values_g_next: Buffer = if (layer.len > 1) layer[layer.len - 1].values_g else in_g;
    var values_next: Buffer = if (layer.len > 1) layer[layer.len - 1].values else in;
    var layer_idx_plus_one: u32 = @intCast(config.len);
    while (layer_idx_plus_one > 0) : (layer_idx_plus_one -= 1) {
        layer_idx = layer_idx_plus_one - 1;
        layer[layer_idx].activation.backward(&backward_cpu, values_next, values_g_next);
        switch (layer[layer_idx].tag) {
            .dense => |*d| {
                size_in = values_next.view().size;
                values_next.moveReshape(.{ .a = 1, .z = 1, .y = size_in.productOfElements(), .x = 1 });
                values_g_next.moveReshape(.{ .a = 1, .z = 1, .y = size_in.productOfElements(), .x = 1 });
                d.backward(&backward_cpu, values_next, &values_g_next, layer[layer_idx].values_g);
                values_next.moveReshape(size_in);
                values_g_next.moveReshape(size_in);
            },
            .convolution => |*c| c.backward(&backward_cpu, values_next, &values_g_next, //
                layer[layer_idx].values, &layer[layer_idx].values_g),
            .reduce => |*r| r.backward(&backward_cpu, &values_g_next, &layer[layer_idx].values_g),
            .split => |*s| s.backward(&backward_cpu, values_next, values_g_next, &layer[layer_idx].values_g),
            // .residual => |r| r.backward(&layer[r.in_layer].values_g, &layer[layer_idx].values_g),
            .residual => util.todo(@src()),
        }
        values_g_next = if (layer_idx == 0) in_g else layer[layer_idx - 1].values_g;
        values_next = if (layer_idx == 0) in else layer[layer_idx - 1].values;
    }
    layer_idx = 0;
    while (layer_idx < config.len) : (layer_idx += 1) {
        switch (layer[layer_idx].tag) {
            .dense => |*d| {
                learn_cpu.unaryMultiply(d.weights_g, 0.01);
                learn_cpu.unaryMultiply(d.biases_g, 0.01);
                learn_cpu.binarySubtract(d.weights, d.weights_g);
                learn_cpu.binarySubtract(d.biases, d.biases_g);
            },
            .convolution => |*c| {
                learn_cpu.unaryMultiply(c.weights_g, 0.01);
                learn_cpu.unaryMultiply(c.biases_g, 0.01);
                learn_cpu.binarySubtract(c.weights, c.weights_g);
                learn_cpu.binarySubtract(c.biases, c.biases_g);
            },
            .reduce => {},
            .split => |*s| {
                learn_cpu.unaryMultiply(s.weights_g, 0.01);
                learn_cpu.unaryMultiply(s.biases_g, 0.01);
                learn_cpu.binarySubtract(s.weights, s.weights_g);
                learn_cpu.binarySubtract(s.biases, s.biases_g);
            },
            .residual => {},
        }
    }

    const forward_optimization_depth: u32 = 10 * forward_cpu.num;
    const forward_compiled: Program = try Program.alloc(runtime, gpa, arena_nn, arena_temp, forward_cpu, //
        forward_optimization_depth, size_global, size_local);
    errdefer forward_compiled.free(runtime);
    const backward_optimization_depth: u32 = 10 * backward_cpu.num;
    const backward_compiled: Program = try Program.alloc(runtime, gpa, arena_nn, arena_temp, backward_cpu, //
        backward_optimization_depth, size_global, size_local);
    errdefer forward_compiled.free(runtime);
    const learn_optimization_depth: u32 = 10 * learn_cpu.num;
    const learn_compiled: Program = try Program.alloc(runtime, gpa, arena_nn, arena_temp, learn_cpu, //
        learn_optimization_depth, size_global, size_local);

    return .{
        .arena = arena_nn_allocator,
        .layer = layer,
        .in = in,
        .in_g = in_g,
        .forward_compiled = forward_compiled,
        .backward_compiled = backward_compiled,
        .learn_compiled = learn_compiled,
        .runtime = runtime,
    };
}
pub fn free(neuralnet: Neuralnet) void {
    neuralnet.in.free(neuralnet.runtime);
    neuralnet.in_g.free(neuralnet.runtime);
    neuralnet.forward_compiled.free(neuralnet.runtime);
    neuralnet.backward_compiled.free(neuralnet.runtime);
    neuralnet.learn_compiled.free(neuralnet.runtime);
    for (neuralnet.layer) |layer| {
        layer.activation.free(neuralnet.runtime);
        layer.values.free(neuralnet.runtime);
        layer.values_g.free(neuralnet.runtime);
        switch (layer.tag) {
            .dense => |*d| d.free(neuralnet.runtime),
            .convolution => |*c| c.free(neuralnet.runtime),
            .reduce => {},
            .split => |*s| s.free(neuralnet.runtime),
            .residual => {},
        }
    }
    neuralnet.arena.deinit();
}
// $TODO Add forward only pass where some additionaly buffer can be intermediaries
pub fn forward(neuralnet: Neuralnet) !void {
    try neuralnet.forward_compiled.run(neuralnet.runtime);
}
/// Input and output buffers have the same a_size as eachother and otherwise the same size of the nn in/output
pub fn backward(neuralnet: *Neuralnet, in: *Buffer, out: *Buffer) !void {
    assert(in.buffer.a_size == out.buffer.a_size);
    assert(in.buffer.offset == 0);
    assert(out.buffer.offset == 0);
    const layers: u32 = @intCast(neuralnet.layer.len);
    const a_size = in.buffer.a_size;
    in.moveReshape(1, in.buffer.z_size, in.buffer.y_size, in.buffer.x_size);
    out.moveReshape(1, out.buffer.z_size, out.buffer.y_size, out.buffer.x_size);
    var a_idx: u32 = 0;
    // $TODO Rework this. This should ideally all be performed on the specified compute device
    while (a_idx < a_size) : (a_idx += 1) {
        in.moveOffset(a_idx, 0, 0, 0);
        out.moveOffset(a_idx, 0, 0, 0);
        neuralnet.in.binarySet(in);
        neuralnet.in.realize();
        try neuralnet.sync(true, true, true, false, false, .sync_to_device);
        try neuralnet.forward();
        try neuralnet.sync(true, true, true, false, false, .sync_to_host);
        neuralnet.layer[layers - 1].values_g.binarySet(&neuralnet.layer[layers - 1].values);
        neuralnet.layer[layers - 1].values_g.binarySubtract(out);
        // Technically there is a ` * 2` here because it's mean square error but that's just a constant factor so it doesn't really matter
        neuralnet.layer[layers - 1].values_g.realize();
        try neuralnet.sync(true, true, true, false, false, .sync_to_device);
        neuralnet.backward_compiled.run(neuralnet.runtime);
    }
}
pub fn learn(neuralnet: Neuralnet) !void {
    try neuralnet.learn_compiled.run(neuralnet.runtime);
}
pub fn init(neuralnet: *Neuralnet, rng: u64) !void {
    const arena_temp = neuralnet.arena.allocator();
    var linearized_temp: Linearized = try .alloc(arena_temp, @intCast(2 * neuralnet.layer.len));
    defer arena_temp.free(linearized_temp.op);
    // Normally I would use PCG here but as I already use PCG in unaryRandom there could be cases with duplicate values in the buffers
    // I don't think that would be the end of the world but it just kinda ugly
    var default_prng = DefaultPrng.init(rng);
    var prng: std.Random = default_prng.random();
    for (neuralnet.layer) |*layer| {
        switch (layer.tag) {
            .dense => |*d| {
                linearized_temp.unaryRandom(d.weights, prng.int(u32));
                linearized_temp.unaryRandom(d.biases, prng.int(u32));
            },
            .convolution => |*c| {
                linearized_temp.unaryRandom(c.weights, prng.int(u32));
                linearized_temp.unaryRandom(c.biases, prng.int(u32));
            },
            .reduce => {},
            .split => |*s| {
                linearized_temp.unaryRandom(s.weights, prng.int(u32));
                linearized_temp.unaryRandom(s.biases, prng.int(u32));
            },
            .residual => {},
        }
    }
    linearized_temp.realize();
    try neuralnet.sync(true, true, true, true, true, .sync_to_device);
}
// $TODO Snyc option for temp buffers
pub fn sync(
    neuralnet: *Neuralnet,
    comptime force: bool,
    comptime in: bool,
    comptime out: bool,
    comptime weights: bool,
    comptime values: bool,
    t: Buffer.SyncStatus,
) !void {
    assert(t != .sync_to_none);
    if (in) {
        if (force) {
            neuralnet.in.syncUpdate(t);
            neuralnet.in_g.syncUpdate(t);
        }
        switch (t) {
            .sync_to_device => {
                try neuralnet.in.syncToDevice(neuralnet.runtime);
                try neuralnet.in_g.syncToDevice(neuralnet.runtime);
            },
            .sync_to_host => {
                try neuralnet.in.syncToHost(neuralnet.runtime);
                try neuralnet.in_g.syncToHost(neuralnet.runtime);
            },
            .sync_to_none => unreachable,
        }
    }
    for (neuralnet.layer, 0..) |*layer, layer_idx| {
        switch (layer.tag) {
            .dense, .convolution, .split => {},
            .reduce, .residual => continue,
        }
        if (weights) {
            const weights_curr: *Buffer = switch (layer.tag) {
                .dense => |*d| &d.weights,
                .convolution => |*c| &c.weights,
                .reduce => unreachable,
                .split => |*s| &s.weights,
                .residual => unreachable,
            };
            const biases_curr: *Buffer = switch (layer.tag) {
                .dense => |*d| &d.biases,
                .convolution => |*c| &c.biases,
                .reduce => unreachable,
                .split => |*s| &s.biases,
                .residual => unreachable,
            };
            const weights_g_curr: *Buffer = switch (layer.tag) {
                .dense => |*d| &d.weights_g,
                .convolution => |*c| &c.weights_g,
                .reduce => unreachable,
                .split => |*s| &s.weights_g,
                .residual => unreachable,
            };
            const biases_g_curr: *Buffer = switch (layer.tag) {
                .dense => |*d| &d.biases_g,
                .convolution => |*c| &c.biases_g,
                .reduce => unreachable,
                .split => |*s| &s.biases_g,
                .residual => unreachable,
            };
            if (force) {
                weights_curr.syncUpdate(t);
                biases_curr.syncUpdate(t);
                weights_g_curr.syncUpdate(t);
                biases_g_curr.syncUpdate(t);
            }
            switch (t) {
                .sync_to_device => {
                    try weights_curr.syncToDevice(neuralnet.runtime);
                    try biases_curr.syncToDevice(neuralnet.runtime);
                    try weights_g_curr.syncToDevice(neuralnet.runtime);
                    try biases_g_curr.syncToDevice(neuralnet.runtime);
                },
                .sync_to_host => {
                    try weights_curr.syncToHost(neuralnet.runtime);
                    try biases_curr.syncToHost(neuralnet.runtime);
                    try weights_g_curr.syncToHost(neuralnet.runtime);
                    try biases_g_curr.syncToHost(neuralnet.runtime);
                },
                .sync_to_none => unreachable,
            }
        }
        if (values or (out and layer_idx == neuralnet.layer.len - 1)) {
            if (force) {
                layer.values.syncUpdate(t);
                layer.values_g.syncUpdate(t);
            }
            switch (t) {
                .sync_to_device => {
                    try layer.values.syncToDevice(neuralnet.runtime);
                    try layer.values_g.syncToDevice(neuralnet.runtime);
                },
                .sync_to_host => {
                    try layer.values.syncToHost(neuralnet.runtime);
                    try layer.values_g.syncToHost(neuralnet.runtime);
                },
                .sync_to_none => unreachable,
            }
        }
    }

    try neuralnet.runtime.queueWait();
}
/// File format v0:
/// Every field is u32
/// Activation in order: none, relu, sigmoid, relu_clipped, relu_leaky, silu, gelu, tanh
/// Arch:
/// 8 version bytes ++
/// dense       -> "d++out++act\n"
/// convolution -> "c++sze++str++pdd++flt++act\n"
/// reduce      -> "r++sze++str++typ\n"
/// split       -> "s++flt++act\n"
/// residual    -> "R++frm++typ\n"
/// ++ 8 bytes XxHash init with version number
/// Param:
/// 8 version bytes ++ layer 0 [weights][biases] ... ++ layer n [weights][biases] ++ 8 bytes XxHash64 hash init with version number
const format_version: u64 = 0;
pub fn save(neuralnet: *Neuralnet, file_param_name: []const u8, file_arch_name: []const u8, force: bool) !void {
    const file_param = try std.fs.cwd().createFile(file_param_name, .{ .exclusive = !force, .truncate = true });
    defer file_param.close();
    const file_arch = try std.fs.cwd().createFile(file_arch_name, .{ .exclusive = !force, .truncate = true });
    defer file_arch.close();
    var buffer: [4096]u8 = @splat(0);
    try file_arch.writeAll(&std.mem.toBytes(format_version));
    try file_param.writeAll(&std.mem.toBytes(format_version));
    var hash_arch = std.hash.XxHash64.init(format_version);
    buffer[0] = 'i';
    @memcpy(buffer[1..5], &std.mem.toBytes(neuralnet.in.buffer.z_size));
    @memcpy(buffer[5..9], &std.mem.toBytes(neuralnet.in.buffer.y_size));
    @memcpy(buffer[9..13], &std.mem.toBytes(neuralnet.in.buffer.x_size));
    buffer[13] = '\n';
    try file_arch.writeAll(buffer[0..14]);
    hash_arch.update(buffer[0..14]);
    var hash_param = std.hash.XxHash64.init(format_version);
    for (neuralnet.layer) |*layer| {
        switch (layer.tag) {
            .dense => |d| {
                buffer[0] = 'd';
                @memcpy(buffer[1..5], &std.mem.toBytes(d.size_out));
                @memcpy(buffer[5..9], &std.mem.toBytes(layer.activation.t));
                buffer[9] = '\n';
                try file_arch.writeAll(buffer[0..10]);
                hash_arch.update(buffer[0..10]);
            },
            .convolution => |c| {
                buffer[0] = 'c';
                @memcpy(buffer[1..5], &std.mem.toBytes(c.kernel_size));
                @memcpy(buffer[5..9], &std.mem.toBytes(c.kernel_stride));
                @memcpy(buffer[9..13], &std.mem.toBytes(c.kernel_padding));
                @memcpy(buffer[13..17], &std.mem.toBytes(c.filters));
                @memcpy(buffer[17..21], &std.mem.toBytes(layer.activation.t));
                buffer[21] = '\n';
                try file_arch.writeAll(buffer[0..22]);
                hash_arch.update(buffer[0..22]);
            },
            .reduce => |r| {
                buffer[0] = 'r';
                @memcpy(buffer[1..5], &std.mem.toBytes(r.kernel_size));
                @memcpy(buffer[5..9], &std.mem.toBytes(r.kernel_stride));
                @memcpy(buffer[9..13], &std.mem.toBytes(r.t));
                buffer[13] = '\n';
                try file_arch.writeAll(buffer[0..14]);
                hash_arch.update(buffer[0..14]);
            },
            .split => |s| {
                buffer[0] = 's';
                @memcpy(buffer[1..5], &std.mem.toBytes(s.filters));
                @memcpy(buffer[5..9], &std.mem.toBytes(layer.activation.t));
                buffer[9] = '\n';
                try file_arch.writeAll(buffer[0..10]);
                hash_arch.update(buffer[0..10]);
            },
            .residual => |r| {
                buffer[0] = 'R';
                @memcpy(buffer[1..5], &std.mem.toBytes(r.in_layer));
                @memcpy(buffer[5..9], &std.mem.toBytes(r.t));
                buffer[9] = '\n';
                try file_arch.writeAll(buffer[0..10]);
                hash_arch.update(buffer[0..10]);
            },
        }
        switch (layer.tag) {
            .dense, .convolution, .split => {},
            .reduce, .residual => continue,
        }
        const weights: *Buffer = switch (layer.tag) {
            .dense => |*d| &d.weights,
            .convolution => |*c| &c.weights,
            .reduce => unreachable,
            .split => |*s| &s.weights,
            .residual => unreachable,
        };
        const biases: *Buffer = switch (layer.tag) {
            .dense => |*d| &d.biases,
            .convolution => |*c| &c.biases,
            .reduce => unreachable,
            .split => |*s| &s.biases,
            .residual => unreachable,
        };
        const weights_bytes: []const u8 = std.mem.sliceAsBytes(weights.buffer.values);
        const biases_bytes: []const u8 = std.mem.sliceAsBytes(biases.buffer.values);
        try file_param.writeAll(weights_bytes);
        try file_param.writeAll(biases_bytes);
        hash_param.update(weights_bytes);
        hash_param.update(biases_bytes);
    }
    try file_arch.writeAll(&std.mem.toBytes(hash_arch.final()));
    try file_param.writeAll(&std.mem.toBytes(hash_param.final()));
}
/// Arbitrary value. Increase if not sufficient.
const file_param_size_max: usize = 4 * 1000 * 1000 * 1000;
/// Arbitrary value. Increase if not sufficient.
const file_arch_size_max: usize = 4 * 1000 * 1000;
/// Returns the number of bytes read
fn readParamsV0(weights: *Buffer, biases: *Buffer, bytes: []const u8) u64 {
    const weights_size: u64 = weights.buffer.values.len *
        @sizeOf(@TypeOf(weights.buffer.values[0]));
    const biases_size: u64 = biases.buffer.values.len * @sizeOf(@TypeOf(biases.buffer.values[0]));
    const weights_slice: []const f32 = @alignCast(std.mem.bytesAsSlice( //
        @TypeOf(biases.buffer.values[0]), bytes[0..weights_size]));
    const biases_slice: []const f32 = @alignCast(std.mem.bytesAsSlice( //
        @TypeOf(biases.buffer.values[0]), bytes[weights_size .. weights_size + biases_size]));
    std.mem.copyBackwards(f32, weights.buffer.values, weights_slice);
    std.mem.copyBackwards(f32, biases.buffer.values, biases_slice);
    return weights_size + biases_size;
}
// Allocator is used to read the entire file at once
pub fn readParams(neuralnet: *Neuralnet, gpa: Allocator, file_param_name: []const u8) !void {
    const file_param = try std.fs.cwd().openFile(file_param_name, .{ .mode = .read_only });
    var idx: u64 = 8;
    const file_param_bytes: []const u8 = file_param.readToEndAlloc(gpa, file_param_size_max) catch |err| switch (err) {
        error.FileTooBig => {
            std.log.err("File {s} exceeds max size of {} bytes. Increase `file_param_size_max` if this is intentional.\n", //
                .{ file_param_name, file_param_size_max });
            return err;
        },
        else => return err,
    };
    defer gpa.free(file_param_bytes);
    const format_version_read: u64 = std.mem.bytesToValue(u64, file_param_bytes[0..8]);
    var hash = std.hash.XxHash64.init(format_version_read);
    for (neuralnet.layer) |*layer| {
        switch (layer.tag) {
            .dense, .convolution, .split => {},
            .reduce, .residual => continue,
        }
        const weights: *Buffer = switch (layer.tag) {
            .dense => |*d| &d.weights,
            .convolution => |*c| &c.weights,
            .reduce => unreachable,
            .split => |*s| &s.weights,
            .residual => unreachable,
        };
        const biases: *Buffer = switch (layer.tag) {
            .dense => |*d| &d.biases,
            .convolution => |*c| &c.biases,
            .reduce => unreachable,
            .split => |*s| &s.biases,
            .residual => unreachable,
        };
        switch (format_version_read) {
            0 => idx += readParamsV0(weights, biases, file_param_bytes[idx..]),
            else => {
                std.log.err("Unrecognized file format version {} for param file {s}\n", //
                    .{ format_version_read, file_param_name });
                unreachable;
            },
        }
        hash.update(std.mem.sliceAsBytes(weights.buffer.values));
        hash.update(std.mem.sliceAsBytes(biases.buffer.values));
    }
    if (file_param_bytes.len != idx + 8) return error.ParamFileWrongSize;
    if (hash.final() != std.mem.bytesToValue(u64, file_param_bytes[idx..]))
        return error.ParamHashMismatch;
}
fn readInputV0(bytes: []const u8) struct { z: u32, y: u32, x: u32 } {
    assert(bytes[0] == 'i');
    return .{
        .z = std.mem.bytesAsValue(u32, bytes[1..5]).*,
        .y = std.mem.bytesAsValue(u32, bytes[5..9]).*,
        .x = std.mem.bytesAsValue(u32, bytes[9..13]).*,
    };
}
fn readArchV0(bytes: []const u8) Layer.Config {
    assert(bytes[0] == 'd' or bytes[0] == 'c' or bytes[0] == 'r' or bytes[0] == 's' or bytes[0] == 'R');
    return switch (bytes[0]) {
        'd' => .{ .dense = .{
            .size_out = std.mem.bytesAsValue(u32, bytes[1..5]).*,
            .activation_kind = std.mem.bytesAsValue(Activation.Kind, bytes[5..9]).*,
        } },
        'c' => .{ .convolution = .{
            .kernel_size = std.mem.bytesAsValue(u32, bytes[1..5]).*,
            .kernel_stride = std.mem.bytesAsValue(u32, bytes[5..9]).*,
            .kernel_padding = std.mem.bytesAsValue(u32, bytes[9..13]).*,
            .filters = std.mem.bytesAsValue(u32, bytes[13..17]).*,
            .activation_kind = std.mem.bytesAsValue(Activation.Kind, bytes[17..21]).*,
        } },
        'r' => .{ .reduce = .{
            .kernel_size = std.mem.bytesAsValue(u32, bytes[1..5]).*,
            .kernel_stride = std.mem.bytesAsValue(u32, bytes[5..9]).*,
            .t = std.mem.bytesAsValue(Reduce.Kind, bytes[9..13]).*,
        } },
        's' => .{ .split = .{
            .filters = std.mem.bytesAsValue(u32, bytes[1..5]).*,
            .activation_kind = std.mem.bytesAsValue(Activation.Kind, bytes[5..9]).*,
        } },
        'R' => .{ .residual = .{
            .in_layer = std.mem.bytesAsValue(u32, bytes[1..5]).*,
            .t = std.mem.bytesAsValue(Residual.Kind, bytes[5..9]).*,
        } },
        else => unreachable,
    };
}
pub fn readArch(gpa: Allocator, file_arch_name: []const u8) !struct {
    config: []const Layer.Config,
    z_in: u32,
    y_in: u32,
    x_in: u32,
} {
    const file_arch = std.fs.cwd().openFile(file_arch_name, .{ .mode = .read_only }) catch |err| switch (err) {
        error.FileTooBig => {
            std.log.err("File {s} exceeds max size of {} bytes. Increase `file_arch_size_max` if this is intentional.\n", //
                .{ file_arch_name, file_arch_size_max });
            return err;
        },
        else => return err,
    };
    defer file_arch.close();
    var idx: u64 = 0;
    const file_arch_bytes: []const u8 = try file_arch.readToEndAlloc(gpa, file_arch_size_max);
    defer gpa.free(file_arch_bytes);
    assert(file_arch_bytes.len > 16); // The version and hash already take up 16 bytes and there has to be other info on top of that
    const newlines: u64 = std.mem.count(u8, file_arch_bytes, "\n");
    assert(newlines >= 2);
    const layers: u64 = newlines - 1; // Take of the newline for the input descriptor
    const format_version_read: u64 = std.mem.bytesToValue(u64, file_arch_bytes[0..8]);
    idx += 8;
    var hash_arch = std.hash.XxHash64.init(format_version_read);
    const arch_input_end: u64 = std.mem.indexOfScalar(u8, file_arch_bytes[idx..], '\n') orelse unreachable;
    const arch_input: []const u8 = file_arch_bytes[idx .. idx + arch_input_end + 1];
    hash_arch.update(arch_input);
    const size_in = switch (format_version_read) {
        0 => readInputV0(arch_input),
        else => unreachable,
    };
    idx += 14;
    const config: []Layer.Config = try gpa.alloc(Layer.Config, layers);
    var layer_idx: u64 = 0;
    while (layer_idx < layers) : (layer_idx += 1) {
        const idx_newline: u64 = std.mem.indexOfScalar(u8, file_arch_bytes[idx..], '\n') orelse unreachable;
        config[layer_idx] = switch (format_version_read) {
            0 => readArchV0(file_arch_bytes[idx .. idx + idx_newline + 1]),
            else => unreachable,
        };
        hash_arch.update(file_arch_bytes[idx .. idx + idx_newline + 1]);
        idx += idx_newline + 1;
    }
    assert(idx + 8 == file_arch_bytes.len);
    const hash: u64 = hash_arch.final();
    const hash_read: u64 = std.mem.bytesAsValue(u64, file_arch_bytes[idx .. idx + 8]).*;
    if (hash != hash_read) {
        return error.ArchHashMismatch;
    }
    return .{
        .config = config,
        .z_in = size_in.z,
        .y_in = size_in.y,
        .x_in = size_in.x,
    };
}
pub fn print(neuralnet: Neuralnet, padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
    if (name) |text| {
        util.log.print("{s}Neuralnet {s}\n", .{ " " ** offset, text });
    } else {
        util.log.print("{s}Neuralnet\n", .{" " ** offset});
    }
    util.log.print("{s}Layers:\n", .{" " ** (offset + padding)});
    for (neuralnet.layer, 0..) |*layer, layer_idx| {
        util.log.print("{s}[{}] ->\n", .{ " " ** (offset + 2 * padding), layer_idx });
        switch (layer.tag) {
            .dense => |d| d.print(padding, offset + 2 * padding, null),
            .convolution => |c| c.print(padding, offset + 2 * padding, null),
            .reduce => |r| r.print(padding, offset + 2 * padding, null),
            .split => |s| s.print(padding, offset + 2 * padding, null),
            .residual => |r| r.print(padding, offset + 2 * padding, null),
        }
    }
}
