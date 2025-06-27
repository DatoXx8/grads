const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const DefaultPrng = std.Random.DefaultPrng;

const Optimization = @import("./compiler/optimize.zig").Optimization;
const Program = @import("./compiler/Program.zig");
const Layer = @import("./Layer.zig");
const Activation = Layer.Activation;
const Dense = Layer.Dense;
const Convolution = Layer.Convolution;
const Reduce = Layer.Reduce;
const Split = Layer.Split;
const Residual = Layer.Residual;
const cl = @import("./runtimes/cl.zig");
const ClContext = cl.ClContext;
const ClDevice = cl.ClDevice;
const ClCommandQueue = cl.ClCommandQueue;
const Tensor = @import("./Tensor.zig");
const Linearized = Tensor.Linearized;
const Op = Tensor.Op;
const Buffer = Tensor.Buffer;
const todo = @import("./util.zig").todo;

pub const Neuralnet = @This();
layer: []Layer,
queue: ClCommandQueue,
in: Tensor,
in_g: Tensor,
forward_cl: Program,
backward_cl: Program,
learn_cl: Program,
forward_cpu: Linearized,
backward_cpu: Linearized,
learn_cpu: Linearized,
pub fn alloc(
    allocator: Allocator,
    z_size: u32,
    y_size: u32,
    x_size: u32,
    config: []const Layer.Config,
    size_global: u32,
    size_local: u32,
    device: ClDevice,
    context: ClContext,
    queue: ClCommandQueue,
) !Neuralnet {
    assert(size_global > 0);
    assert(size_local > 0);
    assert(size_global % size_local == 0);
    assert(z_size > 0);
    assert(y_size > 0);
    assert(x_size > 0);

    var layer: []Layer = try allocator.alloc(Layer, config.len);
    errdefer allocator.free(layer);

    var capacity_forward: u32 = 0;
    var capacity_backward: u32 = 0;
    var capacity_learn: u32 = 0;
    var z_in: u32 = z_size;
    var y_in: u32 = y_size;
    var x_in: u32 = x_size;
    const in: Tensor = try Tensor.alloc(allocator, 1, z_in, y_in, x_in, context, 1);
    var in_g: Tensor = try Tensor.alloc(allocator, 1, z_in, y_in, x_in, context, 1);
    var layer_idx: u32 = 0;
    while (layer_idx < config.len) : (layer_idx += 1) {
        // $TODO Activation and norming
        const size_in: u32 = z_in * y_in * x_in;
        const z_out: u32 = switch (config[layer_idx]) {
            .dense => 1,
            .convolution => |c| c.filters,
            .reduce => z_in,
            .split => |s| z_in * s.filters,
            .residual => z_in,
        };
        const y_out: u32 = switch (config[layer_idx]) {
            .dense => 1,
            .convolution => |c| Convolution.sizeNew(y_in, c.kernel_size, c.kernel_stride, c.kernel_padding),
            .reduce => |r| Reduce.sizeNew(y_in, r.kernel_size, r.kernel_stride),
            .split => y_in,
            .residual => y_in,
        };
        const x_out: u32 = switch (config[layer_idx]) {
            .dense => |d| d.size_out,
            .convolution => |c| Convolution.sizeNew(x_in, c.kernel_size, c.kernel_stride, c.kernel_padding),
            .reduce => |r| Reduce.sizeNew(x_in, r.kernel_size, r.kernel_stride),
            .split => x_in,
            .residual => x_in,
        };
        const forward_cap: u32 = switch (config[layer_idx]) {
            .dense => |d| d.size_out * 3 + 1,
            .convolution => |c| 3 * c.filters * y_out * x_out,
            .reduce => y_out * x_out,
            .split => |s| 3 * s.filters,
            .residual => 1,
        };
        const backward_cap: u32 = switch (config[layer_idx]) {
            .dense => |d| 2 + d.size_out + 4 * size_in,
            .convolution => |c| 2 * c.filters + 6 * c.filters * y_out * x_out,
            .reduce => y_out * x_out,
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
        layer[layer_idx].activation = try Activation.alloc(allocator, switch (config[layer_idx]) {
            .dense => |d| d.activation_type,
            .convolution => |c| c.activation_type,
            .reduce => .none,
            .split => |s| s.activation_type,
            .residual => .none,
        }, z_out, y_out, x_out, context);
        layer[layer_idx].tag = switch (config[layer_idx]) {
            .dense => |d| .{ .dense = try Dense.alloc(allocator, size_in, d.size_out, context) },
            .convolution => |c| .{
                .convolution = try Convolution.alloc(allocator, z_in, y_in, x_in, c.filters, c.kernel_size, //
                    c.kernel_stride, c.kernel_padding, context),
            },
            .reduce => |r| .{ .reduce = Reduce.init(z_in, y_in, x_in, r.kernel_size, r.kernel_stride, r.t) },
            .split => |s| .{ .split = try Split.alloc(allocator, s.filters, z_in, y_in, x_in, context) },
            .residual => |r| .{ .residual = .{ .t = .identity, .in_layer = r.in_layer } },
        };

        layer[layer_idx].values = try Tensor.alloc(allocator, 1, z_out, y_out, x_out, context, forward_cap);
        if (layer_idx == 0) {
            try in_g.linearized.capacityEnsure(allocator, backward_cap);
        } else {
            layer[layer_idx - 1].values_g = try Tensor.alloc(allocator, 1, z_in, y_in, x_in, context, backward_cap);
        }
        if (layer_idx == config.len - 1) {
            layer[layer_idx].values_g = try Tensor.alloc(allocator, 1, z_out, y_out, x_out, context, 2);
        }
        capacity_forward += forward_cap;
        capacity_backward += backward_cap;
        capacity_learn += learn_cap;
        z_in = layer[layer_idx].values.buffer.z_size;
        y_in = layer[layer_idx].values.buffer.y_size;
        x_in = layer[layer_idx].values.buffer.x_size;
    }

    var forward_cpu: Linearized = try Linearized.alloc(allocator, capacity_forward);
    errdefer forward_cpu.free(allocator);
    var backward_cpu: Linearized = try Linearized.alloc(allocator, capacity_backward);
    errdefer backward_cpu.free(allocator);
    var learn_cpu: Linearized = try Linearized.alloc(allocator, capacity_learn);
    errdefer learn_cpu.free(allocator);

    var values_prev: Tensor = in;
    layer_idx = 0;
    while (layer_idx < config.len) : (layer_idx += 1) {
        switch (layer[layer_idx].tag) {
            .dense => |*d| {
                z_in = values_prev.buffer.z_size;
                y_in = values_prev.buffer.y_size;
                x_in = values_prev.buffer.x_size;
                values_prev.moveReshape(1, 1, z_in * y_in * x_in, 1);
                d.forward(&values_prev, &layer[layer_idx].values);
                values_prev.moveReshape(1, z_in, y_in, x_in);
            },
            .convolution => |*c| c.forward(&values_prev, &layer[layer_idx].values),
            .reduce => |*r| r.forward(&values_prev, &layer[layer_idx].values),
            .split => |*s| s.forward(&values_prev, &layer[layer_idx].values),
            // .residual => |*r| r.forward(&layer[r.in_layer].values, &layer[layer_idx].values),
            .residual => todo(@src()),
        }
        layer[layer_idx].activation.forward(&layer[layer_idx].values);
        forward_cpu.concat(&layer[layer_idx].values.linearized);
        values_prev = layer[layer_idx].values;
    }
    var values_g_next: Tensor = if (layer.len > 1) layer[layer.len - 1].values_g else in_g;
    var values_next: Tensor = if (layer.len > 1) layer[layer.len - 1].values else in;
    var layer_idx_plus_one: u32 = @intCast(config.len);
    while (layer_idx_plus_one > 0) : (layer_idx_plus_one -= 1) {
        layer_idx = layer_idx_plus_one - 1;
        switch (layer[layer_idx].tag) {
            .dense => |*d| {
                z_in = values_next.buffer.z_size;
                y_in = values_next.buffer.y_size;
                x_in = values_next.buffer.x_size;
                values_next.moveReshape(1, 1, z_in * y_in * x_in, 1);
                values_g_next.moveReshape(1, 1, z_in * y_in * x_in, 1);
                d.backward(&values_next, &values_g_next, &layer[layer_idx].values_g);
                values_next.moveReshape(1, z_in, y_in, x_in);
                values_g_next.moveReshape(1, z_in, y_in, x_in);
            },
            .convolution => |*c| c.backward(&values_next, &values_g_next, &layer[layer_idx].values, &layer[layer_idx].values_g),
            .reduce => |*r| r.backward(&values_g_next, &layer[layer_idx].values_g),
            .split => |*s| s.backward(&values_next, &values_g_next, &layer[layer_idx].values_g),
            // .residual => |r| r.backward(&layer[r.in_layer].values_g, &layer[layer_idx].values_g),
            .residual => todo(@src()),
        }
        // $FIXME I dunno if this should go above or below
        layer[layer_idx].activation.backward(&values_next, &values_g_next);
        switch (layer[layer_idx].tag) {
            .dense => |*d| {
                backward_cpu.concat(&d.weights_g.linearized);
                backward_cpu.concat(&d.biases_g.linearized);
            },
            .convolution => |*c| {
                backward_cpu.concat(&c.weights_g.linearized);
                backward_cpu.concat(&c.biases_g.linearized);
            },
            .reduce => {},
            .split => |*s| {
                backward_cpu.concat(&s.weights_g.linearized);
                backward_cpu.concat(&s.biases_g.linearized);
            },
            .residual => {},
        }
        backward_cpu.concat(&values_g_next.linearized);
        values_g_next = if (layer_idx == 0) in_g else layer[layer_idx - 1].values_g;
        values_next = if (layer_idx == 0) in else layer[layer_idx - 1].values;
    }
    layer_idx = 0;
    while (layer_idx < config.len) : (layer_idx += 1) {
        switch (layer[layer_idx].tag) {
            .dense => |*d| {
                d.weights_g.unaryMultiply(0.01);
                d.biases_g.unaryMultiply(0.01);
                d.weights.binarySubtract(&d.weights_g);
                d.biases.binarySubtract(&d.biases_g);
                learn_cpu.concat(&d.weights.linearized);
                learn_cpu.concat(&d.biases.linearized);
            },
            .convolution => |*c| {
                c.weights_g.unaryMultiply(0.01);
                c.biases_g.unaryMultiply(0.01);
                c.weights.binarySubtract(&c.weights_g);
                c.biases.binarySubtract(&c.biases_g);
                learn_cpu.concat(&c.weights.linearized);
                learn_cpu.concat(&c.biases.linearized);
            },
            .reduce => {},
            .split => |*s| {
                s.weights_g.unaryMultiply(0.01);
                s.biases_g.unaryMultiply(0.01);
                s.weights.binarySubtract(&s.weights_g);
                s.biases.binarySubtract(&s.biases_g);
                learn_cpu.concat(&s.weights.linearized);
                learn_cpu.concat(&s.biases.linearized);
            },
            .residual => {},
        }
    }
    const forward_cl: Program = try Program.alloc(allocator, forward_cpu, size_global, size_local, .O3, device, context, queue);
    errdefer forward_cl.free(allocator);
    const backward_cl: Program = try Program.alloc(allocator, backward_cpu, size_global, size_local, .O3, device, context, queue);
    errdefer backward_cl.free(allocator);
    const learn_cl: Program = try Program.alloc(allocator, learn_cpu, size_global, size_local, .O3, device, context, queue);
    return .{
        .layer = layer,
        .queue = queue,
        .in = in,
        .in_g = in_g,
        .forward_cl = forward_cl,
        .backward_cl = backward_cl,
        .learn_cl = learn_cl,
        .forward_cpu = forward_cpu,
        .backward_cpu = backward_cpu,
        .learn_cpu = learn_cpu,
    };
}
pub fn free(this: *@This(), allocator: Allocator) void {
    this.in.free(allocator);
    this.in_g.free(allocator);
    this.forward_cpu.free(allocator);
    this.backward_cpu.free(allocator);
    this.learn_cpu.free(allocator);
    this.forward_cl.free(allocator);
    this.backward_cl.free(allocator);
    this.learn_cl.free(allocator);
    for (this.layer) |*layer| {
        layer.activation.free(allocator);
        layer.values.free(allocator);
        layer.values_g.free(allocator);
        switch (layer.tag) {
            .dense => |*d| d.free(allocator),
            .convolution => |*c| c.free(allocator),
            .reduce => {},
            .split => |*s| s.free(allocator),
            .residual => {},
        }
    }
    allocator.free(this.layer);
}
// $TODO / $FIXME The sync handling needs to be reworked
// $TODO Add forward only pass where some additionaly tensors can be intermediaries
pub fn forward(this: *@This(), t: ClDevice.Type) !void {
    switch (t) {
        .cpu => {
            this.forward_cpu.run();
        },
        .gpu => {
            try this.forward_cl.run();
        },
    }
}
/// Input and output buffers have the same a_size as eachother and otherwise the same size of the nn in/output
pub fn backward(this: *@This(), t: ClDevice.Type, in: *Tensor, out: *Tensor) !void {
    assert(in.buffer.a_size == out.buffer.a_size);
    assert(in.buffer.offset == 0);
    assert(out.buffer.offset == 0);
    const layers: u32 = @intCast(this.layer.len);
    const a_size = in.buffer.a_size;
    in.moveReshape(1, in.buffer.z_size, in.buffer.y_size, in.buffer.x_size);
    out.moveReshape(1, out.buffer.z_size, out.buffer.y_size, out.buffer.x_size);
    var a_idx: u32 = 0;
    // $TODO Rework this. This should ideally all be performed on the specified compute device
    while (a_idx < a_size) : (a_idx += 1) {
        in.moveOffset(a_idx, 0, 0, 0);
        out.moveOffset(a_idx, 0, 0, 0);
        this.in.binarySet(in);
        this.in.realize();
        try this.sync(true, true, true, false, false, .sync_to_device);
        try this.forward(t);
        try this.sync(true, true, true, false, false, .sync_to_host);
        this.layer[layers - 1].values_g.binarySet(&this.layer[layers - 1].values);
        this.layer[layers - 1].values_g.binarySubtract(out);
        // Technically there is a ` * 2` here because it's mean square error but that's just a constant factor so it doesn't really matter
        this.layer[layers - 1].values_g.realize();
        try this.sync(true, true, true, false, false, .sync_to_device);
        switch (t) {
            .cpu => {
                this.backward_cpu.run();
            },
            .gpu => {
                try this.backward_cl.run();
            },
        }
    }
}
pub fn learn(this: *@This(), t: ClDevice.Type) !void {
    switch (t) {
        .cpu => {
            this.learn_cpu.run();
        },
        .gpu => {
            try this.learn_cl.run();
        },
    }
}
pub fn init(this: *@This(), rng: u64) !void {
    // Normally I would use PCG here but as I already use PCG in unaryRandom there could be cases with duplicate values in the tensors
    // I don't think that would be the end of the world but it just kinda ugly
    var default_prng = DefaultPrng.init(rng);
    var prng: std.Random = default_prng.random();
    for (this.layer) |*layer| {
        switch (layer.tag) {
            .dense => |*d| {
                d.weights.unaryRandom(prng.int(u32));
                d.biases.unaryRandom(prng.int(u32));
                d.weights.realize();
                d.biases.realize();
            },
            .convolution => |*c| {
                c.weights.unaryRandom(prng.int(u32));
                c.biases.unaryRandom(prng.int(u32));
                c.weights.realize();
                c.biases.realize();
            },
            .reduce => {},
            .split => |*s| {
                s.weights.unaryRandom(prng.int(u32));
                s.biases.unaryRandom(prng.int(u32));
                s.weights.realize();
                s.biases.realize();
            },
            .residual => {},
        }
    }
    try this.sync(true, true, true, true, true, .sync_to_device);
}
// $TODO Snyc option for temp buffers
pub fn sync(
    this: *@This(),
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
            this.in.buffer.syncUpdate(t);
            this.in_g.buffer.syncUpdate(t);
        }
        switch (t) {
            .sync_to_device => {
                try this.in.buffer.syncToDevice(this.queue);
                try this.in_g.buffer.syncToDevice(this.queue);
            },
            .sync_to_host => {
                try this.in.buffer.syncToHost(this.queue);
                try this.in_g.buffer.syncToHost(this.queue);
            },
            .sync_to_none => unreachable,
        }
    }
    for (this.layer, 0..) |*layer, layer_idx| {
        switch (layer.tag) {
            .dense, .convolution, .split => {},
            .reduce, .residual => continue,
        }
        if (weights) {
            const weights_curr: *Tensor = switch (layer.tag) {
                .dense => |*d| &d.weights,
                .convolution => |*c| &c.weights,
                .reduce => unreachable,
                .split => |*s| &s.weights,
                .residual => unreachable,
            };
            const biases_curr: *Tensor = switch (layer.tag) {
                .dense => |*d| &d.biases,
                .convolution => |*c| &c.biases,
                .reduce => unreachable,
                .split => |*s| &s.biases,
                .residual => unreachable,
            };
            const weights_g_curr: *Tensor = switch (layer.tag) {
                .dense => |*d| &d.weights_g,
                .convolution => |*c| &c.weights_g,
                .reduce => unreachable,
                .split => |*s| &s.weights_g,
                .residual => unreachable,
            };
            const biases_g_curr: *Tensor = switch (layer.tag) {
                .dense => |*d| &d.biases_g,
                .convolution => |*c| &c.biases_g,
                .reduce => unreachable,
                .split => |*s| &s.biases_g,
                .residual => unreachable,
            };
            if (force) {
                weights_curr.buffer.syncUpdate(t);
                biases_curr.buffer.syncUpdate(t);
                weights_g_curr.buffer.syncUpdate(t);
                biases_g_curr.buffer.syncUpdate(t);
            }
            switch (t) {
                .sync_to_device => {
                    try weights_curr.buffer.syncToDevice(this.queue);
                    try biases_curr.buffer.syncToDevice(this.queue);
                    try weights_g_curr.buffer.syncToDevice(this.queue);
                    try biases_g_curr.buffer.syncToDevice(this.queue);
                },
                .sync_to_host => {
                    try weights_curr.buffer.syncToHost(this.queue);
                    try biases_curr.buffer.syncToHost(this.queue);
                    try weights_g_curr.buffer.syncToHost(this.queue);
                    try biases_g_curr.buffer.syncToHost(this.queue);
                },
                .sync_to_none => unreachable,
            }
        }
        if (values or (out and layer_idx == this.layer.len - 1)) {
            if (force) {
                layer.values.buffer.syncUpdate(t);
                layer.values_g.buffer.syncUpdate(t);
            }
            switch (t) {
                .sync_to_device => {
                    try layer.values.buffer.syncToDevice(this.queue);
                    try layer.values_g.buffer.syncToDevice(this.queue);
                },
                .sync_to_host => {
                    try layer.values.buffer.syncToHost(this.queue);
                    try layer.values_g.buffer.syncToHost(this.queue);
                },
                .sync_to_none => unreachable,
            }
        }
    }
    try this.in.buffer.syncWait(this.queue);
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
pub fn save(this: *@This(), file_param_name: []const u8, file_arch_name: []const u8, force: bool) !void {
    const file_param = try std.fs.cwd().createFile(file_param_name, .{ .exclusive = !force, .truncate = true });
    errdefer file_param.close();
    defer file_param.close();
    const file_arch = try std.fs.cwd().createFile(file_arch_name, .{ .exclusive = !force, .truncate = true });
    errdefer file_arch.close();
    defer file_arch.close();
    var buffer: [4096]u8 = @splat(0);
    try file_arch.writeAll(&std.mem.toBytes(format_version));
    try file_param.writeAll(&std.mem.toBytes(format_version));
    var hash_arch = std.hash.XxHash64.init(format_version);
    buffer[0] = 'i';
    @memcpy(buffer[1..5], &std.mem.toBytes(this.in.buffer.z_size));
    @memcpy(buffer[5..9], &std.mem.toBytes(this.in.buffer.y_size));
    @memcpy(buffer[9..13], &std.mem.toBytes(this.in.buffer.x_size));
    buffer[13] = '\n';
    try file_arch.writeAll(buffer[0..14]);
    hash_arch.update(buffer[0..14]);
    var hash_param = std.hash.XxHash64.init(format_version);
    for (this.layer) |*layer| {
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
        const weights: *Tensor = switch (layer.tag) {
            .dense => |*d| &d.weights,
            .convolution => |*c| &c.weights,
            .reduce => unreachable,
            .split => |*s| &s.weights,
            .residual => unreachable,
        };
        const biases: *Tensor = switch (layer.tag) {
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
fn readParamsV0(weights: *Tensor, biases: *Tensor, bytes: []const u8) u64 {
    const weights_size: u64 = weights.buffer.values.len * @sizeOf(@TypeOf(weights.buffer.values[0]));
    const biases_size: u64 = biases.buffer.values.len * @sizeOf(@TypeOf(biases.buffer.values[0]));
    const weights_slice: []const f32 = @alignCast(std.mem.bytesAsSlice(@TypeOf(biases.buffer.values[0]), //
        bytes[0..weights_size]));
    const biases_slice: []const f32 = @alignCast(std.mem.bytesAsSlice(@TypeOf(biases.buffer.values[0]), //
        bytes[weights_size .. weights_size + biases_size]));
    std.mem.copyBackwards(f32, weights.buffer.values, weights_slice);
    std.mem.copyBackwards(f32, biases.buffer.values, biases_slice);
    return weights_size + biases_size;
}
// Allocator is used to read the entire file at once
pub fn readParams(this: *@This(), allocator: Allocator, file_param_name: []const u8) !void {
    const file_param = try std.fs.cwd().openFile(file_param_name, .{ .mode = .read_only });
    var idx: u64 = 8;
    const file_param_bytes: []const u8 = file_param.readToEndAlloc(allocator, file_param_size_max) catch |err| switch (err) {
        error.FileTooBig => {
            std.log.err("File {s} exceeds max size of {} bytes. Increase `file_param_size_max` if this is intentional.\n", //
                .{ file_param_name, file_param_size_max });
            return err;
        },
        else => return err,
    };
    errdefer allocator.free(file_param_bytes);
    defer allocator.free(file_param_bytes);
    const format_version_read: u64 = std.mem.bytesToValue(u64, file_param_bytes[0..8]);
    var hash = std.hash.XxHash64.init(format_version_read);
    for (this.layer) |*layer| {
        switch (layer.tag) {
            .dense, .convolution, .split => {},
            .reduce, .residual => continue,
        }
        const weights: *Tensor = switch (layer.tag) {
            .dense => |*d| &d.weights,
            .convolution => |*c| &c.weights,
            .reduce => unreachable,
            .split => |*s| &s.weights,
            .residual => unreachable,
        };
        const biases: *Tensor = switch (layer.tag) {
            .dense => |*d| &d.biases,
            .convolution => |*c| &c.biases,
            .reduce => unreachable,
            .split => |*s| &s.biases,
            .residual => unreachable,
        };
        switch (format_version_read) {
            0 => idx += readParamsV0(weights, biases, file_param_bytes[idx..]),
            else => {
                std.log.err("Unrecognized file format version {} for param file {s}\n", .{ format_version_read, file_param_name });
                unreachable;
            },
        }
        hash.update(std.mem.sliceAsBytes(weights.buffer.values));
        hash.update(std.mem.sliceAsBytes(biases.buffer.values));
    }
    if (file_param_bytes.len != idx + 8) return error.ParamFileWrongSize;
    if (hash.final() != std.mem.bytesToValue(u64, file_param_bytes[idx..])) return error.ParamHashMismatch;
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
            .activation_type = std.mem.bytesAsValue(Activation.Type, bytes[5..9]).*,
        } },
        'c' => .{ .convolution = .{
            .kernel_size = std.mem.bytesAsValue(u32, bytes[1..5]).*,
            .kernel_stride = std.mem.bytesAsValue(u32, bytes[5..9]).*,
            .kernel_padding = std.mem.bytesAsValue(u32, bytes[9..13]).*,
            .filters = std.mem.bytesAsValue(u32, bytes[13..17]).*,
            .activation_type = std.mem.bytesAsValue(Activation.Type, bytes[17..21]).*,
        } },
        'r' => .{ .reduce = .{
            .kernel_size = std.mem.bytesAsValue(u32, bytes[1..5]).*,
            .kernel_stride = std.mem.bytesAsValue(u32, bytes[5..9]).*,
            .t = std.mem.bytesAsValue(Reduce.Type, bytes[9..13]).*,
        } },
        's' => .{ .split = .{
            .filters = std.mem.bytesAsValue(u32, bytes[1..5]).*,
            .activation_type = std.mem.bytesAsValue(Activation.Type, bytes[5..9]).*,
        } },
        'R' => .{ .residual = .{
            .in_layer = std.mem.bytesAsValue(u32, bytes[1..5]).*,
            .t = std.mem.bytesAsValue(Residual.Type, bytes[5..9]).*,
        } },
        else => unreachable,
    };
}
pub fn readArch(allocator: Allocator, file_arch_name: []const u8) !struct { config: []const Layer.Config, z_in: u32, y_in: u32, x_in: u32 } {
    const file_arch = std.fs.cwd().openFile(file_arch_name, .{ .mode = .read_only }) catch |err| switch (err) {
        error.FileTooBig => {
            std.log.err("File {s} exceeds max size of {} bytes. Increase `file_arch_size_max` if this is intentional.\n", //
                .{ file_arch_name, file_arch_size_max });
            return err;
        },
        else => return err,
    };
    errdefer file_arch.close();
    defer file_arch.close();
    var idx: u64 = 0;
    const file_arch_bytes: []const u8 = try file_arch.readToEndAlloc(allocator, file_arch_size_max);
    errdefer allocator.free(file_arch_bytes);
    defer allocator.free(file_arch_bytes);
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
    const config: []Layer.Config = try allocator.alloc(Layer.Config, layers);
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
pub fn print(this: @This(), padding: comptime_int, offset: comptime_int, name: ?[]const u8) void {
    if (name) |text| {
        std.debug.print("{s}Neuralnet {s}\n", .{ " " ** offset, text });
    } else {
        std.debug.print("{s}Neuralnet\n", .{" " ** offset});
    }
    std.debug.print("{s}Layers:\n", .{" " ** (offset + padding)});
    for (this.layer, 0..) |*layer, layer_idx| {
        std.debug.print("{s}[{}] ->\n", .{ " " ** (offset + 2 * padding), layer_idx });
        switch (layer.tag) {
            .dense => |d| d.print(padding, offset + 2 * padding, null),
            .convolution => |c| c.print(padding, offset + 2 * padding, null),
            .reduce => |r| r.print(padding, offset + 2 * padding, null),
            .split => |s| s.print(padding, offset + 2 * padding, null),
            .residual => |r| r.print(padding, offset + 2 * padding, null),
        }
    }
}
// switch (config[layer_idx]) {
//     .dense => |d| ,
//     .convolution => |c| ,
//     .reduce => |r| ,
//     .split => |s| ,
//     .residual => |r| ,
// }
