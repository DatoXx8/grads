const std = @import("std");

// TODO: Get rid of the pollution with passing in an allocator everywhere.
// TODO: Make the alloc and free functions unable to error. That is fairly easy with just having everything as an optional, but that is horribly disgusting
// TODO: I think all of the above can be gotten rid of by having a way to explicity interface with the linearized capacity to increase / set it as necessary
//      -> Do something like tensor.capacity_ensure(std.mem.allocator, u32) to ensure there at least that many spots free
//
// TODO: Implement weightgen and that arnold net thing where there are cubic functions as connections
// TODO: Actual error handling where it is possible
// TODO: Add autograd
// TODO: Generate linearized at comptime so that the compiler can do all the vectorization and the compiler could possibly inline everything so that there
//  is no need for going through the switch statement every time
// TODO: Switch to sea-of-ops ssa pir with explicit dependency fields

const Tensor = @import("./tensor.zig").Tensor;

const assert = std.debug.assert;

const Program = @import("./compiler/program.zig").Program;

const ClDevice = @import("./runtimes/cl.zig").ClDevice;
const ClContext = @import("./runtimes/cl.zig").ClContext;
const ClCommandQueue = @import("./runtimes/cl.zig").ClCommandQueue;

const Neuralnet = @import("./nn.zig").Neuralnet;

pub fn main() !void {
    std.debug.print("Hi :)\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const device: ClDevice = try ClDevice.alloc(.gpu);
    const context: ClContext = try ClContext.alloc(device);
    const queue: ClCommandQueue = try ClCommandQueue.alloc(device, context);

    var tensor: Tensor = try Tensor.alloc(allocator, 1, 1, 2, 2, context);
    defer tensor.free(allocator);
    try tensor.unaryRandom(allocator);
    tensor.realize();
    try tensor.buffer.syncToDevice(queue);
    try tensor.buffer.syncWait(queue);

    var nn: Neuralnet = try Neuralnet.alloc(
        allocator,
        tensor,
        &[_]Neuralnet.Layer.Config{
            // .{ .dense = .{ .size_out = 8, .activation = .none } },
            // .{ .dense = .{ .size_out = 8, .activation = .relu } },
            .{ .split = .{ .filters = 2, .activation = .none } },
        },
        20,
        4,
        .O1,
        context,
        device,
        queue,
    );
    defer nn.free(allocator) catch {};
    try nn.init(allocator);
    try nn.forward(.gpu);
    nn.layers[nn.layers.len - 1].values.print(4, 0, null);
    std.debug.print("Max {}\n", .{try device.maxSizeLocal()});
}
