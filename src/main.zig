const std = @import("std");

// TODO: Implement weightgen and that arnold net thing where there are cubic functions as connections
// TODO: Actual error handling where it is possible
// TODO: Add autograd
// TODO: Generate linearized at comptime so that the compiler can do all the vectorization and the compiler could possibly inline everything so that there
//  is no need for going through the switch statement every time
// TODO: Really need to compress every single struct. DimInfo struct is *huge*, that is probably the biggest target

const Tensor = @import("./tensor.zig").Tensor;

const assert = std.debug.assert;

const Program = @import("./compiler/program.zig").Program;

const ClDevice = @import("./runtimes/cl.zig").ClDevice;
const ClContext = @import("./runtimes/cl.zig").ClContext;
const ClCommandQueue = @import("./runtimes/cl.zig").ClCommandQueue;

const Neuralnet = @import("./nn.zig").Neuralnet;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.detectLeaks();
    const allocator = gpa.allocator();
    const device: ClDevice = try ClDevice.alloc(.gpu);
    const context: ClContext = try ClContext.alloc(device);
    const queue: ClCommandQueue = try ClCommandQueue.alloc(device, context);

    var tensor: Tensor = try Tensor.alloc(allocator, 1, 2, 2, 2, context);
    defer tensor.free(allocator);
    try tensor.linearized.capacityEnsure(allocator, 1);
    tensor.unaryRandom();
    tensor.realize();
    try tensor.buffer.syncToDevice(queue);
    try tensor.buffer.syncWait(queue);

    var nn: Neuralnet = try Neuralnet.alloc(
        allocator,
        tensor,
        &[_]Neuralnet.Layer.Config{
            .{ .dense = .{ .size_out = 4, .activation = .none } },
            // .{ .convolution = .{ .filters = 2, .kernel_size = 4, .kernel_padding = 1, .kernel_stride = 2, .activation = .none } },
            // .{ .filter = .{ .kernel_size = 4, .kernel_padding = 1, .kernel_stride = 2, .activation = .none } },
            // .{ .split = .{ .filters = 2, .activation = .none } },
        },
        20,
        4,
        .O1,
        context,
        device,
        queue,
    );
    defer nn.free(allocator) catch {};
    nn.init();
    try nn.forward(.gpu);
    nn.layers[nn.layers.len - 1].values.print(4, 0, null);
}
