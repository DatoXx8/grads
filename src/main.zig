const std = @import("std");

// TODO: Make optimizer both the standard way and the one casey described that's like perpetually running on a seperate thread
// TODO: Maybe give a list of tensor names to the compiler that do have to hold the right values
//  and should not be inlined away instead of the other way around.
//  I like this solution a lot because it also makes the compiler simulator more interesting when randomizing the "relevant" tensors.
// TODO: Don't use usize where possible to get rid of platform dependant code
// TODO: Implement weightgen and that arnold net thing where there are cubic functions as connections
// TODO: Actual error handling where it is possible
// TODO: Add autograd
// TODO: Really need to compress every single struct. DimInfo struct is *huge*, that is probably the biggest target
// TODO: Analyse /usr/lib/libnvidia-opencl.so

const assert = std.debug.assert;

const Tensor = @import("./tensor.zig").Tensor;

const Program = @import("./compiler/program.zig").Program;

const ClDevice = @import("./runtimes/cl.zig").ClDevice;
const ClContext = @import("./runtimes/cl.zig").ClContext;
const ClCommandQueue = @import("./runtimes/cl.zig").ClCommandQueue;

const Neuralnet = @import("./nn.zig").Neuralnet;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.detectLeaks();
    const allocator = gpa.allocator();

    // TODO: Should probably free these explicitly huh
    const device: ClDevice = try ClDevice.alloc(.gpu);
    const context: ClContext = try ClContext.alloc(device);
    const queue: ClCommandQueue = try ClCommandQueue.alloc(device, context);

    var tensor: Tensor = try Tensor.alloc(allocator, 1, 2, 2, 2, context);
    defer tensor.free(allocator);
    try tensor.linearized.capacityEnsure(allocator, 1);
    tensor.unaryRandom(0);
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
    nn.init(0);
    try nn.forward(.gpu);
    nn.layers[nn.layers.len - 1].values.print(4, 0, null);
}
