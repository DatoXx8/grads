const std = @import("std");
const assert = std.debug.assert;

const Program = @import("./compiler/program.zig").Program;
const Neuralnet = @import("./nn.zig").Neuralnet;
const ClDevice = @import("./runtimes/cl.zig").ClDevice;
const ClContext = @import("./runtimes/cl.zig").ClContext;
const ClCommandQueue = @import("./runtimes/cl.zig").ClCommandQueue;
const Tensor = @import("./tensor.zig").Tensor;

// $FIXME Compiler simulator rng=1743713666904694 opt=O1
// $TODO Make 64 bit version flag in the files from Neuralnet.saveToFile()
// $TODO Make debug flag for compile step that adds debug printing if enabled
// $TODO Make optimizer both the standard way and the one casey described that's like perpetually running on a seperate thread, or make optimize.step()
// $TODO Implement weightgen and that arnold net thing where there are cubic functions as connections
// $TODO Add autograd
// $TODO Add automatic quantization
// $TODO Really need to compress every single struct. DimInfo struct is *huge*, that is probably the biggest target
// $TODO Analyse /usr/lib/libnvidia-opencl.so and /usr/lib/libcuda.so

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.detectLeaks();
    const allocator = gpa.allocator();

    // $TODO Should probably free these explicitly huh
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
        context,
        device,
        queue,
    );
    errdefer nn.free(allocator);
    defer nn.free(allocator);
    nn.init(0);
    try nn.forward(.gpu);
    nn.layers[nn.layers.len - 1].values.print(4, 0, null);
}
