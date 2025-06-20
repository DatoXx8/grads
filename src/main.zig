const std = @import("std");
const assert = std.debug.assert;

const Program = @import("./compiler/program.zig").Program;
const Neuralnet = @import("./nn.zig").Neuralnet;
const Layer = @import("./layer.zig").Layer;
const ClDevice = @import("./runtimes/cl.zig").ClDevice;
const ClContext = @import("./runtimes/cl.zig").ClContext;
const ClCommandQueue = @import("./runtimes/cl.zig").ClCommandQueue;
const Tensor = @import("./tensor.zig").Tensor;

// $TODO Log test fail seeds to file, this requires not changing the random generation scheme
// $TODO Log failing simulator seeds to some test database file
// $TODO Make a way to have a tensor put it's ops in another tensors linearized, maybe call it like external linearized
// $TODO Make unit tests for Neuralnets (forward, backward, learn verifiably with learn cycles putting loss to 0)
// $TODO Factor out all the places in which we create random linearized ops. This also makes it easier to keep consistent prng states across the simulator and profiler
// $TODO Make debug flag for compile step that adds debug printing if enabled
// $TODO Make optimizer both the standard way and the one casey described that's like perpetually running on a seperate thread and make optimizer.step()
// $TODO Implement weightgen and that arnold net thing where there are cubic functions as connections
// $TODO Add autograd
// $TODO Add automatic quantization
// $TODO Really need to compress every single struct. DimInfo struct is *huge*, that is probably the biggest target
// $TODO Analyse /usr/lib/libnvidia-opencl.so and /usr/lib/libcuda.so (/opt/cuda/targets/x86_64-linux/include/cuda.h for PTX)

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.detectLeaks();
    const allocator = gpa.allocator();

    var device: ClDevice = try ClDevice.alloc(.gpu);
    var context: ClContext = try ClContext.alloc(device);
    var queue: ClCommandQueue = try ClCommandQueue.alloc(device, context);
    defer {
        device.free() catch {};
        context.free() catch {};
        queue.free() catch {};
    }

    var nn: Neuralnet = try Neuralnet.alloc(
        allocator,
        2,
        2,
        2,
        &[_]Layer.Config{
            .{ .dense = .{ .size_out = 4, .activation_type = .none } },
            // .{ .convolution = .{ .filters = 2, .kernel_size = 4, .kernel_padding = 1, .kernel_stride = 2, .activation_type = .none } },
            // .{ .split = .{ .filters = 2, .activation_type = .none } },
        },
        20,
        4,
        device,
        context,
        queue,
    );
    errdefer nn.free(allocator);
    defer nn.free(allocator);
    try nn.init(0);
    try nn.sync(true, true, true, true, true, .sync_to_device);
    try nn.forward(.gpu);
    try nn.sync(true, true, true, true, true, .sync_to_host);
    nn.layer[nn.layer.len - 1].values.print(4, 0, null);
}
