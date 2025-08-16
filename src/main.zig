const std = @import("std");
const assert = std.debug.assert;

const Program = @import("compiler/Program.zig");
const Layer = @import("Layer.zig");
const Neuralnet = @import("Neuralnet.zig");
const Runtime = @import("compiler/runtimes/Runtime.zig");
const RuntimeCl = Runtime.RuntimeCl;
const Tensor = @import("Tensor.zig");

// $TODO Make casey style optimizer as an alternative with genetic algorithm to change optimization steps saved in array (When making this add reorder optimization for potentially better loop behavior)
// $TODO Refactor all of the assignments to not be optionals. Just have default values that are equivalent to no optimization.
// $TODO Try out this ZII thing
// $TODO Randomly pertubate the random linearized ops (Change sizes, offsets, op types, unary values etc.)
// $TODO Log test fail seeds to file, this requires not changing the random generation scheme
// $TODO Make a way to have a tensor put it's ops in another tensors linearized, maybe call it like external linearized
// $TODO Make unit tests for Neuralnets (forward, backward, learn verifiably with learn cycles putting loss to 0)
// $TODO Make debug flag for compile step that adds debug printing if enabled
// $TODO Add autograd
// $TODO Add automatic quantization
// $TODO Really need to compress every single struct
// $TODO Implement weightgen and that arnold net thing where there are cubic functions as connections

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.detectLeaks();
    const allocator = gpa.allocator();

    var runtime_cl: RuntimeCl = undefined;
    var runtime: Runtime = runtime_cl.runtime();
    // var runtime_ptx: RuntimePtx = undefined;
    // var runtime: Runtime = runtime_ptx.runtime();
    try runtime.init();
    defer runtime.deinit();

    var nn: Neuralnet = try Neuralnet.alloc(
        runtime,
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
    );
    defer nn.free(allocator);
    try nn.init(0);
    try nn.sync(true, true, true, true, true, .sync_to_device);
    try nn.forward();
    try nn.sync(true, true, true, true, true, .sync_to_host);
    nn.layer[nn.layer.len - 1].values.print(4, 0, null);
}
