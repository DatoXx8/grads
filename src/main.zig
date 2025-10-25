const std = @import("std");
const assert = std.debug.assert;

const Program = @import("compiler/Program.zig");
const Layer = @import("Layer.zig");
const Neuralnet = @import("Neuralnet.zig");
const Runtime = @import("compiler/runtimes/Runtime.zig");
const RuntimeCl = Runtime.RuntimeCl;
const Buffer = @import("Buffer.zig");
const Linearized = @import("Linearized.zig");

// $TODO Get rid of all print methods and just make format functions, this allows saner logging for test failures
// $TODO Try making every kernel it's own source so that compilation is faster in the OpenCl implementation
// $TODO Rework the memory management in PIRs to be more arena style. Maybe make it like the dedit free list style
// $TODO Expose an ArenaAllocator like interface for the Runtimes. Having to call the individual free function just because of those makes no sense
// $TODO Refactor all of the assignments to not be optionals. Just have default values that are equivalent to no optimization.
// $TODO Randomly pertubate the random linearized ops (Change sizes, offsets, op types, unary values etc.)
// $TODO Log test fail seeds to regtest file with textify_linarized
// $TODO Make unit tests for Neuralnets (forward, backward, learn verifiably with learn cycles putting loss to 0)
// $TODO Make debug flag for compile step that adds debug printing if enabled

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
            .{ .dense = .{ .size_out = 4, .activation_kind = .none } },
            // .{ .convolution = .{ .filters = 2, .kernel_size = 2, .kernel_padding = 1, .kernel_stride = 2, .activation_kind = .none } },
            // .{ .split = .{ .filters = 2, .activation_kind = .none } },
            // .{ .reduce = .{ .t = .sum, .kernel_size = 2, .kernel_stride = 1 } },
        },
        20,
        4,
    );
    defer nn.free();
    try nn.init(0);
    try nn.sync(true, true, true, true, true, .sync_to_device);
    try nn.forward();
    try nn.sync(true, true, true, true, true, .sync_to_host);
    nn.layer[nn.layer.len - 1].values.print(4, 0, null);
}
