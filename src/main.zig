const std = @import("std");

// TODO: Get rid of the pollution with passing in an allocator everywhere.
// TODO: Make the alloc and free functions unable to error. That is fairly easy with just having everything as an optional, but that is horribly disgusting
// TODO: I think all of the above can be gotten rid of by having a way to explicity interface with the linearized capacity to increase / set it as necessary
//      -> Do something like tensor.capacity_ensure(std.mem.allocator, u32) to ensure there at least that many spots free
//
//  TODO: Implement weightgen and that arnold net thing where there are cubic functions as connections
//
//      FIND THAT FUCKING RACING GAME SONG WITH THE GREEN THUMBNAIL "STARTING THE WINNER" OR SOME SHIT LIKE THAT. THAT SONG WAS A FUCKING BANGER
//
//      Error in old c implementation at 1732074960

const Tensor = @import("./tensor.zig").Tensor;

const assert = std.debug.assert;

const Program = @import("./compiler/program.zig").Program;

const ClDevice = @import("./runtimes/cl.zig").ClDevice;
const ClContext = @import("./runtimes/cl.zig").ClContext;
const ClCommandQueue = @import("./runtimes/cl.zig").ClCommandQueue;

pub fn main() !void {
    // const stdout_file = std.io.getStdOut().writer();
    // var bw = std.io.bufferedWriter(stdout_file);
    // const stdout = bw.writer();
    // defer bw.flush() catch {};

    std.debug.print("Hi :)\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const device: ClDevice = try ClDevice.alloc(.Gpu);
    const context: ClContext = try ClContext.alloc(device);
    const queue: ClCommandQueue = try ClCommandQueue.alloc(device, context);

    var tensor: Tensor = try Tensor.alloc(allocator, 2, 3, 4, 5, context);
    defer tensor.free(allocator);
    try tensor.unaryAdd(allocator, 4);
    try tensor.linearized.print(4, 0, null);

    const program: Program = try Program.alloc(allocator, tensor.linearized, 9, 3, device, context, queue);
    defer program.free(allocator) catch {};

    tensor.realize();
    try tensor.linearized.print(4, 0, null);
    try tensor.print(4, 0, null);
}
