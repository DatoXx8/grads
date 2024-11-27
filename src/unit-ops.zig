const std = @import("std");

const Tensor = @import("./tensor.zig").Tensor;

const Pcg = @import("./prng.zig").Pcg;

// TODO: Move tests to seperate directory
// TODO: Make options to run with provided seed

test "unit tests for singular ops" {
    const allocator = std.testing.allocator;

    const a_size: u32 = 4;
    const z_size: u32 = 3;
    const y_size: u32 = 2;
    const x_size: u32 = 1;
    var tensor1 = try Tensor.alloc(allocator, a_size, z_size, y_size, x_size, null);
    defer tensor1.free(allocator);
    var tensor2 = try Tensor.alloc(allocator, a_size, z_size, y_size, x_size, null);
    defer tensor2.free(allocator);
    const val1 = try allocator.alloc(f32, a_size * z_size * y_size * x_size);
    defer allocator.free(val1);
    const val2 = try allocator.alloc(f32, a_size * z_size * y_size * x_size);
    defer allocator.free(val2);

    const rng: u64 = @bitCast(std.time.microTimestamp());
    std.debug.print("rng: {} {s} {s}\n", .{ rng, tensor1.buffer.name, tensor2.buffer.name });
    Pcg.init(rng);

    for (0..a_size * z_size * y_size * x_size) |val_idx| {
        val1[val_idx] = Pcg.rand_f32();
        val2[val_idx] = Pcg.rand_f32();
    }
}
