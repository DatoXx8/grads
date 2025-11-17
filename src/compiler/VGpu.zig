//! Virtual GPU simulator, more or less.
//! Used to estimate the cost of a PIR on the current hardware since measuring could be quite slow.
//! Actually running the program is of course more accurate
//! Costs are not in any unit and are just meant to be compared to one another

const std = @import("std");

const util = @import("../util.zig");
const Buffer = @import("../Buffer.zig");
const Linearized = @import("../Linearized.zig");
const Op = Linearized.Op;
const Pir = @import("Pir.zig");

pub const Detail = enum(u8) {
    simple,
    medium,
    complex,
};
pub const Features = struct {
    simd_width_bytes: u32,
    cache_l1_kb: u32, // Per core
    cache_l2_kb: u32, // Shared
};

pub const VGpu = @This();

detail: Detail,
features: Features,

/// Data basically made up
fn costOfOpSimple(kind: Op.Kind) u64 {
    return switch (kind) {
        .unary_add, .binary_add, .expand_add => 1,
        .unary_subtract, .binary_subtract, .expand_subtract => 1,
        .unary_multiply, .binary_multiply, .expand_multiply => 1,
        .unary_divide, .binary_divide, .expand_divide => 20,
        .unary_absolute => 2,
        .unary_max, .binary_max, .expand_max => 2,
        .unary_min, .binary_min, .expand_min => 2,
        .unary_random => 20,
        .unary_set, .binary_set, .expand_set => 1,
        .unary_sign => 1,
        .unary_square => 1,
        .unary_sqrt => 20,
        .unary_tanh => 40,
        .unary_exp => 40,
        .unary_log => 40,
        .unary_reciprocal => 10,
        .reduce_avg, .reduce_sum => 1, // avg takes an extra divide, but who cares
        .reduce_max, .reduce_min => 2,
    };
}
pub fn costEstimate(v_gpu: VGpu, pir: Pir, size_global: u32, size_local: u32) u64 {
    _ = size_local;
    switch (v_gpu.detail) {
        .simple => {
            var cost: u64 = 0;
            var assign_idx: u32 = 0;
            while (assign_idx < pir.assign_num) : (assign_idx += 1) {
                const cost_flat: u64 = 1000; // Execution cost for a kernel, completely made up number

                const a_size: u32 = pir.assign[assign_idx].size.a;
                const z_size: u32 = pir.assign[assign_idx].size.z;
                const y_size: u32 = pir.assign[assign_idx].size.y;
                const x_size: u32 = pir.assign[assign_idx].size.x;
                const repeats: u32 = pir.assign[assign_idx].repeats;
                const kernel_assign_ops: u32 = if (pir.assign[assign_idx].split)
                    std.math.divCeil(u32, repeats * a_size * z_size * y_size * x_size, size_global) catch unreachable
                else
                    (std.math.divCeil(u32, repeats, size_global) catch unreachable) * a_size * z_size * y_size * x_size;
                cost += costOfOpSimple(pir.assign[assign_idx].base.kind) * kernel_assign_ops + cost_flat;
                var inlined_idx: u32 = 0;
                while (inlined_idx < pir.assign[assign_idx].inlined.num) : (inlined_idx += 1) {
                    cost += costOfOpSimple(pir.assign[assign_idx].inlined.base[inlined_idx].kind) * kernel_assign_ops;
                }
            }
            return cost;
        },
        .medium => {
            util.todo(@src());
        },
        .complex => {
            util.todo(@src());
        },
    }
}
