//! Virtual GPU simulator, more or less.
//! Used to estimate the cost of a PIR on the current hardware since measuring could be quite slow.

const std = @import("std");

const Pir = @import("Pir.zig");

const Detail = enum(u8) {
    simple,
    medium,
    complex,
};
const Features = struct {
    simd_width: u32,
    cache_l1_kb: u32, // Per core
    cache_l2_kb: u32, // Shared
};

pub fn VGpu(detail: Detail, feature: Features) type {
    return struct {
        detail: Detail,
        features: Features,

        pub fn cost(pir: Pir) u64 {
            //
        }
    };
}
