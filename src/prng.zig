const math = @import("std").math;

var state: u64 = 0;
const mult: u64 = 6364136223846793005;
const incr: u64 = 1442695040888963407;

/// The method used in rand_f32 always generates 2 random numbers, and it is kinda pointles to just get rid of the other one.
/// I measured the performance in ReleaseSafe and generated 2^32 floats and the times were 54s for the spare and 1:35m for no spare.
/// IMO that is a big enough margin to say that this is definitely faster. It's also less wasteful of randomness if you catch my drift.
var spare_exists: bool = false;
var spare: f32 = 0;
/// The likelyhood of this failing despite trying this many times is ((4 - pi)/4) ^ tries_max, which for tries_max = 40 is 1e-27,
/// which means it will likely only happen at billions of exabytes (= trillions of petabytes). And I seriously doubt anyone will generate *that* many floats with this.
/// In case it fails anyways it returns 0 because that is the least likely to cause damage as that is the mean anyways.
const tries_max = 40;

/// This implementation was tested using PractRand [https://www.pcg-random.org/posts/how-to-test-with-practrand.html] up to 1 TB and it found no statistical anomalies.
pub const Pcg = struct {
    fn rotate_32(x: u32, pivot: u5) u32 {
        return x >> pivot | x << ((-%pivot) & 31);
    }
    pub fn init(x: u64) void {
        state = x;
        // To make sure that re-initing guarantees the same results, as long as you don't use multi-threading ^^
        spare_exists = false;
        spare = 0;
    }
    pub fn rand() u32 {
        var x: u64 = state;
        const pivot: u5 = @truncate(x >> 59);

        state = state *% mult +% incr;
        x ^= x >> 18;
        return Pcg.rotate_32(@truncate(x >> 27), pivot);
    }
    pub fn rand_below(top: u32) u32 {
        if (top == 0 or top == 1) {
            return 0;
        }
        var x: u32 = Pcg.rand();
        var m: u64 = @as(u64, x) *% @as(u64, top);
        var l: u32 = @truncate(m);
        if (l < top) {
            var t: u32 = -%top;
            if (t > top) {
                t -= top;
                if (t >= top) {
                    t %= top;
                }
            }
            while (l < t) {
                x = Pcg.rand();
                m = @as(u64, x) *% @as(u64, top);
                l = @truncate(m);
            }
        }
        return @truncate(m >> 32);
    }
    /// Zig implementation of the Marsaglia polar method from https://en.wikipedia.org/wiki/Marsaglia_polar_method#C++
    /// TODO: Make this thread-safe, actually not trivial if I don't want to abandon the spare value
    pub fn rand_f32() f32 {
        if (spare_exists) {
            spare_exists = false;
            return spare;
        } else {
            var u: f32 = 0;
            var v: f32 = 0;
            var s: f32 = 0;
            for (0..tries_max) |_| {
                u = (@as(f32, @floatFromInt(Pcg.rand())) / @as(f32, @floatFromInt(math.maxInt(u32)))) * 2.0 - 1.0;
                v = (@as(f32, @floatFromInt(Pcg.rand())) / @as(f32, @floatFromInt(math.maxInt(u32)))) * 2.0 - 1.0;
                s = u * u + v * v;
                if (s > 0 and s < 1) {
                    break;
                }
            }
            if (s <= 0 or s >= 1) {
                return 0;
            }
            s = math.sqrt(-2.0 * math.log(f32, math.e, s) / s);
            spare = v * s;
            spare_exists = true;
            return u * s;
        }
    }
};
