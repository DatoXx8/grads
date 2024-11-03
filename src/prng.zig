var state: u64 = 0;
const mult: u64 = 6364136223846793005;
const incr: u64 = 1442695040888963407;

pub const Pcg = struct {
    fn rotate_32(x: u32, pivot: u5) u32 {
        return x >> pivot | x << ((-%pivot) & 31);
    }
    pub fn init(x: u32) void {
        state = x;
    }
    pub fn rand() u32 {
        var x: u64 = state;
        const pivot: u32 = @intCast(x >> 59);

        state = state *% mult +% incr;
        x ^= x >> 18;
        return Pcg.rotate_32(@truncate(x >> 27), @truncate(pivot));
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
};
