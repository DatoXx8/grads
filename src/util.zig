const std = @import("std");
const assert = std.debug.assert;

/// Does nothing for now. Just explicitly documents that some condition can be true or false
pub fn maybe(ok: bool) void {
    assert(ok or !ok);
}

/// Just to make intent clearer instead of saying unreachable or assert(false)
pub fn todo() void {
    assert(false);
}
