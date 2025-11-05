const std = @import("std");
const assert = std.debug.assert;

/// Does nothing for now. Just explicitly documents that some condition can be true or false
pub fn maybe(ok: bool) void {
    assert(ok or !ok);
}

/// Just to make intent clearer instead of saying unreachable or assert(false)
pub fn todo(src: std.builtin.SourceLocation) noreturn {
    std.debug.panic("Reached `todo` in {s}:{}:{} in function {s}\n", .{ src.file, src.line, src.column, src.fn_name });
}

/// Only supports single threaded mode
pub const log = struct {
    var enabled: bool = true;
    pub fn enable() void {
        enabled = true;
    }
    pub fn disable() void {
        enabled = false;
    }
    pub fn print(comptime fmt: []const u8, args: anytype) void {
        if (enabled) {
            std.debug.print(fmt, args);
        }
    }
};
