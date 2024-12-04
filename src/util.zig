// const AssertionErrorType = error{
//     assertion_failure,
// };
// TODO: Make Error type that describes the error
// pub fn assert(ok: bool) !void {
//     if (!ok) {
//         return AssertionErrorType.assertion_failure;
//     }
// }
pub inline fn assert(ok: bool) void {
    if (!ok) {
        unreachable;
    }
}
pub inline fn maybe(ok: bool) void {
    assert(ok or !ok);
}
