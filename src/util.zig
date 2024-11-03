// const AssertionErrorType = error{
//     assertion_failure,
// };
// TODO: Make Error type that describes the error
// pub fn assert(ok: bool) !void {
//     if (!ok) {
//         return AssertionErrorType.assertion_failure;
//     }
// }
pub fn assert(ok: bool) void {
    if (!ok) {
        unreachable;
    }
}
pub fn maybe(ok: bool) void {
    assert(ok or !ok);
}
