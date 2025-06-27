pub const Optimization = @import("./compiler/optimize.zig").Optimization;
pub const Program = @import("./compiler/Program.zig");
pub const Neuralnet = @import("./Neuralnet.zig");
const cl = @import("./runtimes/cl.zig");
pub const ClContext = cl.ClContext;
pub const ClDevice = cl.ClDevice;
pub const ClCommandQueue = cl.ClCommandQueue;
pub const ClError = cl.ClError;
pub const Tensor = @import("./Tensor.zig");
pub const Buffer = Tensor.Buffer;
pub const Op = Tensor.Op;
pub const Linearized = Tensor.Linearized;

