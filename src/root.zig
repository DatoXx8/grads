pub const Optimization = @import("./compiler/optimize.zig").Optimization;
pub const Program = @import("./compiler/program.zig").Program;
pub const Neuralnet = @import("./nn.zig").Neuralnet;
pub const ClContext = @import("./runtimes/cl.zig").ClContext;
pub const ClDevice = @import("./runtimes/cl.zig").ClDevice;
pub const ClCommandQueue = @import("./runtimes/cl.zig").ClCommandQueue;
pub const ClError = @import("./runtimes/cl.zig").ClError;
pub const Tensor = @import("./tensor.zig").Tensor;
pub const Buffer = @import("./tensor.zig").Buffer;
pub const Op = @import("./tensor.zig").Op;
pub const Linearized = @import("./tensor.zig").Linearized;

