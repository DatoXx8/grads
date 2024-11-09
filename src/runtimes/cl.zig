const builtin = @import("builtin");
const std = @import("std");
// const opencl_target_version = @import("opencl_config").opencl_version;
// const opencl_target_version = @import("opencl_config").opencl_version;

const opencl_header_file = switch (builtin.target.os.tag) {
    .macos => "OpenCL/cl.h",
    else => "CL/cl.h",
};

const opencl = @cImport({
    // @cDefine("CL_TARGET_OPENCL_VERSION", opencl_target_version);
    @cInclude(opencl_header_file);
});
