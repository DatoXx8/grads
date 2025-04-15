const std = @import("std");

// $TODO Support compiling to a static lib

fn addExe(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    grads: *std.Build.Module,
    name: []const u8,
    root_file: []const u8,
    run_name: []const u8,
    run_description: []const u8,
) void {
    const exe = b.addExecutable(.{
        .name = name,
        .root_source_file = b.path(root_file),
        .target = target,
        .optimize = optimize,
    });
    exe.addIncludePath(.{
        .cwd_relative = "/usr/include/",
    });
    exe.linkLibC();
    exe.linkSystemLibrary("OpenCL");
    exe.root_module.addImport("grads", grads);
    b.installArtifact(exe);
    const exe_run = b.addRunArtifact(exe);
    exe_run.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        exe_run.addArgs(args);
    }
    const exe_run_step = b.step(run_name, run_description);
    exe_run_step.dependOn(&exe_run.step);
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const grads = b.addModule("grads", .{
        .root_source_file = b.path("src/root.zig"),
    });

    addExe(b, target, optimize, grads, "test", "src/main.zig", "run", //
        "Only used as a temporary main file. Don't use.");
    addExe(b, target, optimize, grads, "unit-ops", "tests/unit-ops.zig", "unit-ops", //
        "Run unit tests for singular ops.");
    addExe(b, target, optimize, grads, "simulate-linearized", "tests/simulate-linearized.zig", "test-linearized", //
        "Run simulation tests for linearized ops vs singular ops.");
    addExe(b, target, optimize, grads, "simulate-compiler", "tests/simulate-compiler.zig", "test-compiler", //
        "Run simulation tests for compiler vs linearized ops.");
    addExe(b, target, optimize, grads, "profile-compiler", "tests/profile-compiler.zig", "profile-compiler", //
        "Run the simulator for the compiler optimisations and overall speed.");
}
