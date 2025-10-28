const std = @import("std");

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
    const mod = b.createModule(.{
        .root_source_file = b.path(root_file),
        .target = target,
        .optimize = optimize,
    });
    const exe = b.addExecutable(.{
        .name = name,
        .root_module = mod,
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
        .target = target,
        .optimize = optimize,
    });

    addExe(b, target, optimize, grads, "test", "src/main.zig", "run", //
        "Only used as a temporary main file. Don't use.");
    addExe(b, target, optimize, grads, "unit_ops", "tests/unit_ops.zig", "unit_ops", //
        "Run unit tests for singular ops.");
    addExe(b, target, optimize, grads, "simulate_linearized", "tests/simulate_linearized.zig", "simulate_linearized", //
        "Run simulation tests for linearized ops vs singular ops.");
    addExe(b, target, optimize, grads, "simulate_compiler", "tests/simulate_compiler.zig", "simulate_compiler", //
        "Run simulation tests for compiler vs linearized ops.");
    addExe(b, target, optimize, grads, "profile_compiler", "tests/profile_compiler.zig", "profile_compiler", //
        "Run the simulator for the compiler optimisations and overall speed.");

    const regression_compiler_mod = b.createModule(.{
        .root_source_file = b.path("tests/regression_compiler.zig"),
        .target = target,
        .optimize = optimize,
    });
    const regression_compiler = b.addTest(.{
        .root_module = regression_compiler_mod,
    });
    regression_compiler.addIncludePath(.{
        .cwd_relative = "/usr/include/",
    });
    regression_compiler.linkLibC();
    regression_compiler.linkSystemLibrary("OpenCL");
    regression_compiler.root_module.addImport("grads", grads);
    b.installArtifact(regression_compiler);
    const regression_compiler_run = b.addRunArtifact(regression_compiler);
    regression_compiler_run.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        regression_compiler_run.addArgs(args);
    }
    const regression_compiler_run_step = b.step("regression_compiler", "Run the regression tests from previous simulator failures");
    regression_compiler_run_step.dependOn(&regression_compiler_run.step);
}
