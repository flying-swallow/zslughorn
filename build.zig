// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

const std = @import("std");

/// Number of indirection-table slots per axis.
///
/// This is an ABI constant of the packed band-texture format, not a tuning knob: the CPU bakes it
/// into the per-shape block layout and the fragment shader indexes the table with it. The upstream
/// C++ carries a TODO (slughorn.hpp:799) admitting nothing enforces the two stay in sync. Here this
/// is the single definition site -- it is exported to Zig via `build_options` and (at M3) passed to
/// slangc as -DSLUG_INDIRECTION_SIZE, so drift becomes a compile error instead of silent garbage.
const indirection_size: u32 = 32;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const options = b.addOptions();
    options.addOption(u32, "indirection_size", indirection_size);

    const zml = b.dependency("zml", .{ .target = target, .optimize = optimize }).module("zml");

    const mod = b.addModule("slughorn", .{
        .root_source_file = b.path("src/slughorn.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.addOptions("build_options", options);
    mod.addImport("zml", zml);

    const lib = b.addLibrary(.{
        .name = "slughorn",
        .linkage = .static,
        .root_module = mod,
    });
    b.installArtifact(lib);

    // -- tests -----------------------------------------------------------------------------------

    const test_mod = b.createModule(.{
        .root_source_file = b.path("test/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "slughorn", .module = mod }},
    });
    test_mod.addOptions("build_options", options);

    const tests = b.addTest(.{ .root_module = test_mod });
    const run_tests = b.addRunArtifact(tests);
    // Fixtures are read at runtime from the source tree, not embedded.
    run_tests.setCwd(b.path("."));

    const unit_tests = b.addTest(.{ .root_module = mod });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_tests.step);

    // -- fixtures --------------------------------------------------------------------------------
    //
    // Deliberately NOT wired into the default or `test` step: fixtures are checked in, so neither
    // building nor testing zslughorn requires a C++ toolchain or a copy of the upstream tree. This
    // step exists to (re)generate them, and `fixtures-verify` to catch drift.

    const slughorn_src = b.option(
        []const u8,
        "slughorn-src",
        "Path to the upstream C++ slughorn checkout (for the `fixtures` step)",
    ) orelse "../slughorn";

    // Band boundaries are computed in float (slughorn.cpp:1029/1036/1080 are literally
    // `minY + snapped * rangeY`) and then feed *comparisons* (`maxY >= lo`, slughorn.cpp:165).
    // Contracting that into an FMA changes its f32 result for ~24% of inputs (measured with a
    // random probe), and a 1-ULP shift in a boundary can flip band membership for a curve whose
    // extremum sits in the gap -- changing the *integer* output bytes. GCC contracts by default;
    // Zig's float mode is .strict and does not. So pin the C++ to match Zig, and keep the baseline
    // ISA (no FMA to contract into) as a second line of defence.
    const fp_flags = [_][]const u8{ "-ffp-contract=off", "-fno-fast-math", "-march=x86-64" };

    const dump_bin = addDumper(b, slughorn_src, &fp_flags);

    const gen = std.Build.Step.Run.create(b, "generate golden fixtures");
    gen.addFileArg(dump_bin);
    // The dumper writes straight into the checked-in fixtures directory; regenerating them is an
    // explicit, reviewable act (`zig build fixtures` then `git diff`).
    gen.addDirectoryArg(b.path("fixtures"));
    gen.has_side_effects = true;

    const fixtures_step = b.step("fixtures", "Regenerate golden fixtures from the C++ (needs g++)");
    fixtures_step.dependOn(&gen.step);

    // -- fixtures-verify -------------------------------------------------------------------------
    //
    // Regenerates into a temp directory and diffs against what is checked in, catching fixtures
    // that have gone stale (or been hand-edited) relative to the upstream C++.
    //
    // Note what this deliberately does NOT assert: that the output is independent of float
    // contraction. It is not, and that is measured rather than assumed -- building the dumper with
    // `-ffp-contract=fast -march=native` changes `decompose_cubic_fine`, because CurveDecomposer's
    // flatness test (`dx*dx + dy*dy` and the cross product in _pointToLineDistSq, slughorn.hpp:1606)
    // contracts into FMAs, which changes which cubics count as flat and therefore how they
    // subdivide. That is precisely why the dumper pins `-ffp-contract=off` to match Zig's `.strict`
    // float mode. An earlier version of this step diffed a contracted build against a
    // non-contracted one and required them to be equal -- a gate that could never pass.
    const gen_fresh = std.Build.Step.Run.create(b, "dump fixtures (fresh, for verification)");
    gen_fresh.addFileArg(dump_bin);
    const fresh_dir = gen_fresh.addOutputDirectoryArg("fresh");

    const diff = b.addSystemCommand(&.{ "diff", "-r", "-q" });
    diff.addDirectoryArg(b.path("fixtures"));
    diff.addDirectoryArg(fresh_dir);
    diff.expectExitCode(0);

    const verify_step = b.step(
        "fixtures-verify",
        "Check the checked-in fixtures still match the upstream C++ (needs g++)",
    );
    verify_step.dependOn(&diff.step);
}

/// Compiles the golden-fixture dumper against the upstream C++ with the given float flags.
fn addDumper(b: *std.Build, slughorn_src: []const u8, flags: []const []const u8) std.Build.LazyPath {
    const cc = b.addSystemCommand(&.{ "g++", "-std=c++20", "-O2" });
    cc.addArgs(flags);
    cc.addArg("-o");
    const bin = cc.addOutputFileArg("slughorn-dump");
    cc.addFileArg(b.path("tools/dump/slughorn_dump.cpp"));
    cc.addArg(b.pathJoin(&.{ slughorn_src, "slughorn/slughorn.cpp" }));
    cc.addArg(b.fmt("-I{s}", .{slughorn_src}));
    return bin;
}
