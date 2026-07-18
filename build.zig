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

    // -- nanosvg backend -------------------------------------------------------------------------
    //
    // Off by default, mirroring upstream's `option(SLUGHORN_NANOSVG ... OFF)` (CMakeLists.txt:45).
    // The dependency is `.lazy`, so with the option off nanosvg is never even fetched and a plain
    // `zig build` stays pure Zig.
    //
    // This is a *separate module* that imports `slughorn`, never the reverse. nanosvg is C and
    // needs libc; keeping it behind its own module boundary is what stops that reaching the core --
    // the same seam the M3 renderer will use to keep GPL/rhi-zig out of the MIT core.

    const enable_nanosvg = b.option(bool, "nanosvg", "Build the NanoSVG backend module") orelse false;

    if (enable_nanosvg) {
        // Each `orelse return` here is the lazy-fetch protocol, not an error path: it abandons this
        // configure pass so the runner can fetch nanosvg and re-run, after which they all resolve.
        // The binding comes from a function rather than deps/nanosvg's `build()` precisely so that
        // last null can reach this scope -- see the comment on `binding` there.
        const nanosvg_pkg = b.lazyImport(@This(), "nanosvg") orelse return;
        const nanosvg_dep = b.lazyDependency("nanosvg", .{}) orelse return;
        const nanosvg = nanosvg_pkg.binding(nanosvg_dep.builder, target, optimize) orelse return;

        const svg_mod = b.addModule("slughorn_nanosvg", .{
            .root_source_file = b.path("src/backends/nanosvg.zig"),
            .target = target,
            .optimize = optimize,
        });
        svg_mod.addImport("slughorn", mod);
        svg_mod.addImport("nanosvg", nanosvg.module);
        svg_mod.linkLibrary(nanosvg.lib);

        // A separate test binary rather than a fifth suite in test/root.zig: that file uses
        // refAllDecls, which would force analysis of the backend import even in builds where the
        // dependency was never fetched.
        const svg_test_mod = b.createModule(.{
            .root_source_file = b.path("test/nanosvg.zig"),
            .target = target,
            .optimize = optimize,
        });
        svg_test_mod.addImport("slughorn", mod);
        svg_test_mod.addImport("slughorn_nanosvg", svg_mod);

        const svg_tests = b.addTest(.{ .root_module = svg_test_mod });
        test_step.dependOn(&b.addRunArtifact(svg_tests).step);
    }

    // -- SDF/MSDF backend ------------------------------------------------------------------------
    //
    // Off by default. Turns a built shape's curves into SDF/MSDF tiles -- the `renderSDF` family
    // from upstream's render.hpp, which upstream delegates to Chlumsky's C++ msdfgen. Here the
    // generator is the user's msdf-zig (a pure-Zig msdfgen port). We take its FreeType-free
    // `msdf-core` module by passing `.font = false`, so the curves -> distance-field path pulls in
    // no FreeType (and dodges mach-freetype's `@cImport`, which the current Zig nightly rejects).
    //
    // A separate module importing `slughorn`, never the reverse -- MSDF stays out of the MIT core.

    const enable_msdf = b.option(bool, "msdf", "Build the SDF/MSDF backend module (needs ../msdf-zig)") orelse false;

    if (enable_msdf) {
        const msdf_dep = b.lazyDependency("msdf_zig", .{
            .target = target,
            .optimize = optimize,
            .font = false,
        }) orelse return;

        const sdf_mod = b.addModule("slughorn_sdf", .{
            .root_source_file = b.path("src/backends/sdf.zig"),
            .target = target,
            .optimize = optimize,
        });
        sdf_mod.addImport("slughorn", mod);
        sdf_mod.addImport("msdf", msdf_dep.module("msdf-core"));

        const sdf_test_mod = b.createModule(.{
            .root_source_file = b.path("test/sdf.zig"),
            .target = target,
            .optimize = optimize,
        });
        sdf_test_mod.addImport("slughorn", mod);
        sdf_test_mod.addImport("slughorn_sdf", sdf_mod);

        const sdf_tests = b.addTest(.{ .root_module = sdf_test_mod });
        test_step.dependOn(&b.addRunArtifact(sdf_tests).step);
    }

    // -- FreeType backend ------------------------------------------------------------------------
    //
    // Off by default. Loads real font glyphs into the atlas via the *system* libfreetype (translate-c
    // wrapper in deps/freetype). A separate module importing `slughorn`, never the reverse -- libc and
    // FreeType stay out of the MIT core, mirroring the nanosvg seam.

    const enable_freetype = b.option(bool, "freetype", "Build the FreeType backend module (needs system libfreetype)") orelse false;

    if (enable_freetype) {
        const ft_pkg = b.lazyImport(@This(), "freetype") orelse return;
        const ft_dep = b.lazyDependency("freetype", .{}) orelse return;
        const ft_mod = ft_pkg.binding(ft_dep.builder, target, optimize);

        const freetype_mod = b.addModule("slughorn_freetype", .{
            .root_source_file = b.path("src/backends/freetype.zig"),
            .target = target,
            .optimize = optimize,
        });
        freetype_mod.addImport("slughorn", mod);
        freetype_mod.addImport("freetype", ft_mod);

        const freetype_test_mod = b.createModule(.{
            .root_source_file = b.path("test/freetype.zig"),
            .target = target,
            .optimize = optimize,
        });
        freetype_test_mod.addImport("slughorn", mod);
        freetype_test_mod.addImport("slughorn_freetype", freetype_mod);

        const freetype_tests = b.addTest(.{ .root_module = freetype_test_mod });
        test_step.dependOn(&b.addRunArtifact(freetype_tests).step);
    }

    // -- Canvas backend --------------------------------------------------------------------------
    //
    // Off by default. A procedural, Path2D-style drawing API that builds a `CompositeShape`. Pure Zig,
    // no external deps -- gated for parity with the other backends and to keep the default suite lean.

    const enable_canvas = b.option(bool, "canvas", "Build the Canvas backend module") orelse false;

    if (enable_canvas) {
        const canvas_mod = b.addModule("slughorn_canvas", .{
            .root_source_file = b.path("src/backends/canvas.zig"),
            .target = target,
            .optimize = optimize,
        });
        canvas_mod.addImport("slughorn", mod);

        const canvas_test_mod = b.createModule(.{
            .root_source_file = b.path("test/canvas.zig"),
            .target = target,
            .optimize = optimize,
        });
        canvas_test_mod.addImport("slughorn", mod);
        canvas_test_mod.addImport("slughorn_canvas", canvas_mod);

        const canvas_tests = b.addTest(.{ .root_module = canvas_test_mod });
        test_step.dependOn(&b.addRunArtifact(canvas_tests).step);
    }

    // -- GPU renderer backend (M3) ---------------------------------------------------------------
    //
    // Off by default. The Vulkan renderer + Slug shader that draws the compiled atlas. It links
    // rhi-zig (GPL-2.0), so it is a *separate module* that imports `slughorn`, never the reverse --
    // the same seam nanosvg/msdf use, here keeping GPL out of the MIT core. The dependency is lazy,
    // so a default `zig build` never fetches rhi and stays MIT/pure-Zig. Needs Vulkan at test time.

    const enable_renderer = b.option(bool, "renderer", "Build the GPU renderer backend module (needs ../rhi-zig, Vulkan)") orelse false;

    if (enable_renderer) {
        const rhi_dep = b.lazyDependency("rhi", .{ .target = target, .optimize = optimize }) orelse return;
        const rhi_mod = rhi_dep.module("rhi");

        const renderer_mod = b.addModule("slughorn_renderer", .{
            .root_source_file = b.path("src/gpu/renderer.zig"),
            .target = target,
            .optimize = optimize,
        });
        renderer_mod.addImport("slughorn", mod);
        renderer_mod.addImport("rhi", rhi_mod);

        // Compile the Slug shaders (GLSL, mirroring the reference GLSL) to SPIR-V with the system
        // glslangValidator and embed them into the module. `-DSLUG_INDIRECTION_SIZE` is sourced
        // from the same `indirection_size` build constant the packer and render.zig use, so the
        // shader's band indexing can never silently drift from the CPU's.
        const vert_spv = compileGlslSpv(b, "shaders/slug.vert", "slug.vert.spv", indirection_size);
        const frag_spv = compileGlslSpv(b, "shaders/slug.frag", "slug.frag.spv", indirection_size);
        renderer_mod.addAnonymousImport("slug_vert_spv", .{ .root_source_file = vert_spv });
        renderer_mod.addAnonymousImport("slug_frag_spv", .{ .root_source_file = frag_spv });

        const renderer_test_mod = b.createModule(.{
            .root_source_file = b.path("test/renderer.zig"),
            .target = target,
            .optimize = optimize,
        });
        renderer_test_mod.addImport("slughorn", mod);
        renderer_test_mod.addImport("slughorn_renderer", renderer_mod);

        const renderer_tests = b.addTest(.{ .root_module = renderer_test_mod });
        test_step.dependOn(&b.addRunArtifact(renderer_tests).step);
    }

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

/// Compiles a GLSL shader to SPIR-V with the system `glslangValidator`, returning a LazyPath to the
/// emitted module for embedding via `addAnonymousImport` + `@embedFile`. The stage is inferred from
/// the source extension (`.vert`/`.frag`). `TEX_WIDTH` is baked for the default 512-wide atlas
/// (log2(512) = 9); a general renderer would pass it as a spec constant.
fn compileGlslSpv(b: *std.Build, src: []const u8, out_name: []const u8, indir: u32) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ "glslangValidator", "-V", "--target-env", "vulkan1.3" });
    run.addArg(b.fmt("-DSLUG_INDIRECTION_SIZE={d}", .{indir}));
    run.addArg("-DTEX_WIDTH=9");
    run.addArg("-o");
    const out = run.addOutputFileArg(out_name);
    run.addFileArg(b.path(src));
    return out;
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
