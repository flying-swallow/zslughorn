// Copyright (c) 2026 Michael Pollind
// SPDX-License-Identifier: MIT
//
// Wrapper package for nanosvg (zlib, (c) 2013-14 Mikko Mononen). No upstream source is vendored
// here -- the package manager fetches it by URL+hash (see build.zig.zon), so this directory holds
// only build glue and the implementation shim.

const std = @import("std");

/// The two halves a consumer needs: the Zig-facing module, and the static library carrying the
/// expanded C implementation. Importing the module without linking the library builds but fails to
/// link, so callers must take both.
pub const Binding = struct {
    module: *std.Build.Module,
    lib: *std.Build.Step.Compile,
};

/// Wires nanosvg up against the fetched upstream source.
///
/// Returns null when upstream has not been fetched yet -- nothing here can be built before the
/// header exists to translate. Callers MUST propagate that (`orelse return`) rather than ignore it,
/// which is what lets the build runner fetch the source and re-run configure.
pub fn binding(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) ?Binding {
    const upstream = b.lazyDependency("nanosvg", .{}) orelse return null;

    // Declarations. NANOSVG_IMPLEMENTATION is *not* defined here, so translate-c sees prototypes
    // only and never walks the ~2900-line body -- that half is compiled from nanosvg_impl.c below.
    const translate_c = b.addTranslateC(.{
        .root_source_file = upstream.path("src/nanosvg.h"),
        .target = target,
        .optimize = optimize,
    });

    const module = b.addModule("nanosvg", .{ .root_source_file = b.path("main.zig") });
    module.addImport("c", translate_c.createModule());

    // Implementation. nanosvg needs string.h/stdlib.h/stdio.h/math.h, hence libc; keeping that here
    // rather than on the consumer is what stops it leaking into the slughorn core.
    const impl = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    impl.addCSourceFile(.{
        .file = b.path("nanosvg_impl.c"),
        .flags = &.{"-std=c99"},
    });
    impl.addIncludePath(upstream.path("src"));

    const lib = b.addLibrary(.{
        .name = "nanosvg",
        .root_module = impl,
        .linkage = .static,
    });

    return .{ .module = module, .lib = lib };
}

/// Nothing to build standalone -- consumers call `binding` instead.
///
/// The work deliberately does not live here. A `build()` that declared the module would run during
/// dependency *resolution*, forcing the upstream fetch on every build that merely resolves this
/// package -- including `zig build` with the backend switched off, which is exactly what the lazy
/// dependency exists to prevent. Returning the binding from a function instead lets the null
/// propagate out to the root, which is the only place that knows whether nanosvg is wanted at all.
pub fn build(b: *std.Build) void {
    _ = b;
}
