// Copyright (c) 2026 Michael Pollind
// SPDX-License-Identifier: MIT
//
// Wrapper package for the *system* FreeType. Unlike deps/nanosvg, nothing is fetched or vendored:
// translate-c reads the installed <freetype/*.h> headers (link_libc defaults on, so it also finds
// <string.h> etc.), and the FT_* symbols are resolved from the system libfreetype at link time.

const std = @import("std");

/// The standard location of the FreeType headers on a Debian/Ubuntu-style system. A more portable
/// build would discover this via pkg-config; hardcoded here since the backend is opt-in and Linux.
const freetype_include = "/usr/include/freetype2";

/// Returns the `freetype` module: Zig-facing declarations (`freetype.c`), linked against system
/// libfreetype. The backend module imports this and never leaks libc into the slughorn core.
pub fn binding(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("ft.h"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    translate_c.addIncludePath(.{ .cwd_relative = freetype_include });

    const module = b.addModule("freetype", .{
        .root_source_file = b.path("main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    module.addImport("c", translate_c.createModule());
    module.linkSystemLibrary("freetype", .{});
    return module;
}

/// Nothing to build standalone -- consumers call `binding`. (Same rationale as deps/nanosvg: a
/// `build()` that declared the module would run during dependency resolution.)
pub fn build(b: *std.Build) void {
    _ = b;
}
