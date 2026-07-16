// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! Integration tests: everything that needs the golden fixtures lives here.
//!
//! Pure unit tests live beside the code they test in `src/` and run via the same `zig build test`.

const std = @import("std");

pub const fixture = @import("fixture.zig");
pub const golden_decompose = @import("golden_decompose.zig");
pub const golden_atlas = @import("golden_atlas.zig");
pub const golden_render = @import("golden_render.zig");
pub const golden_multi = @import("golden_multi.zig");

test {
    std.testing.refAllDecls(@This());
}
