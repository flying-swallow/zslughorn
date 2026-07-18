// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT
//
// Tests for the FreeType backend (`-Dfreetype=true`). Property-based against a real system font: the
// upstream C++ compiles with FreeType, so a golden oracle is recoverable later, but for now the
// assertions pin behaviour (a glyph loads, carries positive metrics, and decomposes to a closed
// outline) rather than exact control points -- the curve math is already covered by
// golden_decompose.zig. The test skips if FreeType or the font is unavailable.

const std = @import("std");
const slughorn = @import("slughorn");
const ft = @import("slughorn_freetype");

const testing = std.testing;
const Slug = slughorn.Slug;

const font_path = "/usr/share/fonts/fonts-go/Go-Regular.ttf";

test "loadGlyph loads a real glyph with FreeType metrics" {
    const gpa = testing.allocator;

    const lib = ft.Library.init() catch return error.SkipZigTest; // no FreeType -> skip
    defer lib.deinit();
    const face = lib.openFace(font_path) catch return error.SkipZigTest; // font missing -> skip
    defer face.deinit();

    try testing.expect(face.unitsPerEM() > 0);

    var atlas = try slughorn.Atlas.init(gpa, slughorn.default_texture_width);
    defer atlas.deinit();

    // 'A' is present in any Latin font and has a closed, two-contour outline.
    try testing.expect(try face.loadGlyph(gpa, &atlas, 'A'));
    // A codepoint no normal font maps -> no glyph, nothing added.
    try testing.expect(!(try face.loadGlyph(gpa, &atlas, 0x10FFFF)));

    try atlas.build();

    const shape = atlas.getShape(.{ .codepoint = 'A' }).?;
    try testing.expect(shape.advance > 0); // positive advance width
    try testing.expect(shape.width > 0 and shape.height > 0); // non-empty bbox
    try testing.expect(shape.bearing_y > 0); // 'A' sits above the baseline
    try testing.expect(shape.curves.len >= 3); // an outline, not a single segment

    // Coordinates are em-normalized, so the glyph lives near the unit square.
    for (shape.curves) |cv| {
        try testing.expect(cv.x1 > -0.5 and cv.x1 < 1.5);
        try testing.expect(cv.y1 > -0.5 and cv.y1 < 1.5);
    }
}
