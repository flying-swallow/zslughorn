// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! Tests for the NanoSVG backend.
//!
//! Unlike every other suite here, these expectations are hand-derived rather than dumped from the
//! upstream C++ -- see DIVERGENCE.md. The C++ SVG backend does not compile against the nanosvg
//! commit it pins, so there is no oracle to diff against.
//!
//! The assertions are chosen to be robust rather than positional: areas and endpoints, not
//! hardcoded control points. A curve-count check would pin down De Casteljau's split behaviour,
//! which is `decompose.zig`'s business and already covered byte-exactly by `golden_decompose.zig`.

const std = @import("std");
const slughorn = @import("slughorn");
const svg = @import("slughorn_nanosvg");

const testing = std.testing;
const Curve = slughorn.Curve;
const Slug = slughorn.Slug;

const dpi: Slug = 96;

/// A right triangle filling the whole 100x100 canvas.
const triangle =
    \\<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
    \\  <path d="M 0 0 L 100 0 L 0 100 Z" fill="black"/>
    \\</svg>
;

/// One shape, two sub-paths: an 80x80 outer square with a 40x40 square inside it. Both are wound
/// the same direction in the source, so only the fill rule distinguishes "hole" from "overlap".
fn ringSvg(comptime fill_rule: []const u8) []const u8 {
    return "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100\" height=\"100\" viewBox=\"0 0 100 100\">" ++
        "<path fill-rule=\"" ++ fill_rule ++ "\" fill=\"black\"" ++
        " d=\"M 10 10 L 90 10 L 90 90 L 10 90 Z M 30 30 L 70 30 L 70 70 L 30 70 Z\"/>" ++
        "</svg>";
}

/// Shoelace area over each curve's chord. Exact for the straight-edged shapes used here, since a
/// line's chord *is* the line.
fn signedArea(curves: []const Curve) f64 {
    var sum: f64 = 0;
    for (curves) |curve| {
        sum += @as(f64, curve.x1) * @as(f64, curve.y3) - @as(f64, curve.x3) * @as(f64, curve.y1);
    }
    return sum * 0.5;
}

/// Decomposes the first shape of `source`. Caller owns the result.
fn decomposeFirst(
    source: []const u8,
    origin: slughorn.Origin,
    auto_metrics: bool,
) !svg.Decomposed {
    const gpa = testing.allocator;

    const image = svg.parseFromMemory(gpa, source, dpi);
    defer image.deinit();

    const scale = image.scale() orelse return error.NoScale;

    var it = image.shapes();
    const shape = it.next() orelse return error.NoShapes;

    return svg.decomposePath(gpa, shape, scale, origin, auto_metrics) orelse error.NoCurves;
}

test "parses a triangle into em-normalized curves" {
    const gpa = testing.allocator;

    var decomposed = try decomposeFirst(triangle, .default, true);
    defer decomposed.deinit(gpa);

    const curves = decomposed.curves.items;
    try testing.expect(curves.len > 0);

    // scale = 1/width = 1/100, and the triangle spans the full canvas, so every coordinate must
    // land in em-space [0, 1].
    for (curves) |curve| {
        for ([_]Slug{ curve.x1, curve.x2, curve.x3 }) |x| {
            try testing.expect(x >= -1e-6 and x <= 1.0 + 1e-6);
        }
        for ([_]Slug{ curve.y1, curve.y2, curve.y3 }) |y| {
            try testing.expect(y >= -1e-6 and y <= 1.0 + 1e-6);
        }
    }

    // The contour is closed: it ends where it started.
    try testing.expectApproxEqAbs(curves[0].x1, curves[curves.len - 1].x3, 1e-6);
    try testing.expectApproxEqAbs(curves[0].y1, curves[curves.len - 1].y3, 1e-6);

    // Area 1/2 in em-space (a half-unit-square triangle), whichever way it winds.
    try testing.expectApproxEqAbs(@as(f64, 0.5), @abs(signedArea(curves)), 1e-5);
}

test "fill-rule=nonzero leaves both sub-paths wound the same way" {
    const gpa = testing.allocator;

    var decomposed = try decomposeFirst(ringSvg("nonzero"), .default, true);
    defer decomposed.deinit(gpa);

    // Same winding => the areas add: (0.8 * 0.8) + (0.4 * 0.4) = 0.80.
    try testing.expectApproxEqAbs(
        @as(f64, 0.80),
        @abs(signedArea(decomposed.curves.items)),
        1e-5,
    );
}

test "fill-rule=evenodd reverses the inner sub-path into a hole" {
    const gpa = testing.allocator;

    var decomposed = try decomposeFirst(ringSvg("evenodd"), .default, true);
    defer decomposed.deinit(gpa);

    // The reversal makes the inner contour subtract: (0.8 * 0.8) - (0.4 * 0.4) = 0.48.
    //
    // This also pins the sub-path *ordering*. The ray-cast only fires for a sub-path with curves
    // already behind it, so if nanosvg's reversed path list were consumed as-is, the inner square
    // would be visited first, no reversal would happen, and this would read 0.80.
    try testing.expectApproxEqAbs(
        @as(f64, 0.48),
        @abs(signedArea(decomposed.curves.items)),
        1e-5,
    );
}

test "sub-paths are decomposed in document order" {
    const gpa = testing.allocator;

    // auto_metrics = false keeps canvas coordinates, so the sub-path starts stay distinguishable
    // (with auto-metrics the outer's start is rebased onto the bbox corner at the origin).
    var decomposed = try decomposeFirst(ringSvg("nonzero"), .default, false);
    defer decomposed.deinit(gpa);

    // The outer square starts at (10, 10) -> em (0.1, 0.1). If the reversed list leaked through,
    // this would be the inner square's (0.3, 0.3).
    const first = decomposed.curves.items[0];
    try testing.expectApproxEqAbs(@as(Slug, 0.1), first.x1, 1e-6);
    try testing.expectApproxEqAbs(@as(Slug, 0.1), first.y1, 1e-6);
}

test "a degenerate bounding box decomposes to nothing" {
    const gpa = testing.allocator;

    const zero_area =
        \\<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
        \\  <path d="M 50 50 L 50 50 Z" fill="black"/>
        \\</svg>
    ;

    const image = svg.parseFromMemory(gpa, zero_area, dpi);
    defer image.deinit();

    const scale = image.scale().?;

    var it = image.shapes();
    if (it.next()) |shape| {
        try testing.expect(svg.decomposePath(gpa, shape, scale, .default, true) == null);
    }
}

test "an image with no positive width has no scale" {
    const gpa = testing.allocator;

    const no_width =
        \\<svg xmlns="http://www.w3.org/2000/svg" width="0" height="0"></svg>
    ;

    const image = svg.parseFromMemory(gpa, no_width, dpi);
    defer image.deinit();

    try testing.expect(image.scale() == null);
    try testing.expect(image.heightEm() == null);
}

test "input that is not SVG yields an empty, unusable image" {
    const gpa = testing.allocator;

    // NanoSVG never rejects anything -- garbage parses to a zero-width image with no shapes, so
    // `scale()` is the only signal a caller gets. Worth pinning: it is the reason parseFromMemory
    // has no error in its signature.
    const image = svg.parseFromMemory(gpa, "definitely not svg", dpi);
    defer image.deinit();

    try testing.expect(image.scale() == null);

    var it = image.shapes();
    try testing.expect(it.next() == null);
}

test "parseFromMemory does not modify the caller's buffer" {
    const gpa = testing.allocator;

    // nsvgParse writes into the buffer it is handed, so the wrapper must be duplicating ours.
    const source = try gpa.dupe(u8, triangle);
    defer gpa.free(source);

    const image = svg.parseFromMemory(gpa, source, dpi);
    defer image.deinit();

    try testing.expectEqualStrings(triangle, source);
}

test "loadShape registers the shape with an atlas" {
    const gpa = testing.allocator;

    var atlas = try slughorn.Atlas.init(gpa, slughorn.default_texture_width);
    defer atlas.deinit();

    const image = svg.parseFromMemory(gpa, triangle, dpi);
    defer image.deinit();

    const scale = image.scale().?;

    var it = image.shapes();
    const shape = it.next().?;

    const key: slughorn.Key = .{ .name = "triangle" };
    const transform = try svg.loadShape(gpa, &atlas, shape, key, scale, .default, true, 1);

    // The triangle's bbox corner is the canvas origin, so it needs no placement offset.
    try testing.expect(transform != null);
    try testing.expectApproxEqAbs(@as(Slug, 0), transform.?.x, 1e-6);
    try testing.expectApproxEqAbs(@as(Slug, 0), transform.?.y, 1e-6);

    try atlas.build();
    try testing.expect(atlas.isBuilt());
    try testing.expect(atlas.getShape(key) != null);
}

test "loadShape with fixed metrics spans the whole viewport" {
    const gpa = testing.allocator;

    var atlas = try slughorn.Atlas.init(gpa, slughorn.default_texture_width);
    defer atlas.deinit();

    // A 100x50 canvas is 2 em wide and 0.5 em tall.
    const wide =
        \\<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50">
        \\  <path d="M 10 10 L 90 10 L 90 40 L 10 40 Z" fill="black"/>
        \\</svg>
    ;

    const image = svg.parseFromMemory(gpa, wide, dpi);
    defer image.deinit();

    const height_em = image.heightEm().?;
    try testing.expectApproxEqAbs(@as(Slug, 0.5), height_em, 1e-6);

    const scale = image.scale().?;

    var it = image.shapes();
    const shape = it.next().?;

    const key: slughorn.Key = .{ .name = "rect" };
    const transform = try svg.loadShape(gpa, &atlas, shape, key, scale, .default, false, height_em);

    // Fixed metrics means the shape keeps canvas coordinates, so no placement offset applies.
    try testing.expect(transform != null);
    try testing.expectApproxEqAbs(@as(Slug, 0), transform.?.x, 1e-6);
    try testing.expectApproxEqAbs(@as(Slug, 0), transform.?.y, 1e-6);

    try atlas.build();
    try testing.expect(atlas.getShape(key) != null);
}

test "shapes are reported visible and filled" {
    const gpa = testing.allocator;

    const image = svg.parseFromMemory(gpa, triangle, dpi);
    defer image.deinit();

    var it = image.shapes();
    const shape = it.next().?;

    try testing.expect(svg.isVisible(shape));
    try testing.expect(svg.isFilled(shape));
    try testing.expect(it.next() == null);
}
