// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT
//
// Tests for the Canvas backend (`-Dcanvas=true`). Property-based (no C++ oracle): they pin the
// drawing-call -> curve -> CompositeShape pipeline (layer count, colors, placement, enclosed area)
// rather than exact control points -- the curve math is covered by golden_decompose.zig.

const std = @import("std");
const slughorn = @import("slughorn");
const cv = @import("slughorn_canvas");

const testing = std.testing;
const Slug = slughorn.Slug;
const Curve = slughorn.Curve;

/// Shoelace area over each curve's chord. Exact for straight edges; an inscribed-polygon under-
/// estimate for curved ones (good enough to bound a circle's area).
fn signedArea(curves: []const Curve) f64 {
    var sum: f64 = 0;
    for (curves) |c| sum += @as(f64, c.x1) * @as(f64, c.y3) - @as(f64, c.x3) * @as(f64, c.y1);
    return sum * 0.5;
}

fn colorNear(col: slughorn.Color, r: Slug, g: Slug, b: Slug) bool {
    return @abs(col[0] - r) < 0.01 and @abs(col[1] - g) < 0.01 and @abs(col[2] - b) < 0.01;
}

test "canvas fills a rect and a circle into a two-layer composite" {
    const gpa = testing.allocator;
    var atlas = try slughorn.Atlas.init(gpa, slughorn.default_texture_width);
    defer atlas.deinit();

    var canvas = cv.Canvas.init(gpa, &atlas);
    defer canvas.deinit();

    canvas.rect(0.1, 0.1, 0.6, 0.4);
    const rect_key = try canvas.fill(slughorn.rgb(1, 0, 0), 1);

    canvas.circle(0.5, 0.5, 0.3);
    const circ_key = try canvas.fill(slughorn.rgb(0, 0, 1), 1);

    try testing.expectEqual(@as(usize, 2), canvas.layerCount());

    var composite = canvas.finalize();
    defer composite.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), composite.layers.items.len);

    // Layers keep their fill color and placement (the shape's bbox corner).
    try testing.expect(colorNear(composite.layers.items[0].color, 1, 0, 0));
    try testing.expect(colorNear(composite.layers.items[1].color, 0, 0, 1));
    try testing.expectApproxEqAbs(@as(Slug, 0.1), composite.layers.items[0].transform.x, 1e-5);
    try testing.expectApproxEqAbs(@as(Slug, 0.1), composite.layers.items[0].transform.y, 1e-5);

    try atlas.build();

    // The rect is a closed quad; shape-local, its area is exactly 0.6 * 0.4 = 0.24.
    const rect_shape = atlas.getShape(rect_key).?;
    try testing.expectApproxEqAbs(@as(f64, 0.24), @abs(signedArea(rect_shape.curves)), 0.005);
    try testing.expectApproxEqAbs(@as(Slug, 0.6), rect_shape.width, 1e-4);
    try testing.expectApproxEqAbs(@as(Slug, 0.4), rect_shape.height, 1e-4);

    // The circle (r = 0.3) encloses ~pi*r^2 = 0.283; the inscribed-polygon area is a bit under.
    const circ_shape = atlas.getShape(circ_key).?;
    const circ_area = @abs(signedArea(circ_shape.curves));
    try testing.expect(circ_area > 0.24 and circ_area < 0.30);
}

test "canvas transform stack offsets committed geometry" {
    const gpa = testing.allocator;
    var atlas = try slughorn.Atlas.init(gpa, slughorn.default_texture_width);
    defer atlas.deinit();

    var canvas = cv.Canvas.init(gpa, &atlas);
    defer canvas.deinit();

    canvas.save();
    canvas.translate(0.2, 0.3);
    canvas.rect(0, 0, 0.4, 0.4); // transformed to bbox min (0.2, 0.3)
    _ = try canvas.fill(slughorn.rgb(0, 1, 0), 1);
    canvas.restore();

    var composite = canvas.finalize();
    defer composite.deinit(gpa);

    try testing.expectApproxEqAbs(@as(Slug, 0.2), composite.layers.items[0].transform.x, 1e-5);
    try testing.expectApproxEqAbs(@as(Slug, 0.3), composite.layers.items[0].transform.y, 1e-5);
}
