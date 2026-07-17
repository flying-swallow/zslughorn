// Copyright (c) 2026 Michael Pollind
// SPDX-License-Identifier: GPL-2.0-only
//
// Tests for the GPU renderer backend (`-Drenderer=true`). Like the nanosvg/sdf suites these run
// against a live backend rather than a byte oracle: here a real Vulkan device. The eventual
// coverage test cross-checks the GPU against `render.zig`; this first case proves the headless
// render + readback path itself works end-to-end.

const std = @import("std");
const slughorn = @import("slughorn");
const renderer = @import("slughorn_renderer");

const Slug = slughorn.Slug;
const Curve = slughorn.Curve;

/// A straight edge as a degenerate quadratic (control point at the chord midpoint).
fn line(x1: Slug, y1: Slug, x3: Slug, y3: Slug) Curve {
    return .{ .x1 = x1, .y1 = y1, .x2 = (x1 + x3) * 0.5, .y2 = (y1 + y3) * 0.5, .x3 = x3, .y3 = y3 };
}

/// RGBA8 unorm stores `round(f * 255)`; allow ±1 for the implementation's rounding of ties.
fn expectChannel(actual: u8, expected: f32) !void {
    const target = @round(expected * 255.0);
    const diff = @abs(@as(f32, @floatFromInt(actual)) - target);
    try std.testing.expect(diff <= 1.0);
}

test "headless clear + readback round-trips the clear color" {
    const gpa = std.testing.allocator;
    const w: u32 = 16;
    const h: u32 = 16;
    const clear = [4]f32{ 0.25, 0.5, 0.75, 1.0 };

    const px = try renderer.clearToRgba8(gpa, w, h, clear);
    defer gpa.free(px);

    try std.testing.expectEqual(@as(usize, w * h * 4), px.len);

    // Every texel of a cleared target is the clear color. Check the first, a middle, and the last.
    for ([_]usize{ 0, (h / 2 * w + w / 2) * 4, (w * h - 1) * 4 }) |base| {
        try expectChannel(px[base + 0], clear[0]);
        try expectChannel(px[base + 1], clear[1]);
        try expectChannel(px[base + 2], clear[2]);
        try expectChannel(px[base + 3], clear[3]);
    }
}

/// A filled triangle wound counter-clockwise, with three slanted edges. No edge is axis-aligned, so
/// none hits the degenerate `0.5/b.y` branch where the GLSL shader and render.hpp deliberately
/// diverge (DIVERGENCE.md) -- keeping this a clean apples-to-apples GPU/CPU comparison.
const triangle = [_]Curve{
    line(0.2, 0.25, 0.8, 0.35),
    line(0.8, 0.35, 0.5, 0.85),
    line(0.5, 0.85, 0.2, 0.25),
};

test "GPU coverage of a triangle matches the render.zig oracle" {
    const gpa = std.testing.allocator;

    var atlas = try slughorn.Atlas.init(gpa, slughorn.default_texture_width);
    defer atlas.deinit();
    try atlas.addShape(.{ .name = "tri" }, .{ .curves = &triangle });
    try atlas.build();

    const shape = atlas.getShape(.{ .name = "tri" }).?;

    // The CPU oracle: the same banded coverage the shader implements, over the same em window.
    var sampler = try slughorn.render.decode(gpa, shape.*, atlas.getCurveTextureData(), atlas.getBandTextureData());
    defer sampler.deinit(gpa);
    const origin = sampler.emOrigin();
    const size = sampler.emSize();

    const n: u32 = 32;
    const grid = (try renderer.renderCoverage(gpa, &atlas, .{ .name = "tri" }, n, origin.x, origin.y, size.x, size.y)).?;
    defer gpa.free(grid);

    const nf: f32 = @floatFromInt(n);
    const ppe_x = nf / size.x;
    const ppe_y = nf / size.y;

    var max_cov: f32 = 0;
    var min_cov: f32 = 1;
    var max_diff: f32 = 0;
    var agree: u32 = 0;
    var row: u32 = 0;
    while (row < n) : (row += 1) {
        var col: u32 = 0;
        while (col < n) : (col += 1) {
            const g = grid[row * n + col];
            try std.testing.expect(!std.math.isNan(g)); // the clean triangle must never produce NaN

            const emx = origin.x + (@as(f32, @floatFromInt(col)) + 0.5) / nf * size.x;
            const emy = origin.y + (@as(f32, @floatFromInt(row)) + 0.5) / nf * size.y;
            const c = sampler.renderSampleBanded(emx, emy, ppe_x, ppe_y).fill;

            const diff = @abs(g - c);
            if (diff <= 0.05) agree += 1;
            max_diff = @max(max_diff, diff);
            max_cov = @max(max_cov, g);
            min_cov = @min(min_cov, g);
        }
    }

    const total = n * n;
    std.debug.print("GPU vs render.zig: {d}/{d} within 0.05, max diff {d:.4}, coverage [{d:.3}, {d:.3}]\n", .{ agree, total, max_diff, min_cov, max_cov });

    // It actually drew a filled glyph: some texel reaches full interior coverage, some is empty.
    try std.testing.expect(max_cov > 0.9);
    try std.testing.expect(min_cov < 0.1);

    // And it matches the CPU banded oracle almost everywhere (a few edge texels may differ by more
    // than the threshold from GPU/CPU float ordering; the bulk must agree tightly).
    try std.testing.expect(agree * 100 >= total * 97);
}
