// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! Tests for the SDF/MSDF backend.
//!
//! Hand-derived expectations, like the nanosvg suite: there is no byte oracle here (msdf-zig is a
//! different implementation from upstream's msdfgen, and the two differ at the LSB by design). The
//! assertions pin the *properties* of a distance field -- edge ~= 0.5, interior > 0.5, exterior
//! < 0.5, MSDF median reconstruction -- not exact texel values.

const std = @import("std");
const slughorn = @import("slughorn");
const sdf = @import("slughorn_sdf");

const testing = std.testing;
const Slug = slughorn.Slug;
const Curve = slughorn.Curve;

/// A straight edge as a degenerate quadratic (control point at the chord midpoint).
fn line(x1: Slug, y1: Slug, x3: Slug, y3: Slug) Curve {
    return .{ .x1 = x1, .y1 = y1, .x2 = (x1 + x3) * 0.5, .y2 = (y1 + y3) * 0.5, .x3 = x3, .y3 = y3 };
}

/// A filled square [0.2, 0.8]^2, wound counter-clockwise, as four line-quadratics.
const square = [_]Curve{
    line(0.2, 0.2, 0.8, 0.2),
    line(0.8, 0.2, 0.8, 0.8),
    line(0.8, 0.8, 0.2, 0.8),
    line(0.2, 0.8, 0.2, 0.2),
};

fn squareAtlas(gpa: std.mem.Allocator) !slughorn.Atlas {
    var atlas = try slughorn.Atlas.init(gpa, slughorn.default_texture_width);
    errdefer atlas.deinit();
    try atlas.addShape(.{ .name = "square" }, .{ .curves = &square });
    try atlas.build();
    return atlas;
}

fn median(a: f32, b: f32, c: f32) f32 {
    return @max(@min(a, b), @min(@max(a, b), c));
}

test "renderSDF of a filled square: interior > 0.5, exterior < 0.5" {
    const gpa = testing.allocator;
    var atlas = try squareAtlas(gpa);
    defer atlas.deinit();

    var grid = (try sdf.renderSDF(gpa, &atlas, .{ .name = "square" }, 48, 0.15)).?;
    defer grid.deinit(gpa);

    try testing.expect(grid.width > 0 and grid.height > 0);
    try testing.expectEqual(@as(u8, 1), grid.channels);

    // The tile is framed on the square's bbox plus the range margin, so its center is the square's
    // center (deep interior) and its corners sit in the exterior margin.
    try testing.expect(grid.at(grid.width / 2, grid.height / 2, 0) > 0.5);
    try testing.expect(grid.at(0, 0, 0) < 0.5);
    try testing.expect(grid.at(grid.width - 1, grid.height - 1, 0) < 0.5);
}

test "renderMSDF: three channels, median reconstructs the sign" {
    const gpa = testing.allocator;
    var atlas = try squareAtlas(gpa);
    defer atlas.deinit();

    var grid = (try sdf.renderMSDF(gpa, &atlas, .{ .name = "square" }, 48, 0.15)).?;
    defer grid.deinit(gpa);

    try testing.expectEqual(@as(u8, 3), grid.channels);

    const cx = grid.width / 2;
    const cy = grid.height / 2;
    const inside = median(grid.at(cx, cy, 0), grid.at(cx, cy, 1), grid.at(cx, cy, 2));
    const outside = median(grid.at(0, 0, 0), grid.at(0, 0, 1), grid.at(0, 0, 2));
    try testing.expect(inside > 0.5);
    try testing.expect(outside < 0.5);
}

test "renderMSDFTile is always square with the shape interior at its center" {
    const gpa = testing.allocator;
    var atlas = try squareAtlas(gpa);
    defer atlas.deinit();

    var grid = (try sdf.renderMSDFTile(gpa, &atlas, .{ .name = "square" }, 64, 0.15)).?;
    defer grid.deinit(gpa);

    try testing.expectEqual(@as(u32, 64), grid.width);
    try testing.expectEqual(@as(u32, 64), grid.height);
    try testing.expectEqual(@as(u8, 3), grid.channels);

    // A square shape has equal aspect, so it fills the tile edge to edge (no margin): center is
    // interior, corner is exterior.
    const inside = median(grid.at(32, 32, 0), grid.at(32, 32, 1), grid.at(32, 32, 2));
    const corner = median(grid.at(0, 0, 0), grid.at(0, 0, 1), grid.at(0, 0, 2));
    try testing.expect(inside > 0.5);
    try testing.expect(corner < 0.5);
}

test "renderMSDFTile letterboxes a non-square shape with a zero margin" {
    const gpa = testing.allocator;

    // A wide, short rectangle: its long axis fills the tile, leaving a top/bottom margin.
    const wide = [_]Curve{
        line(0.1, 0.4, 0.9, 0.4),
        line(0.9, 0.4, 0.9, 0.6),
        line(0.9, 0.6, 0.1, 0.6),
        line(0.1, 0.6, 0.1, 0.4),
    };
    var atlas = try slughorn.Atlas.init(gpa, slughorn.default_texture_width);
    defer atlas.deinit();
    try atlas.addShape(.{ .name = "wide" }, .{ .curves = &wide });
    try atlas.build();

    var grid = (try sdf.renderMSDFTile(gpa, &atlas, .{ .name = "wide" }, 64, 0.1)).?;
    defer grid.deinit(gpa);

    try testing.expectEqual(@as(u32, 64), grid.width);
    try testing.expectEqual(@as(u32, 64), grid.height);

    // The top row falls in the letterbox margin, filled with exactly 0.
    try testing.expectEqual(@as(f32, 0), grid.at(32, 1, 0));
    // The center is still interior.
    const inside = median(grid.at(32, 32, 0), grid.at(32, 32, 1), grid.at(32, 32, 2));
    try testing.expect(inside > 0.5);
}

test "unknown key renders nothing" {
    const gpa = testing.allocator;
    var atlas = try squareAtlas(gpa);
    defer atlas.deinit();

    try testing.expect((try sdf.renderSDF(gpa, &atlas, .{ .name = "missing" }, 48, 0.15)) == null);
}

test "MsdfAtlas records a per-shape layer and packs an RGB32F array" {
    const gpa = testing.allocator;
    var atlas = try squareAtlas(gpa);
    defer atlas.deinit();

    var ma = sdf.MsdfAtlas.init(32);
    defer ma.deinit(gpa);

    const layer = (try ma.request(gpa, &atlas, .{ .name = "square" }, 0.15)).?;
    try testing.expectEqual(@as(i32, 0), layer); // first shape -> layer 0
    try testing.expectEqual(@as(usize, 1), ma.layerCount());

    // The result is written back onto the core Shape (Phase 2's msdf_layer/msdf_range).
    const shape = atlas.getShape(.{ .name = "square" }).?;
    try testing.expectEqual(@as(i32, 0), shape.msdf_layer);
    try testing.expectApproxEqAbs(@as(Slug, 0.15), shape.msdf_range, 1e-6);

    // Re-request dedups to the same layer without generating a second tile.
    const again = (try ma.request(gpa, &atlas, .{ .name = "square" }, 0.15)).?;
    try testing.expectEqual(@as(i32, 0), again);
    try testing.expectEqual(@as(usize, 1), ma.layerCount());

    // The packed array is RGB32F, tile_size square, one layer.
    const td = ma.textureData(gpa);
    defer gpa.free(td.bytes);
    try testing.expectEqual(@as(u32, 32), td.width);
    try testing.expectEqual(@as(u32, 32), td.height);
    try testing.expectEqual(@as(u32, 1), td.depth);
    try testing.expectEqual(slughorn.TextureData.Format.rgb32f, td.format);
    try testing.expectEqual(@as(usize, 32 * 32 * 3 * 4), td.bytes.len);

    // The tile's center texel is interior: median(r, g, b) > 0.5. Read alignment-safely.
    const ti = (16 * 32 + 16) * 3; // first channel of the center texel, in floats
    const r: f32 = @bitCast(std.mem.readInt(u32, td.bytes[(ti + 0) * 4 ..][0..4], .little));
    const g: f32 = @bitCast(std.mem.readInt(u32, td.bytes[(ti + 1) * 4 ..][0..4], .little));
    const b: f32 = @bitCast(std.mem.readInt(u32, td.bytes[(ti + 2) * 4 ..][0..4], .little));
    try testing.expect(median(r, g, b) > 0.5);

    // Unknown key -> null, nothing generated.
    try testing.expect((try ma.request(gpa, &atlas, .{ .name = "nope" }, 0.15)) == null);
    try testing.expectEqual(@as(usize, 1), ma.layerCount());
}
