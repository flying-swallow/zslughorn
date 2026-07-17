// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! SDF/MSDF tile generation from a shape's curves.
//!
//! Ported from the `renderSDF`/`renderMSDF`/`renderMSDFTile` family in `slughorn/render.hpp`, which
//! upstream backs with Chlumsky's C++ msdfgen. Here the generator is the user's msdf-zig (a pure-Zig
//! msdfgen port), reached through its FreeType-free `msdf-core` module: we hand it a `Shape` built
//! from the atlas's retained em-space quadratics, so no font and no FreeType are involved.
//!
//! Like the nanosvg backend, this is a separate module that imports `slughorn`, never the reverse.
//! MSDF is upstream's default-off `SLUGHORN_MSDF` side-channel; it stays out of the MIT core.
//!
//! Scope: the generation primitives only. The core `Shape` now carries the MSDF result fields
//! (`msdf_layer`/`msdf_range`), but the atlas-level machinery that would populate them
//! (`rasterizeSDFAtlas`, `requestMSDF`, the MSDF texture array, serialization) is not ported.

const std = @import("std");
const slughorn = @import("slughorn");
const msdf = @import("msdf");

const oom = slughorn.oom;
const Slug = slughorn.Slug;
const Curve = slughorn.Curve;
const Atlas = slughorn.Atlas;
const Key = slughorn.Key;

const Vec2 = @Vector(2, f64);

/// msdf-zig's distance-field flavours, re-exported. `renderSDF`/`renderMSDF` pick `sdf`/`msdf`;
/// `psdf`/`mtsdf`/`msdf10` are reachable via `renderTyped`.
pub const SdfType = msdf.SdfType;

/// A rasterized distance-field tile.
///
/// Row-major, **row 0 = top** (msdf-zig's native orientation already matches upstream's `Grid`, so
/// unlike `render.hpp:renderSDF` no Y-flip is applied). Values are in [0, 1]: for a plain SDF an
/// edge texel is ~0.5, interior > 0.5, exterior < 0.5; for an MSDF reconstruct the distance with
/// `median(r, g, b)`.
pub const Grid = struct {
    width: u32,
    height: u32,
    channels: u8,
    data: []f32,

    pub fn deinit(self: Grid, gpa: std.mem.Allocator) void {
        gpa.free(self.data);
    }

    /// Channel `ch` of texel (x, y).
    pub fn at(self: Grid, x: u32, y: u32, ch: u8) f32 {
        return self.data[(y * self.width + x) * self.channels + ch];
    }
};

fn v2(x: Slug, y: Slug) Vec2 {
    return .{ @floatCast(x), @floatCast(y) };
}

/// Builds an msdf-core `Shape` from a shape's retained flat quadratic curves.
///
/// Contours are recovered exactly as upstream's `Atlas::getShapeContours` does: a run of curves
/// belongs to one contour until a curve's end point stops matching the next curve's start point
/// (slughorn.cpp:864). Each `Curve` is a quadratic (start / off-curve / end), so it maps to one
/// quadratic `EdgeSegment` -- the same shape `ftConicTo` builds from a font outline.
///
/// Caller owns the result and must free it with `freeShape`.
fn toShape(gpa: std.mem.Allocator, curves: []const Curve) msdf.Shape {
    var shape: msdf.Shape = .{};
    if (curves.len == 0) return shape;

    var contour: msdf.Contour = .{};
    for (curves, 0..) |c, i| {
        oom.must(contour.edges.append(gpa, msdf.EdgeSegment.create(
            v2(c.x1, c.y1),
            v2(c.x2, c.y2),
            v2(c.x3, c.y3),
            null,
            .white,
        )));

        const last = i + 1 == curves.len;
        const brk = !last and (c.x3 != curves[i + 1].x1 or c.y3 != curves[i + 1].y1);
        if (last or brk) {
            oom.must(shape.contours.append(gpa, contour));
            contour = .{};
        }
    }
    return shape;
}

fn freeShape(gpa: std.mem.Allocator, shape: *msdf.Shape) void {
    for (shape.contours.items) |*c| c.edges.deinit(gpa);
    shape.contours.deinit(gpa);
}

/// A raw distance-field bitmap: `data` is `width * height * channels` floats in [0, 1], row 0 top.
pub const Bitmap = struct {
    width: u32,
    height: u32,
    channels: u8,
    data: []f32,

    pub fn deinit(self: Bitmap, gpa: std.mem.Allocator) void {
        gpa.free(self.data);
    }
};

/// Rasterizes shape `key` to a distance field of the given type.
///
/// `tile_size` is the target length of the longer axis in texels; `range` is the em-space spread
/// that maps to the full [0, 1] output (edge at 0.5), matching `render.hpp`'s `range`. The tile is
/// aspect-matched to the shape's bounding box (plus the range margin), so the shorter axis is
/// smaller than `tile_size`. Returns null when the shape is unknown, has no curves, or has an empty
/// bounding box. Caller owns the result.
pub fn renderTyped(
    gpa: std.mem.Allocator,
    atlas: *const Atlas,
    key: Key,
    sdf_type: SdfType,
    tile_size: u32,
    range: Slug,
) !?Bitmap {
    const shape = atlas.getShape(key) orelse return null;
    if (shape.curves.len == 0) return null;

    var s = toShape(gpa, shape.curves);
    defer freeShape(gpa, &s);

    // Tight bounds before normalization; msdf-zig recomputes the same box inside generateFromShape
    // (edge subdivision does not move the bounding box).
    const bounds = s.getBounds(0, 0, 0);
    const bw = bounds.right - bounds.left;
    const bh = bounds.top - bounds.bottom;
    if (!(bw > 0) or !(bh > 0)) return null;

    // Map (tile_size, range) onto msdf-zig's (px_size, px_range) model. The bitmap msdf-zig emits
    // is (bw + px_range/px_size) * px_size on the long axis; choosing px_range = 2*range*px_size
    // makes px_range/px_size == 2*range, so the padding equals `range` on every side (upstream's
    // `getBounds(range)`) and the long axis lands on tile_size.
    const rng: f64 = @floatCast(range);
    const long = @max(bw + 2.0 * rng, bh + 2.0 * rng);
    const px_size_f = @max(1.0, @round(@as(f64, @floatFromInt(tile_size)) / long));
    const px_size: u16 = @intFromFloat(@min(px_size_f, @as(f64, std.math.maxInt(u16))));
    const px_range: u16 = @intFromFloat(@max(1.0, @round(2.0 * rng * @as(f64, @floatFromInt(px_size)))));

    var data = try msdf.generateFromShape(gpa, &s, .{
        .sdf_type = sdf_type,
        .px_size = px_size,
        .px_range = px_range,
        // The atlas curves already carry correct nonzero winding (holes were reversed at decompose
        // time), so preserve it and let msdf-zig's out-of-bounds probe fix only the global sign.
        .orientation = .guess,
    });
    defer data.deinit(gpa);

    const channels = sdf_type.numChannels();
    const pixels: []const u8 = switch (data.pixels) {
        .normal => |p| p,
        // msdf10 is a packed format with no []u8 view; renderTyped's callers never request it.
        .msdf10 => return error.UnsupportedSdfType,
    };

    const out = try gpa.alloc(f32, @as(usize, data.width) * data.height * channels);
    for (pixels, out) |v, *o| o.* = @as(f32, @floatFromInt(v)) / 255.0;

    return .{ .width = data.width, .height = data.height, .channels = channels, .data = out };
}

fn bitmapToGrid(b: Bitmap) Grid {
    return .{ .width = b.width, .height = b.height, .channels = b.channels, .data = b.data };
}

/// Single-channel SDF tile. Ported from `render.hpp:renderSDF`.
pub fn renderSDF(gpa: std.mem.Allocator, atlas: *const Atlas, key: Key, tile_size: u32, range: Slug) !?Grid {
    const bmp = try renderTyped(gpa, atlas, key, .sdf, tile_size, range) orelse return null;
    return bitmapToGrid(bmp);
}

/// Three-channel MSDF tile (reconstruct distance with `median(r, g, b)`). Ported from
/// `render.hpp:renderMSDF`.
pub fn renderMSDF(gpa: std.mem.Allocator, atlas: *const Atlas, key: Key, tile_size: u32, range: Slug) !?Grid {
    const bmp = try renderTyped(gpa, atlas, key, .msdf, tile_size, range) orelse return null;
    return bitmapToGrid(bmp);
}

/// Square `tile_size` x `tile_size` MSDF tile, the shape letterboxed/pillarboxed and centered.
///
/// Ported from `render.hpp:renderMSDFTile`: use this for a `sampler2DArray` where every layer must
/// share dimensions. The margin outside the shape's aspect-matched bitmap is filled with 0 (deep
/// exterior). Returns null under the same conditions as `renderTyped`. Caller owns the result.
pub fn renderMSDFTile(gpa: std.mem.Allocator, atlas: *const Atlas, key: Key, tile_size: u32, range: Slug) !?Grid {
    const bmp = try renderTyped(gpa, atlas, key, .msdf, tile_size, range) orelse return null;
    defer bmp.deinit(gpa);

    const channels = bmp.channels;
    const data = try gpa.alloc(f32, @as(usize, tile_size) * tile_size * channels);
    @memset(data, 0);

    // Center the aspect-matched bitmap; renderTyped sizes its long axis to tile_size, so both axes
    // fit. Clamp defensively in case rounding pushes a dimension one texel over.
    const w = @min(bmp.width, tile_size);
    const h = @min(bmp.height, tile_size);
    const ox = (tile_size - w) / 2;
    const oy = (tile_size - h) / 2;

    for (0..h) |y| {
        const dst_row = (oy + y) * tile_size + ox;
        const src_row = y * bmp.width;
        for (0..w) |x| {
            for (0..channels) |ch| {
                data[(dst_row + x) * channels + ch] = bmp.data[(src_row + x) * channels + ch];
            }
        }
    }

    return .{ .width = tile_size, .height = tile_size, .channels = channels, .data = data };
}
