// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! Band construction: slice a shape into horizontal and vertical bands, record which curves cross
//! each, and build the indirection tables that let the shader find a band in O(1).
//!
//! This is what makes the Slug technique fast: rather than testing every curve per fragment, the
//! shader quantizes the em-coordinate, reads a band index out of a 32-slot table, and only tests
//! the curves in that band -- sorted so it can stop early.
//!
//! Ported from `Atlas::buildShapeBands`, slughorn.cpp:907.

const std = @import("std");
const build_options = @import("build_options");

const types = @import("types.zig");
const errors = @import("errors.zig");

const Slug = types.Slug;
const Curve = types.Curve;
const Shape = types.Shape;
const Origin = types.Origin;

const indirection_size: u32 = build_options.indirection_size;

/// Largest band count an axis may have.
///
/// The indirection tables hold one band index per slot as a `u8`, so the largest addressable index
/// is 255 and thus the largest legal count is 256.
///
/// DIVERGENCE from upstream, which has no such check: `static_cast<uint8_t>(band)` at
/// slughorn.cpp:1083 and :1152 silently truncates, so band 256 wraps to 0 and the shader reads the
/// wrong band's curves -- wrong pixels, no diagnostic. Reachable via `splits_y` with 256+ entries.
/// We reject instead. See DIVERGENCE.md.
pub const max_bands: u32 = 256;

/// One band: the curves crossing it, ordered for the shader's early-out.
pub const BandEntry = struct {
    curve_indices: std.ArrayList(u32) = .empty,

    pub fn deinit(self: *BandEntry, gpa: std.mem.Allocator) void {
        self.curve_indices.deinit(gpa);
    }
};

/// Working state for one shape between `addShape` and `packTextures`.
///
/// Ported from `Atlas::ShapeBuild`, slughorn.hpp:1305. Upstream keeps a `curveCount` alongside the
/// index list; here `curve_indices.items.len` is the single source of truth.
pub const ShapeBuild = struct {
    metrics: Shape = .{},
    curves: std.ArrayList(Curve) = .empty,
    hbands: std.ArrayList(BandEntry) = .empty,
    vbands: std.ArrayList(BandEntry) = .empty,
    splits_x: std.ArrayList(Slug) = .empty,
    splits_y: std.ArrayList(Slug) = .empty,
    /// Band index per quantized em-coordinate slot; `indirection_size` entries each.
    indir_x: std.ArrayList(u8) = .empty,
    indir_y: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *ShapeBuild, gpa: std.mem.Allocator) void {
        for (self.hbands.items) |*b| b.deinit(gpa);
        for (self.vbands.items) |*b| b.deinit(gpa);
        self.hbands.deinit(gpa);
        self.vbands.deinit(gpa);
        self.curves.deinit(gpa);
        self.splits_x.deinit(gpa);
        self.splits_y.deinit(gpa);
        self.indir_x.deinit(gpa);
        self.indir_y.deinit(gpa);
    }
};

/// Does this curve's y-extent overlap [lo, hi]? Ported from slughorn.cpp:164.
pub fn curveIntersectsBandY(c: Curve, lo: Slug, hi: Slug) bool {
    return c.maxY() >= lo and c.minY() <= hi;
}

/// Does this curve's x-extent overlap [lo, hi]? Ported from slughorn.cpp:171.
pub fn curveIntersectsBandX(c: Curve, lo: Slug, hi: Slug) bool {
    return c.maxX() >= lo and c.minX() <= hi;
}

/// Sort context: orders curve indices by an extent, descending, with a total order.
///
/// The descending order is what lets the shader stop scanning a band early. Upstream's comparator
/// is a bare `>` on the extent (slughorn.cpp:1055 for max-x, :1125 for max-y), which is only a
/// *partial* order -- tied curves may come out in any order, and `std::sort` is unstable, so the
/// result depends on the standard library's pivot choices. (Measured against libstdc++: with tied
/// keys the original order survives to n=16 and is scrambled from n=17 on. libc++ would differ
/// again.)
///
/// Adding the index as a tie-break makes the order total, so every correct sort agrees. It is
/// semantically free: the sort exists only for the early-out, so any order among equal-extent
/// curves is equally valid. Given upstream pushes indices ascending (slughorn.cpp:1053/:1116),
/// this is exactly what a `stable_sort` there would produce.
const ExtentSort = struct {
    curves: []const Curve,
    extent: *const fn (Curve) Slug,

    fn lessThan(self: ExtentSort, l: u32, r: u32) bool {
        const el = self.extent(self.curves[l]);
        const er = self.extent(self.curves[r]);
        if (el != er) return el > er;
        return l < r; // total order; see the note above
    }
};

fn maxXOf(c: Curve) Slug {
    return c.maxX();
}
fn maxYOf(c: Curve) Slug {
    return c.maxY();
}

/// Slices `build` into bands and fills its metrics, band lists, and indirection tables.
///
/// `override_metrics` is upstream's `!autoMetrics`: when set, the caller's declared extent is used
/// as the band range rather than the tight curve bbox, so coverage stays correct outside the bbox.
pub fn buildShapeBands(
    gpa: std.mem.Allocator,
    build: *ShapeBuild,
    num_bands_x_in: u32,
    num_bands_y_in: u32,
    override_metrics: bool,
    origin: Origin,
) errors.BuildError!void {
    // An empty shape has no geometry to band; it still packs a (degenerate) block.
    if (build.curves.items.len == 0) return;

    const num_curves = build.curves.items.len;

    // -- band counts: splits win, else auto-pick, else the caller's value ------------------------
    var num_bands_y = num_bands_y_in;
    var num_bands_x = num_bands_x_in;

    if (build.splits_y.items.len != 0) {
        num_bands_y = @intCast(build.splits_y.items.len + 1);
    } else if (num_bands_y == 0) {
        num_bands_y = @intCast(@min(@as(usize, 16), @max(@as(usize, 1), num_curves / 2)));
    }

    if (build.splits_x.items.len != 0) {
        num_bands_x = @intCast(build.splits_x.items.len + 1);
    } else if (num_bands_x == 0) {
        num_bands_x = @intCast(@min(@as(usize, 16), @max(@as(usize, 1), num_curves / 2)));
    }

    if (num_bands_x > max_bands or num_bands_y > max_bands) return error.TooManyBands;

    // -- bounding box ----------------------------------------------------------------------------
    // Seeded with +/-1e9 rather than +/-inf, as upstream (slughorn.cpp:943). Curves carrying NaN
    // or inf are rejected at addShape, so these sentinels are always overwritten.
    var min_x: Slug = 1e9;
    var min_y: Slug = 1e9;
    var max_x: Slug = -1e9;
    var max_y: Slug = -1e9;

    for (build.curves.items) |c| {
        min_x = @min(min_x, c.minX());
        min_y = @min(min_y, c.minY());
        max_x = @max(max_x, c.maxX());
        max_y = @max(max_y, c.maxY());
    }

    var range_x = max_x - min_x;
    var range_y = max_y - min_y;

    if (range_x < 1e-6) range_x = 1e-6;
    if (range_y < 1e-6) range_y = 1e-6;

    // -- metrics ---------------------------------------------------------------------------------
    if (!override_metrics) {
        build.metrics.width = range_x;
        build.metrics.height = range_y;
        build.metrics.bearing_x = min_x;
        build.metrics.bearing_y = max_y; // top of shape; Y-up
        build.metrics.advance = range_x;
    }

    switch (origin.type) {
        .centered => {
            build.metrics.origin_x = range_x / 2;
            build.metrics.origin_y = range_y / 2;
        },
        .pivot, .custom => {
            build.metrics.origin_x = origin.x;
            build.metrics.origin_y = origin.y;
        },
        .default => {},
    }

    build.metrics.origin = origin;

    // With overridden metrics the declared extent becomes the band range, so band transforms and
    // indirection tables line up with the layout bounds rather than the tight curve bbox --
    // required for correct coverage outside the bbox (e.g. tiling).
    if (override_metrics) {
        min_x = build.metrics.bearing_x;
        max_x = build.metrics.bearing_x + build.metrics.width;
        min_y = build.metrics.bearing_y - build.metrics.height;
        max_y = build.metrics.bearing_y;
        range_x = build.metrics.width;
        range_y = build.metrics.height;

        if (range_x < 1e-6) range_x = 1e-6;
        if (range_y < 1e-6) range_y = 1e-6;
    }

    // -- band transform --------------------------------------------------------------------------
    // The shader indexes the indirection table with
    // clamp(int(em_pos * band_scale + band_offset), 0, indirection_size - 1).
    const isize_f: Slug = @floatFromInt(indirection_size);
    build.metrics.band_scale_x = isize_f / range_x;
    build.metrics.band_scale_y = isize_f / range_y;
    build.metrics.band_offset_x = -min_x * build.metrics.band_scale_x;
    build.metrics.band_offset_y = -min_y * build.metrics.band_scale_y;
    build.metrics.band_max_x = num_bands_x - 1;
    build.metrics.band_max_y = num_bands_y - 1;

    // -- horizontal bands (sliced along Y) -------------------------------------------------------
    {
        const hboundaries = gpa.alloc(Slug, num_bands_y + 1) catch @panic("slughorn: oom");
        defer gpa.free(hboundaries);

        computeBoundaries(hboundaries, build.splits_y.items, num_bands_y, min_y, max_y, range_y);

        build.hbands.resize(gpa, num_bands_y) catch @panic("slughorn: oom");
        for (build.hbands.items) |*b| b.* = .{};

        for (build.hbands.items, 0..) |*band, b| {
            const lo = hboundaries[b];
            const hi = hboundaries[b + 1];

            for (build.curves.items, 0..) |c, ci| {
                if (curveIntersectsBandY(c, lo, hi)) {
                    band.curve_indices.append(gpa, @intCast(ci)) catch @panic("slughorn: oom");
                }
            }

            std.sort.pdq(
                u32,
                band.curve_indices.items,
                ExtentSort{ .curves = build.curves.items, .extent = &maxXOf },
                ExtentSort.lessThan,
            );
        }

        buildIndirection(gpa, &build.indir_y, hboundaries, num_bands_y, min_y, range_y);
    }

    // -- vertical bands (sliced along X) ---------------------------------------------------------
    {
        const vboundaries = gpa.alloc(Slug, num_bands_x + 1) catch @panic("slughorn: oom");
        defer gpa.free(vboundaries);

        computeBoundaries(vboundaries, build.splits_x.items, num_bands_x, min_x, max_x, range_x);

        build.vbands.resize(gpa, num_bands_x) catch @panic("slughorn: oom");
        for (build.vbands.items) |*b| b.* = .{};

        for (build.vbands.items, 0..) |*band, b| {
            const lo = vboundaries[b];
            const hi = vboundaries[b + 1];

            for (build.curves.items, 0..) |c, ci| {
                if (curveIntersectsBandX(c, lo, hi)) {
                    band.curve_indices.append(gpa, @intCast(ci)) catch @panic("slughorn: oom");
                }
            }

            std.sort.pdq(
                u32,
                band.curve_indices.items,
                ExtentSort{ .curves = build.curves.items, .extent = &maxYOf },
                ExtentSort.lessThan,
            );
        }

        buildIndirection(gpa, &build.indir_x, vboundaries, num_bands_x, min_x, range_x);
    }
}

/// Fills `out` (length num_bands + 1) with band boundaries in em-space.
///
/// Interior boundaries are snapped to the indirection grid so that a boundary always falls on a
/// slot edge -- otherwise a slot could straddle two bands and the shader's O(1) lookup would pick
/// the wrong one.
///
/// The `min + snapped * range` expression here is the one that must not be contracted into an FMA;
/// see the note in build.zig. Zig's float mode is `.strict`, so it is not.
fn computeBoundaries(
    out: []Slug,
    splits: []const Slug,
    num_bands: u32,
    min_v: Slug,
    max_v: Slug,
    range: Slug,
) void {
    const isize_f: Slug = @floatFromInt(indirection_size);

    out[0] = min_v;
    out[num_bands] = max_v;

    if (splits.len != 0) {
        for (splits, 0..) |s, i| {
            const snapped = @round(s * isize_f) / isize_f;
            out[i + 1] = min_v + snapped * range;
        }
    } else {
        var i: u32 = 1;
        while (i < num_bands) : (i += 1) {
            const fi: Slug = @floatFromInt(i);
            const fn_: Slug = @floatFromInt(num_bands);
            const snapped = @round(fi / fn_ * isize_f) / isize_f;
            out[i] = min_v + snapped * range;
        }
    }
}

/// Builds an indirection table: slot q -> band index for em-fraction (q + 0.5) / indirection_size.
fn buildIndirection(
    gpa: std.mem.Allocator,
    out: *std.ArrayList(u8),
    boundaries: []const Slug,
    num_bands: u32,
    min_v: Slug,
    range: Slug,
) void {
    const isize_f: Slug = @floatFromInt(indirection_size);

    out.resize(gpa, indirection_size) catch @panic("slughorn: oom");

    for (out.items, 0..) |*slot, q| {
        const fq: Slug = @floatFromInt(q);
        const frac = (fq + 0.5) / isize_f;
        const em = min_v + frac * range;

        var band: u32 = num_bands - 1;
        var b: u32 = 0;
        while (b + 1 < num_bands) : (b += 1) {
            if (em < boundaries[b + 1]) {
                band = b;
                break;
            }
        }

        // Safe without truncation: buildShapeBands rejects num_bands > max_bands (256), so the
        // largest index here is 255. Upstream has no such guard -- see `max_bands`.
        std.debug.assert(band < max_bands);
        slot.* = @intCast(band);
    }
}

// ================================================================================================
// Tests
// ================================================================================================

const testing = std.testing;

test "curve/band intersection is inclusive at both edges" {
    const c: Curve = .{ .x1 = 0.2, .y1 = 0.2, .x2 = 0.5, .y2 = 0.5, .x3 = 0.8, .y3 = 0.8 };

    try testing.expect(curveIntersectsBandY(c, 0.0, 1.0)); // fully inside
    try testing.expect(curveIntersectsBandY(c, 0.0, 0.2)); // touches at lo edge
    try testing.expect(curveIntersectsBandY(c, 0.8, 1.0)); // touches at hi edge
    try testing.expect(!curveIntersectsBandY(c, 0.9, 1.0)); // above
    try testing.expect(!curveIntersectsBandY(c, 0.0, 0.1)); // below

    try testing.expect(curveIntersectsBandX(c, 0.0, 0.2));
    try testing.expect(!curveIntersectsBandX(c, 0.9, 1.0));
}

test "uniform boundaries snap to the indirection grid" {
    var out: [3]Slug = undefined;
    computeBoundaries(&out, &.{}, 2, 0.0, 1.0, 1.0);

    try testing.expectEqual(@as(Slug, 0.0), out[0]);
    try testing.expectEqual(@as(Slug, 1.0), out[2]);
    // 1/2 * 32 = 16, round -> 16, /32 -> 0.5
    try testing.expectEqual(@as(Slug, 0.5), out[1]);

    // 3 bands over [0,1]: 1/3*32 = 10.67 -> 11/32; 2/3*32 = 21.3 -> 21/32. Snapped, not exact
    // thirds -- which is the point: every boundary lands on a slot edge.
    var out3: [4]Slug = undefined;
    computeBoundaries(&out3, &.{}, 3, 0.0, 1.0, 1.0);
    try testing.expectEqual(@as(Slug, 11.0 / 32.0), out3[1]);
    try testing.expectEqual(@as(Slug, 21.0 / 32.0), out3[2]);
}

test "explicit splits override band count and are snapped too" {
    var out: [3]Slug = undefined;
    // 0.3 * 32 = 9.6 -> 10/32 = 0.3125
    computeBoundaries(&out, &.{0.3}, 2, 0.0, 1.0, 1.0);
    try testing.expectEqual(@as(Slug, 10.0 / 32.0), out[1]);
}

test "indirection maps slots to bands, and every slot resolves" {
    const gpa = testing.allocator;

    var table: std.ArrayList(u8) = .empty;
    defer table.deinit(gpa);

    const boundaries = [_]Slug{ 0.0, 0.5, 1.0 };
    buildIndirection(gpa, &table, &boundaries, 2, 0.0, 1.0);

    try testing.expectEqual(@as(usize, indirection_size), table.items.len);
    // Lower half -> band 0, upper half -> band 1.
    try testing.expectEqual(@as(u8, 0), table.items[0]);
    try testing.expectEqual(@as(u8, 0), table.items[15]);
    try testing.expectEqual(@as(u8, 1), table.items[16]);
    try testing.expectEqual(@as(u8, 1), table.items[indirection_size - 1]);
}

test "band sort is descending by extent, with ties broken by index" {
    const gpa = testing.allocator;

    var b: ShapeBuild = .{};
    defer b.deinit(gpa);

    // Curves 0 and 2 tie on max-x (1.0); curve 1 is larger (2.0).
    b.curves.append(gpa, .{ .x1 = 0, .y1 = 0, .x2 = 0.5, .y2 = 0.1, .x3 = 1, .y3 = 0 }) catch @panic("slughorn: oom");
    b.curves.append(gpa, .{ .x1 = 0, .y1 = 0, .x2 = 1.0, .y2 = 0.1, .x3 = 2, .y3 = 0 }) catch @panic("slughorn: oom");
    b.curves.append(gpa, .{ .x1 = 0, .y1 = 0, .x2 = 0.5, .y2 = 0.1, .x3 = 1, .y3 = 0 }) catch @panic("slughorn: oom");

    try buildShapeBands(gpa, &b, 1, 1, false, .default);

    try testing.expectEqual(@as(usize, 1), b.hbands.items.len);
    const order = b.hbands.items[0].curve_indices.items;
    try testing.expectEqualSlices(u32, &.{ 1, 0, 2 }, order);
}

test "auto band count is curves/2, clamped to [1, 16]" {
    const gpa = testing.allocator;

    // 1 curve -> max(1, 0) = 1 band.
    {
        var b: ShapeBuild = .{};
        defer b.deinit(gpa);
        b.curves.append(gpa, .{ .x1 = 0, .y1 = 0, .x2 = 0.5, .y2 = 1, .x3 = 1, .y3 = 0 }) catch @panic("slughorn: oom");
        try buildShapeBands(gpa, &b, 0, 0, false, .default);
        try testing.expectEqual(@as(usize, 1), b.hbands.items.len);
        try testing.expectEqual(@as(u32, 0), b.metrics.band_max_y);
    }

    // 100 curves -> min(16, 50) = 16 bands.
    {
        var b: ShapeBuild = .{};
        defer b.deinit(gpa);
        for (0..100) |i| {
            const t: Slug = @as(Slug, @floatFromInt(i)) / 100.0;
            b.curves.append(gpa, .{ .x1 = 0, .y1 = t, .x2 = 0.5, .y2 = t, .x3 = 1, .y3 = t }) catch @panic("slughorn: oom");
        }
        try buildShapeBands(gpa, &b, 0, 0, false, .default);
        try testing.expectEqual(@as(usize, 16), b.hbands.items.len);
        try testing.expectEqual(@as(u32, 15), b.metrics.band_max_y);
    }
}

test "auto metrics come from the curve bounding box" {
    const gpa = testing.allocator;

    var b: ShapeBuild = .{};
    defer b.deinit(gpa);
    b.curves.append(gpa, .{ .x1 = 0.25, .y1 = 0.5, .x2 = 0.5, .y2 = 0.75, .x3 = 0.75, .y3 = 0.5 }) catch @panic("slughorn: oom");

    try buildShapeBands(gpa, &b, 1, 1, false, .default);

    try testing.expectEqual(@as(Slug, 0.5), b.metrics.width); // 0.75 - 0.25
    try testing.expectEqual(@as(Slug, 0.25), b.metrics.height); // 0.75 - 0.5
    try testing.expectEqual(@as(Slug, 0.25), b.metrics.bearing_x); // min x
    try testing.expectEqual(@as(Slug, 0.75), b.metrics.bearing_y); // max y, Y-up
    try testing.expectEqual(@as(Slug, 0.5), b.metrics.advance);
}

test "more than 256 bands is rejected instead of silently truncating" {
    const gpa = testing.allocator;

    var b: ShapeBuild = .{};
    defer b.deinit(gpa);
    b.curves.append(gpa, .{ .x1 = 0, .y1 = 0, .x2 = 0.5, .y2 = 1, .x3 = 1, .y3 = 0 }) catch @panic("slughorn: oom");

    // 256 bands is the most a u8 index can address, so it must be accepted ...
    try buildShapeBands(gpa, &b, 256, 1, false, .default);
    try testing.expectEqual(@as(u32, 255), b.metrics.band_max_x);

    // ... and 257 rejected. Upstream truncates here and reads the wrong band at runtime.
    var b2: ShapeBuild = .{};
    defer b2.deinit(gpa);
    b2.curves.append(gpa, .{ .x1 = 0, .y1 = 0, .x2 = 0.5, .y2 = 1, .x3 = 1, .y3 = 0 }) catch @panic("slughorn: oom");
    try testing.expectError(error.TooManyBands, buildShapeBands(gpa, &b2, 257, 1, false, .default));
}

test "empty shape produces no bands" {
    const gpa = testing.allocator;

    var b: ShapeBuild = .{};
    defer b.deinit(gpa);

    try buildShapeBands(gpa, &b, 0, 0, false, .default);
    try testing.expectEqual(@as(usize, 0), b.hbands.items.len);
    try testing.expectEqual(@as(usize, 0), b.indir_y.items.len);
}

test "degenerate (zero-extent) range is clamped, not divided by zero" {
    const gpa = testing.allocator;

    var b: ShapeBuild = .{};
    defer b.deinit(gpa);
    // A horizontal line: zero y-range.
    b.curves.append(gpa, .{ .x1 = 0, .y1 = 0.5, .x2 = 0.5, .y2 = 0.5, .x3 = 1, .y3 = 0.5 }) catch @panic("slughorn: oom");

    try buildShapeBands(gpa, &b, 1, 1, false, .default);

    // range_y clamps to 1e-6, so the scale is finite rather than inf.
    try testing.expect(std.math.isFinite(b.metrics.band_scale_y));
    try testing.expectEqual(@as(Slug, @as(Slug, indirection_size) / 1e-6), b.metrics.band_scale_y);
}
