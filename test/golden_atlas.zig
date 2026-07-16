// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! The M1 gate: the packed textures must match the upstream C++ byte-for-byte.
//!
//! Tier A only -- one shape per case. With a single shape, iteration order is trivially
//! deterministic, so byte equality is a meaningful assertion. Multi-shape cases cannot be compared
//! this way (upstream's packing order comes from a `std::unordered_map`); see DIVERGENCE.md and
//! `golden_multi.zig`.
//!
//! Bit-exact rather than epsilon, deliberately:
//!   * the band texture is pure integers;
//!   * the curve texture is caller floats copied through with no arithmetic (slughorn.cpp:1348);
//!   * band scale/offset are simple arithmetic, exact once the C++ is built with
//!     -ffp-contract=off (which `zig build fixtures` does -- see DIVERGENCE.md).
//! Any difference is a port bug, and an epsilon would only hide it.

const std = @import("std");
const slughorn = @import("slughorn");
const fixture = @import("fixture.zig");

const testing = std.testing;

/// How a case's shape is described to the atlas. Mirrors the dumper's `Case`.
const Case = struct {
    name: []const u8,
    tex_width: u32,
    key: slughorn.Key,
    info: slughorn.ShapeInfo,
    /// What the Zig port must do. Null means "succeed and match bytes".
    ///
    /// Where this is non-null but the fixture records no C++ throw, the divergence is deliberate
    /// and documented in DIVERGENCE.md.
    expect_error: ?anyerror = null,
};

fn cv(x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) slughorn.Curve {
    return .{ .x1 = x1, .y1 = y1, .x2 = x2, .y2 = y2, .x3 = x3, .y3 = y3 };
}

/// Mirrors the dumper's `manyCurves` exactly, including the distinct extents that keep the band
/// sort's ordering total. See the note there.
fn manyCurves(gpa: std.mem.Allocator, n: u32) []slughorn.Curve {
    const out = gpa.alloc(slughorn.Curve, n) catch @panic("oom");
    for (out, 0..) |*c, i| {
        const t: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n));
        c.* = cv(0.0, t, 0.25 + t * 0.1, t + 0.001, 0.5 + t * 0.5, t);
    }
    return out;
}

fn triangle() []const slughorn.Curve {
    return &.{
        cv(0.0, 0.0, 0.25, 0.5, 0.5, 1.0),
        cv(0.5, 1.0, 0.75, 0.5, 1.0, 0.0),
        cv(1.0, 0.0, 0.5, 0.0, 0.0, 0.0),
    };
}

/// Runs one case and compares against its fixture.
fn check(case: Case) !void {
    const gpa = testing.allocator;

    const path = try std.fmt.allocPrint(gpa, "fixtures/{s}.slgf", .{case.name});
    defer gpa.free(path);

    var fx = try fixture.load(gpa, testing.io, path);
    defer fx.deinit();

    try testing.expectEqual(case.tex_width, fx.tex_width);

    var atlas = try slughorn.Atlas.init(gpa, case.tex_width);
    defer atlas.deinit();

    var diag: slughorn.Diagnostics = .{};
    atlas.diag = &diag;

    try atlas.addShape(case.key, case.info);

    if (case.expect_error) |want| {
        try testing.expectError(want, atlas.build());
        return;
    }

    // The fixture must agree that the C++ succeeded; otherwise the case table is out of date.
    try testing.expect(!fx.threw);
    try atlas.build();

    try testing.expectEqual(@as(usize, 1), fx.shapes.len);
    const want_shape = fx.shapes[0];
    const got_shape = atlas.getShape(case.key) orelse return error.MissingShape;

    // -- per-shape metadata ----------------------------------------------------------------------
    try testing.expectEqual(want_shape.band_tex_x, got_shape.band_tex_x);
    try testing.expectEqual(want_shape.band_tex_y, got_shape.band_tex_y);
    try testing.expectEqual(want_shape.band_max_x, got_shape.band_max_x);
    try testing.expectEqual(want_shape.band_max_y, got_shape.band_max_y);

    try expectBits("band_scale_x", want_shape.band_scale_x, got_shape.band_scale_x);
    try expectBits("band_scale_y", want_shape.band_scale_y, got_shape.band_scale_y);
    try expectBits("band_offset_x", want_shape.band_offset_x, got_shape.band_offset_x);
    try expectBits("band_offset_y", want_shape.band_offset_y, got_shape.band_offset_y);
    try expectBits("bearing_x", want_shape.bearing_x, got_shape.bearing_x);
    try expectBits("bearing_y", want_shape.bearing_y, got_shape.bearing_y);
    try expectBits("width", want_shape.width, got_shape.width);
    try expectBits("height", want_shape.height, got_shape.height);
    try expectBits("advance", want_shape.advance, got_shape.advance);
    try expectBits("origin_x", want_shape.origin_x, got_shape.origin_x);
    try expectBits("origin_y", want_shape.origin_y, got_shape.origin_y);

    // -- packing stats ---------------------------------------------------------------------------
    const want_stats = fx.stats;
    const got_stats = atlas.getPackingStats();
    try testing.expectEqual(want_stats.curve_texels_used, got_stats.curve_texels_used);
    try testing.expectEqual(want_stats.curve_texels_padding, got_stats.curve_texels_padding);
    try testing.expectEqual(want_stats.curve_texels_total, got_stats.curve_texels_total);
    try testing.expectEqual(want_stats.band_texels_used, got_stats.band_texels_used);
    try testing.expectEqual(want_stats.band_texels_padding, got_stats.band_texels_padding);
    try testing.expectEqual(want_stats.band_texels_total, got_stats.band_texels_total);
    try testing.expectEqual(want_stats.band_max_count, got_stats.band_max_count);
    try testing.expectEqual(want_stats.band_max_offset, got_stats.band_max_offset);

    // -- the textures themselves -----------------------------------------------------------------
    try expectTexture(case.name, "curve", fx.curve_texture, atlas.getCurveTextureData().*);
    try expectTexture(case.name, "band", fx.band_texture, atlas.getBandTextureData().*);
}

fn expectBits(what: []const u8, want: f32, got: f32) !void {
    if (@as(u32, @bitCast(want)) != @as(u32, @bitCast(got))) {
        std.debug.print("\n{s}: C++ {d} (0x{X:0>8}) != Zig {d} (0x{X:0>8})\n", .{
            what, want, @as(u32, @bitCast(want)), got, @as(u32, @bitCast(got)),
        });
        return error.FieldMismatch;
    }
}

fn expectTexture(
    case_name: []const u8,
    which: []const u8,
    want: fixture.Texture,
    got: slughorn.TextureData,
) !void {
    try testing.expectEqual(want.width, got.width);
    try testing.expectEqual(want.height, got.height);
    try testing.expectEqual(want.depth, got.depth);
    try testing.expectEqual(want.format, @intFromEnum(got.format));
    try testing.expectEqual(want.bytes.len, got.bytes.len);
    // Guard against a vacuous pass.
    try testing.expect(want.bytes.len > 0);

    if (std.mem.eql(u8, want.bytes, got.bytes)) return;

    // Report the first divergence with its texel coordinates -- a raw slice diff of a 512-wide
    // RGBA32F texture is unreadable.
    const texel_size: usize = if (want.format == 0) 16 else 8;
    for (want.bytes, got.bytes, 0..) |w, g, i| {
        if (w == g) continue;
        const texel = i / texel_size;
        std.debug.print(
            "\n{s}: {s} texture differs at byte {d} (texel {d} = x:{d} y:{d}, byte {d} of texel): C++ 0x{X:0>2} != Zig 0x{X:0>2}\n",
            .{ case_name, which, i, texel, texel % want.width, texel / want.width, i % texel_size, w, g },
        );
        break;
    }
    return error.TextureMismatch;
}

// ================================================================================================
// Cases -- mirrors tools/dump/slughorn_dump.cpp
// ================================================================================================

test "golden: single_curve" {
    try check(.{
        .name = "single_curve",
        .tex_width = 512,
        .key = .{ .codepoint = 'A' },
        .info = .{ .curves = &.{cv(0, 0, 0.5, 1, 1, 0)} },
    });
}

test "golden: empty_shape" {
    try check(.{
        .name = "empty_shape",
        .tex_width = 512,
        .key = .{ .codepoint = 'E' },
        .info = .{ .curves = &.{} },
    });
}

test "golden: triangle" {
    try check(.{
        .name = "triangle",
        .tex_width = 512,
        .key = .{ .name = "triangle" },
        .info = .{ .curves = triangle() },
    });
}

test "golden: band_738_w512 -- the row-fit regression" {
    const gpa = testing.allocator;
    const curves = manyCurves(gpa, 738);
    defer gpa.free(curves);

    try check(.{
        .name = "band_738_w512",
        .tex_width = 512,
        .key = .{ .name = "band738" },
        .info = .{ .curves = curves, .num_bands_x = 1, .num_bands_y = 1 },
        .expect_error = error.BandExceedsTextureRow,
    });
}

test "golden: band_738_w1024 -- the same shape, wider texture" {
    const gpa = testing.allocator;
    const curves = manyCurves(gpa, 738);
    defer gpa.free(curves);

    try check(.{
        .name = "band_738_w1024",
        .tex_width = 1024,
        .key = .{ .name = "band738" },
        .info = .{ .curves = curves, .num_bands_x = 1, .num_bands_y = 1 },
    });
}

test "golden: band_eq_texwidth -- exactly one row fits" {
    const gpa = testing.allocator;
    const curves = manyCurves(gpa, 512);
    defer gpa.free(curves);

    try check(.{
        .name = "band_eq_texwidth",
        .tex_width = 512,
        .key = .{ .name = "eq" },
        .info = .{ .curves = curves, .num_bands_x = 1, .num_bands_y = 1 },
    });
}

test "golden: band_texwidth_plus1 -- one more does not" {
    const gpa = testing.allocator;
    const curves = manyCurves(gpa, 513);
    defer gpa.free(curves);

    try check(.{
        .name = "band_texwidth_plus1",
        .tex_width = 512,
        .key = .{ .name = "plus1" },
        .info = .{ .curves = curves, .num_bands_x = 1, .num_bands_y = 1 },
        .expect_error = error.BandExceedsTextureRow,
    });
}

test "golden: ties_in_sort" {
    const gpa = testing.allocator;
    const curves = try gpa.alloc(slughorn.Curve, 8);
    defer gpa.free(curves);
    for (curves, 0..) |*c, i| {
        const y: f32 = @as(f32, @floatFromInt(i)) * 0.1;
        c.* = cv(0.0, y, 0.5, y + 0.05, 1.0, y);
    }

    try check(.{
        .name = "ties_in_sort",
        .tex_width = 512,
        .key = .{ .name = "ties" },
        .info = .{ .curves = curves, .num_bands_x = 1, .num_bands_y = 1 },
    });
}

test "golden: boundary_exact -- the FP-contraction canary" {
    try check(.{
        .name = "boundary_exact",
        .tex_width = 512,
        .key = .{ .name = "boundary" },
        .info = .{
            .curves = &.{
                cv(0.1, 0.1, 0.3, 1.0 / 3.0, 0.7, 0.5123),
                cv(0.7, 0.5123, 0.9, 1.0 / 7.0, 0.983, 0.017),
                cv(0.983, 0.017, 0.4, 0.06, 0.1, 0.1),
            },
            .num_bands_x = 3,
            .num_bands_y = 3,
        },
    });
}

test "golden: degenerate_collinear" {
    try check(.{
        .name = "degenerate_collinear",
        .tex_width = 512,
        .key = .{ .name = "degen" },
        .info = .{ .curves = &.{
            cv(0.0, 0.5, 0.5, 0.5, 1.0, 0.5),
            cv(1.0, 0.5, 0.5, 0.5, 0.0, 0.5),
        } },
    });
}
