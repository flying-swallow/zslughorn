// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! Step 7: Tier-B -- multi-shape cases, compared semantically rather than byte-for-byte.
//!
//! Why not bytes: upstream iterates a `std::unordered_map` to drive packing (slughorn.cpp:1173,
//! :1194, :1244, :1337) while band texels store *absolute* curve-texture coordinates
//! (slughorn.cpp:1471-1479). A shape's packed block therefore encodes where previously-iterated
//! shapes landed, so the whole layout is a function of libstdc++'s bucket order. Asserting byte
//! equality here would be asserting a property of libstdc++'s hash table, not of slughorn -- and
//! it would break on libc++ without anything being wrong.
//!
//! What is actually required is that each shape *means* the same thing: same curves, same metrics,
//! same coverage. `decode` reads the packed bytes back the way the shader will, so comparing
//! decoded output tests exactly that, independent of where the blocks landed.
//!
//! See DIVERGENCE.md.

const std = @import("std");
const slughorn = @import("slughorn");
const fixture = @import("fixture.zig");

const testing = std.testing;
const render = slughorn.render;

test "multi_shape_3: every shape decodes to the same geometry the C++ produced" {
    const gpa = testing.allocator;

    var fx = try fixture.load(gpa, testing.io, "fixtures/multi_shape_3.slgf");
    defer fx.deinit();

    try testing.expect(!fx.threw);
    try testing.expectEqual(@as(usize, 3), fx.shapes.len);

    var atlas = try slughorn.Atlas.init(gpa, fx.tex_width);
    defer atlas.deinit();

    // Insertion order here is ours to choose; the point of this test is that it does not matter.
    for (fx.shapes) |s| {
        try atlas.addShape(s.key, .{ .curves = s.curves });
    }
    try atlas.build();

    for (fx.shapes) |want| {
        const got = atlas.getShape(want.key) orelse {
            std.debug.print("\nmissing shape {f}\n", .{want.key});
            return error.MissingShape;
        };

        // Metrics are intrinsic to the shape, so these must match exactly even though the block
        // *locations* legitimately differ.
        try expectBits("width", want.width, got.width);
        try expectBits("height", want.height, got.height);
        try expectBits("bearing_x", want.bearing_x, got.bearing_x);
        try expectBits("bearing_y", want.bearing_y, got.bearing_y);
        try expectBits("advance", want.advance, got.advance);
        try expectBits("band_scale_x", want.band_scale_x, got.band_scale_x);
        try expectBits("band_scale_y", want.band_scale_y, got.band_scale_y);
        try expectBits("band_offset_x", want.band_offset_x, got.band_offset_x);
        try expectBits("band_offset_y", want.band_offset_y, got.band_offset_y);

        // Band *counts* are intrinsic too ...
        try testing.expectEqual(want.band_max_x, got.band_max_x);
        try testing.expectEqual(want.band_max_y, got.band_max_y);

        // ... while band *locations* are not, and are deliberately not asserted:
        //   want.band_tex_x / want.band_tex_y

        // The retained curves survive verbatim.
        try testing.expectEqual(want.curves.len, got.curves.len);
        try testing.expectEqualSlices(
            u8,
            std.mem.sliceAsBytes(want.curves),
            std.mem.sliceAsBytes(got.curves),
        );
    }
}

test "multi_shape_3: decoding recovers each shape's curves through the packed textures" {
    // The stronger claim: not just that the Shape structs agree, but that the *bytes we packed*
    // decode back to the right curves for every shape -- i.e. each block is internally consistent
    // wherever it happens to have landed.
    const gpa = testing.allocator;

    var fx = try fixture.load(gpa, testing.io, "fixtures/multi_shape_3.slgf");
    defer fx.deinit();

    var atlas = try slughorn.Atlas.init(gpa, fx.tex_width);
    defer atlas.deinit();

    // Insert in reverse order: if packing order leaked into the decoded meaning, this would break.
    var i = fx.shapes.len;
    while (i > 0) {
        i -= 1;
        try atlas.addShape(fx.shapes[i].key, .{ .curves = fx.shapes[i].curves });
    }
    try atlas.build();

    for (fx.shapes) |want| {
        const shape = atlas.getShape(want.key) orelse return error.MissingShape;

        var sampler = try render.decode(
            gpa,
            shape.*,
            atlas.getCurveTextureData(),
            atlas.getBandTextureData(),
        );
        defer sampler.deinit(gpa);

        try testing.expectEqual(want.curves.len, sampler.curves.len);

        // decode() returns curves in a canonical (sorted-unique) order rather than input order, so
        // compare as sets.
        for (want.curves) |wc| {
            var found = false;
            for (sampler.curves) |gc| {
                if (std.mem.eql(u8, std.mem.asBytes(&wc), std.mem.asBytes(&gc))) found = true;
            }
            if (!found) {
                std.debug.print("\n{f}: curve {f} did not survive the round trip\n", .{ want.key, wc });
                return error.CurveMissing;
            }
        }
    }
}

test "insertion order changes the layout but not the meaning" {
    // Makes the Tier A/B split concrete: the same shapes inserted in two orders produce different
    // *bytes* (which is why multi-shape byte comparison is meaningless) but identical *coverage*
    // (which is why the port is still correct).
    const gpa = testing.allocator;

    const tri = [_]slughorn.Curve{
        .{ .x1 = 0.0, .y1 = 0.0, .x2 = 0.25, .y2 = 0.5, .x3 = 0.5, .y3 = 1.0 },
        .{ .x1 = 0.5, .y1 = 1.0, .x2 = 0.75, .y2 = 0.5, .x3 = 1.0, .y3 = 0.0 },
        .{ .x1 = 1.0, .y1 = 0.0, .x2 = 0.5, .y2 = 0.0, .x3 = 0.0, .y3 = 0.0 },
    };
    const arc = [_]slughorn.Curve{
        .{ .x1 = 0.0, .y1 = 0.0, .x2 = 0.5, .y2 = 0.9, .x3 = 1.0, .y3 = 0.0 },
        .{ .x1 = 1.0, .y1 = 0.0, .x2 = 0.5, .y2 = 0.1, .x3 = 0.0, .y3 = 0.0 },
    };

    const forward = try buildTwo(gpa, .{ .name = "tri" }, &tri, .{ .name = "arc" }, &arc);
    var f_atlas = forward;
    defer f_atlas.deinit();

    const reverse = try buildTwo(gpa, .{ .name = "arc" }, &arc, .{ .name = "tri" }, &tri);
    var r_atlas = reverse;
    defer r_atlas.deinit();

    // The bytes differ -- the blocks landed in different places.
    try testing.expect(!std.mem.eql(
        u8,
        f_atlas.getBandTextureData().bytes,
        r_atlas.getBandTextureData().bytes,
    ));

    // The meaning does not.
    for ([_]slughorn.Key{ .{ .name = "tri" }, .{ .name = "arc" } }) |key| {
        var a = try renderOne(gpa, &f_atlas, key);
        defer a.deinit(gpa);
        var b = try renderOne(gpa, &r_atlas, key);
        defer b.deinit(gpa);

        try testing.expectEqual(a.width, b.width);
        var covered: usize = 0;
        for (a.data, b.data) |x, y| {
            try testing.expectEqual(x, y);
            if (x > 0.5) covered += 1;
        }
        // Not vacuous: there is real coverage to agree about.
        try testing.expect(covered > 0);
    }
}

fn buildTwo(
    gpa: std.mem.Allocator,
    k1: slughorn.Key,
    c1: []const slughorn.Curve,
    k2: slughorn.Key,
    c2: []const slughorn.Curve,
) !slughorn.Atlas {
    var a = try slughorn.Atlas.init(gpa, 512);
    errdefer a.deinit();
    try a.addShape(k1, .{ .curves = c1 });
    try a.addShape(k2, .{ .curves = c2 });
    try a.build();
    return a;
}

fn renderOne(gpa: std.mem.Allocator, atlas: *const slughorn.Atlas, key: slughorn.Key) !render.Grid {
    const shape = atlas.getShape(key) orelse return error.MissingShape;
    var sampler = try render.decode(
        gpa,
        shape.*,
        atlas.getCurveTextureData(),
        atlas.getBandTextureData(),
    );
    defer sampler.deinit(gpa);
    return sampler.renderGrid(gpa, .{ .size_hint = 32 });
}

fn expectBits(what: []const u8, want: f32, got: f32) !void {
    if (@as(u32, @bitCast(want)) != @as(u32, @bitCast(got))) {
        std.debug.print("\n{s}: C++ {d} != Zig {d}\n", .{ what, want, got });
        return error.FieldMismatch;
    }
}
