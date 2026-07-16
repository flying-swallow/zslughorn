// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! Step 6 gate: `render.zig` must reproduce the C++ `render.hpp` coverage output.
//!
//! This is the payoff of porting the oracle. `decode` reads the packed textures back the way the
//! shader will, so agreeing here validates `bands.zig` and `pack.zig` end-to-end -- with no GPU
//! and no reliance on the internal band structures, which are private and discarded by `build()`.
//!
//! Tolerance rather than bit-equality, deliberately: this path runs `sqrt` and divides, where
//! libstdc++ and Zig may legitimately differ by ULPs. Pinning bits would be brittle for no safety
//! gain. The bit-exact match *rate* is reported anyway, so silent drift stays visible -- the
//! textures those grids are decoded from are already compared bit-for-bit in `golden_atlas.zig`.

const std = @import("std");
const slughorn = @import("slughorn");
const fixture = @import("fixture.zig");

const testing = std.testing;
const render = slughorn.render;

/// Coverage is an alpha in [0, 1]; a difference below this is far under one 8-bit level (1/255).
const tolerance: f32 = 1e-5;

fn checkGrids(name: []const u8) !void {
    const gpa = testing.allocator;

    const path = try std.fmt.allocPrint(gpa, "fixtures/{s}.slgf", .{name});
    defer gpa.free(path);

    var fx = try fixture.load(gpa, testing.io, path);
    defer fx.deinit();

    try testing.expect(!fx.threw);
    // Guard against a vacuous pass: this case is only worth naming if it has grids.
    try testing.expect(fx.grids.len > 0);
    try testing.expectEqual(@as(usize, 1), fx.shapes.len);

    // Rebuild the atlas from the fixture's own retained curves, so this tests the render path
    // rather than re-testing shape ingestion.
    var atlas = try slughorn.Atlas.init(gpa, fx.tex_width);
    defer atlas.deinit();

    const key = fx.shapes[0].key;
    try atlas.addShape(key, .{
        .curves = fx.shapes[0].curves,
        .num_bands_x = fx.shapes[0].band_max_x + 1,
        .num_bands_y = fx.shapes[0].band_max_y + 1,
    });
    try atlas.build();

    const shape = atlas.getShape(key) orelse return error.MissingShape;

    var sampler = try render.decode(
        gpa,
        shape.*,
        atlas.getCurveTextureData(),
        atlas.getBandTextureData(),
    );
    defer sampler.deinit(gpa);

    for (fx.grids) |want| {
        var got = try sampler.renderGrid(gpa, .{
            .size_hint = @max(want.width, want.height),
            .banded = want.banded,
        });
        defer got.deinit(gpa);

        try testing.expectEqual(want.width, got.width);
        try testing.expectEqual(want.height, got.height);

        // Guard against a vacuous pass: two all-zero grids agree perfectly and prove nothing.
        var want_covered: usize = 0;
        for (want.data) |v| {
            if (v > 0.01) want_covered += 1;
        }
        if (want_covered == 0) {
            std.debug.print("\n{s}: C++ reference grid has no coverage -- the case proves nothing\n", .{name});
            return error.VacuousGrid;
        }

        var max_abs: f32 = 0;
        var sum_abs: f64 = 0;
        var exact: usize = 0;
        var worst_at: usize = 0;

        for (want.data, got.data, 0..) |w, g, i| {
            const d = @abs(w - g);
            if (d > max_abs) {
                max_abs = d;
                worst_at = i;
            }
            sum_abs += d;
            if (@as(u32, @bitCast(w)) == @as(u32, @bitCast(g))) exact += 1;
        }

        const n = want.data.len;
        const mean_abs = sum_abs / @as(f64, @floatFromInt(n));
        const exact_pct = 100.0 * @as(f64, @floatFromInt(exact)) / @as(f64, @floatFromInt(n));

        if (max_abs > tolerance) {
            std.debug.print(
                "\n{s} ({s}): coverage mismatch\n" ++
                    "  max abs err {e} at pixel {d} (x:{d} y:{d}): C++ {d} vs Zig {d}\n" ++
                    "  mean abs err {e}, bit-exact {d:.1}%\n",
                .{
                    name,                     if (want.banded) "banded" else "unbanded",
                    max_abs,                  worst_at,
                    worst_at % want.width,    worst_at / want.width,
                    want.data[worst_at],      got.data[worst_at],
                    mean_abs,                 exact_pct,
                },
            );
            return error.CoverageMismatch;
        }

        // Non-fatal, but reported: a drop here means the two implementations have started to
        // diverge numerically even though they are still within tolerance.
        if (exact_pct < 100.0) {
            std.debug.print(
                "note: {s} ({s}) within tolerance but only {d:.1}% bit-exact (max {e})\n",
                .{ name, if (want.banded) "banded" else "unbanded", exact_pct, max_abs },
            );
        }
    }
}

test "render matches C++: single_curve" {
    try checkGrids("single_curve");
}

test "render matches C++: triangle" {
    try checkGrids("triangle");
}

test "render matches C++: ties_in_sort" {
    try checkGrids("ties_in_sort");
}

test "render matches C++: boundary_exact" {
    try checkGrids("boundary_exact");
}

test "render matches C++: decompose_cubic_default" {
    try checkGrids("decompose_cubic_default");
}

test "render matches C++: decompose_cubic_fine" {
    try checkGrids("decompose_cubic_fine");
}

test "banded and unbanded agree -- which is the whole point of the bands" {
    // The band structure is an optimization: it must not change the answer. The C++ fixture
    // carries both, so this checks the property on our own output too.
    const gpa = testing.allocator;

    var fx = try fixture.load(gpa, testing.io, "fixtures/triangle.slgf");
    defer fx.deinit();

    var atlas = try slughorn.Atlas.init(gpa, fx.tex_width);
    defer atlas.deinit();

    const key = fx.shapes[0].key;
    try atlas.addShape(key, .{ .curves = fx.shapes[0].curves });
    try atlas.build();

    const shape = atlas.getShape(key) orelse return error.MissingShape;

    var sampler = try render.decode(gpa, shape.*, atlas.getCurveTextureData(), atlas.getBandTextureData());
    defer sampler.deinit(gpa);

    var banded = try sampler.renderGrid(gpa, .{ .size_hint = 32, .banded = true });
    defer banded.deinit(gpa);
    var plain = try sampler.renderGrid(gpa, .{ .size_hint = 32, .banded = false });
    defer plain.deinit(gpa);

    var covered: usize = 0;
    for (banded.data, plain.data) |b, p| {
        try testing.expectApproxEqAbs(p, b, 1e-5);
        if (b > 0.5) covered += 1;
    }

    // Guard against a vacuous pass: an all-zero grid would satisfy the loop above trivially.
    try testing.expect(covered > 0);
}

test "decode round-trips the packed textures back to the original curves" {
    // decode() is the inverse of pack(); if it recovers the curves we put in, the packed bytes
    // really do mean what the shader will read.
    const gpa = testing.allocator;

    var atlas = try slughorn.Atlas.init(gpa, 512);
    defer atlas.deinit();

    const curves = [_]slughorn.Curve{
        .{ .x1 = 0.0, .y1 = 0.0, .x2 = 0.25, .y2 = 0.5, .x3 = 0.5, .y3 = 1.0 },
        .{ .x1 = 0.5, .y1 = 1.0, .x2 = 0.75, .y2 = 0.5, .x3 = 1.0, .y3 = 0.0 },
        .{ .x1 = 1.0, .y1 = 0.0, .x2 = 0.5, .y2 = 0.0, .x3 = 0.0, .y3 = 0.0 },
    };

    try atlas.addShape(.{ .name = "tri" }, .{ .curves = &curves });
    try atlas.build();

    const shape = atlas.getShape(.{ .name = "tri" }) orelse return error.MissingShape;

    var sampler = try render.decode(gpa, shape.*, atlas.getCurveTextureData(), atlas.getBandTextureData());
    defer sampler.deinit(gpa);

    try testing.expectEqual(curves.len, sampler.curves.len);
    // Curves come back bit-exact: the curve texture stores them verbatim.
    for (curves) |want| {
        var found = false;
        for (sampler.curves) |got| {
            if (std.mem.eql(u8, std.mem.asBytes(&want), std.mem.asBytes(&got))) found = true;
        }
        try testing.expect(found);
    }
}
