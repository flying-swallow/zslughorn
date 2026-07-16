// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! Step 3 gate: the Zig `CurveDecomposer` must reproduce the C++ one bit-for-bit.
//!
//! `Atlas::Shape` retains the em-space curves it was handed (slughorn.hpp:771), so each fixture's
//! `shape.curves` *is* the C++ decomposer's output for the same path. We replay the path here and
//! compare.
//!
//! Bit-exact, not epsilon: the decomposer's arithmetic is midpoint averaging and small-integer
//! multiplies on f32, with no transcendentals and no reductions. There is no legitimate source of
//! divergence, so any difference is a port bug, and an epsilon would only hide it.

const std = @import("std");
const slughorn = @import("slughorn");
const fixture = @import("fixture.zig");

const Curve = slughorn.Curve;
const testing = std.testing;

/// Replays a path against a Zig `CurveDecomposer` and asserts the result matches the named
/// fixture's retained curves bit-for-bit.
fn expectMatches(
    name: []const u8,
    tolerance: slughorn.Slug,
    comptime path: fn (d: *slughorn.CurveDecomposer) void,
) !void {
    const gpa = testing.allocator;

    var fx = try fixture.load(gpa, testing.io, name);
    defer fx.deinit();

    try testing.expect(!fx.threw);
    try testing.expectEqual(@as(usize, 1), fx.shapes.len);
    const expected = fx.shapes[0].curves;
    // Guard against a vacuous pass: an empty fixture would make the comparison loop a no-op.
    try testing.expect(expected.len > 0);

    var curves: std.ArrayList(Curve) = .empty;
    defer curves.deinit(gpa);

    var d = slughorn.CurveDecomposer.init(gpa, &curves);
    d.tolerance = tolerance;
    path(&d);

    try testing.expectEqual(expected.len, curves.items.len);

    for (expected, curves.items, 0..) |e, got, i| {
        // Compare the bit patterns: expectEqual on f32 would treat -0.0 and 0.0 as equal, and
        // would say nothing useful about NaN.
        if (!std.mem.eql(u8, std.mem.asBytes(&e), std.mem.asBytes(&got))) {
            std.debug.print(
                "\n{s}: curve {d} differs\n  C++: {f}\n  Zig: {f}\n",
                .{ name, i, e, got },
            );
            return error.CurveMismatch;
        }
    }
}

test "decomposer matches C++: default tolerance (two quads per cubic)" {
    try expectMatches("fixtures/decompose_cubic_default.slgf", slughorn.tolerance_exact, struct {
        fn p(d: *slughorn.CurveDecomposer) void {
            _ = d.moveTo(0, 0)
                .cubicTo(0, 1, 1, 1, 1, 0)
                .cubicTo(1, -0.4, 0.13, -0.77, 0, 0)
                .close();
        }
    }.p);
}

test "decomposer matches C++: fine tolerance (adaptive subdivision)" {
    try expectMatches("fixtures/decompose_cubic_fine.slgf", slughorn.tolerance_fine, struct {
        fn p(d: *slughorn.CurveDecomposer) void {
            _ = d.moveTo(0, 0)
                .cubicTo(0, 1, 1, 1, 1, 0)
                .cubicTo(1, -0.4, 0.13, -0.77, 0, 0)
                .close();
        }
    }.p);
}

test "decomposer matches C++: max-depth termination" {
    try expectMatches("fixtures/decompose_max_depth.slgf", 0, struct {
        fn p(d: *slughorn.CurveDecomposer) void {
            _ = d.moveTo(0, 0).cubicTo(0, 1, 1, 1, 1, 0);
        }
    }.p);
}

test "decomposer matches C++: lines, close, and reversed winding" {
    try expectMatches("fixtures/decompose_lines_reverse.slgf", slughorn.tolerance_exact, struct {
        fn p(d: *slughorn.CurveDecomposer) void {
            _ = d.moveTo(0, 0).lineTo(1, 0).lineTo(1, 1).close();
            const m = d.mark();
            _ = d.moveTo(0.25, 0.25).lineTo(0.5, 0.25).lineTo(0.5, 0.5).close();
            _ = d.reverseFrom(m);
        }
    }.p);
}
