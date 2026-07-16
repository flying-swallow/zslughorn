// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! Path -> quadratic Bezier decomposition.
//!
//! This is the universal entry point into slughorn: anything that can be reduced to quadratic
//! Beziers can be rendered. Every backend (Canvas, FreeType, NanoSVG, ...) funnels through here.
//!
//! Ported from `CurveDecomposer`, slughorn.hpp:1511.

const std = @import("std");
const oom = @import("oom.zig");
const types = @import("types.zig");

const Slug = types.Slug;
const Curve = types.Curve;

/// Flatness thresholds, in curve-space units. Defaults suit em-normalized [0, 1] geometry; scale
/// proportionally for other authoring spaces (slughorn.hpp:1496-1497).
pub const tolerance_draft: Slug = 1e-2; // fast; visible only at large sizes
pub const tolerance_balanced: Slug = 1e-3; // good default for screen work
pub const tolerance_fine: Slug = 1e-4; // high-DPI / print / export

/// The default (slughorn.hpp:1509).
///
/// Despite the name this is the *lowest* quality setting: it is `FLT_MAX`, so `tolerance^2`
/// overflows to infinity, every cubic tests as "flat enough", and each one emits exactly two
/// quadratics with no subdivision. Upstream picked this default so `cubicTo` keeps its original,
/// predictable two-leaf behaviour; lower it to opt into adaptive subdivision.
pub const tolerance_exact: Slug = std.math.floatMax(Slug);

/// Maximum subdivision recursion depth (slughorn.hpp:1602).
///
/// Bounds the work on degenerate input. At depth 8 the chord error is already under 0.4% of the
/// cubic's bounding box -- below any practical rendering threshold.
const max_depth: u32 = 8;

/// Accumulates quadratic Beziers from path commands.
///
/// Allocation failure panics (see `oom.zig`), so none of these methods return errors -- which is
/// what lets them chain the way the C++ does:
///
///     _ = d.moveTo(0, 0).lineTo(1, 0).cubicTo(1, 0.5, 0.5, 1, 0, 1).close();
pub const CurveDecomposer = struct {
    curves: *std.ArrayList(Curve),
    gpa: std.mem.Allocator,

    tolerance: Slug = tolerance_exact,

    // Current point, and the start of the current subpath (for `close`).
    x: Slug = 0,
    y: Slug = 0,
    sx: Slug = 0,
    sy: Slug = 0,

    pub fn init(gpa: std.mem.Allocator, curves: *std.ArrayList(Curve)) CurveDecomposer {
        return .{ .curves = curves, .gpa = gpa };
    }

    pub fn moveTo(self: *CurveDecomposer, x: Slug, y: Slug) *CurveDecomposer {
        self.x = x;
        self.sx = x;
        self.y = y;
        self.sy = y;
        return self;
    }

    /// A line is a quadratic whose control point is the chord midpoint.
    pub fn lineTo(self: *CurveDecomposer, x3: Slug, y3: Slug) *CurveDecomposer {
        self.push(.{
            .x1 = self.x,
            .y1 = self.y,
            .x2 = (self.x + x3) * 0.5,
            .y2 = (self.y + y3) * 0.5,
            .x3 = x3,
            .y3 = y3,
        });
        self.x = x3;
        self.y = y3;
        return self;
    }

    pub fn quadTo(self: *CurveDecomposer, cx: Slug, cy: Slug, x3: Slug, y3: Slug) *CurveDecomposer {
        self.push(.{ .x1 = self.x, .y1 = self.y, .x2 = cx, .y2 = cy, .x3 = x3, .y3 = y3 });
        self.x = x3;
        self.y = y3;
        return self;
    }

    /// Adaptive cubic -> quadratic decomposition via De Casteljau subdivision.
    pub fn cubicTo(
        self: *CurveDecomposer,
        c1x: Slug,
        c1y: Slug,
        c2x: Slug,
        c2y: Slug,
        x3: Slug,
        y3: Slug,
    ) *CurveDecomposer {
        self.cubicAdaptive(self.x, self.y, c1x, c1y, c2x, c2y, x3, y3, 0);
        self.x = x3;
        self.y = y3;
        return self;
    }

    /// Closes the current subpath with a line back to its start. No-op if already there.
    pub fn close(self: *CurveDecomposer) *CurveDecomposer {
        const eps: Slug = 1e-6;
        if (@abs(self.x - self.sx) > eps or @abs(self.y - self.sy) > eps) {
            _ = self.lineTo(self.sx, self.sy);
        }
        return self;
    }

    /// Snapshots the current write position, for a later `reverseFrom`.
    pub fn mark(self: *const CurveDecomposer) usize {
        return self.curves.items.len;
    }

    /// Reverses the winding of everything appended since `pos`.
    ///
    /// Winding direction is how the punch-out effect is achieved: an inner contour wound opposite
    /// to its outer one is cut out of it.
    pub fn reverseFrom(self: *CurveDecomposer, pos: usize) *CurveDecomposer {
        reverseCurves(self.curves.items[pos..]);
        return self;
    }

    fn push(self: *CurveDecomposer, c: Curve) void {
        oom.must(self.curves.append(self.gpa, c));
    }

    /// Squared distance from (px, py) to the infinite line through (ax, ay) -> (bx, by).
    ///
    /// Squared throughout to keep the sqrt out of the hot path; callers compare against
    /// tolerance^2.
    fn pointToLineDistSq(px: Slug, py: Slug, ax: Slug, ay: Slug, bx: Slug, by: Slug) Slug {
        const dx = bx - ax;
        const dy = by - ay;
        const len_sq = dx * dx + dy * dy;

        if (len_sq < 1e-12) {
            // Degenerate chord: fall back to the distance to the start point.
            const ex = px - ax;
            const ey = py - ay;
            return ex * ex + ey * ey;
        }

        // |cross(b-a, a-p)| / |b-a|, squared.
        const cross = dx * (ay - py) - dy * (ax - px);
        return (cross * cross) / len_sq;
    }

    /// True when both interior control points lie within `tolerance` of the p0->p3 chord.
    fn flatEnough(
        self: *const CurveDecomposer,
        p0x: Slug,
        p0y: Slug,
        p1x: Slug,
        p1y: Slug,
        p2x: Slug,
        p2y: Slug,
        p3x: Slug,
        p3y: Slug,
    ) bool {
        // With the default tolerance_exact (FLT_MAX) this overflows to +inf, so every comparison
        // below succeeds and no subdivision ever happens. That is the intended default.
        const tol_sq = self.tolerance * self.tolerance;
        return pointToLineDistSq(p1x, p1y, p0x, p0y, p3x, p3y) <= tol_sq and
            pointToLineDistSq(p2x, p2y, p0x, p0y, p3x, p3y) <= tol_sq;
    }

    /// The leaf operation: emit a cubic as two quadratics via a midpoint split.
    fn emitTwoQuads(
        self: *CurveDecomposer,
        p0x: Slug,
        p0y: Slug,
        p1x: Slug,
        p1y: Slug,
        p2x: Slug,
        p2y: Slug,
        p3x: Slug,
        p3y: Slug,
    ) void {
        const midx = (p0x + 3.0 * p1x + 3.0 * p2x + p3x) * 0.125;
        const midy = (p0y + 3.0 * p1y + 3.0 * p2y + p3y) * 0.125;

        self.push(.{
            .x1 = p0x,
            .y1 = p0y,
            .x2 = (p0x + 3.0 * p1x) * 0.25,
            .y2 = (p0y + 3.0 * p1y) * 0.25,
            .x3 = midx,
            .y3 = midy,
        });
        self.push(.{
            .x1 = midx,
            .y1 = midy,
            .x2 = (3.0 * p2x + p3x) * 0.25,
            .y2 = (3.0 * p2y + p3y) * 0.25,
            .x3 = p3x,
            .y3 = p3y,
        });
    }

    /// Recursive De Casteljau subdivision at t = 0.5.
    fn cubicAdaptive(
        self: *CurveDecomposer,
        p0x: Slug,
        p0y: Slug,
        p1x: Slug,
        p1y: Slug,
        p2x: Slug,
        p2y: Slug,
        p3x: Slug,
        p3y: Slug,
        depth: u32,
    ) void {
        if (depth >= max_depth or self.flatEnough(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y)) {
            self.emitTwoQuads(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y);
            return;
        }

        const m01x = (p0x + p1x) * 0.5;
        const m01y = (p0y + p1y) * 0.5;
        const m12x = (p1x + p2x) * 0.5;
        const m12y = (p1y + p2y) * 0.5;
        const m23x = (p2x + p3x) * 0.5;
        const m23y = (p2y + p3y) * 0.5;

        const m012x = (m01x + m12x) * 0.5;
        const m012y = (m01y + m12y) * 0.5;
        const m123x = (m12x + m23x) * 0.5;
        const m123y = (m12y + m23y) * 0.5;

        const m0123x = (m012x + m123x) * 0.5;
        const m0123y = (m012y + m123y) * 0.5;

        self.cubicAdaptive(p0x, p0y, m01x, m01y, m012x, m012y, m0123x, m0123y, depth + 1);
        self.cubicAdaptive(m0123x, m0123y, m123x, m123y, m23x, m23y, p3x, p3y, depth + 1);
    }
};

/// Reverses the winding of `curves` in place: swap each curve's endpoints (the control point is
/// already symmetric, so it stays put), then reverse the sequence.
///
/// Ported from slughorn.hpp:1581.
pub fn reverseCurves(curves: []Curve) void {
    for (curves) |*c| {
        std.mem.swap(Slug, &c.x1, &c.x3);
        std.mem.swap(Slug, &c.y1, &c.y3);
    }
    std.mem.reverse(Curve, curves);
}

// ================================================================================================
// Tests
// ================================================================================================

const testing = std.testing;

fn expectCurveApprox(expected: Curve, actual: Curve) !void {
    const tol = 1e-6;
    try testing.expectApproxEqAbs(expected.x1, actual.x1, tol);
    try testing.expectApproxEqAbs(expected.y1, actual.y1, tol);
    try testing.expectApproxEqAbs(expected.x2, actual.x2, tol);
    try testing.expectApproxEqAbs(expected.y2, actual.y2, tol);
    try testing.expectApproxEqAbs(expected.x3, actual.x3, tol);
    try testing.expectApproxEqAbs(expected.y3, actual.y3, tol);
}

test "lineTo emits a quadratic with a midpoint control" {
    const gpa = testing.allocator;

    var curves: std.ArrayList(Curve) = .empty;
    defer curves.deinit(gpa);

    var d = CurveDecomposer.init(gpa, &curves);
    _ = d.moveTo(0, 0).lineTo(2, 4);

    try testing.expectEqual(@as(usize, 1), curves.items.len);
    try expectCurveApprox(.{ .x1 = 0, .y1 = 0, .x2 = 1, .y2 = 2, .x3 = 2, .y3 = 4 }, curves.items[0]);
}

test "quadTo passes through verbatim" {
    const gpa = testing.allocator;

    var curves: std.ArrayList(Curve) = .empty;
    defer curves.deinit(gpa);

    var d = CurveDecomposer.init(gpa, &curves);
    _ = d.moveTo(0, 0).quadTo(0.5, 1, 1, 0);

    try testing.expectEqual(@as(usize, 1), curves.items.len);
    try testing.expectEqual(Curve{ .x1 = 0, .y1 = 0, .x2 = 0.5, .y2 = 1, .x3 = 1, .y3 = 0 }, curves.items[0]);
}

test "the default tolerance emits exactly two quads per cubic, without subdividing" {
    const gpa = testing.allocator;

    var curves: std.ArrayList(Curve) = .empty;
    defer curves.deinit(gpa);

    var d = CurveDecomposer.init(gpa, &curves);
    try testing.expectEqual(tolerance_exact, d.tolerance);
    // Deliberately a very un-flat cubic: it still must not subdivide at the default tolerance.
    _ = d.moveTo(0, 0).cubicTo(0, 10, 10, 10, 10, 0);

    try testing.expectEqual(@as(usize, 2), curves.items.len);
    // Midpoint of the cubic: (0 + 3*0 + 3*10 + 10)/8 = 5, (0 + 30 + 30 + 0)/8 = 7.5
    try expectCurveApprox(.{ .x1 = 0, .y1 = 0, .x2 = 0, .y2 = 7.5, .x3 = 5, .y3 = 7.5 }, curves.items[0]);
    try expectCurveApprox(.{ .x1 = 5, .y1 = 7.5, .x2 = 10, .y2 = 7.5, .x3 = 10, .y3 = 0 }, curves.items[1]);
}

test "tolerance_exact squares to infinity, which is what disables subdivision" {
    // Pin the mechanism the default depends on, so a change to Slug's width can't silently
    // re-enable subdivision by making this finite.
    try testing.expect(std.math.isInf(tolerance_exact * tolerance_exact));
}

test "a lower tolerance subdivides, and stays within max_depth" {
    const gpa = testing.allocator;

    var curves: std.ArrayList(Curve) = .empty;
    defer curves.deinit(gpa);

    var d = CurveDecomposer.init(gpa, &curves);
    d.tolerance = tolerance_fine;
    _ = d.moveTo(0, 0).cubicTo(0, 1, 1, 1, 1, 0);

    // Subdivision happened ...
    try testing.expect(curves.items.len > 2);
    // ... and is bounded: at most 2^max_depth leaves, 2 quads each.
    try testing.expect(curves.items.len <= 2 * (@as(usize, 1) << max_depth));
    // The decomposition still spans the original endpoints.
    try expectCurveApprox(.{ .x1 = 0, .y1 = 0, .x2 = 0, .y2 = 0, .x3 = 0, .y3 = 0 }, .{
        .x1 = curves.items[0].x1,
        .y1 = curves.items[0].y1,
        .x2 = 0,
        .y2 = 0,
        .x3 = 0,
        .y3 = 0,
    });
    const last = curves.items[curves.items.len - 1];
    try testing.expectApproxEqAbs(@as(Slug, 1), last.x3, 1e-6);
    try testing.expectApproxEqAbs(@as(Slug, 0), last.y3, 1e-6);
}

test "degenerate cubic terminates at max_depth instead of recursing forever" {
    const gpa = testing.allocator;

    var curves: std.ArrayList(Curve) = .empty;
    defer curves.deinit(gpa);

    var d = CurveDecomposer.init(gpa, &curves);
    d.tolerance = 0; // nothing is ever flat enough
    _ = d.moveTo(0, 0).cubicTo(0, 1, 1, 1, 1, 0);

    // Exactly 2^max_depth leaves, 2 quads each -- the depth cap is what bounds this.
    try testing.expectEqual(@as(usize, 2 * (@as(usize, 1) << max_depth)), curves.items.len);
}

test "close draws back to the subpath start, and is a no-op when already there" {
    const gpa = testing.allocator;

    var curves: std.ArrayList(Curve) = .empty;
    defer curves.deinit(gpa);

    var d = CurveDecomposer.init(gpa, &curves);
    _ = d.moveTo(0, 0).lineTo(1, 0).lineTo(1, 1).close();
    try testing.expectEqual(@as(usize, 3), curves.items.len);
    // The closing line returns to the origin.
    try expectCurveApprox(.{ .x1 = 1, .y1 = 1, .x2 = 0.5, .y2 = 0.5, .x3 = 0, .y3 = 0 }, curves.items[2]);

    // Closing again adds nothing: we are already at the start.
    _ = d.close();
    try testing.expectEqual(@as(usize, 3), curves.items.len);
}

test "reverseFrom flips winding of only the marked range" {
    const gpa = testing.allocator;

    var curves: std.ArrayList(Curve) = .empty;
    defer curves.deinit(gpa);

    var d = CurveDecomposer.init(gpa, &curves);
    _ = d.moveTo(0, 0).lineTo(1, 0); // outer contour: index 0

    const m = d.mark();
    try testing.expectEqual(@as(usize, 1), m);

    _ = d.moveTo(0, 0).lineTo(0, 1).lineTo(1, 1); // inner contour: indices 1, 2
    _ = d.reverseFrom(m);

    try testing.expectEqual(@as(usize, 3), curves.items.len);
    // Untouched.
    try expectCurveApprox(.{ .x1 = 0, .y1 = 0, .x2 = 0.5, .y2 = 0, .x3 = 1, .y3 = 0 }, curves.items[0]);
    // Reversed: the sequence order flipped and each curve's endpoints swapped, so what was the
    // last curve's end is now the first curve's start.
    try expectCurveApprox(.{ .x1 = 1, .y1 = 1, .x2 = 0.5, .y2 = 1, .x3 = 0, .y3 = 1 }, curves.items[1]);
    try expectCurveApprox(.{ .x1 = 0, .y1 = 1, .x2 = 0, .y2 = 0.5, .x3 = 0, .y3 = 0 }, curves.items[2]);
}

test "reverseCurves round-trips" {
    var curves = [_]Curve{
        .{ .x1 = 0, .y1 = 0, .x2 = 0.5, .y2 = 1, .x3 = 1, .y3 = 0 },
        .{ .x1 = 1, .y1 = 0, .x2 = 1.5, .y2 = -1, .x3 = 2, .y3 = 0 },
    };
    const original = curves;

    reverseCurves(&curves);
    try testing.expect(!std.mem.eql(u8, std.mem.asBytes(&original), std.mem.asBytes(&curves)));

    reverseCurves(&curves);
    try testing.expectEqualSlices(u8, std.mem.asBytes(&original), std.mem.asBytes(&curves));
}
