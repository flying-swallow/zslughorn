// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! Procedural drawing front-end: a Path2D-style API that builds a `CompositeShape`.
//!
//! Ported from `slughorn/canvas.hpp` (the core authoring + fill path). A `Path` records subpaths
//! through a `CurveDecomposer`; a `Canvas` owns a `Path`, a current transform, and a growing
//! `CompositeShape`, and each `fill`/`defineShape` commit normalizes the drawn curves to shape-local
//! em-space, registers them with the `Atlas`, and (for `fill`) pushes a `Layer`.
//!
//! Pure Zig, no external dependency -- but a *backend* like the others: it imports `slughorn`, never
//! the reverse.
//!
//! Scope: path geometry (moveTo/lineTo/quadTo/bezierTo/rect/roundedRect/circle/ellipse/arc/close),
//! the transform stack (translate/scale/rotate/save/restore), and flat-color fill / defineShape into
//! a CompositeShape. Deferred (upstream has them, no consumer here yet): stroking, text, gradients
//! in-canvas, arcTo, arc-length sampling, masks, and MSDF.

const std = @import("std");
const slughorn = @import("slughorn");

const Slug = slughorn.Slug;
const Curve = slughorn.Curve;
const Color = slughorn.Color;
const Matrix = slughorn.Matrix;
const Atlas = slughorn.Atlas;
const Key = slughorn.Key;
const ShapeInfo = slughorn.ShapeInfo;
const CurveDecomposer = slughorn.CurveDecomposer;
const CompositeShape = slughorn.CompositeShape;

/// The circle-quadrant cubic control-arm factor: 4/3 * tan(pi/8). Ported from canvas.hpp.
const kappa: Slug = 0.5522847498307936;

/// A drawable path: subpaths recorded as em-space quadratics, plus a current transform.
///
/// The internal `CurveDecomposer` borrows `&active`, so a `Path` must be used through a stable
/// pointer (which `Canvas` guarantees) and not copied by value after the first authoring call.
pub const Path = struct {
    /// The open subpath the decomposer is currently writing into.
    active: std.ArrayList(Curve) = .empty,
    /// Closed subpaths accumulate here.
    pending: std.ArrayList(Curve) = .empty,
    decomposer: CurveDecomposer = undefined,
    bound: bool = false,
    /// Current point in *authoring* space (pre-transform), for arc join decisions.
    pen_x: Slug = 0,
    pen_y: Slug = 0,
    ctm: Matrix = .identity,
    ctm_stack: std.ArrayList(Matrix) = .empty,

    pub fn deinit(self: *Path, gpa: std.mem.Allocator) void {
        self.active.deinit(gpa);
        self.pending.deinit(gpa);
        self.ctm_stack.deinit(gpa);
        self.* = .{};
    }

    fn ensureBound(self: *Path, gpa: std.mem.Allocator) void {
        if (!self.bound) {
            self.decomposer = CurveDecomposer.init(gpa, &self.active);
            self.bound = true;
        }
    }

    fn xf(self: *const Path, x: Slug, y: Slug) [2]Slug {
        const p = self.ctm.apply(.{ x, y });
        return .{ p[0], p[1] };
    }

    // -- transform stack -------------------------------------------------------------------------

    pub fn save(self: *Path, gpa: std.mem.Allocator) void {
        self.ctm_stack.append(gpa, self.ctm) catch @panic("slughorn: oom");
    }

    pub fn restore(self: *Path) void {
        if (self.ctm_stack.pop()) |m| self.ctm = m;
    }

    /// Composes `m` so it applies before the current transform (subsequent draws are transformed by
    /// `m` first, then the existing CTM) -- matching HTML canvas `translate`/`scale`/`rotate`.
    fn compose(self: *Path, m: Matrix) void {
        self.ctm = m.mul(self.ctm);
    }

    pub fn translate(self: *Path, tx: Slug, ty: Slug) void {
        self.compose(Matrix.translation(tx, ty));
    }
    pub fn scale(self: *Path, sx: Slug, sy: Slug) void {
        self.compose(Matrix.scaling(sx, sy));
    }
    pub fn rotate(self: *Path, radians: Slug) void {
        self.compose(Matrix.rotation(radians));
    }
    pub fn resetTransform(self: *Path) void {
        self.ctm = .identity;
    }

    // -- path authoring --------------------------------------------------------------------------

    pub fn moveTo(self: *Path, gpa: std.mem.Allocator, x: Slug, y: Slug) void {
        self.ensureBound(gpa);
        const p = self.xf(x, y);
        _ = self.decomposer.moveTo(p[0], p[1]);
        self.pen_x = x;
        self.pen_y = y;
    }

    pub fn lineTo(self: *Path, gpa: std.mem.Allocator, x: Slug, y: Slug) void {
        self.ensureBound(gpa);
        const p = self.xf(x, y);
        _ = self.decomposer.lineTo(p[0], p[1]);
        self.pen_x = x;
        self.pen_y = y;
    }

    pub fn quadTo(self: *Path, gpa: std.mem.Allocator, cx: Slug, cy: Slug, x: Slug, y: Slug) void {
        self.ensureBound(gpa);
        const cp = self.xf(cx, cy);
        const ep = self.xf(x, y);
        _ = self.decomposer.quadTo(cp[0], cp[1], ep[0], ep[1]);
        self.pen_x = x;
        self.pen_y = y;
    }

    pub fn bezierTo(self: *Path, gpa: std.mem.Allocator, c1x: Slug, c1y: Slug, c2x: Slug, c2y: Slug, x: Slug, y: Slug) void {
        self.ensureBound(gpa);
        const a = self.xf(c1x, c1y);
        const b = self.xf(c2x, c2y);
        const ep = self.xf(x, y);
        _ = self.decomposer.cubicTo(a[0], a[1], b[0], b[1], ep[0], ep[1]);
        self.pen_x = x;
        self.pen_y = y;
    }

    /// Closes the current subpath and moves it to `pending`.
    pub fn closePath(self: *Path, gpa: std.mem.Allocator) void {
        self.ensureBound(gpa);
        _ = self.decomposer.close();
        self.pending.appendSlice(gpa, self.active.items) catch @panic("slughorn: oom");
        self.active.clearRetainingCapacity();
    }

    pub fn rect(self: *Path, gpa: std.mem.Allocator, x: Slug, y: Slug, w: Slug, h: Slug) void {
        self.moveTo(gpa, x, y);
        self.lineTo(gpa, x + w, y);
        self.lineTo(gpa, x + w, y + h);
        self.lineTo(gpa, x, y + h);
        self.closePath(gpa);
    }

    pub fn roundedRect(self: *Path, gpa: std.mem.Allocator, x: Slug, y: Slug, w: Slug, h: Slug, r: Slug) void {
        const kr = kappa * r;
        self.moveTo(gpa, x + r, y);
        self.lineTo(gpa, x + w - r, y);
        self.bezierTo(gpa, x + w - r + kr, y, x + w, y + r - kr, x + w, y + r);
        self.lineTo(gpa, x + w, y + h - r);
        self.bezierTo(gpa, x + w, y + h - r + kr, x + w - r + kr, y + h, x + w - r, y + h);
        self.lineTo(gpa, x + r, y + h);
        self.bezierTo(gpa, x + r - kr, y + h, x, y + h - r + kr, x, y + h - r);
        self.lineTo(gpa, x, y + r);
        self.bezierTo(gpa, x, y + r - kr, x + r - kr, y, x + r, y);
        self.closePath(gpa);
    }

    pub fn ellipse(self: *Path, gpa: std.mem.Allocator, cx: Slug, cy: Slug, rx: Slug, ry: Slug) void {
        const kx = kappa * rx;
        const ky = kappa * ry;
        self.moveTo(gpa, cx + rx, cy);
        self.bezierTo(gpa, cx + rx, cy + ky, cx + kx, cy + ry, cx, cy + ry);
        self.bezierTo(gpa, cx - kx, cy + ry, cx - rx, cy + ky, cx - rx, cy);
        self.bezierTo(gpa, cx - rx, cy - ky, cx - kx, cy - ry, cx, cy - ry);
        self.bezierTo(gpa, cx + kx, cy - ry, cx + rx, cy - ky, cx + rx, cy);
        self.closePath(gpa);
    }

    pub fn circle(self: *Path, gpa: std.mem.Allocator, cx: Slug, cy: Slug, r: Slug) void {
        self.ellipse(gpa, cx, cy, r, r);
    }

    /// Circular arc, `start_angle` to `end_angle` (radians). `ccw` selects the sweep direction.
    /// Ported from `arc` + `_arcSegments` (canvas.hpp:399, :698): the sweep is split into <=90 deg
    /// cubic segments with the tangent-based control arm.
    pub fn arc(self: *Path, gpa: std.mem.Allocator, cx: Slug, cy: Slug, r: Slug, start_angle: Slug, end_angle: Slug, ccw: bool) void {
        var sweep = end_angle - start_angle;
        const tau = 2.0 * std.math.pi;
        if (ccw and sweep > 0) sweep -= tau;
        if (!ccw and sweep < 0) sweep += tau;
        self.arcSegments(gpa, cx, cy, r, start_angle, sweep);
    }

    fn arcSegments(self: *Path, gpa: std.mem.Allocator, cx: Slug, cy: Slug, r: Slug, start_angle: Slug, sweep: Slug) void {
        if (r <= 0 or sweep == 0) return;
        const abs_sweep = @abs(sweep);
        const half_pi = std.math.pi / 2.0;
        const n_segs: usize = @max(1, @as(usize, @intFromFloat(@ceil(abs_sweep / half_pi))));
        const seg_sweep = sweep / @as(Slug, @floatFromInt(n_segs));
        var angle = start_angle;
        var i: usize = 0;
        while (i < n_segs) : (i += 1) {
            const a0 = angle;
            const a1 = angle + seg_sweep;
            const k = (4.0 / 3.0) * @tan(seg_sweep * 0.25);
            const cos0 = @cos(a0);
            const sin0 = @sin(a0);
            const cos1 = @cos(a1);
            const sin1 = @sin(a1);
            const p0x = cx + r * cos0;
            const p0y = cy + r * sin0;
            const p3x = cx + r * cos1;
            const p3y = cy + r * sin1;
            const p1x = p0x - k * r * sin0;
            const p1y = p0y + k * r * cos0;
            const p2x = p3x + k * r * sin1;
            const p2y = p3y - k * r * cos1;
            if (i == 0 and self.active.items.len == 0 and self.pending.items.len == 0) {
                self.moveTo(gpa, p0x, p0y);
            } else if (i == 0) {
                const dx = p0x - self.pen_x;
                const dy = p0y - self.pen_y;
                if (dx * dx + dy * dy > 1e-10) self.lineTo(gpa, p0x, p0y);
            }
            self.bezierTo(gpa, p1x, p1y, p2x, p2y, p3x, p3y);
            angle = a1;
        }
    }

    /// Clears both curve buffers and the pen (HTML `beginPath`). Leaves the transform untouched.
    pub fn clear(self: *Path, gpa: std.mem.Allocator) void {
        _ = gpa;
        self.active.clearRetainingCapacity();
        self.pending.clearRetainingCapacity();
        self.pen_x = 0;
        self.pen_y = 0;
    }
};

/// A stateful `CompositeShape` builder. Draw a path, then commit it with `fill`/`defineShape`;
/// `finalize` hands back the accumulated composite.
pub const Canvas = struct {
    gpa: std.mem.Allocator,
    atlas: *Atlas,
    path: Path = .{},
    composite: CompositeShape = .{},
    auto_key: u21 = 0,

    pub fn init(gpa: std.mem.Allocator, atlas: *Atlas) Canvas {
        return .{ .gpa = gpa, .atlas = atlas };
    }

    pub fn deinit(self: *Canvas) void {
        self.path.deinit(self.gpa);
        self.composite.deinit(self.gpa);
        self.* = undefined;
    }

    // -- authoring forwarding --------------------------------------------------------------------

    pub fn beginPath(self: *Canvas) void {
        self.path.clear(self.gpa);
    }
    pub fn moveTo(self: *Canvas, x: Slug, y: Slug) void {
        self.path.moveTo(self.gpa, x, y);
    }
    pub fn lineTo(self: *Canvas, x: Slug, y: Slug) void {
        self.path.lineTo(self.gpa, x, y);
    }
    pub fn quadTo(self: *Canvas, cx: Slug, cy: Slug, x: Slug, y: Slug) void {
        self.path.quadTo(self.gpa, cx, cy, x, y);
    }
    pub fn bezierTo(self: *Canvas, c1x: Slug, c1y: Slug, c2x: Slug, c2y: Slug, x: Slug, y: Slug) void {
        self.path.bezierTo(self.gpa, c1x, c1y, c2x, c2y, x, y);
    }
    pub fn closePath(self: *Canvas) void {
        self.path.closePath(self.gpa);
    }
    pub fn rect(self: *Canvas, x: Slug, y: Slug, w: Slug, h: Slug) void {
        self.path.rect(self.gpa, x, y, w, h);
    }
    pub fn roundedRect(self: *Canvas, x: Slug, y: Slug, w: Slug, h: Slug, r: Slug) void {
        self.path.roundedRect(self.gpa, x, y, w, h, r);
    }
    pub fn circle(self: *Canvas, cx: Slug, cy: Slug, r: Slug) void {
        self.path.circle(self.gpa, cx, cy, r);
    }
    pub fn ellipse(self: *Canvas, cx: Slug, cy: Slug, rx: Slug, ry: Slug) void {
        self.path.ellipse(self.gpa, cx, cy, rx, ry);
    }
    pub fn arc(self: *Canvas, cx: Slug, cy: Slug, r: Slug, a0: Slug, a1: Slug, ccw: bool) void {
        self.path.arc(self.gpa, cx, cy, r, a0, a1, ccw);
    }
    pub fn save(self: *Canvas) void {
        self.path.save(self.gpa);
    }
    pub fn restore(self: *Canvas) void {
        self.path.restore();
    }
    pub fn translate(self: *Canvas, tx: Slug, ty: Slug) void {
        self.path.translate(tx, ty);
    }
    pub fn scaleXform(self: *Canvas, sx: Slug, sy: Slug) void {
        self.path.scale(sx, sy);
    }
    pub fn rotate(self: *Canvas, radians: Slug) void {
        self.path.rotate(radians);
    }

    // -- commits ---------------------------------------------------------------------------------

    /// Commits the current path as a filled shape and pushes a `Layer` painted `color`. `scale` maps
    /// authoring units into em-space (1.0 if you drew in em-space directly). Returns the shape's key.
    pub fn fill(self: *Canvas, color: Color, scale: Slug) !Key {
        const key: Key = .{ .codepoint = self.auto_key };
        self.auto_key += 1;
        try self.commit(key, scale, color, true);
        return key;
    }

    /// Registers the current path's geometry under `key` without pushing a `Layer` (a shared or
    /// masked shape). Ported from `defineShape` (canvas.hpp:1050).
    pub fn defineShape(self: *Canvas, key: Key, scale: Slug) !void {
        try self.commit(key, scale, undefined, false);
    }

    fn commit(self: *Canvas, key: Key, scale: Slug, color: Color, push_layer: bool) !void {
        // Gather every subpath's curves (closed ones in `pending`, plus any still-open `active`).
        var all: std.ArrayList(Curve) = .empty;
        defer all.deinit(self.gpa);
        all.appendSlice(self.gpa, self.path.pending.items) catch @panic("slughorn: oom");
        all.appendSlice(self.gpa, self.path.active.items) catch @panic("slughorn: oom");
        if (all.items.len == 0) return;

        // Scale authoring units to em-space, then find the bbox.
        var min_x: Slug = std.math.floatMax(Slug);
        var min_y: Slug = std.math.floatMax(Slug);
        var max_x: Slug = -std.math.floatMax(Slug);
        var max_y: Slug = -std.math.floatMax(Slug);
        for (all.items) |*c| {
            c.x1 *= scale;
            c.y1 *= scale;
            c.x2 *= scale;
            c.y2 *= scale;
            c.x3 *= scale;
            c.y3 *= scale;
            min_x = @min(min_x, @min(c.x1, @min(c.x2, c.x3)));
            max_x = @max(max_x, @max(c.x1, @max(c.x2, c.x3)));
            min_y = @min(min_y, @min(c.y1, @min(c.y2, c.y3)));
            max_y = @max(max_y, @max(c.y1, @max(c.y2, c.y3)));
        }

        // Rebase to shape-local coords (bbox corner at the origin); the placement offset becomes the
        // Layer transform. auto_metrics lets the atlas derive metrics from the tight bbox.
        for (all.items) |*c| {
            c.x1 -= min_x;
            c.y1 -= min_y;
            c.x2 -= min_x;
            c.y2 -= min_y;
            c.x3 -= min_x;
            c.y3 -= min_y;
        }

        try self.atlas.addShape(key, .{ .curves = all.items, .auto_metrics = true });

        if (push_layer) {
            self.composite.layers.append(self.gpa, .{
                .key = key,
                .color = color,
                .transform = .{ .x = min_x, .y = min_y },
            }) catch @panic("slughorn: oom");
        }

        self.path.clear(self.gpa);
    }

    /// Number of layers committed so far.
    pub fn layerCount(self: *const Canvas) usize {
        return self.composite.layers.items.len;
    }

    /// Hands back the accumulated composite and resets the builder for the next one. The caller owns
    /// the returned `CompositeShape` and must `deinit` it.
    pub fn finalize(self: *Canvas) CompositeShape {
        const out = self.composite;
        self.composite = .{};
        return out;
    }
};
