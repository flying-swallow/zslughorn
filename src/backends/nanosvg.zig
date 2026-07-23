// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! SVG front-end: NanoSVG paths -> slughorn curves.
//!
//! Ported from `slughorn/nanosvg.hpp`. This is a *backend* module in the sense `decompose.zig`
//! means it -- an input adapter that funnels into `CurveDecomposer`, the universal entry point.
//! It imports `slughorn`, never the reverse: nanosvg is C and needs libc, and that must not reach
//! the core.
//!
//! SVG coordinates are normalized to [0, 1] em-space against the image *width* (`scale = 1/width`,
//! nanosvg.hpp:17), unconditionally. To recover authoring-space positions, multiply a `Transform`
//! by the image's width/height.
//!
//! Scope: filled shapes' path geometry (including the evenodd -> nonzero winding conversion) and the
//! `loadImage` compositing frontend -- an SVG becomes a `CompositeShape` of `Layer`s with flat-color
//! and linear/radial gradient paints. Not ported: strokes (upstream skips them too), the per-shape
//! `ShapeRule` policy config, and `Mask` (upstream's `loadImage` never sets one). See DIVERGENCE.md
//! for the radial-gradient `units` divergence.

const std = @import("std");
const slughorn = @import("slughorn");
const nanosvg = @import("nanosvg");

/// Raw translate-c'd nanosvg. Exposed so callers can reach fields this wrapper does not surface.
pub const c = nanosvg.c;

const Slug = slughorn.Slug;
const Curve = slughorn.Curve;
const Color = slughorn.Color;
const Matrix = slughorn.Matrix;
const Origin = slughorn.Origin;
const ShapeInfo = slughorn.ShapeInfo;
const Atlas = slughorn.Atlas;
const Key = slughorn.Key;
const CurveDecomposer = slughorn.CurveDecomposer;
const GradientInfo = slughorn.GradientInfo;
const GradientStop = slughorn.GradientStop;
const Layer = slughorn.Layer;
const CompositeShape = slughorn.CompositeShape;

/// Upstream hardcodes "px" at both call sites (nanosvg.hpp:677, :704).
const units = "px";

/// Where a decomposed shape sits in the source SVG's space, in em units.
///
/// Re-exported from the core (`slughorn.Transform`, `types.zig`): upstream keeps this in the core
/// because `Layer` carries one, so it now lives there rather than in this backend. Kept as a
/// backend alias so existing call sites (`decomposePath`/`loadShape`) stay unchanged.
pub const Transform = slughorn.Transform;

/// A parsed SVG document. Owns the underlying nanosvg allocation.
pub const Image = struct {
    handle: *c.NSVGimage,

    pub fn deinit(self: Image) void {
        c.nsvgDelete(self.handle);
    }

    pub fn width(self: Image) Slug {
        return self.handle.width;
    }

    pub fn height(self: Image) Slug {
        return self.handle.height;
    }

    /// The em-normalization factor, `1 / width`.
    ///
    /// Null when the image has no positive width, which upstream treats as unnormalizable and
    /// refuses to load (nanosvg.hpp:450-454). A NaN width also lands here, since the comparison is
    /// false either way.
    pub fn scale(self: Image) ?Slug {
        if (!(self.handle.width > 0)) return null;
        return 1.0 / self.handle.width;
    }

    /// Canvas height in em units (`height / width`), for `loadShape`'s fixed-metrics path.
    pub fn heightEm(self: Image) ?Slug {
        const s = self.scale() orelse return null;
        return self.handle.height * s;
    }

    /// Iterates every shape in document order, hidden ones included; filter with `isVisible`.
    pub fn shapes(self: Image) ShapeIterator {
        return .{ .cursor = self.handle.shapes };
    }
};

pub const ShapeIterator = struct {
    cursor: ?*c.NSVGshape,

    pub fn next(self: *ShapeIterator) ?*const c.NSVGshape {
        const shape = self.cursor orelse return null;
        self.cursor = shape.next;
        return shape;
    }
};

/// Coerces one of nanosvg's enum constants to the width of the struct field that stores it.
///
/// The C enums translate to `c_int`/`c_uint`, but the fields holding them are declared `char` /
/// `unsigned char`, so the two never compare directly.
inline fn enumAs(comptime Field: type, comptime value: anytype) Field {
    return @intCast(value);
}

/// Whether the shape's `display` resolved to visible (nanosvg.hpp:468).
pub fn isVisible(shape: *const c.NSVGshape) bool {
    return (shape.flags & enumAs(@TypeOf(shape.flags), c.NSVG_FLAGS_VISIBLE)) != 0;
}

/// Whether the shape is filled at all. Unfilled shapes decompose to curves that would render as
/// nothing (nanosvg.hpp:476 and its neighbours).
pub fn isFilled(shape: *const c.NSVGshape) bool {
    return shape.fill.type != enumAs(@TypeOf(shape.fill.type), c.NSVG_PAINT_NONE);
}

/// Parses SVG source held in memory.
///
/// There is no parse error, because NanoSVG has no notion of invalid input: anything unparseable
/// yields an *empty* image with zero width rather than a failure. `nsvgParse` returns null in
/// exactly one case -- its own parser allocation failing (nanosvg.h:3033-3035) -- which is OOM, and
/// this project panics on OOM rather than reporting it (see the note in `slughorn.zig`). So callers asking "was that
/// really an SVG?" should test `scale()`, which is null precisely when the image is unusable.
///
/// The text is duplicated rather than borrowed: `nsvgParse` needs a NUL terminator and *mutates*
/// the buffer it is handed. The caller's slice is left untouched.
pub fn parseFromMemory(gpa: std.mem.Allocator, svg_text: []const u8, dpi: Slug) Image {
    const buf = gpa.dupeSentinel(u8, svg_text, 0) catch @panic("slughorn: oom");
    defer gpa.free(buf);

    const parsed: ?*c.NSVGimage = c.nsvgParse(buf.ptr, units, dpi);
    const handle: std.mem.Allocator.Error!*c.NSVGimage = parsed orelse error.OutOfMemory;
    return .{ .handle = handle catch @panic("slughorn: oom") };
}

/// Parses an SVG file.
///
/// Reads through Zig rather than calling `nsvgParseFromFile`, which would route the read through C
/// stdio and collapse every I/O failure into the same null return that means OOM.
pub fn parseFromFile(
    gpa: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    dpi: Slug,
) !Image {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
    defer gpa.free(bytes);
    return parseFromMemory(gpa, bytes, dpi);
}

/// One shape's geometry, decomposed into em-space quadratics.
pub const Decomposed = struct {
    curves: std.ArrayList(Curve),
    origin: Origin,
    transform: Transform,

    pub fn deinit(self: *Decomposed, gpa: std.mem.Allocator) void {
        self.curves.deinit(gpa);
    }

    /// Borrows `curves`, so it must not outlive this `Decomposed`. The borrow is why this is a
    /// method rather than a stored field: upstream's `ShapeInfo` owns its curve vector
    /// (`info.curves = std::move(curves)`, nanosvg.hpp:410), where ours points at someone else's
    /// storage and `Atlas.addShape` copies what it keeps (types.zig:208).
    pub fn shapeInfo(self: *const Decomposed) ShapeInfo {
        return .{ .curves = self.curves.items, .origin = self.origin };
    }
};

/// Horizontal rightward ray cast from (px, py) against `curves[0..end]`, returning the crossing
/// count; odd means the point is inside.
///
/// Ported from nanosvg.hpp:304. Each curve is approximated by its chord (x1,y1)->(x3,y3) -- exact
/// for polygons, close enough for smooth shapes.
fn rayCrossings(curves: []const Curve, end: usize, px: Slug, py: Slug) usize {
    var count: usize = 0;
    for (curves[0..end]) |curve| {
        // The chord must straddle the ray's y-level.
        if ((curve.y1 <= py) == (curve.y3 <= py)) continue;

        const t = (py - curve.y1) / (curve.y3 - curve.y1);
        const xi = curve.x1 + t * (curve.x3 - curve.x1);

        if (xi > px) count += 1;
    }
    return count;
}

/// Decomposes one `NSVGshape` into em-space quadratics.
///
/// Returns null when the shape has an empty bounding box or yields no curves. Ported from
/// nanosvg.hpp:323. Caller owns the result.
pub fn decomposePath(
    gpa: std.mem.Allocator,
    shape: *const c.NSVGshape,
    scale: Slug,
    origin: Origin,
    auto_metrics: bool,
) ?Decomposed {
    const min_x = shape.bounds[0] * scale;
    const min_y = shape.bounds[1] * scale;
    const max_x = shape.bounds[2] * scale;
    const max_y = shape.bounds[3] * scale;

    if (max_x <= min_x or max_y <= min_y) return null;

    // With auto-metrics the shape is re-based to its own bbox corner, so bands are tight and no
    // offset is wasted; otherwise it stays in canvas space.
    const off_x: Slug = if (auto_metrics) min_x else 0;
    const off_y: Slug = if (auto_metrics) min_y else 0;

    var curves: std.ArrayList(Curve) = .empty;
    var decomposer = CurveDecomposer.init(gpa, &curves);

    const evenodd = shape.fillRule == enumAs(@TypeOf(shape.fillRule), c.NSVG_FILLRULE_EVENODD);

    // NanoSVG *prepends* each NSVGpath as it parses, so `shape.paths` runs in reverse document
    // order (last sub-path first). Reverse it back: the ray-cast containment test below only works
    // if a sub-path's enclosing outer contour was accumulated before it (nanosvg.hpp:341-347).
    var paths: std.ArrayList(*const c.NSVGpath) = .empty;
    defer paths.deinit(gpa);

    var cursor: ?*const c.NSVGpath = shape.paths;
    while (cursor) |path| : (cursor = path.next) {
        paths.append(gpa, path) catch @panic("slughorn: oom");
    }
    std.mem.reverse(*const c.NSVGpath, paths.items);

    for (paths.items) |path| {
        // Defensive: nanosvg drops short paths itself (`if (p->npts < 4) return;`, nanosvg.h:1051),
        // so this should be unreachable. A cubic needs 4 points.
        if (path.npts < 4) continue;

        const pts = path.pts;
        const start_x = pts[0] * scale - off_x;
        const start_y = pts[1] * scale - off_y;

        const subpath_start = decomposer.mark();
        _ = decomposer.moveTo(start_x, start_y);

        // `pts` is a flat cubic chain -- x0,y0, [c1x,c1y, c2x,c2y, x1,y1], ... -- so each step
        // consumes 3 points and the window slides by 3 (6 floats).
        //
        // The bound is `i + 3 < npts`, where upstream writes `i < npts - 1` (nanosvg.hpp:360).
        // The two agree exactly for every path nanosvg can actually produce, since it emits
        // npts = 1 + 3k ("Expect 1 + N*3 points", nanosvg.h:1057). They differ only if npts is not
        // 1 mod 3, where upstream's bound admits a final iteration that reads up to 4 floats past
        // the end of `pts`. Same output, no out-of-bounds read on malformed input.
        var i: usize = 0;
        const npts: usize = @intCast(path.npts);
        while (i + 3 < npts) : (i += 3) {
            const p = pts + i * 2;
            _ = decomposer.cubicTo(
                p[2] * scale - off_x,
                p[3] * scale - off_y,
                p[4] * scale - off_x,
                p[5] * scale - off_y,
                p[6] * scale - off_x,
                p[7] * scale - off_y,
            );
        }

        if (path.closed != 0) _ = decomposer.close();

        // Evenodd -> nonzero: if this sub-path starts inside an odd number of the sub-paths already
        // accumulated, flip its winding so the nonzero shader cuts the hole. CPU-only, baked into
        // the atlas at load time -- zero GPU cost (nanosvg.hpp:372-379).
        if (evenodd and subpath_start > 0) {
            if (rayCrossings(curves.items, subpath_start, start_x, start_y) % 2 != 0) {
                _ = decomposer.reverseFrom(subpath_start);
            }
        }
    }

    if (curves.items.len == 0) {
        curves.deinit(gpa);
        return null;
    }

    // The origin *data* stored with the shape, in local em-space (nanosvg.hpp:384-393).
    var info_origin = origin;
    switch (origin.type) {
        .pivot => {
            info_origin.x = origin.x * scale - off_x;
            info_origin.y = origin.y * scale - off_y;
        },
        .custom => {
            info_origin.x = origin.x * scale;
            info_origin.y = origin.y * scale;
        },
        .default, .centered => {},
    }

    // Where the caller must place the shape to reconstruct the SVG's layout (nanosvg.hpp:395-404).
    // Without auto-metrics the shape already carries canvas-space coordinates, so it needs no
    // placement offset at all.
    const transform: Transform = if (!auto_metrics) .{} else switch (origin.type) {
        .centered => .{ .x = (min_x + max_x) * 0.5, .y = (min_y + max_y) * 0.5 },
        .pivot => .{ .x = origin.x * scale, .y = origin.y * scale },
        .custom => .{ .x = min_x + origin.x * scale, .y = min_y + origin.y * scale },
        .default => .{ .x = min_x, .y = min_y },
    };

    return .{ .curves = curves, .origin = info_origin, .transform = transform };
}

/// Decomposes one `NSVGshape` and registers it with `atlas` under `key`.
///
/// Returns null when the shape decomposed to nothing; otherwise the transform that places it.
/// Ported from nanosvg.hpp:413.
pub fn loadShape(
    gpa: std.mem.Allocator,
    atlas: *Atlas,
    shape: *const c.NSVGshape,
    key: Key,
    scale: Slug,
    origin: Origin,
    auto_metrics: bool,
    canvas_height_em: Slug,
) !?Transform {
    var decomposed = decomposePath(gpa, shape, scale, origin, auto_metrics) orelse return null;
    defer decomposed.deinit(gpa);

    var info = decomposed.shapeInfo();
    info.auto_metrics = auto_metrics;

    if (!auto_metrics) {
        // Declare the whole SVG viewport as the band extent, so buildShapeBands calibrates over the
        // full canvas rather than the tight curve bbox. Falls back to a unit square when the caller
        // supplied no height (nanosvg.hpp:421-431).
        const h: Slug = if (canvas_height_em > 0) canvas_height_em else 1;
        info.bearing_x = 0;
        info.bearing_y = h;
        info.width = 1;
        info.height = h;
    }

    try atlas.addShape(key, info);
    return decomposed.transform;
}

/// nanosvg packs colors as 0xAABBGGRR (byte order R, G, B, A). Ported from `colorFromNSVG`,
/// nanosvg.hpp:156.
fn colorFromNSVG(abgr: u32) Color {
    return .{
        @as(Slug, @floatFromInt(abgr & 0xFF)) / 255.0,
        @as(Slug, @floatFromInt((abgr >> 8) & 0xFF)) / 255.0,
        @as(Slug, @floatFromInt((abgr >> 16) & 0xFF)) / 255.0,
        @as(Slug, @floatFromInt((abgr >> 24) & 0xFF)) / 255.0,
    };
}

/// 2x3 affine inverse, matching nanosvg's internal `nsvg__xformInverse` (which is behind
/// `NANOSVG_IMPLEMENTATION` and so invisible to translate-c). f64 intermediates, as upstream.
fn xformInverse(inv: *[6]f32, t: [6]f32) void {
    const det = @as(f64, t[0]) * t[3] - @as(f64, t[2]) * t[1];
    if (det > -1e-6 and det < 1e-6) {
        inv.* = .{ 1, 0, 0, 1, 0, 0 };
        return;
    }
    const invdet = 1.0 / det;
    inv[0] = @floatCast(@as(f64, t[3]) * invdet);
    inv[1] = @floatCast(-@as(f64, t[1]) * invdet);
    inv[2] = @floatCast(-@as(f64, t[2]) * invdet);
    inv[3] = @floatCast(@as(f64, t[0]) * invdet);
    inv[4] = @floatCast((@as(f64, t[2]) * t[5] - @as(f64, t[3]) * t[4]) * invdet);
    inv[5] = @floatCast((@as(f64, t[1]) * t[4] - @as(f64, t[0]) * t[5]) * invdet);
}

/// Applies a 2x3 affine to a point. Matches nanosvg's internal `nsvg__xformPoint`.
fn xformPoint(t: [6]f32, x: f32, y: f32) [2]f32 {
    return .{ x * t[0] + y * t[2] + t[4], x * t[1] + y * t[3] + t[5] };
}

/// `t = xx*emX + xy*emY + dx` projects an em point onto the gradient axis. Ported from
/// `buildLinearGradientMatrix`, slughorn.hpp:1393.
fn buildLinearGradientMatrix(x0: Slug, y0: Slug, x1: Slug, y1: Slug) Matrix {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const len_sq = dx * dx + dy * dy;
    if (len_sq < 1e-12) return Matrix.identity;
    const inv = 1.0 / len_sq;
    return .{ .xx = dx * inv, .xy = dy * inv, .dx = -(x0 * dx + y0 * dy) * inv };
}

/// Packs the gradient center into dx/dy and the 2x2 B matrix (em-delta -> normalized gradient space)
/// into the linear part. Ported from `buildAffineRadialGradientMatrix`, slughorn.hpp:1443.
fn buildAffineRadialGradientMatrix(cx: Slug, cy: Slug, b00: Slug, b01: Slug, b10: Slug, b11: Slug) Matrix {
    return .{ .xx = b00, .yx = b10, .xy = b01, .yy = b11, .dx = cx, .dy = cy };
}

/// Parses `shape`'s linear/radial fill gradient into a `GradientInfo`, registers it with `atlas`,
/// and returns the 1-based gradient id (null if the gradient is empty or degenerate).
///
/// Ported from the gradient arm of `loadImage`, nanosvg.hpp:498-618. **Divergence:** upstream applies
/// an object-bounding-box isotropic-radius correction gated on `NSVGgradient::units`, but the pinned
/// nanosvg's *public* gradient struct has no `units` field (it lives on the implementation-internal
/// `NSVGgradientData`), so that correction is dropped -- radials on non-square bounding boxes are
/// slightly off. See DIVERGENCE.md.
fn parseGradient(
    gpa: std.mem.Allocator,
    atlas: *Atlas,
    shape: *const c.NSVGshape,
    scale: Slug,
    shape_opacity: Slug,
    is_radial: bool,
) !?u32 {
    const g = shape.fill.unnamed_0.gradient; // translate-c names the anonymous paint union
    if (g == null or g.*.nstops == 0) return null;

    const nstops: usize = @intCast(g.*.nstops);
    const stops_c = @as([*]const c.NSVGgradientStop, @ptrCast(&g.*.stops))[0..nstops];

    var stops = try gpa.alloc(GradientStop, nstops);
    defer gpa.free(stops); // addGradient copies; this temporary is freed either way
    for (stops_c, 0..) |sc, i| {
        var col = colorFromNSVG(sc.color);
        col[3] = col[3] * shape_opacity;
        stops[i] = .{ .t = sc.offset, .color = col };
    }

    const min_x_em = shape.bounds[0] * scale;
    const min_y_em = shape.bounds[1] * scale;

    // nanosvg stores xform as the inverse of gradient->pixel; invert to get the forward transform.
    var fwd: [6]f32 = undefined;
    xformInverse(&fwd, g.*.xform);

    var info: GradientInfo = .{ .stops = stops };
    if (!is_radial) {
        const p0 = xformPoint(fwd, 0, 0);
        const p1 = xformPoint(fwd, 0, 1);
        info.type = .linear;
        info.transform = buildLinearGradientMatrix(
            p0[0] * scale - min_x_em,
            p0[1] * scale - min_y_em,
            p1[0] * scale - min_x_em,
            p1[1] * scale - min_y_em,
        );
    } else {
        const center = xformPoint(fwd, 0, 0);
        const det = fwd[0] * fwd[3] - fwd[2] * fwd[1];
        if (@abs(det) < 1e-10) return null; // degenerate radial
        const inv_s_det = 1.0 / (scale * det);
        var b00 = fwd[3] * inv_s_det;
        var b01 = -fwd[2] * inv_s_det;
        var b10 = -fwd[1] * inv_s_det;
        var b11 = fwd[0] * inv_s_det;
        if (b11 < 0) { // the shader convention needs b11 > 0
            b00 = -b00;
            b01 = -b01;
            b10 = -b10;
            b11 = -b11;
        }
        info.type = .affine_radial;
        info.transform = buildAffineRadialGradientMatrix(
            center[0] * scale - min_x_em,
            center[1] * scale - min_y_em,
            b00,
            b01,
            b10,
            b11,
        );
        info.inner_radius = 0;
    }
    return try atlas.addGradient(info);
}

/// Loads a whole SVG into a `CompositeShape` -- one `Layer` per visible, filled shape, in document
/// order, with its color/gradient paint and placement. Ported from `loadImage`, nanosvg.hpp:437.
///
/// Returns null when the image is unnormalizable (zero width -- see `Image.scale`). The caller owns
/// the returned `CompositeShape` and must keep `image` alive while using it: name-keyed layers
/// borrow the source shapes' id strings.
///
/// Scope vs upstream: strokes are skipped (upstream skips them too -- stroke-to-fill is unwired), and
/// the per-shape `ShapeRule`/policy config is not ported (every visible filled shape is included).
pub fn loadImage(
    gpa: std.mem.Allocator,
    atlas: *Atlas,
    image: Image,
    origin: Origin,
    auto_metrics: bool,
) !?CompositeShape {
    const scale = image.scale() orelse return null;
    const height_em = image.heightEm() orelse 1;

    var composite: CompositeShape = .{ .advance = 1 }; // normalized image width is always 1.0
    errdefer composite.deinit(gpa);

    var auto_key: u21 = 0;
    var it = image.shapes();
    while (it.next()) |shape| {
        if (!isVisible(shape)) continue;

        const ft = shape.fill.type;
        var color: Color = .{ 1, 1, 1, 1 };
        var gradient_id: u32 = 0;

        if (ft == enumAs(@TypeOf(ft), c.NSVG_PAINT_COLOR)) {
            color = colorFromNSVG(shape.fill.unnamed_0.color);
            color[3] = color[3] * shape.opacity;
            if (color[3] < 1e-4) continue; // fully transparent -> drop
        } else if (ft == enumAs(@TypeOf(ft), c.NSVG_PAINT_LINEAR_GRADIENT) or
            ft == enumAs(@TypeOf(ft), c.NSVG_PAINT_RADIAL_GRADIENT))
        {
            const is_radial = ft == enumAs(@TypeOf(ft), c.NSVG_PAINT_RADIAL_GRADIENT);
            gradient_id = (try parseGradient(gpa, atlas, shape, scale, shape.opacity, is_radial)) orelse continue;
        } else {
            continue; // NSVG_PAINT_NONE / unsupported: skip (no ForceInclude/policy support)
        }

        // Name-keyed by SVG id when present (borrows the id string), else an auto codepoint.
        const id = std.mem.sliceTo(&shape.id, 0);
        const key: Key = if (id.len > 0) .{ .name = id } else blk: {
            defer auto_key += 1;
            break :blk .{ .codepoint = auto_key };
        };

        const transform = (try loadShape(gpa, atlas, shape, key, scale, origin, auto_metrics, height_em)) orelse continue;

        composite.layers.append(gpa, .{
            .key = key,
            .color = color,
            .transform = transform,
            .gradient_id = gradient_id,
        }) catch @panic("slughorn: oom");
    }

    return composite;
}
