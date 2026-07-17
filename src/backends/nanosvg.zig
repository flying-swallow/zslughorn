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
//! Scope: filled shapes' path geometry, including the evenodd -> nonzero winding conversion.
//! Gradients, strokes, and the `CompositeShape`/`Layer` compositing API (`loadImage` upstream) are
//! not ported -- they depend on core types that do not exist here yet.

const std = @import("std");
const slughorn = @import("slughorn");
const nanosvg = @import("nanosvg");

/// Raw translate-c'd nanosvg. Exposed so callers can reach fields this wrapper does not surface.
pub const c = nanosvg.c;

const oom = slughorn.oom;
const Slug = slughorn.Slug;
const Curve = slughorn.Curve;
const Origin = slughorn.Origin;
const ShapeInfo = slughorn.ShapeInfo;
const Atlas = slughorn.Atlas;
const Key = slughorn.Key;
const CurveDecomposer = slughorn.CurveDecomposer;

/// Upstream hardcodes "px" at both call sites (nanosvg.hpp:677, :704).
const units = "px";

/// Where a decomposed shape sits in the source SVG's space, in em units.
///
/// Ported from `Transform`, slughorn.hpp:165. Upstream keeps this in the core because `Layer`
/// carries one; here it is backend-local until `CompositeShape`/`Layer` are ported, at which point
/// it belongs in `types.zig`.
pub const Transform = struct {
    x: Slug = 0,
    y: Slug = 0,
    z: Slug = 0,
};

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
    /// refuses to load (nanosvg.hpp:462-466). A NaN width also lands here, since the comparison is
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
/// this project panics on OOM rather than reporting it (see `oom.zig`). So callers asking "was that
/// really an SVG?" should test `scale()`, which is null precisely when the image is unusable.
///
/// The text is duplicated rather than borrowed: `nsvgParse` needs a NUL terminator and *mutates*
/// the buffer it is handed. The caller's slice is left untouched.
pub fn parseFromMemory(gpa: std.mem.Allocator, svg_text: []const u8, dpi: Slug) Image {
    const buf = oom.must(gpa.dupeSentinel(u8, svg_text, 0));
    defer gpa.free(buf);

    const parsed: ?*c.NSVGimage = c.nsvgParse(buf.ptr, units, dpi);
    const handle: std.mem.Allocator.Error!*c.NSVGimage = parsed orelse error.OutOfMemory;
    return .{ .handle = oom.must(handle) };
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
        oom.must(paths.append(gpa, path));
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
        // The bound is `i + 3 < npts`, where upstream writes `i < npts - 1` (nanosvg.hpp:363).
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
