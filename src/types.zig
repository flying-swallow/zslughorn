// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

const std = @import("std");
const zml = @import("zml");

/// The scalar type for all geometry. `slug_t` upstream (slughorn.hpp:57).
///
/// f32, not f64, and deliberately so: it is what the GPU consumes, and what `render.zig` must use
/// to stay a faithful oracle for the shader.
pub const Slug = f32;

/// A single quadratic Bezier segment: p1 -> p2 (off-curve control) -> p3.
///
/// Coordinates are em-normalized (FreeType's FT_LOAD_NO_SCALE path divided by units_per_EM).
/// Ported from slughorn.hpp:646.
pub const Curve = extern struct {
    x1: Slug,
    y1: Slug,
    x2: Slug,
    y2: Slug,
    x3: Slug,
    y3: Slug,

    pub fn minX(c: Curve) Slug {
        return @min(c.x1, @min(c.x2, c.x3));
    }
    pub fn maxX(c: Curve) Slug {
        return @max(c.x1, @max(c.x2, c.x3));
    }
    pub fn minY(c: Curve) Slug {
        return @min(c.y1, @min(c.y2, c.y3));
    }
    pub fn maxY(c: Curve) Slug {
        return @max(c.y1, @max(c.y2, c.y3));
    }

    pub fn isFinite(c: Curve) bool {
        inline for (@typeInfo(Curve).@"struct".field_names) |name| {
            if (!std.math.isFinite(@field(c, name))) return false;
        }
        return true;
    }

    pub fn format(c: Curve, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("Curve(({d}, {d}) -> ({d}, {d}) -> ({d}, {d}))", .{
            c.x1, c.y1, c.x2, c.y2, c.x3, c.y3,
        });
    }
};

/// An RGBA color, components in [0, 1], indexed `[0]`..`[3]`. Ported from slughorn.hpp:85.
///
/// A `zml.Vec4f32` rather than a struct, because colors get *interpolated* far more than they get
/// field-accessed: a gradient stop is `zml.scalar.lerp(c0, c1, t)`, and element-wise arithmetic
/// comes for free. `zml.color`'s colour-science helpers (`srgb_to_linear`, `luminance`, ...) are
/// free functions over vectors, so they compose with this directly.
pub const Color = zml.Vec4f32;

/// Builds a `Color` from components.
pub fn rgba(r: Slug, g: Slug, b: Slug, a: Slug) Color {
    return .{ r, g, b, a };
}

/// Builds an opaque `Color`, preserving the `a = 1` default the struct form used to carry.
pub fn rgb(r: Slug, g: Slug, b: Slug) Color {
    return .{ r, g, b, 1 };
}

/// A 2D affine transform, stored as the six meaningful entries of a 3x3 matrix.
///
/// Ported from slughorn.hpp:107. Layout matches upstream's naming: `xx`/`yx` are the first
/// column, `xy`/`yy` the second, `dx`/`dy` the translation.
///
/// Kept as six floats rather than a `zml.Mat3f32`: the compact form is what M4 uploads to the GPU
/// as a gradient transform, and `zml` has no 2D affine type -- its `Mat.rotate` is gated on
/// `rows == 4 and cols == 4` and falls through to a bare `unreachable` for a 3x3 (matrix.zig:208),
/// which compiles clean and panics at runtime. Points are `zml.Vec2f32` so this still composes with
/// the rest of `zml`.
pub const Matrix = extern struct {
    xx: Slug = 1,
    yx: Slug = 0,
    xy: Slug = 0,
    yy: Slug = 1,
    dx: Slug = 0,
    dy: Slug = 0,

    pub const identity: Matrix = .{};

    pub fn translation(tx: Slug, ty: Slug) Matrix {
        return .{ .dx = tx, .dy = ty };
    }

    pub fn scaling(sx: Slug, sy: Slug) Matrix {
        return .{ .xx = sx, .yy = sy };
    }

    pub fn rotation(radians: Slug) Matrix {
        const c = @cos(radians);
        const s = @sin(radians);
        return .{ .xx = c, .yx = s, .xy = -s, .yy = c };
    }

    /// Returns `self` followed by `o` (i.e. `o * self` in column-vector convention).
    pub fn mul(self: Matrix, o: Matrix) Matrix {
        return .{
            .xx = self.xx * o.xx + self.yx * o.xy,
            .yx = self.xx * o.yx + self.yx * o.yy,
            .xy = self.xy * o.xx + self.yy * o.xy,
            .yy = self.xy * o.yx + self.yy * o.yy,
            .dx = self.dx * o.xx + self.dy * o.xy + o.dx,
            .dy = self.dx * o.yx + self.dy * o.yy + o.dy,
        };
    }

    /// Transforms a point. Components are indexed `[0]` = x, `[1]` = y.
    pub fn apply(self: Matrix, p: zml.Vec2f32) zml.Vec2f32 {
        return .{
            self.xx * p[0] + self.xy * p[1] + self.dx,
            self.yx * p[0] + self.yy * p[1] + self.dy,
        };
    }

    /// Transforms a direction, ignoring translation. For normals, tangents, and deltas.
    pub fn applyDir(self: Matrix, v: zml.Vec2f32) zml.Vec2f32 {
        return .{
            self.xx * v[0] + self.xy * v[1],
            self.yx * v[0] + self.yy * v[1],
        };
    }
};

/// A shape's placement in a scene, in em units. Ported from `Transform`, slughorn.hpp:165.
///
/// Deliberately *not* an affine: upstream replaced the old per-`Layer` `Matrix` (whose linear part
/// `computeQuad` ignored) with this position-only form. `x`/`y` place a shape via
/// `Shape.computeQuad`; `z` is an optional depth offset for 3D scenes. Gradient geometry still uses
/// `Matrix` (see `GradientInfo`), which genuinely needs the 2x2 linear part. Lives in core because
/// `Layer` (M3/compositing) carries one; the NanoSVG backend already produces it.
pub const Transform = struct {
    x: Slug = 0,
    y: Slug = 0,
    z: Slug = 0,
};

/// The kinds of color ramp a gradient can be. Ported from `GradientInfo::Type`, slughorn.hpp:198.
/// The `transform` `Matrix` on a `GradientInfo` is interpreted per variant.
pub const GradientType = enum {
    /// Along a line: `t = m.xx*emX + m.xy*emY + m.dx`.
    linear,
    /// From a center: `m.dx/dy` = center, `m.xx` = outer radius, `inner_radius` = inner radius.
    radial,
    /// Around a center by angle, over `start_angle`..`end_angle` (turns).
    sweep,
    /// Radial in an affine-warped space: `m` is the 2x2 inverse-affine, `inner_radius` in that space.
    affine_radial,
};

/// One stop in a gradient's color ramp. Ported from `GradientStop`, slughorn.hpp:191.
pub const GradientStop = struct {
    /// Position along the gradient axis, in [0, 1].
    t: Slug = 0,
    color: Color,
};

/// A gradient paint: a color ramp plus the geometry that maps em-space to a ramp position. Ported
/// from `GradientInfo`, slughorn.hpp:196. Registered with `Atlas.addGradient` before `build()`
/// (the atlas-side storage is M5 work); a `Layer` then references the returned id. `stops` is
/// borrowed, mirroring how `ShapeInfo` borrows its curves.
pub const GradientInfo = struct {
    type: GradientType = .linear,
    stops: []const GradientStop = &.{},
    /// Gradient geometry in local em-space; interpreted per `type` (see `GradientType`).
    transform: Matrix = .identity,
    /// Radial/AffineRadial: inner radius in em-space (0 = point center).
    inner_radius: Slug = 0,
    /// Sweep only: arc range in turns, [0, 1].
    start_angle: Slug = 0,
    end_angle: Slug = 1,
};

/// Pixel data ready to hand to a graphics API, exactly as `Atlas::TextureData`
/// (slughorn.hpp:818).
///
/// `bytes` is row-major and **owned by the `Atlas`** that produced it; accessors hand out
/// `*const TextureData`, mirroring upstream's const-reference returns.
pub const TextureData = struct {
    pub const Format = enum(u32) {
        /// Four 32-bit floats per texel. The curve texture.
        rgba32f = 0,
        /// Four 16-bit unsigned ints per texel. The band texture.
        rgba16ui = 1,
        rgba8 = 2,
        rgb32f = 3,

        /// Bytes per texel.
        pub fn texelSize(f: Format) u32 {
            return switch (f) {
                .rgba32f => 16,
                .rgba16ui => 8,
                .rgba8 => 4,
                .rgb32f => 12,
            };
        }
    };

    bytes: []u8 = &.{},
    width: u32 = 0,
    height: u32 = 0,
    /// >0: array texture with this many layers. 0: plain 2D.
    depth: u32 = 0,
    format: Format = .rgba32f,

    pub fn isEmpty(self: TextureData) bool {
        return self.bytes.len == 0;
    }
};

/// Where a shape's transform origin sits relative to its geometry.
///
/// Ported from `ShapeInfo::Origin`, slughorn.hpp:698. This is what the GPU uses as the rotation
/// pivot, so it decides what a rotation of the shape looks like.
pub const Origin = struct {
    pub const Type = enum {
        /// Quad at the bbox corner; origin data (0, 0).
        default,
        /// Quad at the bbox centre; origin data (width/2, height/2) in em-space.
        centered,
        /// Quad at (x, y) in authoring space; origin data is that pivot in local em-space.
        pivot,
        /// Quad at the bbox corner, but origin data is stored verbatim. For shaders that treat the
        /// origin as raw user data rather than a pivot.
        custom,
    };

    type: Type = .default,
    x: Slug = 0,
    y: Slug = 0,

    pub const default: Origin = .{};
    pub const centered: Origin = .{ .type = .centered };

    pub fn atPivot(x: Slug, y: Slug) Origin {
        return .{ .type = .pivot, .x = x, .y = y };
    }

    pub fn eql(a: Origin, b: Origin) bool {
        return a.type == b.type and a.x == b.x and a.y == b.y;
    }
};

/// Everything a caller supplies to describe one shape. Ported from `Atlas::ShapeInfo`,
/// slughorn.hpp:658.
///
/// Curves must be em-normalized (FreeType's FT_LOAD_NO_SCALE path divided by units_per_EM).
pub const ShapeInfo = struct {
    /// Borrowed for the duration of the call; `Atlas` copies what it keeps.
    curves: []const Curve,

    /// Derive width/height/bearing/advance from the curve bounding box. Set false and fill the
    /// metric fields to forward a font's own metrics.
    auto_metrics: bool = true,

    bearing_x: Slug = 0,
    bearing_y: Slug = 0,
    width: Slug = 0,
    height: Slug = 0,
    advance: Slug = 0,

    /// Band grid dimensions. 0 means "pick automatically" for that axis.
    ///
    /// Unsigned here, unlike upstream's signed field (slughorn.hpp:687): the C++ uses negative as
    /// an invalid sentinel and clamps it away at slughorn.cpp:284, which a `u32` makes
    /// unrepresentable instead.
    num_bands_x: u32 = 0,
    num_bands_y: u32 = 0,

    /// Interior split positions as normalized [0, 1] fractions of the shape's range, ascending.
    /// When non-empty these override `num_bands_*`; the resulting band count is len + 1.
    splits_x: []const Slug = &.{},
    splits_y: []const Slug = &.{},

    origin: Origin = .default,
};

/// Everything the renderer needs to draw one shape. Populated by `build()`.
///
/// Ported from `Atlas::Shape`, slughorn.hpp:727. Carries the MSDF result fields (`msdf_layer`,
/// `msdf_range`) so the core `Shape` can hold an MSDF tile reference; the atlas-level machinery that
/// populates them is still unported. The scanline-sweeper fields upstream also has are omitted --
/// that feature is not on the roadmap.
pub const Shape = struct {
    /// Where this shape's band header block starts in the band texture (texel coords).
    band_tex_x: u32 = 0,
    band_tex_y: u32 = 0,

    /// Band index clamping limits (num_bands - 1).
    band_max_x: u32 = 0,
    band_max_y: u32 = 0,

    /// Band-space transform: band_coord = em_pos * band_scale + band_offset.
    band_scale_x: Slug = 0,
    band_scale_y: Slug = 0,
    band_offset_x: Slug = 0,
    band_offset_y: Slug = 0,

    /// Metrics in em-space (normalized to the font's em square, or to the curve bounding box when
    /// `auto_metrics`).
    bearing_x: Slug = 0,
    bearing_y: Slug = 0,
    width: Slug = 0,
    height: Slug = 0,
    advance: Slug = 0,

    /// Em-space pivot, derived from `ShapeInfo.origin` during `build()`.
    origin_x: Slug = 0,
    origin_y: Slug = 0,

    /// The origin spec as supplied, retained for diagnostics and for `computeQuad` branching.
    origin: Origin = .default,

    /// Layer index in the MSDF texture array; -1 = no MSDF tile generated yet. Ported from
    /// `msdfLayer` (slughorn.hpp:762). The field lives here so the core `Shape` can *carry* an MSDF
    /// result; the atlas-level machinery that populates it (`rasterizeSDFAtlas`/`requestMSDF`) is
    /// still unported -- see `src/backends/sdf.zig`.
    msdf_layer: i32 = -1,
    /// Em-space SDF range used when this shape's MSDF tile was generated. Ported from `msdfRange`
    /// (slughorn.hpp:766). Zero until a tile is generated.
    msdf_range: Slug = 0,

    /// The original em-space curves, retained post-build so callers can re-read outlines without
    /// re-running a font backend. Owned by the `Atlas`.
    curves: []Curve = &.{},

    /// The world-space bounding quad for this shape.
    ///
    /// `scale` converts em-space metrics to world units; `expand` is a small extra em-space margin
    /// for AA fringes or rotated content (do not derive it from `scale`). The quad is relative to
    /// (0, 0) -- scene placement is the caller's job.
    ///
    /// Ported from slughorn.hpp:783.
    pub fn computeQuad(self: Shape, tx: Slug, ty: Slug, scale: Slug, expand: Slug) Quad {
        const ox = (tx - self.origin_x) * scale;
        const oy = (ty - self.origin_y) * scale;
        return .{
            .x1 = ox + (self.bearing_x - expand) * scale,
            .y1 = oy + (self.bearing_y - self.height - expand) * scale,
            .x2 = ox + (self.bearing_x + self.width + expand) * scale,
            .y2 = oy + (self.bearing_y + expand) * scale,
        };
    }
};

/// An axis-aligned rectangle. Ported from slughorn.hpp:221.
pub const Quad = extern struct {
    x1: Slug = 0,
    y1: Slug = 0,
    x2: Slug = 0,
    y2: Slug = 0,
};

/// Texture-packing statistics, populated by `build()`. Ported from slughorn.hpp:870.
///
/// `padding` counts texels burned by the row-straddle alignment described in `pack.zig`.
///
/// The two `max` fields are the ones worth watching: utilization ratios cannot reveal how close a
/// build came to the hard uint16 limits, because the textures are sized to just fit whatever data
/// exists and so read ~100% full almost regardless of headroom.
pub const PackingStats = extern struct {
    /// Texels written with actual curve data.
    curve_texels_used: u32 = 0,
    /// Texels wasted to row-alignment bumps.
    curve_texels_padding: u32 = 0,
    /// width * height (allocated).
    curve_texels_total: u32 = 0,

    band_texels_used: u32 = 0,
    band_texels_padding: u32 = 0,
    band_texels_total: u32 = 0,

    /// Largest single band's curve-index list across all shapes. Limit: 65535, and also the
    /// texture width (see `pack.zig`).
    band_max_count: u32 = 0,
    /// Largest per-shape cumulative band-data span (`cursor - shape_start`). Limit: 65535.
    band_max_offset: u32 = 0,
};

test "Curve extents and finiteness" {
    const c: Curve = .{ .x1 = 0, .y1 = 0, .x2 = 0.5, .y2 = 1, .x3 = 1, .y3 = 0 };
    try std.testing.expectEqual(@as(Slug, 0), c.minX());
    try std.testing.expectEqual(@as(Slug, 1), c.maxX());
    try std.testing.expectEqual(@as(Slug, 0), c.minY());
    try std.testing.expectEqual(@as(Slug, 1), c.maxY());
    try std.testing.expect(c.isFinite());

    const nan: Curve = .{ .x1 = 0, .y1 = std.math.nan(Slug), .x2 = 0, .y2 = 0, .x3 = 0, .y3 = 0 };
    try std.testing.expect(!nan.isFinite());
    const inf: Curve = .{ .x1 = std.math.inf(Slug), .y1 = 0, .x2 = 0, .y2 = 0, .x3 = 0, .y3 = 0 };
    try std.testing.expect(!inf.isFinite());
}

test "Matrix identity and composition" {
    const p = Matrix.identity.apply(.{ 3, 4 });
    try std.testing.expectEqual(zml.Vec2f32{ 3, 4 }, p);

    // Scale by 2, then translate by (1, 1).
    const m = Matrix.scaling(2, 2).mul(Matrix.translation(1, 1));
    try std.testing.expectEqual(zml.Vec2f32{ 7, 9 }, m.apply(.{ 3, 4 }));

    // `mul` is ordered, not commutative: translating first puts the scale on the offset too.
    const n = Matrix.translation(1, 1).mul(Matrix.scaling(2, 2));
    try std.testing.expectEqual(zml.Vec2f32{ 8, 10 }, n.apply(.{ 3, 4 }));
}

test "Matrix applyDir ignores translation" {
    const m = Matrix.scaling(2, 3).mul(Matrix.translation(100, 100));
    try std.testing.expectEqual(zml.Vec2f32{ 106, 112 }, m.apply(.{ 3, 4 }));
    try std.testing.expectEqual(zml.Vec2f32{ 6, 12 }, m.applyDir(.{ 3, 4 }));
}

test "Matrix rotation" {
    const m = Matrix.rotation(std.math.pi / 2.0);
    const p = m.apply(.{ 1, 0 });
    // A quarter turn takes +x to +y; sin/cos leave a little dust behind.
    try std.testing.expectApproxEqAbs(@as(Slug, 0), p[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(Slug, 1), p[1], 1e-6);
}

test "Color is a zml vector and interpolates" {
    const black = rgb(0, 0, 0);
    try std.testing.expectEqual(@as(Slug, 1), black[3]);

    const c = rgba(1, 0, 0, 1);
    try std.testing.expectEqual(@as(Slug, 1), c[0]);

    // The reason Color is a vector rather than a struct: gradient stops are just a lerp.
    const mid = zml.scalar.lerp(black, c, 0.5);
    try std.testing.expectEqual(Color{ 0.5, 0, 0, 1 }, mid);
}

test "Curve stays a 24-byte GPU wire format" {
    // Curve is memcmp'd against the C++ golden fixtures and packed into RGBA32F texels. Rebuilding
    // it on zml.Vec2f32 would keep the size but raise alignment to 8; pin both.
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Curve));
    try std.testing.expectEqual(@as(usize, 4), @alignOf(Curve));
}

test "TextureData texel sizes" {
    try std.testing.expectEqual(@as(u32, 16), TextureData.Format.rgba32f.texelSize());
    try std.testing.expectEqual(@as(u32, 8), TextureData.Format.rgba16ui.texelSize());
    try std.testing.expectEqual(@as(u32, 12), TextureData.Format.rgb32f.texelSize());
    const empty: TextureData = .{};
    try std.testing.expect(empty.isEmpty());
}

test "Transform is position-only, zero by default" {
    const t: Transform = .{ .x = 1, .y = 2 };
    try std.testing.expectEqual(@as(Slug, 1), t.x);
    try std.testing.expectEqual(@as(Slug, 2), t.y);
    try std.testing.expectEqual(@as(Slug, 0), t.z);
}

test "GradientInfo carries the upstream defaults" {
    const g: GradientInfo = .{ .stops = &.{
        .{ .t = 0, .color = rgb(1, 0, 0) },
        .{ .t = 1, .color = rgb(0, 0, 1) },
    } };
    try std.testing.expectEqual(GradientType.linear, g.type);
    try std.testing.expect(std.meta.eql(g.transform, Matrix.identity));
    try std.testing.expectEqual(@as(Slug, 0), g.inner_radius);
    try std.testing.expectEqual(@as(Slug, 0), g.start_angle);
    try std.testing.expectEqual(@as(Slug, 1), g.end_angle);
    try std.testing.expectEqual(@as(usize, 2), g.stops.len);
    // Color is a Vec4f32; rgb() sets a = 1, so a stop can be lerped directly.
    try std.testing.expectEqual(@as(Slug, 1), g.stops[0].color[3]);
}

test "Shape carries MSDF result fields defaulted to 'none'" {
    const s: Shape = .{};
    try std.testing.expectEqual(@as(i32, -1), s.msdf_layer); // -1 sentinel = no tile generated
    try std.testing.expectEqual(@as(Slug, 0), s.msdf_range);
}
