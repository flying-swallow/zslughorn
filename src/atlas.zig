// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! The `Atlas`: the whole public surface of the data compiler.
//!
//! Lifecycle is populate -> `build()` -> frozen. After `build()` the textures are immutable, which
//! is the point of the Slug technique: the atlas is compiled once and never rebuilt, no matter how
//! the shapes are subsequently transformed.
//!
//! Ported from `Atlas`, slughorn.hpp:636.

const std = @import("std");

const types = @import("types.zig");
const errors = @import("errors.zig");
const bands_mod = @import("bands.zig");
const pack = @import("pack.zig");
const key_mod = @import("key.zig");

const Slug = types.Slug;
const Curve = types.Curve;
const Shape = types.Shape;
const ShapeInfo = types.ShapeInfo;
const TextureData = types.TextureData;
const PackingStats = types.PackingStats;
const GradientInfo = types.GradientInfo;
const GradientStop = types.GradientStop;
const Key = key_mod.Key;
const ShapeBuild = bands_mod.ShapeBuild;

pub const default_texture_width: u32 = 512;

/// Texels per gradient color strip (upstream `GRADIENT_STRIP_WIDTH`, slughorn.hpp:806). 256 gives
/// 8-bit `t` precision; a sampler's bilinear filtering smooths between texels.
const gradient_strip_width: u32 = 256;

fn stopLessThan(_: void, a: GradientStop, b: GradientStop) bool {
    return a.t < b.t;
}

/// `clamp(v, 0, 1) * 255`, rounded. Matches upstream's `toU8` lambda (no gamma conversion).
fn toU8(v: Slug) u8 {
    return @intFromFloat(@max(0.0, @min(1.0, v)) * 255.0 + 0.5);
}

pub const Atlas = struct {
    gpa: std.mem.Allocator,

    tex_width: u32,
    built: bool = false,

    /// Optional structured detail for the most recent error.
    diag: ?*errors.Diagnostics = null,

    /// Discarded after `build()`.
    build_map: key_mod.KeyMap(ShapeBuild) = .empty,
    /// Live after `build()`.
    shapes: key_mod.KeyMap(Shape) = .empty,

    /// Interned `Key.name` storage; `Key`s handed to us borrow, the atlas owns its copies.
    names: std.ArrayList([]u8) = .empty,

    curve_data: TextureData = .{},
    band_data: TextureData = .{},
    stats: PackingStats = .{},

    /// Registered gradient paints, referenced 1-based by `Layer.gradient_id`. Each owns a copy of
    /// its stops. `build()` rasterizes these into `gradient_data`.
    gradients: std.ArrayList(GradientInfo) = .empty,
    /// The gradient color strips, produced by `build()`: 256 texels wide (`gradient_strip_width`),
    /// one RGBA8 row per gradient. Empty when no gradients were registered.
    gradient_data: TextureData = .{},

    /// Creates an atlas whose textures are `tex_width` texels wide.
    ///
    /// The width must be a positive power of two: the shader uses log2(tex_width) as a bit-shift
    /// count, so any other width corrupts band coordinate wrapping. Upstream asserts this
    /// (slughorn.cpp:196), which compiles out under NDEBUG -- here it is always checked.
    pub fn init(gpa: std.mem.Allocator, tex_width: u32) error{InvalidTextureWidth}!Atlas {
        if (tex_width == 0 or !std.math.isPowerOfTwo(tex_width)) return error.InvalidTextureWidth;

        // The indirection tables alone occupy 2 * indirection_size texels of every shape's block
        // and must fit one row; a narrower atlas could never pack anything.
        if (tex_width < 2 * @import("slughorn.zig").indirection_size) return error.InvalidTextureWidth;

        return .{ .gpa = gpa, .tex_width = tex_width };
    }

    pub fn deinit(self: *Atlas) void {
        for (self.build_map.values()) |*b| b.deinit(self.gpa);
        self.build_map.deinit(self.gpa);

        for (self.shapes.values()) |*s| self.gpa.free(s.curves);
        self.shapes.deinit(self.gpa);

        for (self.names.items) |n| self.gpa.free(n);
        self.names.deinit(self.gpa);

        if (self.curve_data.bytes.len != 0) self.gpa.free(self.curve_data.bytes);
        if (self.band_data.bytes.len != 0) self.gpa.free(self.band_data.bytes);

        for (self.gradients.items) |g| self.gpa.free(g.stops);
        self.gradients.deinit(self.gpa);
        if (self.gradient_data.bytes.len != 0) self.gpa.free(self.gradient_data.bytes);

        self.* = undefined;
    }

    /// Interns a key's name so the atlas does not depend on caller-owned memory.
    fn intern(self: *Atlas, key: Key) Key {
        return switch (key) {
            .codepoint => key,
            .name => |n| blk: {
                // Reuse an existing interned copy when we already have this shape.
                if (self.build_map.getKey(key)) |k| break :blk k;
                if (self.shapes.getKey(key)) |k| break :blk k;
                const copy = self.gpa.dupe(u8, n) catch @panic("slughorn: oom");
                self.names.append(self.gpa, copy) catch @panic("slughorn: oom");
                break :blk .{ .name = copy };
            },
        };
    }

    /// Registers a shape.
    ///
    /// Curves are copied. Coordinates must be finite: the band sort comparator is not a strict
    /// weak ordering under NaN, which would make upstream's `std::sort` undefined behaviour.
    pub fn addShape(
        self: *Atlas,
        key: Key,
        desc: ShapeInfo,
    ) (errors.BuildError || error{ AtlasAlreadyBuilt, NonFiniteCoordinate })!void {
        // Upstream silently ignores this (`if(_built) return;`, slughorn.cpp:267).
        if (self.built) return error.AtlasAlreadyBuilt;

        for (desc.curves) |c| {
            if (!c.isFinite()) return error.NonFiniteCoordinate;
        }

        var b: ShapeBuild = .{};
        errdefer b.deinit(self.gpa);

        b.curves.appendSlice(self.gpa, desc.curves) catch @panic("slughorn: oom");
        b.splits_x.appendSlice(self.gpa, desc.splits_x) catch @panic("slughorn: oom");
        b.splits_y.appendSlice(self.gpa, desc.splits_y) catch @panic("slughorn: oom");

        if (!desc.auto_metrics) {
            b.metrics.bearing_x = desc.bearing_x;
            b.metrics.bearing_y = desc.bearing_y;
            b.metrics.width = desc.width;
            b.metrics.height = desc.height;
            b.metrics.advance = desc.advance;
        }

        try bands_mod.buildShapeBands(
            self.gpa,
            &b,
            desc.num_bands_x,
            desc.num_bands_y,
            !desc.auto_metrics,
            desc.origin,
        );

        const owned = self.intern(key);

        // Replacing an existing shape must not leak the old build.
        if (self.build_map.getPtr(owned)) |existing| existing.deinit(self.gpa);
        self.build_map.put(self.gpa, owned, b) catch @panic("slughorn: oom");
    }

    /// Compiles every registered shape into the textures.
    ///
    /// One-shot: the atlas is frozen afterwards. On failure the atlas is left built-but-invalid,
    /// and the error identifies which invariant the input violated (see `Diagnostics` for detail).
    pub fn build(self: *Atlas) (errors.BuildError || error{AtlasAlreadyBuilt})!void {
        if (self.built) return error.AtlasAlreadyBuilt;

        try pack.packTextures(self.gpa, self.tex_width, &self.build_map, .{
            .curve_data = &self.curve_data,
            .band_data = &self.band_data,
            .shapes = &self.shapes,
            .stats = &self.stats,
            .diag = self.diag,
        });

        self.rasterizeGradients();

        for (self.build_map.values()) |*b| b.deinit(self.gpa);
        self.build_map.clearAndFree(self.gpa);
        self.built = true;
    }

    pub fn isBuilt(self: *const Atlas) bool {
        return self.built;
    }

    /// Borrows; null when no shape is registered under `key`.
    ///
    /// Upstream returns `std::optional<Shape>` (slughorn.hpp:1151), copying the whole retained
    /// curve vector on every call. A by-value copy here would alias the curves slice and
    /// double-free, so this borrows instead.
    pub fn getShape(self: *const Atlas, key: Key) ?*const Shape {
        return self.shapes.getPtr(key);
    }

    /// Registers a gradient paint, returning its 1-based id for `Layer.gradient_id` (0 = flat
    /// color). The stops are copied. Must be called before `build()`.
    ///
    /// Ported from `Atlas::addGradient` (slughorn.cpp:305). Upstream silently returns 0 when the
    /// atlas is already built; here that is `error.AtlasAlreadyBuilt`, matching `addShape`. Only the
    /// registration is ported -- the GPU gradient strip upstream rasterizes during `build()` is not.
    pub fn addGradient(self: *Atlas, info: GradientInfo) error{AtlasAlreadyBuilt}!u32 {
        if (self.built) return error.AtlasAlreadyBuilt;
        var owned = info;
        owned.stops = self.gpa.dupe(GradientStop, info.stops) catch @panic("slughorn: oom");
        self.gradients.append(self.gpa, owned) catch @panic("slughorn: oom");
        return @intCast(self.gradients.items.len);
    }

    /// The gradient registered under `id` (1-based), or null for 0 / out of range. Borrows.
    pub fn getGradient(self: *const Atlas, id: u32) ?*const GradientInfo {
        if (id == 0 or id > self.gradients.items.len) return null;
        return &self.gradients.items[id - 1];
    }

    pub fn gradientCount(self: *const Atlas) usize {
        return self.gradients.items.len;
    }

    pub fn getCurveTextureData(self: *const Atlas) *const TextureData {
        return &self.curve_data;
    }

    pub fn getBandTextureData(self: *const Atlas) *const TextureData {
        return &self.band_data;
    }

    /// The gradient color strips (256 x gradientCount RGBA8; empty when no gradients). A future
    /// shader samples it at `V = (gradient_id - 0.5) / gradientCount`, `U = t` (the ramp parameter).
    pub fn getGradientTextureData(self: *const Atlas) *const TextureData {
        return &self.gradient_data;
    }

    /// Rasterizes each registered gradient into one `gradient_strip_width`-wide RGBA8 row of
    /// `gradient_data`. Ported from `Atlas::rasterizeGradients` (slughorn.cpp:475). Stops are sorted
    /// ascending by `t` (in place, like upstream); each column samples `t = i/(width-1)`, clamped/
    /// padded outside the stop range and per-channel linearly interpolated within it. No premultiply
    /// or gamma conversion -- straight `clamp(v,0,1)*255 + 0.5`.
    fn rasterizeGradients(self: *Atlas) void {
        if (self.gradients.items.len == 0) return;
        const num: u32 = @intCast(self.gradients.items.len);

        const bytes = self.gpa.alloc(u8, @as(usize, gradient_strip_width) * num * 4) catch @panic("slughorn: oom");
        @memset(bytes, 0);
        self.gradient_data = .{ .bytes = bytes, .width = gradient_strip_width, .height = num, .format = .rgba8 };

        for (self.gradients.items, 0..) |*grad, g| {
            const stops = @constCast(grad.stops); // atlas-owned; sort in place, matching upstream
            std.mem.sort(GradientStop, stops, {}, stopLessThan);
            if (stops.len == 0) continue;

            for (0..gradient_strip_width) |i| {
                const t = @as(Slug, @floatFromInt(i)) / @as(Slug, @floatFromInt(gradient_strip_width - 1));

                var col: types.Color = undefined;
                if (t <= stops[0].t) {
                    col = stops[0].color;
                } else if (t >= stops[stops.len - 1].t) {
                    col = stops[stops.len - 1].color;
                } else {
                    col = stops[stops.len - 1].color; // fallback if no bracket matches (shouldn't happen)
                    var s: usize = 0;
                    while (s + 1 < stops.len) : (s += 1) {
                        const t0 = stops[s].t;
                        const t1 = stops[s + 1].t;
                        if (t >= t0 and t <= t1) {
                            const range = t1 - t0;
                            const frac: Slug = if (range > 1e-9) (t - t0) / range else 0;
                            const c0 = stops[s].color;
                            const c1 = stops[s + 1].color;
                            col = c0 + (c1 - c0) * @as(types.Color, @splat(frac));
                            break;
                        }
                    }
                }

                const base = (g * gradient_strip_width + i) * 4;
                bytes[base + 0] = toU8(col[0]);
                bytes[base + 1] = toU8(col[1]);
                bytes[base + 2] = toU8(col[2]);
                bytes[base + 3] = toU8(col[3]);
            }
        }
    }

    /// Sets the MSDF tile result on a built shape. Metadata only -- it does not touch the frozen
    /// packed textures -- so the SDF/MSDF backend (which owns tile generation) can record which layer
    /// a shape landed in and the range used, mirroring upstream's post-build `Atlas::requestMSDF`.
    pub fn setShapeMsdf(self: *Atlas, key: Key, layer: i32, range: Slug) void {
        if (self.shapes.getPtr(key)) |s| {
            s.msdf_layer = layer;
            s.msdf_range = range;
        }
    }

    pub fn getPackingStats(self: *const Atlas) PackingStats {
        return self.stats;
    }

    pub fn getTextureWidth(self: *const Atlas) u32 {
        return self.tex_width;
    }
};

// ================================================================================================
// Tests
// ================================================================================================

const testing = std.testing;

test "texture width must be a positive power of two" {
    const gpa = testing.allocator;

    for ([_]u32{ 0, 3, 100, 513, 1000 }) |bad| {
        try testing.expectError(error.InvalidTextureWidth, Atlas.init(gpa, bad));
    }

    for ([_]u32{ 64, 128, 512, 1024, 4096 }) |good| {
        var a = try Atlas.init(gpa, good);
        defer a.deinit();
        try testing.expectEqual(good, a.getTextureWidth());
    }
}

test "a width too narrow for the indirection block is rejected" {
    // 2 * indirection_size = 64 texels of every shape's block must fit one row.
    try testing.expectError(error.InvalidTextureWidth, Atlas.init(testing.allocator, 32));
}

test "build compiles a shape and freezes the atlas" {
    var a = try Atlas.init(testing.allocator, 512);
    defer a.deinit();

    try a.addShape(.{ .codepoint = 'A' }, .{
        .curves = &.{.{ .x1 = 0, .y1 = 0, .x2 = 0.5, .y2 = 1, .x3 = 1, .y3 = 0 }},
    });

    try testing.expect(!a.isBuilt());
    try a.build();
    try testing.expect(a.isBuilt());

    const s = a.getShape(.{ .codepoint = 'A' }) orelse return error.MissingShape;
    try testing.expectEqual(@as(usize, 1), s.curves.len);

    try testing.expectEqual(@as(u32, 512), a.getCurveTextureData().width);
    try testing.expectEqual(TextureData.Format.rgba32f, a.getCurveTextureData().format);
    try testing.expectEqual(TextureData.Format.rgba16ui, a.getBandTextureData().format);

    try testing.expectEqual(@as(?*const Shape, null), a.getShape(.{ .codepoint = 'Z' }));
}

test "mutating a built atlas is an error, not a silent no-op" {
    var a = try Atlas.init(testing.allocator, 512);
    defer a.deinit();

    try a.addShape(.{ .codepoint = 'A' }, .{
        .curves = &.{.{ .x1 = 0, .y1 = 0, .x2 = 0.5, .y2 = 1, .x3 = 1, .y3 = 0 }},
    });
    try a.build();

    try testing.expectError(error.AtlasAlreadyBuilt, a.addShape(.{ .codepoint = 'B' }, .{
        .curves = &.{.{ .x1 = 0, .y1 = 0, .x2 = 0.5, .y2 = 1, .x3 = 1, .y3 = 0 }},
    }));
}

test "addGradient registers 1-based, copies stops, and errors after build" {
    var a = try Atlas.init(testing.allocator, 512);
    defer a.deinit();

    // Stops in caller-owned (stack) memory: the atlas must copy, not borrow, them.
    const stops = [_]GradientStop{
        .{ .t = 0, .color = types.rgb(1, 0, 0) },
        .{ .t = 1, .color = types.rgb(0, 0, 1) },
    };
    const id1 = try a.addGradient(.{ .type = .linear, .stops = &stops });
    const id2 = try a.addGradient(.{ .type = .affine_radial, .stops = &stops });
    try testing.expectEqual(@as(u32, 1), id1); // 1-based; 0 means "none"
    try testing.expectEqual(@as(u32, 2), id2);
    try testing.expectEqual(@as(usize, 2), a.gradientCount());

    const g = a.getGradient(id1).?;
    try testing.expectEqual(types.GradientType.linear, g.type);
    try testing.expectEqual(@as(usize, 2), g.stops.len);
    try testing.expect(@intFromPtr(g.stops.ptr) != @intFromPtr(&stops)); // copied
    try testing.expectEqual(@as(Slug, 1), g.stops[1].t);

    try testing.expect(a.getGradient(0) == null);
    try testing.expect(a.getGradient(3) == null);

    try a.build();
    try testing.expectError(error.AtlasAlreadyBuilt, a.addGradient(.{ .stops = &stops }));
}

test "build rasterizes gradients into a 256-wide RGBA8 strip, one row per gradient" {
    var a = try Atlas.init(testing.allocator, 512);
    defer a.deinit();

    const stops = [_]GradientStop{
        .{ .t = 0, .color = types.rgb(1, 0, 0) }, // red at t=0
        .{ .t = 1, .color = types.rgb(0, 0, 1) }, // blue at t=1
    };
    _ = try a.addGradient(.{ .stops = &stops });
    try a.addShape(.{ .codepoint = 'A' }, .{
        .curves = &.{.{ .x1 = 0, .y1 = 0, .x2 = 0.5, .y2 = 1, .x3 = 1, .y3 = 0 }},
    });
    try a.build();

    const gd = a.getGradientTextureData();
    try testing.expectEqual(@as(u32, 256), gd.width);
    try testing.expectEqual(@as(u32, 1), gd.height); // one row per gradient
    try testing.expectEqual(types.TextureData.Format.rgba8, gd.format);
    try testing.expectEqual(@as(usize, 256 * 4), gd.bytes.len);

    // Column 0 is the first stop (red), column 255 the last (blue); alpha stays opaque throughout.
    try testing.expectEqual(@as(u8, 255), gd.bytes[0]); // R
    try testing.expectEqual(@as(u8, 0), gd.bytes[2]); // B
    const last = 255 * 4;
    try testing.expectEqual(@as(u8, 0), gd.bytes[last + 0]); // R
    try testing.expectEqual(@as(u8, 255), gd.bytes[last + 2]); // B
    try testing.expectEqual(@as(u8, 255), gd.bytes[last + 3]); // A

    // The midpoint is a ~halfway red/blue blend.
    const mid = 128 * 4;
    try testing.expect(gd.bytes[mid + 0] > 100 and gd.bytes[mid + 0] < 160);
    try testing.expect(gd.bytes[mid + 2] > 100 and gd.bytes[mid + 2] < 160);
    try testing.expectEqual(@as(u8, 255), gd.bytes[mid + 3]);
}

test "non-finite coordinates are rejected at ingest" {
    var a = try Atlas.init(testing.allocator, 512);
    defer a.deinit();

    try testing.expectError(error.NonFiniteCoordinate, a.addShape(.{ .codepoint = 'A' }, .{
        .curves = &.{.{ .x1 = 0, .y1 = std.math.nan(Slug), .x2 = 0.5, .y2 = 1, .x3 = 1, .y3 = 0 }},
    }));
    try testing.expectError(error.NonFiniteCoordinate, a.addShape(.{ .codepoint = 'B' }, .{
        .curves = &.{.{ .x1 = std.math.inf(Slug), .y1 = 0, .x2 = 0.5, .y2 = 1, .x3 = 1, .y3 = 0 }},
    }));
}

test "the atlas owns its key names" {
    var a = try Atlas.init(testing.allocator, 512);
    defer a.deinit();

    {
        // A name that does not outlive the call.
        var buf: [8]u8 = undefined;
        const tmp = try std.fmt.bufPrint(&buf, "logo", .{});
        try a.addShape(.{ .name = tmp }, .{
            .curves = &.{.{ .x1 = 0, .y1 = 0, .x2 = 0.5, .y2 = 1, .x3 = 1, .y3 = 0 }},
        });
        @memset(&buf, 'x'); // scribble over the caller's buffer
    }

    try a.build();
    try testing.expect(a.getShape(.{ .name = "logo" }) != null);
}

test "a band wider than a texture row is rejected, with a usable diagnostic" {
    var a = try Atlas.init(testing.allocator, 512);
    defer a.deinit();

    var diag: errors.Diagnostics = .{};
    a.diag = &diag;

    // 738 curves in a single band: the regression the war story in pack.zig is about.
    var curves: std.ArrayList(Curve) = .empty;
    defer curves.deinit(testing.allocator);
    for (0..738) |i| {
        const t: Slug = @as(Slug, @floatFromInt(i)) / 738.0;
        try curves.append(testing.allocator, .{
            .x1 = 0,
            .y1 = t,
            .x2 = 0.25 + t * 0.125,
            .y2 = t + 0.001,
            .x3 = 0.5 + t * 0.5,
            .y3 = t,
        });
    }

    try a.addShape(.{ .name = "band738" }, .{
        .curves = curves.items,
        .num_bands_x = 1,
        .num_bands_y = 1,
    });

    try testing.expectError(error.BandExceedsTextureRow, a.build());
    try testing.expectEqual(@as(u32, 738), diag.count);
    try testing.expectEqual(@as(u32, 512), diag.tex_width);
    // The diagnostic tells the caller what would actually fix it.
    try testing.expectEqual(@as(u32, 1024), diag.suggested_tex_width);
}

test "the same shape packs at a wider texture" {
    var a = try Atlas.init(testing.allocator, 1024);
    defer a.deinit();

    var curves: std.ArrayList(Curve) = .empty;
    defer curves.deinit(testing.allocator);
    for (0..738) |i| {
        const t: Slug = @as(Slug, @floatFromInt(i)) / 738.0;
        try curves.append(testing.allocator, .{
            .x1 = 0,
            .y1 = t,
            .x2 = 0.25 + t * 0.125,
            .y2 = t + 0.001,
            .x3 = 0.5 + t * 0.5,
            .y3 = t,
        });
    }

    try a.addShape(.{ .name = "band738" }, .{
        .curves = curves.items,
        .num_bands_x = 1,
        .num_bands_y = 1,
    });

    try a.build();
    try testing.expectEqual(@as(u32, 738), a.getPackingStats().band_max_count);
}
