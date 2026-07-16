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

const oom = @import("oom.zig");
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
const Key = key_mod.Key;
const ShapeBuild = bands_mod.ShapeBuild;

pub const default_texture_width: u32 = 512;

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
                const copy = oom.must(self.gpa.dupe(u8, n));
                oom.must(self.names.append(self.gpa, copy));
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

        oom.must(b.curves.appendSlice(self.gpa, desc.curves));
        oom.must(b.splits_x.appendSlice(self.gpa, desc.splits_x));
        oom.must(b.splits_y.appendSlice(self.gpa, desc.splits_y));

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
        oom.must(self.build_map.put(self.gpa, owned, b));
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

    pub fn getCurveTextureData(self: *const Atlas) *const TextureData {
        return &self.curve_data;
    }

    pub fn getBandTextureData(self: *const Atlas) *const TextureData {
        return &self.band_data;
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
