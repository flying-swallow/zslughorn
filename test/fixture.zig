// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! Reader for `.slgf` golden fixtures emitted by `tools/dump/slughorn_dump.cpp`.
//!
//! The fixtures are the contract between the upstream C++ and this port: they are checked in, so
//! building and testing zslughorn needs neither a C++ toolchain nor a copy of the upstream tree.
//! Regenerate with `zig build fixtures -Dslughorn-src=<path>`.

const std = @import("std");
const slughorn = @import("slughorn");

pub const format_version: u32 = 3;

pub const Shape = struct {
    key: slughorn.Key,
    band_tex_x: u32,
    band_tex_y: u32,
    band_max_x: u32,
    band_max_y: u32,
    band_scale_x: f32,
    band_scale_y: f32,
    band_offset_x: f32,
    band_offset_y: f32,
    bearing_x: f32,
    bearing_y: f32,
    width: f32,
    height: f32,
    advance: f32,
    origin_x: f32,
    origin_y: f32,
    curves: []slughorn.Curve,
};

pub const Texture = struct {
    width: u32,
    height: u32,
    depth: u32,
    format: u32,
    bytes: []u8,
};

/// A coverage grid from the C++ `render.hpp` -- the oracle `render.zig` must reproduce.
pub const Grid = struct {
    /// Whether the band acceleration structure was used (what the shader does) or every curve was
    /// walked (the reference the banding must agree with).
    banded: bool,
    width: u32,
    height: u32,
    data: []f32,
};

pub const Fixture = struct {
    /// Heap-allocated so its address is stable.
    ///
    /// An ArenaAllocator must not be copied once handed out: `arena.allocator()` captures
    /// `&arena`, so a by-value copy (as when returning this struct) leaves the allocator writing
    /// into the original while the copy keeps a stale `buffer_list` -- and then frees only part of
    /// it. Holding it by pointer makes that impossible rather than merely avoided.
    arena: *std.heap.ArenaAllocator,
    gpa: std.mem.Allocator,

    name: []const u8,
    tex_width: u32,
    indirection_size: u32,

    /// True if the C++ `build()` raised. `message` then holds what it said.
    threw: bool,
    message: []const u8,

    // Present only when `!threw`.
    shapes: []Shape,
    stats: slughorn.PackingStats,
    curve_texture: Texture,
    band_texture: Texture,
    /// Empty for multi-shape cases and for shapes with no renderable extent.
    grids: []Grid,

    pub fn deinit(self: *Fixture) void {
        const gpa = self.gpa;
        self.arena.deinit();
        gpa.destroy(self.arena);
    }
};

pub const ParseError = error{
    BadMagic,
    BadTrailer,
    BadChecksum,
    UnsupportedVersion,
    /// The fixture was generated with a different INDIRECTION_SIZE than this build uses.
    ///
    /// This is the check that closes the drift loop upstream leaves open (slughorn.hpp:799 carries
    /// a TODO admitting nothing enforces the constant matches the shader's). The packed band
    /// layout is a function of this value, so a mismatch would silently compare unlike things.
    IndirectionSizeMismatch,
    Truncated,
};

const Reader = struct {
    buf: []const u8,
    pos: usize = 0,

    fn take(self: *Reader, n: usize) ParseError![]const u8 {
        if (self.pos + n > self.buf.len) return error.Truncated;
        defer self.pos += n;
        return self.buf[self.pos..][0..n];
    }

    fn u8v(self: *Reader) ParseError!u8 {
        return (try self.take(1))[0];
    }

    fn u32v(self: *Reader) ParseError!u32 {
        return std.mem.readInt(u32, (try self.take(4))[0..4], .little);
    }

    fn u64v(self: *Reader) ParseError!u64 {
        return std.mem.readInt(u64, (try self.take(8))[0..8], .little);
    }

    fn f32v(self: *Reader) ParseError!f32 {
        // The dumper writes the bit pattern, so this pins the exact float.
        return @bitCast(try self.u32v());
    }

    fn magic(self: *Reader, want: *const [4]u8) ParseError!void {
        const got = try self.take(4);
        if (!std.mem.eql(u8, got, want)) return error.BadMagic;
    }

    fn str(self: *Reader, arena: std.mem.Allocator) (ParseError || error{OutOfMemory})![]const u8 {
        const n = try self.u32v();
        return arena.dupe(u8, try self.take(n));
    }
};

/// Loads and validates one fixture. Caller owns the returned `Fixture` and must `deinit` it.
///
/// Paths are relative to the source root; `build.zig` sets the test runner's cwd there.
pub fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Fixture {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
    defer gpa.free(bytes);
    return parse(gpa, bytes);
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Fixture {
    if (bytes.len < 12) return error.Truncated;

    // The trailing checksum covers everything before it, so verify before trusting any of it.
    const body = bytes[0 .. bytes.len - 4];
    const want_sum = std.mem.readInt(u32, bytes[bytes.len - 4 ..][0..4], .little);
    const got_sum = std.hash.Crc32.hash(body);
    if (want_sum != got_sum) return error.BadChecksum;
    if (!std.mem.eql(u8, body[body.len - 4 ..], "ENDF")) return error.BadTrailer;

    const arena = try gpa.create(std.heap.ArenaAllocator);
    errdefer gpa.destroy(arena);
    arena.* = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    var r: Reader = .{ .buf = body[0 .. body.len - 4] };

    try r.magic("SLGF");

    const version = try r.u32v();
    if (version != format_version) return error.UnsupportedVersion;

    const indirection_size = try r.u32v();
    if (indirection_size != slughorn.indirection_size) return error.IndirectionSizeMismatch;

    const name = try r.str(a);
    const tex_width = try r.u32v();

    const threw = (try r.u32v()) != 0;
    const message = try r.str(a);

    var fx: Fixture = .{
        .arena = arena,
        .gpa = gpa,
        .name = name,
        .tex_width = tex_width,
        .indirection_size = indirection_size,
        .threw = threw,
        .message = message,
        .shapes = &.{},
        .stats = .{},
        .curve_texture = undefined,
        .band_texture = undefined,
        .grids = &.{},
    };

    if (threw) return fx;

    const num_shapes = try r.u32v();
    const shapes = try a.alloc(Shape, num_shapes);

    for (shapes) |*s| {
        const kind = try r.u8v();
        const key: slughorn.Key = switch (kind) {
            0 => .{ .codepoint = try r.u32v() },
            1 => .{ .name = try r.str(a) },
            else => return error.Truncated,
        };

        s.* = .{
            .key = key,
            .band_tex_x = try r.u32v(),
            .band_tex_y = try r.u32v(),
            .band_max_x = try r.u32v(),
            .band_max_y = try r.u32v(),
            .band_scale_x = try r.f32v(),
            .band_scale_y = try r.f32v(),
            .band_offset_x = try r.f32v(),
            .band_offset_y = try r.f32v(),
            .bearing_x = try r.f32v(),
            .bearing_y = try r.f32v(),
            .width = try r.f32v(),
            .height = try r.f32v(),
            .advance = try r.f32v(),
            .origin_x = try r.f32v(),
            .origin_y = try r.f32v(),
            .curves = &.{},
        };

        const num_curves = try r.u32v();
        const curves = try a.alloc(slughorn.Curve, num_curves);
        for (curves) |*c| {
            c.* = .{
                .x1 = try r.f32v(),
                .y1 = try r.f32v(),
                .x2 = try r.f32v(),
                .y2 = try r.f32v(),
                .x3 = try r.f32v(),
                .y3 = try r.f32v(),
            };
        }
        s.curves = curves;
    }

    fx.shapes = shapes;

    fx.stats = .{
        .curve_texels_used = try r.u32v(),
        .curve_texels_padding = try r.u32v(),
        .curve_texels_total = try r.u32v(),
        .band_texels_used = try r.u32v(),
        .band_texels_padding = try r.u32v(),
        .band_texels_total = try r.u32v(),
        .band_max_count = try r.u32v(),
        .band_max_offset = try r.u32v(),
    };

    fx.curve_texture = try readTexture(&r, a);
    fx.band_texture = try readTexture(&r, a);

    const num_grids = try r.u32v();
    const grids = try a.alloc(Grid, num_grids);
    for (grids) |*g| {
        g.banded = (try r.u32v()) != 0;
        g.width = try r.u32v();
        g.height = try r.u32v();
        const data = try a.alloc(f32, @as(usize, g.width) * g.height);
        for (data) |*v| v.* = try r.f32v();
        g.data = data;
    }
    fx.grids = grids;

    return fx;
}

fn readTexture(r: *Reader, a: std.mem.Allocator) !Texture {
    const width = try r.u32v();
    const height = try r.u32v();
    const depth = try r.u32v();
    const fmt = try r.u32v();
    const n = try r.u64v();
    const bytes = try a.dupe(u8, try r.take(@intCast(n)));
    return .{ .width = width, .height = height, .depth = depth, .format = fmt, .bytes = bytes };
}

// ================================================================================================
// Gate for step 2: the harness itself must work before any ported logic leans on it.
// ================================================================================================

test "single_curve fixture round-trips" {
    var fx = try load(std.testing.allocator, std.testing.io, "fixtures/single_curve.slgf");
    defer fx.deinit();

    try std.testing.expectEqualStrings("single_curve", fx.name);
    try std.testing.expectEqual(@as(u32, 512), fx.tex_width);
    try std.testing.expectEqual(slughorn.indirection_size, fx.indirection_size);
    try std.testing.expect(!fx.threw);

    try std.testing.expectEqual(@as(usize, 1), fx.shapes.len);
    const s = fx.shapes[0];
    try std.testing.expectEqual(@as(u32, 'A'), s.key.codepoint);
    try std.testing.expectEqual(@as(usize, 1), s.curves.len);
    // The curve survives the round trip bit-exactly.
    try std.testing.expectEqual(slughorn.Curve{ .x1 = 0, .y1 = 0, .x2 = 0.5, .y2 = 1, .x3 = 1, .y3 = 0 }, s.curves[0]);

    // Curve texture is RGBA32F; band texture is RGBA16UI.
    try std.testing.expectEqual(@as(u32, 0), fx.curve_texture.format);
    try std.testing.expectEqual(@as(u32, 1), fx.band_texture.format);
    try std.testing.expectEqual(@as(u32, 512), fx.curve_texture.width);
    try std.testing.expect(fx.curve_texture.bytes.len > 0);
    try std.testing.expect(fx.band_texture.bytes.len > 0);
    // Byte length must agree with width * height * texel size.
    try std.testing.expectEqual(
        @as(usize, fx.curve_texture.width * fx.curve_texture.height * 16),
        fx.curve_texture.bytes.len,
    );
}

test "error fixtures record the C++ exception" {
    var fx = try load(std.testing.allocator, std.testing.io, "fixtures/band_738_w512.slgf");
    defer fx.deinit();

    try std.testing.expect(fx.threw);
    // This is the regression the slughorn.cpp:1416-1434 war story is about.
    try std.testing.expect(std.mem.indexOf(u8, fx.message, "738 curves") != null);
    try std.testing.expect(std.mem.indexOf(u8, fx.message, "does not fit in a texture row") != null);
}

test "the same shape packs at a wider texture" {
    var fx = try load(std.testing.allocator, std.testing.io, "fixtures/band_738_w1024.slgf");
    defer fx.deinit();

    try std.testing.expect(!fx.threw);
    try std.testing.expectEqual(@as(u32, 1024), fx.tex_width);
    try std.testing.expectEqual(@as(u32, 738), fx.stats.band_max_count);
}

test "the texture-row limit is strict, not off-by-one" {
    // 512 curves at width 512 fits; 513 does not. `count > _texWidth`, not `>=`.
    var eq = try load(std.testing.allocator, std.testing.io, "fixtures/band_eq_texwidth.slgf");
    defer eq.deinit();
    try std.testing.expect(!eq.threw);
    try std.testing.expectEqual(@as(u32, 512), eq.stats.band_max_count);

    var plus1 = try load(std.testing.allocator, std.testing.io, "fixtures/band_texwidth_plus1.slgf");
    defer plus1.deinit();
    try std.testing.expect(plus1.threw);
    try std.testing.expect(std.mem.indexOf(u8, plus1.message, "513 curves") != null);
}

test "corrupt fixtures are rejected" {
    const gpa = std.testing.allocator;
    const good = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "fixtures/single_curve.slgf", gpa, .limited(1 << 20));
    defer gpa.free(good);

    // Flipping any byte must trip the checksum.
    const bad = try gpa.dupe(u8, good);
    defer gpa.free(bad);
    bad[64] ^= 0xff;
    try std.testing.expectError(error.BadChecksum, parse(gpa, bad));

    try std.testing.expectError(error.Truncated, parse(gpa, good[0..8]));
}
