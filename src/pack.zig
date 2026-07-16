// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! Texture packing: lay every shape's curves and bands out into the two GPU textures.
//!
//! This is the most delicate code in the library. The layout is not free-form -- it encodes an
//! exact contract with the fragment shader, and the row-straddle invariant below is load-bearing
//! (upstream has a war story about getting it wrong; see `BandPacker.packList`).
//!
//! Ported from `Atlas::packTextures`, slughorn.cpp:1167.

const std = @import("std");
const build_options = @import("build_options");

const oom = @import("oom.zig");
const types = @import("types.zig");
const errors = @import("errors.zig");
const bands_mod = @import("bands.zig");
const key_mod = @import("key.zig");

const Slug = types.Slug;
const Shape = types.Shape;
const TextureData = types.TextureData;
const PackingStats = types.PackingStats;
const ShapeBuild = bands_mod.ShapeBuild;
const BandEntry = bands_mod.BandEntry;
const Key = key_mod.Key;

const indirection_size: u32 = build_options.indirection_size;

/// Advances `cursor` so a span of `span` texels fits without crossing a row boundary.
///
/// Ported from `alignCursorForSpan`, slughorn.cpp:180.
pub fn alignCursorForSpan(cursor: u32, width: u32, span: u32) u32 {
    if (span == 0) return cursor;
    const x = cursor % width;
    if (x + span > width) return cursor + (width - x);
    return cursor;
}

/// Writes texels into a `TextureData`'s byte buffer.
///
/// Replaces upstream's three near-identical `write*Texel` lambdas (slughorn.cpp:1268-1305) and
/// their `reinterpret_cast`s. Out-of-range writes are silently dropped, matching upstream's
/// `if (y >= height) return;` guard.
pub const TexWriter = struct {
    tex: *TextureData,

    pub fn writeRGBA32F(self: TexWriter, idx: u32, r: Slug, g: Slug, b: Slug, a: Slug) void {
        const x = idx % self.tex.width;
        const y = idx / self.tex.width;
        if (y >= self.tex.height) return;

        const off = (@as(usize, y) * self.tex.width + x) * 4 * @sizeOf(f32);
        for ([_]Slug{ r, g, b, a }, 0..) |v, i| {
            std.mem.writeInt(u32, self.tex.bytes[off + i * 4 ..][0..4], @bitCast(v), .little);
        }
    }

    pub fn writeRGBA16UI(self: TexWriter, idx: u32, r: u16, g: u16, b: u16, a: u16) void {
        const x = idx % self.tex.width;
        const y = idx / self.tex.width;
        if (y >= self.tex.height) return;

        const off = (@as(usize, y) * self.tex.width + x) * 4 * @sizeOf(u16);
        for ([_]u16{ r, g, b, a }, 0..) |v, i| {
            std.mem.writeInt(u16, self.tex.bytes[off + i * 2 ..][0..2], v, .little);
        }
    }
};

/// What `packTextures` produces.
pub const Output = struct {
    curve_data: *TextureData,
    band_data: *TextureData,
    shapes: *key_mod.KeyMap(Shape),
    stats: *PackingStats,
    diag: ?*errors.Diagnostics,
};

/// A cursor that records how many texels alignment burned.
const Cursor = struct {
    pos: u32 = 0,
    width: u32,
    padding: *u32,

    fn alignFor(self: *Cursor, span: u32) void {
        const aligned = alignCursorForSpan(self.pos, self.width, span);
        self.padding.* += aligned - self.pos;
        self.pos = aligned;
    }
};

/// Packs one shape's band curve-index lists and fills in its headers.
///
/// This is upstream's `packBandList` lambda (slughorn.cpp:1411), which captured six things and
/// needed to raise -- a real struct rather than a closure.
const BandPacker = struct {
    w: TexWriter,
    tex_width: u32,
    shape_start: u32,
    curve_locs: []const u32,
    headers: []Header,
    stats: *PackingStats,
    cursor: *Cursor,
    key: Key,
    diag: ?*errors.Diagnostics,

    const Header = struct { count: u16 = 0, offset: u16 = 0 };

    fn packList(self: *BandPacker, band_list: []const BandEntry, header_base: u32) errors.BuildError!void {
        for (band_list, 0..) |band, b| {
            const count: u32 = @intCast(band.curve_indices.items.len);

            // CORRECTION: count > _texWidth IS a real, hard limit, not an arbitrary one --
            // a band's curve-index list must fit within a single texture row. The shader's
            // read loop (slug_HFill/slug_VFill in Atlas.shaders.cpp) only wraps the list's
            // STARTING offset via slug_CalcBandLoc(); it then does
            // texelFetch(..., hbandLoc.x + curveIndex, hbandLoc.y, ...) with a FIXED row,
            // so a list longer than one row reads garbage/wrong data past the row boundary.
            // alignCursorForSpan() can only shift where a span STARTS, it cannot make a
            // span wider than the row fit without straddling. An earlier version of this
            // comment claimed the shader already handled multi-row lists and replaced this
            // check with a uint16_t-only one -- that was wrong, and got caught when a real
            // tile (738-curve band) still corrupted at texWidth=512 after that "fix".
            if (count > self.tex_width) {
                if (self.diag) |d| d.* = .{
                    .key = self.key,
                    .count = count,
                    .tex_width = self.tex_width,
                    .suggested_tex_width = std.math.ceilPowerOfTwoAssert(u32, count),
                };
                return error.BandExceedsTextureRow;
            }

            // Separate, independent limit: header.count is a uint16_t regardless of
            // texWidth. Only reachable if _texWidth itself somehow exceeded 65535.
            if (count > 0xffff) {
                if (self.diag) |d| d.* = .{ .key = self.key, .count = count, .tex_width = self.tex_width };
                return error.BandCountOverflow;
            }

            self.stats.band_max_count = @max(self.stats.band_max_count, count);

            self.cursor.alignFor(count);

            const hi = header_base + @as(u32, @intCast(b));
            const span = self.cursor.pos - self.shape_start;

            // offset is relative to shapeStart and also a uint16_t. This is the more
            // realistic overflow case in practice -- it accumulates across every band in
            // the shape (up to 2*INDIRECTION_SIZE of them), so a shape with many
            // moderately-sized bands can hit this even when no single band's count does.
            if (span > 0xffff) {
                if (self.diag) |d| d.* = .{ .key = self.key, .count = span, .tex_width = self.tex_width };
                return error.BandOffsetOverflow;
            }

            self.stats.band_max_offset = @max(self.stats.band_max_offset, span);

            self.headers[hi] = .{ .count = @intCast(count), .offset = @intCast(span) };

            for (band.curve_indices.items) |ci| {
                const loc = self.curve_locs[ci];
                // Absolute curve-texture coordinates. This is what makes a shape's packed block
                // non-relocatable, and therefore what makes iteration order part of the output --
                // see DIVERGENCE.md.
                self.w.writeRGBA16UI(
                    self.cursor.pos,
                    @intCast(loc % self.tex_width),
                    @intCast(loc / self.tex_width),
                    0,
                    0,
                );
                self.cursor.pos += 1;
            }

            self.stats.band_texels_used += count;
        }
    }
};

/// Lays out every shape into the curve and band textures.
///
/// `build_map` is consumed: each shape's curves are moved into the resulting `Shape`.
pub fn packTextures(
    gpa: std.mem.Allocator,
    tex_width: u32,
    build_map: *key_mod.KeyMap(ShapeBuild),
    out: Output,
) errors.BuildError!void {
    // -- pass 1: measure the curve texture ------------------------------------------------------
    var total_curve_texels: u32 = 0;
    for (build_map.values()) |g| {
        for (0..g.curves.items.len) |_| {
            total_curve_texels = alignCursorForSpan(total_curve_texels, tex_width, 2);
            total_curve_texels += 2;
        }
    }

    const curve_tex_height = @max(1, (total_curve_texels + tex_width - 1) / tex_width);

    out.curve_data.* = .{
        .width = tex_width,
        .height = curve_tex_height,
        .format = .rgba32f,
        .bytes = oom.must(gpa.alloc(u8, @as(usize, tex_width) * curve_tex_height * 4 * @sizeOf(f32))),
    };
    @memset(out.curve_data.bytes, 0);

    // -- pass 1: measure the band texture -------------------------------------------------------
    var total_band_texels: u32 = 0;
    for (build_map.values()) |g| {
        const block_size = blockSize(g);

        total_band_texels = alignCursorForSpan(total_band_texels, tex_width, block_size);

        var cursor = total_band_texels + block_size;
        for ([_][]const BandEntry{ g.hbands.items, g.vbands.items }) |list| {
            for (list) |band| {
                const count: u32 = @intCast(band.curve_indices.items.len);
                cursor = alignCursorForSpan(cursor, tex_width, count);
                cursor += count;
            }
        }
        total_band_texels = cursor;
    }

    const band_tex_height = @max(1, (total_band_texels + tex_width - 1) / tex_width);

    out.band_data.* = .{
        .width = tex_width,
        .height = band_tex_height,
        .format = .rgba16ui,
        .bytes = oom.must(gpa.alloc(u8, @as(usize, tex_width) * band_tex_height * 4 * @sizeOf(u16))),
    };
    @memset(out.band_data.bytes, 0);

    // -- pass 2: write ---------------------------------------------------------------------------
    out.stats.* = .{
        .curve_texels_total = tex_width * curve_tex_height,
        .band_texels_total = tex_width * band_tex_height,
    };

    const cw: TexWriter = .{ .tex = out.curve_data };
    const bw: TexWriter = .{ .tex = out.band_data };

    var curve_cursor: Cursor = .{ .width = tex_width, .padding = &out.stats.curve_texels_padding };
    var band_cursor: Cursor = .{ .width = tex_width, .padding = &out.stats.band_texels_padding };

    var it = build_map.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const g = entry.value_ptr;

        // -- curves: two texels each, (x1,y1,x2,y2) then (x3,y3,0,0) ----------------------------
        const curve_locs = oom.must(gpa.alloc(u32, g.curves.items.len));
        defer gpa.free(curve_locs);

        for (g.curves.items, 0..) |c, ci| {
            curve_cursor.alignFor(2);
            curve_locs[ci] = curve_cursor.pos;

            cw.writeRGBA32F(curve_cursor.pos, c.x1, c.y1, c.x2, c.y2);
            cw.writeRGBA32F(curve_cursor.pos + 1, c.x3, c.y3, 0, 0);

            curve_cursor.pos += 2;
            out.stats.curve_texels_used += 2;
        }

        // -- band block ---------------------------------------------------------------------------
        //
        // Per shape:
        //   [indirY x INDIRECTION_SIZE][indirX x INDIRECTION_SIZE][hband headers][vband headers]
        //
        // followed by the curve index lists (aligned, possibly wrapping rows via
        // slug_CalcBandLoc). Shapes with no curves get no indirection block.
        var sd = g.metrics;

        const num_h: u32 = @intCast(g.hbands.items.len);
        const num_v: u32 = @intCast(g.vbands.items.len);
        const num_headers = num_h + num_v;
        const indir_size: u32 = if (num_headers > 0) 2 * indirection_size else 0;
        const block_size = indir_size + num_headers;

        // blockSize is bounded by 2*INDIRECTION_SIZE + numBandHeaders (currently at most
        // 128), so this should never trigger for any sane _texWidth -- but silently
        // `continue`-ing here used to drop the shape's entire band data with no signal
        // at all. Fail loudly instead: a texture too narrow to hold one shape's fixed-size
        // header block is a real configuration error, not something to paper over.
        if (block_size > tex_width) {
            if (out.diag) |d| d.* = .{ .key = key, .count = block_size, .tex_width = tex_width };
            return error.HeaderBlockExceedsTextureRow;
        }

        band_cursor.alignFor(block_size);

        const shape_start = band_cursor.pos;
        sd.band_tex_x = shape_start % tex_width;
        sd.band_tex_y = shape_start / tex_width;

        if (indir_size > 0 and
            g.indir_y.items.len == indirection_size and
            g.indir_x.items.len == indirection_size)
        {
            for (0..indirection_size) |q| {
                const qi: u32 = @intCast(q);
                bw.writeRGBA16UI(shape_start + qi, g.indir_y.items[q], 0, 0, 0);
                bw.writeRGBA16UI(shape_start + indirection_size + qi, g.indir_x.items[q], 0, 0, 0);
            }
            out.stats.band_texels_used += 2 * indirection_size;
        }

        const headers = oom.must(gpa.alloc(BandPacker.Header, num_headers));
        defer gpa.free(headers);
        @memset(headers, .{});

        band_cursor.pos = shape_start + block_size;

        var packer: BandPacker = .{
            .w = bw,
            .tex_width = tex_width,
            .shape_start = shape_start,
            .curve_locs = curve_locs,
            .headers = headers,
            .stats = out.stats,
            .cursor = &band_cursor,
            .key = key,
            .diag = out.diag,
        };

        try packer.packList(g.hbands.items, 0);
        try packer.packList(g.vbands.items, num_h);

        // Headers go at shape_start + indir_size, after the indirection blocks. Written only now
        // because their offsets are not known until the lists above have been placed.
        for (0..num_h) |i| {
            const ii: u32 = @intCast(i);
            bw.writeRGBA16UI(shape_start + indir_size + ii, headers[i].count, headers[i].offset, 0, 0);
        }
        for (0..num_v) |i| {
            const ii: u32 = @intCast(i);
            bw.writeRGBA16UI(
                shape_start + indir_size + num_h + ii,
                headers[num_h + i].count,
                headers[num_h + i].offset,
                0,
                0,
            );
        }

        out.stats.band_texels_used += num_headers;

        // The shape takes ownership of its curves; the ShapeBuild is discarded after this.
        sd.curves = oom.must(g.curves.toOwnedSlice(gpa));

        oom.must(out.shapes.put(gpa, key, sd));
    }
}

fn blockSize(g: ShapeBuild) u32 {
    const num_headers: u32 = @intCast(g.hbands.items.len + g.vbands.items.len);
    const indir: u32 = if (num_headers > 0) 2 * indirection_size else 0;
    return indir + num_headers;
}

// ================================================================================================
// Tests
// ================================================================================================

const testing = std.testing;

test "alignCursorForSpan keeps spans off row boundaries" {
    // Fits in the current row: no bump.
    try testing.expectEqual(@as(u32, 0), alignCursorForSpan(0, 8, 2));
    try testing.expectEqual(@as(u32, 6), alignCursorForSpan(6, 8, 2));

    // Would straddle: bump to the next row.
    try testing.expectEqual(@as(u32, 8), alignCursorForSpan(7, 8, 2));
    try testing.expectEqual(@as(u32, 16), alignCursorForSpan(13, 8, 4));

    // A zero span never moves.
    try testing.expectEqual(@as(u32, 7), alignCursorForSpan(7, 8, 0));

    // Exactly filling the row is fine -- the boundary is not crossed.
    try testing.expectEqual(@as(u32, 4), alignCursorForSpan(4, 8, 4));
}

test "alignCursorForSpan cannot rescue an over-wide span" {
    // A span wider than a row straddles wherever it starts. This is why the count > tex_width
    // check exists rather than relying on alignment -- see BandPacker.packList.
    const width: u32 = 8;
    const span: u32 = 9;
    var cursor: u32 = 0;
    while (cursor < width) : (cursor += 1) {
        const aligned = alignCursorForSpan(cursor, width, span);
        try testing.expect(aligned % width + span > width);
    }
}

test "TexWriter round-trips both formats and drops out-of-range writes" {
    const gpa = testing.allocator;

    var tex: TextureData = .{
        .width = 4,
        .height = 2,
        .format = .rgba32f,
        .bytes = try gpa.alloc(u8, 4 * 2 * 4 * @sizeOf(f32)),
    };
    defer gpa.free(tex.bytes);
    @memset(tex.bytes, 0);

    const w: TexWriter = .{ .tex = &tex };
    w.writeRGBA32F(5, 1.5, -2.5, 3.5, 4.5); // x=1, y=1

    const off = (1 * 4 + 1) * 4 * @sizeOf(f32);
    try testing.expectEqual(@as(f32, 1.5), @as(f32, @bitCast(std.mem.readInt(u32, tex.bytes[off..][0..4], .little))));
    try testing.expectEqual(@as(f32, 4.5), @as(f32, @bitCast(std.mem.readInt(u32, tex.bytes[off + 12 ..][0..4], .little))));

    // Past the last row: dropped, not a buffer overrun.
    w.writeRGBA32F(100, 9, 9, 9, 9);
}

test "TexWriter writes RGBA16UI little-endian" {
    const gpa = testing.allocator;

    var tex: TextureData = .{
        .width = 2,
        .height = 1,
        .format = .rgba16ui,
        .bytes = try gpa.alloc(u8, 2 * 1 * 4 * @sizeOf(u16)),
    };
    defer gpa.free(tex.bytes);
    @memset(tex.bytes, 0);

    const w: TexWriter = .{ .tex = &tex };
    w.writeRGBA16UI(1, 0x1234, 0x5678, 0, 0);

    try testing.expectEqualSlices(u8, &.{ 0x34, 0x12, 0x78, 0x56, 0, 0, 0, 0 }, tex.bytes[8..16]);
}
