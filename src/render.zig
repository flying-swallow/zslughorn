// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! CPU-side Slug coverage emulator: mirrors the GPU fragment shader analytically.
//!
//! This buys three things:
//!
//!  * software rendering and visual validation with no GPU;
//!  * an oracle -- `decode` reverses the packing, so rendering a shape here exercises
//!    `bands.zig` and `pack.zig` end-to-end and proves the packed bytes actually mean what the
//!    shader will think they mean;
//!  * post-build curve access (glyph outlines, stroking) without re-running a font backend.
//!
//! Ported from `render.hpp`. This mirrors **render.hpp**, not the reference GLSL -- the two
//! genuinely differ in one degenerate branch (see `solveHorizPoly`), and matching the C++ is what
//! makes the golden comparison meaningful.
//!
//! Usage:
//!
//!     var s = try render.decode(gpa, shape, atlas.getCurveTextureData(), atlas.getBandTextureData());
//!     defer s.deinit(gpa);
//!     var g = try s.renderGrid(gpa, .{ .size_hint = 128 });
//!     defer g.deinit(gpa);
//!     // g.at(row, col) is coverage in [0, 1]

const std = @import("std");
const build_options = @import("build_options");

const oom = @import("oom.zig");
const types = @import("types.zig");

const Slug = types.Slug;
const Curve = types.Curve;
const Shape = types.Shape;
const TextureData = types.TextureData;

const indirection_size: u32 = build_options.indirection_size;

/// Matches the GLSL's `1.0/65536.0` exactly (render.hpp:314).
const eps: Slug = 1.0 / 65536.0;

/// Coverage result for a single em-space point.
pub const Sample = struct {
    fill: Slug = 0,
    xcov: Slug = 0,
    ycov: Slug = 0,
    xwgt: Slug = 0,
    ywgt: Slug = 0,
    /// Curves examined. Useful for band-tuning: this is what the shader's inner loop costs.
    iters: u32 = 0,
};

/// A 2-D coverage image, row-major, values in [0, 1].
pub const Grid = struct {
    width: u32 = 0,
    height: u32 = 0,
    data: []Slug = &.{},

    pub fn at(self: Grid, row: u32, col: u32) Slug {
        return self.data[row * self.width + col];
    }

    pub fn deinit(self: *Grid, gpa: std.mem.Allocator) void {
        gpa.free(self.data);
        self.* = .{};
    }
};

pub const RenderError = error{
    InvalidShapeDimensions,
    UnexpectedTextureFormat,
    TextureReadOutOfBounds,
};

pub const GridOptions = struct {
    size_hint: u32 = 128,
    margin: Slug = 0,
    /// Use the band acceleration structure (what the shader does). False walks every curve, which
    /// is the reference behaviour a correct banding must agree with.
    banded: bool = true,
};

fn clamp(x: Slug, lo: Slug, hi: Slug) Slug {
    return if (x < lo) lo else if (x > hi) hi else x;
}

/// Classifies which roots of a quadratic lie in the relevant range, as a bit code.
///
/// This is the Slug technique's central trick (render.hpp:322, mirroring the GLSL's
/// `slug_CalcRootCode`). Rather than branching on the signs of the three control values, it packs
/// them into a 3-bit index and looks the answer up in the constant `0x2E74` -- a 16-entry table
/// held in an immediate. Bit 0x01 means "root 1 counts", bit 0x100 means "root 2 counts".
///
/// Pure integer bit manipulation, so it ports across GLSL/C++/Zig unchanged.
fn calcRootCode(y1: Slug, y2: Slug, y3: Slug) u32 {
    // s1/s2/s3 rather than upstream's i1/i2/i3: `i1` is a primitive type name in Zig.
    const s1: u32 = @as(u32, @bitCast(y1)) >> 31;
    const s2: u32 = @as(u32, @bitCast(y2)) >> 30;
    const s3: u32 = @as(u32, @bitCast(y3)) >> 29;

    var shift: u32 = (s2 & 0x2) | (s1 & ~@as(u32, 0x2));
    shift = (s3 & 0x4) | (shift & ~@as(u32, 0x4));

    // shift is 3 bits by construction: i1 contributes bit 0, i2 bit 1, i3 bit 2.
    return (@as(u32, 0x2E74) >> @intCast(shift)) & 0x0101;
}

const Roots = struct { r1: Slug, r2: Slug };

/// Solves for where the curve crosses the horizontal ray through the sample point.
///
/// DIVERGENCE (inherited deliberately): the `@abs(by) >= eps` guard on the degenerate branch
/// exists here and in render.hpp:343, but *not* in the reference GLSL (example:135), which
/// computes `0.5/b.y` unconditionally and so yields +/-inf or NaN where this yields 0. Reachable
/// only when `|ay| < eps` and `|by| < eps` -- a near-degenerate, near-horizontal curve.
///
/// This file mirrors render.hpp, so the guard stays. The GPU shader must mirror the GLSL. Expect
/// the two to disagree in that branch; see DIVERGENCE.md.
fn solveHorizPoly(x1: Slug, y1: Slug, x2: Slug, y2: Slug, x3: Slug, y3: Slug) Roots {
    const ax = x1 - 2.0 * x2 + x3;
    const ay = y1 - 2.0 * y2 + y3;
    const bx = x1 - x2;
    const by = y1 - y2;

    if (@abs(ay) < eps) {
        // Degenerate: the quadratic collapsed to a line.
        const t: Slug = if (@abs(by) >= eps) y1 * (0.5 / by) else 0;
        const x = (ax * t - 2.0 * bx) * t + x1;
        return .{ .r1 = x, .r2 = x };
    }

    const d = @sqrt(@max(by * by - ay * y1, 0));
    const t1 = (by - d) / ay;
    const t2 = (by + d) / ay;

    return .{
        .r1 = (ax * t1 - 2.0 * bx) * t1 + x1,
        .r2 = (ax * t2 - 2.0 * bx) * t2 + x1,
    };
}

/// The vertical counterpart of `solveHorizPoly`; same degenerate-branch note applies.
fn solveVertPoly(x1: Slug, y1: Slug, x2: Slug, y2: Slug, x3: Slug, y3: Slug) Roots {
    const ax = x1 - 2.0 * x2 + x3;
    const ay = y1 - 2.0 * y2 + y3;
    const bx = x1 - x2;
    const by = y1 - y2;

    if (@abs(ax) < eps) {
        const t: Slug = if (@abs(bx) >= eps) x1 * (0.5 / bx) else 0;
        const y = (ay * t - 2.0 * by) * t + y1;
        return .{ .r1 = y, .r2 = y };
    }

    const d = @sqrt(@max(bx * bx - ax * x1, 0));
    const t1 = (bx - d) / ax;
    const t2 = (bx + d) / ax;

    return .{
        .r1 = (ay * t1 - 2.0 * by) * t1 + y1,
        .r2 = (ay * t2 - 2.0 * by) * t2 + y1,
    };
}

/// Combines the x and y coverage estimates into one alpha.
///
/// Takes the better of a weighted blend and a conservative min: the weighted term is accurate on
/// clean edges, while the min term rescues corners where one axis alone under-reports.
fn calcCoverage(xcov: Slug, ycov: Slug, xwgt: Slug, ywgt: Slug) Slug {
    const weighted = @abs(xcov * xwgt + ycov * ywgt) / @max(xwgt + ywgt, eps);
    const conservative = @min(@abs(xcov), @abs(ycov));
    return clamp(@max(weighted, conservative), 0, 1);
}

fn lookupBandIndir(coord_scaled: Slug, indir: []const u8) u32 {
    const q = clamp(coord_scaled, 0, @floatFromInt(indirection_size - 1));
    return indir[@intFromFloat(q)];
}

/// A decoded shape, ready to evaluate coverage.
pub const Sampler = struct {
    shape: Shape = .{},
    curves: []Curve = &.{},

    /// Band `b`'s curves are `hband_indices[hband_offsets[b] .. hband_offsets[b + 1]]`.
    hband_offsets: []u32 = &.{},
    hband_indices: []u32 = &.{},
    vband_offsets: []u32 = &.{},
    vband_indices: []u32 = &.{},

    indir_y: [indirection_size]u8 = @splat(0),
    indir_x: [indirection_size]u8 = @splat(0),

    pub fn deinit(self: *Sampler, gpa: std.mem.Allocator) void {
        gpa.free(self.curves);
        gpa.free(self.hband_offsets);
        gpa.free(self.hband_indices);
        gpa.free(self.vband_offsets);
        gpa.free(self.vband_indices);
        self.* = .{};
    }

    /// The em-space origin of the shape's band range.
    pub fn emOrigin(self: Sampler) struct { x: Slug, y: Slug } {
        return .{
            .x = if (self.shape.band_scale_x != 0) -self.shape.band_offset_x / self.shape.band_scale_x else 0,
            .y = if (self.shape.band_scale_y != 0) -self.shape.band_offset_y / self.shape.band_scale_y else 0,
        };
    }

    /// The em-space extent of the shape's band range.
    pub fn emSize(self: Sampler) struct { x: Slug, y: Slug } {
        const isize_f: Slug = @floatFromInt(indirection_size);
        return .{
            .x = if (self.shape.band_scale_x != 0) isize_f / self.shape.band_scale_x else 0,
            .y = if (self.shape.band_scale_y != 0) isize_f / self.shape.band_scale_y else 0,
        };
    }

    pub fn computeRenderSize(self: Sampler, size_hint: u32) RenderError!struct { w: u32, h: u32 } {
        const w = self.shape.width;
        const h = self.shape.height;
        if (w <= 0 or h <= 0) return error.InvalidShapeDimensions;

        const scale = @as(Slug, @floatFromInt(size_hint)) / @max(w, h);
        return .{
            .w = @intFromFloat(@max(1, @round(w * scale))),
            .h = @intFromFloat(@max(1, @round(h * scale))),
        };
    }

    /// Coverage at one em-space point, testing every curve.
    ///
    /// `ppe_x`/`ppe_y` are pixels-per-em: they set the filter width, i.e. how wide the antialiased
    /// edge is. On the GPU these come from `fwidth`; here the caller states them.
    pub fn renderSample(self: Sampler, rx: Slug, ry: Slug, ppe_x: Slug, ppe_y: Slug) Sample {
        var out: Sample = .{};

        for (self.curves) |c| {
            out.iters += 1;

            const x1 = c.x1 - rx;
            const y1 = c.y1 - ry;
            const x2 = c.x2 - rx;
            const y2 = c.y2 - ry;
            const x3 = c.x3 - rx;
            const y3 = c.y3 - ry;

            accumulateH(&out, x1, y1, x2, y2, x3, y3, ppe_x);
            accumulateV(&out, x1, y1, x2, y2, x3, y3, ppe_y);
        }

        out.fill = calcCoverage(out.xcov, out.ycov, out.xwgt, out.ywgt);
        return out;
    }

    /// Coverage at one em-space point, using the bands -- what the shader actually does.
    ///
    /// Must agree with `renderSample` wherever the banding is correct; that equivalence is the
    /// point of the band structure and is what the tests below check.
    pub fn renderSampleBanded(self: Sampler, rx: Slug, ry: Slug, ppe_x: Slug, ppe_y: Slug) Sample {
        var out: Sample = .{};

        if (self.hband_offsets.len < 2 or self.vband_offsets.len < 2) return out;

        const band_x = lookupBandIndir(rx * self.shape.band_scale_x + self.shape.band_offset_x, &self.indir_x);
        const band_y = lookupBandIndir(ry * self.shape.band_scale_y + self.shape.band_offset_y, &self.indir_y);

        if (band_y + 1 < self.hband_offsets.len) {
            for (self.hband_offsets[band_y]..self.hband_offsets[band_y + 1]) |i| {
                const c = self.curves[self.hband_indices[i]];
                out.iters += 1;

                const x1 = c.x1 - rx;
                const y1 = c.y1 - ry;
                const x2 = c.x2 - rx;
                const y2 = c.y2 - ry;
                const x3 = c.x3 - rx;
                const y3 = c.y3 - ry;

                // The early-out the descending sort exists for: once a curve lies entirely left of
                // the sample, so does every curve after it.
                if (@max(x1, @max(x2, x3)) * ppe_x < -0.5) break;

                accumulateH(&out, x1, y1, x2, y2, x3, y3, ppe_x);
            }
        }

        if (band_x + 1 < self.vband_offsets.len) {
            for (self.vband_offsets[band_x]..self.vband_offsets[band_x + 1]) |i| {
                const c = self.curves[self.vband_indices[i]];
                out.iters += 1;

                const x1 = c.x1 - rx;
                const y1 = c.y1 - ry;
                const x2 = c.x2 - rx;
                const y2 = c.y2 - ry;
                const x3 = c.x3 - rx;
                const y3 = c.y3 - ry;

                if (@max(y1, @max(y2, y3)) * ppe_y < -0.5) break;

                accumulateV(&out, x1, y1, x2, y2, x3, y3, ppe_y);
            }
        }

        out.fill = calcCoverage(out.xcov, out.ycov, out.xwgt, out.ywgt);
        return out;
    }

    /// Rasterizes the shape to a coverage grid.
    pub fn renderGrid(self: Sampler, gpa: std.mem.Allocator, opts: GridOptions) RenderError!Grid {
        const size = try self.computeRenderSize(opts.size_hint);

        const o = self.emOrigin();
        const s = self.emSize();

        var ox = o.x - opts.margin * s.x;
        var oy = o.y - opts.margin * s.y;
        const sx = s.x * (1.0 + 2.0 * opts.margin);
        const sy = s.y * (1.0 + 2.0 * opts.margin);
        _ = &ox;
        _ = &oy;

        var grid: Grid = .{
            .width = size.w,
            .height = size.h,
            .data = oom.must(gpa.alloc(Slug, size.w * size.h)),
        };
        @memset(grid.data, 0);

        // NOTE: upstream passes the grid dimensions as pixels-per-em (render.hpp:279) while
        // stepping the sample point by sx/width -- so this `ppe` is really pixels-per-em-span, and
        // only equals pixels-per-em when sx == 1. Preserved as-is to keep the oracle faithful;
        // callers who need true ppe should call renderSampleBanded directly.
        const ppe_x: Slug = @floatFromInt(size.w);
        const ppe_y: Slug = @floatFromInt(size.h);

        for (0..size.h) |j| {
            for (0..size.w) |i| {
                const u = (@as(Slug, @floatFromInt(i)) + 0.5) / @as(Slug, @floatFromInt(size.w));
                const v = (@as(Slug, @floatFromInt(j)) + 0.5) / @as(Slug, @floatFromInt(size.h));
                const ex = ox + u * sx;
                const ey = oy + v * sy;

                const r = if (opts.banded)
                    self.renderSampleBanded(ex, ey, ppe_x, ppe_y)
                else
                    self.renderSample(ex, ey, ppe_x, ppe_y);

                grid.data[j * size.w + i] = r.fill;
            }
        }

        return grid;
    }
};

fn accumulateH(out: *Sample, x1: Slug, y1: Slug, x2: Slug, y2: Slug, x3: Slug, y3: Slug, ppe_x: Slug) void {
    const code = calcRootCode(y1, y2, y3);
    if (code == 0) return;

    const r = solveHorizPoly(x1, y1, x2, y2, x3, y3);
    const r1 = r.r1 * ppe_x;
    const r2 = r.r2 * ppe_x;

    if (code & 0x01 != 0) {
        out.xcov += clamp(r1 + 0.5, 0, 1);
        out.xwgt = @max(out.xwgt, clamp(1.0 - @abs(r1) * 2.0, 0, 1));
    }
    if (code & 0x100 != 0) {
        out.xcov -= clamp(r2 + 0.5, 0, 1);
        out.xwgt = @max(out.xwgt, clamp(1.0 - @abs(r2) * 2.0, 0, 1));
    }
}

fn accumulateV(out: *Sample, x1: Slug, y1: Slug, x2: Slug, y2: Slug, x3: Slug, y3: Slug, ppe_y: Slug) void {
    const code = calcRootCode(x1, x2, x3);
    if (code == 0) return;

    const r = solveVertPoly(x1, y1, x2, y2, x3, y3);
    const r1 = r.r1 * ppe_y;
    const r2 = r.r2 * ppe_y;

    // Note the signs are opposite the horizontal case: the two axes wind oppositely.
    if (code & 0x01 != 0) {
        out.ycov -= clamp(r1 + 0.5, 0, 1);
        out.ywgt = @max(out.ywgt, clamp(1.0 - @abs(r1) * 2.0, 0, 1));
    }
    if (code & 0x100 != 0) {
        out.ycov += clamp(r2 + 0.5, 0, 1);
        out.ywgt = @max(out.ywgt, clamp(1.0 - @abs(r2) * 2.0, 0, 1));
    }
}

/// Reconstructs a `Sampler` from the packed atlas textures.
///
/// This is the inverse of `pack.zig`, and reading the bytes back the way the shader will is what
/// makes it a real check on the packing rather than a restatement of it.
pub fn decode(
    gpa: std.mem.Allocator,
    shape: Shape,
    curve_tex: *const TextureData,
    band_tex: *const TextureData,
) (RenderError || error{OutOfMemory})!Sampler {
    if (curve_tex.format != .rgba32f) return error.UnexpectedTextureFormat;
    if (band_tex.format != .rgba16ui) return error.UnexpectedTextureFormat;

    var out: Sampler = .{ .shape = shape };

    // A shape with no geometry: no bands to decode.
    if (shape.band_scale_x == 0 or shape.band_scale_y == 0) {
        out.hband_offsets = try gpa.dupe(u32, &.{ 0, 0 });
        out.vband_offsets = try gpa.dupe(u32, &.{ 0, 0 });
        return out;
    }

    const shape_start = shape.band_tex_y * band_tex.width + shape.band_tex_x;
    const num_h = shape.band_max_y + 1;
    const num_v = shape.band_max_x + 1;
    const num_hdrs = num_h + num_v;
    const indir_size: u32 = if (num_hdrs > 0) 2 * indirection_size else 0;

    const band_texels: u32 = band_tex.width * band_tex.height;

    const Rd = struct {
        tex: *const TextureData,
        limit: u32,

        fn bandTexel(self: @This(), idx: u32, ch: u32) RenderError!u16 {
            if (idx >= self.limit) return error.TextureReadOutOfBounds;
            const off = (@as(usize, idx) * 4 + ch) * @sizeOf(u16);
            return std.mem.readInt(u16, self.tex.bytes[off..][0..2], .little);
        }
    };
    const rd: Rd = .{ .tex = band_tex, .limit = band_texels };

    for (0..indirection_size) |q| {
        const qi: u32 = @intCast(q);
        out.indir_y[q] = @intCast(try rd.bandTexel(shape_start + qi, 0));
        out.indir_x[q] = @intCast(try rd.bandTexel(shape_start + indirection_size + qi, 0));
    }

    const Header = struct { count: u32, offset: u32 };
    const headers = try gpa.alloc(Header, num_hdrs);
    defer gpa.free(headers);

    for (headers, 0..) |*h, i| {
        const ii: u32 = @intCast(i);
        h.* = .{
            .count = try rd.bandTexel(shape_start + indir_size + ii, 0),
            .offset = try rd.bandTexel(shape_start + indir_size + ii, 1),
        };
    }

    var global: std.ArrayList(u32) = .empty;
    defer global.deinit(gpa);

    var h_offsets: std.ArrayList(u32) = .empty;
    var h_indices: std.ArrayList(u32) = .empty;
    var v_offsets: std.ArrayList(u32) = .empty;
    var v_indices: std.ArrayList(u32) = .empty;
    errdefer {
        h_offsets.deinit(gpa);
        h_indices.deinit(gpa);
        v_offsets.deinit(gpa);
        v_indices.deinit(gpa);
    }

    const Decoder = struct {
        rd: Rd,
        curve_width: u32,
        shape_start: u32,
        headers: []const Header,
        global: *std.ArrayList(u32),
        gpa: std.mem.Allocator,

        fn list(
            self: @This(),
            header_index: u32,
            num_bands: u32,
            offsets: *std.ArrayList(u32),
            indices: *std.ArrayList(u32),
        ) !void {
            try offsets.append(self.gpa, 0);

            for (0..num_bands) |i| {
                const h = self.headers[header_index + i];
                for (0..h.count) |j| {
                    const at: u32 = self.shape_start + h.offset + @as(u32, @intCast(j));
                    const cx = try self.rd.bandTexel(at, 0);
                    const cy = try self.rd.bandTexel(at, 1);
                    // Each curve occupies two texels, so the texel address halves to a curve index.
                    const idx: u32 = (@as(u32, cy) * self.curve_width + cx) / 2;
                    try indices.append(self.gpa, idx);
                    try self.global.append(self.gpa, idx);
                }
                try offsets.append(self.gpa, @intCast(indices.items.len));
            }
        }
    };

    const dec: Decoder = .{
        .rd = rd,
        .curve_width = curve_tex.width,
        .shape_start = shape_start,
        .headers = headers,
        .global = &global,
        .gpa = gpa,
    };

    try dec.list(0, num_h, &h_offsets, &h_indices);
    try dec.list(num_h, num_v, &v_offsets, &v_indices);

    // Collapse the referenced curves to a dense, deterministic set.
    std.mem.sortUnstable(u32, global.items, {}, std.sort.asc(u32));
    const uniq = blk: {
        var n: usize = 0;
        for (global.items) |v| {
            if (n == 0 or global.items[n - 1] != v) {
                global.items[n] = v;
                n += 1;
            }
        }
        break :blk global.items[0..n];
    };

    var remap: std.AutoHashMapUnmanaged(u32, u32) = .empty;
    defer remap.deinit(gpa);

    const curves = try gpa.alloc(Curve, uniq.len);
    errdefer gpa.free(curves);

    const curve_texels: u32 = curve_tex.width * curve_tex.height;

    for (uniq, 0..) |g, i| {
        const t0 = g * 2;
        const t1 = t0 + 1;
        if (t1 >= curve_texels) return error.TextureReadOutOfBounds;

        const f = struct {
            fn read(tex: *const TextureData, texel: u32, ch: u32) Slug {
                const off = (@as(usize, texel) * 4 + ch) * @sizeOf(f32);
                return @bitCast(std.mem.readInt(u32, tex.bytes[off..][0..4], .little));
            }
        };

        curves[i] = .{
            .x1 = f.read(curve_tex, t0, 0),
            .y1 = f.read(curve_tex, t0, 1),
            .x2 = f.read(curve_tex, t0, 2),
            .y2 = f.read(curve_tex, t0, 3),
            .x3 = f.read(curve_tex, t1, 0),
            .y3 = f.read(curve_tex, t1, 1),
        };
        try remap.put(gpa, g, @intCast(i));
    }

    for (h_indices.items) |*idx| idx.* = remap.get(idx.*).?;
    for (v_indices.items) |*idx| idx.* = remap.get(idx.*).?;

    out.curves = curves;
    out.hband_offsets = try h_offsets.toOwnedSlice(gpa);
    out.hband_indices = try h_indices.toOwnedSlice(gpa);
    out.vband_offsets = try v_offsets.toOwnedSlice(gpa);
    out.vband_indices = try v_indices.toOwnedSlice(gpa);

    return out;
}

// ================================================================================================
// Tests
// ================================================================================================

const testing = std.testing;

test "calcRootCode: the 0x2E74 table is a pure function of the three signs" {
    // All-positive and all-negative control values mean the ray misses: no roots count.
    try testing.expectEqual(@as(u32, 0), calcRootCode(1, 1, 1));
    try testing.expectEqual(@as(u32, 0), calcRootCode(-1, -1, -1));

    // A sign change means the curve crosses the ray, so at least one root counts.
    try testing.expect(calcRootCode(-1, -1, 1) != 0);
    try testing.expect(calcRootCode(1, -1, -1) != 0);
    try testing.expect(calcRootCode(1, 1, -1) != 0);
    try testing.expect(calcRootCode(-1, 1, 1) != 0);

    // Only the two documented bits are ever produced.
    for ([_]Slug{ -1, 1 }) |a| {
        for ([_]Slug{ -1, 1 }) |b| {
            for ([_]Slug{ -1, 1 }) |c| {
                try testing.expectEqual(@as(u32, 0), calcRootCode(a, b, c) & ~@as(u32, 0x0101));
            }
        }
    }
}

test "calcRootCode reads raw sign bits, so -0.0 counts as negative" {
    // The trick indexes on sign bits, not on comparisons, so -0.0 and +0.0 select different table
    // entries. Not a quirk of the port -- the GLSL does the same thing.
    //
    // All-same-sign still means "no crossing" either way, so these agree despite selecting
    // opposite ends of the table (shift 0 vs 7):
    try testing.expectEqual(@as(u32, 0), calcRootCode(0.0, 0.0, 0.0));
    try testing.expectEqual(@as(u32, 0), calcRootCode(-0.0, -0.0, -0.0));

    // But in a mixed case the sign bit is decisive: -0.0 reads as a sign change and so registers a
    // crossing, where +0.0 does not.
    try testing.expectEqual(@as(u32, 0), calcRootCode(0.0, 1.0, 1.0));
    try testing.expect(calcRootCode(-0.0, 1.0, 1.0) != 0);
}

test "solveHorizPoly guards the degenerate divide where the GLSL does not" {
    // |ay| < eps and |by| < eps: the C++ yields 0 for t, the reference GLSL yields inf/NaN.
    // We mirror the C++ -- see the doc comment and DIVERGENCE.md.
    const r = solveHorizPoly(1, 0, 2, 0, 3, 0);
    try testing.expect(std.math.isFinite(r.r1));
    try testing.expect(std.math.isFinite(r.r2));
    // t == 0 -> x == x1
    try testing.expectEqual(@as(Slug, 1), r.r1);
}

test "calcCoverage is clamped to [0, 1] and takes the better estimate" {
    try testing.expectEqual(@as(Slug, 0), calcCoverage(0, 0, 0, 0));
    // Fully covered on both axes.
    try testing.expectEqual(@as(Slug, 1), calcCoverage(1, 1, 1, 1));
    // Wild inputs still clamp.
    try testing.expectEqual(@as(Slug, 1), calcCoverage(5, 5, 1, 1));
    // The conservative min rescues a case the weighted blend under-reports: with zero weights the
    // weighted term is ~0, but both axes agree coverage is 1.
    try testing.expectEqual(@as(Slug, 1), calcCoverage(1, 1, 0, 0));
}

test "lookupBandIndir clamps out-of-range coordinates" {
    var indir: [indirection_size]u8 = @splat(0);
    indir[0] = 7;
    indir[indirection_size - 1] = 9;

    try testing.expectEqual(@as(u32, 7), lookupBandIndir(-100, &indir)); // below -> slot 0
    try testing.expectEqual(@as(u32, 9), lookupBandIndir(1000, &indir)); // above -> last slot
    try testing.expectEqual(@as(u32, 7), lookupBandIndir(0.5, &indir));
}
