// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! FreeType front-end: font glyph outlines -> slughorn curves.
//!
//! Ported from `slughorn/freetype.hpp` (the monochrome path). Like the other backends it imports
//! `slughorn`, never the reverse: FreeType is C and needs libc, kept out of the MIT core.
//!
//! A glyph is loaded with `FT_LOAD_NO_SCALE` (raw font units), its outline walked by
//! `FT_Outline_Decompose` into a `CurveDecomposer`, and everything scaled by `1/units_per_EM` into
//! em-space. Metrics come straight from FreeType (`auto_metrics = false`), and the shape is keyed by
//! its Unicode codepoint.
//!
//! Scope: monochrome outline glyphs. COLR/emoji color glyphs (which upstream turns into a
//! `CompositeShape` of `Layer`s), the uniform two-pass cell layout, and the font-metric tables are
//! not ported. Contour orientation is inherited verbatim from FreeType -- upstream reverses nothing,
//! and the winding-agnostic coverage rule handles it downstream.

const std = @import("std");
const slughorn = @import("slughorn");
const c = @import("freetype").c;

const Slug = slughorn.Slug;
const Curve = slughorn.Curve;
const Atlas = slughorn.Atlas;
const ShapeInfo = slughorn.ShapeInfo;
const CurveDecomposer = slughorn.CurveDecomposer;

pub const Error = error{ FreeTypeInit, FaceOpen };

/// A FreeType library instance. One per thread of use; open faces from it.
pub const Library = struct {
    handle: c.FT_Library,

    pub fn init() Error!Library {
        var handle: c.FT_Library = undefined;
        if (c.FT_Init_FreeType(&handle) != 0) return error.FreeTypeInit;
        return .{ .handle = handle };
    }

    pub fn deinit(self: Library) void {
        _ = c.FT_Done_FreeType(self.handle);
    }

    /// Opens the font at `path` (face index 0). `path` must be NUL-terminated.
    pub fn openFace(self: Library, path: [:0]const u8) Error!Face {
        var handle: c.FT_Face = undefined;
        if (c.FT_New_Face(self.handle, path.ptr, 0, &handle) != 0) return error.FaceOpen;
        return .{ .handle = handle };
    }
};

/// A loaded font face.
pub const Face = struct {
    handle: c.FT_Face,

    pub fn deinit(self: Face) void {
        _ = c.FT_Done_Face(self.handle);
    }

    pub fn unitsPerEM(self: Face) u16 {
        return self.handle.*.units_per_EM;
    }

    /// Loads the glyph for `codepoint`, decomposes its outline into em-space curves, and registers it
    /// with `atlas` under the codepoint key. Returns false when the codepoint has no glyph or the
    /// glyph is not an outline (e.g. a bitmap / color glyph). Ported from `loadGlyph`,
    /// freetype.hpp:1105.
    pub fn loadGlyph(self: Face, gpa: std.mem.Allocator, atlas: *Atlas, codepoint: u21) !bool {
        const face = self.handle;

        const glyph_index = c.FT_Get_Char_Index(face, codepoint);
        if (glyph_index == 0 and codepoint != 0) return false; // .notdef allowed only for cp 0
        if (c.FT_Load_Glyph(face, glyph_index, c.FT_LOAD_NO_SCALE) != 0) return false;
        if (face.*.glyph.*.format != c.FT_GLYPH_FORMAT_OUTLINE) return false;

        const em_scale: Slug = 1.0 / @as(Slug, @floatFromInt(face.*.units_per_EM));

        var curves: std.ArrayList(Curve) = .empty;
        defer curves.deinit(gpa); // addShape copies what it keeps; free our temp either way

        var decomposer = CurveDecomposer.init(gpa, &curves);
        var ctx: OutlineContext = .{ .decomposer = &decomposer, .scale = em_scale };
        const funcs: c.FT_Outline_Funcs = .{
            .move_to = &ftMoveTo,
            .line_to = &ftLineTo,
            .conic_to = &ftConicTo,
            .cubic_to = &ftCubicTo,
            .shift = 0,
            .delta = 0,
        };
        // FT_Outline_Decompose auto-closes each contour (emits the closing line), so no close() call.
        if (c.FT_Outline_Decompose(&face.*.glyph.*.outline, &funcs, &ctx) != 0) return false;

        const m = face.*.glyph.*.metrics;
        const info: ShapeInfo = .{
            .curves = curves.items,
            .auto_metrics = false,
            .bearing_x = @as(Slug, @floatFromInt(m.horiBearingX)) * em_scale,
            .bearing_y = @as(Slug, @floatFromInt(m.horiBearingY)) * em_scale,
            .width = @as(Slug, @floatFromInt(m.width)) * em_scale,
            .height = @as(Slug, @floatFromInt(m.height)) * em_scale,
            .advance = @as(Slug, @floatFromInt(m.horiAdvance)) * em_scale,
        };
        try atlas.addShape(.{ .codepoint = codepoint }, info);
        return true;
    }
};

const OutlineContext = struct {
    decomposer: *CurveDecomposer,
    scale: Slug,
};

/// Font-unit coordinate -> em-space.
fn e(ctx: *OutlineContext, v: c.FT_Pos) Slug {
    return @as(Slug, @floatFromInt(v)) * ctx.scale;
}

fn ctxOf(user: ?*anyopaque) *OutlineContext {
    return @ptrCast(@alignCast(user.?));
}

fn ftMoveTo(to: [*c]const c.FT_Vector, user: ?*anyopaque) callconv(.c) c_int {
    const ctx = ctxOf(user);
    _ = ctx.decomposer.moveTo(e(ctx, to.*.x), e(ctx, to.*.y));
    return 0;
}

fn ftLineTo(to: [*c]const c.FT_Vector, user: ?*anyopaque) callconv(.c) c_int {
    const ctx = ctxOf(user);
    _ = ctx.decomposer.lineTo(e(ctx, to.*.x), e(ctx, to.*.y));
    return 0;
}

fn ftConicTo(control: [*c]const c.FT_Vector, to: [*c]const c.FT_Vector, user: ?*anyopaque) callconv(.c) c_int {
    const ctx = ctxOf(user);
    _ = ctx.decomposer.quadTo(e(ctx, control.*.x), e(ctx, control.*.y), e(ctx, to.*.x), e(ctx, to.*.y));
    return 0;
}

fn ftCubicTo(c1: [*c]const c.FT_Vector, c2: [*c]const c.FT_Vector, to: [*c]const c.FT_Vector, user: ?*anyopaque) callconv(.c) c_int {
    const ctx = ctxOf(user);
    _ = ctx.decomposer.cubicTo(
        e(ctx, c1.*.x),
        e(ctx, c1.*.y),
        e(ctx, c2.*.x),
        e(ctx, c2.*.y),
        e(ctx, to.*.x),
        e(ctx, to.*.y),
    );
    return 0;
}
