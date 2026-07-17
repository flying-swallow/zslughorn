// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! slughorn -- a Zig port of AlphaPixel's C++ slughorn, which implements Eric Lengyel's Slug
//! GPU vector rendering technique.
//!
//! This library contains **no GPU code**, by design and exactly like upstream. It is a data
//! compiler: you feed it quadratic Bezier curves, it emits packed pixel buffers (`TextureData`)
//! plus per-shape metadata. Uploading those to a GPU and running the Slug shader is the caller's
//! job -- see the separate `slughorn-rhi` module for an rhi-zig frontend.
//!
//! Typical use:
//!
//!     var atlas = try slughorn.Atlas.init(gpa, 512);
//!     defer atlas.deinit();
//!     try atlas.addShape(.{ .codepoint = 'A' }, .{ .curves = curves });
//!     try atlas.build();
//!     const curve_tex = atlas.getCurveTextureData();
//!     const band_tex = atlas.getBandTextureData();
//!
//! Note on errors: allocation failure **panics** rather than being returned. See `oom.zig`.

const std = @import("std");
const build_options = @import("build_options");

/// The math library the public types are built on, re-exported so callers get `Vec2f32` and friends
/// without declaring a second dependency -- and, more importantly, so they get *this* copy of it.
/// Two `zml` instances would mean two structurally identical but incompatible vector types.
pub const zml = @import("zml");

pub const oom = @import("oom.zig");

const errors = @import("errors.zig");
pub const Error = errors.Error;
pub const BuildError = errors.BuildError;
pub const Diagnostics = errors.Diagnostics;

const key_mod = @import("key.zig");
pub const Key = key_mod.Key;
pub const KeyContext = key_mod.KeyContext;
pub const KeyMap = key_mod.KeyMap;

const decompose = @import("decompose.zig");
pub const CurveDecomposer = decompose.CurveDecomposer;
pub const reverseCurves = decompose.reverseCurves;
pub const tolerance_draft = decompose.tolerance_draft;
pub const tolerance_balanced = decompose.tolerance_balanced;
pub const tolerance_fine = decompose.tolerance_fine;
pub const tolerance_exact = decompose.tolerance_exact;

const types = @import("types.zig");
pub const Slug = types.Slug;
pub const Curve = types.Curve;
pub const Color = types.Color;
pub const rgba = types.rgba;
pub const rgb = types.rgb;
pub const Matrix = types.Matrix;
pub const Transform = types.Transform;
pub const GradientType = types.GradientType;
pub const GradientStop = types.GradientStop;
pub const GradientInfo = types.GradientInfo;
pub const Quad = types.Quad;
pub const Origin = types.Origin;
pub const ShapeInfo = types.ShapeInfo;
pub const Shape = types.Shape;
pub const TextureData = types.TextureData;
pub const PackingStats = types.PackingStats;

pub const bands = @import("bands.zig");
pub const max_bands = bands.max_bands;

pub const pack = @import("pack.zig");

pub const render = @import("render.zig");

const atlas_mod = @import("atlas.zig");
pub const Atlas = atlas_mod.Atlas;
pub const default_texture_width = atlas_mod.default_texture_width;

/// Indirection-table slots per axis.
///
/// An ABI constant of the packed band format shared with the fragment shader, sourced from a
/// single definition site in build.zig. Upstream keeps this as a `static constexpr` with a TODO
/// (slughorn.hpp:799-802) admitting nothing enforces that the shader agrees; here both sides are
/// generated from the same value.
pub const indirection_size: u32 = build_options.indirection_size;

comptime {
    // The indirection tables occupy 2 * indirection_size texels at the head of every shape's band
    // block, so they must fit within a texture row for even the narrowest legal atlas.
    std.debug.assert(indirection_size > 0);
    std.debug.assert(indirection_size <= 128);
}

test {
    std.testing.refAllDecls(@This());
}
