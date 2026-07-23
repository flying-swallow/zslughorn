// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! Domain errors, and the diagnostic channel that carries the detail Zig error values cannot.
//!
//! Note what is absent: `error.OutOfMemory` -- allocation failure panics (see the note in `slughorn.zig`).

const std = @import("std");
const Key = @import("key.zig").Key;

/// Errors raised by `Atlas.build()`.
///
/// These map to the upstream C++ `throw` sites, plus two hardenings (`TooManyBands`, and the
/// promotion of a debug-only assert to a real error in `Error.InvalidTextureWidth`).
pub const BuildError = error{
    /// A band's curve-index list is wider than one texture row. (slughorn.cpp:1428)
    ///
    /// Hard limit, not a tuning issue -- see the comment on `pack.BandPacker.pack`. Raise the
    /// atlas's texture width to the next power of two >= the band's curve count.
    BandExceedsTextureRow,
    /// A band's curve count exceeds the uint16 band-header field. (slughorn.cpp:1439)
    BandCountOverflow,
    /// A shape's band data spans more than 65535 texels from its block start. (slughorn.cpp:1456)
    BandOffsetOverflow,
    /// A shape's header block alone is wider than one texture row. (slughorn.cpp:1378)
    HeaderBlockExceedsTextureRow,
    /// A shape has more than 255 bands on an axis.
    ///
    /// DIVERGENCE from upstream, which silently truncates: the indirection tables store a band
    /// index per slot as uint8 (`static_cast<uint8_t>(band)` at slughorn.cpp:1083 and :1152) with
    /// no guard, so band 256 wraps to 0 and the shader reads the wrong band. We reject instead.
    TooManyBands,
};

/// Every error this library can return.
pub const Error = BuildError || error{
    /// No shape/composite is registered under the given key. (slughorn.hpp:549,555)
    KeyNotFound,
    /// Texture width is zero or not a power of two.
    ///
    /// DIVERGENCE from upstream, where this is a bare `assert` (slughorn.cpp:196-201) that
    /// compiles out under NDEBUG. The shader uses log2(texWidth) as a bit-shift count, so a
    /// non-power-of-two width silently corrupts band coordinate wrapping. Always checked here.
    InvalidTextureWidth,
    /// A mutating call arrived after `build()`.
    ///
    /// DIVERGENCE from upstream, which silently ignores these (`if(_built) return;`).
    AtlasAlreadyBuilt,
    /// A curve coordinate is NaN or infinite.
    ///
    /// Rejected at ingest because the band sort comparator is a bare `>`, which is not a strict
    /// weak ordering in the presence of NaN -- upstream's `std::sort` has undefined behaviour
    /// (it can read out of bounds) on such input.
    NonFiniteCoordinate,
};

/// Optional structured detail for the most recent error.
///
/// Zig error values carry no payload, but the upstream messages are worth keeping ("X's band has
/// 738 curves, which does not fit in a texture row of width 512"). Attach one of these to an
/// `Atlas` to receive the same detail. Follows the pattern std uses for its own diagnostics.
pub const Diagnostics = struct {
    /// The shape being processed when the error was raised.
    key: ?Key = null,
    /// Context-dependent: the offending band's curve count, band count, or texel span.
    count: u32 = 0,
    /// The atlas texture width in effect.
    tex_width: u32 = 0,
    /// For `BandExceedsTextureRow`: the smallest texture width that would accept this shape.
    suggested_tex_width: u32 = 0,

    pub fn reset(self: *Diagnostics) void {
        self.* = .{};
    }

    /// Renders a human-readable message for `err` from the recorded fields.
    pub fn format(self: Diagnostics, err: Error, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        switch (err) {
            error.BandExceedsTextureRow => try writer.print(
                "band has {d} curves, which does not fit in a texture row of width {d}; " ++
                    "increase the atlas texture width to at least {d}",
                .{ self.count, self.tex_width, self.suggested_tex_width },
            ),
            error.BandCountOverflow => try writer.print(
                "band has {d} curves, exceeding the uint16 band-header capacity (65535)",
                .{self.count},
            ),
            error.BandOffsetOverflow => try writer.print(
                "shape band data spans {d} texels, exceeding the uint16 band-header offset capacity (65535)",
                .{self.count},
            ),
            error.HeaderBlockExceedsTextureRow => try writer.print(
                "shape header block is {d} texels, wider than one texture row of {d}",
                .{ self.count, self.tex_width },
            ),
            error.TooManyBands => try writer.print(
                "shape has {d} bands on one axis, exceeding the 255 the uint8 indirection table can address",
                .{self.count},
            ),
            error.InvalidTextureWidth => try writer.print(
                "texture width {d} is not a positive power of two",
                .{self.tex_width},
            ),
            error.KeyNotFound => try writer.writeAll("no shape registered under that key"),
            error.AtlasAlreadyBuilt => try writer.writeAll("the atlas has already been built"),
            error.NonFiniteCoordinate => try writer.writeAll("curve coordinate is NaN or infinite"),
        }
    }
};
