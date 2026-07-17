//! Raw nanosvg C bindings.
//!
//! `c` is the translate-c'd header. This package intentionally passes it through untouched -- the
//! idiomatic wrapper lives in zslughorn's `src/backends/nanosvg.zig`, which is where the C types
//! get turned into slughorn curves.

pub const c = @import("c");
