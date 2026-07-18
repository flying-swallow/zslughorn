//! Wrapper package for the system FreeType. The translate-c'd headers are re-exported as `c`; the
//! FT_* symbols are resolved from the system `libfreetype` at link time (see build.zig).
pub const c = @import("c");
