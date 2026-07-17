// The implementation half of nanosvg.
//
// nanosvg is a single-header library: this macro is what expands the ~2900-line body, and it must
// be defined in exactly one translation unit. The declarations half is handled separately, by
// running translate-c over the *bare* header (see build.zig) -- with the macro undefined it sees
// only prototypes and never tries to translate the implementation.
//
// The header self-includes string.h/stdlib.h/stdio.h/math.h inside its own guard, so nothing else
// is needed here. The rasterizer (nanosvgrast.h) is deliberately not compiled: slughorn consumes
// vector paths and never rasterizes.

#define NANOSVG_IMPLEMENTATION
#include "nanosvg.h"
