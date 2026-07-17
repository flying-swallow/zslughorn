# zslughorn

A Zig port of [slughorn](https://github.com/AlphaPixel/slughorn) (MIT © AlphaPixel LLC), which
implements Eric Lengyel's **Slug** GPU vector-graphics technique: quadratic Bezier curves are
evaluated exactly, per-fragment, in the shader instead of being tessellated into triangles. Edges
stay perfect at any zoom, transforms are free, and perspective is exact.

**This library contains no GPU code** — by design, exactly like upstream. It is a *data compiler*:
you feed it curves, it emits packed pixel buffers plus per-shape metadata. Uploading those to a GPU
and running the Slug shader is the caller's job. (Upstream pairs with `osgSlug`; this one is built
to pair with [rhi-zig](https://github.com/flying-swallow/rhi-zig).)

> **Status:** Milestone 1. The atlas compiler is complete and validated byte-for-byte against the
> upstream C++. No backends and no renderer yet — see [Roadmap](#roadmap).

## Usage

```zig
const slughorn = @import("slughorn");

var atlas = try slughorn.Atlas.init(gpa, 512); // width must be a power of two
defer atlas.deinit();

try atlas.addShape(.{ .codepoint = 'A' }, .{ .curves = my_curves });
try atlas.build(); // one-shot; the atlas is frozen afterwards

const curves = atlas.getCurveTextureData(); // RGBA32F -> upload as a texture
const bands = atlas.getBandTextureData(); // RGBA16UI -> upload as a texture
```

Build shapes from paths with `CurveDecomposer`:

```zig
var list: std.ArrayList(slughorn.Curve) = .empty;
var d = slughorn.CurveDecomposer.init(gpa, &list);
_ = d.moveTo(0, 0).lineTo(1, 0).cubicTo(1, 0.5, 0.5, 1, 0, 1).close();
```

Render on the CPU with no GPU at all — useful for tests and for validating a shader:

```zig
var s = try slughorn.render.decode(gpa, shape.*, curves, bands);
defer s.deinit(gpa);
var grid = try s.renderGrid(gpa, .{ .size_hint = 128 });
defer grid.deinit(gpa);
// grid.at(row, col) is coverage in [0, 1]
```

### Errors

Allocation failure **panics** rather than being returned: OOM is unrecoverable here, and
propagating it would bury the errors you can actually act on. So the error set is small and every
member is actionable — a band that does not fit a texture row, a uint16 overflow, a non-power-of-two
width. Attach a `Diagnostics` to get the detail Zig error values cannot carry:

```zig
var diag: slughorn.Diagnostics = .{};
atlas.diag = &diag;
atlas.build() catch |err| {
    // e.g. BandExceedsTextureRow: count=738, tex_width=512, suggested_tex_width=1024
};
```

## Testing

```
zig build test                                   # no C++ toolchain needed
zig build fixtures -Dslughorn-src=../slughorn    # regenerate golden fixtures from the C++
zig build fixtures-verify -Dslughorn-src=../slughorn
```

Correctness is anchored to the original rather than to hand-written expectations. A small C++
dumper runs the *real* upstream slughorn and serializes its output into `fixtures/*.slgf`, which
are checked in; the Zig tests compare against those. Two tiers, for a reason:

- **Tier A (single shape)** — packed textures compared **byte-for-byte**, plus all per-shape
  metadata and packing stats. Carries essentially all the invariant coverage.
- **Tier B (multiple shapes)** — compared **semantically**, because upstream's byte layout depends
  on `std::unordered_map` iteration order. See [DIVERGENCE.md](DIVERGENCE.md).

The C++ `render.hpp` is a CPU emulator of the fragment shader, so porting it gives a free oracle:
`decode` reads the packed bytes back the way the shader will, which validates the banding and
packing end-to-end with no GPU.

Fixtures must be generated with `-ffp-contract=off` (which `zig build fixtures` does). This is not
a formality — see [DIVERGENCE.md](DIVERGENCE.md#float-behaviour).

## Divergences

The port deliberately does not reproduce several upstream behaviours, including two real bugs (a
silent `uint8` truncation past 256 bands, and unstable band sorts). All are documented and tested
in **[DIVERGENCE.md](DIVERGENCE.md)**.

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| **M1** | Atlas compiler + golden tests | **done** |
| M2 | Backends: Canvas → FreeType → NanoSVG | NanoSVG paths done; Canvas/FreeType planned |
| M3 | rhi-zig renderer + Slang shader (Vulkan-only) | planned |
| M4 | Gradients, MSDF, compositing | MSDF generation done (msdf-zig); gradients/compositing planned |

M3 will live in a separate module: the core has no graphics dependencies and stays MIT, while the
renderer links rhi-zig (GPL-2.0). See [LICENSE](LICENSE).

## NanoSVG backend

Off by default, mirroring upstream's `SLUGHORN_NANOSVG=OFF`:

```sh
zig build -Dnanosvg=true          # build the backend module
zig build test -Dnanosvg=true     # and run its tests
```

With the option unset, [nanosvg](https://github.com/memononen/nanosvg) is never even downloaded, so
the default build stays pure Zig and needs no network beyond `zml`. The dependency lives in
`deps/nanosvg/` as a wrapper package — build glue and a four-line implementation shim, with the
upstream source fetched by URL+hash rather than vendored. It is pinned to the same commit
`slughorn/ext/nanosvg` tracks, so the C++ and this port parse identically.

`slughorn_nanosvg` is a **separate module** that imports `slughorn`, never the reverse: nanosvg is C
and pulls in libc, and that must not reach the core. Same seam M3 uses for licensing.

```zig
const svg = @import("slughorn_nanosvg");

const image = svg.parseFromMemory(gpa, source, 96);
defer image.deinit();

// NanoSVG never reports bad input -- garbage yields a zero-width image, so this is the check.
const scale = image.scale() orelse return error.NotAnSvg;

var it = image.shapes();
while (it.next()) |shape| {
    if (!svg.isVisible(shape) or !svg.isFilled(shape)) continue;
    _ = try svg.loadShape(gpa, &atlas, shape, .{ .name = "…" }, scale, .default, true, 1);
}
```

Scope: path geometry, including the `fill-rule="evenodd"` → nonzero winding conversion. Gradients,
strokes, and the `CompositeShape`/`Layer` compositing API are not ported — they need core types that
do not exist yet (M4).

## SDF / MSDF backend

Off by default, mirroring upstream's `SLUGHORN_MSDF=OFF`:

```sh
zig build -Dmsdf=true             # build the backend module
zig build test -Dmsdf=true        # and run its tests
```

Turns a built shape's retained curves into single-channel SDF or three-channel MSDF tiles — the
`renderSDF`/`renderMSDF`/`renderMSDFTile` family from upstream's `render.hpp`, which upstream backs
with Chlumsky's C++ [msdfgen](https://github.com/Chlumsky/msdfgen). Here the generator is the user's
[msdf-zig](https://github.com/flying-swallow/msdf-zig) (a pure-Zig msdfgen port), consumed through
its **FreeType-free `msdf-core`** module (`.font = false`). So the curves → distance-field path
pulls in no FreeType: `-Dmsdf=true` fetches no C at all.

```zig
const sdf = @import("slughorn_sdf");

// tile_size = longest axis in texels; range = em-space spread mapped to the full [0,1] output.
var grid = (try sdf.renderSDF(gpa, &atlas, .{ .name = "A" }, 128, 0.1)) orelse return;
defer grid.deinit(gpa);
// grid.at(x, y, 0): edge ~= 0.5, interior > 0.5, exterior < 0.5. Row 0 = top.
```

`slughorn_sdf` is a **separate module** that imports `slughorn`, never the reverse — MSDF stays out
of the MIT core, upstream's default-off side channel. Scope: the generation primitives only.
Upstream's atlas-level machinery (`rasterizeSDFAtlas`, `requestMSDF`, the `Shape.msdfLayer` field,
serialization) is not ported.

## Requirements

- Zig 0.17.0-dev (tracks nightly)
- `g++` with C++20 — **only** to regenerate fixtures, never to build or test
- Nothing extra for `-Dnanosvg=true`: the C is compiled by Zig itself
- `-Dmsdf=true` needs a `../msdf-zig` checkout (sibling path dependency); it compiles no C either
