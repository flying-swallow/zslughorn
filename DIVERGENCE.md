# Divergences from upstream C++ slughorn

Deliberate, tested differences between this port and
[AlphaPixel/slughorn](https://github.com/AlphaPixel/slughorn). Anything here is a decision, not an
accident; the golden fixtures pin everything else to byte equality.

## Upstream bugs this port refuses to reproduce

### 1. Indirection-table truncation past 256 bands

`slughorn.cpp:1083` (`indirY`) and `slughorn.cpp:1152` (`indirX`) write the band index with
`static_cast<uint8_t>(band)` and never check it fits. A shape with more than 256 bands on an axis
therefore wraps -- band 256 becomes band 0 -- and the shader silently reads the wrong band's
curves. Wrong pixels, no diagnostic.

Reachable in practice: `splitsY` with 256+ entries produces 257+ bands. It also needs
`texWidth >= 1024` to get past the header-block check at `slughorn.cpp:1378`, so it is not purely
theoretical.

**Here:** `bands.buildShapeBands` returns `error.TooManyBands` when either axis exceeds
`max_bands` (256 -- the most a `u8` index can address). Covered by a unit test in `src/bands.zig`.

**Worth reporting upstream.**

### 2. The power-of-two texture width check vanishes in release builds

`slughorn.cpp:196-201` guards the invariant with a bare `assert`, which compiles out under
`NDEBUG`. The shader uses `log2(texWidth)` as a bit-shift count, so a non-power-of-two width
silently corrupts band coordinate wrapping in a release build -- exactly where it matters.

**Here:** always checked; `Atlas.init` returns `error.InvalidTextureWidth`.

### 3. Band sorts are unstable, with a comparator that is only a partial order

`slughorn.cpp:1055` (horizontal, by max-x) and `slughorn.cpp:1125` (vertical, by max-y) call
`std::sort` -- unstable -- with a bare `>` comparator. Curves with equal extents may come out in
any order, so the packed bytes depend on the standard library's pivot choices.

This is a reproducible-build bug independent of the port: libstdc++ and libc++ can already disagree
with each other today. Measured against libstdc++ (`n` = count of tied keys):

| n | tied keys keep their original order? |
|---|---|
| ≤ 16 | yes (insertion sort) |
| ≥ 17 | **no** (introsort) |

**Here:** the comparator breaks ties on curve index, making the order total, so every correct sort
agrees. This is semantically free -- the descending sort exists only to let the shader stop
scanning a band early, so any order among equal-extent curves is equally valid. Given upstream
pushes indices ascending (`slughorn.cpp:1053`/`:1116`), a total order is exactly what
`std::stable_sort` would produce there.

Consequence for testing: fixtures deliberately avoid ties above n=16, because at those sizes the
C++ output is an artifact of one standard library rather than a specification. `ties_in_sort`
covers ties at n=8, inside the range where libstdc++ is stable and the two agree exactly.

### 4. NaN input is undefined behaviour

The bare `>` comparator is not a strict weak ordering when a coordinate is NaN, so upstream's
`std::sort` has UB (it can read out of bounds).

**Here:** rejected at ingest with `error.NonFiniteCoordinate`.

## API changes

### Allocation failure panics

Upstream throws `std::bad_alloc` and never catches it. This port panics instead of returning
`error.OutOfMemory`, so public error sets contain only errors a caller can act on:
`oom.must(list.append(gpa, c))`, which is `catch @panic` with the message in one place.

`must` takes an `Allocator.Error` union and switches exhaustively over `error{OutOfMemory}`, so
handing it a richer error set is a compile error rather than a real failure misreported as OOM.
Nothing wraps the caller's allocator: `Atlas` stores and uses the allocator it was given.

Second-order effect: because path operations no longer return errors, `CurveDecomposer`'s fluent
chaining survives the port -- `d.moveTo(0, 0).lineTo(1, 1).close()`.

### Transform types are built on zml

`Color` is a `zml.Vec4f32` (i.e. `@Vector(4, f32)`), not upstream's `{r, g, b, a}` struct
(slughorn.hpp:85). Components are indexed `[0]`..`[3]`; `types.rgb`/`types.rgba` construct one, with
`rgb` preserving the `a = 1` default the struct carried. The trade is field names for arithmetic:
gradient interpolation (M4) becomes `zml.scalar.lerp(c0, c1, t)`, and `zml.color`'s colour-science
helpers are free functions over vectors that then compose directly.

`Matrix` keeps upstream's six-float layout -- it is the form M4 uploads as a gradient transform, and
zml has no 2D affine type -- but `apply` now takes and returns a `zml.Vec2f32` instead of loose
`(x, y)` scalars. `applyDir` is new: the same transform with translation dropped, for tangents and
deltas.

Deliberately **not** built on zml: `Curve` (a GPU wire format -- an `extern struct` memcmp'd against
the golden fixtures, where `Vec2f32` would raise alignment from 4 to 8 for no gain), and everything
in `render.zig`/`bands.zig`/`pack.zig`. The hot path is scalar float math pinned bit-exactly to the
C++, and at least one zml primitive would silently change results: `zml.scalar.clamp` is
`@min(@max(x, lo), hi)`, which returns `0` for `clamp(NaN, 0, 1)`, where render.hpp's ternary chain
(render.hpp:318) returns `NaN`. The solvers can produce NaN in degenerate cases, so the two are not
interchangeable.

### Silent no-ops became errors

`addShape` and friends upstream begin with `if(_built) return;` (`slughorn.cpp:268` and others),
silently discarding the call. Here that is `error.AtlasAlreadyBuilt`.

### Shape lookup borrows instead of copying

`Atlas::getShape` returns `std::optional<Shape>` (`slughorn.hpp:1151`), copying the whole retained
curve vector on every call. Here it returns `?*const Shape`. A by-value Zig copy would alias the
curves slice and double-free.

### Key drops its cached hash

Upstream precomputes and stores a hash in every `Key` (`slughorn.hpp:249-253`) to speed libstdc++'s
hashing and fast-path `operator==`. The value is not observable in the output, and
`std.array_hash_map` with `store_hash = true` already caches hashes in the table -- the same
optimization, from std, without a field that has to be kept in sync with the payload.

### Shape storage is insertion-ordered

Upstream stores shapes in a `std::unordered_map` (`slughorn.hpp:1346`) and iterates it to drive
texture packing (`slughorn.cpp:1173`, `:1194`, `:1244`, `:1337`), while band texels record
*absolute* curve-texture coordinates (`slughorn.cpp:1471-1479`). So a shape's packed block encodes
where previously iterated shapes landed: per-shape blocks are not relocatable, and the entire byte
layout is a function of libstdc++'s bucket order.

**Here:** shapes live in an insertion-ordered map, making the output deterministic and independent
of hash values.

This is why golden testing is tiered:

* **Tier A -- single shape.** Iteration order is trivially deterministic, so the packed textures are
  compared byte-for-byte. Carries essentially all the invariant coverage.
* **Tier B -- multiple shapes.** Compared semantically (via `decode`), because byte equality with
  the C++ would be asserting a property of libstdc++'s hash table, not of slughorn.

### render.zig mirrors render.hpp, not the GLSL

`render.hpp` is described as mirroring the fragment shader, but the two genuinely differ in one
branch. `_solveHorizPoly`/`_solveVertPoly` (`render.hpp:343`, `:364`) guard the degenerate divide:

```cpp
const slug_t t = std::abs(by) >= EPS ? y1 * (0.5_cv / by) : 0_cv;
```

The reference GLSL (`example/slughorn-example-glfw.cpp:135`) does not -- it computes `0.5/b.y`
unconditionally, yielding `±inf`/NaN where the CPU yields 0. Reachable only when `|ay| < EPS` and
`|by| < EPS`: a near-degenerate, near-horizontal curve. (`EPS = 1/65536` matches on both sides.)

`src/render.zig` mirrors **render.hpp**, guard included, because it exists to be the oracle the
golden fixtures are compared against. The M3 Slang shader must mirror the **GLSL** instead. Expect
the two to disagree in that branch, and treat such outliers as expected divergence rather than port
bugs. The `degenerate_collinear` fixture exists to keep the case measured.

Measured result: the coverage grids agree with the C++ **100% bit-exactly** (not merely within
tolerance) on every case, including the 512-curve adaptive-subdivision one. The test gates on a
1e-5 tolerance -- `sqrt` and divide are correctly-rounded IEEE ops so exactness is achievable, but
pinning it would be brittle for no safety gain -- and reports the bit-exact rate as a non-fatal
metric so any future drift is visible.

## Float behaviour

The port is bit-exact against the C++ *when the C++ is built with `-ffp-contract=off`*, which is
what `zig build fixtures` does. This is not a formality:

Building the dumper with `-ffp-contract=fast -march=native` changes `decompose_cubic_fine`'s output.
`CurveDecomposer::_pointToLineDistSq` (`slughorn.hpp:1606`) computes `dx*dx + dy*dy` and a cross
product; contracting those into FMAs changes which cubics test as flat, which changes how they
subdivide. Band boundaries (`minY + snapped * rangeY`, `slughorn.cpp:1029`/`:1036`/`:1080`) are
likewise contraction-sensitive, and they feed comparisons (`maxY >= lo`, `slughorn.cpp:165`) -- so a
1-ULP shift can move a curve between bands and change the *integer* output. A random probe puts the
FMA-vs-mul+add divergence rate for that expression shape at ~24% of inputs.

Zig's float mode is `.strict` and does not contract, so pinning the C++ to match is what makes the
comparison meaningful. `zig build fixtures-verify` checks the checked-in fixtures still match a
fresh build.
