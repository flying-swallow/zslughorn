// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT
//
// Golden-fixture dumper: runs the *real* upstream C++ slughorn and serializes its atlas output so
// the Zig port can be compared against it byte-for-byte.
//
// Build (see build.zig, which owns the canonical command):
//
//   g++ -std=c++20 -O2 -ffp-contract=off -fno-fast-math -march=x86-64 \
//       tools/dump/slughorn_dump.cpp <src>/slughorn/slughorn.cpp -I<src> -o slughorn-dump
//
// The float flags are load-bearing -- see the comment in build.zig.
//
// NOTE ON assert(): we deliberately do NOT define NDEBUG. Upstream guards the power-of-two texture
// width with a bare assert (slughorn.cpp:196-201), which would abort this process rather than
// throw. Those cases therefore have no fixture and are tested Zig-side only; every case here uses
// a legal width.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "slughorn/slughorn.hpp"
#include "slughorn/render.hpp"

using slughorn::Atlas;
using slughorn::Key;

// ================================================================================================
// Writer -- field-by-field, little-endian.
//
// Never memcpy a struct: that would couple the fixture format to C++ ABI padding and to this
// compiler's layout decisions. Every field is written explicitly.
// ================================================================================================

struct Writer {
	std::vector<uint8_t> buf;

	void u8v(uint8_t v) { buf.push_back(v); }

	void u32v(uint32_t v) {
		for(int i = 0; i < 4; i++) buf.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xff));
	}

	void u64v(uint64_t v) {
		for(int i = 0; i < 8; i++) buf.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xff));
	}

	// Written via its bit pattern so the fixture pins the exact f32, not a decimal round-trip.
	void f32v(float v) {
		uint32_t bits;
		std::memcpy(&bits, &v, 4);
		u32v(bits);
	}

	void magic(const char* m) { for(int i = 0; i < 4; i++) buf.push_back(static_cast<uint8_t>(m[i])); }

	void str(const std::string& s) {
		u32v(static_cast<uint32_t>(s.size()));
		buf.insert(buf.end(), s.begin(), s.end());
	}

	void bytes(const std::vector<uint8_t>& b) {
		u64v(static_cast<uint64_t>(b.size()));
		buf.insert(buf.end(), b.begin(), b.end());
	}
};

static uint32_t crc32(const uint8_t* data, size_t len) {
	static uint32_t table[256];
	static bool init = false;

	if(!init) {
		for(uint32_t i = 0; i < 256; i++) {
			uint32_t c = i;
			for(int k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
			table[i] = c;
		}
		init = true;
	}

	uint32_t c = 0xffffffffu;
	for(size_t i = 0; i < len; i++) c = table[(c ^ data[i]) & 0xff] ^ (c >> 8);
	return c ^ 0xffffffffu;
}

// ================================================================================================
// Case definition
// ================================================================================================

struct Case {
	std::string name;
	uint32_t texWidth;
	// key -> shape. Insertion order here is the order addShape() is called; note that upstream's
	// _build is an unordered_map, so this does NOT determine packing order for multi-shape cases.
	// That is exactly why multi-shape cases are compared semantically rather than byte-for-byte.
	std::vector<std::pair<Key, Atlas::ShapeInfo>> shapes;
};

// Coverage-grid resolution. Small on purpose: these are float-per-pixel and land in every
// fixture, and 32x32 is ample to catch a wrong solver or a mis-decoded band.
static constexpr uint32_t kGridSize = 32;

static Atlas::Curve cv(float x1, float y1, float x2, float y2, float x3, float y3) {
	return Atlas::Curve{x1, y1, x2, y2, x3, y3};
}

// A closed unit triangle, as three quadratic curves with collinear control points.
static Atlas::Curves triangle() {
	return {
		cv(0.0f, 0.0f, 0.25f, 0.5f, 0.5f, 1.0f),
		cv(0.5f, 1.0f, 0.75f, 0.5f, 1.0f, 0.0f),
		cv(1.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f)
	};
}

// N curves that all land in the same band (with numBands=1 there is only one), so a single band's
// curve list has exactly N entries. Used to probe the row-fit limit.
//
// Every curve is given a DISTINCT max-x and max-y, and that is load-bearing rather than
// incidental. The band sorts (slughorn.cpp:1055 by max-x, :1125 by max-y) are `std::sort` -- which
// is unstable -- with a bare `>` comparator, so ties are ordered arbitrarily. Measured against
// libstdc++: with all-equal keys the original order survives up to n=16 (insertion sort) and is
// scrambled from n=17 on (introsort). At these sizes, tied keys would therefore make the C++
// output an artifact of one standard library's pivot choices -- not reproducible by any correct
// port, nor even by libc++. Distinct keys make the ordering total, so every correct sort agrees.
//
// Ties are still covered, deliberately, by the `ties_in_sort` case: 8 curves, inside the range
// where libstdc++'s behaviour is stable and a tie-breaking port matches it exactly.
static Atlas::Curves manyCurves(uint32_t n) {
	Atlas::Curves out;
	for(uint32_t i = 0; i < n; i++) {
		const float t = static_cast<float>(i) / static_cast<float>(n);
		// max-x = 0.5 + t*0.5 and max-y = t + 0.001: both strictly increasing in i.
		out.push_back(cv(0.0f, t, 0.25f + t * 0.1f, t + 0.001f, 0.5f + t * 0.5f, t));
	}
	return out;
}

static std::vector<Case> buildCases() {
	std::vector<Case> cases;

	auto simple = [](Atlas::Curves c, int nx = 0, int ny = 0) {
		Atlas::ShapeInfo si;
		si.curves = std::move(c);
		si.numBandsX = nx;
		si.numBandsY = ny;
		return si;
	};

	// -- trivial ---------------------------------------------------------------------------------
	cases.push_back({"single_curve", 512, {{Key(uint32_t('A')), simple({cv(0, 0, 0.5f, 1, 1, 0)})}}});
	cases.push_back({"empty_shape", 512, {{Key(uint32_t('E')), simple({})}}});
	cases.push_back({"triangle", 512, {{Key("triangle"), simple(triangle())}}});

	// -- the 738-curve regression from the slughorn.cpp:1416-1434 war story -----------------------
	// At texWidth=512 the band's 738-entry curve list cannot fit one texture row -> throws.
	// At texWidth=1024 it fits -> succeeds.
	cases.push_back({"band_738_w512", 512, {{Key("band738"), simple(manyCurves(738), 1, 1)}}});
	cases.push_back({"band_738_w1024", 1024, {{Key("band738"), simple(manyCurves(738), 1, 1)}}});

	// -- the `count > _texWidth` boundary is strict ----------------------------------------------
	cases.push_back({"band_eq_texwidth", 512, {{Key("eq"), simple(manyCurves(512), 1, 1)}}});
	cases.push_back({"band_texwidth_plus1", 512, {{Key("plus1"), simple(manyCurves(513), 1, 1)}}});

	// -- sort ties: several curves sharing an identical max-X ------------------------------------
	{
		Atlas::Curves c;
		for(uint32_t i = 0; i < 8; i++) {
			const float y = static_cast<float>(i) * 0.1f;
			// Every curve has max-X exactly 1.0 -> the comparator sees ties throughout.
			c.push_back(cv(0.0f, y, 0.5f, y + 0.05f, 1.0f, y));
		}
		cases.push_back({"ties_in_sort", 512, {{Key("ties"), simple(std::move(c), 1, 1)}}});
	}

	// -- FP-contraction canary -------------------------------------------------------------------
	// Band boundaries are computed as `minY + snapped * rangeY` (slughorn.cpp:1029/1036/1080) and
	// then feed comparisons (`maxY >= lo`, slughorn.cpp:165), so a 1-ULP drift in a boundary can
	// flip band membership and change the integer output bytes. Contracting that expression into
	// an FMA changes its f32 result for ~24% of inputs (measured), which is why the dumper is
	// built with -ffp-contract=off.
	//
	// The coordinates below are deliberately NOT binary fractions. An earlier version of this case
	// used 0.5/0.25, which are exactly representable -- the boundary then computes identically with
	// and without contraction, so the case could not detect the very thing it was named for.
	//
	// Even so, treat this as a sample, not a proof: catching a flip also requires a curve extremum
	// to fall inside the 1-ULP gap. The `fixtures-contract-check` step is the real guarantee -- it
	// builds the dumper both ways and diffs, testing the property directly.
	{
		Atlas::Curves c = {
			cv(0.1f, 0.1f, 0.3f, 1.0f / 3.0f, 0.7f, 0.5123f),
			cv(0.7f, 0.5123f, 0.9f, 1.0f / 7.0f, 0.983f, 0.017f),
			cv(0.983f, 0.017f, 0.4f, 0.06f, 0.1f, 0.1f)
		};
		cases.push_back({"boundary_exact", 512, {{Key("boundary"), simple(std::move(c), 3, 3)}}});
	}

	// -- degenerate: near-collinear, exercising the |ay|<EPS && |by|<EPS branch ------------------
	// render.hpp:342 guards this divide; the reference GLSL (example:135) does not. Recorded so the
	// divergence is measured rather than assumed.
	{
		Atlas::Curves c = {
			cv(0.0f, 0.5f, 0.5f, 0.5f, 1.0f, 0.5f),
			cv(1.0f, 0.5f, 0.5f, 0.5f, 0.0f, 0.5f)
		};
		cases.push_back({"degenerate_collinear", 512, {{Key("degen"), simple(std::move(c))}}});
	}

	// -- CurveDecomposer ------------------------------------------------------------------------
	//
	// Atlas::Shape retains the em-space curves it was given (slughorn.hpp:771), so feeding
	// CurveDecomposer output through addShape() makes the fixture's shape.curves *be* the
	// decomposed output. That lets the existing fixture format check the decomposer byte-exactly,
	// with no separate dump mode.
	{
		// Default tolerance (TOLERANCE_EXACT): always two quads per cubic, no subdivision.
		Atlas::Curves c;
		slughorn::CurveDecomposer d(c);
		d.moveTo(0.0f, 0.0f);
		d.cubicTo(0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f);
		d.cubicTo(1.0f, -0.4f, 0.13f, -0.77f, 0.0f, 0.0f);
		d.close();
		cases.push_back({"decompose_cubic_default", 512, {{Key("dec"), simple(std::move(c))}}});
	}
	{
		// A low tolerance opts into adaptive De Casteljau subdivision.
		Atlas::Curves c;
		slughorn::CurveDecomposer d(c);
		d.tolerance = slughorn::TOLERANCE_FINE;
		d.moveTo(0.0f, 0.0f);
		d.cubicTo(0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f);
		d.cubicTo(1.0f, -0.4f, 0.13f, -0.77f, 0.0f, 0.0f);
		d.close();
		cases.push_back({"decompose_cubic_fine", 512, {{Key("dec"), simple(std::move(c))}}});
	}
	{
		// Degenerate: tolerance 0 means nothing is ever flat, so MAX_DEPTH is what terminates it.
		Atlas::Curves c;
		slughorn::CurveDecomposer d(c);
		d.tolerance = 0.0f;
		d.moveTo(0.0f, 0.0f);
		d.cubicTo(0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f);
		cases.push_back({"decompose_max_depth", 512, {{Key("dec"), simple(std::move(c))}}});
	}
	{
		// Lines, close(), and reversed winding (the punch-out idiom).
		Atlas::Curves c;
		slughorn::CurveDecomposer d(c);
		d.moveTo(0.0f, 0.0f);
		d.lineTo(1.0f, 0.0f);
		d.lineTo(1.0f, 1.0f);
		d.close();
		const size_t m = d.mark();
		d.moveTo(0.25f, 0.25f);
		d.lineTo(0.5f, 0.25f);
		d.lineTo(0.5f, 0.5f);
		d.close();
		d.reverseFrom(m);
		cases.push_back({"decompose_lines_reverse", 512, {{Key("dec"), simple(std::move(c))}}});
	}

	// -- multi-shape: Tier B (semantic compare only; see the packing-order note above) -----------
	cases.push_back({"multi_shape_3", 512, {
		{Key(uint32_t('A')), simple({cv(0, 0, 0.5f, 1, 1, 0)})},
		{Key("logo"), simple(triangle())},
		{Key(uint32_t('B')), simple({cv(0, 0, 0.5f, 0.5f, 1, 1), cv(1, 1, 0.5f, 0, 0, 0)})}
	}});

	return cases;
}

// ================================================================================================
// Serialization
// ================================================================================================

static void writeKey(Writer& w, const Key& k) {
	if(k.type() == Key::Type::Codepoint) {
		w.u8v(0);
		w.u32v(k.codepoint());
	}

	else {
		w.u8v(1);
		w.str(k.name());
	}
}

static void writeShape(Writer& w, const Key& k, const Atlas::Shape& s) {
	writeKey(w, k);

	w.u32v(s.bandTexX);
	w.u32v(s.bandTexY);
	w.u32v(s.bandMaxX);
	w.u32v(s.bandMaxY);

	w.f32v(s.bandScaleX);
	w.f32v(s.bandScaleY);
	w.f32v(s.bandOffsetX);
	w.f32v(s.bandOffsetY);

	w.f32v(s.bearingX);
	w.f32v(s.bearingY);
	w.f32v(s.width);
	w.f32v(s.height);
	w.f32v(s.advance);
	w.f32v(s.originX);
	w.f32v(s.originY);

	w.u32v(static_cast<uint32_t>(s.curves.size()));

	for(const auto& c : s.curves) {
		w.f32v(c.x1); w.f32v(c.y1);
		w.f32v(c.x2); w.f32v(c.y2);
		w.f32v(c.x3); w.f32v(c.y3);
	}
}

static void writeTexture(Writer& w, const Atlas::TextureData& t) {
	w.u32v(t.width);
	w.u32v(t.height);
	w.u32v(t.depth);
	w.u32v(static_cast<uint32_t>(t.format));
	w.bytes(t.bytes);
}

static bool writeCase(const Case& c, const std::string& dir) {
	Writer w;

	w.magic("SLGF");
	w.u32v(3); // format version
	w.u32v(Atlas::INDIRECTION_SIZE);
	w.str(c.name);
	w.u32v(c.texWidth);

	Atlas atlas(c.texWidth);

	for(const auto& [key, info] : c.shapes) atlas.addShape(key, info);

	std::string err;

	try {
		atlas.build();
	}

	catch(const std::exception& e) {
		err = e.what();
	}

	// threw: 1 if build() raised. The *message* is recorded verbatim rather than being classified
	// into a code here -- mapping C++ messages to Zig error values belongs in the Zig test table,
	// where it is readable and where deliberate divergences (e.g. >255 bands) can be stated.
	w.u32v(err.empty() ? 0u : 1u);
	w.str(err);

	if(err.empty()) {
		w.u32v(static_cast<uint32_t>(c.shapes.size()));

		for(const auto& [key, info] : c.shapes) {
			const auto shape = atlas.getShape(key);

			if(!shape) {
				std::fprintf(stderr, "dump: %s: shape vanished after build\n", c.name.c_str());
				return false;
			}

			writeShape(w, key, *shape);
		}

		const auto& st = atlas.getPackingStats();

		w.u32v(st.curveTexelsUsed);
		w.u32v(st.curveTexelsPadding);
		w.u32v(st.curveTexelsTotal);
		w.u32v(st.bandTexelsUsed);
		w.u32v(st.bandTexelsPadding);
		w.u32v(st.bandTexelsTotal);
		w.u32v(st.bandMaxCount);
		w.u32v(st.bandMaxOffset);

		writeTexture(w, atlas.getCurveTextureData());
		writeTexture(w, atlas.getBandTextureData());

		// Coverage grids from render.hpp -- the CPU shader emulator the Zig render.zig must
		// reproduce. Rendered for single-shape cases only (multi-shape cases are Tier B and are
		// not compared this way), and only when the shape has usable dimensions, since
		// computeRenderSize() throws otherwise.
		uint32_t numGrids = 0;
		std::vector<std::pair<uint32_t, slughorn::render::Grid>> grids; // (banded, grid)

		if(c.shapes.size() == 1) {
			const auto& key = c.shapes[0].first;
			const auto shape = atlas.getShape(key);

			if(shape && shape->width > 0.0f && shape->height > 0.0f) {
				const auto sampler = slughorn::render::decode(atlas, key);

				// Both paths: banded is what the shader does, unbanded is the reference every
				// correct banding must agree with.
				for(uint32_t banded = 0; banded <= 1; banded++) {
					grids.emplace_back(banded, sampler.renderGrid(kGridSize, 0.0f, banded != 0));
				}

				numGrids = static_cast<uint32_t>(grids.size());
			}
		}

		w.u32v(numGrids);

		for(const auto& [banded, g] : grids) {
			w.u32v(banded);
			w.u32v(g.width);
			w.u32v(g.height);

			for(const auto v : g.data) w.f32v(v);
		}
	}

	w.magic("ENDF");

	const uint32_t sum = crc32(w.buf.data(), w.buf.size());

	w.u32v(sum);

	const std::string path = dir + "/" + c.name + ".slgf";

	FILE* f = std::fopen(path.c_str(), "wb");

	if(!f) {
		std::fprintf(stderr, "dump: cannot open %s\n", path.c_str());
		return false;
	}

	const size_t n = std::fwrite(w.buf.data(), 1, w.buf.size(), f);

	std::fclose(f);

	if(n != w.buf.size()) {
		std::fprintf(stderr, "dump: short write to %s\n", path.c_str());
		return false;
	}

	std::printf(
		"%-24s %6zu bytes  %s\n",
		c.name.c_str(),
		w.buf.size(),
		err.empty() ? "ok" : ("threw: " + err).substr(0, 60).c_str()
	);

	return true;
}

int main(int argc, char** argv) {
	if(argc < 2) {
		std::fprintf(stderr, "usage: slughorn-dump <output-dir>\n");
		return 2;
	}

	const std::string dir = argv[1];

	for(const auto& c : buildCases()) {
		if(!writeCase(c, dir)) return 1;
	}

	return 0;
}
