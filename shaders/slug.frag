// Copyright (c) 2026 AlphaPixel LLC (original GLSL), Michael Pollind (Vulkan port)
// SPDX-License-Identifier: GPL-2.0-only
//
// Slug fragment stage: analytic glyph coverage from the packed band/curve textures.
//
// This mirrors the reference GLSL (slughorn/example/slughorn-example-glfw.cpp), NOT src/render.zig
// / render.hpp. The one deliberate difference is in the polynomial solve's degenerate branch: the
// reference computes `0.5 / b.y` unconditionally (yielding +/-inf or NaN for a near-degenerate,
// near-axis-aligned curve), where render.hpp guards it to 0. render.zig is pinned to render.hpp for
// the golden fixtures; this shader is pinned to the GLSL. They agree everywhere except that branch
// -- see DIVERGENCE.md. Adapted for Vulkan: separate texture objects fetched samplerlessly, the
// output is raw coverage (an r32f target) rather than a blended color.
#version 450
#extension GL_EXT_samplerless_texture_functions : require

layout(location = 0) in vec2 v_emCoord;
layout(location = 1) flat in vec4 v_bandXform; // bandScaleX/Y, bandOffsetX/Y
layout(location = 2) flat in vec4 v_shapeData; // bandTexX/Y, bandMaxX/Y

layout(set = 0, binding = 0) uniform texture2D u_curveTexture;
layout(set = 0, binding = 1) uniform utexture2D u_bandTexture;

layout(location = 0) out float fragColor;

// log2(atlas texture width). Passed as a define; defaults to log2(512) = 9, the default atlas.
#ifndef TEX_WIDTH
#define TEX_WIDTH 9
#endif
// Must match slughorn's build-option indirection_size (Atlas::INDIRECTION_SIZE).
#ifndef SLUG_INDIRECTION_SIZE
#define SLUG_INDIRECTION_SIZE 32
#endif

uint slug_CalcRootCode(float y1, float y2, float y3) {
    uint i1 = floatBitsToUint(y1) >> 31u;
    uint i2 = floatBitsToUint(y2) >> 30u;
    uint i3 = floatBitsToUint(y3) >> 29u;

    uint shift = (i2 & 2u) | (i1 & ~2u);
    shift = (i3 & 4u) | (shift & ~4u);

    return ((0x2E74u >> shift) & 0x0101u);
}

vec2 slug_SolveHorizPoly(vec4 p12, vec2 p3) {
    vec2 a = p12.xy - p12.zw * 2.0 + p3;
    vec2 b = p12.xy - p12.zw;
    float ra = 1.0 / a.y;
    float rb = 0.5 / b.y; // unconditional -- the deliberate GLSL divergence from render.hpp
    float d = sqrt(max(b.y * b.y - a.y * p12.y, 0.0));
    float t1 = (b.y - d) * ra;
    float t2 = (b.y + d) * ra;
    if (abs(a.y) < 1.0 / 65536.0) { t1 = p12.y * rb; t2 = t1; }
    return vec2(
        (a.x * t1 - b.x * 2.0) * t1 + p12.x,
        (a.x * t2 - b.x * 2.0) * t2 + p12.x
    );
}

vec2 slug_SolveVertPoly(vec4 p12, vec2 p3) {
    vec2 a = p12.xy - p12.zw * 2.0 + p3;
    vec2 b = p12.xy - p12.zw;
    float ra = 1.0 / a.x;
    float rb = 0.5 / b.x;
    float d = sqrt(max(b.x * b.x - a.x * p12.x, 0.0));
    float t1 = (b.x - d) * ra;
    float t2 = (b.x + d) * ra;
    if (abs(a.x) < 1.0 / 65536.0) { t1 = p12.x * rb; t2 = t1; }
    return vec2(
        (a.y * t1 - b.y * 2.0) * t1 + p12.y,
        (a.y * t2 - b.y * 2.0) * t2 + p12.y
    );
}

ivec2 slug_CalcBandLoc(ivec2 glyphLoc, uint offset) {
    ivec2 bandLoc = ivec2(glyphLoc.x + int(offset), glyphLoc.y);
    bandLoc.y += bandLoc.x >> TEX_WIDTH;
    bandLoc.x &= (1 << TEX_WIDTH) - 1;
    return bandLoc;
}

float slug_CalcCoverage(float xcov, float ycov, float xwgt, float ywgt) {
    float coverage = max(
        abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0),
        min(abs(xcov), abs(ycov))
    );
    return clamp(coverage, 0.0, 1.0);
}

float slug_Render(vec2 renderCoord, vec4 bandTransform, ivec2 glyphLoc, ivec2 bandMax) {
    vec2 emsPerPixel = fwidth(renderCoord);
    vec2 pixelsPerEm = 1.0 / emsPerPixel;

    // O(1) band index via the indirection tables (2 fetches per axis).
    int qY = clamp(int(renderCoord.y * bandTransform.y + bandTransform.w), 0, SLUG_INDIRECTION_SIZE - 1);
    int qX = clamp(int(renderCoord.x * bandTransform.x + bandTransform.z), 0, SLUG_INDIRECTION_SIZE - 1);
    int bandY = int(texelFetch(u_bandTexture, ivec2(glyphLoc.x + qY, glyphLoc.y), 0).r);
    int bandX = int(texelFetch(u_bandTexture, ivec2(glyphLoc.x + SLUG_INDIRECTION_SIZE + qX, glyphLoc.y), 0).r);

    // Horizontal bands -- headers at glyphLoc + 2*IS + bandY.
    float xcov = 0.0, xwgt = 0.0;
    uvec2 hbandData = texelFetch(u_bandTexture, ivec2(glyphLoc.x + 2 * SLUG_INDIRECTION_SIZE + bandY, glyphLoc.y), 0).xy;
    ivec2 hbandLoc = slug_CalcBandLoc(glyphLoc, hbandData.y);

    for (int ci = 0; ci < int(hbandData.x); ci++) {
        ivec2 curveLoc = ivec2(texelFetch(u_bandTexture, ivec2(hbandLoc.x + ci, hbandLoc.y), 0).xy);
        vec4 p12 = texelFetch(u_curveTexture, curveLoc, 0) - vec4(renderCoord, renderCoord);
        vec2 p3 = texelFetch(u_curveTexture, ivec2(curveLoc.x + 1, curveLoc.y), 0).xy - renderCoord;

        if (max(max(p12.x, p12.z), p3.x) * pixelsPerEm.x < -0.5) break;

        uint code = slug_CalcRootCode(p12.y, p12.w, p3.y);
        if (code != 0u) {
            vec2 r = slug_SolveHorizPoly(p12, p3) * pixelsPerEm.x;
            if ((code & 1u) != 0u) { xcov += clamp(r.x + 0.5, 0.0, 1.0); xwgt = max(xwgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0)); }
            if (code > 1u) { xcov -= clamp(r.y + 0.5, 0.0, 1.0); xwgt = max(xwgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0)); }
        }
    }

    // Vertical bands -- headers at glyphLoc + 2*IS + numHBands + bandX.
    float ycov = 0.0, ywgt = 0.0;
    uvec2 vbandData = texelFetch(u_bandTexture, ivec2(glyphLoc.x + 2 * SLUG_INDIRECTION_SIZE + bandMax.y + 1 + bandX, glyphLoc.y), 0).xy;
    ivec2 vbandLoc = slug_CalcBandLoc(glyphLoc, vbandData.y);

    for (int ci = 0; ci < int(vbandData.x); ci++) {
        ivec2 curveLoc = ivec2(texelFetch(u_bandTexture, ivec2(vbandLoc.x + ci, vbandLoc.y), 0).xy);
        vec4 p12 = texelFetch(u_curveTexture, curveLoc, 0) - vec4(renderCoord, renderCoord);
        vec2 p3 = texelFetch(u_curveTexture, ivec2(curveLoc.x + 1, curveLoc.y), 0).xy - renderCoord;

        if (max(max(p12.y, p12.w), p3.y) * pixelsPerEm.y < -0.5) break;

        uint code = slug_CalcRootCode(p12.x, p12.z, p3.x);
        if (code != 0u) {
            vec2 r = slug_SolveVertPoly(p12, p3) * pixelsPerEm.y;
            if ((code & 1u) != 0u) { ycov -= clamp(r.x + 0.5, 0.0, 1.0); ywgt = max(ywgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0)); }
            if (code > 1u) { ycov += clamp(r.y + 0.5, 0.0, 1.0); ywgt = max(ywgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0)); }
        }
    }

    return slug_CalcCoverage(xcov, ycov, xwgt, ywgt);
}

void main() {
    ivec2 glyphLoc = ivec2(v_shapeData.xy);
    ivec2 bandMax = ivec2(v_shapeData.zw);
    fragColor = slug_Render(v_emCoord, v_bandXform, glyphLoc, bandMax);
}
