// Copyright (c) 2026 Michael Pollind
// SPDX-License-Identifier: GPL-2.0-only
//
// Samples the atlas MSDF tile array (Phase 4): fetches a shape's tile from the Texture2DArray at its
// layer and reconstructs the signed distance as median(r, g, b) -- the GPU-side counterpart of the
// msdf-zig tile generation. A demonstration that the RGB32F tile array is consumable; the caller
// maps the em coordinate to the tile's [0,1] UV. (The array is uploaded as RGBA32F, which is widely
// supported for sampled images; only rgb is meaningful.)
#version 450
#extension GL_EXT_samplerless_texture_functions : require

layout(location = 0) in vec2 v_emCoord; // tile UV in [0,1]

layout(set = 0, binding = 0) uniform texture2DArray u_msdf; // RGBA32F (rgb = MSDF channels)

// p = (tile_size, layer, 0, 0)
layout(push_constant) uniform PC { vec4 p; } pc;

layout(location = 0) out float fragColor;

float median3(float a, float b, float c) {
    return max(min(a, b), min(max(a, b), c));
}

void main() {
    int ts = int(pc.p.x);
    int layer = int(pc.p.y);
    ivec2 texel = clamp(ivec2(v_emCoord * float(ts)), ivec2(0), ivec2(ts - 1));
    vec3 s = texelFetch(u_msdf, ivec3(texel, layer), 0).rgb;
    fragColor = median3(s.r, s.g, s.b);
}
