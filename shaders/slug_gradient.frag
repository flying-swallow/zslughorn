// Copyright (c) 2026 Michael Pollind
// SPDX-License-Identifier: GPL-2.0-only
//
// Samples the atlas gradient strip (Phase 5): computes the linear ramp parameter t from the em
// coordinate and the gradient's transform, then fetches the color from the strip texture -- the
// GPU-side counterpart of Atlas.rasterizeGradients. A demonstration that the strip is consumable as
// the sampling convention intends (V = row = gradient_id - 1, U = t); coverage is a separate pass
// (slug.frag), so this fills the whole quad with the ramp.
#version 450
#extension GL_EXT_samplerless_texture_functions : require

layout(location = 0) in vec2 v_emCoord;

layout(set = 0, binding = 0) uniform texture2D u_gradientStrip; // RGBA8 unorm, one row per gradient

// grad = (xx, xy, dx, row): linear ramp t = xx*emX + xy*emY + dx; `row` is gradient_id - 1.
layout(push_constant) uniform PC { vec4 grad; } pc;

layout(location = 0) out vec4 fragColor;

void main() {
    float t = clamp(pc.grad.x * v_emCoord.x + pc.grad.y * v_emCoord.y + pc.grad.z, 0.0, 1.0);
    int col = int(t * 255.0 + 0.5);
    int row = int(pc.grad.w);
    fragColor = texelFetch(u_gradientStrip, ivec2(col, row), 0);
}
