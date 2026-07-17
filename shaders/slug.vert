// Copyright (c) 2026 Michael Pollind
// SPDX-License-Identifier: GPL-2.0-only
//
// Slug vertex stage. Passes the per-vertex em coordinate through for smooth interpolation and the
// per-shape band data through flat (constant across the quad). Positions are supplied already in
// clip space by the CPU, so there is no MVP here -- the renderer maps the em window onto the
// viewport directly.
#version 450

layout(location = 0) in vec2 a_position;  // clip-space xy
layout(location = 1) in vec2 a_emCoord;   // em-space coordinate at this vertex
layout(location = 2) in vec4 a_bandXform; // bandScaleX/Y, bandOffsetX/Y
layout(location = 3) in vec4 a_shapeData; // bandTexX/Y, bandMaxX/Y

layout(location = 0) out vec2 v_emCoord;
layout(location = 1) flat out vec4 v_bandXform;
layout(location = 2) flat out vec4 v_shapeData;

void main() {
    v_emCoord = a_emCoord;
    v_bandXform = a_bandXform;
    v_shapeData = a_shapeData;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
