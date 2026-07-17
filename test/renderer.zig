// Copyright (c) 2026 Michael Pollind
// SPDX-License-Identifier: GPL-2.0-only
//
// Tests for the GPU renderer backend (`-Drenderer=true`). Like the nanosvg/sdf suites these run
// against a live backend rather than a byte oracle: here a real Vulkan device. The eventual
// coverage test cross-checks the GPU against `render.zig`; this first case proves the headless
// render + readback path itself works end-to-end.

const std = @import("std");
const renderer = @import("slughorn_renderer");

/// RGBA8 unorm stores `round(f * 255)`; allow ±1 for the implementation's rounding of ties.
fn expectChannel(actual: u8, expected: f32) !void {
    const target = @round(expected * 255.0);
    const diff = @abs(@as(f32, @floatFromInt(actual)) - target);
    try std.testing.expect(diff <= 1.0);
}

test "headless clear + readback round-trips the clear color" {
    const gpa = std.testing.allocator;
    const w: u32 = 16;
    const h: u32 = 16;
    const clear = [4]f32{ 0.25, 0.5, 0.75, 1.0 };

    const px = try renderer.clearToRgba8(gpa, w, h, clear);
    defer gpa.free(px);

    try std.testing.expectEqual(@as(usize, w * h * 4), px.len);

    // Every texel of a cleared target is the clear color. Check the first, a middle, and the last.
    for ([_]usize{ 0, (h / 2 * w + w / 2) * 4, (w * h - 1) * 4 }) |base| {
        try expectChannel(px[base + 0], clear[0]);
        try expectChannel(px[base + 1], clear[1]);
        try expectChannel(px[base + 2], clear[2]);
        try expectChannel(px[base + 3], clear[3]);
    }
}
