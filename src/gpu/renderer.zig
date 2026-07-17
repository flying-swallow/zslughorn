// Copyright (c) 2026 Michael Pollind
// SPDX-License-Identifier: GPL-2.0-only
//
// The GPU renderer backend (M3). Draws the compiled slughorn atlas on the GPU with a Slug
// fragment shader that mirrors the reference GLSL. It links rhi-zig (GPL-2.0), so this module
// imports `slughorn` (MIT) but the core never imports this -- keeping GPL out of the MIT core,
// the same module seam the nanosvg/msdf backends use.
//
// Bring-up is incremental. This first slice establishes the headless Vulkan plumbing --
// instance/device with no window, an offscreen color target, and a render + CPU readback path --
// which is the infrastructure every later slice (shader, atlas textures, coverage cross-check)
// builds on. The clear-and-read entry point exists to prove that plumbing end-to-end against a
// real device before any atlas data is involved.

const std = @import("std");
const rhi = @import("rhi");
const slughorn = @import("slughorn");

/// A headless Vulkan context: a Vulkan instance + logical device, with no window or surface.
/// `rhi.Renderer` is a process-global singleton, so at most one `Gpu` may be live at a time.
pub const Gpu = struct {
    gpa: std.mem.Allocator,
    device: rhi.Device,

    pub fn init(gpa: std.mem.Allocator) !Gpu {
        // No validation layer: it is optional tooling and may be absent, and requesting an
        // unavailable layer fails device-independent instance creation. Headless -- no surface.
        try rhi.Renderer.init(gpa, .{ .vk = .{ .app_name = "slughorn", .enable_validation_layer = false } });
        errdefer rhi.Renderer.deinit();

        var adapters = try rhi.PhysicalAdapter.enumerate_adapters(gpa);
        defer adapters.deinit(gpa);
        if (adapters.items.len == 0) return error.NoVulkanAdapter;

        // Ranks discrete > integrated > cpu, so a real GPU wins when present and llvmpipe is the
        // fallback. `Device.init` copies what it needs out of the adapter, so freeing the list
        // (the deferred `deinit` above) after this is safe.
        const idx = rhi.PhysicalAdapter.default_select_adapter(adapters.items);
        const device = try rhi.Device.init(gpa, &adapters.items[idx]);
        return .{ .gpa = gpa, .device = device };
    }

    pub fn deinit(self: *Gpu) void {
        self.device.deinit();
        rhi.Renderer.deinit();
    }
};

/// Render an offscreen `width` x `height` RGBA8 target cleared to `clear`, and return the pixels
/// read back to host memory (row-major, 4 bytes/texel, caller owns the slice).
///
/// This is the smallest end-to-end exercise of the headless path: image + view creation, a
/// clear-only render pass, the color->transfer barrier, a texture->buffer copy, queue submit, and
/// a host-visible readback. Later slices swap the clear for a real Slug draw over the atlas.
pub fn clearToRgba8(gpa: std.mem.Allocator, width: u32, height: u32, clear: [4]f32) ![]u8 {
    var gpu = try Gpu.init(gpa);
    defer gpu.deinit();
    const dev = &gpu.device;

    var color = try rhi.Image.init(dev, .{
        .format = .rgba8_unorm,
        .width = width,
        .height = height,
        .usage = .{ .color_attachment = true, .transfer_src = true },
        .memory_usage = .prefer_device,
    });
    defer color.deinit(dev);

    var color_view = try rhi.ImageView.init(dev, &color, .{
        .view_type = .color_attachment,
        .format = .rgba8_unorm,
        .aspect = .color,
    });
    defer color_view.deinit(dev);

    const size: usize = @as(usize, width) * @as(usize, height) * 4;
    var readback = try rhi.Buffer.init_general(dev, .{
        .size = size,
        .persistant_map = true,
        .sequential_access = false, // random host access: we read it back
        .buffer_usage = .prefer_host,
        .usage = .{}, // transfer_src/dst default on
    });
    defer readback.deinit(dev);

    var pool = try rhi.Pool.init(dev, &dev.graphics_queue);
    defer pool.deinit(dev);
    var cmd = try rhi.Cmd.init(dev, &pool);
    defer cmd.deinit(dev, &pool);

    try cmd.begin(dev);
    cmd.image_barrier(dev, .{ .image = &color, .before = .{}, .after = .{ .render_target = true } });
    cmd.begin_rendering(dev, .{
        .color_attachments = &.{.{ .view = color_view, .load_op = .clear, .store_op = .store, .clear_color = clear }},
        .render_area = .{ .width = width, .height = height },
    });
    cmd.end_rendering(dev);
    cmd.image_barrier(dev, .{ .image = &color, .before = .{ .render_target = true }, .after = .{ .copy_src = true } });
    cmd.copy_texture_to_buffer(dev, .{ .src = &color, .dst = &readback, .width = width, .height = height, .aspect = .color });
    cmd.buffer_barrier(dev, .{ .buffer = &readback, .before = .{ .copy_dst = true }, .after = .{ .host_read = true } });
    try cmd.end(dev);

    try dev.graphics_queue.submit(dev, .{ .vk = .{ .cmds = &.{&cmd} } });
    try dev.graphics_queue.wait_queue_idle(dev);

    const mapped = readback.mapped_region orelse return error.BufferNotMapped;
    const out = try gpa.alloc(u8, size);
    @memcpy(out, mapped[0..size]);
    return out;
}
