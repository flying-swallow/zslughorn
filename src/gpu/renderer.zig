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

/// The Slug shader stages, compiled from `shaders/slug.{vert,frag}` to SPIR-V by the build and
/// embedded here (see `compileGlslSpv` in build.zig). SPIR-V is a multiple of 4 bytes, which is all
/// `rpi.Program` requires -- it copies the blob into aligned storage.
const vert_spv: []const u8 = @embedFile("slug_vert_spv");
const frag_spv: []const u8 = @embedFile("slug_frag_spv");
const grad_frag_spv: []const u8 = @embedFile("slug_gradient_frag_spv");
const msdf_frag_spv: []const u8 = @embedFile("slug_msdf_frag_spv");

fn mapFormat(f: slughorn.TextureData.Format) rhi.Format {
    return switch (f) {
        .rgba32f => .rgba32_sfloat,
        .rgba16ui => .rgba16_uint,
        .rgba8 => .rgba8_unorm,
        .rgb32f => .rgb32_sfloat,
    };
}

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

/// Render `key`'s coverage on the GPU over the em window `[ox, ox+sx] x [oy, oy+sy]`, into an
/// `n` x `n` grid, and return it row-major (row 0 = top), values in [0, 1]. Null if the key is not
/// in the atlas.
///
/// The window is mapped linearly onto the viewport so a fragment at output pixel (col, row) sees
/// em coordinate `(ox + (col+0.5)/n * sx, oy + (row+0.5)/n * sy)` and a constant `fwidth`, i.e.
/// pixels-per-em `(n/sx, n/sy)`. A caller that evaluates `render.zig`'s sampler at those exact
/// points with those exact pixels-per-em can compare the two coverage fields directly.
pub fn renderCoverage(
    gpa: std.mem.Allocator,
    atlas: *const slughorn.Atlas,
    key: slughorn.Key,
    n: u32,
    ox: f32,
    oy: f32,
    sx: f32,
    sy: f32,
) !?[]f32 {
    const shape = atlas.getShape(key) orelse return null;
    const curve_td = atlas.getCurveTextureData();
    const band_td = atlas.getBandTextureData();

    var gpu = try Gpu.init(gpa);
    defer gpu.deinit();
    const dev = &gpu.device;

    // -- atlas textures, uploaded through host-visible staging buffers --
    var curve_img = try rhi.Image.init(dev, .{
        .format = mapFormat(curve_td.format),
        .width = curve_td.width,
        .height = curve_td.height,
        .usage = .{ .sampled = true, .transfer_dst = true },
        .memory_usage = .prefer_device,
    });
    defer curve_img.deinit(dev);
    var curve_view = try rhi.ImageView.init(dev, &curve_img, .{ .view_type = .shader_resource_2d, .format = mapFormat(curve_td.format), .aspect = .color });
    defer curve_view.deinit(dev);
    var curve_stage = try rhi.Buffer.init_general(dev, .{ .size = curve_td.bytes.len, .persistant_map = true, .sequential_access = true, .buffer_usage = .prefer_host, .usage = .{} });
    defer curve_stage.deinit(dev);
    @memcpy(curve_stage.mapped_region.?[0..curve_td.bytes.len], curve_td.bytes);

    var band_img = try rhi.Image.init(dev, .{
        .format = mapFormat(band_td.format),
        .width = band_td.width,
        .height = band_td.height,
        .usage = .{ .sampled = true, .transfer_dst = true },
        .memory_usage = .prefer_device,
    });
    defer band_img.deinit(dev);
    var band_view = try rhi.ImageView.init(dev, &band_img, .{ .view_type = .shader_resource_2d, .format = mapFormat(band_td.format), .aspect = .color });
    defer band_view.deinit(dev);
    var band_stage = try rhi.Buffer.init_general(dev, .{ .size = band_td.bytes.len, .persistant_map = true, .sequential_access = true, .buffer_usage = .prefer_host, .usage = .{} });
    defer band_stage.deinit(dev);
    @memcpy(band_stage.mapped_region.?[0..band_td.bytes.len], band_td.bytes);

    // -- coverage target (raw float) + host readback --
    var cov = try rhi.Image.init(dev, .{ .format = .r32_sfloat, .width = n, .height = n, .usage = .{ .color_attachment = true, .transfer_src = true }, .memory_usage = .prefer_device });
    defer cov.deinit(dev);
    var cov_view = try rhi.ImageView.init(dev, &cov, .{ .view_type = .color_attachment, .format = .r32_sfloat, .aspect = .color });
    defer cov_view.deinit(dev);
    const cov_bytes: usize = @as(usize, n) * @as(usize, n) * 4;
    var readback = try rhi.Buffer.init_general(dev, .{ .size = cov_bytes, .persistant_map = true, .sequential_access = false, .buffer_usage = .prefer_host, .usage = .{} });
    defer readback.deinit(dev);

    // -- a viewport-filling quad; per-vertex em coords place the em window on the framebuffer --
    const Vertex = extern struct { pos: [2]f32, em: [2]f32, band_xform: [4]f32, shape_data: [4]f32 };
    const bx = [4]f32{ shape.band_scale_x, shape.band_scale_y, shape.band_offset_x, shape.band_offset_y };
    const sd = [4]f32{
        @floatFromInt(shape.band_tex_x),
        @floatFromInt(shape.band_tex_y),
        @floatFromInt(shape.band_max_x),
        @floatFromInt(shape.band_max_y),
    };
    const x1 = ox + sx;
    const y1 = oy + sy;
    // NDC (-1,-1) is the framebuffer top-left, and readback row 0 is the top row, so em y grows
    // downward with the row index -- exactly how the CPU comparison indexes the grid.
    const verts = [6]Vertex{
        .{ .pos = .{ -1, -1 }, .em = .{ ox, oy }, .band_xform = bx, .shape_data = sd },
        .{ .pos = .{ 1, -1 }, .em = .{ x1, oy }, .band_xform = bx, .shape_data = sd },
        .{ .pos = .{ -1, 1 }, .em = .{ ox, y1 }, .band_xform = bx, .shape_data = sd },
        .{ .pos = .{ 1, -1 }, .em = .{ x1, oy }, .band_xform = bx, .shape_data = sd },
        .{ .pos = .{ 1, 1 }, .em = .{ x1, y1 }, .band_xform = bx, .shape_data = sd },
        .{ .pos = .{ -1, 1 }, .em = .{ ox, y1 }, .band_xform = bx, .shape_data = sd },
    };
    var vbuf = try rhi.Buffer.init_general(dev, .{ .size = @sizeOf(@TypeOf(verts)), .persistant_map = true, .sequential_access = true, .buffer_usage = .prefer_host, .usage = .{ .vertex_buffer = true } });
    defer vbuf.deinit(dev);
    @memcpy(vbuf.mapped_region.?[0..@sizeOf(@TypeOf(verts))], std.mem.asBytes(&verts));

    // -- program: two samplerless sampled textures --
    const modules = [_]rhi.rpi.ModuleStage{
        .{ .stage = .vertex, .data = vert_spv },
        .{ .stage = .fragment, .data = frag_spv },
    };
    const layout = rhi.rpi.Layout{ .bindings = &.{
        .{ .name = "u_curveTexture", .set = 0, .binding = 0, .descriptor_type = .sampled_image, .stages = .{ .fragment = true } },
        .{ .name = "u_bandTexture", .set = 0, .binding = 1, .descriptor_type = .sampled_image, .stages = .{ .fragment = true } },
    } };
    var program = try rhi.rpi.Program.initialize(gpa, dev, &modules, layout);
    defer program.deinit(dev);

    // -- record: upload textures, draw, copy back --
    var pool = try rhi.Pool.init(dev, &dev.graphics_queue);
    defer pool.deinit(dev);
    var cmd = try rhi.Cmd.init(dev, &pool);
    defer cmd.deinit(dev, &pool);

    try cmd.begin(dev);

    cmd.image_barrier(dev, .{ .image = &curve_img, .before = .{}, .after = .{ .copy_dst = true } });
    cmd.copy_buffer_to_texture(dev, .{ .src = &curve_stage, .dst = &curve_img, .width = curve_td.width, .height = curve_td.height });
    cmd.image_barrier(dev, .{ .image = &curve_img, .before = .{ .copy_dst = true }, .after = .{ .shader_resource = true } });

    cmd.image_barrier(dev, .{ .image = &band_img, .before = .{}, .after = .{ .copy_dst = true } });
    cmd.copy_buffer_to_texture(dev, .{ .src = &band_stage, .dst = &band_img, .width = band_td.width, .height = band_td.height });
    cmd.image_barrier(dev, .{ .image = &band_img, .before = .{ .copy_dst = true }, .after = .{ .shader_resource = true } });

    cmd.image_barrier(dev, .{ .image = &cov, .before = .{}, .after = .{ .render_target = true } });
    cmd.begin_rendering(dev, .{
        .color_attachments = &.{.{ .view = cov_view, .load_op = .clear, .store_op = .store, .clear_color = .{ 0, 0, 0, 1 } }},
        .render_area = .{ .width = n, .height = n },
    });
    cmd.set_viewport(dev, .{ .width = @floatFromInt(n), .height = @floatFromInt(n) });
    cmd.set_scissor(dev, .{ .width = n, .height = n });
    try program.bindPipeline(dev, &cmd, 1, "slug", .{
        .topology = .triangle_list,
        .colors = &.{.{ .format = .r32_sfloat }},
        .vertex_streams = &.{.{ .binding = 0, .stride = @sizeOf(Vertex) }},
        .vertex_attributes = &.{
            .{ .location = 0, .binding = 0, .format = .rg32_sfloat, .offset = 0 },
            .{ .location = 1, .binding = 0, .format = .rg32_sfloat, .offset = 8 },
            .{ .location = 2, .binding = 0, .format = .rgba32_sfloat, .offset = 16 },
            .{ .location = 3, .binding = 0, .format = .rgba32_sfloat, .offset = 32 },
        },
    });
    try program.bindDescriptors(dev, &cmd, 0, &.{
        rhi.rpi.DescriptorBinding.init("u_curveTexture", rhi.Descriptor.sampledImage(dev, &curve_view), 0),
        rhi.rpi.DescriptorBinding.init("u_bandTexture", rhi.Descriptor.sampledImage(dev, &band_view), 0),
    }, .graphics);
    cmd.bind_vertex_buffer(dev, &vbuf, 0);
    cmd.draw(dev, .{ .vertex_count = 6 });
    cmd.end_rendering(dev);

    cmd.image_barrier(dev, .{ .image = &cov, .before = .{ .render_target = true }, .after = .{ .copy_src = true } });
    cmd.copy_texture_to_buffer(dev, .{ .src = &cov, .dst = &readback, .width = n, .height = n });
    cmd.buffer_barrier(dev, .{ .buffer = &readback, .before = .{ .copy_dst = true }, .after = .{ .host_read = true } });
    try cmd.end(dev);

    try dev.graphics_queue.submit(dev, .{ .vk = .{ .cmds = &.{&cmd} } });
    try dev.graphics_queue.wait_queue_idle(dev);

    const mapped = readback.mapped_region orelse return error.BufferNotMapped;
    const out = try gpa.alloc(f32, @as(usize, n) * @as(usize, n));
    @memcpy(std.mem.sliceAsBytes(out), mapped[0..cov_bytes]);
    return out;
}

/// Render gradient `gradient_id`'s color ramp across the em window `[ox, ox+sx] x [oy, oy+sy]` into
/// an `n` x `n` RGBA8 grid (row 0 = top), sampling the atlas gradient strip on the GPU. Null if the
/// gradient or its strip is absent. This is the GPU counterpart of `Atlas.rasterizeGradients`: it
/// consumes the Phase-5 strip via the intended `(V = gradient_id - 1, U = t)` convention, computing
/// the linear ramp `t = xx*emX + xy*emY + dx` from the gradient's transform.
pub fn renderGradient(
    gpa: std.mem.Allocator,
    atlas: *const slughorn.Atlas,
    gradient_id: u32,
    n: u32,
    ox: f32,
    oy: f32,
    sx: f32,
    sy: f32,
) !?[]u8 {
    const grad = atlas.getGradient(gradient_id) orelse return null;
    const strip_td = atlas.getGradientTextureData();
    if (strip_td.isEmpty()) return null;

    var gpu = try Gpu.init(gpa);
    defer gpu.deinit();
    const dev = &gpu.device;

    // -- the gradient strip texture (RGBA8), uploaded via a host staging buffer --
    var strip_img = try rhi.Image.init(dev, .{
        .format = .rgba8_unorm,
        .width = strip_td.width,
        .height = strip_td.height,
        .usage = .{ .sampled = true, .transfer_dst = true },
        .memory_usage = .prefer_device,
    });
    defer strip_img.deinit(dev);
    var strip_view = try rhi.ImageView.init(dev, &strip_img, .{ .view_type = .shader_resource_2d, .format = .rgba8_unorm, .aspect = .color });
    defer strip_view.deinit(dev);
    var strip_stage = try rhi.Buffer.init_general(dev, .{ .size = strip_td.bytes.len, .persistant_map = true, .sequential_access = true, .buffer_usage = .prefer_host, .usage = .{} });
    defer strip_stage.deinit(dev);
    @memcpy(strip_stage.mapped_region.?[0..strip_td.bytes.len], strip_td.bytes);

    // -- RGBA8 color target + readback --
    var color = try rhi.Image.init(dev, .{ .format = .rgba8_unorm, .width = n, .height = n, .usage = .{ .color_attachment = true, .transfer_src = true }, .memory_usage = .prefer_device });
    defer color.deinit(dev);
    var color_view = try rhi.ImageView.init(dev, &color, .{ .view_type = .color_attachment, .format = .rgba8_unorm, .aspect = .color });
    defer color_view.deinit(dev);
    const out_bytes: usize = @as(usize, n) * @as(usize, n) * 4;
    var readback = try rhi.Buffer.init_general(dev, .{ .size = out_bytes, .persistant_map = true, .sequential_access = false, .buffer_usage = .prefer_host, .usage = .{} });
    defer readback.deinit(dev);

    // -- viewport-filling quad (same layout as slug.vert; band/shape data unused here) --
    const Vertex = extern struct { pos: [2]f32, em: [2]f32, band_xform: [4]f32, shape_data: [4]f32 };
    const z4 = [4]f32{ 0, 0, 0, 0 };
    const x1 = ox + sx;
    const y1 = oy + sy;
    const verts = [6]Vertex{
        .{ .pos = .{ -1, -1 }, .em = .{ ox, oy }, .band_xform = z4, .shape_data = z4 },
        .{ .pos = .{ 1, -1 }, .em = .{ x1, oy }, .band_xform = z4, .shape_data = z4 },
        .{ .pos = .{ -1, 1 }, .em = .{ ox, y1 }, .band_xform = z4, .shape_data = z4 },
        .{ .pos = .{ 1, -1 }, .em = .{ x1, oy }, .band_xform = z4, .shape_data = z4 },
        .{ .pos = .{ 1, 1 }, .em = .{ x1, y1 }, .band_xform = z4, .shape_data = z4 },
        .{ .pos = .{ -1, 1 }, .em = .{ ox, y1 }, .band_xform = z4, .shape_data = z4 },
    };
    var vbuf = try rhi.Buffer.init_general(dev, .{ .size = @sizeOf(@TypeOf(verts)), .persistant_map = true, .sequential_access = true, .buffer_usage = .prefer_host, .usage = .{ .vertex_buffer = true } });
    defer vbuf.deinit(dev);
    @memcpy(vbuf.mapped_region.?[0..@sizeOf(@TypeOf(verts))], std.mem.asBytes(&verts));

    // -- program: one sampled texture (the strip) + a fragment push constant (the linear transform) --
    const modules = [_]rhi.rpi.ModuleStage{
        .{ .stage = .vertex, .data = vert_spv },
        .{ .stage = .fragment, .data = grad_frag_spv },
    };
    const layout = rhi.rpi.Layout{
        .bindings = &.{.{ .name = "u_gradientStrip", .set = 0, .binding = 0, .descriptor_type = .sampled_image, .stages = .{ .fragment = true } }},
        .push_constant = .{ .stages = .{ .fragment = true }, .size = 16 },
    };
    var program = try rhi.rpi.Program.initialize(gpa, dev, &modules, layout);
    defer program.deinit(dev);

    var pool = try rhi.Pool.init(dev, &dev.graphics_queue);
    defer pool.deinit(dev);
    var cmd = try rhi.Cmd.init(dev, &pool);
    defer cmd.deinit(dev, &pool);

    try cmd.begin(dev);

    cmd.image_barrier(dev, .{ .image = &strip_img, .before = .{}, .after = .{ .copy_dst = true } });
    cmd.copy_buffer_to_texture(dev, .{ .src = &strip_stage, .dst = &strip_img, .width = strip_td.width, .height = strip_td.height });
    cmd.image_barrier(dev, .{ .image = &strip_img, .before = .{ .copy_dst = true }, .after = .{ .shader_resource = true } });

    cmd.image_barrier(dev, .{ .image = &color, .before = .{}, .after = .{ .render_target = true } });
    cmd.begin_rendering(dev, .{
        .color_attachments = &.{.{ .view = color_view, .load_op = .clear, .store_op = .store, .clear_color = .{ 0, 0, 0, 1 } }},
        .render_area = .{ .width = n, .height = n },
    });
    cmd.set_viewport(dev, .{ .width = @floatFromInt(n), .height = @floatFromInt(n) });
    cmd.set_scissor(dev, .{ .width = n, .height = n });
    try program.bindPipeline(dev, &cmd, 2, "slug_gradient", .{
        .topology = .triangle_list,
        .colors = &.{.{ .format = .rgba8_unorm }},
        .vertex_streams = &.{.{ .binding = 0, .stride = @sizeOf(Vertex) }},
        .vertex_attributes = &.{
            .{ .location = 0, .binding = 0, .format = .rg32_sfloat, .offset = 0 },
            .{ .location = 1, .binding = 0, .format = .rg32_sfloat, .offset = 8 },
            .{ .location = 2, .binding = 0, .format = .rgba32_sfloat, .offset = 16 },
            .{ .location = 3, .binding = 0, .format = .rgba32_sfloat, .offset = 32 },
        },
    });
    // Push the linear ramp transform + the strip row (gradient_id - 1).
    const pc = [4]f32{ grad.transform.xx, grad.transform.xy, grad.transform.dx, @floatFromInt(gradient_id - 1) };
    program.pushConstants(dev, &cmd, std.mem.asBytes(&pc), 0);
    try program.bindDescriptors(dev, &cmd, 0, &.{
        rhi.rpi.DescriptorBinding.init("u_gradientStrip", rhi.Descriptor.sampledImage(dev, &strip_view), 0),
    }, .graphics);
    cmd.bind_vertex_buffer(dev, &vbuf, 0);
    cmd.draw(dev, .{ .vertex_count = 6 });
    cmd.end_rendering(dev);

    cmd.image_barrier(dev, .{ .image = &color, .before = .{ .render_target = true }, .after = .{ .copy_src = true } });
    cmd.copy_texture_to_buffer(dev, .{ .src = &color, .dst = &readback, .width = n, .height = n });
    cmd.buffer_barrier(dev, .{ .buffer = &readback, .before = .{ .copy_dst = true }, .after = .{ .host_read = true } });
    try cmd.end(dev);

    try dev.graphics_queue.submit(dev, .{ .vk = .{ .cmds = &.{&cmd} } });
    try dev.graphics_queue.wait_queue_idle(dev);

    const mapped = readback.mapped_region orelse return error.BufferNotMapped;
    const out = try gpa.alloc(u8, out_bytes);
    @memcpy(out, mapped[0..out_bytes]);
    return out;
}

/// Sample the MSDF tile array `td` (RGB32F, one shape per layer -- Phase 4's `MsdfAtlas.textureData`)
/// at `layer` on the GPU and return the reconstructed coverage `median(r,g,b)` as an `n` x `n` grid
/// (row 0 = top). Null if the array is empty or `layer` is out of range. This is the GPU counterpart
/// of the msdf-zig tile generation: it proves the tile array is consumable and the median
/// reconstruction runs on the GPU. The RGB32F data is repacked to RGBA32F for upload (3-channel
/// 32-bit sampled images are not universally supported); only rgb is meaningful.
pub fn renderMsdf(gpa: std.mem.Allocator, td: *const slughorn.TextureData, layer: u32, n: u32) !?[]f32 {
    if (td.isEmpty() or layer >= td.depth) return null;
    const ts = td.width; // tiles are square
    const texels: usize = @as(usize, td.width) * td.height * td.depth;

    // RGB32F -> RGBA32F (rgb + 1), reading the source floats alignment-safely.
    const rgba = try gpa.alloc(f32, texels * 4);
    defer gpa.free(rgba);
    var i: usize = 0;
    while (i < texels) : (i += 1) {
        const so = i * 3 * 4;
        rgba[i * 4 + 0] = @bitCast(std.mem.readInt(u32, td.bytes[so + 0 ..][0..4], .little));
        rgba[i * 4 + 1] = @bitCast(std.mem.readInt(u32, td.bytes[so + 4 ..][0..4], .little));
        rgba[i * 4 + 2] = @bitCast(std.mem.readInt(u32, td.bytes[so + 8 ..][0..4], .little));
        rgba[i * 4 + 3] = 1.0;
    }

    var gpu = try Gpu.init(gpa);
    defer gpu.deinit();
    const dev = &gpu.device;

    // -- the tile array texture (RGBA32F, td.depth layers) --
    var arr = try rhi.Image.init(dev, .{
        .format = .rgba32_sfloat,
        .width = td.width,
        .height = td.height,
        .array_layers = td.depth,
        .usage = .{ .sampled = true, .transfer_dst = true },
        .memory_usage = .prefer_device,
    });
    defer arr.deinit(dev);
    var arr_view = try rhi.ImageView.init(dev, &arr, .{ .view_type = .shader_resource_2d_array, .format = .rgba32_sfloat, .aspect = .color, .layer_num = td.depth });
    defer arr_view.deinit(dev);
    const arr_bytes = rgba.len * 4;
    var arr_stage = try rhi.Buffer.init_general(dev, .{ .size = arr_bytes, .persistant_map = true, .sequential_access = true, .buffer_usage = .prefer_host, .usage = .{} });
    defer arr_stage.deinit(dev);
    @memcpy(arr_stage.mapped_region.?[0..arr_bytes], std.mem.sliceAsBytes(rgba));

    // -- R32F coverage target + readback --
    var cov = try rhi.Image.init(dev, .{ .format = .r32_sfloat, .width = n, .height = n, .usage = .{ .color_attachment = true, .transfer_src = true }, .memory_usage = .prefer_device });
    defer cov.deinit(dev);
    var cov_view = try rhi.ImageView.init(dev, &cov, .{ .view_type = .color_attachment, .format = .r32_sfloat, .aspect = .color });
    defer cov_view.deinit(dev);
    const cov_bytes: usize = @as(usize, n) * @as(usize, n) * 4;
    var readback = try rhi.Buffer.init_general(dev, .{ .size = cov_bytes, .persistant_map = true, .sequential_access = false, .buffer_usage = .prefer_host, .usage = .{} });
    defer readback.deinit(dev);

    // -- quad; emCoord carries the tile UV [0,1] --
    const Vertex = extern struct { pos: [2]f32, em: [2]f32, band_xform: [4]f32, shape_data: [4]f32 };
    const z4 = [4]f32{ 0, 0, 0, 0 };
    const verts = [6]Vertex{
        .{ .pos = .{ -1, -1 }, .em = .{ 0, 0 }, .band_xform = z4, .shape_data = z4 },
        .{ .pos = .{ 1, -1 }, .em = .{ 1, 0 }, .band_xform = z4, .shape_data = z4 },
        .{ .pos = .{ -1, 1 }, .em = .{ 0, 1 }, .band_xform = z4, .shape_data = z4 },
        .{ .pos = .{ 1, -1 }, .em = .{ 1, 0 }, .band_xform = z4, .shape_data = z4 },
        .{ .pos = .{ 1, 1 }, .em = .{ 1, 1 }, .band_xform = z4, .shape_data = z4 },
        .{ .pos = .{ -1, 1 }, .em = .{ 0, 1 }, .band_xform = z4, .shape_data = z4 },
    };
    var vbuf = try rhi.Buffer.init_general(dev, .{ .size = @sizeOf(@TypeOf(verts)), .persistant_map = true, .sequential_access = true, .buffer_usage = .prefer_host, .usage = .{ .vertex_buffer = true } });
    defer vbuf.deinit(dev);
    @memcpy(vbuf.mapped_region.?[0..@sizeOf(@TypeOf(verts))], std.mem.asBytes(&verts));

    const modules = [_]rhi.rpi.ModuleStage{
        .{ .stage = .vertex, .data = vert_spv },
        .{ .stage = .fragment, .data = msdf_frag_spv },
    };
    const layout = rhi.rpi.Layout{
        .bindings = &.{.{ .name = "u_msdf", .set = 0, .binding = 0, .descriptor_type = .sampled_image, .stages = .{ .fragment = true } }},
        .push_constant = .{ .stages = .{ .fragment = true }, .size = 16 },
    };
    var program = try rhi.rpi.Program.initialize(gpa, dev, &modules, layout);
    defer program.deinit(dev);

    var pool = try rhi.Pool.init(dev, &dev.graphics_queue);
    defer pool.deinit(dev);
    var cmd = try rhi.Cmd.init(dev, &pool);
    defer cmd.deinit(dev, &pool);

    try cmd.begin(dev);

    cmd.image_barrier(dev, .{ .image = &arr, .before = .{}, .after = .{ .copy_dst = true } });
    cmd.copy_buffer_to_texture(dev, .{ .src = &arr_stage, .dst = &arr, .width = td.width, .height = td.height, .depth = 1, .layer_count = td.depth });
    cmd.image_barrier(dev, .{ .image = &arr, .before = .{ .copy_dst = true }, .after = .{ .shader_resource = true } });

    cmd.image_barrier(dev, .{ .image = &cov, .before = .{}, .after = .{ .render_target = true } });
    cmd.begin_rendering(dev, .{
        .color_attachments = &.{.{ .view = cov_view, .load_op = .clear, .store_op = .store, .clear_color = .{ 0, 0, 0, 1 } }},
        .render_area = .{ .width = n, .height = n },
    });
    cmd.set_viewport(dev, .{ .width = @floatFromInt(n), .height = @floatFromInt(n) });
    cmd.set_scissor(dev, .{ .width = n, .height = n });
    try program.bindPipeline(dev, &cmd, 3, "slug_msdf", .{
        .topology = .triangle_list,
        .colors = &.{.{ .format = .r32_sfloat }},
        .vertex_streams = &.{.{ .binding = 0, .stride = @sizeOf(Vertex) }},
        .vertex_attributes = &.{
            .{ .location = 0, .binding = 0, .format = .rg32_sfloat, .offset = 0 },
            .{ .location = 1, .binding = 0, .format = .rg32_sfloat, .offset = 8 },
            .{ .location = 2, .binding = 0, .format = .rgba32_sfloat, .offset = 16 },
            .{ .location = 3, .binding = 0, .format = .rgba32_sfloat, .offset = 32 },
        },
    });
    const pc = [4]f32{ @floatFromInt(ts), @floatFromInt(layer), 0, 0 };
    program.pushConstants(dev, &cmd, std.mem.asBytes(&pc), 0);
    try program.bindDescriptors(dev, &cmd, 0, &.{
        rhi.rpi.DescriptorBinding.init("u_msdf", rhi.Descriptor.sampledImage(dev, &arr_view), 0),
    }, .graphics);
    cmd.bind_vertex_buffer(dev, &vbuf, 0);
    cmd.draw(dev, .{ .vertex_count = 6 });
    cmd.end_rendering(dev);

    cmd.image_barrier(dev, .{ .image = &cov, .before = .{ .render_target = true }, .after = .{ .copy_src = true } });
    cmd.copy_texture_to_buffer(dev, .{ .src = &cov, .dst = &readback, .width = n, .height = n });
    cmd.buffer_barrier(dev, .{ .buffer = &readback, .before = .{ .copy_dst = true }, .after = .{ .host_read = true } });
    try cmd.end(dev);

    try dev.graphics_queue.submit(dev, .{ .vk = .{ .cmds = &.{&cmd} } });
    try dev.graphics_queue.wait_queue_idle(dev);

    const mapped = readback.mapped_region orelse return error.BufferNotMapped;
    const out = try gpa.alloc(f32, @as(usize, n) * @as(usize, n));
    @memcpy(std.mem.sliceAsBytes(out), mapped[0..cov_bytes]);
    return out;
}
