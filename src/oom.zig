// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

//! Out-of-memory policy: allocation failure panics; it is never reported to callers.
//!
//! Rationale: OOM is unrecoverable here. Threading `error.OutOfMemory` through every signature
//! would bury the errors a caller can actually act on (a band that does not fit a texture row, a
//! uint16 overflow) under one they cannot. This also matches the upstream C++, which throws
//! `std::bad_alloc` and never catches it.

const std = @import("std");

/// Unwraps an allocation, panicking if it failed.
///
/// `oom.must(list.append(gpa, curve))` -- the panic fires here, at the call site that asked for
/// the memory, so that is where the stack trace points.
///
/// Only accepts `Allocator.Error` unions. The switch is exhaustive over `error{OutOfMemory}`, so
/// passing a richer error set (a `BuildError`, say) is a compile error rather than a real failure
/// silently reported as OOM.
pub inline fn must(x: anytype) @typeInfo(@TypeOf(x)).error_union.payload {
    return x catch |err| switch (err) {
        error.OutOfMemory => @panic("slughorn: out of memory"),
    };
}

test "must passes successful allocations through" {
    const gpa = std.testing.allocator;

    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(gpa);

    for (0..1000) |i| must(list.append(gpa, @intCast(i)));

    try std.testing.expectEqual(@as(usize, 1000), list.items.len);
    try std.testing.expectEqual(@as(u32, 999), list.items[999]);

    const buf = must(gpa.alloc(u8, 128));
    defer gpa.free(buf);
    try std.testing.expectEqual(@as(usize, 128), buf.len);
}

test "must forwards the payload type unchanged" {
    const gpa = std.testing.allocator;
    const slice: []u8 = must(gpa.alloc(u8, 4));
    defer gpa.free(slice);
    try std.testing.expectEqual([]u8, @TypeOf(slice));
}
