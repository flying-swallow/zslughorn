// Copyright (c) 2026 AlphaPixel LLC (original C++ slughorn), Michael Pollind (Zig port)
// SPDX-License-Identifier: MIT

const std = @import("std");

/// Identifies a shape within an `Atlas`: either a Unicode codepoint (a glyph) or an arbitrary
/// name (hand-authored content).
///
/// Ported from slughorn.hpp:245. Two deliberate changes:
///
///  * The cached hash field is dropped. Upstream precomputes and stores it (slughorn.hpp:249-253)
///    to speed libstdc++'s hashing and to fast-path `operator==`. It is not observable in the
///    atlas output, and `std.array_hash_map` with `store_hash = true` already caches the hash in
///    the table itself -- the same optimization, provided by std, without a field that must be
///    kept in sync with the payload.
///
///  * `name` borrows. `Atlas` interns its own copy on insert; a `Key` you build to look something
///    up does not own anything.
pub const Key = union(enum) {
    codepoint: u32,
    name: []const u8,

    pub fn eql(a: Key, b: Key) bool {
        return switch (a) {
            .codepoint => |cp| switch (b) {
                .codepoint => |other| cp == other,
                .name => false,
            },
            .name => |n| switch (b) {
                .codepoint => false,
                .name => |other| std.mem.eql(u8, n, other),
            },
        };
    }

    pub fn format(self: Key, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        switch (self) {
            .codepoint => |cp| {
                // Match upstream's debug output: printable ASCII shows as the character.
                if (cp >= 0x20 and cp < 0x7f) {
                    try writer.print("U+{X:0>4} '{c}'", .{ cp, @as(u8, @intCast(cp)) });
                } else {
                    try writer.print("U+{X:0>4}", .{cp});
                }
            },
            .name => |n| try writer.print("\"{s}\"", .{n}),
        }
    }
};

/// Hash/equality context for `std.array_hash_map`.
///
/// The tag is mixed into the hash so the codepoint and name namespaces cannot collide even when
/// their payload hashes agree -- the same guarantee upstream documents at slughorn.hpp:280-281,
/// obtained here by seeding rather than by upstream's ad-hoc re-mix.
///
/// The concrete hash *values* differ from the C++ and that is fine: nothing observable depends on
/// them, because the map is insertion-ordered (see `Atlas`).
pub const KeyContext = struct {
    // Note: array_hash_map contexts hash to u32; std.HashMap contexts hash to u64.
    pub fn hash(_: KeyContext, k: Key) u32 {
        var h = std.hash.Wyhash.init(@intFromEnum(std.meta.activeTag(k)));
        switch (k) {
            .codepoint => |cp| h.update(std.mem.asBytes(&cp)),
            .name => |n| h.update(n),
        }
        return @truncate(h.final());
    }

    pub fn eql(_: KeyContext, a: Key, b: Key, _: usize) bool {
        return a.eql(b);
    }
};

/// Insertion-ordered map from `Key` to `V`.
///
/// Insertion order is load-bearing, not incidental. Upstream stores shapes in a
/// `std::unordered_map` (slughorn.hpp:1346) and iterates it to drive texture packing
/// (slughorn.cpp:1173, 1194, 1244, 1337) -- while band texels record *absolute* curve-texture
/// coordinates (slughorn.cpp:1471-1479). So a shape's packed block encodes where previously
/// iterated shapes landed, and the whole byte layout is a function of libstdc++'s bucket order:
/// not reproducible across standard libraries, let alone across a port. Ordering by insertion
/// makes our output deterministic and independent of hash values.
pub fn KeyMap(comptime V: type) type {
    return std.array_hash_map.Custom(Key, V, KeyContext, true);
}

test "Key equality and namespace disjointness" {
    const a: Key = .{ .codepoint = 65 };
    const b: Key = .{ .codepoint = 65 };
    const c: Key = .{ .codepoint = 66 };
    const n: Key = .{ .name = "A" };
    const n2: Key = .{ .name = "A" };

    try std.testing.expect(a.eql(b));
    try std.testing.expect(!a.eql(c));
    try std.testing.expect(n.eql(n2));
    // A codepoint and a name never compare equal, whatever their payloads.
    try std.testing.expect(!a.eql(n));
    try std.testing.expect(!n.eql(a));
}

test "Key round-trips through KeyMap" {
    const gpa = std.testing.allocator;
    var map: KeyMap(u32) = .empty;
    defer map.deinit(gpa);

    try map.put(gpa, .{ .codepoint = 65 }, 100);
    try map.put(gpa, .{ .name = "logo" }, 200);
    try map.put(gpa, .{ .codepoint = 66 }, 300);

    try std.testing.expectEqual(@as(?u32, 100), map.get(.{ .codepoint = 65 }));
    try std.testing.expectEqual(@as(?u32, 200), map.get(.{ .name = "logo" }));
    try std.testing.expectEqual(@as(?u32, 300), map.get(.{ .codepoint = 66 }));
    try std.testing.expectEqual(@as(?u32, null), map.get(.{ .name = "missing" }));
    // "A" as a name must not find codepoint 65.
    try std.testing.expectEqual(@as(?u32, null), map.get(.{ .name = "A" }));

    // Iteration follows insertion order, which is what makes packing reproducible.
    const keys = map.keys();
    try std.testing.expectEqual(@as(usize, 3), keys.len);
    try std.testing.expectEqual(@as(u32, 65), keys[0].codepoint);
    try std.testing.expectEqualStrings("logo", keys[1].name);
    try std.testing.expectEqual(@as(u32, 66), keys[2].codepoint);
}

test "Key formats readably" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("U+0041 'A'", try std.fmt.bufPrint(&buf, "{f}", .{Key{ .codepoint = 65 }}));
    try std.testing.expectEqualStrings("U+1F600", try std.fmt.bufPrint(&buf, "{f}", .{Key{ .codepoint = 0x1F600 }}));
    try std.testing.expectEqualStrings("\"logo\"", try std.fmt.bufPrint(&buf, "{f}", .{Key{ .name = "logo" }}));
}
