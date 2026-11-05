// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace grt {
// Plane capacity: rays (blocked = false), or the padded capacity of the archive's
// [blockY][blockX][localY:8][localX:4] pixel order (blocked = true; shaders/bwd_archive.glsl).
// Logical ray tensors and hit counts always remain row-major.
inline uint32_t archiveRayCount(uint32_t w, uint32_t h, uint32_t b, bool blocked) {
    const uint64_t logicalPlane = uint64_t(w) * h;
    if (!w || !h || !b || logicalPlane > UINT32_MAX / uint64_t(b))
        throw std::runtime_error("Invalid archive image dimensions");
    const uint64_t plane = blocked ? ((uint64_t(w)+3)/4) * ((uint64_t(h)+7)/8) * 32 : logicalPlane;
    if (plane > UINT32_MAX / uint64_t(b))
        throw std::runtime_error("Padded archive exceeds 32-bit ray addressing");
    return uint32_t(plane * b);
}
// One allocation remains [slot][stored ray]. Each descriptor covers whole slot planes;
// shader indexing is relative to that descriptor, always below maxRange bytes.
struct ArchiveLayout {
    uint32_t rays = 0;
    uint32_t slots = 0;
    uint32_t slotsPerPage = 0;
    uint32_t pageCount = 0;

    uint64_t planeBytes() const { return uint64_t(rays) * sizeof(uint32_t); }
    uint64_t bytes() const { return planeBytes() * slots; }
    uint32_t pageShift() const {
        uint32_t shift = 0;
        for (uint32_t size = slotsPerPage; size > 1; size >>= 1) ++shift;
        return shift;
    }
    uint64_t offset(uint32_t page) const { return planeBytes() * slotsPerPage * page; }
    uint64_t range(uint32_t page) const {
        if (page >= pageCount) throw std::out_of_range("archive page");
        return planeBytes() * std::min(slotsPerPage, slots - page * slotsPerPage);
    }
};

inline ArchiveLayout planArchive(uint64_t rays, uint32_t slots, uint64_t maxRange,
                                 uint64_t alignment, uint32_t descriptors) {
    if (!rays || rays > UINT32_MAX || !slots || !maxRange || !alignment || !descriptors)
        throw std::runtime_error("Invalid archive dimensions or device limits");
    // Even on a device with a larger limit, keep each view in the 32-bit range.
    maxRange = std::min(maxRange, uint64_t(UINT32_MAX));
    const uint64_t plane = rays * sizeof(uint32_t);
    if (plane > maxRange)
        throw std::runtime_error("One archive slot plane exceeds maxStorageBufferRange");
    const uint64_t capacity = std::min(uint64_t(slots), maxRange / plane);
    if (capacity == slots) {
        // A single view needs only the actual slots, even if its logical page
        // capacity rounds up. Power-of-two pages make shader addressing cheap.
        uint32_t perPage = 1;
        while (perPage < slots) perPage *= 2;
        return {uint32_t(rays), slots, perPage, 1u};
    }
    uint32_t perPage = 1;
    while (uint64_t(perPage) * 2 <= capacity) perPage *= 2;
    const uint32_t pages = uint32_t((uint64_t(slots) + perPage - 1) / perPage);
    // Vulkan alignment is a power of two. Smaller power-of-two pages cannot
    // fix an offset alignment failure of the largest feasible page.
    if (pages > 1 && (plane * perPage) % alignment != 0)
        throw std::runtime_error("Archive page offsets cannot meet storage-buffer alignment");
    if (pages > descriptors)
        throw std::runtime_error("Archive exceeds available descriptor pages; reduce resolution or hit capacity");
    return {uint32_t(rays), slots, perPage, pages};
}
} // namespace grt
