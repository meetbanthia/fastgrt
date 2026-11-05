// SPDX-License-Identifier: Apache-2.0
// Include after bindings.glsl. Every archive user shares this addressing.
// Host specializes the array length to fit device descriptor limits and fills
// unused descriptors with a valid view of page zero.
#extension GL_EXT_nonuniform_qualifier : require
layout(constant_id = 17) const uint BWD_ARCHIVE_PAGES = 1u;

#if defined(BWD_ARCHIVE_READ_ONLY)
layout(set = 0, binding = 17, scalar) readonly buffer BwdHitIds { uint v[]; } bwdIds[BWD_ARCHIVE_PAGES];
#elif defined(BWD_ARCHIVE_WRITE_ONLY)
layout(set = 0, binding = 17, scalar) writeonly buffer BwdHitIds { uint v[]; } bwdIds[BWD_ARCHIVE_PAGES];
#else
layout(set = 0, binding = 17, scalar) buffer BwdHitIds { uint v[]; } bwdIds[BWD_ARCHIVE_PAGES];
#endif

// Select the page BEFORE multiplying by the ray count: a global offset could
// overflow even though every descriptor-local address is representable.
uint archivePage(uint slot) { return slot >> params.archivePageShift; }
uint archiveCapacity() { return min(uint(BWD_MAXH), params.archiveMaxSlots); }
// Stored ray index: rays are kept in 4x8 pixel blocks ([blockY][blockX][8][4]),
// so a subgroup's slot reads and writes touch few cache lines. Call once per
// active ray; slot accessors take this stored index.
uint archiveRayIndex(uint pix, uint x, uint y, uint b) {
    const uint blocksX = (params.frameW + 3u) / 4u;
    const uint blocksY = (params.frameH + 7u) / 8u;
    return ((b * blocksY + y / 8u) * blocksX + x / 4u) * 32u
           + (y % 8u) * 4u + x % 4u;
}
uint archiveIndex(uint slot, uint storedPix) {
    const uint slotMask = (1u << params.archivePageShift) - 1u;
    return (slot & slotMask) * params.archivePlaneRays + storedPix;
}
#ifndef BWD_ARCHIVE_WRITE_ONLY
uint archiveLoad(uint slot, uint storedPix) {
    return bwdIds[nonuniformEXT(archivePage(slot))].v[archiveIndex(slot, storedPix)];
}
#endif
#ifndef BWD_ARCHIVE_READ_ONLY
void archiveStore(uint slot, uint storedPix, uint id) {
    bwdIds[nonuniformEXT(archivePage(slot))].v[archiveIndex(slot, storedPix)] = id;
}
#endif
