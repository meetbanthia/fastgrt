// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <stdexcept>

namespace grt {
// The per-ray hit-id scratch is stored as [4x2 pixel group][slot][pixel within group] (shaders/rt_payload.glsl),
// padding the right and bottom edges of the launch to whole groups.
inline uint32_t scratchRayCount(uint32_t w, uint32_t h, uint32_t b) {
    if (!w || !h || !b || uint64_t(w)*h > UINT32_MAX / uint64_t(b))
        throw std::runtime_error("Invalid scratch image dimensions");
    const uint64_t plane = ((uint64_t(w)+3)/4) * ((uint64_t(h)+1)/2) * 8;
    if (plane > UINT32_MAX / uint64_t(b))
        throw std::runtime_error("Scratch exceeds 32-bit ray addressing");
    return uint32_t(plane*b);
}
inline uint64_t scratchBytes(uint32_t rays, uint32_t slots, uint64_t maxRange) {
    if (!rays || !slots || uint64_t(rays)*slots > UINT32_MAX)
        throw std::runtime_error("Scratch exceeds 32-bit slot addressing");
    const uint64_t bytes = uint64_t(rays)*slots*sizeof(uint32_t);
    if (bytes > maxRange) throw std::runtime_error("Scratch exceeds maxStorageBufferRange");
    return bytes;
}
} // namespace grt
