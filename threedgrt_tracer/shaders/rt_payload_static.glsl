// SPDX-License-Identifier: Apache-2.0
// Include after the any-hit prd declaration. Every payload array access is
// literal-indexed in SPIR-V; this enables, but does not guarantee, registers.

float payloadDistanceAt(uint slot) {
    float value = INF_T;
#define PAYLOAD_SLOT(I) if (slot == uint(I)) value = prd.hdt[I];
#include "rt_payload_slots.glsl"
#undef PAYLOAD_SLOT
    return value;
}

void setPayloadDistance(uint slot, float value) {
#define PAYLOAD_SLOT(I) if (slot == uint(I)) prd.hdt[I] = value;
#include "rt_payload_slots.glsl"
#undef PAYLOAD_SLOT
}

uint payloadMaxIndex(float m, uint mi) {
    // Ascending scan with strict >: do not replace with >= or a differently
    // ordered reduction, ties decide which slot is evicted.
#define PAYLOAD_SLOT(I) if (prd.hdt[I] > m) { m = prd.hdt[I]; mi = uint(I); }
#include "rt_payload_slots.glsl"
#undef PAYLOAD_SLOT
    prd.maxHitDistance = m;
    return mi;
}
