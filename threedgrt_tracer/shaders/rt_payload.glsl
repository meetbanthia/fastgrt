// SPDX-License-Identifier: Apache-2.0
//
// Ray payload for the RT-pipeline 3DGRT forward (rgen <-> rint/rahit/rmiss).
//
// Carries the current window's k nearest particle hit distances between the
// any-hit shader (writer) and the raygen shader (reader / integrator); the hit
// ids go to a per-ray scratch buffer instead (see HitPayload). Physical
// register/stack placement and handoff costs are driver-dependent; the storage
// qualifier does not guarantee on-chip residency.
//
// Requires MAX_HITS_PER_TRACE (from bindings.glsl) to be in scope.

// Scratch belongs to a launch, not to a frame pixel: any-hit and raygen MUST use
// the same index. It is stored as [4x2 pixel group][slot][pixel within group],
// padding the right/bottom edges of the launch. Addresses never depend on
// subgroup IDs.
uint hitScratchBase() {
    const uint groupsX = (gl_LaunchSizeEXT.x + 3u) / 4u;
    const uint groupsY = (gl_LaunchSizeEXT.y + 1u) / 2u;
    const uint group = (gl_LaunchIDEXT.z * groupsY + gl_LaunchIDEXT.y / 2u)
                       * groupsX + gl_LaunchIDEXT.x / 4u;
    const uint lane = (gl_LaunchIDEXT.y % 2u) * 4u + gl_LaunchIDEXT.x % 4u;
    return group * uint(MAX_HITS_PER_TRACE) * 8u + lane;
}
#define hitScratchSlot(base, slot) ((base) + (slot) * 8u)

struct HitPayload {
    // The any-hit's decisions (max compare, evict, rescan) only ever read hdt, so
    // the hit ids live in a per-ray scratch buffer instead (binding 16, write-only
    // in the any-hit, read back by the raygen): 4 B/hit of payload, exact. Every
    // any-hit invocation pays payload bytes; only buffered inserts pay the
    // (fire-and-forget) global id store.
    float hdt[MAX_HITS_PER_TRACE];   // hit distances along the ray
    // The any-hit keeps the k-nearest buffer UNSORTED (O(1) append / O(1)
    // evict-max instead of an O(k) insertion sort on the traversal critical
    // path); the raygen sorts once per window (loadAndSortHits in rt_sort.glsl).
    // The buffer holds the same hits as a sorted one, so integration is identical.
    uint cntMax;                     // hits stored (:16) | index of the max slot (<<16)
    float maxHitDistance;           // hdt[cntMax >> 16], or -INF_T for an empty window
};
