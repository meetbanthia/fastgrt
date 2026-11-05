#version 460
// SPDX-License-Identifier: Apache-2.0
//
// 3DGRT tracer -- MISS.
//
// Because the any-hit shader ignores every intersection (gather-all), no hit is
// ever committed and the miss shader always runs at the end of traversal. There
// is nothing to do: the payload already holds the gathered hits, which raygen
// integrates. Present only because traceRayEXT requires a miss group.

#extension GL_EXT_ray_tracing : require

// No payload declaration: this miss never touches it.

void main() {
}
