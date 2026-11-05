// SPDX-License-Identifier: Apache-2.0
//
// Shared descriptor-set / parameter layout for all 3DGRT Vulkan shaders.
//
// The parameter UBO mirrors (a subset of) PipelineParameters from the OptiX
// implementation. Buffers that were OptiX "PackedTensorAccessor32" /
// device-pointer launch params are now plain storage buffers whose memory is
// shared with PyTorch CUDA tensors through external-memory interop (see
// src/vulkan_tracer.cpp on the host side).
//
// IMPORTANT: this layout must stay byte-compatible with `ShaderParams` in
// include/3dgrt/vulkan_tracer.h. Keep the two in sync.

#extension GL_EXT_scalar_block_layout : require
// fp16 SH coefficient storage (radiance.sh is float16_t[]; converted to float on read)
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// --- ParticleDensity, byte-identical to the CUDA struct (48 bytes, scalar) ---
struct ParticleDensity {
    vec3  position;
    float density;
    vec4  quaternion;   // (w, x, y, z)
    vec3  scale;
    float padding;
};

// --- Uniform parameters (mirrors host ShaderParams) -------------------------
layout(set = 0, binding = 0, scalar) uniform ParamsUBO {
    vec4  rayToWorld0;     // row 0 of the 3x4 ray-to-world matrix
    vec4  rayToWorld1;     // row 1
    vec4  rayToWorld2;     // row 2
    vec4  aabbMin;         // scene AABB min (xyz), w unused
    vec4  aabbMax;         // scene AABB max (xyz), w unused
    uint  frameW;
    uint  frameH;
    uint  frameB;
    float minTransmittance;
    float minResponse;     // hitMinGaussianResponse
    float minAlpha;        // alphaMinThreshold
    float maxAlpha;        // particle_kernel_max_alpha (0.99)
    float maxParticleSqDist;   // hitMaxParticleSquaredDistance (9.0)
    int   kernelDegree;        // generalized-gaussian degree
    int   sphDegree;           // active SH degree
    uint  archivePageShift;    // log2(slots per archive descriptor page)
    uint  archiveMaxSlots;     // slots allocated per ray
    uint  archivePlaneRays;    // archive slot-plane stride (blocked pixel order, padded)
    uint  statePlaneRays;      // split-state plane stride (rays of the frame)
} params;

const float EPS_T   = 1e-9;
const float INF_T   = 1e20;
const uint  INVALID = 0xFFFFFFFFu;
// bwdCnt high bit: the split forward's fixed rounds did not finish this ray, so the
// tail raygen marched the rest and the archive holds only its first rounds. The apply
// skips such rays and the overflow backward re-traces them whole (exact, no cap).
const uint  BWD_CNT_OVERFLOW = 0x80000000u;
// k-buffer depth: hits gathered per traceRayEXT window. The windowed march is
// exact for any k (identical accepted hits per ray), so k only sets performance.
// The any-hit keeps the window unsorted (O(1) append / evict-max) with only the
// hit distances in the payload and the ids in a per-ray scratch buffer (4 B/hit
// of payload), for which k=24 measured fastest. The host mirrors this value
// (kHitsPerWindow in src/vulkan_tracer.cpp).
#define MAX_HITS_PER_TRACE 24
// Hit-archive slots per ray: the split forward's starting round count (16) x k,
// rounded up to a power of two. The host mirrors this value (kArchiveSlots).
#define BWD_MAXH 512

// Compile-time specialization of the render options, matching the OptiX build
// (-DENABLE_NORMALS / -DENABLE_HIT_COUNTS template args): setup_3dgrt.py passes
// these from the config.
#define NORMALS_ON (ENABLE_NORMALS != 0)
#define HITCOUNTS_ON (ENABLE_HIT_COUNTS != 0)
