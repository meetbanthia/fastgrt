// SPDX-License-Identifier: Apache-2.0
//
// VulkanTracer: a drop-in replacement for the OptiX-based OptixTracer. It
// exposes the same Python-facing methods (build_bvh / trace / trace_bwd) and
// constructor signature, but executes 3DGRT with Vulkan ray-tracing pipelines
// and compute shaders. PyTorch CUDA tensors are shared with Vulkan through
// CUDA<->Vulkan external-memory interop (no host round-trips).
//
// Training step, per frame:
//   buildBVH  the per-gaussian geometry and the BVH are rebuilt, or refitted in place
//             between rebuilds (render.max_consecutive_bvh_update), asynchronously:
//             the next trace waits for the build only when it needs it.
//   trace     the split forward: gather/blend rounds (an RT launch gathers each ray's
//             next window of hits, a compute pass blends it) archive every ray's
//             accepted hits, adapting the round count to the scene; a tail launch
//             finishes the rays still alive after the rounds.
//   traceBwd  the backward replays the archived hits in a compute pass (no second
//             traversal) and sums the gradients across the subgroup before the atomic
//             writes (render.subgroup_atomics); an overflow launch re-traces only the
//             rays longer than the archive.
// A forward that no backward follows (autograd off) runs a single raygen instead.

#pragma once

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <cstdint>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>
#include <3dgrt/archive_layout.h>

// Host mirror of the GLSL ParamsUBO (shaders/bindings.glsl). All members are
// 4 bytes, so this packs identically to the GL_EXT_scalar_block_layout UBO.
struct ShaderParams {
    float rayToWorld0[4];
    float rayToWorld1[4];
    float rayToWorld2[4];
    float aabbMin[4];
    float aabbMax[4];
    uint32_t frameW;
    uint32_t frameH;
    uint32_t frameB;
    float minTransmittance;
    float minResponse;
    float minAlpha;
    float maxAlpha;
    float maxParticleSqDist;
    int32_t kernelDegree;
    int32_t sphDegree;
    uint32_t archivePageShift;     // log2(slots per descriptor page)
    uint32_t archiveMaxSlots;      // actual host allocation capacity
    uint32_t archivePlaneRays;     // physical plane stride, including edge padding
    uint32_t statePlaneRays;       // split-state plane stride
};

static_assert(offsetof(ShaderParams, archivePageShift) == 120, "ParamsUBO archive ABI");
static_assert(offsetof(ShaderParams, statePlaneRays) == 132, "ParamsUBO state ABI");
static_assert(sizeof(ShaderParams) == 136, "ParamsUBO scalar layout size");

// Host mirror of the AabbParams UBO of build_icosahedron.comp / build_instances.comp.
struct AabbParams {
    uint32_t numGaussians;
    int32_t kernelDegree;
    int32_t densityClamping;
    float minResponse;
    uint32_t blasRefLo;      // unit-box BLAS device address (instances build)
    uint32_t blasRefHi;
    uint32_t writeIndices;   // build_icosahedron: rewrite the (count-only) triangle indices
};

// A Vulkan buffer whose device memory is exported and imported into CUDA, so it
// can be filled / read from PyTorch tensors with device-to-device copies.
struct SharedBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize usedSize = 0;     // logical size, excluding allocation slack
    VkBufferUsageFlags usage = 0;
    void* cudaPtr = nullptr;                 // CUDA-visible pointer into `memory`
    cudaExternalMemory_t extMem = nullptr;   // CUDA import handle
    int fd = -1;                             // opaque-fd export handle (Linux)
};

class VulkanTracer {
public:
    VulkanTracer(const std::string& path,
                 const std::string& cuda_path,
                 const std::string& pipeline,
                 const std::string& backwardPipeline,
                 const std::string& primitive,
                 float particleKernelDegree,
                 float particleKernelMinResponse,
                 bool particleKernelDensityClamping,
                 int particleRadianceSphDegree,
                 bool enableNormals,
                 bool enableHitCounts);
    ~VulkanTracer();

    // Frees the training-only hit archive (tens of GB at full resolution). Call once training is over, e.g. before
    // the final test: later forwards without a backward bind a placeholder instead, and a recording forward
    // allocates the full archive again.
    void releaseTrainingBuffers();

    // rebuild: build the BVH from scratch (else refit it in place when allow_update permits and the gaussian count
    // is unchanged). allow_update: build it refittable and refit it when rebuild is 0.
    void buildBVH(torch::Tensor mogPos,
                  torch::Tensor mogRot,
                  torch::Tensor mogScl,
                  torch::Tensor mogDns,
                  unsigned int rebuild,
                  bool allow_update,
                  int64_t iteration = -1);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    trace(uint32_t frameNumber,
          torch::Tensor rayToWorld,
          torch::Tensor rayOri,
          torch::Tensor rayDir,
          torch::Tensor particleDensity,
          torch::Tensor particleRadiance,
          uint32_t renderOpts,
          int sphDegree,
          float minTransmittance);

    std::tuple<torch::Tensor, torch::Tensor>
    traceBwd(uint32_t frameNumber,
             torch::Tensor rayToWorld,
             torch::Tensor rayOri,
             torch::Tensor rayDir,
             torch::Tensor rayRad,
             torch::Tensor rayDns,
             torch::Tensor rayHit,
             torch::Tensor rayNrm,
             torch::Tensor particleDensity,
             torch::Tensor particleRadiance,
             torch::Tensor rayRadGrd,
             torch::Tensor rayDnsGrd,
             torch::Tensor rayHitGrd,
             torch::Tensor rayNrmGrd,
             uint32_t renderOpts,
             int sphDegree,
             float minTransmittance);

private:
    // ---- Vulkan core ----
    void createInstance();
    void pickDeviceAndQueues();   // picks the device matching the CUDA device UUID
    void createCommandPool();
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props);

    // ---- buffers / interop ----
    SharedBuffer createSharedBuffer(VkDeviceSize size, VkBufferUsageFlags usage, bool cudaExport = true);
    void destroyBuffer(SharedBuffer& b);
    void uploadSphHalf(const torch::Tensor& particleRadiance, uint32_t gNum, torch::Tensor& tmp);
    void ensureSharedBuffer(SharedBuffer& b, VkDeviceSize size, VkBufferUsageFlags usage, bool cudaExport = true);
    VkDescriptorBufferInfo checkedBufferInfo(const SharedBuffer& b) const;
    void prepareScratch(uint32_t w, uint32_t h, uint32_t batches);
    void prepareArchive(uint32_t w, uint32_t h, uint32_t batches);
    std::vector<VkDescriptorBufferInfo> archiveBufferInfos() const;

    // ---- acceleration structures ----
    void buildUnitBoxBlas();     // single unit-box [-1,1]^3 procedural BLAS, built once (primitive_type=instances)
    // Record the BVH builds into one command buffer (no waits): the icosahedron BLAS, a TLAS.
    void recordTriangleBlas(VkCommandBuffer cmd, uint32_t numGaussians, bool rebuild, bool updatable);
    void recordTlas(VkCommandBuffer cmd, uint32_t numInstances, bool updatable, bool rebuild);
    void createSceneBoundsBuffer();
    void readSceneBounds();
    // Complete the BVH build buildBVH submitted without a host wait: wait for its fence, free its command buffer
    // and descriptor set, read the scene bounds. No-op if none is pending.
    void finishPendingBvh();

    // ---- pipelines / descriptors ----
    VkShaderModule loadShaderModule(const std::string& spvPath);
    void createPipelines();
    void createRtPipeline();      // ray-tracing pipeline + shader binding table
    void createDescriptorPool();

    // ---- dispatch helpers ----
    VkCommandBuffer beginOneShot();
    void endOneShotAndWait(VkCommandBuffer cmd);
    // Per-frame trace dispatches reuse one command buffer + fence instead of
    // allocating/recording/freeing one and waiting for the queue to go idle.
    VkCommandBuffer beginTraceCmd();
    void submitTraceAndWait();

    // configuration
    std::string m_modulePath;
    int m_kernelDegree = 4;
    float m_minResponse = 0.0113f;
    bool m_densityClamping = true;
    int m_sphNumCoeffs = 16;
    bool m_enableNormals = false;
    bool m_enableHitCounts = true;
    // primitive_type=instances, the reference's instanced primitive: one TLAS instance per gaussian over a unit-box
    // procedural BLAS, intersected in the instance's object frame. Otherwise (icosahedron) one BLAS holds every
    // gaussian's icosahedron (20 triangles) under a single TLAS instance.
    bool m_instances = false;
    static constexpr uint32_t kIcosaVerts = 12;
    static constexpr uint32_t kIcosaTris = 20;
    static constexpr float kMaxAlpha = 0.99f;
    static constexpr float kMinAlpha = 1.0f / 255.0f;
    static constexpr float kMaxParticleSqDist = 9.0f;

    // Vulkan handles
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    uint32_t m_queueFamily = 0;
    VkQueue m_queue = VK_NULL_HANDLE;
    VkCommandPool m_cmdPool = VK_NULL_HANDLE;
    VkDescriptorPool m_descPool = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties m_memProps{};
    VkPhysicalDeviceLimits m_limits{};
    uint32_t m_archiveDescriptorCount = 1;
    grt::ArchiveLayout m_archiveLayout{};
    bool m_archiveReleased = false;     // releaseTrainingBuffers ran; no recording forward since

    // ray-tracing function pointers (loaded via vkGetDeviceProcAddr)
    PFN_vkGetBufferDeviceAddress m_vkGetBufferDeviceAddress = nullptr;
    PFN_vkCreateAccelerationStructureKHR m_vkCreateAccelerationStructureKHR = nullptr;
    PFN_vkDestroyAccelerationStructureKHR m_vkDestroyAccelerationStructureKHR = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR m_vkGetAccelerationStructureBuildSizesKHR = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR m_vkCmdBuildAccelerationStructuresKHR = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR m_vkGetAccelerationStructureDeviceAddressKHR = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR m_vkCreateRayTracingPipelinesKHR = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR m_vkGetRayTracingShaderGroupHandlesKHR = nullptr;
    PFN_vkCmdTraceRaysKHR m_vkCmdTraceRaysKHR = nullptr;
    bool m_rtPipelineSupported = false;   // device advertises VK_KHR_ray_tracing_pipeline

    // pipelines
    VkDescriptorSetLayout m_traceSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_tracePipelineLayout = VK_NULL_HANDLE;
    // RT pipeline: four raygens (split gather, split tail, single-launch forward, overflow backward) share one
    // miss and one hit group
    VkPipeline m_rtPipeline = VK_NULL_HANDLE;
    SharedBuffer m_sbtBuffer;            // shader binding table records
    VkStridedDeviceAddressRegionKHR m_sbtRgenSplit{};     // split forward: one gather window per round
    VkStridedDeviceAddressRegionKHR m_sbtRgenTail{};      // split forward: finishes the rays alive after the rounds
    VkStridedDeviceAddressRegionKHR m_sbtRgenPlain{};     // forward that no backward follows
    VkStridedDeviceAddressRegionKHR m_sbtRgenOverflow{};  // backward of the rays longer than the archive
    VkStridedDeviceAddressRegionKHR m_sbtMiss{};
    VkStridedDeviceAddressRegionKHR m_sbtHit{};
    VkStridedDeviceAddressRegionKHR m_sbtCall{};   // empty (no callable shaders)
    VkPipeline m_fwdBlendPipeline = VK_NULL_HANDLE;   // split forward: blends each gathered window
    VkPipeline m_bwdApplyPipeline = VK_NULL_HANDLE;   // backward: replays the archived hits
    SharedBuffer m_bwdHitIds, m_bwdHitCnt;           // hit archive + per-ray slot counts (+ tail counter)
    SharedBuffer m_splitStateF;   // binding 20: split-forward state planes (T, cursor, radiance, depth, ...)
    uint32_t m_splitStateRays = 0;
    // the split forward's round count for the next frame (adapted to the share of rays left alive)
    uint32_t m_adaptiveRounds = 0;

    VkDescriptorSetLayout m_buildSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_buildPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_buildPipeline = VK_NULL_HANDLE;   // build_icosahedron.comp or build_instances.comp

    // acceleration structures + geometry
    SharedBuffer m_vertBuf, m_idxBuf; // icosahedron triangle geometry, filled by build_icosahedron.comp
    SharedBuffer m_posBuf, m_rotBuf, m_sclBuf, m_dnsBuf;  // shared with build_bvh tensors
    VkAccelerationStructureKHR m_blas = VK_NULL_HANDLE;
    SharedBuffer m_blasBuffer;
    SharedBuffer m_blasScratch;
    uint32_t m_blasPrimCount = 0;     // primitives in m_blas (a refit needs the same count)
    bool m_blasUpdatable = false;     // m_blas was built with ALLOW_UPDATE
    VkDeviceSize m_tlasCapacity = 0;  // size m_tlas was created with (reused while it fits)
    uint32_t m_tlasInstCount = 0;     // instances the current m_tlas was built over
    bool m_tlasUpdatable = false;     // m_tlas was built with ALLOW_UPDATE (instances)
    // scene bounds reduced by the build shader: 6 ordered-float uints in host-visible memory
    VkBuffer m_sceneBoundsBuf = VK_NULL_HANDLE;
    VkDeviceMemory m_sceneBoundsMem = VK_NULL_HANDLE;
    uint32_t* m_sceneBoundsPtr = nullptr;
    // the triangle indices only depend on the gaussian count: rewritten only when the count or
    // any shared buffer changed (resource generation)
    uint32_t m_geomIndexCount = 0;
    uint64_t m_geomIndexGen = ~uint64_t(0);
    VkAccelerationStructureKHR m_tlas = VK_NULL_HANDLE;
    SharedBuffer m_tlasBuffer;
    SharedBuffer m_tlasScratch;
    SharedBuffer m_instanceBuffer;
    // instances: a single unit-box BLAS (built once)
    VkAccelerationStructureKHR m_unitBlas = VK_NULL_HANDLE;
    SharedBuffer m_unitBlasBuffer, m_unitBoxBuf;

    uint32_t m_numGaussians = 0;
    // Set by buildBVH (the gaussians changed), cleared by trace/traceBwd after they
    // upload density+SH. Lets a static model (inference: one buildBVH, many traces;
    // or a training iter's fwd+bwd sharing one upload) skip the redundant density/SH
    // copy + fp16 convert. Safe because gaussians only change via buildBVH.
    bool m_gaussiansDirty = true;
    // The last forward's ray inputs and outputs (tensor data pointers) and what they were
    // traced with. Autograd hands that frame's backward the same saved tensors (version-
    // checked, so unmodified), and the shared buffers still hold their contents unless a
    // trace or a reallocation came in between: the backward then skips re-uploading them.
    struct FwdIoRecord {
        bool valid = false;
        uint32_t frame = 0;
        long rays = 0;
        uint64_t resourceGen = 0;
        const void* ptrs[5] = {};   // rayOri, rayDir, rayRad, rayDns, rayHit
    } m_lastFwdIo;
    float m_sceneAabbMin[3] = {-1.f, -1.f, -1.f};
    float m_sceneAabbMax[3] = {1.f, 1.f, 1.f};

    // persistent trace dispatch state: one reusable command buffer + fence, and
    // cached descriptor sets rewritten only when a bound resource was recreated
    // (tracked by m_resourceGen, bumped by ensureSharedBuffer and TLAS recreation).
    VkCommandBuffer m_traceCmd = VK_NULL_HANDLE;
    VkFence m_traceFence = VK_NULL_HANDLE;
    // the BVH build in flight: completed by finishPendingBvh
    VkFence m_bvhFence = VK_NULL_HANDLE;
    VkCommandBuffer m_bvhPendingCmd = VK_NULL_HANDLE;
    VkDescriptorSet m_bvhPendingDs = VK_NULL_HANDLE;
    VkDescriptorSet m_fwdDs = VK_NULL_HANDLE;
    VkDescriptorSet m_bwdDs = VK_NULL_HANDLE;
    uint64_t m_resourceGen = 1;                 // starts above the "never written" marker
    uint64_t m_fwdDsGen = 0;
    uint64_t m_bwdDsGen = 0;

    // reusable UBOs / shared IO buffers (grown lazily)
    SharedBuffer m_paramsUbo;        // ShaderParams (host visible)
    SharedBuffer m_aabbParamsUbo;    // AabbParams
    SharedBuffer m_densityBuf, m_radianceBuf;
    SharedBuffer m_rayOriBuf, m_rayDirBuf;
    SharedBuffer m_outRad, m_outDns, m_outHit, m_outNrm, m_outHc, m_outVis;
    SharedBuffer m_gradRad, m_gradDns, m_gradHit;             // input grads (bwd)
    SharedBuffer m_densityGrad, m_radianceGrad;               // output grads (bwd)
    SharedBuffer m_hitScratch;                                // per-ray hit-id scratch, binding 16

    int m_cudaDevice = 0;
};
