// SPDX-License-Identifier: Apache-2.0
//
// VulkanTracer implementation: 3DGRT on Vulkan ray-tracing pipelines and compute
// shaders with CUDA<->Vulkan external-memory interop. See include/3dgrt/vulkan_tracer.h
// for the passes of a training step.
//
// This file targets Linux + NVIDIA + Vulkan 1.3 (VK_KHR_ray_tracing_pipeline,
// VK_KHR_acceleration_structure) + CUDA. Buffers shared with PyTorch are
// exported from Vulkan as opaque file descriptors and imported into CUDA.

#include <3dgrt/vulkan_tracer.h>
#include <3dgrt/scratch_layout.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// error-check helpers
// ---------------------------------------------------------------------------
#define VK_CHECK(x)                                                                       \
    do {                                                                                  \
        VkResult err = (x);                                                               \
        if (err != VK_SUCCESS) {                                                          \
            throw std::runtime_error(std::string("Vulkan error ") + std::to_string(err) + \
                                     " at " + __FILE__ ":" + std::to_string(__LINE__));    \
        }                                                                                 \
    } while (0)

#define CUDA_CHECK(x)                                                                          \
    do {                                                                                       \
        cudaError_t err = (x);                                                                 \
        if (err != cudaSuccess) {                                                              \
            throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(err) +    \
                                     " at " + __FILE__ ":" + std::to_string(__LINE__));         \
        }                                                                                      \
    } while (0)

namespace {
// k-buffer depth: hits gathered per traceRayEXT window (MAX_HITS_PER_TRACE in shaders/bindings.glsl).
constexpr uint32_t kHitsPerWindow = 24;
// Hit-archive slots per ray (BWD_MAXH in shaders/bindings.glsl): the split forward's starting round count x k,
// rounded up to a power of two. It caps the split forward's round count at kArchiveSlots / kHitsPerWindow.
constexpr uint32_t kArchiveSlots = 512;
// The split forward's round count starts at kSplitRoundsStart and then follows the scene: each frame the next
// frame's rounds go up by one while more than 1% of the rays reached the tail raygen and down by one below 0.25%,
// between kSplitRoundsMin and the archive's capacity in windows.
constexpr uint32_t kSplitRoundsStart = 16;
constexpr uint32_t kSplitRoundsMin = 4;
// Growth factor for shared-buffer reallocation. Exact-fit growth reallocates every gaussian-sized buffer at every
// densification event with a monotonically larger size, so no freed block is ever reusable, which was measured to
// slow the trace down progressively over a training run. Slack growth collapses O(events) reallocations into
// O(log growth).
constexpr double kAllocSlack = 2.2;

cudaStream_t currentStream() { return at::cuda::getCurrentCUDAStream().stream(); }

// The instance buffer holds one instance (icosahedron: written by vkCmdUpdateBuffer) or one per gaussian
// (instances: written by build_instances.comp).
const VkBufferUsageFlags kInstanceBufferUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
    | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
} // namespace

// ===========================================================================
// Construction
// ===========================================================================
VulkanTracer::VulkanTracer(const std::string& path,
                           const std::string& /*cuda_path*/,
                           const std::string& /*pipeline*/,
                           const std::string& /*backwardPipeline*/,
                           const std::string& primitive,
                           float particleKernelDegree,
                           float particleKernelMinResponse,
                           bool particleKernelDensityClamping,
                           int particleRadianceSphDegree,
                           bool enableNormals,
                           bool enableHitCounts)
    : m_modulePath(path),
      m_kernelDegree(static_cast<int>(particleKernelDegree)),
      m_minResponse(particleKernelMinResponse),
      m_densityClamping(particleKernelDensityClamping),
      m_sphNumCoeffs((particleRadianceSphDegree + 1) * (particleRadianceSphDegree + 1)),
      m_enableNormals(enableNormals),
      m_enableHitCounts(enableHitCounts),
      m_instances(primitive == "instances") {
    if (primitive != "icosahedron" && primitive != "instances")
        throw std::runtime_error("render.primitive_type=" + primitive +
                                 " is not supported by the Vulkan tracer (use icosahedron or instances)");
    CUDA_CHECK(cudaGetDevice(&m_cudaDevice));

    createInstance();
    pickDeviceAndQueues();
    if (!m_rtPipelineSupported)
        throw std::runtime_error("VK_KHR_ray_tracing_pipeline not supported on this device");
    createCommandPool();
    createDescriptorPool();
    createPipelines();
}

VulkanTracer::~VulkanTracer() {
    if (m_device) {
        vkDeviceWaitIdle(m_device);

        auto destroyAS = [&](VkAccelerationStructureKHR& as) {
            if (as) { m_vkDestroyAccelerationStructureKHR(m_device, as, nullptr); as = VK_NULL_HANDLE; }
        };
        destroyAS(m_blas);
        destroyAS(m_tlas);
        destroyAS(m_unitBlas);

        for (SharedBuffer* b : {&m_vertBuf, &m_idxBuf, &m_posBuf, &m_rotBuf, &m_sclBuf, &m_dnsBuf,
                                &m_blasBuffer, &m_blasScratch, &m_tlasBuffer, &m_tlasScratch,
                                &m_instanceBuffer, &m_paramsUbo, &m_aabbParamsUbo,
                                &m_densityBuf, &m_radianceBuf, &m_rayOriBuf, &m_rayDirBuf,
                                &m_outRad, &m_outDns, &m_outHit, &m_outNrm, &m_outHc, &m_outVis,
                                &m_gradRad, &m_gradDns, &m_gradHit, &m_densityGrad, &m_radianceGrad,
                                &m_sbtBuffer, &m_hitScratch, &m_bwdHitIds, &m_bwdHitCnt, &m_splitStateF,
                                &m_unitBlasBuffer, &m_unitBoxBuf}) {
            destroyBuffer(*b);
        }

        if (m_rtPipeline) vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
        if (m_bwdApplyPipeline) vkDestroyPipeline(m_device, m_bwdApplyPipeline, nullptr);
        if (m_fwdBlendPipeline) vkDestroyPipeline(m_device, m_fwdBlendPipeline, nullptr);
        if (m_buildPipeline) vkDestroyPipeline(m_device, m_buildPipeline, nullptr);
        if (m_tracePipelineLayout) vkDestroyPipelineLayout(m_device, m_tracePipelineLayout, nullptr);
        if (m_buildPipelineLayout) vkDestroyPipelineLayout(m_device, m_buildPipelineLayout, nullptr);
        if (m_traceSetLayout) vkDestroyDescriptorSetLayout(m_device, m_traceSetLayout, nullptr);
        if (m_buildSetLayout) vkDestroyDescriptorSetLayout(m_device, m_buildSetLayout, nullptr);
        if (m_traceFence) vkDestroyFence(m_device, m_traceFence, nullptr);
        if (m_bvhFence) vkDestroyFence(m_device, m_bvhFence, nullptr);
        if (m_sceneBoundsMem) { vkUnmapMemory(m_device, m_sceneBoundsMem); vkFreeMemory(m_device, m_sceneBoundsMem, nullptr); }
        if (m_sceneBoundsBuf) vkDestroyBuffer(m_device, m_sceneBoundsBuf, nullptr);
        if (m_descPool) vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
        if (m_cmdPool)  vkDestroyCommandPool(m_device, m_cmdPool, nullptr);
        vkDestroyDevice(m_device, nullptr);
    }
    if (m_instance) vkDestroyInstance(m_instance, nullptr);
}

// ===========================================================================
// Vulkan context
// ===========================================================================
void VulkanTracer::createInstance() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "3dgrt-vulkan";
    app.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> instExt = {
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    };

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = static_cast<uint32_t>(instExt.size());
    ci.ppEnabledExtensionNames = instExt.data();
    VK_CHECK(vkCreateInstance(&ci, nullptr, &m_instance));
}

void VulkanTracer::pickDeviceAndQueues() {
    // Match the Vulkan physical device to the active CUDA device by UUID.
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, m_cudaDevice));

    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(m_instance, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    VK_CHECK(vkEnumeratePhysicalDevices(m_instance, &count, devices.data()));

    for (auto dev : devices) {
        VkPhysicalDeviceIDProperties idProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &idProps;
        vkGetPhysicalDeviceProperties2(dev, &props2);
        if (std::memcmp(idProps.deviceUUID, prop.uuid.bytes, VK_UUID_SIZE) == 0) {
            m_physicalDevice = dev;
            break;
        }
    }
    if (!m_physicalDevice && !devices.empty()) {
        std::cerr << "[3dgrt-vulkan] WARNING: could not match CUDA device UUID; using device 0\n";
        m_physicalDevice = devices[0];
    }
    if (!m_physicalDevice) throw std::runtime_error("No Vulkan physical device found");

    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &m_memProps);
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
    m_limits = props.limits;

    // The hit archive (binding 17) is an array of storage-buffer views. Every binding is visible to each trace
    // stage: reserve the other storage buffers plus the UBO and acceleration structure in the aggregate limit.
    const uint32_t storageBudget = std::min(m_limits.maxPerStageDescriptorStorageBuffers,
                                            m_limits.maxDescriptorSetStorageBuffers);
    if (storageBudget <= 20 || m_limits.maxPerStageResources <= 22)
        throw std::runtime_error("Device descriptor limits cannot accommodate the hit archive");
    // Up to 64 page views: a full-resolution frame of ~17M rays (MipNeRF360 garden) holds 32 slots per 4 GiB view,
    // so its 512 slots need 16 pages. Unused views alias page 0; the page count of a layout that fits fewer pages
    // does not change (planArchive), so neither do its addresses.
    m_archiveDescriptorCount = std::min(64u, std::min(storageBudget - 20, m_limits.maxPerStageResources - 22));

    // The archive is indexed by a per-ray (divergent) page: enable nonuniform SSBO descriptor indexing.
    VkPhysicalDeviceDescriptorIndexingFeatures descriptorIndexing{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    supported.pNext = &descriptorIndexing;
    vkGetPhysicalDeviceFeatures2(m_physicalDevice, &supported);
    if (!descriptorIndexing.shaderStorageBufferArrayNonUniformIndexing ||
        !supported.features.shaderStorageBufferArrayDynamicIndexing)
        throw std::runtime_error("Paged hit archives require storage-buffer array indexing support");
    descriptorIndexing = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    descriptorIndexing.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
    VkPhysicalDeviceFeatures enabledFeatures{};
    enabledFeatures.shaderStorageBufferArrayDynamicIndexing = VK_TRUE;

    // pick a compute-capable queue family
    uint32_t qCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &qCount, nullptr);
    std::vector<VkQueueFamilyProperties> qProps(qCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &qCount, qProps.data());
    m_queueFamily = 0;
    for (uint32_t i = 0; i < qCount; ++i) {
        if (qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { m_queueFamily = i; break; }
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = m_queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    // The tracer needs the ray-tracing pipeline (the constructor throws if absent).
    {
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extCount, exts.data());
        for (const auto& e : exts) {
            if (std::strcmp(e.extensionName, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) == 0)
                m_rtPipelineSupported = true;
        }
    }
    if (!m_rtPipelineSupported) return;

    std::vector<const char*> devExt = {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME,
        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,          // fp16 SH storage
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,    // fp16 conversion in shaders
    };

    // fp16 SH storage: read float16_t from SSBOs (storageBuffer16BitAccess) + convert in-shader (shaderFloat16)
    VkPhysicalDevice16BitStorageFeatures stor16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    stor16.storageBuffer16BitAccess = VK_TRUE;

    VkPhysicalDeviceShaderFloat16Int8Features f16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    f16.shaderFloat16 = VK_TRUE;
    f16.pNext = &stor16;

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    atomicFloat.shaderBufferFloat32AtomicAdd = VK_TRUE;
    atomicFloat.shaderBufferFloat32Atomics = VK_TRUE;
    atomicFloat.pNext = &f16;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    accel.accelerationStructure = VK_TRUE;
    accel.pNext = &atomicFloat;

    VkPhysicalDeviceBufferDeviceAddressFeatures bda{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    bda.bufferDeviceAddress = VK_TRUE;
    bda.pNext = &accel;

    VkPhysicalDeviceScalarBlockLayoutFeatures scalar{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES};
    scalar.scalarBlockLayout = VK_TRUE;
    scalar.pNext = &bda;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipe{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    rtPipe.rayTracingPipeline = VK_TRUE;
    // required for VK_PIPELINE_CREATE_RAY_TRACING_SKIP_AABBS_BIT_KHR (triangle-only
    // traversal specialization; OptiX equivalent: usesPrimitiveTypeFlags = TRIANGLE)
    rtPipe.rayTraversalPrimitiveCulling = VK_TRUE;
    rtPipe.pNext = &scalar;
    descriptorIndexing.pNext = &rtPipe;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &descriptorIndexing;
    dci.pEnabledFeatures = &enabledFeatures;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(devExt.size());
    dci.ppEnabledExtensionNames = devExt.data();
    VK_CHECK(vkCreateDevice(m_physicalDevice, &dci, nullptr, &m_device));

    vkGetDeviceQueue(m_device, m_queueFamily, 0, &m_queue);

    auto load = [&](const char* name) { return vkGetDeviceProcAddr(m_device, name); };
    m_vkGetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddress)load("vkGetBufferDeviceAddress");
    m_vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)load("vkCreateAccelerationStructureKHR");
    m_vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)load("vkDestroyAccelerationStructureKHR");
    m_vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)load("vkGetAccelerationStructureBuildSizesKHR");
    m_vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)load("vkCmdBuildAccelerationStructuresKHR");
    m_vkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)load("vkGetAccelerationStructureDeviceAddressKHR");
    m_vkCreateRayTracingPipelinesKHR = (PFN_vkCreateRayTracingPipelinesKHR)load("vkCreateRayTracingPipelinesKHR");
    m_vkGetRayTracingShaderGroupHandlesKHR = (PFN_vkGetRayTracingShaderGroupHandlesKHR)load("vkGetRayTracingShaderGroupHandlesKHR");
    m_vkCmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)load("vkCmdTraceRaysKHR");
}

void VulkanTracer::createCommandPool() {
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = m_queueFamily;
    VK_CHECK(vkCreateCommandPool(m_device, &ci, nullptr, &m_cmdPool));
}

uint32_t VulkanTracer::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < m_memProps.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (m_memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw std::runtime_error("No suitable Vulkan memory type");
}

VkCommandBuffer VulkanTracer::beginOneShot() {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = m_cmdPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(m_device, &ai, &cmd));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    return cmd;
}

void VulkanTracer::endOneShotAndWait(VkCommandBuffer cmd) {
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(m_queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(m_queue));   // coarse sync (one-time builds only; traces use the fence path)
    vkFreeCommandBuffers(m_device, m_cmdPool, 1, &cmd);
}

// Persistent command buffer + fence for the per-frame trace dispatches: created
// once and reused, and the wait is a fence on this submission only.
VkCommandBuffer VulkanTracer::beginTraceCmd() {
    if (!m_traceCmd) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = m_cmdPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(m_device, &ai, &m_traceCmd));
    }
    if (!m_traceFence) {
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(m_device, &fi, nullptr, &m_traceFence));
    }
    // pool has RESET_COMMAND_BUFFER_BIT: vkBeginCommandBuffer implicitly resets
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(m_traceCmd, &bi));
    return m_traceCmd;
}

void VulkanTracer::submitTraceAndWait() {
    VK_CHECK(vkEndCommandBuffer(m_traceCmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_traceCmd;
    VK_CHECK(vkQueueSubmit(m_queue, 1, &si, m_traceFence));
    VK_CHECK(vkWaitForFences(m_device, 1, &m_traceFence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(m_device, 1, &m_traceFence));
}

// ===========================================================================
// Buffers + CUDA interop
// ===========================================================================
// cudaExport=false: a Vulkan-only buffer (CUDA never reads it). Plain device-local memory is faster for the large
// buffers the GPU passes stream through (the BVH and its build inputs, the hit scratch and archive): the BLAS refit
// of bicycle x4 (6.25M gaussians) reads its inputs ~5x slower from exported memory.
SharedBuffer VulkanTracer::createSharedBuffer(VkDeviceSize size, VkBufferUsageFlags usage, bool cudaExport) {
    SharedBuffer b;
    b.size = size;
    b.usedSize = size;
    b.usage = usage;

    VkExternalMemoryBufferCreateInfo extBuf{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    extBuf.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.pNext = cudaExport ? &extBuf : nullptr;
    bi.size = size;
    bi.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(m_device, &bi, nullptr, &b.buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(m_device, b.buffer, &req);

    VkExportMemoryAllocateInfo exportInfo{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    flags.pNext = cudaExport ? &exportInfo : nullptr;

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &flags;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(m_device, &mai, nullptr, &b.memory));
    VK_CHECK(vkBindBufferMemory(m_device, b.buffer, b.memory, 0));
    if (!cudaExport) return b;   // Vulkan-only buffer: no fd, no CUDA mapping (cudaPtr stays null)

    // export to opaque fd, import into CUDA
    auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(m_device, "vkGetMemoryFdKHR");
    VkMemoryGetFdInfoKHR fdInfo{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    fdInfo.memory = b.memory;
    fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VK_CHECK(vkGetMemoryFdKHR(m_device, &fdInfo, &b.fd));

    cudaExternalMemoryHandleDesc desc{};
    desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = b.fd;
    desc.size = req.size;
    CUDA_CHECK(cudaImportExternalMemory(&b.extMem, &desc));

    cudaExternalMemoryBufferDesc bufDesc{};
    bufDesc.offset = 0;
    bufDesc.size = req.size;
    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&b.cudaPtr, b.extMem, &bufDesc));

    return b;
}

void VulkanTracer::destroyBuffer(SharedBuffer& b) {
    // CUDA requires the mapped buffer to be cudaFree'd BEFORE destroying the external
    // memory object; skipping it leaks the device memory (so it is never reclaimed by
    // Vulkan either).
    if (b.cudaPtr) { cudaFree(b.cudaPtr); b.cudaPtr = nullptr; }
    if (b.extMem)  { cudaDestroyExternalMemory(b.extMem); b.extMem = nullptr; }
    if (b.buffer)  { vkDestroyBuffer(m_device, b.buffer, nullptr); b.buffer = VK_NULL_HANDLE; }
    if (b.memory)  { vkFreeMemory(m_device, b.memory, nullptr); b.memory = VK_NULL_HANDLE; }
    b.size = 0;
    b.usedSize = 0;
    b.usage = 0;
    b.fd = -1;
}

void VulkanTracer::ensureSharedBuffer(SharedBuffer& b, VkDeviceSize size, VkBufferUsageFlags usage, bool cudaExport) {
    if (b.buffer && b.size >= size) {
        if (b.usedSize != size) { b.usedSize = size; ++m_resourceGen; }
        return;
    }
    // Growing an existing buffer: over-allocate by kAllocSlack so steady growth (densification)
    // reallocates O(log growth) times instead of every event. First allocation stays exact-fit.
    VkDeviceSize alloc = size;
    if (b.buffer) {
        VkDeviceSize slack = (VkDeviceSize)((double)size * kAllocSlack);
        alloc = slack > size ? slack : size;
    }
    if (b.buffer) destroyBuffer(b);
    b = createSharedBuffer(alloc, usage, cudaExport);
    b.usedSize = size;
    ++m_resourceGen;   // cached descriptor sets referencing the old buffer are stale
}

VkDescriptorBufferInfo VulkanTracer::checkedBufferInfo(const SharedBuffer& b) const {
    const uint64_t limit = (b.usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
        ? m_limits.maxUniformBufferRange : m_limits.maxStorageBufferRange;
    if (!b.buffer || !b.usedSize || b.usedSize > b.size || b.usedSize > limit)
        throw std::runtime_error("Invalid descriptor buffer range: bytes=" + std::to_string(b.usedSize)
                                 + " limit=" + std::to_string(limit));
    return {b.buffer, 0, b.usedSize};
}

void VulkanTracer::prepareScratch(uint32_t w, uint32_t h, uint32_t batches) {
    const uint32_t rays = grt::scratchRayCount(w, h, batches);
    const uint64_t bytes = grt::scratchBytes(rays, kHitsPerWindow, m_limits.maxStorageBufferRange);
    ensureSharedBuffer(m_hitScratch, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
}

void VulkanTracer::releaseTrainingBuffers() {
    finishPendingBvh();
    VK_CHECK(vkQueueWaitIdle(m_queue));   // nothing in flight may still use the archive
    if (m_bwdHitIds.buffer) destroyBuffer(m_bwdHitIds);
    m_archiveLayout = {};
    m_archiveReleased = true;
    ++m_resourceGen;                      // cached descriptor sets referenced the freed buffer
}

void VulkanTracer::prepareArchive(uint32_t w, uint32_t h, uint32_t batches) {
    const uint32_t rays = grt::archiveRayCount(w, h, batches, false);
    const uint32_t storedRays = grt::archiveRayCount(w, h, batches, true);
    const auto layout = grt::planArchive(storedRays, kArchiveSlots, m_limits.maxStorageBufferRange,
                                         m_limits.minStorageBufferOffsetAlignment, m_archiveDescriptorCount);
    const bool changed = layout.rays != m_archiveLayout.rays || layout.slots != m_archiveLayout.slots;
    ensureSharedBuffer(m_bwdHitIds, layout.bytes(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    // + 1 word: the tail raygen's count of rays alive at the round cap
    ensureSharedBuffer(m_bwdHitCnt, sizeof(uint32_t) * (rays + 1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (changed) {
        m_archiveLayout = layout;
        ++m_resourceGen; // offsets/ranges can change even without reallocating
    }
}

std::vector<VkDescriptorBufferInfo> VulkanTracer::archiveBufferInfos() const {
    std::vector<VkDescriptorBufferInfo> views(m_archiveDescriptorCount);
    for (uint32_t i = 0; i < m_archiveDescriptorCount; ++i) {
        const uint32_t page = i < m_archiveLayout.pageCount ? i : 0u;
        const uint64_t offset = m_archiveLayout.offset(page), range = m_archiveLayout.range(page);
        if (!range || range > m_limits.maxStorageBufferRange ||
            offset % m_limits.minStorageBufferOffsetAlignment ||
            offset > m_bwdHitIds.usedSize || range > m_bwdHitIds.usedSize - offset)
            throw std::runtime_error("Invalid hit-archive descriptor view");
        views[i] = {m_bwdHitIds.buffer, offset, range};
    }
    return views;
}

static VkDeviceAddress bufferAddress(VkDevice dev, PFN_vkGetBufferDeviceAddress fn, VkBuffer buf) {
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buf;
    return fn(dev, &info);
}

// ===========================================================================
// Pipelines + descriptor sets
// ===========================================================================
VkShaderModule VulkanTracer::loadShaderModule(const std::string& spvPath) {
    std::ifstream f(spvPath, std::ios::ate | std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open SPIR-V: " + spvPath);
    size_t sz = (size_t)f.tellg();
    std::vector<char> data(sz);
    f.seekg(0);
    f.read(data.data(), sz);
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = sz;
    ci.pCode = reinterpret_cast<const uint32_t*>(data.data());
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(m_device, &ci, nullptr, &mod));
    return mod;
}

void VulkanTracer::createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 3> sizes{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8},
        // two trace sets (forward, backward) each hold ~20 storage views + the archive pages, plus the build sets
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128 + 2 * m_archiveDescriptorCount},
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 8},
    }};
    VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.maxSets = 16;
    ci.poolSizeCount = (uint32_t)sizes.size();
    ci.pPoolSizes = sizes.data();
    VK_CHECK(vkCreateDescriptorPool(m_device, &ci, nullptr, &m_descPool));
}

// Trace descriptor set (shaders/bindings.glsl and the trace shaders):
//   0 params UBO, 1 TLAS, 2 density, 3 SH (fp16), 4/5 ray origins/directions,
//   6/7/8 forward radiance/density/hit distances, 9 normals, 10 hit counts, 11 visibility,
//   10/11/12 in the backward: incoming radiance/density/hit-distance gradients,
//   14/15 gaussian gradients, 16 per-ray hit-id scratch, 17 hit archive (paged views),
//   18 per-ray archive slot counts, 20 split-forward state planes.
void VulkanTracer::createPipelines() {
    const std::string spvDir = m_modulePath + "/shaders/build/";

    // ---- trace descriptor-set layout, shared by the RT pipeline and the compute passes ----
    const VkShaderStageFlags traceStages =
        VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR |
        VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
    {
        std::vector<VkDescriptorSetLayoutBinding> binds;
        binds.push_back({0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, traceStages, nullptr});
        binds.push_back({1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, traceStages, nullptr});
        for (uint32_t i : {2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 14u, 15u, 16u, 17u, 18u, 20u})
            binds.push_back({i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, i == 17 ? m_archiveDescriptorCount : 1u,
                             traceStages, nullptr});
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = (uint32_t)binds.size();
        ci.pBindings = binds.data();
        VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &m_traceSetLayout));

        // push constant: the split forward's round index (gather raygen + blend pass)
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = sizeof(uint32_t);

        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &m_traceSetLayout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_tracePipelineLayout));
    }

    // ---- geometry-build descriptor-set layout: 0 UBO + 1..4 gaussians (pos, rot, scl, dns),
    //      5 vertices | instances, 6 triangle indices (icosahedron), 7 scene bounds ----
    {
        std::vector<VkDescriptorSetLayoutBinding> binds;
        binds.push_back({0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        for (uint32_t i = 1; i <= 7; ++i)
            binds.push_back({i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = (uint32_t)binds.size();
        ci.pBindings = binds.data();
        VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &m_buildSetLayout));

        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &m_buildSetLayout;
        VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_buildPipelineLayout));
    }

    auto makeComputePipeline = [&](const std::string& spv, VkPipelineLayout layout) {
        VkSpecializationMapEntry archiveEntry{17, 0, sizeof(m_archiveDescriptorCount)};
        VkSpecializationInfo archiveSpec{1, &archiveEntry, sizeof(m_archiveDescriptorCount), &m_archiveDescriptorCount};
        VkShaderModule mod = loadShaderModule(spv);
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName = "main";
        stage.pSpecializationInfo = &archiveSpec;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = stage;
        ci.layout = layout;
        VkPipeline pipe;
        VK_CHECK(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe));
        vkDestroyShaderModule(m_device, mod, nullptr);
        return pipe;
    };

    m_buildPipeline = makeComputePipeline(spvDir + (m_instances ? "build_instances.comp.spv"
                                                                : "build_icosahedron.comp.spv"), m_buildPipelineLayout);

    // the apply's gradient scatter is written for 32-wide subgroups (lane sets in one uint)
    VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 sgProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    sgProps.pNext = &sg;
    vkGetPhysicalDeviceProperties2(m_physicalDevice, &sgProps);
    if (sg.subgroupSize != 32)
        throw std::runtime_error("The backward apply pass needs 32-wide subgroups (device has " +
                                 std::to_string(sg.subgroupSize) + ")");
    m_bwdApplyPipeline = makeComputePipeline(spvDir + "trace_backward_apply.comp.spv", m_tracePipelineLayout);
    m_fwdBlendPipeline = makeComputePipeline(spvDir + "trace_forward_split_blend.comp.spv", m_tracePipelineLayout);

    createRtPipeline();
}

// ===========================================================================
// Ray-tracing pipeline + shader binding table
// ===========================================================================
void VulkanTracer::createRtPipeline() {
    const std::string spvDir = m_modulePath + "/shaders/build/";
    VkSpecializationMapEntry archiveEntry{17, 0, sizeof(m_archiveDescriptorCount)};
    VkSpecializationInfo archiveSpec{1, &archiveEntry, sizeof(m_archiveDescriptorCount), &m_archiveDescriptorCount};

    // ---- shader stages ----
    // Icosahedron: TRIANGLES hit group with NO intersection shader (hardware triangle test); the gaussian math runs
    // only on gathered hits. primitive_type=instances: the reference's instanced primitive -- a PROCEDURAL hit group
    // (object-frame intersection + any-hit with id = instance index) under the same raygens (their back-face culling
    // flag only applies to triangles).
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    auto addStage = [&](const std::string& spv, VkShaderStageFlagBits st) -> uint32_t {
        VkPipelineShaderStageCreateInfo s{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        s.stage = st; s.module = loadShaderModule(spvDir + spv); s.pName = "main";
        s.pSpecializationInfo = &archiveSpec;
        stages.push_back(s);
        return (uint32_t)stages.size() - 1;
    };
    const uint32_t sRgenSplit    = addStage("trace_forward_split_gather.rgen.spv", VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    const uint32_t sRgenTail     = addStage("trace_forward_split_tail.rgen.spv", VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    const uint32_t sRgenPlain    = addStage("trace_forward.rgen.spv", VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    const uint32_t sRgenOverflow = addStage("trace_backward_overflow.rgen.spv", VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    const uint32_t sMiss         = addStage("trace_forward.rmiss.spv", VK_SHADER_STAGE_MISS_BIT_KHR);
    const uint32_t sRint  = m_instances ? addStage("trace_forward_inst.rint.spv", VK_SHADER_STAGE_INTERSECTION_BIT_KHR)
                                        : VK_SHADER_UNUSED_KHR;
    const uint32_t sRahit = addStage(m_instances ? "trace_forward_inst.rahit.spv" : "trace_forward_tri.rahit.spv",
                                     VK_SHADER_STAGE_ANY_HIT_BIT_KHR);

    // ---- shader groups: [0..3] raygens (split gather, split tail, plain forward, overflow backward), [4] miss,
    //      [5] hit ----
    constexpr uint32_t kRgenGroups = 4, gMiss = 4, gHit = 5;
    std::array<VkRayTracingShaderGroupCreateInfoKHR, 6> groups{};
    for (auto& g : groups) {
        g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.generalShader = g.closestHitShader = g.anyHitShader = g.intersectionShader = VK_SHADER_UNUSED_KHR;
    }
    const uint32_t rgens[kRgenGroups] = {sRgenSplit, sRgenTail, sRgenPlain, sRgenOverflow};
    for (uint32_t i = 0; i < kRgenGroups; ++i) {
        groups[i].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groups[i].generalShader = rgens[i];
    }
    groups[gMiss].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[gMiss].generalShader = sMiss;
    groups[gHit].type = m_instances ? VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR
                                    : VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[gHit].anyHitShader = sRahit;
    groups[gHit].intersectionShader = sRint;   // VK_SHADER_UNUSED_KHR for triangles

    VkRayTracingPipelineCreateInfoKHR ci{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    // Triangle path: promise the driver no procedural (AABB) primitives will be traced,
    // so it can compile a triangle-only traversal loop. OptiX does the same via
    // usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE.
    if (!m_instances) ci.flags |= VK_PIPELINE_CREATE_RAY_TRACING_SKIP_AABBS_BIT_KHR;
    ci.stageCount = (uint32_t)stages.size();
    ci.pStages = stages.data();
    ci.groupCount = (uint32_t)groups.size();
    ci.pGroups = groups.data();
    ci.maxPipelineRayRecursionDepth = 1;   // raygen traces; no nested traceRay
    ci.layout = m_tracePipelineLayout;
    VK_CHECK(m_vkCreateRayTracingPipelinesKHR(m_device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &ci, nullptr, &m_rtPipeline));

    for (auto& s : stages) vkDestroyShaderModule(m_device, s.module, nullptr);

    // ---- shader binding table: [ rgen x4 ] [ miss ] [ hit ], one record each ----
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &rtProps;
    vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2);

    auto alignUp = [](VkDeviceSize v, VkDeviceSize a) { return (v + a - 1) & ~(a - 1); };
    const uint32_t handleSize        = rtProps.shaderGroupHandleSize;
    const VkDeviceSize handleAligned = alignUp(handleSize, rtProps.shaderGroupHandleAlignment);
    const VkDeviceSize baseAlign     = rtProps.shaderGroupBaseAlignment;

    const VkDeviceSize rgenStride = alignUp(handleAligned, baseAlign);   // a raygen region holds one record
    VkStridedDeviceAddressRegionKHR* rgenRegions[kRgenGroups] = {&m_sbtRgenSplit, &m_sbtRgenTail, &m_sbtRgenPlain,
                                                                 &m_sbtRgenOverflow};
    for (auto* r : rgenRegions) { r->stride = rgenStride; r->size = rgenStride; }
    m_sbtMiss.stride = handleAligned;
    m_sbtMiss.size   = alignUp(handleAligned, baseAlign);
    m_sbtHit.stride  = handleAligned;
    m_sbtHit.size    = alignUp(handleAligned, baseAlign);

    std::vector<uint8_t> handles(groups.size() * handleSize);
    VK_CHECK(m_vkGetRayTracingShaderGroupHandlesKHR(m_device, m_rtPipeline, 0, (uint32_t)groups.size(),
                                                    handles.size(), handles.data()));

    const VkDeviceSize rgenRegion = kRgenGroups * rgenStride;
    const VkDeviceSize sbtSize = rgenRegion + m_sbtMiss.size + m_sbtHit.size;
    m_sbtBuffer = createSharedBuffer(sbtSize,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    const VkDeviceAddress base = bufferAddress(m_device, m_vkGetBufferDeviceAddress, m_sbtBuffer.buffer);

    std::vector<uint8_t> host(sbtSize, 0);
    for (uint32_t i = 0; i < kRgenGroups; ++i) {
        rgenRegions[i]->deviceAddress = base + i * rgenStride;
        std::memcpy(host.data() + i * rgenStride, handles.data() + i * handleSize, handleSize);
    }
    m_sbtMiss.deviceAddress = base + rgenRegion;
    m_sbtHit.deviceAddress  = base + rgenRegion + m_sbtMiss.size;
    std::memcpy(host.data() + rgenRegion, handles.data() + gMiss * handleSize, handleSize);
    std::memcpy(host.data() + rgenRegion + m_sbtMiss.size, handles.data() + gHit * handleSize, handleSize);
    CUDA_CHECK(cudaMemcpy(m_sbtBuffer.cudaPtr, host.data(), sbtSize, cudaMemcpyHostToDevice));
}

// ===========================================================================
// Acceleration structures
// ===========================================================================
// primitive_type=instances: the reference's unit primitive (createParticleInstanceAS) -- one procedural AABB
// [-1,1]^3, built once and shared by every gaussian's instance. The intersection shader decides the hits.
void VulkanTracer::buildUnitBoxBlas() {
    if (m_unitBlas) return;   // built once

    const VkAabbPositionsKHR box{-1.f, -1.f, -1.f, 1.f, 1.f, 1.f};
    const VkBufferUsageFlags geomUsage =
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    ensureSharedBuffer(m_unitBoxBuf, sizeof(box), geomUsage);
    CUDA_CHECK(cudaMemcpy(m_unitBoxBuf.cudaPtr, &box, sizeof(box), cudaMemcpyHostToDevice));

    VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geom.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    // not opaque, any-hit at most once per primitive per ray (OptiX REQUIRE_SINGLE_ANYHIT_CALL, as the reference)
    geom.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    geom.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    geom.geometry.aabbs.data.deviceAddress = bufferAddress(m_device, m_vkGetBufferDeviceAddress, m_unitBoxBuf.buffer);
    geom.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);

    VkAccelerationStructureBuildGeometryInfoKHR build{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    build.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.geometryCount = 1;
    build.pGeometries = &geom;

    uint32_t numBoxes = 1;
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    m_vkGetAccelerationStructureBuildSizesKHR(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build, &numBoxes, &sizes);

    ensureSharedBuffer(m_unitBlasBuffer, sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    VkAccelerationStructureCreateInfoKHR asci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    asci.buffer = m_unitBlasBuffer.buffer; asci.size = sizes.accelerationStructureSize; asci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    VK_CHECK(m_vkCreateAccelerationStructureKHR(m_device, &asci, nullptr, &m_unitBlas));

    SharedBuffer scratch = createSharedBuffer(sizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    build.dstAccelerationStructure = m_unitBlas;
    build.scratchData.deviceAddress = bufferAddress(m_device, m_vkGetBufferDeviceAddress, scratch.buffer);
    VkAccelerationStructureBuildRangeInfoKHR range{}; range.primitiveCount = numBoxes;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
    VkCommandBuffer cmd = beginOneShot();
    m_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &pRange);
    endOneShotAndWait(cmd);
    destroyBuffer(scratch);
}

// Host-visible (Vulkan-only) buffer the geometry build reduces the scene bounds into.
void VulkanTracer::createSceneBoundsBuffer() {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = 32;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(m_device, &bci, nullptr, &m_sceneBoundsBuf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(m_device, m_sceneBoundsBuf, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(m_device, &mai, nullptr, &m_sceneBoundsMem));
    VK_CHECK(vkBindBufferMemory(m_device, m_sceneBoundsBuf, m_sceneBoundsMem, 0));
    void* ptr = nullptr;
    VK_CHECK(vkMapMemory(m_device, m_sceneBoundsMem, 0, 32, 0, &ptr));
    m_sceneBoundsPtr = static_cast<uint32_t*>(ptr);
}

// Record the icosahedron BLAS into cmd: refitted in place (same primitives, moved vertices) when updatable, not
// asked to rebuild and built updatable over the same primitive count; else a full build.
void VulkanTracer::recordTriangleBlas(VkCommandBuffer cmd, uint32_t numGaussians, bool rebuild, bool updatable) {
    const uint32_t numVerts = numGaussians * kIcosaVerts;
    const uint32_t numTris  = numGaussians * kIcosaTris;
    VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    // NOT opaque: the any-hit must see triangle *candidates* to gather several hits (it never confirms, mirroring
    // the OptiX gather + ignore). NO_DUPLICATE_ANY_HIT: any-hit at most once per primitive per ray (OptiX
    // REQUIRE_SINGLE_ANYHIT_CALL, as the vk_gaussian_splatting reference).
    geom.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    geom.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geom.geometry.triangles.vertexData.deviceAddress = bufferAddress(m_device, m_vkGetBufferDeviceAddress, m_vertBuf.buffer);
    geom.geometry.triangles.vertexStride = sizeof(float) * 3;
    geom.geometry.triangles.maxVertex = numVerts - 1;
    geom.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    geom.geometry.triangles.indexData.deviceAddress = bufferAddress(m_device, m_vkGetBufferDeviceAddress, m_idxBuf.buffer);

    VkAccelerationStructureBuildGeometryInfoKHR build{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    build.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    if (updatable) build.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.geometryCount = 1;
    build.pGeometries = &geom;
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    m_vkGetAccelerationStructureBuildSizesKHR(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build, &numTris, &sizes);
    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = numTris;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

    if (updatable && !rebuild && m_blas && m_blasUpdatable && m_blasPrimCount == numTris) {
        build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        build.srcAccelerationStructure = m_blas;
        build.dstAccelerationStructure = m_blas;
        ensureSharedBuffer(m_blasScratch, sizes.updateScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    } else {
        if (m_blas) { m_vkDestroyAccelerationStructureKHR(m_device, m_blas, nullptr); m_blas = VK_NULL_HANDLE; }
        ensureSharedBuffer(m_blasBuffer, sizes.accelerationStructureSize,
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, false);
        VkAccelerationStructureCreateInfoKHR asci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        asci.buffer = m_blasBuffer.buffer; asci.size = sizes.accelerationStructureSize;
        asci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        VK_CHECK(m_vkCreateAccelerationStructureKHR(m_device, &asci, nullptr, &m_blas));
        ensureSharedBuffer(m_blasScratch, std::max(sizes.buildScratchSize, updatable ? sizes.updateScratchSize : 0),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
        build.dstAccelerationStructure = m_blas;
        m_blasPrimCount = numTris;
        m_blasUpdatable = updatable;
    }
    build.scratchData.deviceAddress = bufferAddress(m_device, m_vkGetBufferDeviceAddress, m_blasScratch.buffer);
    m_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &pRange);
}

// Record a TLAS build over m_instanceBuffer into cmd. The TLAS object is kept while its size fits, so the trace
// descriptor sets (which bind it) stay valid across steps. updatable (instances): built with ALLOW_UPDATE and updated
// in place (MODE_UPDATE) when the caller does not ask for a rebuild and the instance count is unchanged, as the
// icosahedron BLAS is refitted between rebuilds.
void VulkanTracer::recordTlas(VkCommandBuffer cmd, uint32_t numInstances, bool updatable, bool rebuild) {
    VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geom.geometry.instances.data.deviceAddress = bufferAddress(m_device, m_vkGetBufferDeviceAddress, m_instanceBuffer.buffer);
    VkAccelerationStructureBuildGeometryInfoKHR build{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    if (updatable) build.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.geometryCount = 1;
    build.pGeometries = &geom;
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    m_vkGetAccelerationStructureBuildSizesKHR(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build, &numInstances, &sizes);
    const bool update = updatable && !rebuild && m_tlas && m_tlasUpdatable && m_tlasInstCount == numInstances
                     && m_tlasCapacity >= sizes.accelerationStructureSize;
    if (update) {
        build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        build.srcAccelerationStructure = m_tlas;
        build.dstAccelerationStructure = m_tlas;
        ensureSharedBuffer(m_tlasScratch, sizes.updateScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
        build.scratchData.deviceAddress = bufferAddress(m_device, m_vkGetBufferDeviceAddress, m_tlasScratch.buffer);
        VkAccelerationStructureBuildRangeInfoKHR urange{};
        urange.primitiveCount = numInstances;
        const VkAccelerationStructureBuildRangeInfoKHR* pURange = &urange;
        m_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &pURange);
        return;
    }
    if (!m_tlas || m_tlasCapacity < sizes.accelerationStructureSize) {
        if (m_tlas) { m_vkDestroyAccelerationStructureKHR(m_device, m_tlas, nullptr); m_tlas = VK_NULL_HANDLE; }
        ensureSharedBuffer(m_tlasBuffer, sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, false);
        VkAccelerationStructureCreateInfoKHR asci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        asci.buffer = m_tlasBuffer.buffer; asci.size = sizes.accelerationStructureSize;
        asci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        VK_CHECK(m_vkCreateAccelerationStructureKHR(m_device, &asci, nullptr, &m_tlas));
        m_tlasCapacity = sizes.accelerationStructureSize;
        ++m_resourceGen;   // the TLAS handle changed; cached descriptor sets are stale
    }
    ensureSharedBuffer(m_tlasScratch, std::max(sizes.buildScratchSize, updatable ? sizes.updateScratchSize : 0),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    build.dstAccelerationStructure = m_tlas;
    build.scratchData.deviceAddress = bufferAddress(m_device, m_vkGetBufferDeviceAddress, m_tlasScratch.buffer);
    m_tlasInstCount = numInstances;
    m_tlasUpdatable = updatable;
    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = numInstances;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
    m_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &pRange);
}

// ===========================================================================
// build_bvh
// ===========================================================================
// One command buffer: the geometry build (icosahedron vertices/indices, or the instances) reduces the scene bounds,
// then the BLAS (icosahedron) and the TLAS. It is submitted without a host wait: the next trace, traceBwd or buildBVH
// waits for it just before it needs the result (finishPendingBvh), so the build overlaps the host and CUDA work in
// between (the end of the training step and the next step's input copies).
void VulkanTracer::buildBVH(torch::Tensor mogPos,
                            torch::Tensor mogRot,
                            torch::Tensor mogScl,
                            torch::Tensor mogDns,
                            unsigned int rebuild,
                            bool allow_update,
                            int64_t /*iteration*/) {
    finishPendingBvh();   // its buffers are about to be rewritten
    const uint32_t gNum = (uint32_t)mogPos.size(0);
    m_numGaussians = gNum;
    cudaStream_t stream = currentStream();

    const VkBufferUsageFlags geomUsage =
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    // shared input buffers for the geometry-build compute
    ensureSharedBuffer(m_posBuf, sizeof(float) * 3 * gNum, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureSharedBuffer(m_rotBuf, sizeof(float) * 4 * gNum, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureSharedBuffer(m_sclBuf, sizeof(float) * 3 * gNum, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureSharedBuffer(m_dnsBuf, sizeof(float) * 1 * gNum, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureSharedBuffer(m_aabbParamsUbo, sizeof(AabbParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    if (m_instances) {
        buildUnitBoxBlas();   // built once
        ensureSharedBuffer(m_instanceBuffer, sizeof(VkAccelerationStructureInstanceKHR) * gNum, kInstanceBufferUsage);
    } else {
        ensureSharedBuffer(m_vertBuf, sizeof(float) * 3 * kIcosaVerts * gNum, geomUsage, false);
        ensureSharedBuffer(m_idxBuf, sizeof(uint32_t) * 3 * kIcosaTris * gNum, geomUsage, false);
    }

    CUDA_CHECK(cudaMemcpyAsync(m_posBuf.cudaPtr, mogPos.data_ptr(), sizeof(float)*3*gNum, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m_rotBuf.cudaPtr, mogRot.data_ptr(), sizeof(float)*4*gNum, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m_sclBuf.cudaPtr, mogScl.data_ptr(), sizeof(float)*3*gNum, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m_dnsBuf.cudaPtr, mogDns.data_ptr(), sizeof(float)*1*gNum, cudaMemcpyDeviceToDevice, stream));

    // build_icosahedron rewrites the triangle indices only when they can have changed
    const bool writeIdx = !m_instances && (gNum != m_geomIndexCount || m_resourceGen != m_geomIndexGen);
    if (!m_sceneBoundsBuf) createSceneBoundsBuffer();
    AabbParams ap{gNum, m_kernelDegree, m_densityClamping ? 1 : 0, m_minResponse, 0u, 0u, writeIdx ? 1u : 0u};
    if (m_instances) {
        VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
        ai.accelerationStructure = m_unitBlas;
        VkDeviceAddress a = m_vkGetAccelerationStructureDeviceAddressKHR(m_device, &ai);
        ap.blasRefLo = (uint32_t)(a & 0xFFFFFFFFull);
        ap.blasRefHi = (uint32_t)(a >> 32);
    }
    CUDA_CHECK(cudaMemcpyAsync(m_aabbParamsUbo.cudaPtr, &ap, sizeof(ap), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // descriptor set for the geometry-build compute (freed by finishPendingBvh)
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = m_descPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &m_buildSetLayout;
    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(m_device, &dsai, &ds));

    auto bufInfo = [this](const SharedBuffer& b) { return checkedBufferInfo(b); };
    VkDescriptorBufferInfo bi0 = bufInfo(m_aabbParamsUbo);
    VkDescriptorBufferInfo bi1 = bufInfo(m_posBuf), bi2 = bufInfo(m_rotBuf),
                           bi3 = bufInfo(m_sclBuf), bi4 = bufInfo(m_dnsBuf);
    VkDescriptorBufferInfo bi5 = bufInfo(m_instances ? m_instanceBuffer : m_vertBuf);
    VkDescriptorBufferInfo bi6 = m_instances ? VkDescriptorBufferInfo{} : bufInfo(m_idxBuf);
    VkDescriptorBufferInfo bi7{m_sceneBoundsBuf, 0, 32};
    std::vector<VkWriteDescriptorSet> w;
    auto pushW = [&](uint32_t binding, VkDescriptorType type, VkDescriptorBufferInfo* info) {
        VkWriteDescriptorSet ww{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        ww.dstSet = ds; ww.dstBinding = binding; ww.descriptorCount = 1;
        ww.descriptorType = type; ww.pBufferInfo = info; w.push_back(ww);
    };
    pushW(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &bi0);
    pushW(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &bi1);
    pushW(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &bi2);
    pushW(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &bi3);
    pushW(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &bi4);
    pushW(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &bi5);
    if (!m_instances) pushW(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &bi6);
    pushW(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &bi7);
    vkUpdateDescriptorSets(m_device, (uint32_t)w.size(), w.data(), 0, nullptr);

    VkCommandBuffer cmd = beginOneShot();
    vkCmdFillBuffer(cmd, m_sceneBoundsBuf, 0, 12, 0xFFFFFFFFu);   // min: largest ordered value
    vkCmdFillBuffer(cmd, m_sceneBoundsBuf, 12, 12, 0u);           // max: smallest ordered value
    VkMemoryBarrier fb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    fb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    fb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &fb, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_buildPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_buildPipelineLayout, 0, 1, &ds, 0, nullptr);
    vkCmdDispatch(cmd, (gNum + 255) / 256, 1, 1);
    // make the reduced bounds visible to the host read after the wait
    VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    hb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &hb, 0, nullptr, 0, nullptr);
    VkMemoryBarrier gb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};   // vertices/indices/instances -> AS build input
    gb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    gb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &gb, 0, nullptr, 0, nullptr);
    if (m_instances) {
        // build_instances wrote one instance per gaussian over the unit BLAS (barrier above): TLAS only
        recordTlas(cmd, gNum, allow_update, rebuild != 0);
    } else {
        recordTriangleBlas(cmd, gNum, rebuild != 0, allow_update);
        // single instance referencing the (possibly re-created) BLAS, written on the queue
        VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
        ai.accelerationStructure = m_blas;
        VkAccelerationStructureInstanceKHR inst{};
        inst.transform.matrix[0][0] = inst.transform.matrix[1][1] = inst.transform.matrix[2][2] = 1.0f;
        inst.mask = 0xFF;
        // No facing flag: with the default facing convention + backface culling in the
        // raygen, the ray reports each icosahedron's ENTRY face (matches OptiX and the
        // vk_gaussian_splatting reference, which sets no facing flag). Setting
        // TRIANGLE_FRONT_COUNTERCLOCKWISE (= TRIANGLE_FLIP_FACING) inverted the facing,
        // so culling kept the EXIT faces: hits were sorted by exit distance, integrating
        // large gaussians too late (+40% accepted hits/ray vs OptiX on the same ckpt).
        inst.flags = 0;
        inst.accelerationStructureReference = m_vkGetAccelerationStructureDeviceAddressKHR(m_device, &ai);
        ensureSharedBuffer(m_instanceBuffer, sizeof(inst), kInstanceBufferUsage);
        vkCmdUpdateBuffer(cmd, m_instanceBuffer.buffer, 0, sizeof(inst), &inst);
        VkMemoryBarrier ib{VK_STRUCTURE_TYPE_MEMORY_BARRIER};   // BLAS + instance -> TLAS build
        ib.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_TRANSFER_WRITE_BIT;
        ib.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &ib, 0, nullptr, 0, nullptr);
        recordTlas(cmd, 1, false, true);
    }
    VK_CHECK(vkEndCommandBuffer(cmd));
    if (!m_bvhFence) {
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(m_device, &fi, nullptr, &m_bvhFence));
    }
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(m_queue, 1, &si, m_bvhFence));
    // Completed by finishPendingBvh, which reads the scene bounds. Nothing the build reads or writes is
    // touched before then (the next writer is the next build).
    m_bvhPendingCmd = cmd;
    m_bvhPendingDs = ds;

    if (writeIdx) m_geomIndexCount = gNum;
    m_geomIndexGen = m_resourceGen;   // after every buffer change this build made

    // The gaussians just changed: the next trace/traceBwd must re-upload density+SH.
    m_gaussiansDirty = true;
}

void VulkanTracer::readSceneBounds() {
    auto decode = [](uint32_t u) {
        const uint32_t b = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
        float f; std::memcpy(&f, &b, sizeof(f)); return f;
    };
    for (int i = 0; i < 3; ++i) {
        m_sceneAabbMin[i] = decode(m_sceneBoundsPtr[i]);
        m_sceneAabbMax[i] = decode(m_sceneBoundsPtr[3 + i]);
    }
}

void VulkanTracer::finishPendingBvh() {
    if (!m_bvhPendingCmd) return;
    VK_CHECK(vkWaitForFences(m_device, 1, &m_bvhFence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(m_device, 1, &m_bvhFence));
    vkFreeCommandBuffers(m_device, m_cmdPool, 1, &m_bvhPendingCmd);
    vkFreeDescriptorSets(m_device, m_descPool, 1, &m_bvhPendingDs);
    m_bvhPendingCmd = VK_NULL_HANDLE;
    m_bvhPendingDs = VK_NULL_HANDLE;
    readSceneBounds();
}

// ===========================================================================
// helpers to fill the params UBO
// ===========================================================================
static void fillParams(ShaderParams& p, torch::Tensor rayToWorld,
                       const float aabbMin[3], const float aabbMax[3],
                       uint32_t W, uint32_t H, uint32_t B,
                       float minTransmittance, float minResponse, float minAlpha, float maxAlpha,
                       float maxParticleSqDist, int kernelDegree, int sphDegree) {
    torch::Tensor r = rayToWorld.cpu().contiguous();
    const float* rd = r.data_ptr<float>();   // 3x4 row-major
    for (int i = 0; i < 4; ++i) { p.rayToWorld0[i] = rd[0 + i]; p.rayToWorld1[i] = rd[4 + i]; p.rayToWorld2[i] = rd[8 + i]; }
    for (int i = 0; i < 3; ++i) { p.aabbMin[i] = aabbMin[i]; p.aabbMax[i] = aabbMax[i]; }
    p.aabbMin[3] = p.aabbMax[3] = 0.f;
    p.frameW = W; p.frameH = H; p.frameB = B;
    p.minTransmittance = minTransmittance; p.minResponse = minResponse;
    p.minAlpha = minAlpha; p.maxAlpha = maxAlpha; p.maxParticleSqDist = maxParticleSqDist;
    p.kernelDegree = kernelDegree; p.sphDegree = sphDegree;
}

// Writes the SH coefficients as fp16 into m_radianceBuf on the current stream: converted in place
// through a tensor view of the shared buffer when the layout matches (no temporary, no copy),
// else through a temporary (kept alive in tmp until the caller's stream sync).
void VulkanTracer::uploadSphHalf(const torch::Tensor& particleRadiance, uint32_t gNum, torch::Tensor& tmp) {
    const int64_t cols = int64_t(m_sphNumCoeffs) * 3;
    if (particleRadiance.is_contiguous() && particleRadiance.numel() == int64_t(gNum) * cols) {
        torch::from_blob(m_radianceBuf.cudaPtr, {int64_t(gNum), cols},
                         torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA))
            .copy_(particleRadiance.view({int64_t(gNum), cols}));
        return;
    }
    tmp = particleRadiance.to(torch::kHalf).contiguous();
    CUDA_CHECK(cudaMemcpyAsync(m_radianceBuf.cudaPtr, tmp.data_ptr(), sizeof(uint16_t) * cols * gNum,
                               cudaMemcpyDeviceToDevice, currentStream()));
}

// ===========================================================================
// trace (forward)
// ===========================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
VulkanTracer::trace(uint32_t frameNumber,
                    torch::Tensor rayToWorld,
                    torch::Tensor rayOri,
                    torch::Tensor rayDir,
                    torch::Tensor particleDensity,
                    torch::Tensor particleRadiance,
                    uint32_t renderOpts,
                    int sphDegree,
                    float minTransmittance) {
    const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const long B = rayOri.size(0), H = rayOri.size(1), W = rayOri.size(2);
    const long N = B * H * W;   // rays
    const uint32_t gNum = (uint32_t)particleDensity.size(0);
    cudaStream_t stream = currentStream();

    // renderOpts bit 0 (tracer.py RenderOpts.NO_RECORD, set when autograd is off): no backward follows this
    // forward, so it runs the single-launch raygen once over the frame instead of the split rounds and records
    // nothing. The split exists to record cheaply; without recording the single launch is the faster one.
    const bool noRecord = (renderOpts & 1u) != 0;
    if (!noRecord && m_enableNormals)
        throw std::runtime_error("render.enable_normals is not supported for training (the split forward does not "
                                 "integrate normals); it is available for rendering only");

    // force contiguity before fixed-size device copies (see the traceBwd note)
    particleDensity  = particleDensity.contiguous();
    particleRadiance = particleRadiance.contiguous();
    rayOri = rayOri.contiguous();
    rayDir = rayDir.contiguous();

    torch::Tensor rayRad = torch::empty({B, H, W, 3}, opts);
    torch::Tensor rayDns = torch::empty({B, H, W, 1}, opts);
    torch::Tensor rayHit = torch::empty({B, H, W, 2}, opts);
    // normals / hit counts: downloaded when enabled; zeros otherwise (no shader writes them)
    torch::Tensor rayNrm = m_enableNormals ? torch::empty({B, H, W, 3}, opts) : torch::zeros({B, H, W, 3}, opts);
    torch::Tensor rayHc  = m_enableHitCounts ? torch::empty({B, H, W, 1}, opts) : torch::zeros({B, H, W, 1}, opts);
    torch::Tensor vis    = torch::zeros({gNum, 1}, opts);

    // buffers shared with PyTorch: inputs (gaussians, rays) and outputs (the out* buffers)
    ensureSharedBuffer(m_densityBuf,  sizeof(float) * 12 * gNum, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureSharedBuffer(m_radianceBuf, sizeof(uint16_t) * m_sphNumCoeffs * 3 * gNum, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);  // SH stored fp16
    ensureSharedBuffer(m_rayOriBuf, sizeof(float) * 3 * N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureSharedBuffer(m_rayDirBuf, sizeof(float) * 3 * N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const VkBufferUsageFlags outUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    ensureSharedBuffer(m_outRad, sizeof(float) * 3 * N, outUsage);   // accumulated radiance
    ensureSharedBuffer(m_outDns, sizeof(float) * 1 * N, outUsage);   // accumulated density (1 - transmittance)
    ensureSharedBuffer(m_outHit, sizeof(float) * 2 * N, outUsage);   // hit distance, last processed hit t
    ensureSharedBuffer(m_outNrm, sizeof(float) * 3 * N, outUsage);   // written only when normals are enabled
    ensureSharedBuffer(m_outHc,  sizeof(float) * 1 * N, outUsage);   // written only when hit counts are enabled
    ensureSharedBuffer(m_outVis, sizeof(int)   * 1 * gNum, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);   // per gaussian
    // The RT pipeline links the overflow backward raygen too, so the bindings it uses statically
    // (12 gradHit, 14 densityGrad, 15 radianceGrad) must be valid in the forward set
    // (VUID-vkCmdTraceRaysKHR-None-08114). Same sizes as traceBwd: no extra allocation.
    ensureSharedBuffer(m_gradHit, sizeof(float) * 1 * N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureSharedBuffer(m_densityGrad,  sizeof(float) * 12 * gNum, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureSharedBuffer(m_radianceGrad, sizeof(float) * m_sphNumCoeffs * 3 * gNum, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureSharedBuffer(m_paramsUbo, sizeof(ShaderParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    prepareScratch((uint32_t)W, (uint32_t)H, (uint32_t)B);
    // The hit archive, written by the split forward. After releaseTrainingBuffers a forward that records nothing
    // binds a placeholder archive for one 1x1 frame: its descriptors must stay valid but it is never read or
    // written. A recording forward frees the placeholder first, so the full archive is allocated exact-fit as on
    // the first step.
    if (m_archiveReleased && !noRecord) { destroyBuffer(m_bwdHitIds); m_archiveReleased = false; }
    if (m_archiveReleased) prepareArchive(1, 1, 1);
    else prepareArchive((uint32_t)W, (uint32_t)H, (uint32_t)B);
    if (!noRecord) {
        // split-forward state: 8 float planes; NaN bits (0xFF memset) mark uninitialized state -- the shaders
        // lazy-init on first touch. The window handoff lives in the archive (the gather writes it at round*k,
        // the blend masks rejects), and the round index arrives by push constant.
        const uint32_t stateRays = (uint32_t)N;
        ensureSharedBuffer(m_splitStateF, sizeof(float) * 8 * stateRays, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        CUDA_CHECK(cudaMemsetAsync(m_splitStateF.cudaPtr, 0xFF, sizeof(float) * 8 * stateRays, stream));
        m_splitStateRays = stateRays;
    }

    // upload inputs (device->device). The gaussians (density + SH) only change when
    // buildBVH runs, so upload them once per build and reuse across traces (static
    // inference: skip on frames 2..N; training: the backward reuses the forward's
    // upload). The fp16 SH convert is the biggest single item, so it is skipped too.
    torch::Tensor rad16;   // function scope: stays alive until the stream sync below
    if (m_gaussiansDirty) {
        CUDA_CHECK(cudaMemcpyAsync(m_densityBuf.cudaPtr,  particleDensity.data_ptr(),  sizeof(float)*12*gNum, cudaMemcpyDeviceToDevice, stream));
        uploadSphHalf(particleRadiance, gNum, rad16);   // SH fp32 -> fp16 (half the read bandwidth in-shader)
        m_gaussiansDirty = false;
    }
    CUDA_CHECK(cudaMemcpyAsync(m_rayOriBuf.cudaPtr, rayOri.data_ptr(), sizeof(float)*3*N, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m_rayDirBuf.cudaPtr, rayDir.data_ptr(), sizeof(float)*3*N, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(m_outVis.cudaPtr, 0, sizeof(int)*gNum, stream));

    // as late as possible: the copies above run on the GPU alongside the pending build
    finishPendingBvh();
    ShaderParams p{};
    fillParams(p, rayToWorld, m_sceneAabbMin, m_sceneAabbMax, (uint32_t)W, (uint32_t)H, (uint32_t)B,
               minTransmittance, m_minResponse, kMinAlpha, kMaxAlpha, kMaxParticleSqDist, m_kernelDegree, sphDegree);
    p.archivePageShift = m_archiveLayout.pageShift();
    p.archiveMaxSlots = m_archiveLayout.slots;
    p.archivePlaneRays = m_archiveLayout.rays;
    p.statePlaneRays = m_splitStateRays;
    CUDA_CHECK(cudaMemcpyAsync(m_paramsUbo.cudaPtr, &p, sizeof(p), cudaMemcpyHostToDevice, stream));
    // the tail raygen counts the rays alive after the rounds into the word after the per-ray counts
    if (!noRecord)
        CUDA_CHECK(cudaMemsetAsync(static_cast<uint32_t*>(m_bwdHitCnt.cudaPtr) + N, 0, sizeof(uint32_t), stream));

    // wait for the copies, so Vulkan reads the uploaded memory
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // descriptor set: allocated once, rewritten only when a bound resource was
    // recreated (buffer growth or TLAS recreation), tracked by m_resourceGen. Safe to
    // rewrite here: every prior submission has completed (fence/queue-idle waits).
    if (!m_fwdDs) {
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = m_descPool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &m_traceSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_device, &dsai, &m_fwdDs));
    }
    if (m_fwdDsGen != m_resourceGen) {
        auto bufInfo = [this](const SharedBuffer& b) { return checkedBufferInfo(b); };
        VkDescriptorBufferInfo b0 = bufInfo(m_paramsUbo);
        VkDescriptorBufferInfo b2 = bufInfo(m_densityBuf), b3 = bufInfo(m_radianceBuf),
                               b4 = bufInfo(m_rayOriBuf), b5 = bufInfo(m_rayDirBuf),
                               b6 = bufInfo(m_outRad), b7 = bufInfo(m_outDns), b8 = bufInfo(m_outHit),
                               b9 = bufInfo(m_outNrm), b10 = bufInfo(m_outHc), b11 = bufInfo(m_outVis),
                               b12 = bufInfo(m_gradHit), b14 = bufInfo(m_densityGrad), b15 = bufInfo(m_radianceGrad),
                               b16 = bufInfo(m_hitScratch), b18 = bufInfo(m_bwdHitCnt);
        std::vector<VkDescriptorBufferInfo> b17 = archiveBufferInfos();
        VkDescriptorBufferInfo b20{};

        VkWriteDescriptorSetAccelerationStructureKHR asWrite{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
        asWrite.accelerationStructureCount = 1;
        asWrite.pAccelerationStructures = &m_tlas;

        std::vector<VkWriteDescriptorSet> writes;
        auto pushBuf = [&](uint32_t binding, VkDescriptorType t, VkDescriptorBufferInfo* info, uint32_t count = 1) {
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = m_fwdDs; w.dstBinding = binding; w.descriptorCount = count; w.descriptorType = t; w.pBufferInfo = info;
            writes.push_back(w);
        };
        pushBuf(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &b0);
        { VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
          w.pNext = &asWrite; w.dstSet = m_fwdDs; w.dstBinding = 1; w.descriptorCount = 1;
          w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; writes.push_back(w); }
        pushBuf(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b2);
        pushBuf(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b3);
        pushBuf(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b4);
        pushBuf(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b5);
        pushBuf(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b6);
        pushBuf(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b7);
        pushBuf(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b8);
        pushBuf(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b9);
        pushBuf(10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b10);
        pushBuf(11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b11);
        pushBuf(12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b12);   // overflow backward raygen (same pipeline)
        pushBuf(14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b14);
        pushBuf(15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b15);
        pushBuf(16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b16);
        pushBuf(17, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, b17.data(), (uint32_t)b17.size());
        pushBuf(18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b18);
        if (m_splitStateF.buffer) { b20 = bufInfo(m_splitStateF); pushBuf(20, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b20); }
        vkUpdateDescriptorSets(m_device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
        m_fwdDsGen = m_resourceGen;
    }

    VkCommandBuffer cmd = beginTraceCmd();
    if (!noRecord) {
        // ---- split forward: gather -> blend rounds, then the tail. RT and compute are separate bind points,
        // so each pipeline + the shared descriptor set binds once; the loop is trace / barrier / dispatch /
        // barrier. Dead rays cost one state read per extra round.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_tracePipelineLayout, 0, 1, &m_fwdDs, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fwdBlendPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tracePipelineLayout, 0, 1, &m_fwdDs, 0, nullptr);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        // Invariant: the archive must hold every slot the rounds can write (rounds x k <= kArchiveSlots), or the
        // blend would composite hits the archive dropped and the backward would miss their gradients.
        if (!m_adaptiveRounds) m_adaptiveRounds = kSplitRoundsStart;
        const uint32_t rounds = std::min(m_adaptiveRounds, kArchiveSlots / kHitsPerWindow);
        for (uint32_t r = 0; r < rounds; ++r) {
            vkCmdPushConstants(cmd, m_tracePipelineLayout, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(r), &r);
            m_vkCmdTraceRaysKHR(cmd, &m_sbtRgenSplit, &m_sbtMiss, &m_sbtHit, &m_sbtCall,
                                (uint32_t)W, (uint32_t)H, (uint32_t)B);
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            vkCmdDispatch(cmd, ((uint32_t)W + 7) / 8, ((uint32_t)H + 7) / 8, (uint32_t)B);
            if (r + 1 < rounds)
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &mb, 0, nullptr, 0, nullptr);
        }
        // One full-frame tail launch finishes every ray still alive at the round cap (all others return after
        // one state read). The RT pipeline and descriptor set are still bound from the loop above.
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &mb, 0, nullptr, 0, nullptr);
        m_vkCmdTraceRaysKHR(cmd, &m_sbtRgenTail, &m_sbtMiss, &m_sbtHit, &m_sbtCall,
                            (uint32_t)W, (uint32_t)H, (uint32_t)B);
    } else {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_tracePipelineLayout, 0, 1, &m_fwdDs, 0, nullptr);
        // the pipeline links the split gather raygen, which reads the push constant
        const uint32_t round0 = 0u;
        vkCmdPushConstants(cmd, m_tracePipelineLayout, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(round0), &round0);
        m_vkCmdTraceRaysKHR(cmd, &m_sbtRgenPlain, &m_sbtMiss, &m_sbtHit, &m_sbtCall,
                            (uint32_t)W, (uint32_t)H, (uint32_t)B);
    }
    submitTraceAndWait();

    // download outputs
    CUDA_CHECK(cudaMemcpyAsync(rayRad.data_ptr(), m_outRad.cudaPtr, sizeof(float)*3*N, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(rayDns.data_ptr(), m_outDns.cudaPtr, sizeof(float)*1*N, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(rayHit.data_ptr(), m_outHit.cudaPtr, sizeof(float)*2*N, cudaMemcpyDeviceToDevice, stream));
    if (m_enableNormals)
        CUDA_CHECK(cudaMemcpyAsync(rayNrm.data_ptr(), m_outNrm.cudaPtr, sizeof(float)*3*N, cudaMemcpyDeviceToDevice, stream));
    if (m_enableHitCounts)
        CUDA_CHECK(cudaMemcpyAsync(rayHc.data_ptr(),  m_outHc.cudaPtr,  sizeof(float)*1*N, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(vis.data_ptr(),    m_outVis.cudaPtr, sizeof(int)*gNum,  cudaMemcpyDeviceToDevice, stream));
    uint32_t tailRays = 0;
    if (!noRecord)
        CUDA_CHECK(cudaMemcpyAsync(&tailRays, static_cast<uint32_t*>(m_bwdHitCnt.cudaPtr) + N, sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!noRecord) {
        // the next frame's round count: up while more than 1% of the rays reached the tail, down below 0.25%
        const uint32_t maxRounds = std::max(1u, kArchiveSlots / kHitsPerWindow);
        const uint32_t minRounds = std::min(std::max(1u, kSplitRoundsMin), maxRounds);
        const double tailShare = double(tailRays) / double(N);
        if (tailShare > 0.01 && m_adaptiveRounds < maxRounds) ++m_adaptiveRounds;
        else if (tailShare < 0.0025 && m_adaptiveRounds > minRounds) --m_adaptiveRounds;
        m_adaptiveRounds = std::min(std::max(m_adaptiveRounds, minRounds), maxRounds);
    }

    m_lastFwdIo.valid = true;
    m_lastFwdIo.frame = frameNumber;
    m_lastFwdIo.rays = N;
    m_lastFwdIo.resourceGen = m_resourceGen;
    const void* fwdIo[5] = {rayOri.data_ptr(), rayDir.data_ptr(), rayRad.data_ptr(), rayDns.data_ptr(), rayHit.data_ptr()};
    std::copy(fwdIo, fwdIo + 5, m_lastFwdIo.ptrs);
    return {rayRad, rayDns, rayHit, rayNrm, rayHc, vis};
}

// ===========================================================================
// traceBwd (backward)
// ===========================================================================
std::tuple<torch::Tensor, torch::Tensor>
VulkanTracer::traceBwd(uint32_t frameNumber,
                       torch::Tensor rayToWorld,
                       torch::Tensor rayOri,
                       torch::Tensor rayDir,
                       torch::Tensor rayRad,
                       torch::Tensor rayDns,
                       torch::Tensor rayHit,
                       torch::Tensor /*rayNrm*/,
                       torch::Tensor particleDensity,
                       torch::Tensor particleRadiance,
                       torch::Tensor rayRadGrd,
                       torch::Tensor rayDnsGrd,
                       torch::Tensor rayHitGrd,
                       torch::Tensor /*rayNrmGrd*/,
                       uint32_t /*renderOpts*/,
                       int sphDegree,
                       float minTransmittance) {
    const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const long B = rayOri.size(0), H = rayOri.size(1), W = rayOri.size(2);
    const long N = B * H * W;
    const uint32_t gNum = (uint32_t)particleDensity.size(0);
    cudaStream_t stream = currentStream();

    // Autograd hands back gradient tensors that may be non-contiguous (views /
    // strided). We copy fixed-size blocks from data_ptr(), so force contiguity
    // first or the device-to-device copy reads past the storage (cudaErrorInvalidValue).
    particleDensity  = particleDensity.contiguous();
    particleRadiance = particleRadiance.contiguous();
    rayOri = rayOri.contiguous();
    rayDir = rayDir.contiguous();
    rayRad = rayRad.contiguous();
    rayDns = rayDns.contiguous();
    rayHit = rayHit.contiguous();
    rayRadGrd = rayRadGrd.contiguous();
    rayDnsGrd = rayDnsGrd.contiguous();
    rayHitGrd = rayHitGrd.contiguous();

    // fully overwritten by the gradient downloads at the end
    torch::Tensor densityGrad  = torch::empty({particleDensity.size(0), particleDensity.size(1)}, opts);
    torch::Tensor radianceGrad = torch::empty({particleRadiance.size(0), particleRadiance.size(1)}, opts);

    ensureSharedBuffer(m_densityBuf,  sizeof(float) * 12 * gNum, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureSharedBuffer(m_radianceBuf, sizeof(uint16_t) * m_sphNumCoeffs * 3 * gNum, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);  // SH stored fp16
    ensureSharedBuffer(m_rayOriBuf, sizeof(float) * 3 * N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureSharedBuffer(m_rayDirBuf, sizeof(float) * 3 * N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureSharedBuffer(m_outRad, sizeof(float) * 3 * N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);   // fwd radiance
    ensureSharedBuffer(m_outDns, sizeof(float) * 1 * N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);   // fwd density
    ensureSharedBuffer(m_outHit, sizeof(float) * 2 * N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);   // fwd hit (2)
    ensureSharedBuffer(m_gradRad, sizeof(float) * 3 * N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureSharedBuffer(m_gradDns, sizeof(float) * 1 * N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureSharedBuffer(m_gradHit, sizeof(float) * 1 * N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureSharedBuffer(m_densityGrad,  sizeof(float) * 12 * gNum, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureSharedBuffer(m_radianceGrad, sizeof(float) * m_sphNumCoeffs * 3 * gNum, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureSharedBuffer(m_paramsUbo, sizeof(ShaderParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    prepareScratch((uint32_t)W, (uint32_t)H, (uint32_t)B);
    prepareArchive((uint32_t)W, (uint32_t)H, (uint32_t)B);

    // Reuse the forward's density+SH upload when the gaussians are unchanged (same
    // training iter: buildBVH -> trace() uploaded and cleared the flag -> backward skips).
    torch::Tensor rad16;   // function scope: stays alive until the stream sync below
    if (m_gaussiansDirty) {
        CUDA_CHECK(cudaMemcpyAsync(m_densityBuf.cudaPtr,  particleDensity.data_ptr(),  sizeof(float)*12*gNum, cudaMemcpyDeviceToDevice, stream));
        uploadSphHalf(particleRadiance, gNum, rad16);   // SH fp32 -> fp16 (half the read bandwidth in-shader)
        m_gaussiansDirty = false;
    }
    // This frame's forward left its ray inputs and outputs in these buffers, and autograd
    // passes back the very tensors it saved: skip the 12 floats/ray of re-upload.
    const void* fwdIo[5] = {rayOri.data_ptr(), rayDir.data_ptr(), rayRad.data_ptr(), rayDns.data_ptr(), rayHit.data_ptr()};
    const bool fwdIoResident = m_lastFwdIo.valid && m_lastFwdIo.frame == frameNumber && m_lastFwdIo.rays == N &&
                               m_lastFwdIo.resourceGen == m_resourceGen &&
                               std::equal(fwdIo, fwdIo + 5, m_lastFwdIo.ptrs);
    if (!fwdIoResident) {
        CUDA_CHECK(cudaMemcpyAsync(m_rayOriBuf.cudaPtr, rayOri.data_ptr(), sizeof(float)*3*N, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(m_rayDirBuf.cudaPtr, rayDir.data_ptr(), sizeof(float)*3*N, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(m_outRad.cudaPtr, rayRad.data_ptr(), sizeof(float)*3*N, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(m_outDns.cudaPtr, rayDns.data_ptr(), sizeof(float)*1*N, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(m_outHit.cudaPtr, rayHit.data_ptr(), sizeof(float)*2*N, cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(m_gradRad.cudaPtr, rayRadGrd.data_ptr(), sizeof(float)*3*N, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m_gradDns.cudaPtr, rayDnsGrd.data_ptr(), sizeof(float)*1*N, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(m_gradHit.cudaPtr, rayHitGrd.data_ptr(), sizeof(float)*1*N, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(m_densityGrad.cudaPtr, 0, sizeof(float)*12*gNum, stream));
    CUDA_CHECK(cudaMemsetAsync(m_radianceGrad.cudaPtr, 0, sizeof(float)*m_sphNumCoeffs*3*gNum, stream));

    finishPendingBvh();   // normally a no-op: the forward completed it
    ShaderParams p{};
    fillParams(p, rayToWorld, m_sceneAabbMin, m_sceneAabbMax, (uint32_t)W, (uint32_t)H, (uint32_t)B,
               minTransmittance, m_minResponse, kMinAlpha, kMaxAlpha, kMaxParticleSqDist, m_kernelDegree, sphDegree);
    p.archivePageShift = m_archiveLayout.pageShift();
    p.archiveMaxSlots = m_archiveLayout.slots;
    p.archivePlaneRays = m_archiveLayout.rays;
    p.statePlaneRays = m_splitStateRays;
    CUDA_CHECK(cudaMemcpyAsync(m_paramsUbo.cudaPtr, &p, sizeof(p), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // descriptor set: persistent, rewritten only when a bound resource changed
    // (same generation scheme as the forward set; see trace()).
    if (!m_bwdDs) {
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = m_descPool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &m_traceSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_device, &dsai, &m_bwdDs));
    }
    if (m_bwdDsGen != m_resourceGen) {
        auto bufInfo = [this](const SharedBuffer& b) { return checkedBufferInfo(b); };
        VkDescriptorBufferInfo b0 = bufInfo(m_paramsUbo);
        VkDescriptorBufferInfo b2 = bufInfo(m_densityBuf), b3 = bufInfo(m_radianceBuf),
                               b4 = bufInfo(m_rayOriBuf), b5 = bufInfo(m_rayDirBuf),
                               b6 = bufInfo(m_outRad), b7 = bufInfo(m_outDns), b8 = bufInfo(m_outHit),
                               b10 = bufInfo(m_gradRad), b11 = bufInfo(m_gradDns), b12 = bufInfo(m_gradHit),
                               b14 = bufInfo(m_densityGrad), b15 = bufInfo(m_radianceGrad),
                               b16 = bufInfo(m_hitScratch), b18 = bufInfo(m_bwdHitCnt);
        std::vector<VkDescriptorBufferInfo> b17 = archiveBufferInfos();

        VkWriteDescriptorSetAccelerationStructureKHR asWrite{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
        asWrite.accelerationStructureCount = 1;
        asWrite.pAccelerationStructures = &m_tlas;

        std::vector<VkWriteDescriptorSet> writes;
        auto pushBuf = [&](uint32_t binding, VkDescriptorType t, VkDescriptorBufferInfo* info, uint32_t count = 1) {
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = m_bwdDs; w.dstBinding = binding; w.descriptorCount = count; w.descriptorType = t; w.pBufferInfo = info;
            writes.push_back(w);
        };
        pushBuf(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &b0);
        { VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
          w.pNext = &asWrite; w.dstSet = m_bwdDs; w.dstBinding = 1; w.descriptorCount = 1;
          w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; writes.push_back(w); }
        pushBuf(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b2);
        pushBuf(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b3);
        pushBuf(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b4);
        pushBuf(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b5);
        pushBuf(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b6);
        pushBuf(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b7);
        pushBuf(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b8);
        pushBuf(10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b10);
        pushBuf(11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b11);
        pushBuf(12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b12);
        pushBuf(14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b14);
        pushBuf(15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b15);
        pushBuf(16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b16);
        pushBuf(17, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, b17.data(), (uint32_t)b17.size());
        pushBuf(18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b18);
        // the overflow trace runs the pipeline that also links the forward raygens: bind what those use
        // statically (9 normals, 20 split state) whenever the forward allocated them
        VkDescriptorBufferInfo b9 = bufInfo(m_outNrm), b20{};
        pushBuf(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b9);
        if (m_splitStateF.buffer) { b20 = bufInfo(m_splitStateF); pushBuf(20, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &b20); }
        vkUpdateDescriptorSets(m_device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
        m_bwdDsGen = m_resourceGen;
    }

    VkCommandBuffer cmd = beginTraceCmd();
    // The shared RT pipeline links the split gather raygen, which reads the push constant: it must be set
    // before any trace on it (VUID-vkCmdTraceRaysKHR-None-08601).
    const uint32_t round0 = 0u;
    vkCmdPushConstants(cmd, m_tracePipelineLayout, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(round0), &round0);
    // The forward archived the accepted-hit lists: no gather, the gradient pass replays them. (Barrier for
    // cross-submission visibility of the forward's writes; its writers are the split blend compute pass and the
    // tail raygen -- source both stages.)
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_bwdApplyPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tracePipelineLayout, 0, 1, &m_bwdDs, 0, nullptr);
    vkCmdDispatch(cmd, ((uint32_t)W + 7) / 8, ((uint32_t)H + 7) / 8, (uint32_t)B);
    // The rays the forward's tail finished past the archive (bwdCnt flagged) had their archived prefix replayed
    // by the apply, which left their backward state in the split state planes; the overflow raygen traces the rest.
    VkMemoryBarrier ob{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    ob.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    ob.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &ob, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_tracePipelineLayout, 0, 1, &m_bwdDs, 0, nullptr);
    m_vkCmdTraceRaysKHR(cmd, &m_sbtRgenOverflow, &m_sbtMiss, &m_sbtHit, &m_sbtCall,
                        (uint32_t)W, (uint32_t)H, (uint32_t)B);
    submitTraceAndWait();

    CUDA_CHECK(cudaMemcpyAsync(densityGrad.data_ptr(),  m_densityGrad.cudaPtr,  sizeof(float)*12*gNum, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(radianceGrad.data_ptr(), m_radianceGrad.cudaPtr, sizeof(float)*m_sphNumCoeffs*3*gNum, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return {densityGrad, radianceGrad};
}
