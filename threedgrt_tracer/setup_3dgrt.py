# SPDX-License-Identifier: Apache-2.0
#
# Build helper for the Vulkan 3DGRT tracer:
#   1. compile the GLSL shaders to SPIR-V (glslc, or glslangValidator),
#   2. JIT-compile + load the C++/Vulkan PyTorch extension (lib3dgrt_cc).
#
# Targets Linux + NVIDIA + Vulkan 1.3 (VK_KHR_ray_tracing_pipeline) + CUDA. Requires
# the Vulkan SDK (headers, libvulkan and a SPIR-V compiler) to be installed; set
# VULKAN_SDK if it is not on the default search paths.

import logging
import os
import shutil
import subprocess

import torch.utils.cpp_extension

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(__file__)
_SHADERS = os.path.join(_HERE, "shaders")
_SHADER_BUILD = os.path.join(_SHADERS, "build")

# Each entry: (source shader, output .spv name, extra -D defines).
_SHADER_VARIANTS = [
    # acceleration-structure builds: per-gaussian icosahedra (icosahedron) / box instances (instances)
    ("build_icosahedron.comp", "build_icosahedron.comp.spv", []),
    ("build_instances.comp", "build_instances.comp.spv", []),
    # hit group: the any-hit gathers each window's k nearest hits; the instances primitive adds an
    # intersection shader (the icosahedron uses the hardware triangle test)
    ("trace_forward.rahit", "trace_forward_tri.rahit.spv", ["TRIANGLE_PRIM"]),
    ("trace_forward.rahit", "trace_forward_inst.rahit.spv", ["INSTANCE_PRIM"]),
    ("trace_forward.rint", "trace_forward_inst.rint.spv", []),
    ("trace_forward.rmiss", "trace_forward.rmiss.spv", []),
    # training forward: gather/blend rounds, then the tail raygen finishes the rays still alive
    ("trace_forward_split_gather.rgen", "trace_forward_split_gather.rgen.spv", []),
    ("trace_forward_split_blend.comp", "trace_forward_split_blend.comp.spv", []),
    ("trace_forward.rgen", "trace_forward_split_tail.rgen.spv", ["SPLIT_TAIL"]),
    # forward that no backward follows (rendering, validation)
    ("trace_forward.rgen", "trace_forward.rgen.spv", []),
    # backward: replay of the archived hits, then the overflow raygen for the rays past the archive
    ("trace_backward_apply.comp", "trace_backward_apply.comp.spv", []),
    ("trace_backward.rgen", "trace_backward_overflow.rgen.spv", []),
]


def _find_glsl_compiler():
    # Prefer glslc: it runs spirv-opt (-O below), so the driver starts from
    # optimized SPIR-V -- the closest analog of the OptiX kernels' nvcc -O3.
    # glslangValidator emits unoptimized SPIR-V and is kept as the fallback.
    for name in ("glslc", "glslangValidator"):
        path = shutil.which(name)
        if path:
            return name, path
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        for name in ("glslc", "glslangValidator"):
            cand = os.path.join(sdk, "bin", name)
            if os.path.exists(cand):
                return name, cand
    raise RuntimeError(
        "No SPIR-V compiler found (glslangValidator or glslc). Install the Vulkan SDK "
        "and/or set VULKAN_SDK."
    )


def compile_shaders(base_defines, subgroup_atomics=True):
    """Compile every shader variant into shaders/build (recompiled on every import, so config changes apply)."""
    os.makedirs(_SHADER_BUILD, exist_ok=True)
    name, compiler = _find_glsl_compiler()

    def _compile(src_name, out_name, defines):
        src = os.path.join(_SHADERS, src_name)
        dst = os.path.join(_SHADER_BUILD, out_name)
        if name == "glslangValidator":
            cmd = [compiler, "--target-env", "vulkan1.3"]
            cmd += [f"-D{d}" for d in defines]
            cmd += ["-I" + _SHADERS, "-o", dst, src]
        else:  # glslc
            cmd = [compiler, "--target-env=vulkan1.3", "-O"]
            cmd += [f"-D{d}" for d in defines]
            cmd += ["-I", _SHADERS, "-o", dst, src]
        logger.info("Compiling shader %s -> %s", src_name, dst)
        subprocess.run(cmd, check=True)

    for src_name, out_name, extra_defines in _SHADER_VARIANTS:
        defines = list(base_defines) + list(extra_defines)
        # render.subgroup_atomics=false: the backward's apply pass adds each lane's gradients with its own global
        # atomics instead of summing them across the subgroup first (to measure what the subgroup scatter saves)
        if src_name == "trace_backward_apply.comp" and not subgroup_atomics:
            defines.append("APPLY_DIRECT_ATOMICS")
        _compile(src_name, out_name, defines)


def _vulkan_paths():
    include_dirs, library_dirs = [], []
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        include_dirs.append(os.path.join(sdk, "include"))
        library_dirs.append(os.path.join(sdk, "lib"))
    # The conda toolchain's linker does NOT search /usr/lib by default. Prefer
    # the system loader (libvulkan.so) -- it's what ldconfig resolves at runtime,
    # so link- and run-time match. (Both the system loader and the SDK loader are
    # just dispatch shims to the installed driver/ICD; the version doesn't matter.)
    for d in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/lib/x86_64-linux-gnu"):
        if os.path.isdir(d):
            library_dirs.append(d)
    # Fallback: newer SDKs nest the loader under lib/VulkanLoader/lib instead of lib/.
    if sdk:
        nested = os.path.join(sdk, "lib", "VulkanLoader", "lib")
        if os.path.isdir(nested):
            library_dirs.append(nested)
    return include_dirs, library_dirs


def setup_3dgrt(conf):
    if conf.render.primitive_type not in ("icosahedron", "instances"):
        raise ValueError(
            f"render.primitive_type={conf.render.primitive_type!r} is not supported by the Vulkan tracer "
            "(use 'icosahedron' or 'instances')"
        )
    # Compile-time specialization matching the OptiX build: OptiX passes -DENABLE_NORMALS /
    # -DENABLE_HIT_COUNTS and bakes the kernel degree as a template parameter. Without these, the
    # disabled branches stay in the raygen integrate loop and inflate its register allocation.
    base_defines = [
        f"SPH_NUM_COEFFS={(conf.render.particle_radiance_sph_degree + 1) ** 2}",
        f"ENABLE_NORMALS={1 if conf.render.enable_normals else 0}",
        f"ENABLE_HIT_COUNTS={1 if conf.render.enable_hitcounts else 0}",
        f"KERNEL_DEGREE={int(conf.render.particle_kernel_degree)}",
    ]
    compile_shaders(base_defines, subgroup_atomics=conf.render.get("subgroup_atomics", True))

    vk_includes, vk_libdirs = _vulkan_paths()

    include_paths = [os.path.join(_HERE, "include")] + vk_includes
    source_files = ["src/vulkan_tracer.cpp", "bindings.cpp"]
    source_paths = [os.path.join(_HERE, fn) for fn in source_files]

    ldflags = ["-lvulkan", "-lcudart"]
    for d in vk_libdirs:
        ldflags.append(f"-L{d}")

    return torch.utils.cpp_extension.load(
        name="lib3dgrt_cc",
        sources=source_paths,
        extra_cflags=["-O3", "-std=c++17"],
        extra_include_paths=include_paths,
        extra_ldflags=ldflags,
        with_cuda=True,
        verbose=True,
    )
