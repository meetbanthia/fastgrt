// SPDX-License-Identifier: Apache-2.0
//
// pybind module for the Vulkan 3DGRT tracer. The class name and method
// signatures mirror the original OptiX binding so threedgrt_tracer/tracer.py
// can drive either backend unchanged.

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <3dgrt/vulkan_tracer.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<VulkanTracer>(m, "VulkanTracer")
        .def(pybind11::init<const std::string&, const std::string&, const std::string&, const std::string&,
                            const std::string&, float, float, bool, int, bool, bool>())
        // The calls wait on Vulkan fences for most of a training step; they touch no Python objects, so release
        // the GIL meanwhile and let the data loader, TensorBoard and progress threads run instead of piling up.
        .def("trace", &VulkanTracer::trace, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("trace_bwd", &VulkanTracer::traceBwd, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("build_bvh", &VulkanTracer::buildBVH,
             pybind11::arg("mogPos"), pybind11::arg("mogRot"), pybind11::arg("mogScl"),
             pybind11::arg("mogDns"), pybind11::arg("rebuild"), pybind11::arg("allow_update"),
             pybind11::arg("iteration") = -1, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("release_training_buffers", &VulkanTracer::releaseTrainingBuffers,
             pybind11::call_guard<pybind11::gil_scoped_release>());
}
