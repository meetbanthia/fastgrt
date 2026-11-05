# SPDX-License-Identifier: Apache-2.0
#
# Vulkan port of the 3DGRT tracer. The public Python API is identical to the
# original OptiX-based tracer so the rest of the threedgrt pipeline is
# unchanged -- only the C++/GPU backend differs (Vulkan ray-query instead of
# OptiX). See README for the OptiX -> Vulkan mapping.

from .tracer import Tracer

__all__ = ["Tracer"]
