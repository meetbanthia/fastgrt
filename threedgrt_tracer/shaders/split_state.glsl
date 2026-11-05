// SPDX-License-Identifier: Apache-2.0
// The split forward's per-ray state: eight float planes, [plane * statePlaneRays + pix].
layout(set = 0, binding = 20, scalar) buffer SplitStateF { float v[]; } stateF;
