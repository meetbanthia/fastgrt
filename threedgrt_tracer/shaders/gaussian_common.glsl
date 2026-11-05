// SPDX-License-Identifier: Apache-2.0
//
// Gaussian particle math, ported 1:1 from the CUDA reference implementation:
//   threedgrt_tracer/include/3dgrt/kernels/cuda/gaussianParticles.cuh
// and the slang models. Matrix/vector conventions match mathUtils.h:
//   - float33 is row-major; rows are R0, R1, R2.
//   - (v * M) == vec3(dot(R0,v), dot(R1,v), dot(R2,v))   [used for ray->canonical]
//   - (M * v) == M^T * v                                  [used for normals]

// ---- rotation matrix from a (w,x,y,z) quaternion (== quaternionWXYZToMatrix)
// Returns the three rows of the 3x3 rotation matrix.
void quatToMatrix(vec4 q, out vec3 R0, out vec3 R1, out vec3 R2) {
    float r = q.x, x = q.y, y = q.z, z = q.w;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float rx = r * x, ry = r * y, rz = r * z;
    R0 = vec3(1.0 - 2.0 * (yy + zz), 2.0 * (xy + rz),       2.0 * (xz - ry));
    R1 = vec3(2.0 * (xy - rz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz + rx));
    R2 = vec3(2.0 * (xz + ry),       2.0 * (yz - rx),       1.0 - 2.0 * (xx + yy));
}

// (v * M): rows dotted with v
vec3 mulVM(vec3 v, vec3 R0, vec3 R1, vec3 R2) {
    return vec3(dot(R0, v), dot(R1, v), dot(R2, v));
}

// (M * v): == M^T * v
vec3 mulMV(vec3 R0, vec3 R1, vec3 R2, vec3 v) {
    return vec3(R0.x * v.x + R1.x * v.y + R2.x * v.z,
                R0.y * v.x + R1.y * v.y + R2.y * v.z,
                R0.z * v.x + R1.z * v.y + R2.z * v.z);
}

vec3 safeNormalize(vec3 v) {
    float l = dot(v, v);
    return l > 0.0 ? v * inversesqrt(l) : v;
}

// ---- generalized gaussian response (== particleResponse<degree>) -----------
float particleResponse(int degree, float grayDist) {
#ifdef KERNEL_DEGREE
    degree = KERNEL_DEGREE;   // compile-time fold (== the OptiX template parameter)
#endif
    switch (degree) {
        case 8: { float s = -0.000685871056241; float d2 = grayDist * grayDist; return exp(s * d2 * d2); }
        case 5: { float s = -0.0185185185185;   return exp(s * grayDist * grayDist * sqrt(grayDist)); }
        case 4: { float s = -0.0555555555556;   return exp(s * grayDist * grayDist); }
        case 3: { float s = -0.166666666667;    return exp(s * grayDist * sqrt(grayDist)); }
        case 1: { float s = -1.5;               return exp(s * sqrt(grayDist)); }
        case 0: { float s = -0.329630334487;    return max(1.0 + s * sqrt(grayDist), 0.0); }
        default:{ float s = -0.5;               return exp(s * grayDist); }   // quadratic
    }
}

// ---- spherical-harmonics radiance (== radianceFromSpH, clamped) ------------
const float SH_C0 = 0.28209479177387814;
const float SH_C1 = 0.4886025119029199;
const float SH_C2[5] = float[5](1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
                                -1.0925484305920792, 0.5462742152960396);
const float SH_C3[7] = float[7](-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
                                -0.4570457994644658, 1.445305721320277, -0.5900435899266435);

// NOTE: functions that read/write storage buffers (radianceFromSpH,
// processHitFwd, intersectInstanceParticle, processHitBwd) live in
// gaussian_ops.glsl / gaussian_ops_bwd.glsl. Those files are #included by each
// compute shader AFTER its SSBOs are declared, so they can reference the global
// buffer variables directly (GLSL cannot pass runtime-sized arrays to
// functions). The radiance SH buffer is laid out as
//   radiance.sh[particleIdx * SPH_NUM_COEFFS * 3 + coeffIdx * 3 + channel].

// ===========================================================================
// Pure backward-pass helpers (no storage buffers), ported from mathUtils.h /
// gaussianParticles.cuh. Only needed by the backward shader.
// ===========================================================================

// == particleResponseGrd: d(response)/d(grayDist) * gresGrd, given gres.
float particleResponseGrd(int degree, float grayDist, float gres, float gresGrd) {
#ifdef KERNEL_DEGREE
    degree = KERNEL_DEGREE;   // compile-time fold (== the OptiX template parameter)
#endif
    switch (degree) {
        case 8: { float s = -0.000685871056241 * (0.5*8.0); float d2 = grayDist*grayDist; return s*d2*grayDist*gres*gresGrd; }
        case 5: { float s = -0.0185185185185   * (0.5*5.0); return s*grayDist*sqrt(grayDist)*gres*gresGrd; }
        case 4: { float s = -0.0555555555556   * (0.5*4.0); return s*grayDist*gres*gresGrd; }
        case 3: { float s = -0.166666666667    * (0.5*3.0); return s*sqrt(grayDist)*gres*gresGrd; }
        case 1: { float s = -1.5               * (0.5*1.0); return s*sqrt(grayDist)*gres*gresGrd; }
        case 0: { float s = -0.329630334487;                return gres > 0.0 ? (0.5*s*inversesqrt(grayDist))*gresGrd : 0.0; }
        default:{ float s = -0.5;                           return s*gres*gresGrd; }
    }
}

// == matmul_bw_vec(m, gdt) == M^T * gdt with M rows R0,R1,R2.
vec3 matmul_bw_vec(vec3 R0, vec3 R1, vec3 R2, vec3 g) {
    return vec3(g.x*R0.x + g.y*R1.x + g.z*R2.x,
                g.x*R0.y + g.y*R1.y + g.z*R2.y,
                g.x*R0.z + g.y*R1.z + g.z*R2.z);
}

// == safe_normalize_bw(v, d_out)
vec3 safe_normalize_bw(vec3 v, vec3 d_out) {
    float l = dot(v, v);
    if (l > 0.0) {
        float il = inversesqrt(l);
        float il3 = il * il * il;
        return il * d_out - il3 * vec3(
            d_out.x*(v.x*v.x) + d_out.y*(v.y*v.x) + d_out.z*(v.z*v.x),
            d_out.x*(v.x*v.y) + d_out.y*(v.y*v.y) + d_out.z*(v.z*v.y),
            d_out.x*(v.x*v.z) + d_out.y*(v.y*v.z) + d_out.z*(v.z*v.z));
    }
    return vec3(0.0);
}

// == matmul_bw_quat(p, g, q): gradient wrt quaternion of (p * R(q)).
// p is the input vector, g is the upstream grad of the output vector.
vec4 matmul_bw_quat(vec3 p, vec3 g, vec4 q) {
    vec3 d0 = g.x * p;
    vec3 d1 = g.y * p;
    vec3 d2 = g.z * p;
    float r = q.x, x = q.y, y = q.z, z = q.w;
    float dr = 0.0, dx = 0.0, dy = 0.0, dz = 0.0;

    // m[0].x = (1 - 2(yy+zz))
    dy += -4.0*y*d0.x; dz += -4.0*z*d0.x;
    // m[0].y = 2(xy + rz)
    dr += 2.0*z*d0.y; dx += 2.0*y*d0.y; dy += 2.0*x*d0.y; dz += 2.0*r*d0.y;
    // m[0].z = 2(xz - ry)
    dr += -2.0*y*d0.z; dx += 2.0*z*d0.z; dy += -2.0*r*d0.z; dz += 2.0*x*d0.z;

    // m[1].x = 2(xy - rz)
    dr += -2.0*z*d1.x; dx += 2.0*y*d1.x; dy += 2.0*x*d1.x; dz += -2.0*r*d1.x;
    // m[1].y = (1 - 2(xx+zz))
    dx += -4.0*x*d1.y; dz += -4.0*z*d1.y;
    // m[1].z = 2(yz + rx)
    dr += 2.0*x*d1.z; dx += 2.0*r*d1.z; dy += 2.0*z*d1.z; dz += 2.0*y*d1.z;

    // m[2].x = 2(xz + ry)
    dr += 2.0*y*d2.x; dx += 2.0*z*d2.x; dy += 2.0*r*d2.x; dz += 2.0*x*d2.x;
    // m[2].y = 2(yz - rx)
    dr += -2.0*x*d2.y; dx += -2.0*r*d2.y; dy += 2.0*z*d2.y; dz += 2.0*y*d2.y;
    // m[2].z = (1 - 2(xx+yy))
    dx += -4.0*x*d2.z; dy += -4.0*y*d2.z;

    return vec4(dr, dx, dy, dz);
}
