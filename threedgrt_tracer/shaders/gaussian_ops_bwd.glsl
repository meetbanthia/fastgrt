// SPDX-License-Identifier: Apache-2.0
//
// Backward Gaussian operations that touch storage buffers. #include AFTER the
// SSBO declarations:
//   density.particles[]   (read,   ParticleDensity[])
//   radiance.sh[]         (read,   float[])
//   densityGrad.particles[] (atomic write, ParticleDensity[])
//   radGrad.sh[]          (atomic write, float[])
// and after gaussian_common.glsl. Requires GL_EXT_shader_atomic_float.
// Ported 1:1 from processHitBwd / radianceFromSpHBwd in gaussianParticles.cuh.
//
// Gradient sinks: GRAD_SPH(base, c, v) adds v to SH coefficient c of the particle at
// radGrad.sh[base..]; GRAD_DNS(pid, field, off, v) adds v to densityGrad.particles[pid].field,
// whose float offset in ParticleDensity is off (position 0-2, density 3, quaternion 4-7,
// scale 8-10). Both default to atomics; the apply pass (STAGED_SCATTER) redefines them
// to stage the hit's gradient row in shared memory for a subgroup-cooperative scatter.
#ifndef GRAD_SPH
#define GRAD_SPH(base, c, v) atomicAdd(radGrad.sh[(base) + (c)], (v))
#endif
#ifndef GRAD_DNS
#define GRAD_DNS(pid, field, off, v) atomicAdd(densityGrad.particles[pid].field, (v))
#endif

// SH backward: accumulate grads into radGrad.sh, return clamped radiance grad.
vec3 radianceFromSpHBwd(int deg, uint base, vec3 rdir, float weight, vec3 rayRadGrd) {
    // unclamped radiance value
#ifdef SPH_WORDS
    float h[48];
    loadSph(base / 48u, deg, h);
    #define C(I) vec3(h[(I)*3+0], h[(I)*3+1], h[(I)*3+2])
#else
    #define C(I) vec3(float(radiance.sh[base+(I)*3+0]), float(radiance.sh[base+(I)*3+1]), float(radiance.sh[base+(I)*3+2]))
#endif
    vec3 gradu = SH_C0 * C(0);
    {
        float x = rdir.x, y = rdir.y, z = rdir.z;
        if (deg > 0) {
            gradu = gradu - SH_C1*y*C(1) + SH_C1*z*C(2) - SH_C1*x*C(3);
            if (deg > 1) {
                float xx=x*x, yy=y*y, zz=z*z, xy=x*y, yz=y*z, xz=x*z;
                gradu = gradu + SH_C2[0]*xy*C(4) + SH_C2[1]*yz*C(5)
                              + SH_C2[2]*(2.0*zz-xx-yy)*C(6) + SH_C2[3]*xz*C(7) + SH_C2[4]*(xx-yy)*C(8);
                if (deg > 2) {
                    gradu = gradu + SH_C3[0]*y*(3.0*xx-yy)*C(9) + SH_C3[1]*xy*z*C(10)
                                  + SH_C3[2]*y*(4.0*zz-xx-yy)*C(11) + SH_C3[3]*z*(2.0*zz-3.0*xx-3.0*yy)*C(12)
                                  + SH_C3[4]*x*(4.0*zz-xx-yy)*C(13) + SH_C3[5]*z*(xx-yy)*C(14)
                                  + SH_C3[6]*x*(xx-3.0*yy)*C(15);
                }
            }
        }
    }
    gradu += 0.5;
    vec3 grad = max(gradu, vec3(0.0));

    vec3 dL_dRGB = rayRadGrd * weight;
    dL_dRGB.x *= (gradu.x > 0.0 ? 1.0 : 0.0);
    dL_dRGB.y *= (gradu.y > 0.0 ? 1.0 : 0.0);
    dL_dRGB.z *= (gradu.z > 0.0 ? 1.0 : 0.0);

    #define ADD(I, V) { vec3 vv = (V); GRAD_SPH(base, (I)*3+0, vv.x); GRAD_SPH(base, (I)*3+1, vv.y); GRAD_SPH(base, (I)*3+2, vv.z); }
    ADD(0, SH_C0 * dL_dRGB);
    if (deg > 0) {
        float x = rdir.x, y = rdir.y, z = rdir.z;
        ADD(1, (-SH_C1*y) * dL_dRGB);
        ADD(2, ( SH_C1*z) * dL_dRGB);
        ADD(3, (-SH_C1*x) * dL_dRGB);
        if (deg > 1) {
            float xx=x*x, yy=y*y, zz=z*z, xy=x*y, yz=y*z, xz=x*z;
            ADD(4, (SH_C2[0]*xy) * dL_dRGB);
            ADD(5, (SH_C2[1]*yz) * dL_dRGB);
            ADD(6, (SH_C2[2]*(2.0*zz-xx-yy)) * dL_dRGB);
            ADD(7, (SH_C2[3]*xz) * dL_dRGB);
            ADD(8, (SH_C2[4]*(xx-yy)) * dL_dRGB);
            if (deg > 2) {
                ADD(9,  (SH_C3[0]*y*(3.0*xx-yy)) * dL_dRGB);
                ADD(10, (SH_C3[1]*xy*z) * dL_dRGB);
                ADD(11, (SH_C3[2]*y*(4.0*zz-xx-yy)) * dL_dRGB);
                ADD(12, (SH_C3[3]*z*(2.0*zz-3.0*xx-3.0*yy)) * dL_dRGB);
                ADD(13, (SH_C3[4]*x*(4.0*zz-xx-yy)) * dL_dRGB);
                ADD(14, (SH_C3[5]*z*(xx-yy)) * dL_dRGB);
                ADD(15, (SH_C3[6]*x*(xx-3.0*yy)) * dL_dRGB);
            }
        }
    }
    #undef ADD
    #undef C
    return grad;
}

// processHitBwd: replays a single gaussian's contribution in reverse and
// accumulates analytic gradients into densityGrad / radGrad. Returns whether the
// hit was accepted (i.e. whether it produced gradients).
bool processHitBwd(vec3 rayOrigin, vec3 rayDirection, uint pid,
                   float minResponse, float minAlpha, float minTransmittance, int sphDegree,
                   float integratedTransmittance,
                   inout float transmittance, float transmittanceGrad,
                   vec3 integratedRadiance, inout vec3 radianceV, vec3 radianceGrad,
                   float integratedDepth, inout float depth, float depthGrad) {
    ParticleDensity p = loadParticle(pid);
    vec4 grot = p.quaternion;
    vec3 gscl = p.scale;
    vec3 gpos = p.position;
    float gdns = p.density;

    vec3 R0, R1, R2;
    quatToMatrix(grot, R0, R1, R2);
    vec3 giscl  = 1.0 / gscl;
    vec3 gposc  = rayOrigin - gpos;
    vec3 gposcr = mulVM(gposc, R0, R1, R2);
    vec3 gro    = giscl * gposcr;
    vec3 rayDirR = mulVM(rayDirection, R0, R1, R2);
    vec3 grdu   = giscl * rayDirR;
    vec3 grd    = safeNormalize(grdu);

    vec3 gcrod  = cross(grd, gro);
    float grayDist = dot(gcrod, gcrod);

    float gres   = particleResponse(params.kernelDegree, grayDist);
    float galpha = min(params.maxAlpha, gres * gdns);

    if (!((gres > minResponse) && (galpha > minAlpha))) return false;

    vec3  grdd    = grd * dot(grd, -gro);
    vec3  grds    = gscl * grdd;
    float gsqdist = dot(grds, grds);
    float gdist   = sqrt(gsqdist);

    float weight      = galpha * transmittance;
    float nextTransmit = (1.0 - galpha) * transmittance;

    depth += weight * gdist;
    float residualHitT = max((nextTransmit <= minTransmittance) ? 0.0 : (integratedDepth - depth) / nextTransmit, 0.0);

    float galphaRayHitGrd = (gdist - residualHitT) * transmittance * depthGrad;
    vec3  grdsRayHitGrd   = gsqdist > 0.0 ? ((2.0*grds*weight) / (2.0*gdist)) * depthGrad : vec3(0.0);

    vec3  gsclRayHitGrd = grdd * grdsRayHitGrd;
    float grdScaledDot  = dot(grdsRayHitGrd * gscl, grd);
    vec3  grdRayHitGrd = gscl * grdsRayHitGrd * dot(grd, -gro) - gro * grdScaledDot;
    vec3  groRayHitGrd = -grd * grdScaledDot;

    float residualTrm     = galpha < 0.999999 ? integratedTransmittance / (1.0 - galpha) : transmittance;
    float galphaRayDnsGrd = residualTrm * -transmittanceGrad;

    uint base = pid * uint(SPH_NUM_COEFFS) * 3u;
    vec3 grad = radianceFromSpHBwd(sphDegree, base, rayDirection, weight, radianceGrad);

    vec3 rayRad = weight * grad;
    radianceV += rayRad;
    vec3 residualRayRad = max((nextTransmit <= minTransmittance) ? vec3(0.0) : (integratedRadiance - radianceV) / nextTransmit, vec3(0.0));

    float commonRad = transmittance * ((grad.x - residualRayRad.x) * radianceGrad.x
                                     + (grad.y - residualRayRad.y) * radianceGrad.y
                                     + (grad.z - residualRayRad.z) * radianceGrad.z);

    float dDensity = gres * (galphaRayHitGrd + galphaRayDnsGrd + commonRad);
    GRAD_DNS(pid, density, 3, dDensity);

    float gresGrd = gdns * (galphaRayHitGrd + galphaRayDnsGrd + commonRad);
    float grayDistGrd = particleResponseGrd(params.kernelDegree, grayDist, gres, gresGrd);

    vec3 gc = cross(grd, gro);
    vec3 gcrodGrd = 2.0 * gc * grayDistGrd;
    vec3 grdGrd = vec3(gcrodGrd.z*gro.y - gcrodGrd.y*gro.z,
                       gcrodGrd.x*gro.z - gcrodGrd.z*gro.x,
                       gcrodGrd.y*gro.x - gcrodGrd.x*gro.y);
    vec3 groGrd = vec3(gcrodGrd.y*grd.z - gcrodGrd.z*grd.y,
                       gcrodGrd.z*grd.x - gcrodGrd.x*grd.z,
                       gcrodGrd.x*grd.y - gcrodGrd.y*grd.x);

    vec3 gsclGrdGro = vec3(-gposcr.x/(gscl.x*gscl.x),
                           -gposcr.y/(gscl.y*gscl.y),
                           -gposcr.z/(gscl.z*gscl.z)) * (groGrd + groRayHitGrd);
    vec3 gposcrGrd = giscl * (groGrd + groRayHitGrd);

    vec3 gposcGrd     = matmul_bw_vec(R0, R1, R2, gposcrGrd);
    vec4 grotGrdPoscr = matmul_bw_quat(gposc, gposcrGrd, grot);

    vec3 rayMoGPosGrd = -gposcGrd;
    GRAD_DNS(pid, position.x, 0, rayMoGPosGrd.x);
    GRAD_DNS(pid, position.y, 1, rayMoGPosGrd.y);
    GRAD_DNS(pid, position.z, 2, rayMoGPosGrd.z);

    vec3 grduGrd = safe_normalize_bw(grdu, grdGrd + grdRayHitGrd);

    GRAD_DNS(pid, scale.x, 8,  gsclRayHitGrd.x + gsclGrdGro.x + (-rayDirR.x/(gscl.x*gscl.x))*grduGrd.x);
    GRAD_DNS(pid, scale.y, 9,  gsclRayHitGrd.y + gsclGrdGro.y + (-rayDirR.y/(gscl.y*gscl.y))*grduGrd.y);
    GRAD_DNS(pid, scale.z, 10, gsclRayHitGrd.z + gsclGrdGro.z + (-rayDirR.z/(gscl.z*gscl.z))*grduGrd.z);

    vec3 rayDirRGrd = giscl * grduGrd;
    vec4 grotGrdRayDirR = matmul_bw_quat(rayDirection, rayDirRGrd, grot);
    GRAD_DNS(pid, quaternion.x, 4, grotGrdPoscr.x + grotGrdRayDirR.x);
    GRAD_DNS(pid, quaternion.y, 5, grotGrdPoscr.y + grotGrdRayDirR.y);
    GRAD_DNS(pid, quaternion.z, 6, grotGrdPoscr.z + grotGrdRayDirR.z);
    GRAD_DNS(pid, quaternion.w, 7, grotGrdPoscr.w + grotGrdRayDirR.w);

    transmittance = nextTransmit;
    return true;
}
