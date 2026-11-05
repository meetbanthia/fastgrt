// SPDX-License-Identifier: Apache-2.0
//
// Forward Gaussian operations that touch storage buffers. #include AFTER the
// SSBO declarations (density.particles[], radiance.sh[]) and after
// gaussian_common.glsl. SPH_NUM_COEFFS must be defined (specialization or
// #define) before inclusion.

// A particle's ParticleDensity is 48 B, three aligned 16-byte words. Read as words (binding 2
// again, as an alias of density.particles) it takes 3 loads instead of 12, same values.
layout(set = 0, binding = 2, std430) readonly buffer DensityWords { vec4 w[]; } densityWords;
ParticleDensity loadParticle(uint pid) {
    const vec4 a = densityWords.w[pid * 3u], b = densityWords.w[pid * 3u + 1u], c = densityWords.w[pid * 3u + 2u];
    return ParticleDensity(a.xyz, a.w, b, c.xyz, c.w);
}

#if SPH_NUM_COEFFS == 16
// A particle's SH coefficients are 48 fp16 values, 96 B: six 16-byte words. Reading them as
// words (binding 3 again, as an alias of radiance.sh) takes 6 loads per particle instead of
// 48 two-byte ones, with the same values. Only the words the degree uses are read.
layout(set = 0, binding = 3, std430) readonly buffer RadianceWords { uvec4 w[]; } radianceWords;
#define SPH_WORDS
#define SPH_UNPACK(q) \
    if ((q) * 8u < used) { \
        const uvec4 u = radianceWords.w[pid * 6u + (q)]; \
        vec2 p2 = unpackHalf2x16(u.x); h[(q) * 8 + 0] = p2.x; h[(q) * 8 + 1] = p2.y; \
        p2 = unpackHalf2x16(u.y);      h[(q) * 8 + 2] = p2.x; h[(q) * 8 + 3] = p2.y; \
        p2 = unpackHalf2x16(u.z);      h[(q) * 8 + 4] = p2.x; h[(q) * 8 + 5] = p2.y; \
        p2 = unpackHalf2x16(u.w);      h[(q) * 8 + 6] = p2.x; h[(q) * 8 + 7] = p2.y; \
    }
// h[i] = coefficient i / 3, channel i % 3 of particle pid, for the coefficients degree deg uses
void loadSph(uint pid, int deg, out float h[48]) {
    for (int i = 0; i < 48; ++i) h[i] = 0.0;
    const uint used = 3u * uint((deg + 1) * (deg + 1));
    SPH_UNPACK(0u) SPH_UNPACK(1u) SPH_UNPACK(2u) SPH_UNPACK(3u) SPH_UNPACK(4u) SPH_UNPACK(5u)
}
#undef SPH_UNPACK
#endif

// SH radiance decode reading the global radiance.sh[] buffer.
// (== radianceFromSpH(deg, coeffs, dir) with clamp to >= 0)
vec3 radianceFromSpH(int deg, uint base, vec3 dir) {
#ifdef SPH_WORDS
    float h[48];
    loadSph(base / 48u, deg, h);
    #define C(I) vec3(h[(I)*3 + 0], h[(I)*3 + 1], h[(I)*3 + 2])
#else
    #define C(I) vec3(float(radiance.sh[base + (I)*3 + 0]), float(radiance.sh[base + (I)*3 + 1]), float(radiance.sh[base + (I)*3 + 2]))
#endif
    vec3 rad = SH_C0 * C(0);
    if (deg > 0) {
        float x = dir.x, y = dir.y, z = dir.z;
        rad = rad - SH_C1 * y * C(1) + SH_C1 * z * C(2) - SH_C1 * x * C(3);
        if (deg > 1) {
            float xx = x*x, yy = y*y, zz = z*z;
            float xy = x*y, yz = y*z, xz = x*z;
            rad = rad + SH_C2[0]*xy*C(4) + SH_C2[1]*yz*C(5)
                      + SH_C2[2]*(2.0*zz - xx - yy)*C(6)
                      + SH_C2[3]*xz*C(7) + SH_C2[4]*(xx - yy)*C(8);
            if (deg > 2) {
                rad = rad + SH_C3[0]*y*(3.0*xx - yy)*C(9)
                          + SH_C3[1]*xy*z*C(10)
                          + SH_C3[2]*y*(4.0*zz - xx - yy)*C(11)
                          + SH_C3[3]*z*(2.0*zz - 3.0*xx - 3.0*yy)*C(12)
                          + SH_C3[4]*x*(4.0*zz - xx - yy)*C(13)
                          + SH_C3[5]*z*(xx - yy)*C(14)
                          + SH_C3[6]*x*(xx - 3.0*yy)*C(15);
            }
        }
    }
    #undef C
    rad += 0.5;
    return max(rad, vec3(0.0));
}

// == intersectInstanceParticle : the reference's instanced-primitive test (primitive_type=instances), a port of
// intersectInstanceParticle in the OptiX gaussianParticles.cuh. The ray is in the instance's object frame (the unit
// box [-1,1]^3 scaled by kernelScale*scale, rotated and translated); its direction is not normalized, so a ray
// parameter is the same in the object and the world frame. The hit is the point of the ray closest to the centre.
bool intersectInstanceParticle(vec3 objRayOrigin, vec3 objRayDirection,
                               float minHitDistance, float maxHitDistance,
                               out float hitDistance) {
    float numerator   = -dot(objRayOrigin, objRayDirection);
    float denominator = 1.0 / dot(objRayDirection, objRayDirection);
    hitDistance       = numerator * denominator;
    if (hitDistance > minHitDistance && hitDistance < maxHitDistance) {
        vec3 gcrod = cross(safeNormalize(objRayDirection), objRayOrigin);
        return dot(gcrod, gcrod) * denominator < params.maxParticleSqDist;
    }
    return false;
}

// == processHit (forward). Front-to-back compositing of a single gaussian.
// Returns true if the hit was accepted (contributed).
bool processHitFwd(vec3 rayOrigin, vec3 rayDirection, uint pid,
                   inout float transmittance, inout vec3 radianceAccum,
                   inout float depthAccum, inout vec3 normalAccum) {
    ParticleDensity p = loadParticle(pid);
    vec3 R0, R1, R2;
    quatToMatrix(p.quaternion, R0, R1, R2);
    vec3 giscl = 1.0 / p.scale;
    vec3 gposc = rayOrigin - p.position;
    vec3 gro   = giscl * mulVM(gposc, R0, R1, R2);
    vec3 grd   = safeNormalize(giscl * mulVM(rayDirection, R0, R1, R2));

    vec3  gcrod = cross(grd, gro);
    float grayDist = dot(gcrod, gcrod);

    float gres   = particleResponse(params.kernelDegree, grayDist);
    float galpha = min(params.maxAlpha, gres * p.density);

    bool acceptHit = (gres > params.minResponse) && (galpha > params.minAlpha);
    if (acceptHit) {
        float weight = galpha * transmittance;

        vec3  grds = p.scale * grd * dot(grd, -gro);
        float hitT = sqrt(dot(grds, grds));

        uint base = pid * uint(SPH_NUM_COEFFS) * 3u;
        vec3 grad = radianceFromSpH(params.sphDegree, base, rayDirection);

        radianceAccum += grad * weight;
        transmittance *= (1.0 - galpha);
        depthAccum    += hitT * weight;

        if (NORMALS_ON) {
            const float ellipsoidSqRadius = 9.0;
            vec3 particleScaleRotated = mulMV(R0, R1, R2, p.scale);
            normalAccum += weight * safeNormalize((gro + grd * (dot(grd, -gro) - sqrt(ellipsoidSqRadius - grayDist)))
                                                  * particleScaleRotated);
        }
    }
    return acceptHit;
}
