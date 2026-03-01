// optix_kernels.cu - OptiX 9.1 device programs for vrad_nextgen
// Compiled to PTX by compile_ptx.bat

#include <optix.h>

struct PointLight {
  float posX, posY, posZ;
  float colR, colG, colB;
  float constAttn, linearAttn, quadAttn;
};

struct GPUFaceInfo {
  float lmS[4];
  float lmT[4];
  int lmMinsS, lmMinsT;
  int lmW, lmH;
  int luxelOffset;
  float reflR, reflG, reflB;
  float normalX, normalY, normalZ;
};

struct LaunchParams {
  OptixTraversableHandle traversable;
  float *outputBuffer;
  int numSamples;
  float sunDirX, sunDirY, sunDirZ;
  float sunColorR, sunColorG, sunColorB;
  int hasSun;
  PointLight *lights;
  int numLights;
  float *samplePositions;
  float *sampleNormals;
  int *triToFace;
  GPUFaceInfo *faceInfos;
  float *bounceLightIn;
  int bouncePassIndex; // -1 = direct, 0+ = bounce
  int numBounceRays;
  int totalLuxels;
};

extern "C" {
__constant__ LaunchParams params;
}

__device__ unsigned int pcg_hash(unsigned int input) {
  unsigned int state = input * 747796405u + 2891336453u;
  unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

__device__ unsigned int next_pcg(unsigned int &state) {
  unsigned int oldstate = state;
  state = oldstate * 747796405u + 2891336453u;
  unsigned int word =
      ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
  return (word >> 22u) ^ word;
}

__device__ float pcg_float(unsigned int &state) {
  unsigned int r = next_pcg(state);
  return (float)(r & 0x00FFFFFF) / (float)0x01000000;
}

__device__ void cosine_hemisphere(float u1, float u2, float nx, float ny,
                                  float nz, float &dx, float &dy, float &dz) {
  float r = sqrtf(u1);
  float theta = 6.283185307f * u2;
  float lx = r * cosf(theta);
  float ly = r * sinf(theta);
  float lz = sqrtf(1.0f - u1);

  float upX, upY, upZ;
  if (fabsf(nx) < 0.9f) {
    upX = 1;
    upY = 0;
    upZ = 0;
  } else {
    upX = 0;
    upY = 1;
    upZ = 0;
  }

  float tx = upY * nz - upZ * ny;
  float ty = upZ * nx - upX * nz;
  float tz = upX * ny - upY * nx;
  float tlen = sqrtf(tx * tx + ty * ty + tz * tz);
  if (tlen < 1e-8f) {
    dx = nx;
    dy = ny;
    dz = nz;
    return;
  }
  tx /= tlen;
  ty /= tlen;
  tz /= tlen;

  float bx = ny * tz - nz * ty;
  float by = nz * tx - nx * tz;
  float bz = nx * ty - ny * tx;

  dx = lx * tx + ly * bx + lz * nx;
  dy = lx * ty + ly * by + lz * ny;
  dz = lx * tz + ly * bz + lz * nz;
}

extern "C" __global__ void __raygen__lightmap() {
  const uint3 idx = optixGetLaunchIndex();
  int sampleIdx = idx.x;
  if (sampleIdx >= params.numSamples)
    return;

  float px = params.samplePositions[sampleIdx * 3 + 0];
  float py = params.samplePositions[sampleIdx * 3 + 1];
  float pz = params.samplePositions[sampleIdx * 3 + 2];

  float nx = params.sampleNormals[sampleIdx * 3 + 0];
  float ny = params.sampleNormals[sampleIdx * 3 + 1];
  float nz = params.sampleNormals[sampleIdx * 3 + 2];

  float totalR = 0, totalG = 0, totalB = 0;
  // Surface offset bias: must match reference VRAD's 1.0-unit offset
  // (ComputeIlluminationPointAndNormalsSSE: pInfo->m_Points += faceNormal).
  // 0.5 was too small — caused self-intersection noise at high luxel density.
  float bias = 1.0f;

  if (params.bouncePassIndex < 0) {
    if (params.hasSun) {
      float NdotL =
          nx * params.sunDirX + ny * params.sunDirY + nz * params.sunDirZ;
      if (NdotL > 0.0f) {
        unsigned int p0 = 0, p1 = 0;
        optixTrace(params.traversable,
                   make_float3(px + nx * bias, py + ny * bias, pz + nz * bias),
                   make_float3(params.sunDirX, params.sunDirY, params.sunDirZ),
                   0.0f, 1e16f, 0.0f, OptixVisibilityMask(255),
                   OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                       OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                   0, 1, 0, p0, p1);
        if (p0 == 0xFFFFFFFF) {
          totalR += NdotL * params.sunColorR;
          totalG += NdotL * params.sunColorG;
          totalB += NdotL * params.sunColorB;
        }
      }
    }
    for (int li = 0; li < params.numLights; li++) {
      PointLight light = params.lights[li];
      float dx = light.posX - px, dy = light.posY - py, dz = light.posZ - pz;
      float distSq = dx * dx + dy * dy + dz * dz;
      if (distSq < 1e-6f)
        continue;
      float dist = sqrtf(distSq);
      float invDist = 1.0f / dist;
      float ldx = dx * invDist, ldy = dy * invDist, ldz = dz * invDist;
      float NdotL = nx * ldx + ny * ldy + nz * ldz;
      if (NdotL <= 0.0f)
        continue;
      float attenDist = fmaxf(dist, 1.0f);
      float atten = light.constAttn + light.linearAttn * attenDist +
                    light.quadAttn * attenDist * attenDist;
      if (atten < 1e-6f)
        atten = 1e-6f;
      float falloff = 1.0f / atten;
      unsigned int p0 = 0, p1 = 0;
      optixTrace(params.traversable,
                 make_float3(px + nx * bias, py + ny * bias, pz + nz * bias),
                 make_float3(ldx, ldy, ldz), 0.0f, dist - bias, 0.0f,
                 OptixVisibilityMask(255),
                 OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                     OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                 0, 1, 0, p0, p1);
      if (p0 == 0xFFFFFFFF) {
        totalR += NdotL * falloff * light.colR;
        totalG += NdotL * falloff * light.colG;
        totalB += NdotL * falloff * light.colB;
      }
    }
  } else {
    unsigned int seed =
        pcg_hash(sampleIdx * 1973 + params.bouncePassIndex * 9277);
    for (int i = 0; i < params.numBounceRays; i++) {
      float r1 = pcg_float(seed);
      float r2 = pcg_float(seed);

      float dx, dy, dz;
      cosine_hemisphere(r1, r2, nx, ny, nz, dx, dy, dz);

      unsigned int p0 = 0xFFFFFFFF, p1 = 0;
      optixTrace(params.traversable,
                 make_float3(px + nx * bias, py + ny * bias, pz + nz * bias),
                 make_float3(dx, dy, dz), 0.0f, 99999.0f, 0.0f,
                 OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0,
                 p1);

      if (p0 != 0xFFFFFFFF) {
        int faceIndex = params.triToFace[p0];
        GPUFaceInfo fi = params.faceInfos[faceIndex];

        float t = __int_as_float(p1);
        float hx = px + nx * bias + dx * t;
        float hy = py + ny * bias + dy * t;
        float hz = pz + nz * bias + dz * t;

        float u = fi.lmS[0] * hx + fi.lmS[1] * hy + fi.lmS[2] * hz + fi.lmS[3];
        float v = fi.lmT[0] * hx + fi.lmT[1] * hy + fi.lmT[2] * hz + fi.lmT[3];
        int luxU = (int)floorf(u) - fi.lmMinsS;
        int luxV = (int)floorf(v) - fi.lmMinsT;

        if (luxU < 0)
          luxU = 0;
        if (luxU >= fi.lmW)
          luxU = fi.lmW - 1;
        if (luxV < 0)
          luxV = 0;
        if (luxV >= fi.lmH)
          luxV = fi.lmH - 1;

        int offset = fi.luxelOffset + (luxV * fi.lmW + luxU);

        float hitR = params.bounceLightIn[offset * 3 + 0];
        float hitG = params.bounceLightIn[offset * 3 + 1];
        float hitB = params.bounceLightIn[offset * 3 + 2];

        totalR += hitR * fi.reflR;
        totalG += hitG * fi.reflG;
        totalB += hitB * fi.reflB;
      }
    }

    float invN = 1.0f / (float)params.numBounceRays;
    totalR *= invN;
    totalG *= invN;
    totalB *= invN;
  }

  params.outputBuffer[sampleIdx * 3 + 0] = totalR;
  params.outputBuffer[sampleIdx * 3 + 1] = totalG;
  params.outputBuffer[sampleIdx * 3 + 2] = totalB;
}

extern "C" __global__ void __closesthit__radiance() {
  optixSetPayload_0(optixGetPrimitiveIndex());
  float t = optixGetRayTmax();
  optixSetPayload_1(*(unsigned int *)&t);
}

extern "C" __global__ void __miss__radiance() { optixSetPayload_0(0xFFFFFFFF); }
