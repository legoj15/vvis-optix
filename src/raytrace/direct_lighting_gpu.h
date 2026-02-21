//========= GPU Direct Lighting Support ============//
//
// Purpose: Data structures and functions for GPU-accelerated direct lighting
//
//=============================================================================//

#ifndef DIRECT_LIGHTING_GPU_H
#define DIRECT_LIGHTING_GPU_H

#ifdef VRAD_RTX_CUDA_SUPPORT

#include "gpu_scene_data.h"
#include "raytrace_shared.h"

// GPU-friendly light structure matching directlight_t
// NOTE: Uses explicit float fields instead of float3_t to guarantee identical
// struct layout between MSVC (host) and NVCC (device PTX) compilation.
// float3_t (typedef float3 on NVCC) can introduce padding differences.
struct GPULight {
  float origin_x, origin_y, origin_z;
  float intensity_x, intensity_y, intensity_z;
  float normal_x, normal_y, normal_z; // For emit_surface and emit_spotlight

  int type; // emit_point=0, emit_surface=1, emit_spotlight=2, emit_skylight=3
  int facenum; // -1 for point lights, face index for surface lights

  // Attenuation
  float constant_attn;
  float linear_attn;
  float quadratic_attn;

  // Spotlight parameters
  float stopdot;  // cos(inner cone angle)
  float stopdot2; // cos(outer cone angle)
  float exponent;

  // Fade distances
  float startFadeDistance;
  float endFadeDistance;
  float capDist;

  // Lightstyle index (0 = always-on, 1-63 = named/switchable)
  int style;
};

// Shadow ray for direct lighting
struct GPUShadowRay {
  float3_t origin;    // Sample position (offset by epsilon from surface)
  float3_t direction; // Normalized direction to light
  float tmax;         // Distance to light minus epsilon
  int sampleIndex;    // Index into face's sample array
  int lightIndex;     // Index into GPU light array
  int faceIndex;      // Face number for result accumulation
};

// Result of shadow ray trace
struct GPUShadowResult {
  int visible; // 1 if light is visible, 0 if blocked
  float hitT;  // Hit distance (for debugging)
};

// Direct lighting batch for GPU processing
struct DirectLightingBatch {
  GPUShadowRay *rays;
  GPUShadowResult *results;
  int numRays;
  int maxRays;
};

// Initialize GPU direct lighting with pre-converted lights from vrad
void InitDirectLightingGPU(const GPULight *lights, int numLights);

// Shutdown GPU direct lighting
void ShutdownDirectLightingGPU();

// Trace shadow rays for direct lighting
void TraceShadowBatch(const GPUShadowRay *rays, GPUShadowResult *results,
                      int numRays);

// Get GPU light count
int GetGPULightCount();

// Get GPU light array (device pointer)
GPULight *GetGPULights();

// Get host light by index (for falloff calculations on CPU)
const GPULight *GetHostLight(int index);

#endif // VRAD_RTX_CUDA_SUPPORT

#endif // DIRECT_LIGHTING_GPU_H
