//========= GPU Direct Lighting Implementation ============//
//
// Purpose: GPU-accelerated direct lighting using OptiX
//
//=============================================================================//

#include "direct_lighting_gpu.h"

#ifdef VRAD_RTX_CUDA_SUPPORT

#include "raytrace_optix.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

// Static storage for GPU lights
static std::vector<GPULight> s_hostLights;
static GPULight *s_deviceLights = nullptr;
static int s_numLights = 0;
static bool s_initialized = false;

//-----------------------------------------------------------------------------
// Initialize GPU direct lighting with pre-converted lights
// Called from vrad side with already-converted GPULight array
//-----------------------------------------------------------------------------
void InitDirectLightingGPU(const GPULight *lights, int numLights) {
  if (s_initialized) {
    ShutdownDirectLightingGPU();
  }

  s_numLights = numLights;

  if (s_numLights == 0) {
    s_initialized = true;
    return;
  }

  // Copy to host vector
  s_hostLights.assign(lights, lights + numLights);

  // Allocate and upload to GPU
  size_t lightSize = s_numLights * sizeof(GPULight);
  cudaError_t err = cudaMalloc(&s_deviceLights, lightSize);
  if (err != cudaSuccess) {
    printf("InitDirectLightingGPU: cudaMalloc failed: %s\n",
           cudaGetErrorString(err));
    return;
  }

  err = cudaMemcpy(s_deviceLights, s_hostLights.data(), lightSize,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf("InitDirectLightingGPU: cudaMemcpy failed: %s\n",
           cudaGetErrorString(err));
    cudaFree(s_deviceLights);
    s_deviceLights = nullptr;
    return;
  }

  s_initialized = true;
  printf("InitDirectLightingGPU: Uploaded %d lights to GPU\n", s_numLights);
}

//-----------------------------------------------------------------------------
// Shutdown GPU direct lighting
//-----------------------------------------------------------------------------
void ShutdownDirectLightingGPU() {
  if (s_deviceLights) {
    cudaFree(s_deviceLights);
    s_deviceLights = nullptr;
  }
  s_hostLights.clear();
  s_numLights = 0;
  s_initialized = false;
}

//-----------------------------------------------------------------------------
// Trace shadow rays using OptiX
//-----------------------------------------------------------------------------
void TraceShadowBatch(const GPUShadowRay *rays, GPUShadowResult *results,
                      int numRays) {
  if (!s_initialized || numRays <= 0) {
    return;
  }

  // Convert GPUShadowRay to RayBatch for existing TraceBatch infrastructure
  std::vector<RayBatch> rayBatch(numRays);
  std::vector<RayResult> rayResults(numRays);

  for (int i = 0; i < numRays; i++) {
    rayBatch[i].origin = rays[i].origin;
    rayBatch[i].direction = rays[i].direction;
    rayBatch[i].tmin = 1e-4f; // Small offset to avoid self-intersection
    rayBatch[i].tmax = rays[i].tmax;
    rayBatch[i].skip_id = -1; // No skip for shadow rays
  }

  // Use existing TraceBatch
  RayTraceOptiX::TraceBatch(rayBatch.data(), rayResults.data(), numRays);

  // Convert results
  for (int i = 0; i < numRays; i++) {
    // Visible if no hit (hit_id == -1)
    results[i].visible = (rayResults[i].hit_id == -1) ? 1 : 0;
    results[i].hitT = rayResults[i].hit_t;
  }
}

//-----------------------------------------------------------------------------
// Get GPU light count
//-----------------------------------------------------------------------------
int GetGPULightCount() { return s_numLights; }

//-----------------------------------------------------------------------------
// Get GPU lights device pointer
//-----------------------------------------------------------------------------
GPULight *GetGPULights() { return s_deviceLights; }

//-----------------------------------------------------------------------------
// Get host light by index (for falloff calculations on CPU)
//-----------------------------------------------------------------------------
const GPULight *GetHostLight(int index) {
  if (index >= 0 && index < s_numLights) {
    return &s_hostLights[index];
  }
  return nullptr;
}

//=============================================================================
// GPU Scene Data Upload (Phase 1 infrastructure)
//=============================================================================

static GPUSampleData *s_deviceSamples = nullptr;
static GPUFaceInfo *s_deviceFaceInfos = nullptr;
static GPUClusterLightList *s_deviceClusterLists = nullptr;
static int *s_deviceClusterLightIndices = nullptr;
static int s_numSamples = 0;
static int s_numFaceInfos = 0;
static int s_numClusterLists = 0;
static int s_numClusterLightEntries = 0;
static bool s_sceneDataUploaded = false;

static GPULightOutput *s_deviceLightOutput = nullptr;
static int s_lightOutputCount = 0;

// Helper: allocate + upload a typed array to VRAM.
// Returns device pointer on success, nullptr on failure.
template <typename T>
static T *CudaUpload(const T *hostData, int count, const char *label) {
  if (count <= 0 || !hostData)
    return nullptr;

  T *devicePtr = nullptr;
  size_t bytes = (size_t)count * sizeof(T);
  cudaError_t err = cudaMalloc(&devicePtr, bytes);
  if (err != cudaSuccess) {
    printf("UploadGPUSceneData: cudaMalloc(%s, %.2f MB) failed: %s\n", label,
           bytes / (1024.0 * 1024.0), cudaGetErrorString(err));
    return nullptr;
  }

  err = cudaMemcpy(devicePtr, hostData, bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf("UploadGPUSceneData: cudaMemcpy(%s) failed: %s\n", label,
           cudaGetErrorString(err));
    cudaFree(devicePtr);
    return nullptr;
  }

  printf("  %-28s %10d items  %8.2f MB\n", label, count,
         bytes / (1024.0 * 1024.0));
  return devicePtr;
}

void UploadGPUSceneData(const GPUSampleData *samples, int numSamples,
                        const GPUFaceInfo *faceInfos, int numFaces,
                        const GPUClusterLightList *clusterLists,
                        int numClusters, const int *clusterLightIndices,
                        int numClusterLightEntries) {
  if (s_sceneDataUploaded) {
    ShutdownGPUSceneData();
  }

  printf("UploadGPUSceneData: Uploading to VRAM...\n");

  s_deviceSamples = CudaUpload(samples, numSamples, "GPUSampleData");
  s_deviceFaceInfos = CudaUpload(faceInfos, numFaces, "GPUFaceInfo");
  s_deviceClusterLists =
      CudaUpload(clusterLists, numClusters, "GPUClusterLightList");
  s_deviceClusterLightIndices = CudaUpload(
      clusterLightIndices, numClusterLightEntries, "ClusterLightIdx");

  s_numSamples = numSamples;
  s_numFaceInfos = numFaces;
  s_numClusterLists = numClusters;
  s_numClusterLightEntries = numClusterLightEntries;

  // Compute total VRAM used by scene data
  size_t totalBytes = (size_t)numSamples * sizeof(GPUSampleData) +
                      (size_t)numFaces * sizeof(GPUFaceInfo) +
                      (size_t)numClusters * sizeof(GPUClusterLightList) +
                      (size_t)numClusterLightEntries * sizeof(int);

  printf("UploadGPUSceneData: Total VRAM for scene data: %.2f MB\n",
         totalBytes / (1024.0 * 1024.0));
  printf("  %d samples, %d faces, %d clusters, %d cluster-light entries\n",
         numSamples, numFaces, numClusters, numClusterLightEntries);

  s_sceneDataUploaded = true;
}

void AllocateDirectLightingOutput(int numSamples) {
  if (s_deviceLightOutput) {
    cudaFree(s_deviceLightOutput);
    s_deviceLightOutput = nullptr;
  }

  s_lightOutputCount = numSamples;
  size_t bytes = (size_t)numSamples * sizeof(GPULightOutput);
  cudaError_t err = cudaMalloc(&s_deviceLightOutput, bytes);
  if (err != cudaSuccess) {
    printf("AllocateDirectLightingOutput: cudaMalloc(%.2f MB) failed: %s\n",
           bytes / (1024.0 * 1024.0), cudaGetErrorString(err));
    s_deviceLightOutput = nullptr;
    return;
  }

  // Initialize: zero all r/g/b accumulators, set styleMap to -1 (unused).
  // cudaMemset(0) alone won't work since -1 != 0 for styleMap entries.
  GPULightOutput *hostInit = new GPULightOutput[numSamples];
  memset(hostInit, 0, bytes);
  for (int i = 0; i < numSamples; i++) {
    for (int s = 0; s < GPU_MAXLIGHTMAPS; s++) {
      hostInit[i].styleMap[s] = -1;
    }
  }
  err =
      cudaMemcpy(s_deviceLightOutput, hostInit, bytes, cudaMemcpyHostToDevice);
  delete[] hostInit;
  if (err != cudaSuccess) {
    printf("AllocateDirectLightingOutput: cudaMemcpy init failed: %s\n",
           cudaGetErrorString(err));
  }

  printf("AllocateDirectLightingOutput: %d samples (%.2f MB)\n", numSamples,
         bytes / (1024.0 * 1024.0));
}

void DownloadDirectLightingOutput(GPULightOutput *hostBuffer, int numSamples) {
  if (!s_deviceLightOutput || numSamples <= 0)
    return;

  size_t bytes = (size_t)numSamples * sizeof(GPULightOutput);
  cudaError_t err = cudaMemcpy(hostBuffer, s_deviceLightOutput, bytes,
                               cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("DownloadDirectLightingOutput: cudaMemcpy failed: %s\n",
           cudaGetErrorString(err));
  }
}

void ShutdownGPUSceneData() {
  if (s_deviceSamples) {
    cudaFree(s_deviceSamples);
    s_deviceSamples = nullptr;
  }
  if (s_deviceFaceInfos) {
    cudaFree(s_deviceFaceInfos);
    s_deviceFaceInfos = nullptr;
  }
  if (s_deviceClusterLists) {
    cudaFree(s_deviceClusterLists);
    s_deviceClusterLists = nullptr;
  }
  if (s_deviceClusterLightIndices) {
    cudaFree(s_deviceClusterLightIndices);
    s_deviceClusterLightIndices = nullptr;
  }
  if (s_deviceLightOutput) {
    cudaFree(s_deviceLightOutput);
    s_deviceLightOutput = nullptr;
  }
  s_numSamples = 0;
  s_numFaceInfos = 0;
  s_numClusterLists = 0;
  s_numClusterLightEntries = 0;
  s_lightOutputCount = 0;
  s_sceneDataUploaded = false;
}

int GetGPUSampleCount() { return s_numSamples; }
int GetGPUFaceInfoCount() { return s_numFaceInfos; }
int GetGPUClusterCount() { return s_numClusterLists; }

GPUSampleData *GetDeviceSamples() { return s_deviceSamples; }
GPUFaceInfo *GetDeviceFaceInfos() { return s_deviceFaceInfos; }
GPUClusterLightList *GetDeviceClusterLightLists() {
  return s_deviceClusterLists;
}
int *GetDeviceClusterLightIndices() { return s_deviceClusterLightIndices; }
GPULightOutput *GetDeviceDirectLightingOutput() { return s_deviceLightOutput; }

//=============================================================================
// SS Sub-Position Upload — saves/restores original scene samples
//=============================================================================
static GPUSampleData *s_savedOriginalSamples = nullptr;
static int s_savedOriginalSampleCount = 0;

void UploadSSSubPositions(const GPUSampleData *subPositions, int count) {
  // Save original device samples pointer (first time only)
  if (!s_savedOriginalSamples) {
    s_savedOriginalSamples = s_deviceSamples;
    s_savedOriginalSampleCount = s_numSamples;
  } else {
    // Free previous SS upload (but not the saved original)
    if (s_deviceSamples && s_deviceSamples != s_savedOriginalSamples) {
      cudaFree(s_deviceSamples);
    }
  }

  // Upload SS sub-positions as the new device samples
  s_deviceSamples = CudaUpload(subPositions, count, "SSSubPositions");
  s_numSamples = count;
}

void RestoreOriginalSamples() {
  if (s_savedOriginalSamples) {
    // Free the current SS samples (if different from original)
    if (s_deviceSamples && s_deviceSamples != s_savedOriginalSamples) {
      cudaFree(s_deviceSamples);
    }
    s_deviceSamples = s_savedOriginalSamples;
    s_numSamples = s_savedOriginalSampleCount;
    s_savedOriginalSamples = nullptr;
    s_savedOriginalSampleCount = 0;
  }
}

#endif // VRAD_RTX_CUDA_SUPPORT
