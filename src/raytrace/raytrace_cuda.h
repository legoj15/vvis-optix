#ifndef RAYTRACE_CUDA_H
#define RAYTRACE_CUDA_H

#include "mathlib/mathlib.h"
#include "raytrace_shared.h"
#include "tier0/basetypes.h"
#include "tier0/threadtools.h"
#include "tier1/utlvector.h"

// Forward declarations
struct CacheOptimizedTriangle;
struct CacheOptimizedKDNode;

//-----------------------------------------------------------------------------
// GPU Ray Tracing Interface
// Provides hardware-accelerated ray tracing using CUDA
//-----------------------------------------------------------------------------
class RayTraceGPU {
public:
  // Initialize CUDA context and check device capabilities
  // Returns true on success, false if GPU is not available
  static bool Initialize();

  // Upload scene geometry to GPU memory
  // Must be called after Initialize() and before TraceBatch()
  static void
  BuildScene(const CUtlBlockVector<CacheOptimizedTriangle> &triangles,
             const CUtlVector<CacheOptimizedKDNode> &nodes,
             const CUtlVector<int32> &triangle_indices, const Vector &scene_min,
             const Vector &scene_max);

  // Trace a batch of rays on the GPU (synchronous)
  // rays: Input ray array
  // results: Output intersection results (must be pre-allocated)
  // ray_count: Number of rays to trace
  static void TraceBatch(const RayBatch *rays, RayResult *results,
                         int ray_count);

  // Clean up GPU resources
  static void Shutdown();

  // Query GPU status
  static bool IsInitialized() { return s_bInitialized; }
  static const char *GetDeviceName() { return s_szDeviceName; }
  static int GetDeviceMemoryMB() { return s_nDeviceMemoryMB; }

private:
  static bool s_bInitialized;
  static char s_szDeviceName[256];
  static int s_nDeviceMemoryMB;

  // Device memory pointers
  static CUDAKDNode *s_d_nodes;
  static CUDATriangle *s_d_triangles;
  static int *s_d_triangle_indices;

  // Scene bounds
  static float3_t s_scene_min;
  static float3_t s_scene_max;

  // Temporary batch buffers
  static RayBatch *s_d_rays;
  static RayResult *s_d_results;
  static int s_max_batch_size;

  // Mutex for thread-safe access to GPU buffers
  static CThreadMutex s_Mutex;
};

#endif // RAYTRACE_CUDA_H
