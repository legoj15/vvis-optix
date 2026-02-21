#line 1
#ifdef VRAD_RTX_CUDA_SUPPORT
#include "raytrace.h"
#include "raytrace_cuda.h"
#include "tier0/dbg.h"
#include "tier1/strtools.h"
#include "tier1/utlvector.h"
#include <cuda_runtime.h>

// External CUDA kernel launcher (defined in raytrace_cuda.cu)
extern "C" void LaunchRayTraceBatchKernel(const CUDATraceParams &params,
                                          int block_size, int grid_size);

// Static member initialization
bool RayTraceGPU::s_bInitialized = false;
char RayTraceGPU::s_szDeviceName[256] = {0};
int RayTraceGPU::s_nDeviceMemoryMB = 0;

CUDAKDNode *RayTraceGPU::s_d_nodes = nullptr;
CUDATriangle *RayTraceGPU::s_d_triangles = nullptr;
int *RayTraceGPU::s_d_triangle_indices = nullptr;

float3_t RayTraceGPU::s_scene_min = {0, 0, 0};
float3_t RayTraceGPU::s_scene_max = {0, 0, 0};

RayBatch *RayTraceGPU::s_d_rays = nullptr;
RayResult *RayTraceGPU::s_d_results = nullptr;
int RayTraceGPU::s_max_batch_size = 0;
CThreadMutex RayTraceGPU::s_Mutex;

//-----------------------------------------------------------------------------
// Helper: Check CUDA errors
//-----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      Warning("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,                \
              cudaGetErrorString(err));                                        \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_VOID(call)                                                  \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      Warning("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,                \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

//-----------------------------------------------------------------------------
// Initialize GPU
//-----------------------------------------------------------------------------
bool RayTraceGPU::Initialize() {
  if (s_bInitialized) {
    return true;
  }

  // Check CUDA device availability
  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0) {
    Warning("No CUDA-capable GPU found!\n");
    return false;
  }

  // Use device 0
  CUDA_CHECK(cudaSetDevice(0));

  // Get device properties
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  V_strncpy(s_szDeviceName, prop.name, sizeof(s_szDeviceName));
  s_nDeviceMemoryMB = (int)(prop.totalGlobalMem / (1024 * 1024));

  Msg("GPU Ray Tracing Initialized\n");
  Msg("  Device: %s\n", s_szDeviceName);
  Msg("  Memory: %d MB\n", s_nDeviceMemoryMB);
  Msg("  Compute Capability: %d.%d\n", prop.major, prop.minor);

  // Allocate persistent batch buffers (500k rays initially)
  s_max_batch_size = 500000;
  CUDA_CHECK(cudaMalloc(&s_d_rays, s_max_batch_size * sizeof(RayBatch)));
  CUDA_CHECK(cudaMalloc(&s_d_results, s_max_batch_size * sizeof(RayResult)));

  s_bInitialized = true;
  return true;
}

//-----------------------------------------------------------------------------
// Build scene on GPU
//-----------------------------------------------------------------------------
void RayTraceGPU::BuildScene(
    const CUtlBlockVector<CacheOptimizedTriangle> &triangles,
    const CUtlVector<CacheOptimizedKDNode> &nodes,
    const CUtlVector<int32> &triangle_indices, const Vector &scene_min,
    const Vector &scene_max) {
  if (!s_bInitialized) {
    Warning("RayTraceGPU::BuildScene called before Initialize!\n");
    return;
  }

  Msg("Building GPU acceleration structure...\n");

  // Store scene bounds
  s_scene_min.x = scene_min.x;
  s_scene_min.y = scene_min.y;
  s_scene_min.z = scene_min.z;

  s_scene_max.x = scene_max.x;
  s_scene_max.y = scene_max.y;
  s_scene_max.z = scene_max.z;

  // Free old scene data if exists
  if (s_d_nodes)
    cudaFree(s_d_nodes);
  if (s_d_triangles)
    cudaFree(s_d_triangles);
  if (s_d_triangle_indices)
    cudaFree(s_d_triangle_indices);

  // Upload KD-tree nodes
  int node_count = nodes.Count();
  CUDA_CHECK_VOID(cudaMalloc(&s_d_nodes, node_count * sizeof(CUDAKDNode)));

  CUDAKDNode *h_nodes = new CUDAKDNode[node_count];
  for (int i = 0; i < node_count; i++) {
    h_nodes[i].Children = nodes[i].Children;
    h_nodes[i].SplitValue = nodes[i].SplittingPlaneValue;
  }
  CUDA_CHECK_VOID(cudaMemcpy(s_d_nodes, h_nodes,
                             node_count * sizeof(CUDAKDNode),
                             cudaMemcpyHostToDevice));
  delete[] h_nodes;

  // Upload triangle indices
  int index_count = triangle_indices.Count();
  CUDA_CHECK_VOID(cudaMalloc(&s_d_triangle_indices, index_count * sizeof(int)));
  CUDA_CHECK_VOID(cudaMemcpy(s_d_triangle_indices, triangle_indices.Base(),
                             index_count * sizeof(int),
                             cudaMemcpyHostToDevice));

  // Upload triangles (convert to CUDA format)
  int triangle_count = triangles.Count();
  CUDA_CHECK_VOID(
      cudaMalloc(&s_d_triangles, triangle_count * sizeof(CUDATriangle)));

  CUDATriangle *h_triangles = new CUDATriangle[triangle_count];
  for (int i = 0; i < triangle_count; i++) {
    const TriIntersectData_t &tri = triangles[i].m_Data.m_IntersectData;

    h_triangles[i].nx = tri.m_flNx;
    h_triangles[i].ny = tri.m_flNy;
    h_triangles[i].nz = tri.m_flNz;
    h_triangles[i].d = tri.m_flD;
    h_triangles[i].triangle_id = tri.m_nTriangleID;

    for (int j = 0; j < 6; j++) {
      h_triangles[i].edge_eqs[j] = tri.m_ProjectedEdgeEquations[j];
    }

    h_triangles[i].coord_select0 = tri.m_nCoordSelect0;
    h_triangles[i].coord_select1 = tri.m_nCoordSelect1;
    h_triangles[i].flags = tri.m_nFlags;
    h_triangles[i].unused = 0;
  }
  CUDA_CHECK_VOID(cudaMemcpy(s_d_triangles, h_triangles,
                             triangle_count * sizeof(CUDATriangle),
                             cudaMemcpyHostToDevice));
  delete[] h_triangles;

  Msg("  Nodes: %d\n", node_count);
  Msg("  Triangles: %d\n", triangle_count);
  Msg("  Indices: %d\n", index_count);
  Msg("GPU scene upload complete.\n");
}

//-----------------------------------------------------------------------------
// Trace batch of rays
//-----------------------------------------------------------------------------
void RayTraceGPU::TraceBatch(const RayBatch *rays, RayResult *results,
                             int ray_count) {
  if (!s_bInitialized) {
    Warning("RayTraceGPU::TraceBatch called before Initialize!\n");
    return;
  }

  if (ray_count == 0) {
    return;
  }

  AUTO_LOCK(s_Mutex);

  // Handle batches larger than our buffer
  int offset = 0;
  while (offset < ray_count) {
    int batch = min(ray_count - offset, s_max_batch_size);

    // Upload rays
    CUDA_CHECK_VOID(cudaMemcpy(s_d_rays, rays + offset,
                               batch * sizeof(RayBatch),
                               cudaMemcpyHostToDevice));

    // Launch kernel
    CUDATraceParams params;
    params.nodes = s_d_nodes;
    params.triangle_indices = s_d_triangle_indices;
    params.triangles = s_d_triangles;
    params.rays = s_d_rays;
    params.results = s_d_results;
    params.num_rays = batch;
    params.scene_min = s_scene_min;
    params.scene_max = s_scene_max;

    int block_size = 256;
    int grid_size = (batch + block_size - 1) / block_size;

    LaunchRayTraceBatchKernel(params, block_size, grid_size);

    // Download results
    CUDA_CHECK_VOID(cudaMemcpy(results + offset, s_d_results,
                               batch * sizeof(RayResult),
                               cudaMemcpyDeviceToHost));

    offset += batch;
  }

  // Ensure completion
  CUDA_CHECK_VOID(cudaDeviceSynchronize());
}

//-----------------------------------------------------------------------------
// Shutdown
//-----------------------------------------------------------------------------
void RayTraceGPU::Shutdown() {
  if (!s_bInitialized) {
    return;
  }

  if (s_d_nodes)
    cudaFree(s_d_nodes);
  if (s_d_triangles)
    cudaFree(s_d_triangles);
  if (s_d_triangle_indices)
    cudaFree(s_d_triangle_indices);
  if (s_d_rays)
    cudaFree(s_d_rays);
  if (s_d_results)
    cudaFree(s_d_results);

  s_d_nodes = nullptr;
  s_d_triangles = nullptr;
  s_d_triangle_indices = nullptr;
  s_d_rays = nullptr;
  s_d_results = nullptr;

  cudaDeviceReset();

  s_bInitialized = false;

  Msg("GPU Ray Tracing shut down.\n");
}
#endif
