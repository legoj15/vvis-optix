#ifndef RAYTRACE_SHARED_H
#define RAYTRACE_SHARED_H

// Shared structures for CPU/GPU ray tracing
// Compatible with both C++ and CUDA code

#ifdef __CUDACC__
// CUDA compilation path
#include <cuda_runtime.h>
typedef float3 float3_t;
#else
// C++ compilation path
struct float3_t {
  float x, y, z;
};
#endif

// KD-Tree Node Type Flags
#define KDNODE_STATE_XSPLIT 0
#define KDNODE_STATE_YSPLIT 1
#define KDNODE_STATE_ZSPLIT 2
#define KDNODE_STATE_LEAF 3

// Cache-optimized KD-Tree node structure
// Matches VRAD's CacheOptimizedKDNode
struct CUDAKDNode {
  int Children;     // lower 2 bits = split type, upper 30 bits = child/triangle
                    // index
  float SplitValue; // split position (for internal nodes) or triangle count
                    // (for leaves)
};

// Triangle intersection data structure
// Matches VRAD's TriIntersectData_t
struct CUDATriangle {
  // Plane equation: nx*x + ny*y + nz*z + d = 0
  float nx, ny, nz;
  float d;

  int triangle_id; // Original triangle ID (includes flags like TRACE_ID_SKY)

  // Edge equations for barycentric test (6 floats = 3 edges * 2 coefficients)
  float edge_eqs[6];

  // Coordinate selection for projection (0=YZ, 1=XZ, 2=XY)
  unsigned char coord_select0;
  unsigned char coord_select1;
  unsigned char flags;
  unsigned char unused;
};

// Sky face triangle ID flag — matches TRACE_ID_SKY in vrad.h
#define TRACE_ID_SKY_GPU 0x01000000

// Ray input data for GPU tracing
struct RayBatch {
  float3_t origin;
  float3_t direction;
  float tmin;
  float tmax;
  int skip_id; // Triangle ID to skip (for self-intersection avoidance)
};

// Ray intersection result
struct RayResult {
  float hit_t;     // Distance to intersection (-1 if no hit)
  int hit_id;      // Triangle index that was hit (-1 if no hit)
  float3_t normal; // Surface normal at hit point
};

// Per-triangle texture shadow data (only for FCACHETRI_TRANSPARENT triangles)
// Maps material entry index → UV coords + alpha texture atlas location
struct GPUTextureShadowTri {
  float u0, v0;    // UV for vertex 0
  float u1, v1;    // UV for vertex 1
  float u2, v2;    // UV for vertex 2
  int atlasOffset; // Byte offset into flattened alpha texture atlas
  short texWidth;
  short texHeight;
};

// Parameters for CUDA kernel launches
struct CUDATraceParams {
  const CUDAKDNode *nodes;
  const int *triangle_indices;
  const CUDATriangle *triangles;
  const RayBatch *rays;
  RayResult *results;
  int num_rays;
  float3_t scene_min;
  float3_t scene_max;
};

// Parameters for bounce GatherLight CUDA kernel (CSR format)
struct BounceGatherParams {
  // CSR transfer lists
  const long long *csrOffsets; // [numPatches+1] - start of each patch's
                               // transfers (long long to support >2B transfers)
  const int *csrPatch;         // [totalTransfers] - source patch index
  const float *csrWeight; // [totalTransfers] - transfer weight (normalized)

  // Per-patch data (read-only during bounces)
  const float3_t
      *emitlight; // [numPatches] - emitted light (updated per bounce)
  const float3_t *reflectivity; // [numPatches] - surface reflectivity
  const float3_t *patchOrigin;  // [numPatches] - patch origin position
  const float3_t *patchNormal;  // [numPatches] - patch flat normal

  // Output
  float3_t *addlight; // [numPatches] - accumulated bounce light (non-bump)

  // Bump-mapped patch support
  const int
      *needsBumpmap;     // [numPatches] - 1 if patch needs bumpmap, 0 otherwise
  const int *faceNumber; // [numPatches] - face index for bump normal lookup
  float3_t *
      addlightBump; // [numPatches * 3] - bump light for 3 bump vecs (if needed)

  int numPatches;
  int patchOffset; // Global offset: kernel thread j maps to global patch
                   // (j + patchOffset). Used for chunked streaming where
                   // patchOrigin/patchNormal are global arrays but the kernel
                   // processes a subset of patches.
};

#endif // RAYTRACE_SHARED_H
