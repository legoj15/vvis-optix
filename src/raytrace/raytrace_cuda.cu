#include "raytrace_shared.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_STACK_DEPTH 64
#define EPSILON 1e-10f // Match CPU's FourEpsilons value

//-----------------------------------------------------------------------------
// Device helper functions
//-----------------------------------------------------------------------------
__device__ inline float3_t make_float3_t(float x, float y, float z) {
  float3_t v;
  v.x = x;
  v.y = y;
  v.z = z;
  return v;
}

__device__ inline float3_t operator-(const float3_t &a, const float3_t &b) {
  return make_float3_t(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float dot(const float3_t &a, const float3_t &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float get_coord(const float3_t &v, int axis) {
  if (axis == 0)
    return v.x;
  if (axis == 1)
    return v.y;
  return v.z;
}

//-----------------------------------------------------------------------------
// Single ray KD-tree traversal
//-----------------------------------------------------------------------------
__device__ RayResult TraceSingleRayDevice(
    const RayBatch &ray, const CUDAKDNode *nodes, const int *triangle_indices,
    const CUDATriangle *triangles, float3_t scene_min, float3_t scene_max) {
  RayResult result;
  result.hit_t = 1e30f; // Initialize to large value for closest-hit logic
  result.hit_id = -1;
  result.normal = make_float3_t(0, 0, 0);

  // Compute robust inverse direction
  float3_t invDir;
  invDir.x = 1.0f / (fabsf(ray.direction.x) < 1e-12f
                         ? (ray.direction.x < 0 ? -1e-12f : 1e-12f)
                         : ray.direction.x);
  invDir.y = 1.0f / (fabsf(ray.direction.y) < 1e-12f
                         ? (ray.direction.y < 0 ? -1e-12f : 1e-12f)
                         : ray.direction.y);
  invDir.z = 1.0f / (fabsf(ray.direction.z) < 1e-12f
                         ? (ray.direction.z < 0 ? -1e-12f : 1e-12f)
                         : ray.direction.z);

  // Initial ray-box intersection against scene bounds
  float tmin = ray.tmin;
  float tmax = ray.tmax;

  for (int i = 0; i < 3; i++) {
    float bmin = get_coord(scene_min, i);
    float bmax = get_coord(scene_max, i);
    float rorg = get_coord(ray.origin, i);
    float rinv = get_coord(invDir, i);

    float t0 = (bmin - rorg) * rinv;
    float t1 = (bmax - rorg) * rinv;

    if (rinv < 0.0f) {
      float tmp = t0;
      t0 = t1;
      t1 = tmp;
    }

    tmin = fmaxf(tmin, t0);
    tmax = fminf(tmax, t1);
  }

  if (tmin > tmax) {
    return result; // Ray misses scene bounds
  }

  // KD-tree traversal stack
  struct StackNode {
    int nodeIdx;
    float tmin, tmax;
  };
  StackNode stack[MAX_STACK_DEPTH];
  int stackPtr = 0;

  int currNodeIdx = 0;
  float curr_tmin = tmin;
  float curr_tmax = tmax;

  while (true) {
    // Load current node
    int nodeChildren = nodes[currNodeIdx].Children;
    float nodeSplitValue = nodes[currNodeIdx].SplitValue;
    int type = nodeChildren & 3;

    if (type == KDNODE_STATE_LEAF) {
      // Leaf node: intersect triangles
      int triCount = __float_as_int(nodeSplitValue);
      int triStart = nodeChildren >> 2;

      for (int i = 0; i < triCount; i++) {
        int triIdx = __ldg(&triangle_indices[triStart + i]);
        const CUDATriangle *pTri = &triangles[triIdx];

        int tri_id = __ldg(&pTri->triangle_id);
        if (tri_id == ray.skip_id) {
          continue;
        }

        // Ray-plane intersection
        float nx = __ldg(&pTri->nx);
        float ny = __ldg(&pTri->ny);
        float nz = __ldg(&pTri->nz);
        float d = __ldg(&pTri->d);

        float3_t normal = make_float3_t(nx, ny, nz);
        float denom = dot(ray.direction, normal);

        if (fabsf(denom) < EPSILON) {
          continue; // Ray parallel to triangle
        }

        float isect_t = (d - dot(ray.origin, normal)) / denom;

        // CPU checks: isect_t > 0 AND isect_t < closest_hit
        // This ensures hit is in front of ray and closer than current best
        if (isect_t <= 0.0f) {
          continue; // Behind ray origin
        }

        if (isect_t >= result.hit_t) {
          continue; // Already have closer hit
        }

        // Compute hit point
        float3_t hitPoint;
        hitPoint.x = ray.origin.x + isect_t * ray.direction.x;
        hitPoint.y = ray.origin.y + isect_t * ray.direction.y;
        hitPoint.z = ray.origin.z + isect_t * ray.direction.z;

        // Project to 2D for edge tests
        unsigned char c0 = __ldg(&pTri->coord_select0);
        unsigned char c1 = __ldg(&pTri->coord_select1);

        float hitc0 = get_coord(hitPoint, c0);
        float hitc1 = get_coord(hitPoint, c1);

        // Edge equation tests (barycentric) - CPU uses >= 0, so we do the same
        float e0 = __ldg(&pTri->edge_eqs[0]);
        float e1 = __ldg(&pTri->edge_eqs[1]);
        float e2 = __ldg(&pTri->edge_eqs[2]);
        float B0 = e0 * hitc0 + e1 * hitc1 + e2;
        if (B0 < 0.0f) {
          continue;
        }

        float e3 = __ldg(&pTri->edge_eqs[3]);
        float e4 = __ldg(&pTri->edge_eqs[4]);
        float e5 = __ldg(&pTri->edge_eqs[5]);
        float B1 = e3 * hitc0 + e4 * hitc1 + e5;
        if (B1 < 0.0f) {
          continue;
        }

        // CPU: B0 + B1 <= 1
        if (B0 + B1 > 1.0f) {
          continue;
        }

        // Valid hit!
        result.hit_t = isect_t;
        result.hit_id = triIdx;
        result.normal = normal;
      }

      // Early out if we found a hit in the current leaf that's closer than
      // remaining nodes
      if (result.hit_id != -1 && result.hit_t <= curr_tmax) {
        // Wait, this is subtle. We can't break if result.hit_t > curr_tmin
        // because there might be other triangles in the same leaf or preceding
        // leaves. But we visit nodes front-to-back. So if we hit something in
        // the leaf and it's behind the current TMax, we don't need to check any
        // nodes beyond TMax. Correct check is at the pop step.
      }

      // Pop from stack
      if (stackPtr == 0)
        break;
      StackNode sn = stack[--stackPtr];
      currNodeIdx = sn.nodeIdx;
      curr_tmin = sn.tmin;
      curr_tmax = sn.tmax;

      // Early exit if we have a hit closer than this node's entry point
      if (result.hit_id != -1 && result.hit_t <= curr_tmin)
        break;
      continue;
    }

    // Internal node: traverse children
    int axis = type;
    float splitVal = nodeSplitValue;
    int leftIdx = nodeChildren >> 2;
    int rightIdx = leftIdx + 1;

    float rdir = get_coord(ray.direction, axis);
    float tSplit =
        (splitVal - get_coord(ray.origin, axis)) * get_coord(invDir, axis);

    int frontChild, backChild;
    if (rdir < 0.0f) {
      frontChild = rightIdx;
      backChild = leftIdx;
    } else {
      frontChild = leftIdx;
      backChild = rightIdx;
    }

    if (tSplit > curr_tmax) {
      // Split plane is beyond exit - only front child is relevant
      currNodeIdx = frontChild;
    } else if (tSplit < curr_tmin) {
      // Split plane is before entry - only back child is relevant
      currNodeIdx = backChild;
    } else {
      // Straddling - must visit both. Push back, visit front first.
      if (stackPtr < MAX_STACK_DEPTH) {
        stack[stackPtr++] = {backChild, tSplit, curr_tmax};
      }
      currNodeIdx = frontChild;
      curr_tmax = tSplit;
    }
  }

  return result;
}

//-----------------------------------------------------------------------------
// CUDA Kernel: Trace batch of rays
//-----------------------------------------------------------------------------
__global__ void RayTraceBatchKernel(CUDATraceParams params) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= params.num_rays) {
    return;
  }

  const RayBatch ray = params.rays[idx];

  RayResult result = TraceSingleRayDevice(
      ray, params.nodes, params.triangle_indices, params.triangles,
      params.scene_min, params.scene_max);

  params.results[idx] = result;
}

//-----------------------------------------------------------------------------
// Host-callable kernel launcher
//-----------------------------------------------------------------------------
extern "C" void LaunchRayTraceBatchKernel(const CUDATraceParams &params,
                                          int block_size, int grid_size) {
  RayTraceBatchKernel<<<grid_size, block_size>>>(params);
}

//-----------------------------------------------------------------------------
// Bounce GatherLight CUDA Kernels
//-----------------------------------------------------------------------------

// Non-bump-mapped patches: simple weighted accumulation
__global__ void GatherLightKernel(BounceGatherParams params) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= params.numPatches)
    return;

  // Global patch index for per-patch array lookups
  int gj = j + params.patchOffset;

  // Skip bump-mapped patches (handled by bump kernel)
  if (params.needsBumpmap[gj])
    return;

  // CSR is chunk-local: indexed by j
  long long start = params.csrOffsets[j];
  long long end = params.csrOffsets[j + 1];

  float sx = 0.0f, sy = 0.0f, sz = 0.0f;

  for (long long k = start; k < end; k++) {
    int srcPatch = params.csrPatch[k];
    float weight = params.csrWeight[k];

    // v = emitlight[srcPatch] * reflectivity[srcPatch]
    float3_t emit = params.emitlight[srcPatch];
    float3_t refl = params.reflectivity[srcPatch];

    sx += emit.x * refl.x * weight;
    sy += emit.y * refl.y * weight;
    sz += emit.z * refl.z * weight;
  }

  float3_t result;
  result.x = sx;
  result.y = sy;
  result.z = sz;
  params.addlight[gj] = result;
}

// Bump-mapped patches: needs delta direction and dot products against normals
// This kernel handles patches with needsBumpmap=1
// Note: bump normals are precomputed and stored in patchBumpNormals
// For simplicity, we approximate by using the same transfer weight for all
// bump directions, scaled by the dot product of delta with each normal.
// The CPU version re-normalizes the transfer by dividing out the flat normal
// dot and multiplying by each bump normal dot.
__global__ void
GatherLightBumpKernel(BounceGatherParams params,
                      const float3_t *patchBumpNormals // [totalPatches * 4]
) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= params.numPatches)
    return;

  // Global patch index for per-patch array lookups
  int gj = j + params.patchOffset;

  // Only process bump-mapped patches
  if (!params.needsBumpmap[gj])
    return;

  // CSR is chunk-local: indexed by j
  long long start = params.csrOffsets[j];
  long long end = params.csrOffsets[j + 1];

  // Self patch data: use global index gj
  float3_t patchOrig = params.patchOrigin[gj];
  float3_t patchNrm = params.patchNormal[gj];

  // Load the 4 normals for this patch: use global index gj
  float3_t normals[4];
  for (int n = 0; n < 4; n++) {
    normals[n] = patchBumpNormals[gj * 4 + n];
  }

  // Accumulators for 4 bump directions
  float bsx[4] = {0, 0, 0, 0};
  float bsy[4] = {0, 0, 0, 0};
  float bsz[4] = {0, 0, 0, 0};

  for (long long k = start; k < end; k++) {
    int srcPatch = params.csrPatch[k];
    float weight = params.csrWeight[k];

    // Source patch data: srcPatch is already a global index
    float3_t srcOrig = params.patchOrigin[srcPatch];
    float3_t emit = params.emitlight[srcPatch];
    float3_t refl = params.reflectivity[srcPatch];

    // v = emitlight * reflectivity
    float vx = emit.x * refl.x;
    float vy = emit.y * refl.y;
    float vz = emit.z * refl.z;

    // delta = normalize(srcOrigin - patchOrigin)
    float dx = srcOrig.x - patchOrig.x;
    float dy = srcOrig.y - patchOrig.y;
    float dz = srcOrig.z - patchOrig.z;
    float invLen = rsqrtf(dx * dx + dy * dy + dz * dz + 1e-30f);
    dx *= invLen;
    dy *= invLen;
    dz *= invLen;

    // Remove normal already factored into transfer: scale = 1 / dot(delta,
    // patchNormal)
    float dnorm = dx * patchNrm.x + dy * patchNrm.y + dz * patchNrm.z;
    if (fabsf(dnorm) < 1e-10f)
      continue;
    float scale = weight / dnorm;

    float svx = vx * scale;
    float svy = vy * scale;
    float svz = vz * scale;

    // Accumulate for each bump normal direction
    for (int n = 0; n < 4; n++) {
      float d = dx * normals[n].x + dy * normals[n].y + dz * normals[n].z;
      if (d <= 0.0f)
        continue;
      bsx[n] += svx * d;
      bsy[n] += svy * d;
      bsz[n] += svz * d;
    }
  }

  // Write results using global index gj
  float3_t r0;
  r0.x = bsx[0];
  r0.y = bsy[0];
  r0.z = bsz[0];
  params.addlight[gj] = r0;

  // Write 3 bump results
  for (int n = 0; n < 3; n++) {
    float3_t r;
    r.x = bsx[n + 1];
    r.y = bsy[n + 1];
    r.z = bsz[n + 1];
    params.addlightBump[gj * 3 + n] = r;
  }
}

//-----------------------------------------------------------------------------
// Host-callable bounce kernel launchers
//-----------------------------------------------------------------------------
extern "C" void LaunchGatherLightKernel(const BounceGatherParams &params,
                                        int block_size) {
  int grid_size = (params.numPatches + block_size - 1) / block_size;
  GatherLightKernel<<<grid_size, block_size>>>(params);
}

extern "C" void LaunchGatherLightBumpKernel(const BounceGatherParams &params,
                                            const float3_t *patchBumpNormals,
                                            int block_size) {
  int grid_size = (params.numPatches + block_size - 1) / block_size;
  GatherLightBumpKernel<<<grid_size, block_size>>>(params, patchBumpNormals);
}
