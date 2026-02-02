#include "cuda_runtime.h"
#include "tier0/dbg.h"
#include "vis.h"
#include "vvis_cuda.h"
#include <algorithm>
#include <vector>

// Scale factor for float precision
#define VIS_EPSILON 0.01f

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      Msg("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,                     \
          cudaGetErrorString(err));                                            \
      exit(1);                                                                 \
    }                                                                          \
  }

// ----------------------------------------------------------------------------------
// Data Structures (Device & Host)
// ----------------------------------------------------------------------------------

#define MAX_POINTS_ON_WINDING_GPU 64
#define MAX_DEPTH_GPU 64

struct CUDAPortal {
  Vector origin;
  float radius;
  Vector planeNormal;
  float planeDist;
  int numPoints;
  int windingOffset; // Index into global g_dAllWindingPoints
  int leaf;          // The leaf this portal leads TO
};

struct CUDALeaf {
  int portalOffset; // Index into g_dLeafPortals
  int numPortals;
};

struct CUDAWinding {
  int numpoints;
  Vector points[MAX_POINTS_ON_WINDING_GPU];
};

struct CUDASeparator {
  Vector normal;
  float dist;
};

// ----------------------------------------------------------------------------------
// Global Device Pointers
// ----------------------------------------------------------------------------------

__constant__ int c_numPortals;
__constant__ int c_portalClusters;
__constant__ int c_portalBytes;
__constant__ float c_visRadiusClient;
__constant__ bool c_bUseRadiusClient;

static CUDAPortal *g_dPortals = nullptr;
static Vector *g_dWindingPoints = nullptr;
static CUDALeaf *g_dLeafs = nullptr;
static int *g_dLeafPortals = nullptr;

static unsigned char *g_dPortalFlood = nullptr;
static unsigned char *g_dPortalVis = nullptr;

static int *g_dStack_leafIdx = nullptr;
static int *g_dStack_lastPortalIdx = nullptr;
static int *g_dStack_currentPortalIter = nullptr;
static Vector *g_dStack_sourcePoints = nullptr;
static int *g_dStack_sourceNumPoints = nullptr;
static Vector *g_dStack_passPoints = nullptr;
static int *g_dStack_passNumPoints = nullptr;
static unsigned char *g_dStack_mightSee = nullptr;

// ----------------------------------------------------------------------------------
// Helper Math (Device)
// ----------------------------------------------------------------------------------

__device__ inline float DotProductGPU(const Vector &a, const Vector &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline void CrossProductGPU(const Vector &a, const Vector &b,
                                       Vector &result) {
  result.x = a.y * b.z - a.z * b.y;
  result.y = a.z * b.x - a.x * b.z;
  result.z = a.x * b.y - a.y * b.x;
}

__device__ inline void VectorSubtractGPU(const Vector &a, const Vector &b,
                                         Vector &c) {
  c.x = a.x - b.x;
  c.y = a.y - b.y;
  c.z = a.z - b.z;
}

__device__ inline void VectorCopyGPU(const Vector &src, Vector &dst) {
  dst.x = src.x;
  dst.y = src.y;
  dst.z = src.z;
}

__device__ inline void VectorScaleGPU(Vector &v, float s) {
  v.x *= s;
  v.y *= s;
  v.z *= s;
}

__device__ inline void VectorNegateGPU(Vector &v) {
  v.x = -v.x;
  v.y = -v.y;
  v.z = -v.z;
}

__device__ inline void VectorNormalizeGPU(Vector &v) {
  float len = sqrtf(DotProductGPU(v, v));
  if (len > 1e-6f)
    VectorScaleGPU(v, 1.0f / len);
}

__device__ bool BuildSeparatorGPU(const CUDAWinding *w1, const CUDAWinding *w2,
                                  const Vector &portalNormal,
                                  CUDASeparator *sep) {
  for (int i = 0; i < w1->numpoints; i++) {
    int i2 = (i + 1) % w1->numpoints;
    Vector v1, v2;
    VectorSubtractGPU(w1->points[i2], w1->points[i], v1);
    for (int j = 0; j < w2->numpoints; j++) {
      VectorSubtractGPU(w2->points[j], w1->points[i], v2);
      Vector normal;
      CrossProductGPU(v2, v1, normal);
      if (DotProductGPU(normal, normal) < 1e-4f)
        continue;
      VectorNormalizeGPU(normal);
      float dist = DotProductGPU(w1->points[i], normal);

      bool side1 = true;
      for (int k = 0; k < w1->numpoints; k++) {
        if (DotProductGPU(w1->points[k], normal) - dist < -VIS_EPSILON) {
          side1 = false;
          break;
        }
      }
      if (!side1) {
        VectorNegateGPU(normal);
        dist = -dist;
        side1 = true;
        for (int k = 0; k < w1->numpoints; k++) {
          if (DotProductGPU(w1->points[k], normal) - dist < -VIS_EPSILON) {
            side1 = false;
            break;
          }
        }
      }
      if (!side1)
        continue;

      bool side2 = true;
      for (int k = 0; k < w2->numpoints; k++) {
        if (DotProductGPU(w2->points[k], normal) - dist > VIS_EPSILON) {
          side2 = false;
          break;
        }
      }
      if (side2) {
        VectorCopyGPU(normal, sep->normal);
        sep->dist = dist;
        return true;
      }
    }
  }
  return false;
}

__device__ inline void CopyWindingGPU(const CUDAWinding *src,
                                      CUDAWinding *dst) {
  dst->numpoints = src->numpoints;
  for (int i = 0; i < src->numpoints; i++) {
    VectorCopyGPU(src->points[i], dst->points[i]);
  }
}

__device__ void ChopWindingGPU(const CUDAWinding *in, CUDAWinding *out,
                               const Vector &normal, float dist) {
  float dists[MAX_POINTS_ON_WINDING_GPU + 4];
  int sides[MAX_POINTS_ON_WINDING_GPU + 4];
  int counts[3] = {0, 0, 0};

  for (int i = 0; i < in->numpoints; i++) {
    float dot = DotProductGPU(in->points[i], normal) - dist;
    dists[i] = dot;
    if (dot > VIS_EPSILON)
      sides[i] = 0;
    else if (dot < -VIS_EPSILON)
      sides[i] = 1;
    else
      sides[i] = 2;
    counts[sides[i]]++;
  }

  if (!counts[0]) {
    out->numpoints = 0;
    return;
  }
  if (!counts[1]) {
    CopyWindingGPU(in, out);
    return;
  }

  out->numpoints = 0;
  sides[in->numpoints] = sides[0];
  dists[in->numpoints] = dists[0];

  for (int i = 0; i < in->numpoints; i++) {
    const Vector &p1 = in->points[i];
    if (sides[i] == 2) {
      if (out->numpoints < MAX_POINTS_ON_WINDING_GPU)
        VectorCopyGPU(p1, out->points[out->numpoints++]);
      continue;
    }
    if (sides[i] == 0) {
      if (out->numpoints < MAX_POINTS_ON_WINDING_GPU)
        VectorCopyGPU(p1, out->points[out->numpoints++]);
    }
    if (sides[i + 1] == 2 || sides[i + 1] == sides[i])
      continue;
    if (out->numpoints < MAX_POINTS_ON_WINDING_GPU) {
      const Vector &p2 = in->points[(i + 1) % in->numpoints];
      float dot = dists[i] / (dists[i] - dists[i + 1]);
      Vector mid;
      mid.x = p1.x + dot * (p2.x - p1.x);
      mid.y = p1.y + dot * (p2.y - p1.y);
      mid.z = p1.z + dot * (p2.z - p1.z);
      VectorCopyGPU(mid, out->points[out->numpoints++]);
    }
  }
}

// ----------------------------------------------------------------------------------
// Kernels
// ----------------------------------------------------------------------------------

__global__ void BasePortalVisKernel(const CUDAPortal *portals,
                                    const Vector *windingPoints,
                                    unsigned char *outVis, int numPortals,
                                    int portalBytes) {
  int p1Idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (p1Idx >= numPortals)
    return;

  const CUDAPortal *p1 = &portals[p1Idx];
  unsigned char *myVisRow = outVis + (size_t)p1Idx * portalBytes;

  for (int p2Idx = 0; p2Idx < numPortals; p2Idx++) {
    if (p1Idx == p2Idx)
      continue;
    const CUDAPortal *p2 = &portals[p2Idx];

    bool p2_hasFront = false;
    for (int k = 0; k < p2->numPoints; k++) {
      Vector pt;
      VectorCopyGPU(windingPoints[p2->windingOffset + k], pt);
      float d = DotProductGPU(pt, p1->planeNormal) - p1->planeDist;
      if (d > VIS_EPSILON) {
        p2_hasFront = true;
        break;
      }
    }
    if (!p2_hasFront)
      continue;

    bool p1_hasFront = false;
    for (int k = 0; k < p1->numPoints; k++) {
      Vector pt;
      VectorCopyGPU(windingPoints[p1->windingOffset + k], pt);
      float d = DotProductGPU(pt, p2->planeNormal) - p2->planeDist;
      if (d < -VIS_EPSILON) {
        p1_hasFront = true;
        break;
      }
    }
    if (!p1_hasFront)
      continue;

    myVisRow[p2Idx >> 3] |= (1 << (p2Idx & 7));
  }
}

__global__ void PortalFlowKernel(
    int numHostPortals, int portalBytes, CUDAPortal *portals, CUDALeaf *leafs,
    int *leafPortals, Vector *windingPoints, unsigned char *portalFlood,
    unsigned char *portalVis, int *s_leafIdx, int *s_lastPortalIdx,
    int *s_currentPortalIter, Vector *s_sourcePoints, int *s_sourceNumPoints,
    Vector *s_passPoints, int *s_passNumPoints, unsigned char *s_mightSee) {
  int srcPortalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (srcPortalIdx >= numHostPortals)
    return;

#define STACK_ADDR_ARG(d, n) ((d) * (n) + srcPortalIdx)

  const CUDAPortal *pSrc = &portals[srcPortalIdx];
  s_leafIdx[STACK_ADDR_ARG(0, numHostPortals)] = pSrc->leaf;
  s_lastPortalIdx[STACK_ADDR_ARG(0, numHostPortals)] = -1;
  s_currentPortalIter[STACK_ADDR_ARG(0, numHostPortals)] = 0;

  // Init MightSee (Depth 0)
  unsigned char *srcFlood = portalFlood + (size_t)srcPortalIdx * portalBytes;
  unsigned char *dstMightSee0 =
      s_mightSee + (size_t)STACK_ADDR_ARG(0, numHostPortals) * portalBytes;
  for (int b = 0; b < portalBytes; b++) {
    dstMightSee0[b] = srcFlood[b];
  }

  int srcNumPts = pSrc->numPoints;
  if (srcNumPts > MAX_POINTS_ON_WINDING_GPU)
    srcNumPts = MAX_POINTS_ON_WINDING_GPU;
  s_sourceNumPoints[STACK_ADDR_ARG(0, numHostPortals)] = srcNumPts;
  for (int i = 0; i < srcNumPts; i++)
    VectorCopyGPU(windingPoints[pSrc->windingOffset + i],
                  s_sourcePoints[STACK_ADDR_ARG(0, numHostPortals) *
                                     MAX_POINTS_ON_WINDING_GPU +
                                 i]);

  s_passNumPoints[STACK_ADDR_ARG(0, numHostPortals)] = srcNumPts;
  for (int i = 0; i < srcNumPts; i++)
    VectorCopyGPU(s_sourcePoints[STACK_ADDR_ARG(0, numHostPortals) *
                                     MAX_POINTS_ON_WINDING_GPU +
                                 i],
                  s_passPoints[STACK_ADDR_ARG(0, numHostPortals) *
                                   MAX_POINTS_ON_WINDING_GPU +
                               i]);

  unsigned char *floodBits = portalFlood + (size_t)srcPortalIdx * portalBytes;
  unsigned char *portalVisRow = portalVis + (size_t)srcPortalIdx * portalBytes;

  int depth = 0;
  while (depth >= 0) {
    int curAddr = STACK_ADDR_ARG(depth, numHostPortals);
    int leafIdx = s_leafIdx[curAddr];
    if (leafIdx < 0) {
      depth--;
      continue;
    }

    CUDALeaf leafData = leafs[leafIdx];
    int startI = s_currentPortalIter[curAddr];
    bool pushed = false;

    // Limit check
    if (depth >= MAX_DEPTH_GPU) {
      depth--;
      continue;
    }

    unsigned char *curMightSee = s_mightSee + (size_t)curAddr * portalBytes;

    for (int i = startI; i < leafData.numPortals; i++) {
      int pIdx = leafPortals[leafData.portalOffset + i];
      if (pIdx < 0 || pIdx >= numHostPortals || pIdx == srcPortalIdx)
        continue;
      if (pIdx == (s_lastPortalIdx[curAddr] ^ 1))
        continue;

      // Check bits in current MightSee
      if (!(curMightSee[pIdx >> 3] & (1 << (pIdx & 7))))
        continue;

      // PRUNING: Check if this branch can see anything NEW
      unsigned char *pPFlood = portalFlood + (size_t)pIdx * portalBytes;

      bool potential = false;
      unsigned char *nextMightSee = nullptr;
      if (depth < MAX_DEPTH_GPU - 1) {
        int nextAddr = STACK_ADDR_ARG(depth + 1, numHostPortals);
        nextMightSee = s_mightSee + (size_t)nextAddr * portalBytes;
      }

      // We process the loop to:
      // 1. Calculate nextMightSee (intersection of curMightSee and
      // p->portalflood)
      // 2. Check if (nextMightSee & ~portalVisRow) has any bits set.

      for (int b = 0; b < portalBytes; b++) {
        unsigned char m = curMightSee[b] & pPFlood[b];
        if (m & ~portalVisRow[b]) {
          potential = true;
        }
        if (nextMightSee)
          nextMightSee[b] = m;
      }

      // Counter: Considered (Pruning Check)
      // atomicAdd(&g_dDebugCounters[0], 1);

      if (!potential) {
        // Optimization: If we can't see anything new, skip geometric checks and
        // recursion
        continue;
      }

      // ... Geometric Checks ...

      const CUDAPortal *pP = &portals[pIdx];
      Vector clipNormal;
      VectorCopyGPU(pP->planeNormal, clipNormal);
      VectorNegateGPU(clipNormal);
      float clipDist = -pP->planeDist;

      CUDAWinding curSource;
      curSource.numpoints = s_sourceNumPoints[curAddr];
      for (int k = 0; k < curSource.numpoints; k++)
        VectorCopyGPU(s_sourcePoints[curAddr * MAX_POINTS_ON_WINDING_GPU + k],
                      curSource.points[k]);

      CUDAWinding nextSource;
      ChopWindingGPU(&curSource, &nextSource, clipNormal, clipDist);
      if (nextSource.numpoints == 0)
        continue;

      CUDAWinding nextPass;
      nextPass.numpoints = pP->numPoints;
      if (nextPass.numpoints > MAX_POINTS_ON_WINDING_GPU)
        nextPass.numpoints = MAX_POINTS_ON_WINDING_GPU;
      for (int k = 0; k < nextPass.numpoints; k++)
        VectorCopyGPU(windingPoints[pP->windingOffset + k], nextPass.points[k]);

      Vector rootNormal;
      VectorCopyGPU(pSrc->planeNormal, rootNormal);
      float rootDist = pSrc->planeDist;
      CUDAWinding tempPass;
      ChopWindingGPU(&nextPass, &tempPass, rootNormal, rootDist);
      CopyWindingGPU(&tempPass, &nextPass);
      if (nextPass.numpoints == 0)
        continue;

      if (depth > 0) {
        CUDAWinding rootW;
        int rootAddr0 = STACK_ADDR_ARG(0, numHostPortals);
        rootW.numpoints = s_sourceNumPoints[rootAddr0];
        if (rootW.numpoints > MAX_POINTS_ON_WINDING_GPU)
          rootW.numpoints = MAX_POINTS_ON_WINDING_GPU;
        for (int k = 0; k < rootW.numpoints; k++)
          VectorCopyGPU(
              s_sourcePoints[rootAddr0 * MAX_POINTS_ON_WINDING_GPU + k],
              rootW.points[k]);

        CUDAWinding prevPass;
        int passAddr = STACK_ADDR_ARG(depth, numHostPortals);
        prevPass.numpoints = s_passNumPoints[passAddr];
        if (prevPass.numpoints > MAX_POINTS_ON_WINDING_GPU)
          prevPass.numpoints = MAX_POINTS_ON_WINDING_GPU;
        for (int v = 0; v < prevPass.numpoints; v++)
          VectorCopyGPU(s_passPoints[passAddr * MAX_POINTS_ON_WINDING_GPU + v],
                        prevPass.points[v]);

        CUDASeparator sep;
        if (BuildSeparatorGPU(&rootW, &prevPass, rootNormal, &sep)) {
          CUDAWinding tempSource;
          ChopWindingGPU(&nextSource, &tempSource, sep.normal, sep.dist);
          CopyWindingGPU(&tempSource, &nextSource);
          if (nextSource.numpoints == 0)
            continue;
        }
      }

      portalVisRow[pIdx >> 3] |= (1 << (pIdx & 7));

      if (depth < MAX_DEPTH_GPU - 1) {
        int nextAddr = STACK_ADDR_ARG(depth + 1, numHostPortals);
        s_leafIdx[nextAddr] = pP->leaf;
        s_lastPortalIdx[nextAddr] = pIdx;
        s_currentPortalIter[curAddr] = i + 1;
        s_currentPortalIter[nextAddr] = 0;
        s_sourceNumPoints[nextAddr] = nextSource.numpoints;
        for (int k = 0; k < nextSource.numpoints; k++)
          VectorCopyGPU(
              nextSource.points[k],
              s_sourcePoints[nextAddr * MAX_POINTS_ON_WINDING_GPU + k]);
        s_passNumPoints[nextAddr] = nextPass.numpoints;
        for (int k = 0; k < nextPass.numpoints; k++)
          VectorCopyGPU(nextPass.points[k],
                        s_passPoints[nextAddr * MAX_POINTS_ON_WINDING_GPU + k]);
        depth++;
        pushed = true;
        break;
      }
    }
    if (!pushed)
      depth--;
  }
}

// ----------------------------------------------------------------------------------
// Host Interface
// ----------------------------------------------------------------------------------

void RunBasePortalVisCUDA() {
  Msg("Starting CUDA BasePortalVis...\n");
  int numPortals = g_numportals * 2;
  std::vector<CUDAPortal> hPortals(numPortals);
  std::vector<Vector> hWindingPoints;
  for (int i = 0; i < numPortals; i++) {
    hPortals[i].origin = portals[i].origin;
    hPortals[i].radius = portals[i].radius;
    hPortals[i].planeNormal = portals[i].plane.normal;
    hPortals[i].planeDist = portals[i].plane.dist;
    hPortals[i].numPoints = portals[i].winding->numpoints;
    hPortals[i].windingOffset = (int)hWindingPoints.size();
    hPortals[i].leaf = portals[i].leaf;
    for (int k = 0; k < portals[i].winding->numpoints; k++)
      hWindingPoints.push_back(portals[i].winding->points[k]);
  }
  CHECK_CUDA(cudaMalloc(&g_dPortals, numPortals * sizeof(CUDAPortal)));
  CHECK_CUDA(cudaMemcpy(g_dPortals, hPortals.data(),
                        numPortals * sizeof(CUDAPortal),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMalloc(&g_dWindingPoints, hWindingPoints.size() * sizeof(Vector)));
  CHECK_CUDA(cudaMemcpy(g_dWindingPoints, hWindingPoints.data(),
                        hWindingPoints.size() * sizeof(Vector),
                        cudaMemcpyHostToDevice));
  int visSize = numPortals * portalbytes;
  CHECK_CUDA(cudaMalloc(&g_dPortalVis, visSize));
  CHECK_CUDA(cudaMemset(g_dPortalVis, 0, visSize));
  float r = (float)g_VisRadius;
  cudaMemcpyToSymbol(c_visRadiusClient, &r, sizeof(float));
  cudaMemcpyToSymbol(c_bUseRadiusClient, &g_bUseRadius, sizeof(bool));
  cudaMemcpyToSymbol(c_numPortals, &numPortals, sizeof(int));
  cudaMemcpyToSymbol(c_portalBytes, &portalbytes, sizeof(int));
  cudaDeviceSetLimit(cudaLimitStackSize, 16384);
  int blockSize = 128;
  int numBlocks = (numPortals + blockSize - 1) / blockSize;
  BasePortalVisKernel<<<numBlocks, blockSize>>>(
      g_dPortals, g_dWindingPoints, g_dPortalVis, numPortals, portalbytes);
  cudaDeviceSynchronize();
  std::vector<unsigned char> hVis(visSize);
  cudaMemcpy(hVis.data(), g_dPortalVis, visSize, cudaMemcpyDeviceToHost);

  int totalBaseVis = 0;
  for (int i = 0; i < numPortals; i++) {
    if (!portals[i].portalfront) {
      portals[i].portalfront = (byte *)malloc(portalbytes);
      memset(portals[i].portalfront, 0, portalbytes);
    }
    memcpy(portals[i].portalfront, &hVis[i * portalbytes], portalbytes);

    // Count bits for debug
    for (int j = 0; j < portalbytes; j++) {
      unsigned char b = hVis[i * portalbytes + j];
      while (b) {
        if (b & 1)
          totalBaseVis++;
        b >>= 1;
      }
    }
  }
  Msg("CUDA BasePortalVis Total Bits Set: %d\n", totalBaseVis);

  cudaFree(g_dPortals);
  cudaFree(g_dWindingPoints);
  cudaFree(g_dPortalVis);
  g_dPortals = nullptr;
  g_dWindingPoints = nullptr;
  g_dPortalVis = nullptr;
  Msg("CUDA BasePortalVis Finished.\n");
}

void RunPortalFlowCUDA() {
  Msg("Starting CUDA PortalFlow (Global Stacks)...\n");
  int numPortals = g_numportals * 2;
  std::vector<CUDALeaf> hLeafs(portalclusters);
  std::vector<int> hLeafPortals;
  for (int i = 0; i < portalclusters; i++) {
    hLeafs[i].portalOffset = (int)hLeafPortals.size();
    hLeafs[i].numPortals = leafs[i].portals.Count();
    for (int j = 0; j < leafs[i].portals.Count(); j++)
      hLeafPortals.push_back((int)(leafs[i].portals[j] - portals));
  }
  std::vector<CUDAPortal> hPortals(numPortals);
  std::vector<Vector> hWindingPoints;
  for (int i = 0; i < numPortals; i++) {
    hPortals[i].origin = portals[i].origin;
    hPortals[i].planeNormal = portals[i].plane.normal;
    hPortals[i].planeDist = portals[i].plane.dist;
    hPortals[i].numPoints = portals[i].winding->numpoints;
    hPortals[i].windingOffset = (int)hWindingPoints.size();
    hPortals[i].leaf = portals[i].leaf;
    for (int k = 0; k < portals[i].winding->numpoints; k++)
      hWindingPoints.push_back(portals[i].winding->points[k]);
  }
  Msg("Pre-calculating SimpleFlood on CPU...\n");
  std::vector<unsigned char> hFlood(numPortals * portalbytes);
  for (int i = 0; i < numPortals; i++) {
    if (!portals[i].portalflood) {
      portals[i].portalflood = (byte *)malloc(portalbytes);
      memset(portals[i].portalflood, 0, portalbytes);
    }
    SimpleFlood(&portals[i], portals[i].leaf);
    memcpy(&hFlood[i * portalbytes], portals[i].portalflood, portalbytes);
  }
  CHECK_CUDA(cudaMalloc(&g_dPortals, numPortals * sizeof(CUDAPortal)));
  CHECK_CUDA(cudaMemcpy(g_dPortals, hPortals.data(),
                        numPortals * sizeof(CUDAPortal),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMalloc(&g_dWindingPoints, hWindingPoints.size() * sizeof(Vector)));
  CHECK_CUDA(cudaMemcpy(g_dWindingPoints, hWindingPoints.data(),
                        hWindingPoints.size() * sizeof(Vector),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&g_dLeafs, portalclusters * sizeof(CUDALeaf)));
  CHECK_CUDA(cudaMemcpy(g_dLeafs, hLeafs.data(),
                        portalclusters * sizeof(CUDALeaf),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&g_dLeafPortals, hLeafPortals.size() * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(g_dLeafPortals, hLeafPortals.data(),
                        hLeafPortals.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&g_dPortalFlood, numPortals * portalbytes));
  CHECK_CUDA(cudaMemcpy(g_dPortalFlood, hFlood.data(), numPortals * portalbytes,
                        cudaMemcpyHostToDevice));

  // Seed PortalVis from portalfront (BasePortalVis results)
  std::vector<unsigned char> hBaseVis(numPortals * portalbytes);
  for (int i = 0; i < numPortals; i++) {
    if (portals[i].portalfront) {
      memcpy(&hBaseVis[i * portalbytes], portals[i].portalfront, portalbytes);
    }
  }
  CHECK_CUDA(cudaMalloc(&g_dPortalVis, numPortals * portalbytes));
  CHECK_CUDA(cudaMemcpy(g_dPortalVis, hBaseVis.data(), numPortals * portalbytes,
                        cudaMemcpyHostToDevice));

  size_t numEntries = (size_t)numPortals * MAX_DEPTH_GPU;
  CHECK_CUDA(cudaMalloc(&g_dStack_leafIdx, numEntries * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&g_dStack_lastPortalIdx, numEntries * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&g_dStack_currentPortalIter, numEntries * sizeof(int)));
  CHECK_CUDA(
      cudaMalloc(&g_dStack_sourcePoints,
                 numEntries * MAX_POINTS_ON_WINDING_GPU * sizeof(Vector)));
  CHECK_CUDA(cudaMalloc(&g_dStack_sourceNumPoints, numEntries * sizeof(int)));
  CHECK_CUDA(
      cudaMalloc(&g_dStack_passPoints,
                 numEntries * MAX_POINTS_ON_WINDING_GPU * sizeof(Vector)));
  CHECK_CUDA(cudaMalloc(&g_dStack_passNumPoints, numEntries * sizeof(int)));

  // NEW: MightSee Stack
  size_t mightSeeSize =
      numEntries * portalbytes; // Each stack frame has 'portalbytes'
  CHECK_CUDA(cudaMalloc(&g_dStack_mightSee, mightSeeSize));

  cudaMemcpyToSymbol(c_numPortals, &numPortals, sizeof(int));
  cudaMemcpyToSymbol(c_portalBytes, &portalbytes, sizeof(int));
  cudaDeviceSetLimit(cudaLimitStackSize, 32768);

  int blockSize = 128;
  int numBlocks = (numPortals + blockSize - 1) / blockSize;
  Msg("Launching PortalFlowKernel <<<%d, %d>>>\n", numBlocks, blockSize);
  PortalFlowKernel<<<numBlocks, blockSize>>>(
      numPortals, portalbytes, g_dPortals, g_dLeafs, g_dLeafPortals,
      g_dWindingPoints, g_dPortalFlood, g_dPortalVis, g_dStack_leafIdx,
      g_dStack_lastPortalIdx, g_dStack_currentPortalIter, g_dStack_sourcePoints,
      g_dStack_sourceNumPoints, g_dStack_passPoints, g_dStack_passNumPoints,
      g_dStack_mightSee);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    Error("Kernel Launch Error: %s\n", cudaGetErrorString(err));

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
    Error("Kernel Sync Error: %s\n", cudaGetErrorString(err));

  std::vector<unsigned char> hOutVis(numPortals * portalbytes);
  cudaMemcpy(hOutVis.data(), g_dPortalVis, numPortals * portalbytes,
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < numPortals; i++) {
    if (!portals[i].portalvis)
      portals[i].portalvis = (byte *)malloc(portalbytes);
    memcpy(portals[i].portalvis, &hOutVis[i * portalbytes], portalbytes);
    portals[i].status = stat_done;
  }
  cudaFree(g_dPortals);
  cudaFree(g_dWindingPoints);
  cudaFree(g_dLeafs);
  cudaFree(g_dLeafPortals);
  cudaFree(g_dPortalFlood);
  cudaFree(g_dPortalVis);
  cudaFree(g_dStack_leafIdx);
  cudaFree(g_dStack_lastPortalIdx);
  cudaFree(g_dStack_currentPortalIter);
  cudaFree(g_dStack_sourcePoints);
  cudaFree(g_dStack_sourceNumPoints);
  cudaFree(g_dStack_passPoints);
  cudaFree(g_dStack_passNumPoints);
  cudaFree(g_dStack_mightSee);
  g_dPortals = nullptr;
  g_dWindingPoints = nullptr;
  g_dLeafs = nullptr;
  g_dLeafPortals = nullptr;
  g_dPortalFlood = nullptr;
  g_dPortalVis = nullptr;
  g_dStack_leafIdx = nullptr;
  g_dStack_lastPortalIdx = nullptr;
  g_dStack_currentPortalIter = nullptr;
  g_dStack_sourcePoints = nullptr;
  g_dStack_sourceNumPoints = nullptr;
  g_dStack_passPoints = nullptr;
  g_dStack_passNumPoints = nullptr;
  g_dStack_mightSee = nullptr;
  Msg("CUDA PortalFlow Finished.\n");
}
