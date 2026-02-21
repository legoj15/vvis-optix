#ifndef VISIBILITY_GPU_H
#define VISIBILITY_GPU_H

#include "raytrace_shared.h"

//-----------------------------------------------------------------------------
// GPU Visibility Structures
// Shared between Host (C++) and Device (CUDA/OptiX)
//-----------------------------------------------------------------------------

struct GPUPatch {
  float3_t origin;
  float3_t normal;
  float planeDist;
  int faceNumber;
  int child1;        // index into g_Patches
  int child2;        // index into g_Patches
  int ndxNextParent; // index into g_Patches
  float area;
  int clusterNumber;
};

struct GPUVisSceneData {
  // Cluster -> Leaf Patch mapping (Flattened)
  int *clusterLeafOffsets;
  int *clusterLeafIndices;

  // Face -> Root Patch mapping
  // g_FacePatches in VRAD contains *all* patches, but we often need the root.
  // Or we can just use faceParents from VRAD.
  int *faceParentPatch;

  // All patches (flattened array)
  GPUPatch *patches;
  int numPatches;

  // PVS Data
  uint8_t *pvsData; // Compressed PVS array (if used on GPU)
                    // Or we might just pass the visible cluster list per launch
};

// Result of a visibility query (a pair that can see each other)
struct VisiblePair {
  int shooter;
  int receiver;
};

#endif // VISIBILITY_GPU_H
