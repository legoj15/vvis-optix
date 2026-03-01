//=============================================================================
// staticprop_rebake.h
//
// UV re-unwrap and texture rebake for static prop lightmap support.
// When a model has overlapping UVs that prevent per-texel lightmapping,
// this module re-unwraps the UVs via xatlas and rebakes all referenced
// textures into a new atlas. The result is embedded in the BSP pakfile.
//=============================================================================
#ifndef STATICPROP_REBAKE_H
#define STATICPROP_REBAKE_H

#include "mathlib/vector2d.h"
#include "studio.h"
#include "tier1/utlvector.h"

// Forward declarations
class CUtlBuffer;
struct IZip;
class IFileSystem;

//-----------------------------------------------------------------------------
// Result of UV re-unwrapping via xatlas.
//-----------------------------------------------------------------------------
struct RewrapResult_t {
  // Per output-vertex new UV (normalized 0-1).
  // Indexed by the xatlas output vertex index.
  CUtlVector<Vector2D> m_NewUVs;

  // Maps each xatlas output vertex back to the original VVD vertex index.
  // m_VertexXref[outputIdx] = original VVD global index.
  CUtlVector<int> m_VertexXref;

  // New index buffer (triangle list). Length = original triangle count * 3.
  // Indices refer to the output vertex array above.
  CUtlVector<unsigned int> m_NewIndices;

  // Per-triangle material name (same order as m_NewIndices / 3).
  CUtlVector<const char *> m_TriMaterialNames;

  // Atlas dimensions chosen by xatlas.
  int m_nAtlasWidth;
  int m_nAtlasHeight;
  int m_nOutputVertexCount;
};

//-----------------------------------------------------------------------------
// Detect UV overlap in a model's LOD-0 mesh.
// Returns the fraction of texels with conflicting claims (0.0 = no overlap).
//-----------------------------------------------------------------------------
float DetectUVOverlap(studiohdr_t *pStudioHdr, CUtlBuffer &vvdBuf,
                      CUtlBuffer &vtxBuf, int nTestResolution = 256);

//-----------------------------------------------------------------------------
// Re-unwrap model UVs to be non-overlapping using xatlas.
// Returns true on success, populates pResult.
//-----------------------------------------------------------------------------
bool RewrapModelUVs(studiohdr_t *pStudioHdr, CUtlBuffer &vvdBuf,
                    CUtlBuffer &vtxBuf, int nMaxAtlasSize,
                    RewrapResult_t *pResult);

//-----------------------------------------------------------------------------
// Rebake all textures for a model using the new UV layout.
// Creates new VTFs and VMTs in the BSP pakfile.
// pPatchedCdMaterials receives the new $cdmaterials path for the QC.
//-----------------------------------------------------------------------------
bool RebakeTexturesForModel(studiohdr_t *pStudioHdr, CUtlBuffer &vvdBuf,
                            CUtlBuffer &vtxBuf, const RewrapResult_t &rewrap,
                            const char *pMapBase, const char *pPatchedModelName,
                            char *pPatchedCdMaterials, int nCdMatSize);

//-----------------------------------------------------------------------------
// Write an SMD using re-unwrapped UVs and patched material names.
//-----------------------------------------------------------------------------
bool WriteSMDWithRewrappedUVs(const char *pSmdPath, studiohdr_t *pStudioHdr,
                              CUtlBuffer &vvdBuf, const RewrapResult_t &rewrap,
                              const char *pPatchedMaterialPrefix);

#endif // STATICPROP_REBAKE_H
