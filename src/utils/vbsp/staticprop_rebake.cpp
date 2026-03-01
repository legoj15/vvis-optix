//=============================================================================
// staticprop_rebake.cpp
//
// UV re-unwrap and texture rebake for static prop lightmap support.
//=============================================================================
#include "staticprop_rebake.h"
#include "bitmap/imageformat.h"
#include "bsplib.h"
#include "materialpatch.h"
#include "optimize.h"
#include "studio.h"
#include "tier1/UtlBuffer.h"
#include "tier1/strtools.h"
#include "tier1/utlvector.h"
#include "vbsp.h"
#include "vtf/vtf.h"
#include "xatlas.h"

using namespace OptimizedModel;

//-----------------------------------------------------------------------------
// Helper: load a VTF from the filesystem, decompress to BGRA8888.
// Returns an IVTFTexture* that the caller must destroy.
//-----------------------------------------------------------------------------
static IVTFTexture *LoadAndDecompressVTF(const char *pVtfPath) {
  CUtlBuffer buf;
  if (!g_pFullFileSystem->ReadFile(pVtfPath, NULL, buf)) {
    return NULL;
  }

  IVTFTexture *pTex = CreateVTFTexture();
  if (!pTex->Unserialize(buf)) {
    DestroyVTFTexture(pTex);
    return NULL;
  }

  // Convert to BGRA8888 for easy pixel access
  pTex->ConvertImageFormat(IMAGE_FORMAT_BGRA8888, false);
  return pTex;
}

//-----------------------------------------------------------------------------
// Helper: sample a BGRA8888 texture with bilinear filtering at UV coordinates.
// UV is in [0,1] range, wrapping is applied.
//-----------------------------------------------------------------------------
static void SampleTextureBilinear(const unsigned char *pData, int nWidth,
                                  int nHeight, float u, float v,
                                  unsigned char *pOutBGRA) {
  // Wrap UVs to [0,1)
  u = u - floorf(u);
  v = v - floorf(v);

  float fx = u * (float)nWidth - 0.5f;
  float fy = v * (float)nHeight - 0.5f;

  int x0 = (int)floorf(fx);
  int y0 = (int)floorf(fy);
  float sx = fx - (float)x0;
  float sy = fy - (float)y0;

  // Wrap pixel coordinates
  int x1 = (x0 + 1) % nWidth;
  int y1 = (y0 + 1) % nHeight;
  x0 = ((x0 % nWidth) + nWidth) % nWidth;
  y0 = ((y0 % nHeight) + nHeight) % nHeight;

  const unsigned char *p00 = &pData[(y0 * nWidth + x0) * 4];
  const unsigned char *p10 = &pData[(y0 * nWidth + x1) * 4];
  const unsigned char *p01 = &pData[(y1 * nWidth + x0) * 4];
  const unsigned char *p11 = &pData[(y1 * nWidth + x1) * 4];

  for (int c = 0; c < 4; ++c) {
    float val = (1.0f - sx) * (1.0f - sy) * p00[c] + sx * (1.0f - sy) * p10[c] +
                (1.0f - sx) * sy * p01[c] + sx * sy * p11[c];
    pOutBGRA[c] = (unsigned char)clamp((int)(val + 0.5f), 0, 255);
  }
}

//-----------------------------------------------------------------------------
// DetectUVOverlap
//-----------------------------------------------------------------------------
float DetectUVOverlap(studiohdr_t *pStudioHdr, CUtlBuffer &vvdBuf,
                      CUtlBuffer &vtxBuf, int nTestResolution) {
  FileHeader_t *pVtxHdr = (FileHeader_t *)vtxBuf.Base();
  vertexFileHeader_t *pVvdHdr = (vertexFileHeader_t *)vvdBuf.Base();
  if (!pVtxHdr || !pVvdHdr)
    return 0.0f;

  // Apply VVD fixups
  mstudiovertex_t *pRawVerts =
      (mstudiovertex_t *)((byte *)pVvdHdr + pVvdHdr->vertexDataStart);
  mstudiovertex_t *pVerts = pRawVerts;
  mstudiovertex_t *pFixedVerts = NULL;
  int numLod0Verts = pVvdHdr->numLODVertexes[0];

  if (pVvdHdr->numFixups > 0) {
    pFixedVerts = new mstudiovertex_t[numLod0Verts];
    vertexFileFixup_t *pFixups =
        (vertexFileFixup_t *)((byte *)pVvdHdr + pVvdHdr->fixupTableStart);
    int target = 0;
    for (int f = 0; f < pVvdHdr->numFixups; ++f) {
      if (pFixups[f].lod < 0)
        continue;
      if (target + pFixups[f].numVertexes > numLod0Verts)
        break;
      memcpy(&pFixedVerts[target], &pRawVerts[pFixups[f].sourceVertexID],
             pFixups[f].numVertexes * sizeof(mstudiovertex_t));
      target += pFixups[f].numVertexes;
    }
    pVerts = pFixedVerts;
  }

  const int totalPixels = nTestResolution * nTestResolution;

  struct TexelClaim {
    bool claimed;
    bool conflicted;
    Vector normal;
  };

  CUtlVector<TexelClaim> claims;
  claims.SetCount(totalPixels);
  memset(claims.Base(), 0, totalPixels * sizeof(TexelClaim));

  int totalClaimed = 0;
  int totalConflicts = 0;

  for (int bp = 0; bp < pStudioHdr->numbodyparts && bp < pVtxHdr->numBodyParts;
       ++bp) {
    mstudiobodyparts_t *pBodyPart = pStudioHdr->pBodypart(bp);
    BodyPartHeader_t *pVtxBodyPart = pVtxHdr->pBodyPart(bp);

    for (int m = 0; m < pBodyPart->nummodels && m < pVtxBodyPart->numModels;
         ++m) {
      mstudiomodel_t *pModel = pBodyPart->pModel(m);
      ModelHeader_t *pVtxModel = pVtxBodyPart->pModel(m);
      if (pVtxModel->numLODs < 1)
        continue;
      ModelLODHeader_t *pVtxLod = pVtxModel->pLOD(0);
      int modelVertexBase = pModel->vertexindex / sizeof(mstudiovertex_t);

      for (int meshIdx = 0;
           meshIdx < pModel->nummeshes && meshIdx < pVtxLod->numMeshes;
           ++meshIdx) {
        mstudiomesh_t *pMesh = pModel->pMesh(meshIdx);
        MeshHeader_t *pVtxMesh = pVtxLod->pMesh(meshIdx);

        for (int sg = 0; sg < pVtxMesh->numStripGroups; ++sg) {
          StripGroupHeader_t *pStripGroup = pVtxMesh->pStripGroup(sg);
          for (int s = 0; s < pStripGroup->numStrips; ++s) {
            StripHeader_t *pStrip = pStripGroup->pStrip(s);
            if (!(pStrip->flags & STRIP_IS_TRILIST))
              continue;

            for (int idx = 0; idx < pStrip->numIndices - 2; idx += 3) {
              int vvdIdx[3];
              Vector2D uv[3];
              Vector pos[3];

              for (int vi = 0; vi < 3; ++vi) {
                int globalIdx = pStrip->indexOffset + idx + vi;
                unsigned short vtxVertIdx = *pStripGroup->pIndex(globalIdx);
                Vertex_t *pVtxVert = pStripGroup->pVertex(vtxVertIdx);
                vvdIdx[vi] = modelVertexBase + pMesh->vertexoffset +
                             pVtxVert->origMeshVertID;
                if (vvdIdx[vi] < 0 || vvdIdx[vi] >= numLod0Verts)
                  vvdIdx[vi] = 0;
                uv[vi].x = pVerts[vvdIdx[vi]].m_vecTexCoord.x;
                uv[vi].y = pVerts[vvdIdx[vi]].m_vecTexCoord.y;
                pos[vi] = pVerts[vvdIdx[vi]].m_vecPosition;
              }

              // Face normal
              Vector edge1 = pos[1] - pos[0];
              Vector edge2 = pos[2] - pos[0];
              Vector faceNormal;
              CrossProduct(edge1, edge2, faceNormal);
              VectorNormalize(faceNormal);

              // Rasterize triangle in UV space to detect overlaps
              // Simple scanline rasterization
              float minU = min(uv[0].x, min(uv[1].x, uv[2].x));
              float maxU = max(uv[0].x, max(uv[1].x, uv[2].x));
              float minV = min(uv[0].y, min(uv[1].y, uv[2].y));
              float maxV = max(uv[0].y, max(uv[1].y, uv[2].y));

              // Wrap to [0,1] for overlap detection
              minU = minU - floorf(minU);
              maxU = maxU - floorf(maxU);
              minV = minV - floorf(minV);
              maxV = maxV - floorf(maxV);
              if (maxU < minU)
                maxU = 1.0f;
              if (maxV < minV)
                maxV = 1.0f;

              int px0 = max(0, (int)(minU * nTestResolution));
              int px1 = min(nTestResolution - 1, (int)(maxU * nTestResolution));
              int py0 = max(0, (int)(minV * nTestResolution));
              int py1 = min(nTestResolution - 1, (int)(maxV * nTestResolution));

              for (int py = py0; py <= py1; ++py) {
                for (int px = px0; px <= px1; ++px) {
                  int linearPos = py * nTestResolution + px;
                  TexelClaim &claim = claims[linearPos];
                  if (!claim.claimed) {
                    claim.claimed = true;
                    claim.normal = faceNormal;
                    totalClaimed++;
                  } else if (!claim.conflicted) {
                    float dot = DotProduct(claim.normal, faceNormal);
                    if (dot < 0.5f) {
                      claim.conflicted = true;
                      totalConflicts++;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if (pFixedVerts)
    delete[] pFixedVerts;

  if (totalClaimed == 0)
    return 0.0f;
  return (float)totalConflicts / (float)totalClaimed;
}

//-----------------------------------------------------------------------------
// RewrapModelUVs — use xatlas to generate non-overlapping UV atlas.
//-----------------------------------------------------------------------------
bool RewrapModelUVs(studiohdr_t *pStudioHdr, CUtlBuffer &vvdBuf,
                    CUtlBuffer &vtxBuf, int nMaxAtlasSize,
                    RewrapResult_t *pResult) {
  FileHeader_t *pVtxHdr = (FileHeader_t *)vtxBuf.Base();
  vertexFileHeader_t *pVvdHdr = (vertexFileHeader_t *)vvdBuf.Base();
  if (!pVtxHdr || !pVvdHdr)
    return false;

  // Apply VVD fixups
  mstudiovertex_t *pRawVerts =
      (mstudiovertex_t *)((byte *)pVvdHdr + pVvdHdr->vertexDataStart);
  mstudiovertex_t *pVerts = pRawVerts;
  mstudiovertex_t *pFixedVerts = NULL;
  int numLod0Verts = pVvdHdr->numLODVertexes[0];

  if (pVvdHdr->numFixups > 0) {
    pFixedVerts = new mstudiovertex_t[numLod0Verts];
    vertexFileFixup_t *pFixups =
        (vertexFileFixup_t *)((byte *)pVvdHdr + pVvdHdr->fixupTableStart);
    int target = 0;
    for (int f = 0; f < pVvdHdr->numFixups; ++f) {
      if (pFixups[f].lod < 0)
        continue;
      if (target + pFixups[f].numVertexes > numLod0Verts)
        break;
      memcpy(&pFixedVerts[target], &pRawVerts[pFixups[f].sourceVertexID],
             pFixups[f].numVertexes * sizeof(mstudiovertex_t));
      target += pFixups[f].numVertexes;
    }
    pVerts = pFixedVerts;
  }

  // Collect all vertices (positions + normals) and triangles from LOD 0.
  CUtlVector<float> positions;  // x,y,z per vertex
  CUtlVector<float> normals;    // x,y,z per vertex
  CUtlVector<uint32_t> indices; // triangle indices
  CUtlVector<const char *> triMaterialNames;

  // Map: our collected vertex index → global VVD index
  CUtlVector<int> vertToVvdIdx;

  // We need a flattened vertex list. Since different meshes share VVD verts,
  // we flatten everything into one big list and remap indices.
  // However, xatlas needs contiguous vertex arrays. We'll deduplicate by VVD
  // index.
  CUtlVector<int> vvdToCollected;
  vvdToCollected.SetCount(numLod0Verts);
  for (int i = 0; i < numLod0Verts; ++i)
    vvdToCollected[i] = -1;

  for (int bp = 0; bp < pStudioHdr->numbodyparts && bp < pVtxHdr->numBodyParts;
       ++bp) {
    mstudiobodyparts_t *pBodyPart = pStudioHdr->pBodypart(bp);
    BodyPartHeader_t *pVtxBodyPart = pVtxHdr->pBodyPart(bp);

    for (int m = 0; m < pBodyPart->nummodels && m < pVtxBodyPart->numModels;
         ++m) {
      mstudiomodel_t *pModel = pBodyPart->pModel(m);
      ModelHeader_t *pVtxModel = pVtxBodyPart->pModel(m);
      if (pVtxModel->numLODs < 1)
        continue;
      ModelLODHeader_t *pVtxLod = pVtxModel->pLOD(0);
      int modelVertexBase = pModel->vertexindex / sizeof(mstudiovertex_t);

      for (int meshIdx = 0;
           meshIdx < pModel->nummeshes && meshIdx < pVtxLod->numMeshes;
           ++meshIdx) {
        mstudiomesh_t *pMesh = pModel->pMesh(meshIdx);
        MeshHeader_t *pVtxMesh = pVtxLod->pMesh(meshIdx);

        const char *pMaterialName = "default";
        if (pMesh->material >= 0 && pMesh->material < pStudioHdr->numtextures) {
          pMaterialName = pStudioHdr->pTexture(pMesh->material)->pszName();
        }

        for (int sg = 0; sg < pVtxMesh->numStripGroups; ++sg) {
          StripGroupHeader_t *pStripGroup = pVtxMesh->pStripGroup(sg);
          for (int s = 0; s < pStripGroup->numStrips; ++s) {
            StripHeader_t *pStrip = pStripGroup->pStrip(s);
            if (!(pStrip->flags & STRIP_IS_TRILIST))
              continue;

            for (int idx = 0; idx < pStrip->numIndices - 2; idx += 3) {
              uint32_t triIdx[3];
              for (int vi = 0; vi < 3; ++vi) {
                int globalIdx = pStrip->indexOffset + idx + vi;
                unsigned short vtxVertIdx = *pStripGroup->pIndex(globalIdx);
                Vertex_t *pVtxVert = pStripGroup->pVertex(vtxVertIdx);
                int vvdIdx = modelVertexBase + pMesh->vertexoffset +
                             pVtxVert->origMeshVertID;
                if (vvdIdx < 0 || vvdIdx >= numLod0Verts)
                  vvdIdx = 0;

                // Deduplicate by VVD index
                if (vvdToCollected[vvdIdx] == -1) {
                  int newIdx = positions.Count() / 3;
                  vvdToCollected[vvdIdx] = newIdx;
                  vertToVvdIdx.AddToTail(vvdIdx);

                  positions.AddToTail(pVerts[vvdIdx].m_vecPosition.x);
                  positions.AddToTail(pVerts[vvdIdx].m_vecPosition.y);
                  positions.AddToTail(pVerts[vvdIdx].m_vecPosition.z);

                  normals.AddToTail(pVerts[vvdIdx].m_vecNormal.x);
                  normals.AddToTail(pVerts[vvdIdx].m_vecNormal.y);
                  normals.AddToTail(pVerts[vvdIdx].m_vecNormal.z);
                }
                triIdx[vi] = (uint32_t)vvdToCollected[vvdIdx];
              }
              indices.AddToTail(triIdx[0]);
              indices.AddToTail(triIdx[1]);
              indices.AddToTail(triIdx[2]);
              triMaterialNames.AddToTail(pMaterialName);
            }
          }
        }
      }
    }
  }

  if (indices.Count() == 0) {
    if (pFixedVerts)
      delete[] pFixedVerts;
    return false;
  }

  int nInputVertices = positions.Count() / 3;
  int nInputTriangles = indices.Count() / 3;
  Msg("  xatlas: %d vertices, %d triangles\n", nInputVertices, nInputTriangles);

  // --- Run xatlas ---
  xatlas::Atlas *atlas = xatlas::Create();

  xatlas::MeshDecl meshDecl;
  meshDecl.vertexPositionData = positions.Base();
  meshDecl.vertexPositionStride = sizeof(float) * 3;
  meshDecl.vertexNormalData = normals.Base();
  meshDecl.vertexNormalStride = sizeof(float) * 3;
  meshDecl.indexData = indices.Base();
  meshDecl.indexCount = indices.Count();
  meshDecl.indexFormat = xatlas::IndexFormat::UInt32;
  meshDecl.vertexCount = nInputVertices;

  xatlas::AddMeshError err = xatlas::AddMesh(atlas, meshDecl);
  if (err != xatlas::AddMeshError::Success) {
    Warning("  xatlas AddMesh failed: %s\n", xatlas::StringForEnum(err));
    xatlas::Destroy(atlas);
    if (pFixedVerts)
      delete[] pFixedVerts;
    return false;
  }

  xatlas::ChartOptions chartOptions;
  chartOptions.maxIterations = 4;

  xatlas::PackOptions packOptions;
  packOptions.resolution = nMaxAtlasSize;
  packOptions.padding = 4;
  packOptions.bilinear = true;
  packOptions.bruteForce = false;
  packOptions.createImage = false;

  xatlas::Generate(atlas, chartOptions, packOptions);

  if (atlas->meshCount == 0 || atlas->width == 0 || atlas->height == 0) {
    Warning("  xatlas Generate produced no output\n");
    xatlas::Destroy(atlas);
    if (pFixedVerts)
      delete[] pFixedVerts;
    return false;
  }

  Msg("  xatlas: atlas %dx%d, %d charts, %d output vertices\n", atlas->width,
      atlas->height, atlas->chartCount, atlas->meshes[0].vertexCount);

  // --- Extract results ---
  const xatlas::Mesh &outMesh = atlas->meshes[0];

  // VTF requires dimensions to be multiples of 4 (block alignment).
  // Round up and clamp to the requested maximum.
  int atlasW = (int)((atlas->width + 3) & ~3);
  int atlasH = (int)((atlas->height + 3) & ~3);
  if (atlasW > nMaxAtlasSize)
    atlasW = nMaxAtlasSize;
  if (atlasH > nMaxAtlasSize)
    atlasH = nMaxAtlasSize;

  pResult->m_nAtlasWidth = atlasW;
  pResult->m_nAtlasHeight = atlasH;
  pResult->m_nOutputVertexCount = (int)outMesh.vertexCount;

  // Build new UVs and vertex xref
  pResult->m_NewUVs.SetCount(outMesh.vertexCount);
  pResult->m_VertexXref.SetCount(outMesh.vertexCount);

  for (uint32_t v = 0; v < outMesh.vertexCount; ++v) {
    const xatlas::Vertex &xv = outMesh.vertexArray[v];
    // Normalize UVs to [0,1] using the rounded atlas dimensions
    pResult->m_NewUVs[v].x = xv.uv[0] / (float)atlasW;
    pResult->m_NewUVs[v].y = xv.uv[1] / (float)atlasH;
    // xref maps output vertex → input vertex → VVD index
    pResult->m_VertexXref[v] = vertToVvdIdx[xv.xref];
  }

  // Build new index buffer
  pResult->m_NewIndices.SetCount(outMesh.indexCount);
  for (uint32_t i = 0; i < outMesh.indexCount; ++i) {
    pResult->m_NewIndices[i] = outMesh.indexArray[i];
  }

  // Copy material names per triangle, mapping xatlas output back to original
  // Since xatlas reorders triangles by chart, we must reconstruct the
  // assignment
  int nOutputTriangles = outMesh.indexCount / 3;
  pResult->m_TriMaterialNames.SetCount(nOutputTriangles);
  for (int t = 0; t < nOutputTriangles; ++t) {
    pResult->m_TriMaterialNames[t] = "default";
  }

  int tOut = 0;
  for (uint32_t c = 0; c < outMesh.chartCount; ++c) {
    const xatlas::Chart &chart = outMesh.chartArray[c];
    for (uint32_t f = 0; f < chart.faceCount; ++f) {
      uint32_t origFaceIdx = chart.faceArray[f];
      if (tOut < nOutputTriangles && origFaceIdx < (uint32_t)nInputTriangles) {
        pResult->m_TriMaterialNames[tOut] = triMaterialNames[origFaceIdx];
      }
      tOut++;
    }
  }

  if (tOut != nOutputTriangles) {
    Warning("  xatlas map warning: %d reconstructed faces vs %d output "
            "triangles!\n",
            tOut, nOutputTriangles);
  }

  xatlas::Destroy(atlas);
  if (pFixedVerts)
    delete[] pFixedVerts;
  return true;
}

//-----------------------------------------------------------------------------
// RebakeTexturesForModel
//-----------------------------------------------------------------------------
bool RebakeTexturesForModel(studiohdr_t *pStudioHdr, CUtlBuffer &vvdBuf,
                            CUtlBuffer &vtxBuf, const RewrapResult_t &rewrap,
                            const char *pMapBase, const char *pPatchedModelName,
                            char *pPatchedCdMaterials, int nCdMatSize) {
  vertexFileHeader_t *pVvdHdr = (vertexFileHeader_t *)vvdBuf.Base();
  if (!pVvdHdr)
    return false;

  // Apply VVD fixups (same as above)
  mstudiovertex_t *pRawVerts =
      (mstudiovertex_t *)((byte *)pVvdHdr + pVvdHdr->vertexDataStart);
  mstudiovertex_t *pVerts = pRawVerts;
  mstudiovertex_t *pFixedVerts = NULL;
  int numLod0Verts = pVvdHdr->numLODVertexes[0];

  if (pVvdHdr->numFixups > 0) {
    pFixedVerts = new mstudiovertex_t[numLod0Verts];
    vertexFileFixup_t *pFixups =
        (vertexFileFixup_t *)((byte *)pVvdHdr + pVvdHdr->fixupTableStart);
    int target = 0;
    for (int f = 0; f < pVvdHdr->numFixups; ++f) {
      if (pFixups[f].lod < 0)
        continue;
      if (target + pFixups[f].numVertexes > numLod0Verts)
        break;
      memcpy(&pFixedVerts[target], &pRawVerts[pFixups[f].sourceVertexID],
             pFixups[f].numVertexes * sizeof(mstudiovertex_t));
      target += pFixups[f].numVertexes;
    }
    pVerts = pFixedVerts;
  }

  // Build the patched $cdmaterials path
  V_snprintf(pPatchedCdMaterials, nCdMatSize, "maps/%s/staticprop_rebake",
             pMapBase);

  int nTriCount = rewrap.m_NewIndices.Count() / 3;

  // Collect unique material names
  CUtlVector<const char *> uniqueMaterials;
  for (int t = 0; t < nTriCount; ++t) {
    const char *pName = rewrap.m_TriMaterialNames[t];
    bool found = false;
    for (int u = 0; u < uniqueMaterials.Count(); ++u) {
      if (V_stricmp(uniqueMaterials[u], pName) == 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      uniqueMaterials.AddToTail(pName);
    }
  }

  IZip *pak = GetPakFile();

  for (int matIdx = 0; matIdx < uniqueMaterials.Count(); ++matIdx) {
    const char *pMaterialName = uniqueMaterials[matIdx];

    // Find the $basetexture for this material.
    // Search the model's $cdmaterials paths.
    char baseTextureName[512] = {};
    char fullMaterialPath[512];
    bool bFoundTexture = false;

    for (int cd = 0; cd < pStudioHdr->numcdtextures; ++cd) {
      const char *pCdPath = pStudioHdr->pCdtexture(cd);
      V_snprintf(fullMaterialPath, sizeof(fullMaterialPath), "%s%s", pCdPath,
                 pMaterialName);

      if (GetValueFromMaterial(fullMaterialPath, "$basetexture",
                               baseTextureName, sizeof(baseTextureName))) {
        bFoundTexture = true;
        break;
      }
    }

    if (!bFoundTexture) {
      Warning("  Rebake: Could not find $basetexture for material '%s'\n",
              pMaterialName);
      continue;
    }

    // Load the source VTF
    char srcVtfPath[512];
    V_snprintf(srcVtfPath, sizeof(srcVtfPath), "materials/%s.vtf",
               baseTextureName);

    IVTFTexture *pSrcTex = LoadAndDecompressVTF(srcVtfPath);
    if (!pSrcTex) {
      Warning("  Rebake: Could not load VTF '%s'\n", srcVtfPath);
      continue;
    }

    int srcW = pSrcTex->Width();
    int srcH = pSrcTex->Height();
    unsigned char *pSrcData = pSrcTex->ImageData(0, 0, 0);

    Msg("  Rebake: %s (%dx%d) -> atlas (%dx%d)\n", pMaterialName, srcW, srcH,
        rewrap.m_nAtlasWidth, rewrap.m_nAtlasHeight);

    // Create destination VTF
    IVTFTexture *pDstTex = CreateVTFTexture();
    if (!pDstTex->Init(rewrap.m_nAtlasWidth, rewrap.m_nAtlasHeight, 1,
                       IMAGE_FORMAT_BGRA8888,
                       TEXTUREFLAGS_NOMIP | TEXTUREFLAGS_NOLOD, 1)) {
      Warning("  Rebake: Failed to init destination VTF %dx%d\n",
              rewrap.m_nAtlasWidth, rewrap.m_nAtlasHeight);
      DestroyVTFTexture(pSrcTex);
      continue;
    }

    unsigned char *pDstData = pDstTex->ImageData(0, 0, 0);
    if (!pDstData) {
      Warning("  Rebake: ImageData returned NULL for destination VTF\n");
      DestroyVTFTexture(pDstTex);
      DestroyVTFTexture(pSrcTex);
      continue;
    }
    int dstW = rewrap.m_nAtlasWidth;
    int dstH = rewrap.m_nAtlasHeight;
    memset(pDstData, 0, dstW * dstH * 4);

    // Count triangles and pixels for this material
    int nMatTris = 0;
    int nPixelsWritten = 0;

    // Rasterize each triangle that uses this material
    for (int t = 0; t < nTriCount; ++t) {
      if (V_stricmp(rewrap.m_TriMaterialNames[t], pMaterialName) != 0)
        continue;

      nMatTris++;

      // Use same winding swap (0, 2, 1) as the SMD writer so the rasterizer
      // draws the exact same oriented triangle as the engine will render.
      uint32_t i0 = rewrap.m_NewIndices[t * 3 + 0];
      uint32_t i1 = rewrap.m_NewIndices[t * 3 + 2];
      uint32_t i2 = rewrap.m_NewIndices[t * 3 + 1];

      // New UVs (destination texture space). We DO NOT apply the 1.0 - V flip
      // here! Why? studiomdl flips the V coordinate when processing the SMD
      // (1.0 - (1.0 - V) = V). This means the engine ultimately samples the VTF
      // using the exact raw xatlas UV coordinate. The rasterizer must match
      // what the engine samples.
      Vector2D newUV[3] = {rewrap.m_NewUVs[i0], rewrap.m_NewUVs[i1],
                           rewrap.m_NewUVs[i2]};

      // Old UVs (source texture space) — from the original VVD vertices.
      int vvd0 = rewrap.m_VertexXref[i0];
      int vvd1 = rewrap.m_VertexXref[i1];
      int vvd2 = rewrap.m_VertexXref[i2];

      Vector2D oldUV[3];
      oldUV[0].x = pVerts[vvd0].m_vecTexCoord.x;
      oldUV[0].y = pVerts[vvd0].m_vecTexCoord.y;
      oldUV[1].x = pVerts[vvd1].m_vecTexCoord.x;
      oldUV[1].y = pVerts[vvd1].m_vecTexCoord.y;
      oldUV[2].x = pVerts[vvd2].m_vecTexCoord.x;
      oldUV[2].y = pVerts[vvd2].m_vecTexCoord.y;

      // Ensure proper floating point interpolation for UVs.
      // xatlas returns new UVs in [0, 1] range.
      // VTF textures are addressed top-to-bottom.
      float fMinX = min(newUV[0].x, min(newUV[1].x, newUV[2].x)) * dstW;
      float fMaxX = max(newUV[0].x, max(newUV[1].x, newUV[2].x)) * dstW;
      float fMinY = min(newUV[0].y, min(newUV[1].y, newUV[2].y)) * dstH;
      float fMaxY = max(newUV[0].y, max(newUV[1].y, newUV[2].y)) * dstH;

      int px0 = max(0, (int)floorf(fMinX));
      int px1 = min(dstW - 1, (int)ceilf(fMaxX));
      int py0 = max(0, (int)floorf(fMinY));
      int py1 = min(dstH - 1, (int)ceilf(fMaxY));

      // Triangle vertices in pixel space
      float ax = newUV[0].x * dstW, ay = newUV[0].y * dstH;
      float bx = newUV[1].x * dstW, by = newUV[1].y * dstH;
      float cx = newUV[2].x * dstW, cy = newUV[2].y * dstH;

      // Barycentric denominator
      float denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
      if (fabsf(denom) < 1e-6f)
        continue;
      float invDenom = 1.0f / denom;

      for (int py = py0; py <= py1; ++py) {
        for (int px = px0; px <= px1; ++px) {
          float ppx = (float)px + 0.5f;
          float ppy = (float)py + 0.5f;

          // Barycentric coordinates
          float w0 =
              ((by - cy) * (ppx - cx) + (cx - bx) * (ppy - cy)) * invDenom;
          float w1 =
              ((cy - ay) * (ppx - cx) + (ax - cx) * (ppy - cy)) * invDenom;
          float w2 = 1.0f - w0 - w1;

          // Inside triangle test (with small margin for edge coverage)
          if (w0 < -0.01f || w1 < -0.01f || w2 < -0.01f)
            continue;

          // Interpolate old UV using barycentric coords
          float srcU = w0 * oldUV[0].x + w1 * oldUV[1].x + w2 * oldUV[2].x;
          float srcV = w0 * oldUV[0].y + w1 * oldUV[1].y + w2 * oldUV[2].y;

          // Sample source texture
          unsigned char pixel[4];
          SampleTextureBilinear(pSrcData, srcW, srcH, srcU, srcV, pixel);

          // Write to destination
          int dstOffset = (py * dstW + px) * 4;
          pDstData[dstOffset + 0] = pixel[0];
          pDstData[dstOffset + 1] = pixel[1];
          pDstData[dstOffset + 2] = pixel[2];
          pDstData[dstOffset + 3] = pixel[3];
          nPixelsWritten++;
        }
      }
    }

    Msg("  Rebake: %d triangles, %d pixels written for '%s'\n", nMatTris,
        nPixelsWritten, pMaterialName);

    // Serialize and embed in BSP pakfile
    char dstVtfName[512];
    V_snprintf(dstVtfName, sizeof(dstVtfName),
               "materials/maps/%s/staticprop_rebake/%s.vtf", pMapBase,
               pMaterialName);
    V_strlower(dstVtfName);

    CUtlBuffer vtfBuf;
    if (pDstTex->Serialize(vtfBuf)) {
      AddBufferToPak(pak, dstVtfName, vtfBuf.Base(), vtfBuf.TellPut(), false);
      Msg("  Embedded rebaked VTF: %s (%dx%d BGRA8888)\n", dstVtfName, dstW,
          dstH);
    }

    DestroyVTFTexture(pDstTex);
    DestroyVTFTexture(pSrcTex);

    // Create patched VMT
    char dstVmtName[512];
    V_snprintf(dstVmtName, sizeof(dstVmtName),
               "materials/maps/%s/staticprop_rebake/%s.vmt", pMapBase,
               pMaterialName);
    V_strlower(dstVmtName);

    char vmtContent[1024];
    char relativeVtfPath[512];
    V_snprintf(relativeVtfPath, sizeof(relativeVtfPath),
               "maps/%s/staticprop_rebake/%s", pMapBase, pMaterialName);
    V_strlower(relativeVtfPath);

    V_snprintf(vmtContent, sizeof(vmtContent),
               "\"VertexLitGeneric\"\n"
               "{\n"
               "\t\"$basetexture\" \"%s\"\n"
               "\t\"$model\" \"1\"\n"
               "}\n",
               relativeVtfPath);

    AddBufferToPak(pak, dstVmtName, vmtContent, V_strlen(vmtContent), false);
    Msg("  Embedded patched VMT: %s\n", dstVmtName);
  }

  if (pFixedVerts)
    delete[] pFixedVerts;

  return true;
}

//-----------------------------------------------------------------------------
// WriteSMDWithRewrappedUVs — write SMD using xatlas output vertices/indices.
//-----------------------------------------------------------------------------
bool WriteSMDWithRewrappedUVs(const char *pSmdPath, studiohdr_t *pStudioHdr,
                              CUtlBuffer &vvdBuf, const RewrapResult_t &rewrap,
                              const char *pPatchedMaterialPrefix) {
  vertexFileHeader_t *pVvdHdr = (vertexFileHeader_t *)vvdBuf.Base();
  if (!pVvdHdr)
    return false;

  // Apply VVD fixups
  mstudiovertex_t *pRawVerts =
      (mstudiovertex_t *)((byte *)pVvdHdr + pVvdHdr->vertexDataStart);
  mstudiovertex_t *pVerts = pRawVerts;
  mstudiovertex_t *pFixedVerts = NULL;
  int numLod0Verts = pVvdHdr->numLODVertexes[0];

  if (pVvdHdr->numFixups > 0) {
    pFixedVerts = new mstudiovertex_t[numLod0Verts];
    vertexFileFixup_t *pFixups =
        (vertexFileFixup_t *)((byte *)pVvdHdr + pVvdHdr->fixupTableStart);
    int target = 0;
    for (int f = 0; f < pVvdHdr->numFixups; ++f) {
      if (pFixups[f].lod < 0)
        continue;
      if (target + pFixups[f].numVertexes > numLod0Verts)
        break;
      memcpy(&pFixedVerts[target], &pRawVerts[pFixups[f].sourceVertexID],
             pFixups[f].numVertexes * sizeof(mstudiovertex_t));
      target += pFixups[f].numVertexes;
    }
    pVerts = pFixedVerts;
  }

  FILE *fp = fopen(pSmdPath, "w");
  if (!fp) {
    if (pFixedVerts)
      delete[] pFixedVerts;
    return false;
  }

  // Header
  fprintf(fp, "// SMD generated by VBSP staticprop_rebake (UV re-unwrapped)\n");
  fprintf(fp, "version 1\n");

  // Nodes
  fprintf(fp, "nodes\n");
  for (int b = 0; b < pStudioHdr->numbones; ++b) {
    mstudiobone_t *pBone = pStudioHdr->pBone(b);
    fprintf(fp, "  %d \"%s\" %d\n", b, pBone->pszName(), pBone->parent);
  }
  fprintf(fp, "end\n");

  // Skeleton
  fprintf(fp, "skeleton\n");
  fprintf(fp, "time 0\n");
  for (int b = 0; b < pStudioHdr->numbones; ++b) {
    mstudiobone_t *pBone = pStudioHdr->pBone(b);
    fprintf(fp, "  %d  %f %f %f  %f %f %f\n", b, pBone->pos.x, pBone->pos.y,
            pBone->pos.z, pBone->rot.x, pBone->rot.y, pBone->rot.z);
  }
  fprintf(fp, "end\n");

  // Triangles — use xatlas output indices and UVs
  fprintf(fp, "triangles\n");

  int nTriCount = rewrap.m_NewIndices.Count() / 3;
  for (int t = 0; t < nTriCount; ++t) {
    // Material name — use the rebaked material name
    fprintf(fp, "%s\n", rewrap.m_TriMaterialNames[t]);

    for (int vi = 0; vi < 3; ++vi) {
      // SMD winding order: 0, 2, 1. VTX order is 0, 1, 2. We MUST swap 1 and 2
      // so the physical face normal points outward in the engine, avoiding
      // black lighting and mirrored textures (from viewing the back-face).
      int windIdx = (vi == 1) ? 2 : (vi == 2) ? 1 : 0;
      uint32_t outVertIdx = rewrap.m_NewIndices[t * 3 + windIdx];
      int vvdIdx = rewrap.m_VertexXref[outVertIdx];

      if (vvdIdx < 0 || vvdIdx >= numLod0Verts)
        vvdIdx = 0;

      mstudiovertex_t &vert = pVerts[vvdIdx];
      Vector2D newUV = rewrap.m_NewUVs[outVertIdx];
      int boneIdx = vert.m_BoneWeights.bone[0];

      // Counter-rotate positions and normals. studiomdl's $staticprop applies
      // a transform (X' = -Y, Y' = X). We pre-apply its inverse (X = Y', Y =
      // -X') so the double-transform yields the original correctly aligned
      // result.
      float smdPosX = vert.m_vecPosition.y;
      float smdPosY = -vert.m_vecPosition.x;
      float smdPosZ = vert.m_vecPosition.z;
      float smdNrmX = vert.m_vecNormal.y;
      float smdNrmY = -vert.m_vecNormal.x;
      float smdNrmZ = vert.m_vecNormal.z;

      // Write counter-rotated VVD positions/normals
      fprintf(fp, "  %d  %f %f %f  %f %f %f  %f %f", boneIdx, smdPosX, smdPosY,
              smdPosZ, smdNrmX, smdNrmY, smdNrmZ, newUV.x, 1.0f - newUV.y);

      int numWeights = vert.m_BoneWeights.numbones;
      if (numWeights > 1) {
        fprintf(fp, " %d", numWeights);
        for (int bw = 0; bw < numWeights; ++bw) {
          fprintf(fp, " %d %f", vert.m_BoneWeights.bone[bw],
                  vert.m_BoneWeights.weight[bw]);
        }
      }
      fprintf(fp, "\n");
    }
  }

  fprintf(fp, "end\n");
  fclose(fp);

  if (pFixedVerts)
    delete[] pFixedVerts;
  return true;
}
