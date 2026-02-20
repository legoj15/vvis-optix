#ifndef VVIS_OPTIX_H
#define VVIS_OPTIX_H

#include "mathlib/vector.h"
#include "tier1/utlvector.h"

// Structs shared between Host (C++) and Device (PTX)
struct VVIS_GPUPortal {
  float origin_x, origin_y, origin_z;
  float normal_x, normal_y, normal_z;
  float dist;
  int numPoints;
  int windingOffset;
};

// Launch parameters
struct VVIS_OptixLaunchParams {
  unsigned int num_portals;
  unsigned int portal_bytes;
  VVIS_GPUPortal *portals;
  Vector *winding_points;
  unsigned char *portal_flood;
  unsigned char *portal_vis;
  unsigned long long traversable;
};

// Interface class
class CVVisOptiX {
public:
  static bool Initialize();
  static void BuildScene();
  static void TraceVisibility(int numPortals, int portalBytes,
                              VVIS_GPUPortal *d_portals, Vector *d_windings,
                              unsigned char *d_portalFlood,
                              unsigned char *d_portalVis);
  static void Shutdown();
  static bool IsInitialized();
};

#endif // VVIS_OPTIX_H
