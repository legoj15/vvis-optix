# VRAD-RTX Architecture and Experiments

This document chronicles the major GPU offloading strategies attempted in VRAD-RTX, detailing what worked, what failed, and why certain architectures were adopted or reverted. This provides context for the current hybrid CPU/GPU workflow.

---

## 1. Direct Lighting Offload Strategies

In the pursuit of massively parallelizing the `BuildFacelights` phase, several batching strategies were tested to feed the OptiX BVH.

### Strategy 4: Thread-Local Face Batches (Current & Successful)
- **Architecture**: Injects `TraceShadowBatch` at the end of the CPU's `GatherSampleLightAt4Points` loop, collecting rays for a single `facelight_t` and firing them on the GPU.
- **Why it worked**: Kept the engine's per-face memory locality intact.
- **The Parity Breakthrough**: Early tests showed a persistent 27% difference between CPU and GPU. This was resolved by adopting a **Hybrid Visibility Model**:
  - **Standard Point/Sport/Area Lights**: Trace visibility rays on GPU, but retain all complex polynomial/SSE emittance math on the CPU.
  - **Environment/Skydome Lights**: Execute entirely on the CPU (bypassing the GPU batcher).
- **Result**: Achieved clinical bit-parity (Average RGB difference: ~0.03/255) because the exact same SIMD Lambertian math was performed, merely substituting the final ray test with OptiX.

### Strategy 5: Global Batching (Failed & Reverted)
- **Architecture**: Attempted to offload *most* of the direct lighting process by skipping inline CPU evaluation and pushing all lights and all faces into a massive multi-million ray GPU payload, processed deferred at the end of `RadWorld_Go`.
- **Why it failed**: 
  1. **CPU Collection Bottleneck**: The CPU parsing needed to pack 4 million rays and their attributes (origins, intensities, directions) into `SSDeferredRay_t` structures took *longer* than simply doing the raytrace inline. 
  2. **Yield Deficit**: Because Skylights bypass the batcher along with other PVS-culled lights, the GPU was only receiving ~17% of the total scene light rays, making it impossible to apply correct aggregate math asynchronously.
  3. **Visual Grid Artifacts**: Combining the deferred GPU standard light results with the immediate CPU skylight results caused fundamental blending misalignments (a 32% error).
- **Result**: Reverted. The time "saved" by OptiX was completely erased by the CPU memory-copy overhead for the mega-batch.

---

## 2. Supersample GPU Pipeline

### Supersample Collection (Failed & Reverted)
- **Architecture**: Attempted a 3-phase replacement for `SupersampleLightAtPoint` (Phase 1 CPU Gradient Check -> Phase 2 GPU OptiX Trace -> Phase 3 CPU Apply).
- **Why it failed**:
  - The supersample logic requires evaluating highly branchy winding tests, gradient falloffs, and PVS thresholds—an inherently scalar, CPU-friendly workload.
  - Performance cratered: The GPU trace took **0.03 seconds**, but the CPU overhead of packing the metadata structs (`SSFaceBatchEntry`) for the GPU took **49.11 seconds**. 
  - Total lighting time went from **116s (CPU)** to **117.7s (GPU-SS)**.
- **Rule of Thumb Exerted**: The GPU's role must remain focused on massively uniform workloads (unconditional shadow ray tracing, dense Bounces, matrix multiples).
- **Result**: Reverted. 

---

## Conclusion
The current `vrad_rtx` leverages a tightly coupled hybrid loop where the CPU maintains architectural control and mathematically complex gradient filtering, delegating only the binary "Is Occluded?" ray traversals for standard faces to OptiX.
