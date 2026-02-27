# Resolved Regressions Archive

An archive of critical regressions encountered during the VRAD-RTX integration that required significant logic realignment with the Source SDK.

---

## 1. Bounce Light / Radiosity Energy Crisis
- **Symptoms**: Bounce lighting was 75% darker visually than the CPU reference, even when millions of transfers were generated.
- **Root Cause**: The RTX visibility iteration loop only evaluated patches against `HitID == -1` (Total Miss). It discarded rays that technically "hit" the destination patch geometry.
- **The SDK Standard**: The original VRAD raytracer considers a path visible if it misses everything, **OR** if the hit distance is greater than or equal to the total segment length: `if (hit.dist >= ray_length)`.
- **Resolution**: Restored the `hit_t >= tmax` logic across all CPU evaluation stages (`CTransferMaker::Finish`) and GPU output buffers (`ProcessResult`). This immediately restored the missing 65% of bounce energy.

## 2. Radiosity Refusal (Falsely Forced CPU State)
- **Symptoms**: CPU Control test runs (running `vrad_rtx.exe` *without* `-cuda`) generated only ~150 transfers compared to the normal ~37 million, resulting in scenes with no radiosity bounce.
- **Root Cause**: A rogue debug flag `g_bUseGPU = true;` was left uncommented in `vismat.cpp:862`.
- **Mechanics**: During a CPU-only run, forcing this flag routed all visibility queries into the GPU's OptiX batch queue. Since the kernel was never dispatched, the transfers were never completed or read, causing radiosity to fail instantly due to a lack of valid paths.
- **Resolution**: Removed constant overrides of the global flag; command-line constraints safely dictate the execution path.

## 3. Visibility Matrix Parity Optimization Drift
- **Symptoms**: Initial VRAD-RTX transfers dropped from 37 million down to 6.8 million paths, despite visually looking similar on low tier maps. High-frequency detail required those missing 31 million transfers.
- **Root Cause**: The engine uses a heuristic in `TestPatchToPatch`: `if (DotProduct(tmp, tmp) * 0.0625 < patch2->area)` to decide whether to stop recursing and just use the parent's generic visibility.
- **Experiments**: Lowered the coefficient to `0.0` (Brute Force test on every leaf), yielding 180+ million transfers but spiking processing time to 58s.
- **Resolution**: The SDK baseline coefficient of `0.0625` was functionally correct. The missing 31 million transfers were actually a symptom of the [Energy Crisis missing `hit_t >= tmax` test](#1-bounce-light--radiosity-energy-crisis) aggressively rejecting valid parent paths before they could be recorded. Once the target limits were fixed, `0.0625` safely generated perfectly matched transfer counts.

## 4. Uninitialized Geometry Trace (100% Miss Bug)
- **Symptoms**: Shadows bypass occluders entirely yielding hyper-bright direct lighting. Logs show `216 rays traced, 216 misses`.
- **Root Cause**: `TraceShadowBatch` was firing rays with uninitialized or clipped `tmax` ranges because the ray length was calculated *before* the delta direction was properly formulated.
- **Resolution**: Ray formulations enforce standard linear dependency: generate `dir = (p2 - p1)` FIRST, normalize to fetch distance, and THEN clamp `tmax`.
