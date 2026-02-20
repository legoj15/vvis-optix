# VVIS OptiX

A hardware-accelerated drop-in replacement for the Source Engine's `vvis.exe` (Visibility Processor).

## Key Features

- **OptiX Hardware Acceleration:** Shifts the core PortalFlow raytracing workload to the RT Cores on modern NVIDIA GPUs.
- **Optimized in-game rendering:** Greater precision from thousands of rays allows for the same perceptible visibility, while reducing rendered visleafs out of sight.
- **Drop-in Compatibility:** Built as `vvis_optix.exe`, it works seamlessly with existing map compilation workflows.
- **Hybrid Architecture:** Combines fast CPU Setup/BasePortalVis bounds checking with GPU hardware raytracing for maximum efficiency.

## Build Requirements

- Visual Studio 2026 (untested on 2022 and below)
- NVIDIA CUDA SDK 13.1 (untested on older versions)
- NVIDIA OptiX SDK 9.1.0 (untested on older versions)

## Usage

Tested on an RTX 3090.

Replace your existing `vvis.exe` with `vvis_optix.exe` and `vvis_dll_optix.dll` in your tooling pipeline. The OptiX kernels are auto-compiled at runtime or can be pre-compiled via the included `.ptx` logic. Filesystem handling has been altered to allow for this build of vvis to be placed anywhere on your computer (doesn't have to be in a Source bin folder).

To invoke the hardware-accelerated path, run it with the `-cuda` flag:

```
vvis_optix.exe -cuda -game <path_to_mod_folder> <mapname>
```

If the `-cuda` flag is omitted, it falls back to a parallelized CPU path.

NOTE: If using with Garry's Mod, use the `-binroot` flag to point at the root of Source SDK 2013 Multiplayer, as Garry's Mod's binaries are incompatible.

## Architecture

This project splits visibility processing into a modern hybrid model:
1. **CPU BasePortalVis / SimpleFlood:** Fast memory bounds checking and topological setup using optimized CPU paths.
2. **GPU PortalFlow (OptiX):** The heavy inner-loop raycasting, ported to OptiX trace calls. It utilizes stochastic sampling across portal boundaries for massive concurrency.

Detailed architectural breakdowns can be found in the associated development documentation and commit history.

## Testing Suite

See [`.\game\bin\x64`](https://github.com/legoj15/vvis-optix/tree/master/game/bin/x64) for the Python/PowerShell automation suite used to validate parity between the reference SDK and the new GPU implementation.

### Validation Example (`run_tests.ps1`)

This script runs both the standard `vvis` and the new `vvis_optix` on a test map and validates the results. Be sure to have copied the vvis_optix executable and it's application extension to the bin\x64 folder of the 2013 MP SDK Base.

## License

Derived from the Source SDK 2013. See [LICENSE](LICENSE) for details.
