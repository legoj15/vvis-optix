# VRAD-RTX Build & Troubleshooting Guide

Common issues encountered when compiling and testing the VRAD-RTX toolchain.

---

## Build System Issues

### MSB4126: Invalid Solution Configuration (win64 vs x64)
- **Error**: `The specified solution configuration "Release|x64" is invalid.`
- **Cause**: VPC generates the parent `.sln` file using `win64` as the platform string, but individual `.vcxproj` files expect `x64`.
- **Fix**: When building the solution file via MSBuild, use `/p:Platform=win64`. When building `vrad.vcxproj` directly, use `/p:Platform=x64`.

### Linker Error (LNK2019: InitGPUDirectLighting)
- **Error**: `unresolved external symbol InitGPUDirectLighting`
- **Cause**: Adding new `.cpp` bridge files (like `direct_lighting_gpu_vrad.cpp`) via Visual Studio GUI does not persist.
- **Fix**: You **must** add the new source file explicitly to `vrad_dll.vpc` under the source files section, and then re-run `createvradprojects.bat` to regenerate the project.

### LNK2038 Mismatch (_ITERATOR_DEBUG_LEVEL)
- **Error**: `mismatch detected for '_ITERATOR_DEBUG_LEVEL': value '0' doesn't match value '2'`
- **Cause**: Mixing Debug library targets with Release executables, typically in shared dependencies like `lzma.lib`.
- **Fix**: Clear the `src/lib/public/x64/` directory of the offending `.lib` and explicitly rebuild the entire solution with `/p:Configuration=Release`.

### VPC / Batch Script Paths
- **Error**: `'vpc.exe' is not recognized` when running `createvradprojects.bat`.
- **Fix**: The bat file must be executed from the root `src\` directory to properly resolve the `vpc_scripts` relative paths, not from `src/utils/vrad/`.

## GPU/OptiX Diagnostics

### OptiX SBT Record Mismatch Crash
- **Symptom**: "OptiX Error: Illegal memory access" or rays passing directly through all geometry.
- **Cause**: The number of records specified in `OptixBuildInputTriangleArray` does not match the size or layout of the Shader Binding Table (SBT) initialized on the host.
- **Fix**: Ensure `numSbtRecords` precisely tracks the material hitgroups pushed onto the BVH builder.

### Visibility Buffer Overflow (Heap Corruption)
- **Symptom**: `CRITICAL ERROR: vrad_rtx.exe failed with exit code UNKNOWN (NULL)` at the end of execution during array teardown, usually following massive transfer counts.
- **Cause**: The leaf-based visibility kernel returned more visible pairs than the `s_maxVisibilityPairs` allocated on the GPU. Writing past the end of the CUDA buffer causes hidden heap corruption that crashes during `cudaFree`.
- **Fix**: Scale up `s_maxVisibilityPairs` in `raytrace_optix.cpp` based on the map density (currently safe at 100,000,000 pairs).

### Powershell Process `$null` Exit Codes
- **Symptom**: CI script thinks the build failed with `Exit code UNKNOWN`, yet the log says the build succeeded and the BSP timestamp updated.
- **Cause**: PowerShell 5.1's `Start-Process -PassThru` contains bugs retrieving the exit code of native C++ apps once process handles are torn down.
- **Fix**: Use PowerShell 7 (`pwsh`) to run the automated tests.

### "Plaid Pattern" Artifacts in Direct Lighting
- **Symptom**: Checkered shadowing artifacts across contiguous flat surfaces.
- **Cause**: GPU rays self-intersecting with the source geometry.
- **Fix**: Ensure the visibility kernel `tmin` rests at `1e-4f` (not `1e-6f`).
