# VVIS CUDA

A CUDA-accelerated implementation of VVIS for Source SDK 2013 to improve visibility compilation performance.

## Description

This project replaces the standard CPU-based `vvis` tool with a GPU-accelerated version using CUDA. This significantly reduces compile times for complex maps by leveraging the parallel processing power of NVIDIA GPUs for visibility calculations.

## Build Requirements

- **NVIDIA GPU** with CUDA 13.1 support.
- [**CUDA 13.1 SDK**](https://developer.nvidia.com/cuda-downloads)
- **Source SDK 2013 Multiplayer** ([installed via Steam](steam://install/243750)).
- **Visual Studio 2022 or later** with:
    - Desktop development with C++ workload.
    - MSVC v143 build tools.
    - Windows SDK 10 or 11.

## Building

1.  Clone the repository:
    ```bash
    git clone https://github.com/legoj15/vvis-cuda
    ```
2.  Navigate to `src` and run the project creation script:
    ```bat
    createvvisprojects.bat
    ```
3.  Open `vvis.sln` in Visual Studio 2022 or later.
4.  Build the solution (Release configuration recommended for performance).

## Usage & Testing

To test the CUDA implementation, copy the provided batch script (which wraps the PowerShell script) and folder with test maps to the 2013 MP SDK's bin\x64 folder. The files are located here:

[`.\game\bin\x64`](https://github.com/legoj15/vvis-cuda/tree/master/game/bin/x64)

This script runs both the standard `vvis` and the new `vvis_cuda` on a test map and validates the results. Be sure to have copied the vvis_cuda executable and it's application extension to the bin\x64 folder of the 2013 MP SDK Base.

## License

Derived from the Source SDK 2013. See [LICENSE](LICENSE) for details.
