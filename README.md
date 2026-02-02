# VVIS CUDA

A CUDA-accelerated implementation of VVIS for Source SDK 2013 to improve visibility compilation performance.

## Description

This project replaces the standard CPU-based `vvis` tool with a GPU-accelerated version using CUDA. This significantly reduces compile times for complex maps by leveraging the parallel processing power of NVIDIA GPUs for visibility calculations.

## Requirements

- **NVIDIA GPU** with CUDA support.
- **Source SDK 2013 Multiplayer** ([installed via Steam](steam://install/243750)).
- **Visual Studio 2022 or later** with:
    - Desktop development with C++ workload.
    - MSVC v143 build tools.
    - Windows SDK 10 or 11.

## Building

1.  Clone the repository:
    ```bash
    git clone https://github.com/ValveSoftware/source-sdk-2013
    ```
2.  Navigate to `src` and run the project creation script:
    ```bat
    createvvisprojects.bat
    ```
3.  Open `vvis.sln` in Visual Studio 2022 or later.
4.  Build the solution (Release configuration recommended).

## Usage & Testing

To test the CUDA implementation, use the provided batch script (which wraps the PowerShell script) for correct execution on Windows:

```bat
.\game\bin\x64\test_vvis_cuda.bat
```

This script runs both the standard `vvis` and the new `vvis_cuda` on a test map and validates the results.

## License

Derived from the Source SDK 2013. See [LICENSE](LICENSE) for details.
