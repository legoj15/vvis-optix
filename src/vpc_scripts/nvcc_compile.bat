@echo off
setlocal enabledelayedexpansion

if "%CUDA_PATH%"=="" (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" (
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
        echo Info: CUDA_PATH not set, using default: !CUDA_PATH!
    ) else (
        echo Error: CUDA_PATH environment variable is not set and default v13.1 path not found.
        exit /b 1
    )
)

set "CUDA_BIN_PATH=%CUDA_PATH%\bin"

rem Use the explicit path to cl.exe
rem Setup VS environment vars
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"

rem Use cl.exe from path (set by vcvars)
set "CL_PATH=cl.exe"

if not exist "!CL_PATH!" (
    for %%X in (cl.exe) do (set CL_PATH=%%~$PATH:X)
)

if "!CL_PATH!"=="" (
    echo Error: cl.exe not found in PATH after vcvars64.bat
    exit /b 1
)

echo Found CL at: !CL_PATH!

echo DEBUG: Running NVCC...
rem Removed -I"..\thirdparty\optix\include" and changed -ptx to -c for object compilation
"%CUDA_BIN_PATH%\nvcc.exe" -ccbin "!CL_PATH!" --use-local-env -m64 -D_WIN64 -D_WIN32 -c -O3 -arch=sm_75 -Wno-deprecated-gpu-targets -allow-unsupported-compiler --use_fast_math %*

if %ERRORLEVEL% neq 0 (
    echo Error: NVCC failed with error level %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

endlocal
