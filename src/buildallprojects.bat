@echo off
setlocal enabledelayedexpansion

:: -----------------------------------------------------------------------
:: VVIS Build Script
:: -----------------------------------------------------------------------

:: 1. Locate MSBuild
set "MSBUILD_PATH="
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe`) do (
        set "MSBUILD_PATH=%%i"
    )
)

if "!MSBUILD_PATH!"=="" (
    echo [ERROR] MSBuild.exe not found. Please ensure Visual Studio is installed.
    exit /b 1
)

echo [INFO] Found MSBuild: "!MSBUILD_PATH!"

:: 2. Set Build Parameters
set "SLN_FILE=mapping_tools.sln"
set "CONFIG=Release"
set "PLATFORM=win64"

:: 3. Execute Build
echo [INFO] Building %SLN_FILE% (%CONFIG%^|%PLATFORM%)...
echo.

"!MSBUILD_PATH!" "%SLN_FILE%" ^
    /p:Configuration=%CONFIG% ^
    /p:Platform=%PLATFORM% ^
    /m ^
    /t:Build ^
    /fl ^
    /flp:LogFile=build_tools.log;Verbosity=normal ^
    /consoleloggerparameters:Summary;Verbosity=minimal

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed. See build_tools.log for details.
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Build completed successfully.
endlocal
