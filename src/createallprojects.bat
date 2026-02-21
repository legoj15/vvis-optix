@echo off
setlocal

echo.
echo ================================================================
echo Creating VVIS Projects (VS2026)
echo ================================================================
echo.

devtools\bin\vpc.exe /2022 /define:VS2026 /define:SOURCESDK +everything /mksln mapping_tools.sln

if ERRORLEVEL 1 goto BuildFailed
goto BuildOK

:BuildFailed
echo.
echo *** ERROR! VPC Failed to generate projects! ***
echo.
goto End

:BuildOK
echo.
echo Projects generated successfully.
echo.

:End
