param([int]$TimeoutExtensionMinutes = 0)

$SDK_BIN = $PSScriptRoot
$MOD_DIR = "..\..\sourcetest"
$LOG_FILE = "test_vvis_cuda.log"
$MAP_NAME = "validation"
$CONTROL_LOG = "bsp_unit_tests\control\$MAP_NAME.log"
$CUDA_LOG = "bsp_unit_tests\test\$MAP_NAME.log"

# Clean up any existing tool processes
$tools = @("vbsp", "vvis", "vrad", "vvis_cuda", "vrad_rtx")
foreach ($tool in $tools) {
    Get-Process -Name $tool -ErrorAction SilentlyContinue | Stop-Process -Force
}

# Clear or create the log file with UTF-8 encoding
"" | Out-File -FilePath $LOG_FILE -Encoding utf8

function Write-LogMessage {
    param([string]$Message, [bool]$ToLog = $true)
    Write-Host $Message
    if ($ToLog) {
        $Message | Out-File -FilePath $LOG_FILE -Append -Encoding utf8
    }
}

Write-LogMessage "--- Compiling control map (VBSP) ---" -ToLog $false
& "$SDK_BIN\vbsp.exe" -game $MOD_DIR bsp_unit_tests\control\$MAP_NAME
if ($LASTEXITCODE -ne 0) {
    Write-LogMessage "CRITICAL ERROR: vbsp.exe (control) failed with exit code $LASTEXITCODE."
    exit 1
}

Write-LogMessage "--- Copying control BSP to test location ---" -ToLog $false
Copy-Item "bsp_unit_tests\control\$MAP_NAME.bsp" "bsp_unit_tests\test\$MAP_NAME.bsp" -Force

Write-LogMessage "--- Compiling control map (VVIS) ---"
# Clear the native log immediately before running VVIS to capture ONLY VVIS output
Remove-Item $CONTROL_LOG -ErrorAction SilentlyContinue

$start = Get-Date
& ".\vvis.exe" -game $MOD_DIR bsp_unit_tests\control\$MAP_NAME
if ($LASTEXITCODE -ne 0) {
    Write-LogMessage "CRITICAL ERROR: vvis.exe (control) failed with exit code $LASTEXITCODE."
    exit 1
}
$controlTime = (Get-Date) - $start

# Append the native CPU log content
if (Test-Path $CONTROL_LOG) {
    Write-LogMessage "`n--- VVIS Control Native Log ---"
    Get-Content $CONTROL_LOG | Out-File -FilePath $LOG_FILE -Append -Encoding utf8
}

Write-LogMessage "--- Compiling test map (VVIS CUDA) ---"
# Clear the native log immediately before running VVIS CUDA to capture ONLY VVIS CUDA output
Remove-Item $CUDA_LOG -ErrorAction SilentlyContinue

$start = Get-Date
# Use Start-Process to allow monitoring for timeout/hang
$process = Start-Process -FilePath ".\vvis_cuda.exe" -ArgumentList "-cuda -game $MOD_DIR bsp_unit_tests\test\$MAP_NAME" -PassThru -NoNewWindow
$timedOut = $false
$maxSeconds = ($controlTime.TotalSeconds * 1.5) + ($TimeoutExtensionMinutes * 60)
$hasWarnedByExceedingControl = $false

while (-not $process.HasExited) {
    $elapsed = (Get-Date) - $start
    
    if (($elapsed.TotalSeconds -gt $controlTime.TotalSeconds) -and (-not $hasWarnedByExceedingControl)) {
        $remaining = [math]::Max(0, [math]::Round($maxSeconds - $elapsed.TotalSeconds))
        Write-LogMessage "WARNING: Test run has exceeded the control test time ($([math]::Round($controlTime.TotalSeconds))s)."
        Write-LogMessage "Reporting: Script will auto-terminate in $remaining seconds if it does not complete in time."
        $hasWarnedByExceedingControl = $true
    }

    if ($elapsed.TotalSeconds -gt $maxSeconds) {
        $process | Stop-Process -Force
        $timedOut = $true
        
        # Sample current usage snapshot
        $currentGpu = nvidia-smi --query-gpu=utilization.gpu --format="csv,noheader,nounits" 2>$null
        $currentCpu = (Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction SilentlyContinue).CounterSamples.CookedValue
        
        Write-LogMessage "CRITICAL ERROR: vvis_cuda hung or is significantly slower than control test!"
        if ($currentCpu) { Write-LogMessage "CPU Usage at time of termination: $($currentCpu.ToString('F1'))%" }
        if ($currentGpu) { Write-LogMessage "GPU Usage at time of termination: $($currentGpu.Trim())%" }
        Write-LogMessage "---"
        Write-LogMessage "Notice: You can extend the runtime by adding: -TimeoutExtensionMinutes <minutes>"
        Write-Error "CRITICAL ERROR: vvis_cuda hung or is significantly slower than control test! Terminating before completion..."
        break
    }
    Start-Sleep -Seconds 1
}

if (-not $timedOut -and $process.ExitCode -ne 0) {
    Write-LogMessage "CRITICAL ERROR: vvis_cuda.exe failed with exit code $($process.ExitCode)."
    exit 1
}

if ($timedOut) {
    $cudaTime = (Get-Date) - $start
}
else {
    $cudaTime = (Get-Date) - $start
}

# Append the native GPU log content if it exists
if (Test-Path $CUDA_LOG) {
    Write-LogMessage "`n--- VVIS CUDA Native Log ---"
    Get-Content $CUDA_LOG | Out-File -FilePath $LOG_FILE -Append -Encoding utf8
}

Write-LogMessage "`n--- Timing Summary ---"
Write-LogMessage "VVIS Control Time: $($controlTime.TotalSeconds.ToString("F2")) seconds"
if ($timedOut) {
    Write-LogMessage "VVIS CUDA Time:    Did Not Finish"
}
else {
    Write-LogMessage "VVIS CUDA Time:    $($cudaTime.TotalSeconds.ToString("F2")) seconds"
}

if ($cudaTime.TotalSeconds -gt ($controlTime.TotalSeconds - 2)) {
    Write-LogMessage "RESULT: FAIL (CUDA is not >2s faster than CPU)"
}
else {
    Write-LogMessage "RESULT: PASS"
}

if ($timedOut) {
    Write-LogMessage "--- Skipping VVIS Output Comparison due to hang ---"
}
else {
    Write-LogMessage "--- Comparing VVIS Outputs ---" -ToLog $false
    # Use /LB1 to stop comparison after 1 mismatch to save time, since we only care IF they differ
    $fcOutput = fc.exe /b /LB1 bsp_unit_tests\control\$MAP_NAME.bsp bsp_unit_tests\test\$MAP_NAME.bsp 2>&1
    $fcOutput | Select-Object -First 20 | Write-Host
    if ($LASTEXITCODE -ne 0) {
        Write-LogMessage "Warning: VVIS .bsp outputs differ! Initiating visual comparison..."
        
        $GAME_EXE = Join-Path $PSScriptRoot "..\..\hl2.exe"
        $GAME_MAPS = Join-Path $PSScriptRoot "..\..\sourcetest\maps"
        $GAME_SCREENSHOTS = Join-Path $PSScriptRoot "..\..\sourcetest\screenshots"
        # Map name is now defined at the top

        # 1. Copy control bsp
        Copy-Item "bsp_unit_tests\control\$MAP_NAME.bsp" "$GAME_MAPS\$MAP_NAME.bsp" -Force
        
        # 2. Run hl2.exe for control
        Write-LogMessage "Running hl2.exe for control screenshot..."
        $gameArgs = "-game", "sourcetest", "-novid", "+sv_cheats 1", "+mat_wireframe 3", "+map $MAP_NAME", "+cl_mouselook 0", "+r_drawportals 1", "+cl_drawhud 0", "+r_drawviewmodel 0", "+mat_leafvis 3", "+wait 1000", "+screenshot", "+quit"
        $proc = Start-Process -FilePath $GAME_EXE -ArgumentList $gameArgs -Wait -PassThru
        if ($proc.ExitCode -ne 0) {
            Write-LogMessage "CRITICAL ERROR: hl2.exe exited with code $($proc.ExitCode) during control run."
            exit 1
        }

        # 3. Move control screenshot
        $controlTga = "screenshot_control.tga"
        if (Test-Path "$GAME_SCREENSHOTS\${MAP_NAME}0000.tga") {
            Move-Item "$GAME_SCREENSHOTS\${MAP_NAME}0000.tga" $controlTga -Force
        }
        else {
            Write-LogMessage "CRITICAL ERROR: Control screenshot not found!"
            exit 1
        }

        # 4. Copy test bsp
        Copy-Item "bsp_unit_tests\test\$MAP_NAME.bsp" "$GAME_MAPS\$MAP_NAME.bsp" -Force

        # 5. Run hl2.exe for test
        Write-LogMessage "Running hl2.exe for test screenshot..."
        $gameArgs = "-game", "sourcetest", "-novid", "+sv_cheats 1", "+mat_wireframe 3", "+map $MAP_NAME", "+cl_mouselook 0", "+r_drawportals 1", "+cl_drawhud 0", "+r_drawviewmodel 0", "+mat_leafvis 3", "+wait 1000", "+screenshot", "+quit"
        $proc = Start-Process -FilePath $GAME_EXE -ArgumentList $gameArgs -Wait -PassThru
        if ($proc.ExitCode -ne 0) {
            Write-LogMessage "CRITICAL ERROR: hl2.exe exited with code $($proc.ExitCode) during test run."
            exit 1
        }

        # 6. Move test screenshot
        $testTga = "screenshot_test.tga"
        if (Test-Path "$GAME_SCREENSHOTS\${MAP_NAME}0000.tga") {
            Move-Item "$GAME_SCREENSHOTS\${MAP_NAME}0000.tga" $testTga -Force
        }
        else {
            Write-LogMessage "CRITICAL ERROR: Test screenshot not found!"
            exit 1
        }

        # 7. Compare screenshots
        Write-LogMessage "Comparing screenshots using tgadiff..."
        $diffTga = "screenshot_diff.tga"
        $diffOutput = .\tgadiff.exe $controlTga $testTga $diffTga 2>&1
        $diffOutput | Out-File -FilePath $LOG_FILE -Append -Encoding utf8
        
        $diffMatch = $diffOutput | Select-String "Difference: ([\d\.]+)%"
        if ($diffMatch) {
            $percentDiff = [double]$diffMatch.Matches.Groups[1].Value
            Write-LogMessage "Visual Difference: $percentDiff%"
            if ($percentDiff -ge 15.0) {
                # 85% match means < 15% difference
                Write-LogMessage "RESULT: FAIL (Visual difference $percentDiff% >= 15%)"
                exit 1
            }
            else {
                Write-LogMessage "RESULT: PASS (Visual difference $percentDiff% < 15%)"
            }
        }
        else {
            Write-LogMessage "Warning: Could not parse tgadiff output. Assuming major difference."
            exit 1
        }
    }
    else {
        Write-LogMessage "VVIS .bsp outputs match."
    }
}

if ($timedOut) {
    exit 1
}
