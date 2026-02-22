# VRAD-RTX Three-Way Test Harness
# Shared engine for all validation tests. Called by run_tests.ps1 with a
# per-test configuration hashtable.
#
# This file contains ALL the logic that was previously duplicated across
# test_vrad_optix.ps1, test_vrad_optix_quick.ps1, test_vrad_optix_props.ps1,
# test_vrad_optix_radiosity.ps1, and test_vrad_optix_supersampling.ps1.

param(
    [Parameter(Mandatory)][hashtable]$TestConfig,
    [int]$TimeoutExtensionMinutes = 0,
    [switch]$SkipVisualCheck = $false
)

# --- Unpack test config ---
$MAP_NAME = $TestConfig.MapName
$EXTRA_ARGS = $TestConfig.ExtraArgs          # array of strings
$TIMEOUT_MULT = $TestConfig.TimeoutMultiplier
$REF_CTRL_TOL = $TestConfig.RefCtrlTolerance
$CTRL_TEST_TOL = $TestConfig.CtrlTestTolerance
$LIGHTMAP_THRESH = $TestConfig.LightmapThreshold
$ARCHIVE_SUFFIX = $TestConfig.ArchiveSuffix

# --- Constants ---
$MOD_DIR = "..\..\mod_hl2mp"
$LOG_FILE = "test_vrad_optix_$ARCHIVE_SUFFIX.log"

$REF_DIR = "bsp_unit_tests\reference"
$CONTROL_DIR = "bsp_unit_tests\control"
$TEST_DIR = "bsp_unit_tests\test"

$REF_LOG = "$REF_DIR\$MAP_NAME.log"
$CONTROL_LOG = "$CONTROL_DIR\$MAP_NAME.log"
$CUDA_LOG = "$TEST_DIR\$MAP_NAME.log"

$GAME_EXE = "E:\Steam\steamapps\common\Source SDK Base 2013 Multiplayer\hl2.exe"
$GAME_MAPS = "E:\Steam\steamapps\common\Source SDK Base 2013 Multiplayer\sourcetest\maps"
$GAME_SCREENSHOTS = "E:\Steam\steamapps\common\Source SDK Base 2013 Multiplayer\sourcetest\screenshots"

# --- Process cleanup ---
$tools = @("vbsp", "vvis", "vrad", "vvis_cuda", "vrad_rtx")
foreach ($tool in $tools) {
    Get-Process -Name $tool -ErrorAction SilentlyContinue | Stop-Process -Force
}

# --- Logging ---
"" | Out-File -FilePath $LOG_FILE -Encoding utf8

function Write-LogMessage {
    param([string]$Message, [bool]$ToLog = $true)
    Write-Host $Message
    if ($ToLog) {
        $Message | Out-File -FilePath $LOG_FILE -Append -Encoding utf8
    }
}

# --- Screenshot helper ---
function Take-Screenshot {
    param([string]$BspPath, [string]$TargetTga)

    Write-LogMessage "Taking screenshot for $BspPath..."

    # 1. Copy bsp
    Copy-Item $BspPath "$GAME_MAPS\$MAP_NAME.bsp" -Force

    # 2. Run hl2.exe
    $gameArgs = "-game", "sourcetest", "-novid", "-sw", "-w", "2560", "-h", "1440", "+sv_cheats 1", "+map $MAP_NAME", "+cl_mouselook 0", "+cl_drawhud 0", "+r_drawviewmodel 0", "+mat_fullbright 2", "+wait 1000", "+screenshot", "+quit"
    $proc = Start-Process -FilePath $GAME_EXE -ArgumentList $gameArgs -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        Write-LogMessage "CRITICAL ERROR: hl2.exe exited with code $($proc.ExitCode) for map $BspPath."
        throw "FAIL"
    }

    # 3. Move screenshot
    if (Test-Path "$GAME_SCREENSHOTS\${MAP_NAME}0000.tga") {
        Move-Item "$GAME_SCREENSHOTS\${MAP_NAME}0000.tga" $TargetTga -Force
    }
    else {
        Write-LogMessage "CRITICAL ERROR: Screenshot not found for $BspPath!"
        throw "FAIL"
    }
}

# ============================================================
#  Main test flow
# ============================================================
$testFailed = $false
try {

    Write-LogMessage "=== Test: $ARCHIVE_SUFFIX ($MAP_NAME) ==="

    # Ensure directories exist
    if (!(Test-Path $REF_DIR)) { New-Item -ItemType Directory -Path $REF_DIR | Out-Null }
    if (!(Test-Path $CONTROL_DIR)) { New-Item -ItemType Directory -Path $CONTROL_DIR | Out-Null }
    if (!(Test-Path $TEST_DIR)) { New-Item -ItemType Directory -Path $TEST_DIR | Out-Null }

    # Refresh maps from source
    $MAP_SRC = "bsp_unit_tests\$MAP_NAME.bsp"
    if (!(Test-Path $MAP_SRC)) {
        Write-LogMessage "CRITICAL ERROR: Source map $MAP_SRC not found!"
        throw "FAIL"
    }
    Copy-Item $MAP_SRC "$CONTROL_DIR\$MAP_NAME.bsp" -Force
    Copy-Item $MAP_SRC "$TEST_DIR\$MAP_NAME.bsp" -Force

    $HasReference = Test-Path ".\vrad.exe"

    # --- Phase 1: Reference Run (vrad.exe, CPU only) ---
    # Skip recompilation if the source BSP and test manifest haven't changed.
    # We hash the manifest (vrad_tests.psd1) because per-test parameters like
    # ExtraArgs and tolerances live there, not in this shared harness script.
    $MANIFEST_PATH = Join-Path $PSScriptRoot "vrad_tests.psd1"
    $REF_HASH_FILE = "$REF_DIR\$MAP_NAME.ref_hash"
    $currentHash = (Get-FileHash $MAP_SRC -Algorithm SHA256).Hash + ":" + (Get-FileHash $MANIFEST_PATH -Algorithm SHA256).Hash
    $cachedHash = if (Test-Path $REF_HASH_FILE) { (Get-Content $REF_HASH_FILE -Raw).Trim() } else { "" }
    $needsRefCompile = ($currentHash -ne $cachedHash) -or !(Test-Path "$REF_DIR\$MAP_NAME.bsp")

    $refTime = New-TimeSpan
    $refTimeCached = $false
    if ($HasReference) {
        if ($needsRefCompile) {
            Copy-Item $MAP_SRC "$REF_DIR\$MAP_NAME.bsp" -Force
            Write-LogMessage "--- Compiling reference map (Original vrad.exe) ---"
            $fullLogPath = Join-Path (Get-Location).Path $REF_LOG
            Write-LogMessage "vrad.exe (SDK Reference) Log: $fullLogPath"
            $start = Get-Date
            & ".\vrad.exe" @EXTRA_ARGS -game $MOD_DIR "$REF_DIR\$MAP_NAME" *>$null
            if ($LASTEXITCODE -ne 0) {
                Write-LogMessage "WARNING: vrad.exe (reference) failed with exit code $LASTEXITCODE. Skipping reference comparison."
                $HasReference = $false
            }
            else {
                $refTime = (Get-Date) - $start
                $currentHash | Out-File -FilePath $REF_HASH_FILE -Encoding utf8 -NoNewline
                $refTime.TotalSeconds.ToString("F2") | Out-File -FilePath "$REF_DIR\$MAP_NAME.ref_time" -Encoding utf8 -NoNewline
            }
        }
        else {
            Write-LogMessage "Reference unchanged (BSP and script identical), using cached result."
            $refTimeFile = "$REF_DIR\$MAP_NAME.ref_time"
            if (Test-Path $refTimeFile) {
                $refTime = [TimeSpan]::FromSeconds([double](Get-Content $refTimeFile -Raw).Trim())
                $refTimeCached = $true
            }
        }
    }
    else {
        Write-LogMessage "vrad.exe not found, skipping reference pass."
    }

    # --- Phase 2: Control Run (vrad_rtx.exe CPU) ---
    Write-LogMessage "--- Compiling control map (vrad_rtx.exe CPU only) ---"
    $fullLogPath = Join-Path (Get-Location).Path $CONTROL_LOG
    Write-LogMessage "vrad_rtx.exe CPU Log: $fullLogPath"
    $start = Get-Date
    & ".\vrad_rtx.exe" @EXTRA_ARGS -game $MOD_DIR "$CONTROL_DIR\$MAP_NAME" *>$null
    if ($LASTEXITCODE -ne 0) {
        Write-LogMessage "CRITICAL ERROR: vrad_rtx.exe (control, CPU only) failed with exit code $LASTEXITCODE."
        throw "FAIL"
    }
    $controlTime = (Get-Date) - $start

    # --- Phase 3: Reference vs Control Comparison ---
    if ($HasReference) {
        Write-LogMessage "`n--- Comparing Reference vs Control (CPU Parity Check) ---"
        $pythonDiff = python bsp_diff_lightmaps.py "$REF_DIR\$MAP_NAME.bsp" "$CONTROL_DIR\$MAP_NAME.bsp" --threshold $LIGHTMAP_THRESH 2>&1
        $pythonDiff | Write-Host
        $pythonDiff | Out-File -FilePath $LOG_FILE -Append -Encoding utf8

        if ($pythonDiff -like "*All lightmaps are identical.*") {
            Write-LogMessage "RESULT: PASS (All lightmaps are identical)"
        }
        else {
            if ($SkipVisualCheck) {
                Write-LogMessage "WARNING: Lightmaps differ. Visual comparison skipped (Ref vs Control)."
            }
            else {
                Write-LogMessage "Lightmaps differ. Initiating visual comparison (Ref vs Control)..."
                if ($needsRefCompile -or !(Test-Path "screenshot_ref_$MAP_NAME.tga")) {
                    Take-Screenshot "$REF_DIR\$MAP_NAME.bsp" "screenshot_ref_$MAP_NAME.tga"
                }
                else {
                    Write-LogMessage "Using cached reference screenshot."
                }
                Take-Screenshot "$CONTROL_DIR\$MAP_NAME.bsp" "screenshot_control_$MAP_NAME.tga"

                $diffOutput = .\tgadiff.exe screenshot_ref_$MAP_NAME.tga screenshot_control_$MAP_NAME.tga screenshot_diff_ref_ctrl_$MAP_NAME.tga 2>&1
                $diffMatch = $diffOutput | Select-String "Difference: ([\d\.]+)%"
                if ($diffMatch) {
                    $percentDiff = [double]$diffMatch.Matches.Groups[1].Value
                    Write-LogMessage "Visual Difference (Ref vs Control): $percentDiff%"
                    if ($percentDiff -gt $REF_CTRL_TOL) {
                        Write-LogMessage "CRITICAL ERROR: Fundamental issue detected! CPU tests should be identical. Difference: $percentDiff% > $REF_CTRL_TOL% tolerance."
                        throw "FAIL"
                    }
                    else {
                        Write-LogMessage "CPU Visual tests passed within $REF_CTRL_TOL% margin of error."
                    }
                }
                else {
                    Write-LogMessage "CRITICAL ERROR: Could not parse tgadiff output for Ref vs Control."
                    throw "FAIL"
                }
            }
        }
    }

    # --- Phase 4: Test Run (vrad_rtx.exe -cuda) ---
    Write-LogMessage "--- Compiling test map (vrad_rtx.exe -cuda) ---"
    $fullLogPath = Join-Path (Get-Location).Path $CUDA_LOG
    Write-LogMessage "vrad_rtx.exe -cuda Log: $fullLogPath"
    $start = Get-Date

    # Use .NET Process class with event-based async output draining.
    # CRITICAL: Do NOT use Task.Run() with PowerShell scriptblocks to drain
    # stdout/stderr -- PS scriptblocks don't reliably run on .NET ThreadPool
    # threads, causing the pipe buffer (4KB on Windows) to fill up and deadlock
    # the child process when it tries to write (typically around bounce 12-13).
    # Instead, use BeginOutputReadLine/BeginErrorReadLine which use native .NET
    # async callbacks that are guaranteed to drain the pipes.
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = (Resolve-Path ".\vrad_rtx.exe").Path
    $cudaArgs = @($EXTRA_ARGS) + @("-game", $MOD_DIR, "-cuda", "$TEST_DIR\$MAP_NAME")
    $psi.Arguments = ($cudaArgs | Where-Object { $_ }) -join " "
    $psi.WorkingDirectory = (Get-Location).Path
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true

    try {
        $process = [System.Diagnostics.Process]::Start($psi)
    }
    catch {
        Write-LogMessage "CRITICAL ERROR: Failed to start vrad_rtx.exe -cuda: $($_.Exception.Message)"
        throw "FAIL"
    }

    if ($null -eq $process) {
        Write-LogMessage "CRITICAL ERROR: Failed to start vrad_rtx.exe -cuda. Process object is NULL."
        throw "FAIL"
    }

    # Use .NET event-based async draining (reliable, no deadlock)
    $cudaOutput = [System.Text.StringBuilder]::new()
    $cudaErrors = [System.Text.StringBuilder]::new()

    $outEvent = Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action {
        if ($null -ne $Event.SourceEventArgs.Data) {
            $Event.MessageData.AppendLine($Event.SourceEventArgs.Data)
        }
    } -MessageData $cudaOutput

    $errEvent = Register-ObjectEvent -InputObject $process -EventName ErrorDataReceived -Action {
        if ($null -ne $Event.SourceEventArgs.Data) {
            $Event.MessageData.AppendLine($Event.SourceEventArgs.Data)
        }
    } -MessageData $cudaErrors

    $process.BeginOutputReadLine()
    $process.BeginErrorReadLine()

    $timedOut = $false
    $maxSeconds = ($controlTime.TotalSeconds * $TIMEOUT_MULT) + ($TimeoutExtensionMinutes * 60)
    $hasWarnedByExceedingControl = $false

    while (-not $process.HasExited) {
        $elapsed = (Get-Date) - $start
        if (($elapsed.TotalSeconds -gt $controlTime.TotalSeconds) -and (-not $hasWarnedByExceedingControl)) {
            $remaining = [math]::Round($maxSeconds - $elapsed.TotalSeconds)
            Write-LogMessage "WARNING: Test run has exceeded the control test time ($([math]::Round($controlTime.TotalSeconds))s). Will terminate in ${remaining}s."
            $hasWarnedByExceedingControl = $true
        }
        if ($elapsed.TotalSeconds -gt $maxSeconds) {
            $process.Kill()
            $timedOut = $true
            Write-LogMessage "CRITICAL ERROR: vrad_rtx -cuda hung or is significantly slower than control test! Use -TimeoutExtensionMinutes <minutes> to extend wait time."
            break
        }
        Start-Sleep -Seconds 1
    }

    if (-not $timedOut) {
        $process.WaitForExit()
    }

    # Clean up event subscriptions
    Unregister-Event -SourceIdentifier $outEvent.Name
    Unregister-Event -SourceIdentifier $errEvent.Name
    Remove-Job -Job $outEvent -Force
    Remove-Job -Job $errEvent -Force

    # Save captured stderr if any
    $errText = $cudaErrors.ToString()
    if ($errText.Trim()) {
        $errText | Out-File -FilePath "$env:TEMP\vrad_rtx_cuda_err.txt" -Encoding utf8
    }

    # Final refresh to ensure exit code is captured
    $exitCode = if ($process.HasExited) { $process.ExitCode } else { $null }

    if (-not $timedOut -and ($null -eq $exitCode -or $exitCode -ne 0)) {
        $errMessage = if ($null -eq $exitCode) { "UNKNOWN (NULL)" } else { $exitCode }
        Write-LogMessage "CRITICAL ERROR: vrad_rtx.exe -cuda failed with exit code $errMessage."
        throw "FAIL"
    }
    $cudaTime = (Get-Date) - $start

    # --- Timing Summary ---
    Write-LogMessage "`n--- Timing Summary ---"
    if ($HasReference) {
        $cachedLabel = if ($refTimeCached) { " (cached)" } else { "" }
        Write-LogMessage "vrad.exe Time (Source SDK 2013 Unmodified):`t$($refTime.TotalSeconds.ToString("F2"))s$cachedLabel"
    }
    Write-LogMessage "vrad_rtx.exe Time (CPU only):`t`t`t$($controlTime.TotalSeconds.ToString("F2"))s"
    if ($timedOut) {
        Write-LogMessage "vrad_rtx.exe -cuda Time (GPU accelerated):`tDid not finish"
    }
    else {
        Write-LogMessage "vrad_rtx.exe -cuda Time (GPU accelerated):`t$($cudaTime.TotalSeconds.ToString("F2"))s"
    }

    # --- Phase 5: Control vs Test (GPU Parity Check) ---
    if (-not $timedOut) {
        Write-LogMessage "`n--- Comparing Control vs Test (GPU Parity Check) ---"
        $pythonDiff = python bsp_diff_lightmaps.py "$CONTROL_DIR\$MAP_NAME.bsp" "$TEST_DIR\$MAP_NAME.bsp" --threshold $LIGHTMAP_THRESH 2>&1
        $pythonDiff | Write-Host
        $pythonDiff | Out-File -FilePath $LOG_FILE -Append -Encoding utf8

        if ($pythonDiff -like "*All lightmaps are identical.*") {
            Write-LogMessage "RESULT: PASS (All lightmaps are identical)"
        }
        else {
            if ($SkipVisualCheck) {
                Write-LogMessage "WARNING: Lightmaps differ. Visual comparison skipped (Control vs Test)."
            }
            else {
                Write-LogMessage "Initiating visual comparison for GPU..."
                if (!(Test-Path "screenshot_control_$MAP_NAME.tga")) {
                    Take-Screenshot "$CONTROL_DIR\$MAP_NAME.bsp" "screenshot_control_$MAP_NAME.tga"
                }
                Take-Screenshot "$TEST_DIR\$MAP_NAME.bsp" "screenshot_test_$MAP_NAME.tga"

                $diffOutput = .\tgadiff.exe screenshot_control_$MAP_NAME.tga screenshot_test_$MAP_NAME.tga screenshot_diff_ctrl_test_$MAP_NAME.tga 2>&1
                $diffMatch = $diffOutput | Select-String "Difference: ([\d\.]+)%"
                if ($diffMatch) {
                    $percentDiff = [double]$diffMatch.Matches.Groups[1].Value
                    Write-LogMessage "Visual Difference (Control vs Test): $percentDiff%"
                    if ($percentDiff -ge $CTRL_TEST_TOL) {
                        Write-LogMessage "RESULT: FAIL (Visual difference $percentDiff% >= $CTRL_TEST_TOL%)"
                        throw "FAIL"
                    }
                    else {
                        Write-LogMessage "RESULT: PASS (Visual difference $percentDiff% < $CTRL_TEST_TOL%)"
                    }
                }
                else {
                    Write-LogMessage "Warning: Could not parse tgadiff output for Control vs Test."
                    throw "FAIL"
                }
            }
        }
    }

}
catch {
    $testFailed = $true
}

# --- Archive Logs ---
$ARCHIVE_DIR = "bsp_unit_test_logs"
if (!(Test-Path $ARCHIVE_DIR)) { New-Item -ItemType Directory -Path $ARCHIVE_DIR | Out-Null }

# Convert any generated TGA files to PNG for easier viewing
Write-LogMessage "Converting TGA screenshots to PNG..."
$tgaFiles = @(
    "screenshot_ref_$MAP_NAME.tga",
    "screenshot_control_$MAP_NAME.tga",
    "screenshot_test_$MAP_NAME.tga",
    "screenshot_diff_ref_ctrl_$MAP_NAME.tga",
    "screenshot_diff_ctrl_test_$MAP_NAME.tga"
)
$tgaArgs = @()
foreach ($tga in $tgaFiles) {
    if (Test-Path $tga) {
        $tgaArgs += $tga
    }
}
if ($tgaArgs.Count -gt 0) {
    python tga2png.py $tgaArgs | Write-Host
}

$timestamp = (Get-Date).ToString("yyyy-MM-ddTHH-mm-ss")
$RUN_DIR = "$ARCHIVE_DIR\${timestamp}_$ARCHIVE_SUFFIX"
New-Item -ItemType Directory -Path $RUN_DIR | Out-Null

$logMap = @{
    $LOG_FILE    = "test_$ARCHIVE_SUFFIX.log"
    $REF_LOG     = "vrad_reference.log"
    $CONTROL_LOG = "vrad_control.log"
    $CUDA_LOG    = "vrad_cuda.log"
}
foreach ($entry in $logMap.GetEnumerator()) {
    if (Test-Path $entry.Key) {
        Move-Item $entry.Key "$RUN_DIR\$($entry.Value)" -Force
    }
}

# Archive PNG screenshots alongside logs
$pngFiles = @(
    "screenshot_ref_$MAP_NAME.png",
    "screenshot_control_$MAP_NAME.png",
    "screenshot_test_$MAP_NAME.png",
    "screenshot_diff_ref_ctrl_$MAP_NAME-alpha.png",
    "screenshot_diff_ctrl_test_$MAP_NAME-alpha.png",
    "screenshot_diff_ref_ctrl_$MAP_NAME-alphacontrast.png",
    "screenshot_diff_ctrl_test_$MAP_NAME-alphacontrast.png"
)
foreach ($png in $pngFiles) {
    if (Test-Path $png) {
        Copy-Item $png "$RUN_DIR\$png" -Force
    }
}
Write-LogMessage "Logs and screenshots archived to $RUN_DIR"

if ($testFailed -or $timedOut) { exit 1 }
