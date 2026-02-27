# VRAD-RTX Active Bugs

This document tracks identified bugs inside the engine that currently require script-side mitigations or future code-level resolutions.

---

## 1. The "Lock Trap" Deadlock (Bounce #13 Hang)

### Symptoms
When running `vrad_rtx.exe -cuda` in specific automated or piped environments (like Windows PowerShell 5.1 tests), the process will completely hang during the Multithreaded Radiosity phase (e.g., at `Bounce #13`).
- **Log Signature**: Print progress stops exactly halfway through a `0...1...2...x` block inside `RunThreadsOn(uiPatchCount, true, GatherLight)`.

### Root Cause
An architectural deadlock involving the Windows IO Pipe Buffer and the Source engine thread dispatcher.
1. The global thread dispatcher `GetThreadWork()` holds a critical section: `ThreadLock()`.
2. Inside that lock, it calls `UpdatePacifier()` to print progress dots.
3. `UpdatePacifier` performs blocking synchronous I/O (`Msg()`).
4. By Bounce #12 or #13, the process has dumped enough logging (especially during GPU Visibility phases) to **saturate the 4KB/64KB OS Pipe Buffer**.
5. Once the OS buffer is full, the next thread to call `Msg()` blocks indefinitely, waiting for the receiving script to drain the pipe.
6. Because the thread is blocked while holding the `crit` section, all other threads stall attempting to grab `GetThreadWork()`. The main thread stalls on `WaitForMultipleObjects(INFINITE)`, creating a permanent deadlock.

### Current Mitigation (Script-Side)
- The test scripts (`test_vrad_optix.ps1`) have been updated to use asynchronous stream readers (`Register-ObjectEvent` or background tasks in PowerShell 7) to continuously aggressively drain `stdout`/`stderr` without blocking execution. 
- Using modern PowerShell 7 (`pwsh`) ensures better async buffer draining.

### Required Resolution (Engine-Side)
To prevent this in all shell environments, `UpdatePacifier` must be decoupled from the thread dispatcher lock. 
**Action**: Move the call to `UpdatePacifier(...)` **outside** the `ThreadLock()` / `ThreadUnlock()` blocks within `src/utils/common/threads.cpp`.
