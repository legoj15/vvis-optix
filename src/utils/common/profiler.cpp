//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose:
//
//=============================================================================//

#include "profiler.h"
#include "tier0/dbg.h"
#include <pdh.h>
#include <pdhmsg.h>
#include <windows.h>

// NVML includes - try to include dynamically or use a local definition if
// headers aren't easily available in include path For now, we'll assume the
// build system can find nvml.h or we'll define what we need. To avoid
// compilation dependencies on having the CUDA toolkit installed in a specific
// location for everyone, we can use LoadLibrary for NVML, but the plan said to
// link against nvml.lib. Let's try to include nvml.h. If it fails to build,
// we'll fix the include path.
#include <nvml.h>

CProfiler g_Profiler;

CProfiler::CProfiler() {
  m_bInitialized = false;
  m_hPdhQuery = NULL;
  m_hPdhCounter = NULL;
  m_bNvmlInitialized = false;
  m_NvmlDevice = NULL;
}

CProfiler::~CProfiler() { Shutdown(); }

void CProfiler::Initialize() {
  if (m_bInitialized)
    return;

  // --- CPU Profiling Setup (PDH) ---
  PDH_STATUS status = PdhOpenQuery(NULL, 0, (HQUERY *)&m_hPdhQuery);
  if (status == ERROR_SUCCESS) {
    // Add counter for total processor time
    // We use the "Processor(_Total)\% Processor Time" counter (English)
    // Alternatively, PdhAddEnglishCounter is safer for non-English Windows
    status = PdhAddEnglishCounter((HQUERY)m_hPdhQuery,
                                  "\\Processor(_Total)\\% Processor Time", 0,
                                  (HCOUNTER *)&m_hPdhCounter);
    if (status != ERROR_SUCCESS) {
      // Fallback to localized name if english fails (though English API should
      // work)
      status = PdhAddCounter((HQUERY)m_hPdhQuery,
                             "\\Processor(_Total)\\% Processor Time", 0,
                             (HCOUNTER *)&m_hPdhCounter);
    }

    if (status == ERROR_SUCCESS) {
      // Collect initial data
      PdhCollectQueryData((HQUERY)m_hPdhQuery);
    } else {
      m_hPdhQuery = NULL;
      m_hPdhCounter = NULL;
      Warning("Failed to initialize CPU profiling counter. Error: 0x%x\n",
              status);
    }
  } else {
    Warning("Failed to open PDH query for CPU profiling. Error: 0x%x\n",
            status);
  }

  // --- GPU Profiling Setup (NVML) ---
  nvmlReturn_t nvmlResult = nvmlInit();
  if (nvmlResult == NVML_SUCCESS) {
    // Get handle to the first GPU (index 0)
    nvmlResult = nvmlDeviceGetHandleByIndex(0, (nvmlDevice_t *)&m_NvmlDevice);
    if (nvmlResult == NVML_SUCCESS) {
      m_bNvmlInitialized = true;
    } else {
      Warning("Failed to get NVML device handle for GPU 0: %s\n",
              nvmlErrorString(nvmlResult));
      nvmlShutdown();
    }
  } else {
    // Don't warn too loudly, user might not have an NVIDIA GPU
    // Warning( "Failed to initialize NVML: %s\n", nvmlErrorString(nvmlResult)
    // );
  }

  m_bInitialized = true;
}

void CProfiler::Shutdown() {
  if (m_hPdhQuery) {
    PdhCloseQuery((HQUERY)m_hPdhQuery);
    m_hPdhQuery = NULL;
    m_hPdhCounter = NULL;
  }

  if (m_bNvmlInitialized) {
    nvmlShutdown();
    m_bNvmlInitialized = false;
    m_NvmlDevice = NULL;
  }

  m_bInitialized = false;
}

double CProfiler::GetCpuUsage() {
  if (!m_hPdhCounter)
    return 0.0;

  PDH_FMT_COUNTERVALUE counterVal;
  PDH_STATUS status = PdhCollectQueryData((HQUERY)m_hPdhQuery);

  if (status == ERROR_SUCCESS) {
    status = PdhGetFormattedCounterValue((HCOUNTER)m_hPdhCounter,
                                         PDH_FMT_DOUBLE, NULL, &counterVal);
    if (status == ERROR_SUCCESS) {
      return counterVal.doubleValue;
    }
  }

  return 0.0;
}

double CProfiler::GetGpuUsage() {
  if (!m_bNvmlInitialized || !m_NvmlDevice)
    return -1.0;

  // Try to specific process utilization first (Compute/CUDA isolation)
  unsigned int pid = GetCurrentProcessId();
  nvmlProcessUtilizationSample_t *pSamples = NULL;
  unsigned int sampleCount = 0;
  unsigned long long lastSeenTimestamp = 0;

  // First call to get count
  nvmlReturn_t result = nvmlDeviceGetProcessUtilization(
      (nvmlDevice_t)m_NvmlDevice, NULL, &sampleCount, lastSeenTimestamp);

  if (result == NVML_ERROR_INSUFFICIENT_SIZE) {
    pSamples = new nvmlProcessUtilizationSample_t[sampleCount];
    if (pSamples) {
      // Just to be safe, checking allocation
      result =
          nvmlDeviceGetProcessUtilization((nvmlDevice_t)m_NvmlDevice, pSamples,
                                          &sampleCount, lastSeenTimestamp);
    }
  }

  double usage = -1.0;

  if (result == NVML_SUCCESS && pSamples) {
    // We successfully got process samples. Defaults to 0 if we are not found
    // (idle).
    usage = 0.0;
    bool bFound = false;
    for (unsigned int i = 0; i < sampleCount; i++) {
      if (pSamples[i].pid == pid) {
        // smUtil returns percentage of time SM was active
        usage += (double)pSamples[i].smUtil;
        bFound = true;
      }
    }
    // Note: If we didn't find our PID, it means we had 0 utilization in the
    // last window, so usage=0.0 is correct.
  }

  if (pSamples) {
    delete[] pSamples;
  }

  // Fallback to global utilization if process-specific query failed (e.g. not
  // supported)
  if (usage < 0.0) {
    nvmlUtilization_t utilization;
    result =
        nvmlDeviceGetUtilizationRates((nvmlDevice_t)m_NvmlDevice, &utilization);

    if (result == NVML_SUCCESS) {
      return (double)utilization.gpu;
    }
  } else {
    return usage;
  }

  return -1.0;
}
