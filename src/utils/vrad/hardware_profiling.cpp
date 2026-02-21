//========================================================================
// hardware_profiling.cpp - Hardware resource usage profiling for VRAD RTX
//
// Tracks system RAM, VRAM, CPU utilization, and GPU utilization at
// key phase boundaries during lighting compilation.
//========================================================================

#include "tier0/dbg.h"
#include "tier0/platform.h"

#include "hardware_profiling.h"

#ifdef _WIN32
#include <windows.h>

// Avoid including psapi.h which conflicts with Source SDK on newer Windows
// SDKs. Declare the struct and import the K32 function directly from kernel32.
typedef struct {
  DWORD cb;
  DWORD PageFaultCount;
  SIZE_T PeakWorkingSetSize;
  SIZE_T WorkingSetSize;
  SIZE_T QuotaPeakPagedPoolUsage;
  SIZE_T QuotaPagedPoolUsage;
  SIZE_T QuotaPeakNonPagedPoolUsage;
  SIZE_T QuotaNonPagedPoolUsage;
  SIZE_T PagefileUsage;
  SIZE_T PeakPagefileUsage;
} HWP_PROCESS_MEMORY_COUNTERS;

extern "C" __declspec(dllimport) BOOL WINAPI K32GetProcessMemoryInfo(
    HANDLE Process, HWP_PROCESS_MEMORY_COUNTERS *ppsmemCounters, DWORD cb);
#endif

#ifdef VRAD_RTX_CUDA_SUPPORT
#include "raytrace_optix.h"
#endif

//-----------------------------------------------------------------------------
// Internal state
//-----------------------------------------------------------------------------
static double s_flInitTime = 0.0;
static double s_flLastSnapshotTime = 0.0;
static FILETIME s_ftLastKernelTime = {};
static FILETIME s_ftLastUserTime = {};

// Peak tracking
static size_t s_nPeakRAM_MB = 0;
static size_t s_nPeakWorkingSet_MB = 0;
static size_t s_nPeakVRAMUsed_MB = 0;
static float s_flPeakCPU = 0.0f;

//-----------------------------------------------------------------------------
// GPU Host Memory Tracker — per-category ledger
//-----------------------------------------------------------------------------
static const int MAX_HOSTMEM_CATEGORIES = 32;

struct HostMemCategory {
  const char *name;           // Category name (static string, not owned)
  volatile long long current; // Current allocated bytes
  volatile long long peak;    // Peak allocated bytes
};

static HostMemCategory s_hostMemCategories[MAX_HOSTMEM_CATEGORIES];
static volatile long s_numHostMemCategories = 0;
static volatile long long s_hostMemTotalCurrent = 0;
static volatile long long s_hostMemTotalPeak = 0;

// NVML function pointers (dynamically loaded)
typedef int (*nvmlInit_t)(void);
typedef int (*nvmlShutdown_t)(void);
typedef int (*nvmlDeviceGetHandleByIndex_t)(unsigned int, void **);
typedef int (*nvmlDeviceGetUtilizationRates_t)(void *,
                                               struct nvmlUtilization_st *);

struct nvmlUtilization_st {
  unsigned int gpu;
  unsigned int memory;
};

static HMODULE s_hNVML = NULL;
static nvmlDeviceGetUtilizationRates_t s_pfnGetUtilization = NULL;
static void *s_nvmlDevice = NULL;

//-----------------------------------------------------------------------------
// Helpers
//-----------------------------------------------------------------------------
static size_t GetProcessRAM_MB(size_t *pPeakMB = nullptr) {
#ifdef _WIN32
  HWP_PROCESS_MEMORY_COUNTERS pmc = {};
  pmc.cb = sizeof(pmc);
  if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    if (pPeakMB)
      *pPeakMB = pmc.PeakWorkingSetSize / (1024 * 1024);
    return pmc.WorkingSetSize / (1024 * 1024);
  }
#endif
  if (pPeakMB)
    *pPeakMB = 0;
  return 0;
}

static float ComputeCPUUsage() {
#ifdef _WIN32
  FILETIME ftCreation, ftExit, ftKernel, ftUser;
  if (!GetProcessTimes(GetCurrentProcess(), &ftCreation, &ftExit, &ftKernel,
                       &ftUser)) {
    return -1.0f;
  }

  // Convert to 64-bit values
  ULARGE_INTEGER kernelNow, userNow, kernelLast, userLast;
  kernelNow.LowPart = ftKernel.dwLowDateTime;
  kernelNow.HighPart = ftKernel.dwHighDateTime;
  userNow.LowPart = ftUser.dwLowDateTime;
  userNow.HighPart = ftUser.dwHighDateTime;
  kernelLast.LowPart = s_ftLastKernelTime.dwLowDateTime;
  kernelLast.HighPart = s_ftLastKernelTime.dwHighDateTime;
  userLast.LowPart = s_ftLastUserTime.dwLowDateTime;
  userLast.HighPart = s_ftLastUserTime.dwHighDateTime;

  // CPU time delta (in 100ns units)
  ULONGLONG cpuDelta = (kernelNow.QuadPart - kernelLast.QuadPart) +
                       (userNow.QuadPart - userLast.QuadPart);

  // Wall time delta
  double wallDelta = Plat_FloatTime() - s_flLastSnapshotTime;
  if (wallDelta <= 0.0)
    return -1.0f;

  // Convert CPU delta from 100ns to seconds
  double cpuSeconds = (double)cpuDelta / 10000000.0;

  // Normalize by number of logical cores
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  int nCores = si.dwNumberOfProcessors;
  if (nCores < 1)
    nCores = 1;

  float usage = (float)(cpuSeconds / (wallDelta * nCores)) * 100.0f;
  if (usage > 100.0f)
    usage = 100.0f;

  // Update last times
  s_ftLastKernelTime = ftKernel;
  s_ftLastUserTime = ftUser;
  s_flLastSnapshotTime = Plat_FloatTime();

  return usage;
#else
  return -1.0f;
#endif
}

static float GetGPUUtilization() {
  if (!s_pfnGetUtilization || !s_nvmlDevice)
    return -1.0f;

  nvmlUtilization_st util = {};
  int result = s_pfnGetUtilization(s_nvmlDevice, &util);
  if (result != 0) // NVML_SUCCESS == 0
    return -1.0f;

  return (float)util.gpu;
}

static void InitNVML() {
#ifdef _WIN32
  s_hNVML = LoadLibraryA("nvml.dll");
  if (!s_hNVML) {
    // Try the typical NVIDIA driver path
    char path[MAX_PATH];
    if (GetEnvironmentVariableA("ProgramFiles", path, MAX_PATH)) {
      strcat_s(path, "\\NVIDIA Corporation\\NVSMI\\nvml.dll");
      s_hNVML = LoadLibraryA(path);
    }
  }
  // Also try the System32 path where newer drivers put it
  if (!s_hNVML) {
    char path[MAX_PATH];
    if (GetSystemDirectoryA(path, MAX_PATH)) {
      strcat_s(path, "\\nvml.dll");
      s_hNVML = LoadLibraryA(path);
    }
  }

  if (!s_hNVML) {
    return; // NVML not available, GPU utilization will be N/A
  }

  auto pfnInit = (nvmlInit_t)GetProcAddress(s_hNVML, "nvmlInit_v2");
  if (!pfnInit)
    pfnInit = (nvmlInit_t)GetProcAddress(s_hNVML, "nvmlInit");

  auto pfnGetHandle = (nvmlDeviceGetHandleByIndex_t)GetProcAddress(
      s_hNVML, "nvmlDeviceGetHandleByIndex_v2");
  if (!pfnGetHandle)
    pfnGetHandle = (nvmlDeviceGetHandleByIndex_t)GetProcAddress(
        s_hNVML, "nvmlDeviceGetHandleByIndex");

  s_pfnGetUtilization = (nvmlDeviceGetUtilizationRates_t)GetProcAddress(
      s_hNVML, "nvmlDeviceGetUtilizationRates");

  if (!pfnInit || !pfnGetHandle || !s_pfnGetUtilization) {
    FreeLibrary(s_hNVML);
    s_hNVML = NULL;
    s_pfnGetUtilization = NULL;
    return;
  }

  if (pfnInit() != 0) {
    FreeLibrary(s_hNVML);
    s_hNVML = NULL;
    s_pfnGetUtilization = NULL;
    return;
  }

  if (pfnGetHandle(0, &s_nvmlDevice) != 0) {
    s_nvmlDevice = NULL;
    s_pfnGetUtilization = NULL;
  }
#endif
}

static void ShutdownNVML() {
#ifdef _WIN32
  if (s_hNVML) {
    auto pfnShutdown = (nvmlShutdown_t)GetProcAddress(s_hNVML, "nvmlShutdown");
    if (pfnShutdown)
      pfnShutdown();
    FreeLibrary(s_hNVML);
    s_hNVML = NULL;
  }
  s_pfnGetUtilization = NULL;
  s_nvmlDevice = NULL;
#endif
}

//-----------------------------------------------------------------------------
// Public API
//-----------------------------------------------------------------------------
void HardwareProfile_Init() {
  s_flInitTime = Plat_FloatTime();
  s_flLastSnapshotTime = s_flInitTime;
  s_nPeakRAM_MB = 0;
  s_nPeakWorkingSet_MB = 0;
  s_nPeakVRAMUsed_MB = 0;
  GPUHostMem_Reset();
  s_flPeakCPU = 0.0f;

#ifdef _WIN32
  // Capture initial CPU times
  FILETIME ftCreation, ftExit;
  GetProcessTimes(GetCurrentProcess(), &ftCreation, &ftExit,
                  &s_ftLastKernelTime, &s_ftLastUserTime);
#endif

  InitNVML();

  // Take initial snapshot
  HardwareProfile_Snapshot("Initialization");
}

void HardwareProfile_Snapshot(const char *pszLabel) {
  // --- System RAM (current + OS peak) ---
  size_t peakWorkingSetMB = 0;
  size_t ramMB = GetProcessRAM_MB(&peakWorkingSetMB);
  if (ramMB > s_nPeakRAM_MB)
    s_nPeakRAM_MB = ramMB;
  if (peakWorkingSetMB > s_nPeakWorkingSet_MB)
    s_nPeakWorkingSet_MB = peakWorkingSetMB;

  // --- VRAM ---
  size_t vramFreeMB = 0, vramTotalMB = 0;
  size_t vramUsedMB = 0;
  bool bHasVRAM = false;

#ifdef VRAD_RTX_CUDA_SUPPORT
  if (RayTraceOptiX::IsInitialized()) {
    bHasVRAM = RayTraceOptiX::GetVRAMUsage(vramFreeMB, vramTotalMB);
    if (bHasVRAM) {
      vramUsedMB = vramTotalMB - vramFreeMB;
      if (vramUsedMB > s_nPeakVRAMUsed_MB)
        s_nPeakVRAMUsed_MB = vramUsedMB;
    }
  }
#endif

  // --- CPU Usage ---
  float cpuUsage = ComputeCPUUsage();
  if (cpuUsage > s_flPeakCPU)
    s_flPeakCPU = cpuUsage;

  // --- GPU Usage ---
  float gpuUsage = GetGPUUtilization();

  // --- Print ---
  Msg("[HW Profile] %s:\n", pszLabel);
  Msg("  System RAM:  %zu MB (Peak: %zu MB, OS Peak: %zu MB)\n", ramMB,
      s_nPeakRAM_MB, peakWorkingSetMB);

  if (bHasVRAM) {
    float vramPct = vramTotalMB > 0
                        ? (float)vramUsedMB / (float)vramTotalMB * 100.0f
                        : 0.0f;
    Msg("  GPU VRAM:    %zu MB / %zu MB (%.1f%%)\n", vramUsedMB, vramTotalMB,
        vramPct);
  } else {
    Msg("  GPU VRAM:    N/A\n");
  }

  if (cpuUsage >= 0.0f)
    Msg("  CPU Usage:   %.1f%%\n", cpuUsage);
  else
    Msg("  CPU Usage:   N/A\n");

  if (gpuUsage >= 0.0f)
    Msg("  GPU Usage:   %.1f%%\n", gpuUsage);
  else
    Msg("  GPU Usage:   N/A (NVML not available)\n");

  // --- GPU Host Memory Breakdown ---
  GPUHostMem_Print();
}

void HardwareProfile_PrintSummary() {
  Msg("\nHardware Usage Summary:\n");
  Msg("----------------------\n");
  Msg("Peak System RAM:       %zu MB (sampled)\n", s_nPeakRAM_MB);
  Msg("Peak Working Set (OS): %zu MB (true high-water mark)\n",
      s_nPeakWorkingSet_MB);

  // GPU host memory peak
  long long hostMemPeakBytes = s_hostMemTotalPeak;
  if (hostMemPeakBytes > 0) {
    Msg("Peak GPU Host Mem:     %lld MB (tracked)\n",
        hostMemPeakBytes / (1024LL * 1024LL));
  }

  if (s_nPeakVRAMUsed_MB > 0)
    Msg("Peak GPU VRAM:         %zu MB\n", s_nPeakVRAMUsed_MB);
  else
    Msg("Peak GPU VRAM:         N/A\n");

  if (s_flPeakCPU > 0.0f)
    Msg("Peak CPU Usage:        %.1f%%\n", s_flPeakCPU);
  else
    Msg("Peak CPU Usage:        N/A\n");

  Msg("----------------------\n");

  ShutdownNVML();
}

//-----------------------------------------------------------------------------
// GPU Host Memory Tracker — Implementation
//-----------------------------------------------------------------------------

// Find or create a category slot. Returns index, or -1 if full.
static int FindOrCreateCategory(const char *name) {
  // First, look for existing
  long count = s_numHostMemCategories;
  for (long i = 0; i < count; i++) {
    if (s_hostMemCategories[i].name == name) // pointer compare (static strings)
      return (int)i;
  }

  // Not found — create a new slot (interlocked to avoid races)
  long idx = InterlockedIncrement(&s_numHostMemCategories) - 1;
  if (idx >= MAX_HOSTMEM_CATEGORIES) {
    // Out of slots — decrement back and return -1
    InterlockedDecrement(&s_numHostMemCategories);
    return -1;
  }
  s_hostMemCategories[idx].name = name;
  s_hostMemCategories[idx].current = 0;
  s_hostMemCategories[idx].peak = 0;
  return (int)idx;
}

void GPUHostMem_Track(const char *category, long long deltaBytes) {
  if (deltaBytes == 0)
    return;

  int idx = FindOrCreateCategory(category);
  if (idx < 0)
    return;

  HostMemCategory &cat = s_hostMemCategories[idx];

  // Interlocked add to current
  long long newVal =
      InterlockedExchangeAdd64(&cat.current, deltaBytes) + deltaBytes;

  // Update peak if new high
  long long oldPeak = cat.peak;
  while (newVal > oldPeak) {
    long long prev = InterlockedCompareExchange64(&cat.peak, newVal, oldPeak);
    if (prev == oldPeak)
      break;
    oldPeak = prev;
  }

  // Update global total
  long long newTotal =
      InterlockedExchangeAdd64(&s_hostMemTotalCurrent, deltaBytes) + deltaBytes;

  long long oldTotalPeak = s_hostMemTotalPeak;
  while (newTotal > oldTotalPeak) {
    long long prev = InterlockedCompareExchange64(&s_hostMemTotalPeak, newTotal,
                                                  oldTotalPeak);
    if (prev == oldTotalPeak)
      break;
    oldTotalPeak = prev;
  }
}

void GPUHostMem_Print() {
  long count = s_numHostMemCategories;
  if (count <= 0)
    return;

  Msg("  GPU Host Memory Breakdown:\n");
  long long totalCurrent = 0;
  long long totalPeak = 0;
  for (long i = 0; i < count; i++) {
    const HostMemCategory &cat = s_hostMemCategories[i];
    long long cur = cat.current;
    long long pk = cat.peak;
    totalCurrent += cur;
    totalPeak += pk;

    // Only show categories that have had allocations
    if (pk > 0) {
      Msg("    %-28s %6lld MB  (peak: %lld MB)\n", cat.name,
          cur / (1024LL * 1024LL), pk / (1024LL * 1024LL));
    }
  }
  Msg("    %-28s %6lld MB  (peak: %lld MB)\n", "TOTAL TRACKED",
      totalCurrent / (1024LL * 1024LL), totalPeak / (1024LL * 1024LL));
}

void GPUHostMem_Reset() {
  for (int i = 0; i < MAX_HOSTMEM_CATEGORIES; i++) {
    s_hostMemCategories[i].name = nullptr;
    s_hostMemCategories[i].current = 0;
    s_hostMemCategories[i].peak = 0;
  }
  s_numHostMemCategories = 0;
  s_hostMemTotalCurrent = 0;
  s_hostMemTotalPeak = 0;
}

//-----------------------------------------------------------------------------
// Auto-compute optimal GPU ray batch size from available physical memory.
//
// Memory model per thread (threshold counts ray entries):
//   RayBatch:            36 bytes/ray (origin, direction, tmin, tmax, skip_id)
//   RayResult:           20 bytes/ray (allocated during flush, freed
//   immediately) Subtotal:            ~56 bytes/ray steady-state
//
//   CUtlVector doubling: peak memory during growth is ~3× steady state
//   (old buffer + new buffer coexist during realloc-copy).
//   Effective per-ray: ~56 × 3 ≈ 170 bytes
//
// Budget: 25% of available physical RAM (leaves 75% for VRAD's base data:
// patches, lightmaps, cluster light lists, face data, etc.)
//-----------------------------------------------------------------------------
int AutoComputeGPURayBatchSize(int numThreads) {
  const int MIN_BATCH = 50000;
  const int MAX_BATCH = 2000000; // 2M rays — TraceBatch internally chunks at 1M
  const int FALLBACK = 500000;   // 500K conservative fallback
  const int BYTES_PER_RAY = 56;  // 36 RayBatch + 20 RayResult transient

  if (numThreads < 1)
    numThreads = 1;

#ifdef _WIN32
  // Query available physical RAM
  MEMORYSTATUSEX memInfo = {};
  memInfo.dwLength = sizeof(memInfo);
  if (!GlobalMemoryStatusEx(&memInfo)) {
    Msg("  AutoBatch: GlobalMemoryStatusEx failed, using fallback %d\n",
        FALLBACK);
    return FALLBACK;
  }

  unsigned long long availPhysBytes = memInfo.ullAvailPhys;
  unsigned long long totalPhysBytes = memInfo.ullTotalPhys;

  // Budget: 50% of available RAM. Ray buffers are transient and small
  // relative to face data — being generous here improves GPU throughput
  // by reducing flush frequency and mutex contention.
  unsigned long long budget = availPhysBytes / 2;

  // Compute per-thread ray count
  unsigned long long perThread =
      budget / ((unsigned long long)numThreads * BYTES_PER_RAY);

  int result = (int)min(perThread, (unsigned long long)MAX_BATCH);
  result = max(result, MIN_BATCH);

  // Log details for diagnostics
  double budgetGB = (double)budget / (1024.0 * 1024.0 * 1024.0);
  double perThreadMB = (double)result * BYTES_PER_RAY / (1024.0 * 1024.0);
  Msg("  AutoBatch: %.1f GB avail / %.1f GB total, budget=%.1f GB, "
      "%d threads -> %d rays/thread (%.0f MB/thread peak)\n",
      (double)availPhysBytes / (1024.0 * 1024.0 * 1024.0),
      (double)totalPhysBytes / (1024.0 * 1024.0 * 1024.0), budgetGB, numThreads,
      result, perThreadMB);

  return result;
#else
  // Non-Windows: use fallback
  return FALLBACK;
#endif
}
