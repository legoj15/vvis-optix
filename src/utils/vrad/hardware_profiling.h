//========================================================================
// hardware_profiling.h - Hardware resource usage profiling for VRAD RTX
//========================================================================

#ifndef HARDWARE_PROFILING_H
#define HARDWARE_PROFILING_H

// Initialize profiling (takes baseline snapshot)
void HardwareProfile_Init();

// Take a snapshot and log current hardware usage with a label
void HardwareProfile_Snapshot(const char *pszLabel);

// Print peak usage summary (called at end of run)
void HardwareProfile_PrintSummary();

// Auto-compute optimal GPU ray batch size based on available physical RAM.
// Returns rays-per-thread value clamped to [10000, 2000000].
// numThreads = worker thread count for BuildFacelights.
int AutoComputeGPURayBatchSize(int numThreads);

//-----------------------------------------------------------------------------
// GPU Host Memory Tracking
//
// Lightweight per-category ledger for tracking host-side memory allocations
// made by the GPU pipeline. Call Track() on alloc (+bytes) and free (-bytes).
// Thread-safe via interlocked operations.
//-----------------------------------------------------------------------------

// Track a host memory allocation or free for a named category.
// deltaBytes > 0 for allocation, < 0 for free.
void GPUHostMem_Track(const char *category, long long deltaBytes);

// Print all tracked categories with current and peak usage.
void GPUHostMem_Print();

// Reset all tracking to zero.
void GPUHostMem_Reset();

#endif // HARDWARE_PROFILING_H
