//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose:
//
//=============================================================================//

#ifndef PROFILER_H
#define PROFILER_H
#ifdef _WIN32
#pragma once
#endif

class CProfiler {
public:
  CProfiler();
  ~CProfiler();

  void Initialize();
  void Shutdown();

  // Returns CPU usage as a percentage (0.0 to 100.0)
  double GetCpuUsage();

  // Returns GPU usage as a percentage (0.0 to 100.0)
  // Returns -1.0 if GPU profiling is not available
  double GetGpuUsage();

private:
  bool m_bInitialized;

  // CPU Profiling (PDH)
  void *m_hPdhQuery;   // HQUERY
  void *m_hPdhCounter; // HCOUNTER

  // GPU Profiling (NVML)
  bool m_bNvmlInitialized;
  void *m_NvmlDevice; // nvmlDevice_t
};

extern CProfiler g_Profiler;

#endif // PROFILER_H
