#include "pacifier.h"
#include "basetypes.h"
#include "tier0/dbg.h"
#include <stdio.h>


#include <windows.h>

static volatile long g_LastPacifierDrawn = -1;
static bool g_bPacifierSuppressed = false;

#define clamp(a, b, c) ((a) > (c) ? (c) : ((a) < (b) ? (b) : (a)))

void StartPacifier(char const *pPrefix) {
  Msg("%s", pPrefix);
  g_LastPacifierDrawn = -1;
  UpdatePacifier(0.001f);
}

void UpdatePacifier(float flPercent) {
  int iCur = (int)(flPercent * 40.0f);
  iCur = clamp(iCur, 0, 40);

  if (g_bPacifierSuppressed)
    return;

  // Atomically advance g_LastPacifierDrawn to iCur.
  // Only the thread that successfully advances it prints the output.
  long oldVal = g_LastPacifierDrawn;
  while (oldVal < iCur) {
    long prev =
        InterlockedCompareExchange(&g_LastPacifierDrawn, (long)iCur, oldVal);
    if (prev == oldVal) {
      // We won the race — print milestones from oldVal+1 to iCur
      for (int i = (int)oldVal + 1; i <= iCur; i++) {
        if (!(i % 4)) {
          Msg("%d", i / 4);
        } else {
          if (i != 40) {
            Msg(".");
          }
        }
      }
      break;
    }
    // Another thread advanced it — re-read and retry
    oldVal = prev;
  }
}

void EndPacifier(bool bCarriageReturn) {
  UpdatePacifier(1);

  if (bCarriageReturn && !g_bPacifierSuppressed)
    Msg("\n");
}

void SuppressPacifier(bool bSuppress) { g_bPacifierSuppressed = bSuppress; }
