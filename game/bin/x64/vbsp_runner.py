"""
VBSP Runner — invoke VBSP with -countverts and parse the vertex count.

This module provides a clean interface for the optimizer to call VBSP
as an external process and get the exact vertex count for a VMF file.

Usage:
    from vbsp_runner import count_vertices
    verts = count_vertices(vbsp_exe, vmf_path, game_dir)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional




class VBSPError(Exception):
    """Raised when VBSP execution fails."""
    pass


class VBSPNotFoundError(VBSPError):
    """Raised when the VBSP executable is not found."""
    pass


class VBSPCompileError(VBSPError):
    """Raised when VBSP reports an error during compilation."""
    pass


class VBSPParseError(VBSPError):
    """Raised when the VERTCOUNT output cannot be parsed."""
    pass


from collections import namedtuple

# Result of a vertex count operation
VertexCount = namedtuple('VertexCount', [
    'count', 'exceeded',
    'leafface_count', 'leafface_exceeded',
    'face_count', 'face_exceeded',
])

# Regex to match the machine-parseable count outputs
_VERTCOUNT_RE = re.compile(r'^VERTCOUNT:\s*(\d+)', re.MULTILINE)
_EXCEEDED_RE = re.compile(r'^VERTCOUNT_EXCEEDED:\s*true', re.MULTILINE)
_LEAFFACECOUNT_RE = re.compile(r'^LEAFFACECOUNT:\s*(\d+)', re.MULTILINE)
_LEAFFACE_EXCEEDED_RE = re.compile(r'^LEAFFACECOUNT_EXCEEDED:\s*true', re.MULTILINE)
_FACECOUNT_RE = re.compile(r'^FACECOUNT:\s*(\d+)', re.MULTILINE)
_FACE_EXCEEDED_RE = re.compile(r'^FACECOUNT_EXCEEDED:\s*true', re.MULTILINE)


def find_vbsp(search_dir: Optional[Path] = None) -> Optional[Path]:
    """Find vbsp_lmo.exe, searching in the given directory or alongside this script.
    
    Search order:
        1. Explicit search_dir if provided
        2. Same directory as this script (game/bin/x64/)
        3. Parent directories (game/bin/)
    """
    candidates = []
    
    if search_dir:
        candidates.append(Path(search_dir) / 'vbsp_lmo.exe')
    
    # Same directory as this script
    script_dir = Path(__file__).resolve().parent
    candidates.append(script_dir / 'vbsp_lmo.exe')
    
    # Parent
    candidates.append(script_dir.parent / 'vbsp_lmo.exe')
    
    for path in candidates:
        if path.is_file():
            return path
    
    return None


def count_vertices(vbsp_exe: Path,
                   vmf_path: Path,
                   game_dir: Path,
                   bin_root: Path = None,
                   timeout: int = 120,
                   verbose: bool = False,
                   max_retries: int = 2) -> VertexCount:
    """Run VBSP with -countverts to get the total vertex count of the map.
    
    Args:
        vbsp_exe: Path to vbsp_lmo.exe
        vmf_path: Path to the VMF file to check
        game_dir: Game directory for VBSP's -game flag
        bin_root: Bin root directory for VBSP's -binroot flag
        timeout: Maximum seconds to wait for VBSP (default: 120)
        verbose: If True, print VBSP's full output
        max_retries: Number of attempts (default: 2, covers Steam init failures)
        
    Returns:
        Integer vertex count from VBSP's VERTCOUNT output
        
    Raises:
        VBSPNotFoundError: If vbsp_exe doesn't exist
        VBSPCompileError: If VBSP exits with an error after all retries
        VBSPParseError: If VERTCOUNT line is not found in output
        TimeoutError: If VBSP exceeds the timeout
    """
    vbsp_exe = Path(vbsp_exe).resolve()
    vmf_path = Path(vmf_path).resolve()
    game_dir = Path(game_dir).resolve()
    
    if not vbsp_exe.is_file():
        raise VBSPNotFoundError(f"VBSP executable not found: {vbsp_exe}")
    
    if not vmf_path.is_file():
        raise VBSPError(f"VMF file not found: {vmf_path}")
    
    if not game_dir.is_dir():
        raise VBSPError(f"Game directory not found: {game_dir}")
    
    # Build command line
    cmd = [
        str(vbsp_exe),
        '-countverts',
        '-game', str(game_dir),
    ]
    if bin_root:
        cmd.extend(['-binroot', str(Path(bin_root).resolve())])
    cmd.append(str(vmf_path))
    
    env = os.environ.copy()
    if bin_root:
        # Prepend the binroot to PATH so that VBSP can load MaterialSystem.dll and other engine DLLs
        bin_path_x64 = str((Path(bin_root) / 'bin' / 'x64').resolve())
        bin_path_32 = str((Path(bin_root) / 'bin').resolve())
        env['PATH'] = f"{bin_path_x64}{os.pathsep}{bin_path_32}{os.pathsep}{env.get('PATH', '')}"
    
    if verbose:
        print(f"  VBSP: {' '.join(cmd)}", flush=True)
    
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(vbsp_exe.parent),
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"VBSP timed out after {timeout}s on {vmf_path.name}")
        except FileNotFoundError:
            raise VBSPNotFoundError(f"Could not execute: {vbsp_exe}")
        
        # Combine stdout and stderr for parsing
        full_output = result.stdout + '\n' + result.stderr
        
        if verbose:
            for line in result.stdout.splitlines():
                print(f"    [vbsp] {line}", flush=True)
        
        # Always try to parse VERTCOUNT first (even on non-zero exit)
        match = _VERTCOUNT_RE.search(full_output)
        if match:
            exceeded = bool(_EXCEEDED_RE.search(full_output))
            
            # Parse leafface count (may not be present in older builds)
            lf_match = _LEAFFACECOUNT_RE.search(full_output)
            leafface_count = int(lf_match.group(1)) if lf_match else 0
            leafface_exceeded = bool(_LEAFFACE_EXCEEDED_RE.search(full_output))
            
            # Parse face count
            fc_match = _FACECOUNT_RE.search(full_output)
            face_count = int(fc_match.group(1)) if fc_match else 0
            face_exceeded = bool(_FACE_EXCEEDED_RE.search(full_output))
            
            return VertexCount(
                int(match.group(1)), exceeded,
                leafface_count, leafface_exceeded,
                face_count, face_exceeded,
            )
        
        # Non-zero exit without VERTCOUNT — may be intermittent (e.g. Steam init)
        if result.returncode != 0:
            # Show full output for diagnosis
            output_tail = full_output.strip()[-500:]
            last_error = VBSPCompileError(
                f"VBSP exited with code {result.returncode} "
                f"(attempt {attempt}/{max_retries}):\n{output_tail}")
            if attempt < max_retries:
                if verbose:
                    print(f"  VBSP attempt {attempt} failed, retrying...",
                          flush=True)
                continue
            raise last_error
        
        # Zero exit but no VERTCOUNT — shouldn't happen with -countverts build
        raise VBSPParseError(
            f"VERTCOUNT not found in VBSP output. "
            f"Is this VBSP built with -countverts support?\n"
            f"Output (last 500 chars): {full_output[-500:]}")
    
    # Should never reach here, but just in case
    raise last_error or VBSPError("VBSP failed after all retries")


def write_temp_vmf(root, output_path: Path) -> Path:
    """Write a temporary VMF file for VBSP vertex counting.
    
    Args:
        root: Parsed VMF KVNode tree (with modified lightmapscales)
        output_path: Path to write the temp VMF
        
    Returns:
        Path to the written VMF file
    """
    from vmf_parser import VMFWriter
    
    writer = VMFWriter()
    writer.write_file(root, output_path)
    return output_path


def compile_bsp(vbsp_exe: Path,
                vmf_path: Path,
                game_dir: Path,
                bin_root: Path = None,
                timeout: int = 300,
                verbose: bool = False,
                max_retries: int = 2,
                extra_args: list = None) -> Path:
    """Run a full VBSP compile to produce a BSP file.
    
    Unlike count_vertices() which uses -countverts (early exit),
    this performs a complete VBSP compilation so the resulting BSP
    can be used for VRAD light counting.
    
    Args:
        vbsp_exe: Path to vbsp_lmo.exe
        vmf_path: Path to the VMF file to compile
        game_dir: Game directory for VBSP's -game flag
        bin_root: Bin root directory for VBSP's -binroot flag
        timeout: Maximum seconds to wait for VBSP (default: 300)
        verbose: If True, print VBSP's full output
        max_retries: Number of attempts (default: 2)
        extra_args: Additional command-line arguments for VBSP
        
    Returns:
        Path to the compiled BSP file
        
    Raises:
        VBSPNotFoundError: If vbsp_exe doesn't exist
        VBSPCompileError: If VBSP exits with an error after all retries
    """
    vbsp_exe = Path(vbsp_exe).resolve()
    vmf_path = Path(vmf_path).resolve()
    game_dir = Path(game_dir).resolve()
    
    if not vbsp_exe.is_file():
        raise VBSPNotFoundError(f"VBSP executable not found: {vbsp_exe}")
    
    if not vmf_path.is_file():
        raise VBSPError(f"VMF file not found: {vmf_path}")
    
    if not game_dir.is_dir():
        raise VBSPError(f"Game directory not found: {game_dir}")
    
    # Build command line
    cmd = [
        str(vbsp_exe),
        '-game', str(game_dir),
    ]
    if bin_root:
        cmd.extend(['-binroot', str(Path(bin_root).resolve())])
    
    if extra_args:
        cmd.extend(extra_args)
        
    cmd.append(str(vmf_path))
    
    env = os.environ.copy()
    if bin_root:
        # Prepend the binroot to PATH so that VBSP can load MaterialSystem.dll and other engine DLLs
        bin_path_x64 = str((Path(bin_root) / 'bin' / 'x64').resolve())
        bin_path_32 = str((Path(bin_root) / 'bin').resolve())
        env['PATH'] = f"{bin_path_x64}{os.pathsep}{bin_path_32}{os.pathsep}{env.get('PATH', '')}"
    
    if verbose:
        print(f"  VBSP compile: {' '.join(cmd)}", flush=True)
    
    # Record pre-existing BSP state so we can detect stale files
    bsp_path = vmf_path.with_suffix('.bsp')
    pre_mtime = bsp_path.stat().st_mtime if bsp_path.is_file() else None
    
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(vbsp_exe.parent),
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"VBSP timed out after {timeout}s on {vmf_path.name}")
        except FileNotFoundError:
            raise VBSPNotFoundError(f"Could not execute: {vbsp_exe}")
        
        full_output = result.stdout + '\n' + result.stderr
        
        if verbose:
            for line in result.stdout.splitlines():
                print(f"    [vbsp] {line}", flush=True)
        
        # Check for BSP file — VBSP often exits with code 1 for
        # non-fatal warnings (e.g. leaked maps: "Using unoptimized BSP
        # data on this map may cause engine crashes!") but still produces
        # a valid BSP.
        #
        # IMPORTANT: Only accept BSPs that were freshly created or modified
        # by this invocation. A stale BSP from a prior compile must not be
        # silently reused — that leads to optimizing against wrong data.
        if bsp_path.is_file():
            post_mtime = bsp_path.stat().st_mtime
            is_fresh = (pre_mtime is None or post_mtime > pre_mtime)
            
            if is_fresh:
                if result.returncode != 0 and verbose:
                    print(f"  VBSP exited with code {result.returncode} "
                          f"but BSP was produced (non-fatal warning)",
                          flush=True)
                return bsp_path
            else:
                # BSP exists but wasn't touched — stale from a previous run
                if verbose:
                    print(f"  WARNING: BSP file exists but was NOT modified "
                          f"by this VBSP invocation (stale)", flush=True)
        
        if result.returncode == 0 and not bsp_path.is_file():
            raise VBSPError(
                f"VBSP completed but BSP file not found: {bsp_path}")
        
        # Non-zero exit AND no fresh BSP produced — real failure
        output_tail = full_output.strip()[-500:]
        last_error = VBSPCompileError(
            f"VBSP exited with code {result.returncode} "
            f"(attempt {attempt}/{max_retries}):\n{output_tail}")
        if attempt < max_retries:
            if verbose:
                print(f"  VBSP attempt {attempt} failed, retrying...",
                      flush=True)
            continue
        raise last_error
    
    raise last_error or VBSPError("VBSP failed after all retries")

