"""
VMT Checker — detect materials with %detailtype defined.

Scans VMT files across the Source engine search path (loose files + VPKs)
to identify materials that define %detailtype. These materials need a minimum
lightmapscale (typically ≥ 5) to ensure VBSP generates detail props correctly.

Usage:
    from vmt_checker import scan_materials
    detail_mats = scan_materials(material_names, game_dir)
    # detail_mats is a set of material names (lowercase) that have %detailtype
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from vpk_reader import VPKReader


def scan_materials(
    material_names: Set[str] | List[str],
    game_dir: Path,
    verbose: bool = False,
) -> Set[str]:
    """Scan materials for %detailtype across game search paths.

    Args:
        material_names: Set of material names as they appear in the VMF
                        (e.g. 'nature/blendgroundtograss001a'). Case-insensitive.
        game_dir: Game directory (the --game dir, e.g. game/mod_hl2mp).
                  Used to locate gameinfo.txt and resolve search paths.
        verbose: Print detailed progress.

    Returns:
        Set of material names (lowercase) that have %detailtype defined.
    """
    # Normalize input
    materials_to_check = {m.lower().replace('\\', '/') for m in material_names}

    if not materials_to_check:
        return set()

    # Resolve search paths from gameinfo.txt
    search_paths = _resolve_search_paths(game_dir, verbose=verbose)

    # Open VPKs referenced in search paths
    vpk_cache: Dict[str, VPKReader] = {}

    found_detail: Set[str] = set()
    checked: Set[str] = set()

    for mat_name in materials_to_check:
        vmt_path = f"materials/{mat_name}.vmt"

        # Try each search path in order
        for sp in search_paths:
            if mat_name in checked:
                break

            if sp.suffix.lower() == '.vpk':
                # VPK search path
                vpk_key = str(sp).lower()
                if vpk_key not in vpk_cache:
                    dir_vpk = _find_dir_vpk(sp)
                    if dir_vpk and dir_vpk.exists():
                        try:
                            vpk_cache[vpk_key] = VPKReader(dir_vpk)
                            if verbose:
                                print(f"    Opened VPK: {dir_vpk.name} "
                                      f"({len(vpk_cache[vpk_key])} files)")
                        except Exception as e:
                            if verbose:
                                print(f"    Failed to open VPK {dir_vpk}: {e}")
                            vpk_cache[vpk_key] = None
                    else:
                        vpk_cache[vpk_key] = None

                vpk = vpk_cache.get(vpk_key)
                if vpk is None:
                    continue

                data = vpk.read_file(vmt_path)
                if data is not None:
                    checked.add(mat_name)
                    if _vmt_has_detailtype(data):
                        found_detail.add(mat_name)
            else:
                # Loose file search path
                full_path = sp / vmt_path
                if full_path.exists():
                    checked.add(mat_name)
                    try:
                        with open(full_path, 'rb') as f:
                            data = f.read()
                        if _vmt_has_detailtype(data):
                            found_detail.add(mat_name)
                    except Exception:
                        pass

    if verbose and materials_to_check - checked:
        not_found = materials_to_check - checked
        print(f"    {len(not_found)} materials not found in any search path",
              flush=True)

    return found_detail


def _vmt_has_detailtype(data: bytes) -> bool:
    """Check if VMT file data contains a %detailtype definition."""
    try:
        text = data.decode('utf-8', errors='replace').lower()
    except Exception:
        return False

    # Match: "%detailtype" or $detailtype (some VMTs use $ instead of %)
    # Must appear as a key, not inside a comment
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('//'):
            continue
        if '%detailtype' in line or '$detailtype' in line:
            return True
    return False


def _resolve_search_paths(
    game_dir: Path,
    verbose: bool = False,
) -> List[Path]:
    """Resolve Source engine search paths from gameinfo.txt.

    Returns a list of Path objects in search order. Each is either:
      - A directory (for loose file lookups)
      - A .vpk path (will be opened by the VPK reader)
    """
    gameinfo = game_dir / 'gameinfo.txt'
    if not gameinfo.exists():
        if verbose:
            print(f"    WARNING: gameinfo.txt not found at {gameinfo}")
        return [game_dir]

    # Resolve |all_source_engine_paths| from SDK_EXEC_DIR environment variable
    # SDK_EXEC_DIR typically points to bin/x64 inside the SDK root,
    # but |all_source_engine_paths| is the SDK root itself (contains hl2/, platform/, etc.)
    sdk_dir = os.environ.get('SDK_EXEC_DIR', '')
    if sdk_dir:
        # Walk up from SDK_EXEC_DIR to find the SDK root (parent of bin/)
        candidate = Path(sdk_dir)
        while candidate != candidate.parent:
            if (candidate / 'hl2').is_dir():
                break
            candidate = candidate.parent
        all_source = candidate
    else:
        # Fallback: assume it's the parent of the game dir
        all_source = game_dir.parent


    if verbose:
        print(f"    |all_source_engine_paths| → {all_source}")

    gameinfo_path = game_dir

    search_paths: List[Path] = []

    try:
        with open(gameinfo, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception:
        return [game_dir]

    # Parse SearchPaths block
    in_search_paths = False
    brace_depth = 0

    for line in content.splitlines():
        stripped = line.strip()

        # Track brace depth to find SearchPaths block
        if 'SearchPaths' in stripped and not stripped.startswith('//'):
            in_search_paths = True
            continue

        if in_search_paths:
            if stripped == '{':
                brace_depth += 1
                continue
            if stripped == '}':
                brace_depth -= 1
                if brace_depth <= 0:
                    break
                continue

            if stripped.startswith('//') or not stripped:
                continue

            # Parse: key<tab>value
            # We only care about 'game' paths (not 'platform', 'gamebin', etc.)
            parts = stripped.split(None, 1)
            if len(parts) < 2:
                continue

            path_type = parts[0].lower()
            path_value = parts[1].strip()

            # Only process 'game' paths (game, game+mod, game+game_write, etc.)
            if 'game' not in path_type:
                continue

            # Skip custom/* wildcard entries
            if path_value.endswith('/*'):
                continue

            # Resolve tokens — ensure trailing separator for proper path joining
            gi_str = str(gameinfo_path)
            if not gi_str.endswith(os.sep):
                gi_str += os.sep
            as_str = str(all_source)
            if not as_str.endswith(os.sep):
                as_str += os.sep
            path_value = path_value.replace('|gameinfo_path|', gi_str)
            path_value = path_value.replace('|all_source_engine_paths|', as_str)

            resolved = Path(path_value)
            if not resolved.is_absolute():
                # Relative to the game root (parent of game_dir)
                resolved = game_dir.parent / path_value

            search_paths.append(resolved)

    # Always include game_dir itself as a fallback
    if game_dir not in search_paths:
        search_paths.insert(0, game_dir)

    if verbose:
        print(f"    Resolved {len(search_paths)} search paths")

    return search_paths


def _find_dir_vpk(vpk_path: Path) -> Optional[Path]:
    """Given a VPK path from gameinfo.txt, find the corresponding _dir.vpk.

    gameinfo.txt references VPKs without the _dir suffix, e.g.:
        hl2/hl2_textures.vpk
    The actual directory file is:
        hl2/hl2_textures_dir.vpk
    """
    # If it already ends with _dir.vpk, use as-is
    if vpk_path.stem.endswith('_dir'):
        return vpk_path

    # Insert _dir before .vpk
    dir_vpk = vpk_path.with_name(vpk_path.stem + '_dir.vpk')
    return dir_vpk
