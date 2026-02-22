#!/usr/bin/env python3
"""
lmoptimizer — VMF Lightmap Scale Optimizer

Preprocesses a VMF file to intelligently adjust per-face lightmapscale values,
reducing VBSP vertex counts on maps with high-resolution lightmap settings.

Two modes:
  1. Simulation mode (default): Build geometry, simulate lighting, classify faces
  2. BSP mode (--bsp): Read real lightmap data from compiled BSP — faster & more accurate

The vertex budget solver (BSP mode) maximizes faces at lightmapscale=1,
degrading the most uniform faces first to fit within VBSP's vertex limit.

Usage:
    python lmoptimizer.py input.vmf --bsp map.bsp [options]
    python lmoptimizer.py input.vmf --bsp map.bsp --vbsp vbsp_lmo.exe --game C:\\hl2\\hl2 [options]
    python lmoptimizer.py input.vmf [-o output.vmf] [options]
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running as a script from any directory
_this_dir = Path(__file__).resolve().parent
_parent_dir = _this_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

# Force UTF-8 output on Windows (cp1252 can't handle box-drawing chars)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from vmf_parser import VMFParser, VMFWriter, VMFBrush, KVNode, extract_brushes, extract_lights


def main():
    parser = argparse.ArgumentParser(
        description="VMF Lightmap Scale Optimizer — reduce VBSP vertex counts "
                    "by adjusting lightmapscale on uniformly-lit faces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lmoptimizer.py mymap.vmf --bsp mymap.bsp --dry-run
  python lmoptimizer.py mymap.vmf --bsp mymap.bsp --vbsp vbsp_lmo.exe --game C:\\hl2\\hl2
  python lmoptimizer.py mymap.vmf --bsp mymap.bsp -o mymap_optimized.vmf
  python lmoptimizer.py mymap.vmf --bsp mymap.bsp --vertex-budget 60000
        """,
    )

    parser.add_argument('input', type=str, help='Input VMF file path')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output VMF file path (default: input_optimized.vmf)')
    parser.add_argument('--bsp', type=str, default=None,
                        help='Read lightmap data from compiled BSP (fastest, most accurate)')

    # Vertex budget solver (BSP mode)
    parser.add_argument('--vertex-budget', type=int, default=65536,
                        help='Target vertex limit for budget solver (default: 65536)')
    parser.add_argument('--headroom', type=int, default=3277,
                        help='Safety margin subtracted from vertex budget. '
                             'Prevents near-limit crashes with vvis -fast '
                             '(default: 3277 = 5%% of 65536)')
    parser.add_argument('--max-lightmap-dim', type=int, default=32,
                        help="VBSP's g_maxLightmapDimension (default: 32)")

    # VBSP integration (ground-truth vertex counting)
    parser.add_argument('--vbsp', type=str, default=None,
                        help='Path to vbsp_lmo.exe with -countverts support '
                             '(auto-detects in same directory if omitted)')
    parser.add_argument('--game', type=str, default=None,
                        help='Game directory for VBSP -game flag '
                             '(required when --vbsp is used)')

    # VRAD integration (surface light counting)
    parser.add_argument('--vrad', type=str, default=None,
                        help='Path to vrad_rtx.exe with -countlights support '
                             '(auto-detects if omitted)')
    parser.add_argument('--vvis', type=str, default=None,
                        help='Path to vvis.exe (auto-detects alongside VBSP '
                             'if omitted). Required for accurate light '
                             'counting — without VIS data, VRAD '
                             'under-counts surface lights.')
    parser.add_argument('--vrad-game', type=str, default=None,
                        help='Separate game directory for VRAD (used for '
                             'light counting). If omitted, uses --game.\n'
                             'Useful when VBSP and VRAD need different '
                             'game directories (e.g. sourcetest vs garrysmod).')
    parser.add_argument('--lights', type=str, default=None,
                        help='Path to a custom .rad lights file to forward '
                             'to VRAD during light counting '
                             '(e.g. E:\\lights_custom.rad).')
    parser.add_argument('--light-budget', type=int, default=32767,
                        help='Maximum surface lights in compiled BSP '
                             '(default: 32767 — GMod limit). '
                             'Set to 0 to disable light budget enforcement.')

    # Classification thresholds (simulation mode / fallback)
    parser.add_argument('--dark-threshold', type=float, default=5.0,
                        help='Max luminance for "uniformly dark" (default: 5.0)')
    parser.add_argument('--variance-threshold', type=float, default=10.0,
                        help='Variance threshold for "uniformly lit" (default: 10.0)')
    parser.add_argument('--max-scale', type=int, default=128,
                        help='lightmapscale for dark faces (default: 128)')
    parser.add_argument('--uniform-scale', type=int, default=32,
                        help='lightmapscale for uniform faces (default: 32)')
    parser.add_argument('--transition-scale', type=int, default=16,
                        help='lightmapscale for transition faces (default: 16)')
    parser.add_argument('--detail-scale', type=int, default=1,
                        help='lightmapscale for high-detail faces in BSP mode (default: 1)')
    parser.add_argument('--detail-min-scale', type=int, default=5,
                        help='Minimum lightmapscale for %%detailtype materials (default: 5)')
    parser.add_argument('--gradient-tolerance', type=float, default=None,
                        nargs='?', const=0.5,
                        help='Enable monotonic gradient pre-promotion with the given '
                             'luminance tolerance (default when flag is present: 0.5). '
                             'Omit this flag entirely to disable gradient promotion.')

    # Simulation-mode controls
    parser.add_argument('--sample-spacing', type=float, default=16.0,
                        help='World units between sample points (default: 16)')
    parser.add_argument('--min-input-scale', type=int, default=4,
                        help='Only optimize faces with scale below this (default: 4)')
    parser.add_argument('--no-sim', action='store_true',
                        help='Skip lighting simulation — set ALL eligible faces '
                             'to uniform-scale (fast path for large maps)')

    # Coplanar unification mode
    parser.add_argument('--strict-coplanar', action='store_true',
                        help='Use strict coplanar unification (require matching '
                             'material and texture axes). Default is broad '
                             'plane-only grouping for visual coherence.')

    parser.add_argument('--dry-run', action='store_true',
                        help='Report statistics without writing output')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print per-face classification details')

    # Aggressive carving
    parser.add_argument('--chop', action='store_true',
                        help='Single-cut carving: split brushes with mixed-uniformity '
                             'faces along one axis-aligned plane')
    parser.add_argument('--multichop', action='store_true',
                        help='Multi-cut carving: multiple splits per face to isolate '
                             'all contiguous uniform regions (implies --chop)')

    # Visibility (player position simulator)
    parser.add_argument('--visibility-check', action='store_true',
                        help='Perform raycast-based reachability and visibility checks')
    parser.add_argument('--visibility-data', type=str, default=None,
                        help='Load pre-computed visibility JSON instead of running checks')
    parser.add_argument('--vis-debug', action='store_true',
                        help='Paint never-visible faces with dev_measuregeneric01b')
    parser.add_argument('--rtx', action='store_true',
                        help='Pass -rtx to VRAD for GPU-accelerated lighting (if supported)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force a fresh VBSP/VVIS/VRAD compile even if a cached BSP exists')
    parser.add_argument('--vis-workers', type=int, default=0,
                        help='Number of parallel worker processes for visibility (0=auto)')

    args = parser.parse_args()

    # ─── Resolve paths ────────────────────────────────────────────────────────
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = input_path.with_name(
            input_path.stem + '_optimized' + input_path.suffix)

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║          VMF Lightmap Scale Optimizer            ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    if args.bsp:
        print(f"  BSP:    {Path(args.bsp).resolve()}")
    if args.vbsp or _try_find_vbsp():
        vbsp_path = Path(args.vbsp).resolve() if args.vbsp else _try_find_vbsp()
        print(f"  VBSP:   {vbsp_path}")
    if args.vrad or _try_find_vrad():
        vrad_disp = Path(args.vrad).resolve() if args.vrad else _try_find_vrad()
        print(f"  VRAD:   {vrad_disp}")
    if args.light_budget > 0:
        print(f"  Light budget: {args.light_budget:,}")
    print()

    # ─── Phase 1: Parse VMF ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    print("[1/5] Parsing VMF...", flush=True)
    vmf_parser = VMFParser()
    root = vmf_parser.parse_file(input_path)
    t1 = time.perf_counter()
    print(f"  Parsed in {t1 - t0:.2f}s", flush=True)

    # ─── Route: BSP mode (with optional VBSP verification) or auto-compile ────
    if args.bsp:
        result = _run_bsp_mode(args, root, t0)
    else:
        # If no BSP is provided, Auto-Compile a baseline BSP and run the pipeline
        if getattr(args, 'no_sim', False) or getattr(args, 'skip_compile', False):
            result = _run_sim_mode(args, root, t0)
            _print_sim_results(result, args)
            _write_output(result, root, output_path, args, t0)
            return

        result = _run_auto_compile_mode(args, root, input_path, output_path, t0)
        
    if result is None:
        return

    # ─── Report (BSP mode with solver) ────────────────────────────────────────
    if 'vbsp_solver_result' in result:
        # VBSP-in-the-loop solver — already has ground-truth counts
        if args.bsp: # Since auto-compile prints and writes internally
            _print_vbsp_solver_results(result, args)
            _write_output(result, root, output_path, args, t0)
        # No separate verification needed — solver already used VBSP
    else:
        # Calibration-factor solver fallback
        if args.bsp:
            _print_solver_results(result, args)
            _write_output(result, root, output_path, args, t0)
        
        # ─── VBSP verification pass (only for calibration solver) ─────────────
        if not args.dry_run:
            vbsp_path = _resolve_vbsp(args)
            game_dir = _resolve_game_dir(args)
            if vbsp_path and game_dir:
                _run_vbsp_verification(vbsp_path, game_dir, output_path, args)

    # ─── Light budget enforcement (post-solver) ───────────────────────────────
    if not args.dry_run and args.light_budget > 0:
        vbsp_path = _resolve_vbsp(args)
        vrad_path = _resolve_vrad(args)
        vvis_path = _resolve_vvis(args, vbsp_path)
        game_dir = _resolve_game_dir(args)
        # Use --vrad-game if specified, otherwise fall back to --game
        vrad_game_dir = Path(args.vrad_game).resolve() if args.vrad_game else game_dir
        lights_rad = Path(args.lights).resolve() if args.lights else None
        if vbsp_path and vrad_path and game_dir:
            side_map = _build_vmf_side_map(root)
            _run_light_budget_enforcement(
                root=root,
                side_map=side_map,
                output_path=output_path,
                vbsp_exe=vbsp_path,
                vrad_exe=vrad_path,
                vvis_exe=vvis_path,
                game_dir=game_dir,
                vrad_game_dir=vrad_game_dir,
                lights_rad=lights_rad,
                light_budget=args.light_budget,
                verbose=args.verbose,
            )
        elif args.light_budget > 0:
            if not vrad_path:
                print("\n  (No vrad_rtx.exe found — skipping light budget check)",
                      flush=True)
            if not vbsp_path:
                print("  (No vbsp_lmo.exe found — skipping light budget check)",
                      flush=True)


def _print_solver_results(result, args):
    """Print vertex budget solver results."""
    sr = result['solver_result']
    under = sr.optimized_verts <= sr.vertex_budget

    print()
    print("┌──────────────────────────────────────────────────┐")
    print("│              Vertex Budget Results               │")
    print("├──────────────────────────────────────────────────┤")
    print(f"│  Vertex budget:          {sr.vertex_budget:>8,}               │")
    print(f"│  VMF current estimate:   {sr.initial_verts:>8,}               │")
    print(f"│  All-at-scale-1:         {sr.all_at_one_verts:>8,}               │")
    print(f"│  After optimization:     {sr.optimized_verts:>8,}               │")
    print("├──────────────────────────────────────────────────┤")
    print(f"│  Faces at scale=1:       {sr.faces_at_scale_1:>8}               │")
    print(f"│  Faces degraded:         {sr.faces_degraded:>8}               │")
    print(f"│  Max scale assigned:     {sr.max_scale_assigned:>8}               │")
    print(f"│  Faces skipped (tools):  {sr.faces_skipped:>8}               │")
    if hasattr(sr, 'faces_pre_promoted') and sr.faces_pre_promoted > 0:
        print(f"│  Faces pre-promoted:     {sr.faces_pre_promoted:>8}               │")
    if hasattr(sr, 'faces_gradient_promoted') and sr.faces_gradient_promoted > 0:
        print(f"│  Gradient pre-promoted:  {sr.faces_gradient_promoted:>8}               │")
    print(f"│  Solver iterations:      {sr.iterations:>8}               │")
    print("├──────────────────────────────────────────────────┤")

    if under:
        margin = sr.vertex_budget - sr.optimized_verts
        saved_from_ideal = sr.all_at_one_verts - sr.optimized_verts
        print(f"│  ✓ UNDER BUDGET by {margin:>6,} verts             │")
        print(f"│  Quality cost: {saved_from_ideal:>6,} verts from ideal      │")
    else:
        over = sr.optimized_verts - sr.vertex_budget
        print(f"│  ✗ OVER BUDGET by {over:>7,} verts              │")
        print(f"│  All faces maxed out — map needs fewer          │")
        print(f"│  brushes or a higher vertex limit.              │")

    if sr.degraded_detail_faces > 0:
        print(f"│  ⚠ {sr.degraded_detail_faces} lighting-detail faces degraded     │")

    print("└──────────────────────────────────────────────────┘")


def _print_vbsp_solver_results(result, args):
    """Print VBSP-in-the-loop solver results (ground truth)."""
    sr = result['vbsp_solver_result']
    budget = sr.vertex_budget
    under = (sr.vbsp_vertex_count <= budget and
             sr.vbsp_leafface_count <= budget and
             sr.vbsp_face_count <= budget)

    print()
    print("┌──────────────────────────────────────────────────┐")
    print("│        VBSP Ground-Truth Solver Results          │")
    print("├──────────────────────────────────────────────────┤")
    print(f"│  Budget (per limit):     {budget:>8,}               │")
    print(f"│  VBSP vertices:          {sr.vbsp_vertex_count:>8,}               │")
    print(f"│  VBSP leaffaces:         {sr.vbsp_leafface_count:>8,}               │")
    print(f"│  VBSP faces:             {sr.vbsp_face_count:>8,}               │")
    print(f"│  Binding limit:        {sr.binding_limit:>10}               │")
    print("├──────────────────────────────────────────────────┤")
    print(f"│  Baseline scale:         {sr.baseline_scale:>8}               │")
    print(f"│  Faces promoted to 1:    {sr.faces_promoted:>8}               │")
    print(f"│  Faces at baseline:      {sr.faces_at_baseline:>8}               │")
    print(f"│  Faces skipped (tools):  {sr.faces_skipped:>8}               │")
    if hasattr(sr, 'faces_pre_promoted') and sr.faces_pre_promoted > 0:
        print(f"│  Faces pre-promoted:     {sr.faces_pre_promoted:>8}               │")
    if hasattr(sr, 'faces_gradient_promoted') and sr.faces_gradient_promoted > 0:
        print(f"│  Gradient pre-promoted:  {sr.faces_gradient_promoted:>8}               │")
    print(f"│  VBSP invocations:       {sr.vbsp_calls:>8}               │")
    print(f"│  Solve time:           {sr.solve_time:>7.1f}s               │")
    print("├──────────────────────────────────────────────────┤")

    if under:
        worst = max(sr.vbsp_vertex_count, sr.vbsp_leafface_count, sr.vbsp_face_count)
        margin = budget - worst
        print(f"│  ✓ ALL LIMITS OK, margin: {margin:>5,} ({sr.binding_limit})  │")
    else:
        worst = max(sr.vbsp_vertex_count, sr.vbsp_leafface_count, sr.vbsp_face_count)
        over = worst - budget
        print(f"│  ✗ OVER BUDGET by {over:>7,} ({sr.binding_limit})       │")
        print(f"│  Map has too many brushes for this budget.      │")

    print("└──────────────────────────────────────────────────┘")


def _print_sim_results(result, args):
    """Print simulation mode results."""
    if result is None:
        return
    stats = result['stats']
    print()
    print("┌──────────────────────────────────────────────────┐")
    print("│                  Results                         │")
    print("├──────────────────────────────────────────────────┤")
    print(f"│  Total faces analyzed:     {stats['total']:>6}               │")
    print(f"│  Uniform dark  → {stats.get('dark_scale', '???'):>3}:     {stats.get('uniform_dark', 0):>6}               │")
    print(f"│  Uniform lit   → {stats.get('uniform_scale', '???'):>3}:     {stats.get('uniform_lit', 0):>6}               │")
    print(f"│  Transition    → {stats.get('transition_scale', '???'):>3}:     {stats.get('transition', 0):>6}               │")
    if 'high_detail' in stats:
        print(f"│  High detail   →   {stats.get('detail_scale', '?'):>1}:     {stats.get('high_detail', 0):>6}               │")
    else:
        print(f"│  High variation (kept):    {stats.get('high_variation', 0):>6}               │")
    print(f"│  Skipped:                  {stats.get('skipped', 0):>6}               │")
    print(f"│  Faces modified:           {stats.get('modified', 0):>6}               │")
    print("└──────────────────────────────────────────────────┘")


def _write_output(result, root, output_path, args, t0):
    """Write the output VMF if not dry-run."""
    if result is None:
        return
    if args.dry_run:
        print("\n  [DRY RUN] No output file written.")
    else:
        print(f"\n  Writing optimized VMF to: {output_path}", flush=True)
        writer = VMFWriter()
        writer.write_file(root, output_path)
        t_end = time.perf_counter()
        print(f"  Total time: {t_end - t0:.2f}s")
        print(f"\n  Done! Run VBSP on the optimized VMF and compare vertex counts.")


# ═══════════════════════════════════════════════════════════════════════════════
#  BSP MODE — vertex budget solver with real lightmap data
# ═══════════════════════════════════════════════════════════════════════════════

def _run_bsp_mode(args, root, t0: float) -> dict | None:
    """Run the BSP-based optimization pipeline with vertex budget solver."""
    from bsp_reader import BSPReader, match_bsp_to_vmf
    from vertex_estimator import compute_face_extent, FaceExtent, estimate_map_vertices
    from budget_solver import solve_vertex_budget
    from geometry import build_all_faces, build_brush_faces

    bsp_path = Path(args.bsp).resolve()
    if not bsp_path.exists():
        print(f"ERROR: BSP file not found: {bsp_path}", file=sys.stderr)
        sys.exit(1)

    max_lm_dim = args.max_lightmap_dim

    # ─── Phase 2: Build geometry + compute face extents ───────────────────────
    t1 = time.perf_counter()
    print("[2/5] Building geometry and computing face extents...", flush=True)

    # Extract brushes with entity classification
    brush_entity_map = _build_brush_entity_map(root)
    brushes = extract_brushes(root)
    print(f"  Found {len(brushes)} brushes", flush=True)

    # Build face polygons via CSG
    all_geometry_faces = build_all_faces(brushes)
    print(f"  Built {len(all_geometry_faces)} face polygons", flush=True)

    # Compute face extents from polygon vertices + VMF axis data
    face_extents: List[FaceExtent] = []
    side_map = _build_vmf_side_map(root)

    for gf in all_geometry_faces:
        side_node = side_map.get(gf.side_id)
        if side_node is None:
            continue

        uaxis_str = side_node.get_property('uaxis') or ''
        vaxis_str = side_node.get_property('vaxis') or ''

        entity_class = brush_entity_map.get(gf.brush_id, '')

        fe = compute_face_extent(
            gf.vertices, uaxis_str, vaxis_str,
            gf.lightmapscale, gf.side_id, gf.material,
            brush_entity=entity_class,
            has_displacement=bool(side_node.get_children_by_name('dispinfo')),
        )
        face_extents.append(fe)

    t2 = time.perf_counter()
    print(f"  Computed extents for {len(face_extents)} faces in {t2 - t1:.2f}s",
          flush=True)

    # ─── Phase 2.5: Scan materials for %detailtype ────────────────────────────
    detail_type_materials = set()
    detail_min_scale = args.detail_min_scale
    game_dir = _resolve_game_dir(args)
    
    if game_dir:
        print("[2.5/6] Scanning materials for %detailtype...", flush=True)
        from vmt_checker import scan_materials
        
        # Collect unique material names from map faces
        unique_materials = {fe.material.lower().replace('\\', '/') for fe in face_extents}
        print(f"  {len(unique_materials)} unique materials in map", flush=True)
        
        detail_type_materials = scan_materials(
            unique_materials, game_dir, verbose=args.verbose)
        
        if detail_type_materials:
            print(f"  Found {len(detail_type_materials)} materials with %detailtype "
                  f"(min scale: {detail_min_scale}):", flush=True)
            for mat in sorted(detail_type_materials):
                print(f"    • {mat}", flush=True)
        else:
            print(f"  No %detailtype materials found", flush=True)
    else:
        print("  (No --game dir — skipping %detailtype scan)", flush=True)

    # ─── Raw initial vertex estimate (uncalibrated, at VMF's current scales) ────
    raw_initial_verts, _ = estimate_map_vertices(
        face_extents, scales=None, max_lm_dim=max_lm_dim)
    budget = args.vertex_budget - args.headroom
    if args.headroom <= 0:
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║  ⚠ WARNING: --headroom is 0!                    ║")
        print("  ║  The solver will fill to 100% of BSP limits.    ║")
        print("  ║  Maps compiled this way may crash the engine    ║")
        print("  ║  when run with vvis -fast or without vvis.      ║")
        print("  ║  Use full vvis.exe for stable results.          ║")
        print("  ╚══════════════════════════════════════════════════╝")
        print()

    # ─── Phase 3: Read BSP lightmap data ──────────────────────────────────────
    print("[3/6] Reading BSP lightmap data...", flush=True)
    bsp = BSPReader(bsp_path)
    bsp.read()

    # Read actual VBSP vertex count for calibration
    bsp_actual_verts = bsp.read_vertex_count()
    print(f"  BSP actual vertex count: {bsp_actual_verts:,}", flush=True)

    # Detect what lightmapscale the BSP was compiled at
    bsp_compile_scale = bsp.get_bsp_lightmapscale()
    print(f"  BSP compilation lightmapscale: {bsp_compile_scale}", flush=True)

    # Compute raw estimate at the BSP's compilation scale (not VMF's current)
    bsp_scale_overrides = {fe.side_id: bsp_compile_scale for fe in face_extents}
    raw_at_bsp_scale, _ = estimate_map_vertices(
        face_extents, scales=bsp_scale_overrides, max_lm_dim=max_lm_dim)

    # Calibration factor: BSP truth / our raw estimate at the SAME scale
    if raw_at_bsp_scale > 0:
        calibration_factor = bsp_actual_verts / raw_at_bsp_scale
    else:
        calibration_factor = 1.0
    print(f"  Calibration factor: {calibration_factor:.4f} "
          f"(raw@scale{bsp_compile_scale}: {raw_at_bsp_scale:,})", flush=True)

    # Show calibrated initial estimate (at VMF's current scales)
    cal_initial_verts = int(raw_initial_verts * calibration_factor)
    if cal_initial_verts > budget:
        over = cal_initial_verts - budget
        pct = over / budget * 100
        print(f"\n  ⚠ Current VMF: ~{cal_initial_verts:,} calibrated vertices", flush=True)
        print(f"    OVER budget by {over:,} ({pct:.1f}%)", flush=True)
    else:
        under = budget - cal_initial_verts
        print(f"\n  Current VMF: ~{cal_initial_verts:,} calibrated vertices", flush=True)
        print(f"    Under budget by {under:,}", flush=True)
    print()

    bsp_faces = bsp.extract_all_face_data(verbose=args.verbose)
    t3 = time.perf_counter()
    print(f"  Extracted {len(bsp_faces)} renderable faces in {t3 - t2:.2f}s",
          flush=True)


    # ─── Phase 4: Match BSP faces to VMF sides ───────────────────────────────
    print("[4/6] Matching BSP faces to VMF sides...", flush=True)

    # Compute plane normals for VMF sides from their plane points
    vmf_side_info = _build_vmf_side_plane_info(root)
    print(f"  {len(vmf_side_info)} VMF sides with plane data", flush=True)

    # Run two-phase matching
    face_data = match_bsp_to_vmf(bsp_faces, vmf_side_info, verbose=args.verbose)
    t4 = time.perf_counter()
    print(f"  Matched {len(face_data)} VMF sides in {t4 - t3:.2f}s", flush=True)

    if not face_data:
        print("\n  No BSP faces could be matched to VMF sides.")
        return None

    # ─── Phase 4.1: Visibility oracle (never-visible → uniform dark) ──────────
    never_visible_sides: set = set()
    nv_bsp_faces: set = set()
    do_vis = getattr(args, 'visibility_check', False) or getattr(args, 'visibility_data', None)

    if do_vis:
        # Determine source: pre-computed JSON or inline computation
        if args.visibility_data:
            vis_path = Path(args.visibility_data).resolve()
            if vis_path.exists():
                import json as _json
                with open(vis_path) as vf:
                    vis_json = _json.load(vf)
                vis_faces = vis_json.get('faces', {})
                nv_bsp_faces = {int(k) for k, v in vis_faces.items()
                                if not v.get('visible', True)}
                print(f"  Loaded visibility data: {vis_json.get('visible_count', '?')} visible, "
                      f"{vis_json.get('never_visible_count', '?')} never-visible BSP faces",
                      flush=True)
            else:
                print(f"  WARNING: Visibility data not found: {vis_path}", flush=True)
        else:
            # Run inline: reachability → visibility
            print("  Running player position simulator...", flush=True)
            from collision import CollisionWorld
            from reachability import ReachabilityMap
            from visibility import VisibilityOracle

            t_vis0 = time.perf_counter()
            world = CollisionWorld(bsp, verbose=args.verbose)

            # Phase A: Reachability BFS
            rmap = ReachabilityMap(world, grid_res=32.0, verbose=args.verbose)
            reach_count = rmap.run()
            if reach_count == 0:
                print("  WARNING: No reachable cells — skipping visibility check",
                      flush=True)
            else:
                eye_positions = rmap.get_eye_positions()
                print(f"  Reachability: {reach_count} cells, "
                      f"{len(eye_positions)} eye positions", flush=True)

                # Phase B: Visibility classification
                oracle = VisibilityOracle(bsp, world, eye_positions,
                                         verbose=args.verbose)
                vis_results = oracle.classify_faces()
                nv_bsp_faces = {fi for fi, info in vis_results.items()
                                if not info.get('visible', True)}
                vis_count = len(vis_results) - len(nv_bsp_faces)
                t_vis1 = time.perf_counter()
                print(f"  Visibility: {vis_count} visible, "
                      f"{len(nv_bsp_faces)} never-visible BSP faces "
                      f"({t_vis1 - t_vis0:.1f}s)", flush=True)

        # Cross-reference: override luminances for never-visible VMF sides
        raw_nv_sides = set()
        texlight_sides = set()
        
        # Build global texlight material set first so we can identify ALL texlight sides
        texlight_mats = set()
        try:
            from bsp_reader import _find_game_lights_rad
            game_rad = _find_game_lights_rad(game_dir)
            if game_rad:
                texlight_mats |= _parse_texlight_materials(game_rad)
        except Exception: pass
        
        if getattr(args, 'lights', None) and Path(args.lights).is_file():
            texlight_mats |= _parse_texlight_materials(Path(args.lights))

        # Identify texlight surface materials for ALL sides in the map
        for side_id in face_data.keys():
            node = side_map.get(side_id)
            mat = (node.get_property('material') or '').upper().replace('\\\\', '/').strip('/') if node else ''
            if mat in texlight_mats:
                texlight_sides.add(side_id)

        if nv_bsp_faces:
            for side_id, fld in face_data.items():
                bsp_indices = set(fld.bsp_face_indices)
                if bsp_indices and bsp_indices.issubset(nv_bsp_faces):
                    raw_nv_sides.add(side_id)
            
            if raw_nv_sides:
                never_visible_sides = raw_nv_sides - texlight_sides

                for side_id in never_visible_sides:
                    fld = face_data[side_id]
                    fld.luminances = [0.0]

            print(f"  {len(never_visible_sides)} VMF sides → max scale "
                  f"(never-visible), {len(texlight_sides)} texlights skipped in map", flush=True)

    # ─── Phase 4.5: Aggressive brush carving (--chop / --multichop) ────────────
    do_chop = getattr(args, 'chop', False) or getattr(args, 'multichop', False)
    allow_multi = getattr(args, 'multichop', False)
    carve_results = []
    
    if do_chop:
        print("[4.5/6] Aggressive brush carving...", flush=True)
        from brush_carver import find_carve_candidates, apply_carves
        
        # Build brush lookup for carver
        brush_map = {b.id: b._node for b in brushes if b._node}
        
        candidates = find_carve_candidates(
            lighting_data=face_data,
            brushes=brushes,
            side_map=side_map,
            brush_map=brush_map,
            face_extents=face_extents,
            allow_entities=True,
            verbose=args.verbose,
        )
        
        if candidates:
            if args.dry_run:
                print(f"  [DRY RUN] Would carve {len(candidates)} brush faces",
                      flush=True)
                for c in candidates:
                    print(f"    • brush {c.brush_id}, side {c.side_id}: "
                          f"{len(c.uniform_regions)} uniform + "
                          f"{len(c.varied_regions)} varied sub-faces", flush=True)
            else:
                carve_results, promoted_side_ids = apply_carves(
                    vmf_root=root,
                    candidates=candidates,
                    face_extents=face_extents,
                    allow_multi=allow_multi,
                    verbose=True,
                )
                
                if carve_results:
                    total_savings = sum(r.estimated_savings for r in carve_results)
                    total_new = sum(r.new_brush_count for r in carve_results)
                    print(f"  Carved {len(carve_results)} brushes into "
                          f"{total_new} pieces (est. savings: ~{total_savings} verts)",
                          flush=True)
                    
                    # Rebuild geometry and face extents for the modified VMF
                    print("  Rebuilding geometry after carving...", flush=True)
                    brushes = extract_brushes(root)
                    brush_entity_map = _build_brush_entity_map(root)
                    all_geometry_faces = build_all_faces(brushes)
                    
                    face_extents = []
                    side_map = _build_vmf_side_map(root)
                    
                    for gf in all_geometry_faces:
                        side_node = side_map.get(gf.side_id)
                        if side_node is None:
                            continue
                        uaxis_str = side_node.get_property('uaxis') or ''
                        vaxis_str = side_node.get_property('vaxis') or ''
                        entity_class = brush_entity_map.get(gf.brush_id, '')
                        fe = compute_face_extent(
                            gf.vertices, uaxis_str, vaxis_str,
                            gf.lightmapscale, gf.side_id, gf.material,
                            brush_entity=entity_class,
                            has_displacement=bool(side_node.get_children_by_name('dispinfo')),
                        )
                        face_extents.append(fe)
                    
                    print(f"  Rebuilt: {len(brushes)} brushes, "
                          f"{len(face_extents)} face extents", flush=True)
                    
                    # Inject synthetic lighting data for promoted carved faces
                    # so the solver's pre-promotion step treats them as uniform
                    if promoted_side_ids:
                        from bsp_reader import FaceLightmapData
                        for sid in promoted_side_ids:
                            if sid not in face_data:
                                face_data[sid] = FaceLightmapData(
                                    vmf_side_id=sid,
                                    material='',
                                    bsp_face_indices=[],
                                    luminances=[0.0],
                                )
                        print(f"  Injected {len(promoted_side_ids)} synthetic "
                              f"lighting entries for carved dark-uniform faces",
                              flush=True)
                else:
                    print("  No viable carves found", flush=True)
        else:
            print("  No carve candidates found (all faces are "
                  "fully uniform or fully varied)", flush=True)

    # ─── Phase 5: Solve vertex budget ─────────────────────────────────────────
    vbsp_path = _resolve_vbsp(args)
    game_dir = _resolve_game_dir(args)
    
    if vbsp_path and game_dir:
        # ─── VBSP-in-the-loop solver (ground truth) ───────────────────────────────
        print("[5/6] Solving vertex budget with VBSP ground truth...", flush=True)
        from vbsp_solver import solve_with_vbsp
        
        # Compute coplanar groups for budget-aware solver integration
        coplanar_grps = None
        if not getattr(args, 'strict_coplanar', False):
            face_verts = {gf.side_id: gf.vertices for gf in all_geometry_faces}
            coplanar_grps = _compute_coplanar_groups(
                side_map=side_map,
                face_vertices=face_verts,
                face_extents=face_extents,
                detail_type_materials=detail_type_materials,
            )
            if coplanar_grps:
                total_faces = sum(len(g) for g in coplanar_grps)
                print(f"  {len(coplanar_grps)} coplanar groups "
                      f"({total_faces} faces) will be promoted atomically",
                      flush=True)
        
        input_path = Path(args.input).resolve()
        vbsp_result = solve_with_vbsp(
            face_extents=face_extents,
            lighting_data=face_data,
            vmf_root=root,
            side_map=side_map,
            vbsp_exe=vbsp_path,
            game_dir=game_dir,
            input_vmf=input_path,
            vertex_budget=budget,
            max_lm_dim=max_lm_dim,
            detail_type_materials=detail_type_materials,
            detail_min_scale=detail_min_scale,
            gradient_tolerance=args.gradient_tolerance,
            coplanar_groups=coplanar_grps,
            verbose=True,
        )
        
        # Post-solver unification only for strict mode (which increases scales, safe)
        unified = 0
        if getattr(args, 'strict_coplanar', False):
            unified = _unify_coplanar_strict(
                scales=vbsp_result.scales,
                side_map=side_map,
                face_data=face_data,
                face_extents=face_extents,
                max_lm_dim=max_lm_dim,
                verbose=True,
            )
        
        # Re-apply unified scales to VMF
        if unified > 0:
            for side_id, new_scale in vbsp_result.scales.items():
                node = side_map.get(side_id)
                if node:
                    node.set_property('lightmapscale', str(new_scale))
        
        modified = sum(1 for sid, scale in vbsp_result.scales.items()
                       if scale != _get_lightmapscale_default(side_map.get(sid)))
        
        return {
            'vbsp_solver_result': vbsp_result,
            'modified': modified,
            'never_visible_sides': never_visible_sides,
            'raw_nv_sides': raw_nv_sides if 'raw_nv_sides' in locals() else never_visible_sides,
            'texlight_sides': texlight_sides if 'texlight_sides' in locals() else set(),
            'texlight_mats': texlight_mats if 'texlight_mats' in locals() else set(),
        }
    else:
        # ─── Calibration-factor solver (fallback) ─────────────────────────────
        print("[5/6] Solving vertex budget (calibration fallback)...", flush=True)
        if not vbsp_path:
            print("  (No vbsp_lmo.exe found — using calibration factor)", flush=True)
        if not game_dir:
            print("  (No --game dir specified — using calibration factor)", flush=True)
        
        solver_result = solve_vertex_budget(
            face_extents=face_extents,
            lighting_data=face_data,
            vertex_budget=budget,
            max_lm_dim=max_lm_dim,
            calibration_factor=calibration_factor,
            detail_type_materials=detail_type_materials,
            detail_min_scale=detail_min_scale,
            gradient_tolerance=args.gradient_tolerance,
            verbose=args.verbose or True,  # Always show solver progress
        )
        
        # Unify coplanar groups
        if getattr(args, 'strict_coplanar', False):
            _unify_coplanar_strict(
                scales=solver_result.scales,
                side_map=side_map,
                face_data=face_data,
                face_extents=face_extents,
                max_lm_dim=max_lm_dim,
                verbose=True,
            )
        else:
            # Build vertex lookup from geometry faces for adjacency testing
            face_verts = {gf.side_id: gf.vertices for gf in all_geometry_faces}
            _unify_coplanar_groups(
                scales=solver_result.scales,
                side_map=side_map,
                face_data=face_data,
                face_extents=face_extents,
                face_vertices=face_verts,
                detail_type_materials=detail_type_materials,
                gradient_tolerance=args.gradient_tolerance,
                verbose=True,
            )
        
        # Apply scales to VMF
        modified = 0
        if not args.dry_run:
            for side_id, new_scale in solver_result.scales.items():
                node = side_map.get(side_id)
                if node is None:
                    continue
                current = _get_lightmapscale(node)
                if new_scale != current:
                    node.set_property('lightmapscale', str(new_scale))
                    modified += 1
        else:
            for side_id, new_scale in solver_result.scales.items():
                node = side_map.get(side_id)
                if node is None:
                    continue
                current = _get_lightmapscale(node)
                if new_scale != current:
                    modified += 1
        
        return {
            'solver_result': solver_result,
            'modified': modified,
            'never_visible_sides': never_visible_sides,
            'raw_nv_sides': raw_nv_sides if 'raw_nv_sides' in locals() else never_visible_sides,
            'texlight_sides': texlight_sides if 'texlight_sides' in locals() else set(),
            'texlight_mats': texlight_mats if 'texlight_mats' in locals() else set(),
        }


def _build_brush_entity_map(root) -> Dict[int, str]:
    """Map brush IDs to their parent entity classnames.
    
    Returns a dict: brush_id → entity_classname.
    Worldspawn brushes map to "" (empty string).
    func_detail brushes map to "func_detail", etc.
    """
    brush_entity: Dict[int, str] = {}

    # Worldspawn brushes
    for world_node in root.get_children_by_name('world'):
        for solid in world_node.get_children_by_name('solid'):
            bid = solid.get_property('id')
            if bid:
                brush_entity[int(bid)] = ''

    # Entity brushes
    for entity_node in root.get_all_recursive('entity'):
        classname = entity_node.get_property('classname') or ''
        for solid in entity_node.get_children_by_name('solid'):
            bid = solid.get_property('id')
            if bid:
                brush_entity[int(bid)] = classname

    return brush_entity


def _build_vmf_side_plane_info(root) -> dict:
    """Build VMF side plane info for BSP matching: side_id → {normal, dist, material}."""
    from bsp_reader import _canonicalize_plane
    vmf_side_info = {}
    for side_node in root.get_all_recursive('side'):
        sid_str = side_node.get_property('id')
        if not sid_str:
            continue
        sid = int(sid_str)
        plane_str = side_node.get_property('plane') or ''
        material = side_node.get_property('material') or ''

        groups = re.findall(r'\(([^)]+)\)', plane_str)
        if len(groups) < 3:
            continue

        pts = []
        for g in groups[:3]:
            parts = g.strip().split()
            pts.append((float(parts[0]), float(parts[1]), float(parts[2])))

        p1, p2, p3 = pts
        e1 = (p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])
        e2 = (p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2])
        nx = e1[1]*e2[2] - e1[2]*e2[1]
        ny = e1[2]*e2[0] - e1[0]*e2[2]
        nz = e1[0]*e2[1] - e1[1]*e2[0]
        length = math.sqrt(nx*nx + ny*ny + nz*nz)
        if length < 1e-10:
            continue
        nx /= length
        ny /= length
        nz /= length
        dist = nx*p1[0] + ny*p1[1] + nz*p1[2]

        # Canonicalize so VMF normals match BSP convention
        nx, ny, nz, dist = _canonicalize_plane(nx, ny, nz, dist)

        vmf_side_info[sid] = {
            'normal': (nx, ny, nz),
            'dist': dist,
            'material': material,
        }

    return vmf_side_info


def _build_vmf_side_map(root) -> dict:
    """Build a mapping from VMF side ID → side KVNode for in-place modification."""
    side_map = {}
    for side_node in root.get_all_recursive('side'):
        side_id_str = side_node.get_property('id')
        if side_id_str:
            try:
                side_map[int(side_id_str)] = side_node
            except ValueError:
                pass
    return side_map


def _unify_coplanar_strict(
    scales: Dict[int, int],
    side_map: dict,
    face_data: dict,
    face_extents: list,
    max_lm_dim: int = 32,
    verbose: bool = False,
) -> int:
    """Unify lightmapscale for coplanar face groups with uniform lighting.
    
    When multiple brushes share a coplanar face with the same material and
    texture alignment, and ALL faces have perfectly uniform BSP lighting,
    assign them all the same max_useful_scale based on the combined extent.
    
    This targets void-box scenarios where manually carved brush lids produce
    multiple faces with different scales that should match.
    
    Modifies `scales` dict in-place. Returns the number of faces unified.
    """
    from vertex_estimator import min_scale_no_subdivision
    import re
    
    # Build extent lookup
    extent_by_id = {fe.side_id: fe for fe in face_extents}
    
    # Group faces by coplanar key: (plane_key, material, uaxis, vaxis)
    groups: Dict[tuple, list] = {}
    
    for side_id in side_map:
        node = side_map.get(side_id)
        if node is None:
            continue
        
        plane_str = node.get_property('plane')
        material = (node.get_property('material') or '').upper()
        uaxis = node.get_property('uaxis')
        vaxis = node.get_property('vaxis')
        
        if not plane_str or not uaxis or not vaxis:
            continue
        
        # Parse plane to get normal and distance for grouping
        # Round to avoid floating-point mismatch
        matches = re.findall(r'\(([^)]+)\)', plane_str)
        if len(matches) < 3:
            continue
        pts = [tuple(float(x) for x in m.split()) for m in matches[:3]]
        
        # Compute plane normal from 3 points
        v1 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1], pts[1][2] - pts[0][2])
        v2 = (pts[2][0] - pts[0][0], pts[2][1] - pts[0][1], pts[2][2] - pts[0][2])
        nx = v1[1]*v2[2] - v1[2]*v2[1]
        ny = v1[2]*v2[0] - v1[0]*v2[2]
        nz = v1[0]*v2[1] - v1[1]*v2[0]
        length = (nx*nx + ny*ny + nz*nz) ** 0.5
        if length < 1e-8:
            continue
        nx, ny, nz = nx/length, ny/length, nz/length
        dist = nx*pts[0][0] + ny*pts[0][1] + nz*pts[0][2]
        
        # Round for grouping (avoid float noise)
        plane_key = (round(nx, 4), round(ny, 4), round(nz, 4), round(dist, 1))
        
        group_key = (plane_key, material, uaxis, vaxis)
        groups.setdefault(group_key, []).append(side_id)
    
    # Process groups with 2+ faces
    unified_count = 0
    for group_key, side_ids in groups.items():
        if len(side_ids) < 2:
            continue
        
        # Check: group qualifies if all BSP-matched faces have near-uniform
        # lighting. Faces WITHOUT BSP data are OK (BSP merged them away).
        # At least one face must have REAL BSP data (not synthetic/injected).
        UNIFORM_LUM_RANGE = 0.0  # only perfectly uniform faces
        has_any_bsp = False
        all_eligible = True
        for sid in side_ids:
            ld = face_data.get(sid)
            if ld is None:
                continue  # No BSP data = face was merged away, eligible
            # Only count real BSP data (synthetic injected data has empty indices)
            is_real_bsp = bool(ld.bsp_face_indices)
            if is_real_bsp:
                has_any_bsp = True
            if not (ld.is_perfectly_uniform
                or ld.luminance_range <= UNIFORM_LUM_RANGE):
                all_eligible = False
                break
        
        if not all_eligible or not has_any_bsp:
            continue
        
        # Compute max_useful_scale from the largest face extent in the group.
        # This is the scale that eliminates subdivision for the biggest piece,
        # ensuring all pieces in the group share that same scale.
        max_extent_s = 0.0
        max_extent_t = 0.0
        for sid in side_ids:
            fe = extent_by_id.get(sid)
            if fe:
                max_extent_s = max(max_extent_s, fe.extent_s)
                max_extent_t = max(max_extent_t, fe.extent_t)
        
        if max_extent_s < 1e-6 and max_extent_t < 1e-6:
            continue
        
        unified_scale = min_scale_no_subdivision(max_extent_s, max_extent_t, max_lm_dim)
        
        # Sanity cap: don't produce unreasonably large scales
        if unified_scale > 16:
            continue
        
        # Apply to all members
        for sid in side_ids:
            scales[sid] = unified_scale
        
        unified_count += len(side_ids)
        if verbose:
            material = group_key[1]
            print(f"  Unified {len(side_ids)} coplanar uniform faces "
                  f"(mat={material}, scale={unified_scale})", flush=True)
    
    return unified_count


def _compute_coplanar_groups(
    side_map: dict,
    face_vertices: dict,
    face_extents: list,
    detail_type_materials: Optional[Set[str]] = None,
) -> List[List[int]]:
    """Compute connected components of touching coplanar faces.
    
    Groups faces by plane (ignoring material/texture), then clusters
    each plane group by AABB adjacency. Returns only groups with 2+
    faces.
    
    Excludes tool-textured faces, skip-entity faces, and %detailtype faces.
    Does NOT filter by uniform/gradient — the solver handles that.
    """
    from vertex_estimator import should_skip_face
    import re
    
    detail_set = detail_type_materials or set()
    extent_by_id = {fe.side_id: fe for fe in face_extents}
    
    # Group faces by plane key
    groups: Dict[tuple, list] = {}
    
    for side_id in side_map:
        node = side_map.get(side_id)
        if node is None:
            continue
        
        material = (node.get_property('material') or '').upper()
        fe = extent_by_id.get(side_id)
        entity_class = fe.brush_entity if fe else ''
        
        if should_skip_face(material, entity_class):
            continue
        
        mat_lower = material.lower().replace('\\', '/')
        if mat_lower in detail_set:
            continue
        
        if side_id not in face_vertices:
            continue
        
        plane_str = node.get_property('plane')
        if not plane_str:
            continue
        
        matches = re.findall(r'\(([^)]+)\)', plane_str)
        if len(matches) < 3:
            continue
        pts = [tuple(float(x) for x in m.split()) for m in matches[:3]]
        
        v1 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1], pts[1][2] - pts[0][2])
        v2 = (pts[2][0] - pts[0][0], pts[2][1] - pts[0][1], pts[2][2] - pts[0][2])
        nx = v1[1]*v2[2] - v1[2]*v2[1]
        ny = v1[2]*v2[0] - v1[0]*v2[2]
        nz = v1[0]*v2[1] - v1[1]*v2[0]
        length = (nx*nx + ny*ny + nz*nz) ** 0.5
        if length < 1e-8:
            continue
        nx, ny, nz = nx/length, ny/length, nz/length
        dist = nx*pts[0][0] + ny*pts[0][1] + nz*pts[0][2]
        
        plane_key = (round(nx, 4), round(ny, 4), round(nz, 4), round(dist, 1))
        groups.setdefault(plane_key, []).append(side_id)
    
    # Cluster each plane group by AABB adjacency
    result = []
    for plane_key, side_ids in groups.items():
        if len(side_ids) < 2:
            continue
        
        aabbs = {}
        for sid in side_ids:
            verts = face_vertices.get(sid)
            if verts and len(verts) >= 3:
                mins = (min(v[0] for v in verts),
                        min(v[1] for v in verts),
                        min(v[2] for v in verts))
                maxs = (max(v[0] for v in verts),
                        max(v[1] for v in verts),
                        max(v[2] for v in verts))
                aabbs[sid] = (mins, maxs)
        
        adjacency: Dict[int, set] = {sid: set() for sid in side_ids if sid in aabbs}
        aabb_items = list(aabbs.items())
        for i in range(len(aabb_items)):
            sid_a, (mins_a, maxs_a) = aabb_items[i]
            for j in range(i + 1, len(aabb_items)):
                sid_b, (mins_b, maxs_b) = aabb_items[j]
                if _aabbs_touch(mins_a, maxs_a, mins_b, maxs_b):
                    adjacency[sid_a].add(sid_b)
                    adjacency[sid_b].add(sid_a)
        
        components = _find_connected_components(
            [sid for sid in side_ids if sid in aabbs], adjacency)
        
        for component in components:
            if len(component) >= 2:
                result.append(component)
    
    return result


def _aabbs_touch(mins_a, maxs_a, mins_b, maxs_b, epsilon: float = 1.0) -> bool:
    """Check if two AABBs touch or overlap (within epsilon tolerance)."""
    for i in range(3):
        if maxs_a[i] + epsilon < mins_b[i] or maxs_b[i] + epsilon < mins_a[i]:
            return False
    return True


def _find_connected_components(side_ids: list, adjacency: dict) -> list:
    """Find connected components via BFS on an adjacency graph.
    
    Args:
        side_ids: list of side IDs
        adjacency: dict mapping side_id → set of adjacent side_ids
    
    Returns:
        List of lists, each being a connected component.
    """
    visited = set()
    components = []
    for sid in side_ids:
        if sid in visited:
            continue
        # BFS from this node
        component = []
        queue = [sid]
        visited.add(sid)
        while queue:
            current = queue.pop(0)
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _unify_coplanar_groups(
    scales: Dict[int, int],
    side_map: dict,
    face_data: dict,
    face_extents: list,
    face_vertices: dict,
    detail_type_materials: Optional[Set[str]] = None,
    gradient_tolerance: Optional[float] = None,
    verbose: bool = False,
) -> int:
    """Unify lightmapscale for touching coplanar face groups (broad grouping).
    
    Groups faces by plane only (ignoring material and texture axes), then
    clusters each plane group into connected components based on spatial
    adjacency (AABB overlap). Only faces within the same connected component
    are unified.
    
    Within each component, faces that are perfectly uniform or have a simple
    gradient are excluded — they keep their solver-assigned max_useful_scale.
    The remaining "complex" faces are unified to the minimum (best-quality)
    scale already assigned to any member, preventing visual incoherence.
    
    Excludes tool-textured faces, skip-entity faces, and %detailtype faces.
    
    Modifies `scales` dict in-place. Returns the number of faces unified.
    """
    from vertex_estimator import should_skip_face
    import re
    
    detail_set = detail_type_materials or set()
    
    # Build a lookup for face extent data by side_id
    extent_by_id = {fe.side_id: fe for fe in face_extents}
    
    # Group faces by coplanar key: plane only
    groups: Dict[tuple, list] = {}
    
    for side_id in side_map:
        node = side_map.get(side_id)
        if node is None:
            continue
        
        # Get material and check for exclusions
        material = (node.get_property('material') or '').upper()
        fe = extent_by_id.get(side_id)
        entity_class = fe.brush_entity if fe else ''
        
        # Skip tool textures and non-geometry entities
        if should_skip_face(material, entity_class):
            continue
        
        # Skip %detailtype materials
        mat_lower = material.lower().replace('\\', '/')
        if mat_lower in detail_set:
            continue
        
        # Must have vertex data for adjacency testing
        if side_id not in face_vertices:
            continue
        
        plane_str = node.get_property('plane')
        if not plane_str:
            continue
        
        # Parse plane to get normal and distance for grouping
        matches = re.findall(r'\(([^)]+)\)', plane_str)
        if len(matches) < 3:
            continue
        pts = [tuple(float(x) for x in m.split()) for m in matches[:3]]
        
        # Compute plane normal from 3 points
        v1 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1], pts[1][2] - pts[0][2])
        v2 = (pts[2][0] - pts[0][0], pts[2][1] - pts[0][1], pts[2][2] - pts[0][2])
        nx = v1[1]*v2[2] - v1[2]*v2[1]
        ny = v1[2]*v2[0] - v1[0]*v2[2]
        nz = v1[0]*v2[1] - v1[1]*v2[0]
        length = (nx*nx + ny*ny + nz*nz) ** 0.5
        if length < 1e-8:
            continue
        nx, ny, nz = nx/length, ny/length, nz/length
        dist = nx*pts[0][0] + ny*pts[0][1] + nz*pts[0][2]
        
        # Round for grouping (avoid float noise)
        plane_key = (round(nx, 4), round(ny, 4), round(nz, 4), round(dist, 1))
        
        groups.setdefault(plane_key, []).append(side_id)
    
    # Process groups — cluster into connected components by adjacency
    unified_count = 0
    for plane_key, side_ids in groups.items():
        if len(side_ids) < 2:
            continue
        
        # Build AABBs for each face in this plane group
        aabbs = {}
        for sid in side_ids:
            verts = face_vertices.get(sid)
            if verts and len(verts) >= 3:
                mins = (min(v[0] for v in verts),
                        min(v[1] for v in verts),
                        min(v[2] for v in verts))
                maxs = (max(v[0] for v in verts),
                        max(v[1] for v in verts),
                        max(v[2] for v in verts))
                aabbs[sid] = (mins, maxs)
        
        # Build adjacency graph via AABB overlap
        adjacency: Dict[int, set] = {sid: set() for sid in side_ids if sid in aabbs}
        aabb_items = list(aabbs.items())
        for i in range(len(aabb_items)):
            sid_a, (mins_a, maxs_a) = aabb_items[i]
            for j in range(i + 1, len(aabb_items)):
                sid_b, (mins_b, maxs_b) = aabb_items[j]
                if _aabbs_touch(mins_a, maxs_a, mins_b, maxs_b):
                    adjacency[sid_a].add(sid_b)
                    adjacency[sid_b].add(sid_a)
        
        # Find connected components
        components = _find_connected_components(
            [sid for sid in side_ids if sid in aabbs], adjacency)
        
        # Process each connected component independently
        for component in components:
            if len(component) < 2:
                continue
            
            # Partition into uniform/gradient (excluded) and complex (unified)
            complex_ids = []
            excluded_ids = []
            
            for sid in component:
                ld = face_data.get(sid)
                is_uniform = ld.is_perfectly_uniform if ld else False
                is_gradient = (ld.is_monotonic_gradient(gradient_tolerance)
                               if ld and gradient_tolerance is not None else False)
                
                if is_uniform or is_gradient:
                    excluded_ids.append(sid)
                else:
                    complex_ids.append(sid)
            
            if len(complex_ids) < 2:
                continue
            
            # Find the minimum (best-quality) scale among the complex faces
            min_scale = None
            for sid in complex_ids:
                s = scales.get(sid)
                if s is not None:
                    if min_scale is None or s < min_scale:
                        min_scale = s
            
            if min_scale is None:
                continue
            
            # Apply the unified scale to all complex faces in the component
            changed = 0
            for sid in complex_ids:
                old = scales.get(sid)
                if old is not None and old != min_scale:
                    scales[sid] = min_scale
                    changed += 1
            
            if changed > 0:
                unified_count += len(complex_ids)
                if verbose:
                    print(f"  Unified {len(complex_ids)} touching coplanar faces "
                          f"(plane={plane_key}, scale={min_scale}, "
                          f"{len(excluded_ids)} uniform/gradient excluded)",
                          flush=True)
    
    if verbose and unified_count > 0:
        print(f"  Total coplanar-unified: {unified_count} faces", flush=True)
    
    return unified_count


def _get_lightmapscale(side_node) -> int:
    """Get lightmapscale from a VMF side KVNode, defaulting to 16."""
    val = side_node.get_property('lightmapscale')
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return 16


def _get_lightmapscale_default(side_node) -> int:
    """Get lightmapscale from a side node, returning 0 if node is None."""
    if side_node is None:
        return 0
    return _get_lightmapscale(side_node)


def _try_find_vbsp() -> Optional[Path]:
    """Try to find vbsp_lmo.exe in the same directory as this script."""
    from vbsp_runner import find_vbsp
    return find_vbsp()


def _try_find_vrad() -> Optional[Path]:
    """Try to find vrad_rtx.exe via auto-detection."""
    from vrad_runner import find_vrad
    return find_vrad()


def _resolve_vbsp(args) -> Optional[Path]:
    """Resolve the VBSP executable path from args or auto-detection."""
    if args.vbsp:
        path = Path(args.vbsp).resolve()
        if path.is_file():
            return path
        print(f"  WARNING: --vbsp path not found: {path}", file=sys.stderr)
        return None
    # Auto-detect in same directory
    return _try_find_vbsp()


def _resolve_vvis(args, vbsp_path: Optional[Path] = None) -> Optional[Path]:
    """Resolve VVIS path from args, or auto-detect alongside VBSP.
    
    VVIS is needed to generate VIS data before VRAD -countlights.
    Without VIS data, VRAD sets numbounce=0 which causes SubdividePatches
    to skip, resulting in drastically under-counted surface lights.
    """
    if hasattr(args, 'vvis') and args.vvis:
        path = Path(args.vvis).resolve()
        if path.is_file():
            return path
        print(f"  WARNING: --vvis path not found: {path}", file=sys.stderr)
        return None
    # Auto-detect: look for vvis.exe alongside VBSP
    if vbsp_path:
        candidate = vbsp_path.parent / 'vvis.exe'
        if candidate.is_file():
            return candidate
    # Try the game dir's bin/x64
    if hasattr(args, 'game') and args.game:
        game_path = Path(args.game).resolve()
        candidate = game_path.parent / 'bin' / 'x64' / 'vvis.exe'
        if candidate.is_file():
            return candidate
    print("  WARNING: Could not find vvis.exe — light counting may "
          "be inaccurate without VIS data.", file=sys.stderr)
    return None


def _resolve_vrad(args) -> Optional[Path]:
    """Resolve the VRAD executable path from args or auto-detection."""
    if hasattr(args, 'vrad') and args.vrad:
        path = Path(args.vrad).resolve()
        if path.is_file():
            return path
        print(f"  WARNING: --vrad path not found: {path}", file=sys.stderr)
        return None
    # Auto-detect
    return _try_find_vrad()


def _resolve_game_dir(args) -> Optional[Path]:
    """Resolve the game directory from args.

    Also validates that the game installation has a bin/x64 folder,
    which is required for 64-bit VBSP.  Incompatible installations
    (e.g. Source SDK Base 2013 Singleplayer, Half-Life 2) lack this
    folder and produce silently wrong results.
    """
    if args.game:
        path = Path(args.game).resolve()
        if not path.is_dir():
            print(f"  WARNING: --game path not found: {path}", file=sys.stderr)
            return None
        # The game dir is e.g. <root>/sourcetest or <root>/hl2mp.
        # The engine binaries live in <root>/bin/x64.
        bin_x64 = path.parent / 'bin' / 'x64'
        if not bin_x64.is_dir():
            print(f"\n  ╔══════════════════════════════════════════════════╗",
                  file=sys.stderr)
            print(f"  ║  ERROR: Incompatible --game directory!           ║",
                  file=sys.stderr)
            print(f"  ╠══════════════════════════════════════════════════╣",
                  file=sys.stderr)
            print(f"  ║  {str(path)[:48]:<48}║",
                  file=sys.stderr)
            print(f"  ║                                                  ║",
                  file=sys.stderr)
            print(f"  ║  Missing: bin/x64 folder in parent directory.   ║",
                  file=sys.stderr)
            print(f"  ║  This game does not have 64-bit engine binaries.║",
                  file=sys.stderr)
            print(f"  ║                                                  ║",
                  file=sys.stderr)
            print(f"  ║  Compatible examples:                            ║",
                  file=sys.stderr)
            print(f"  ║   • Source SDK Base 2013 Multiplayer/sourcetest  ║",
                  file=sys.stderr)
            print(f"  ║   • Half-Life 2 Deathmatch/hl2mp                ║",
                  file=sys.stderr)
            print(f"  ║  Incompatible:                                   ║",
                  file=sys.stderr)
            print(f"  ║   • Source SDK Base 2013 Singleplayer/sourcetest ║",
                  file=sys.stderr)
            print(f"  ║   • Half-Life 2/hl2                             ║",
                  file=sys.stderr)
            print(f"  ╚══════════════════════════════════════════════════╝",
                  file=sys.stderr)
            sys.exit(1)
        return path
    return None


def _run_vbsp_verification(vbsp_path: Path, game_dir: Path,
                           output_vmf: Path, args) -> None:
    """Run VBSP -countverts on the optimized VMF to verify all BSP limits."""
    from vbsp_runner import count_vertices, VBSPError

    print()
    print("┌──────────────────────────────────────────────────┐")
    print("│          VBSP Verification                       │")
    print("├──────────────────────────────────────────────────┤")

    try:
        vc = count_vertices(
            vbsp_path, output_vmf, game_dir,
            verbose=args.verbose,
        )
        budget = args.vertex_budget
        worst = max(vc.count, vc.leafface_count, vc.face_count)
        under = worst <= budget

        print(f"│  VBSP vertices:          {vc.count:>8,}               │")
        print(f"│  VBSP leaffaces:         {vc.leafface_count:>8,}               │")
        print(f"│  VBSP faces:             {vc.face_count:>8,}               │")
        if vc.exceeded or vc.leafface_exceeded or vc.face_exceeded:
            print(f"│  ⚠ Count exceeded (hit expanded array limit)    │")
        print(f"│  Budget:                 {budget:>8,}               │")

        if under:
            margin = budget - worst
            print(f"│  ✓ VERIFIED ALL LIMITS OK by {margin:>5,}            │")
        else:
            over = worst - budget
            print(f"│  ✗ STILL OVER BUDGET by {over:>7,}              │")
            print(f"│  The solver estimate was inaccurate —           │")
            print(f"│  consider running again with VBSP feedback.     │")

    except VBSPError as e:
        print(f"│  ✗ VBSP verification failed:                    │")
        for line in str(e).splitlines()[:3]:
            print(f"│    {line:<46}│")

    print("└──────────────────────────────────────────────────┘")

def _parse_texlight_materials(rad_path: Path) -> set:
    """Parse a lights.rad file and return the set of texlight material names.

    Returns uppercase-normalized material names (with / separators).
    Skips 'forcetextureshadow' lines and comments.

    Format per line:
        [ldr:|hdr:] material_name R G B intensity
        forcetextureshadow model.mdl
        // comment
    """
    materials = set()
    try:
        with open(rad_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('//'):
                    continue
                # Skip forcetextureshadow lines
                if line.lower().startswith('forcetextureshadow'):
                    continue
                # Strip optional ldr:/hdr: prefix
                if line.lower().startswith(('ldr:', 'hdr:')):
                    line = line[4:].strip()
                # First token is the material name
                parts = line.split()
                if len(parts) >= 4:  # material R G B [intensity]
                    mat = parts[0].upper().replace('\\', '/').strip('/')
                    materials.add(mat)
    except (OSError, IOError):
        pass
    return materials


def _find_game_lights_rad(game_dir: Path) -> Optional[Path]:
    """Find the game's default lights.rad file.

    VRAD searches for 'lights.rad' in the game's filesystem search paths.
    The most common location is directly in the game directory.
    """
    if game_dir:
        candidate = game_dir / 'lights.rad'
        if candidate.is_file():
            return candidate
    return None


def _run_vvis_fast(vvis_exe: Path, bsp_path: Path, game_dir: Path,
                   verbose: bool = False, timeout: int = 120) -> bool:
    """Run VVIS -fast on a BSP to generate minimal VIS data.
    
    VRAD requires VIS data to perform SubdividePatches(); without it,
    numbounce is set to 0 and subdivision is skipped, causing surface
    lights from texlights to be drastically under-counted.
    
    Returns True on success, False on failure.
    """
    bsp_no_ext = str(bsp_path.with_suffix(''))
    import subprocess
    cmd = [
        str(vvis_exe),
        '-fast',
        '-game', str(game_dir),
        bsp_no_ext,
    ]
    if verbose:
        print(f"  VVIS: {' '.join(cmd)}", flush=True)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(vvis_exe.parent),
        )
        if verbose:
            for line in result.stdout.splitlines():
                print(f"    [vvis] {line}", flush=True)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        if verbose:
            print(f"  VVIS error: {e}", flush=True)
        return False


def _run_light_budget_enforcement(
    root,
    side_map: dict,
    output_path: Path,
    vbsp_exe: Path,
    vrad_exe: Path,
    vvis_exe: Path | None = None,
    game_dir: Path = None,
    vrad_game_dir: Path | None = None,
    lights_rad: Path | None = None,
    light_budget: int = 32767,
    verbose: bool = False,
) -> None:
    """Compile the optimized VMF, count surface lights, and degrade if over budget.
    
    This is a post-solver step that iteratively increases all non-skipped face
    lightmapscales until the surface light count is within the budget.
    """
    import shutil
    import tempfile
    from vbsp_runner import compile_bsp, VBSPError
    from vrad_runner import count_lights, VRADError, GMOD_LIGHT_LIMIT
    from vertex_estimator import should_skip_face
    
    print()
    print("┌──────────────────────────────────────────────────┐")
    print("│          Surface Light Budget Check              │")
    print("├──────────────────────────────────────────────────┤")
    
    # Set up a temp directory for compile iterations
    temp_dir = Path(tempfile.mkdtemp(prefix='lmopt_lights_'))
    temp_vmf = temp_dir / output_path.name
    
    # Apply Vis-Debug if enabled
    vis_debug = getattr(args, 'vis_debug', False) if 'args' in globals() or 'args' in locals() else False
    if vis_debug and 'bsp_result' in locals():
        nv_sides = bsp_result.get('raw_nv_sides', bsp_result.get('never_visible_sides', set()))
        texlight_skip = bsp_result.get('texlight_sides', set())
        _VIS_MAT   = 'dev/dev_measuregeneric01'
        _INVIS_MAT = 'dev/dev_measuregeneric01b'
        for world_node in root.get_children_by_name('world'):
            for solid in world_node.get_children_by_name('solid'):
                for side in solid.get_children_by_name('side'):
                    cur_mat = (side.get_property('material') or '').upper().replace('\\', '/')
                    if cur_mat.startswith('TOOLS/'): continue
                    sid = side.get_property('id')
                    if sid:
                        sid_int = int(sid)
                        if sid_int in texlight_skip: continue
                        mat = _INVIS_MAT if sid_int in nv_sides else _VIS_MAT
                        side.set_property('material', mat)
    # Note: args.vis_debug is better, let's pass it cleanly. Wait, we can't easily access args or bsp_result here without modifying signature.
    
    try:
        # Copy the optimized VMF to temp dir for compilation
        shutil.copy2(output_path, temp_vmf)
        
        # Phase 1: Compile with VBSP
        print(f"│  Compiling with VBSP...                         │", flush=True)
        try:
            bsp_path = compile_bsp(
                vbsp_exe, temp_vmf, game_dir,
                verbose=verbose, timeout=300,
                extra_args=['-emitsideids'])
        except (VBSPError, TimeoutError) as e:
            print(f"│  ✗ VBSP compilation failed:                     │")
            for line in str(e).splitlines()[:2]:
                print(f"│    {line[:46]:<46}│")
            print("└──────────────────────────────────────────────────┘")
            return
        
        # Phase 1b: Run VVIS -fast for VIS data (required for accurate light counting)
        # Without VIS, VRAD sets numbounce=0 and SubdividePatches() is skipped,
        # causing surface lights from texlights to be drastically under-counted.
        if vvis_exe:
            # Use vrad_game_dir for VVIS if specified (for consistent gameinfo/mount)
            vvis_gd = vrad_game_dir or game_dir
            print(f"│  Running VVIS -fast for VIS data...              │", flush=True)
            if not _run_vvis_fast(vvis_exe, bsp_path, vvis_gd,
                                  verbose=verbose, timeout=120):
                print(f"│  ⚠ VVIS -fast failed — counts may be low        │")
        else:
            print(f"│  ⚠ No VVIS — surface light counts may be low     │")
        
        # Use vrad_game_dir for VRAD if specified, otherwise use game_dir
        vrad_gd = vrad_game_dir or game_dir
        
        # Phase 2: Count lights with VRAD
        print(f"│  Counting surface lights with VRAD...            │", flush=True)
        if verbose:
            print(f"│  VRAD game dir: {str(vrad_gd)[:33]:<33}│", flush=True)
            if lights_rad:
                print(f"│  Lights .rad:   {str(lights_rad)[:33]:<33}│", flush=True)
        try:
            lc = count_lights(
                vrad_exe, bsp_path, vrad_gd,
                verbose=verbose, timeout=600,
                lights_rad=lights_rad)
        except (VRADError, TimeoutError) as e:
            print(f"│  ✗ VRAD light counting failed:                  │")
            for line in str(e).splitlines()[:2]:
                print(f"│    {line[:46]:<46}│")
            print("└──────────────────────────────────────────────────┘")
            return
        
        print(f"│  Surface lights: {lc.count:>8,}                      │", flush=True)
        print(f"│  Light budget:   {light_budget:>8,}                      │", flush=True)
        
        if lc.count <= light_budget:
            margin = light_budget - lc.count
            pct = lc.count / light_budget * 100
            print(f"│  ✓ UNDER BUDGET by {margin:>6,} ({pct:.0f}%)             │")
            print("└──────────────────────────────────────────────────┘")
            return
        
        # Phase 3: Targeted degradation — only bump texlight-emitting faces
        print("├──────────────────────────────────────────────────┤")
        print(f"│  ⚠ OVER BUDGET — degrading emissive surfaces... │")
        print("├──────────────────────────────────────────────────┤", flush=True)
        
        from vmf_parser import VMFWriter
        writer = VMFWriter()
        
        # Collect texlight material names from lights.rad files
        texlight_mats = set()
        # Game's default lights.rad
        game_rad = _find_game_lights_rad(vrad_gd)
        if game_rad:
            texlight_mats |= _parse_texlight_materials(game_rad)
        # Custom -lights file
        if lights_rad and Path(lights_rad).is_file():
            texlight_mats |= _parse_texlight_materials(Path(lights_rad))
        
        if not texlight_mats:
            print(f"│  ✗ No texlight materials found in lights.rad    │")
            print(f"│  Cannot determine which faces emit light.       │")
            print("└──────────────────────────────────────────────────┘")
            return
        
        print(f"│  Texlight materials found: {len(texlight_mats):>4}                 │")
        
        # Identify emissive faces in the VMF
        emissive_sides = {}
        for side_id, node in side_map.items():
            material = (node.get_property('material') or '').upper()
            # Normalize: strip leading path separators, compare basename
            mat_upper = material.replace('\\', '/').strip('/')
            if mat_upper in texlight_mats:
                emissive_sides[side_id] = node
        
        print(f"│  Emissive faces in VMF:    {len(emissive_sides):>4}                 │")
        
        if not emissive_sides:
            print(f"│  No emissive faces found — lights are from      │")
            print(f"│  entities, not surfaces. Cannot reduce.          │")
            print("└──────────────────────────────────────────────────┘")
            return
        
        max_iterations = 15
        max_scale_cap = 128
        
        for iteration in range(1, max_iterations + 1):
            # Increase only emissive face scales by +1
            bumped = 0
            capped = 0
            for side_id, node in emissive_sides.items():
                current = _get_lightmapscale(node)
                new_scale = min(current + 1, max_scale_cap)
                if new_scale != current:
                    node.set_property('lightmapscale', str(new_scale))
                    bumped += 1
                else:
                    capped += 1
            
            if bumped == 0:
                print(f"│  All emissive faces at max — cannot reduce more │")
                break
            
            # Re-write VMF, re-compile, re-count
            writer.write_file(root, temp_vmf)
            
            try:
                bsp_path = compile_bsp(
                    vbsp_exe, temp_vmf, game_dir,
                    verbose=False, timeout=300,
                    extra_args=['-emitsideids'])
            except (VBSPError, TimeoutError) as e:
                print(f"│  ✗ VBSP failed on iteration {iteration}               │")
                break
            
            # Run VVIS -fast for VIS data
            if vvis_exe:
                vvis_gd = vrad_game_dir or game_dir
                if not _run_vvis_fast(vvis_exe, bsp_path, vvis_gd,
                                      verbose=False, timeout=120):
                    print(f"│  ⚠ VVIS failed on iteration {iteration}               │")
            
            try:
                lc = count_lights(
                    vrad_exe, bsp_path, vrad_gd,
                    verbose=False, timeout=600,
                    lights_rad=lights_rad)
            except (VRADError, TimeoutError) as e:
                print(f"│  ✗ VRAD failed on iteration {iteration}               │")
                break
            
            status = '✓' if lc.count <= light_budget else '…'
            print(f"│  [{status}] Iter {iteration}: bumped {bumped:>5} faces → "
                  f"{lc.count:>7,} lights   │", flush=True)
            
            if lc.count <= light_budget:
                # Success! Write the final optimized VMF
                writer.write_file(root, output_path)
                margin = light_budget - lc.count
                print("├──────────────────────────────────────────────────┤")
                print(f"│  ✓ UNDER BUDGET by {margin:>6,}                      │")
                print(f"│  Degradation iterations: {iteration:>3}                    │")
                print(f"│  Output updated: {output_path.name:<31}│")
                print("└──────────────────────────────────────────────────┘")
                return
        
        # Failed to fit within budget
        over = lc.count - light_budget
        print("├──────────────────────────────────────────────────┤")
        print(f"│  ✗ STILL OVER BUDGET by {over:>7,}              │")
        print(f"│  Map has too many light-emitting surfaces.      │")
        print(f"│  Consider reducing surface light materials.     │")
        print("└──────────────────────────────────────────────────┘")
        
        # Still write the best effort
        writer.write_file(root, output_path)
    
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTO-COMPILE MODE
# ═══════════════════════════════════════════════════════════════════════════════

def _run_auto_compile_mode(args, root, input_path: Path, output_path: Path, t0: float) -> dict | None:
    import tempfile
    import shutil
    from vbsp_runner import compile_bsp, VBSPError

    print("  Mode:   Auto-compile", flush=True)

    vbsp_path = _resolve_vbsp(args)
    game_dir = _resolve_game_dir(args)
    if not vbsp_path or not game_dir:
        print("  ERROR: VBSP and Game Dir are required for Auto-Compile mode.", file=sys.stderr)
        return None

    temp_dir = Path(tempfile.mkdtemp(prefix='lmopt_auto_'))
    temp_bsp = temp_dir / (input_path.stem + '.bsp')
    
    try:
        print(f"\\n[2/8] Compiling baseline BSP for reachability...", flush=True)
        try:
            compile_bsp(
                vbsp_path, input_path, game_dir,
                verbose=args.verbose, timeout=300,
                extra_args=['-emitsideids']
            )
            compiled_bsp_source = input_path.with_suffix('.bsp')
            shutil.copy2(compiled_bsp_source, temp_bsp)
        except (VBSPError, TimeoutError) as e:
            print("  ✗ Failed to auto-compile baseline BSP.", file=sys.stderr)
            return None

        # Point args.bsp to our newly compiled bsp
        args.bsp = str(temp_bsp)
        
        # Run standard BSP pipeline
        result = _run_bsp_mode(args, root, t0)

        if result is None:
            return None

        if 'vbsp_solver_result' in result:
            _print_vbsp_solver_results(result, args)
        else:
            _print_solver_results(result, args)

        # Apply Vis-Debug proxy textures if active
        vis_debug = getattr(args, 'vis_debug', False)
        if vis_debug:
            nv_sides = result.get('raw_nv_sides', result.get('never_visible_sides', set()))
            texlight_skip = result.get('texlight_sides', set())
            texlight_mats = result.get('texlight_mats', set())
            _VIS_MAT   = 'dev/dev_measuregeneric01'
            _INVIS_MAT = 'dev/dev_measuregeneric01b'
            
            painted = 0
            for world_node in root.get_children_by_name('world'):
                for solid in world_node.get_children_by_name('solid'):
                    for side in solid.get_children_by_name('side'):
                        cur_mat = (side.get_property('material') or '').upper().replace('\\\\', '/').strip('/')
                        if cur_mat.startswith('TOOLS/'): continue
                        if cur_mat in texlight_mats: continue
                        
                        sid = side.get_property('id')
                        if sid:
                            sid_int = int(sid)
                            if sid_int in texlight_skip: continue
                            mat = _INVIS_MAT if sid_int in nv_sides else _VIS_MAT
                            side.set_property('material', mat)
                            painted += 1
            print(f"\\n  [Vis Debug] Painted {painted} world faces with proxy textures.", flush=True)

        # Write output file immediately!
        from vmf_parser import VMFWriter
        VMFWriter().write_file(root, output_path)

        # Return the result so light_budget_enforcement can run!
        return result

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULATION MODE — original pipeline with geometry + raycasting
# ═══════════════════════════════════════════════════════════════════════════════

def _run_sim_mode(args, root, t0: float) -> dict | None:
    """Run the simulation-based optimization pipeline.
    
    NOTE: Simulation mode requires the light_sim and classifier modules,
    which are not included in this distribution. Use --bsp mode instead.
    """
    print("ERROR: Simulation mode is not available in this distribution.",
          file=sys.stderr)
    print("  Use --bsp mode with a compiled BSP file for optimization.",
          file=sys.stderr)
    print("  Example: python lmoptimizer.py mymap.vmf --bsp mymap.bsp",
          file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
    main()
