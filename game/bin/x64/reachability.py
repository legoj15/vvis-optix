#!/usr/bin/env python3
"""
Reachability — BFS flood-fill to find every position a player can reach.

Given a compiled BSP, this module discretizes the map into a 3D grid and
performs breadth-first search from player spawn points, simulating walking,
falling, jumping, and crouching to determine all reachable positions.

Usage:
    python reachability.py --bsp map.bsp [--grid-res 32] [--output reach.json] [--verbose]

Output:
    JSON file with reachable cells, their eye heights, and metadata.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from bsp_reader import BSPReader
from collision import (
    CollisionWorld, HULL_STAND, HULL_CROUCH,
    EYE_HEIGHT_STAND, EYE_HEIGHT_CROUCH,
    STEP_HEIGHT, JUMP_HEIGHT, CROUCH_JUMP_HEIGHT,
    MASK_PLAYERSOLID,
)

# Small Z offset so hull bottom doesn't touch the floor surface
GROUND_CLEARANCE = 0.5

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ─── Cell states ──────────────────────────────────────────────────────────────

CELL_UNKNOWN = 0
CELL_SOLID   = 1
CELL_EMPTY   = 2
CELL_REACH_STAND  = 3   # Reachable while standing
CELL_REACH_CROUCH = 4   # Reachable only while crouching

# ─── Grid coordinate system ──────────────────────────────────────────────────

@dataclass
class GridConfig:
    """Grid configuration derived from BSP world bounds."""
    res: float              # Grid cell size in world units
    mins: Tuple[float, float, float]  # World-space origin (snapped to grid)
    count: Tuple[int, int, int]       # Grid dimensions (nx, ny, nz)
    
    def world_to_grid(self, wx: float, wy: float, wz: float) -> Tuple[int, int, int]:
        """Convert world coordinates to grid indices."""
        gx = int((wx - self.mins[0]) / self.res)
        gy = int((wy - self.mins[1]) / self.res)
        gz = int((wz - self.mins[2]) / self.res)
        return (gx, gy, gz)
    
    def grid_to_world(self, gx: int, gy: int, gz: int) -> Tuple[float, float, float]:
        """Convert grid indices to world-space center of cell."""
        wx = self.mins[0] + (gx + 0.5) * self.res
        wy = self.mins[1] + (gy + 0.5) * self.res
        wz = self.mins[2] + gz * self.res  # Z is cell bottom (feet position)
        return (wx, wy, wz)
    
    def in_bounds(self, gx: int, gy: int, gz: int) -> bool:
        """Check if grid coordinates are within bounds."""
        return (0 <= gx < self.count[0] and
                0 <= gy < self.count[1] and
                0 <= gz < self.count[2])
    
    @property
    def total_cells(self) -> int:
        return self.count[0] * self.count[1] * self.count[2]


def make_grid_config(world_mins: Tuple[float, float, float],
                     world_maxs: Tuple[float, float, float],
                     res: float) -> GridConfig:
    """Create a grid config from world bounds."""
    # Snap mins down and maxs up to grid boundaries
    mx = math.floor(world_mins[0] / res) * res
    my = math.floor(world_mins[1] / res) * res
    mz = math.floor(world_mins[2] / res) * res
    
    nx = int(math.ceil((world_maxs[0] - mx) / res)) + 1
    ny = int(math.ceil((world_maxs[1] - my) / res)) + 1
    nz = int(math.ceil((world_maxs[2] - mz) / res)) + 1
    
    return GridConfig(res=res, mins=(mx, my, mz), count=(nx, ny, nz))


# ─── Reachability BFS ─────────────────────────────────────────────────────────

class ReachabilityMap:
    """BFS flood-fill reachability calculator.
    
    Discretizes the map into a 3D grid and finds all positions reachable
    from player spawn points via walking, jumping, falling, and crouching.
    """

    def __init__(self, world: CollisionWorld, grid_res: float = 16.0,
                 verbose: bool = False, bsp=None):
        self.world = world
        self.verbose = verbose
        
        # Get world bounds and shrink to playable area
        world_mins, world_maxs = self._compute_play_bounds(world)
        self.grid = make_grid_config(world_mins, world_maxs, grid_res)
        
        # Cell state array — flat for performance
        # Index = gz * (nx * ny) + gy * nx + gx
        total = self.grid.total_cells
        self._states = bytearray(total)  # All CELL_UNKNOWN initially
        
        # Track reachable cells with their properties
        self.reachable: Dict[Tuple[int,int,int], int] = {}  # (gx,gy,gz) → state
        self.parents: Dict[Tuple[int,int,int], Tuple[int,int,int]] = {}  # Track BFS path: (gx, gy, gz) -> (px, py, pz)
        
        # Build supplemental floor map from BSP face geometry.
        # This covers func_detail floors that are invisible to hull traces.
        self._face_floor_map: Dict[Tuple[int,int], List[float]] = {}
        if bsp is not None:
            self._build_face_floor_map(bsp, grid_res)
        
        if verbose:
            print(f"  Grid: {self.grid.count[0]}×{self.grid.count[1]}×{self.grid.count[2]} "
                  f"= {total:,} cells @ {grid_res}u resolution")
            print(f"  World: ({world_mins[0]:.0f},{world_mins[1]:.0f},{world_mins[2]:.0f}) - "
                  f"({world_maxs[0]:.0f},{world_maxs[1]:.0f},{world_maxs[2]:.0f})")
            if self._face_floor_map:
                print(f"  Face floor map: {len(self._face_floor_map)} grid cells "
                      f"with floor data from BSP faces")

    @staticmethod
    def _compute_play_bounds(world: CollisionWorld) -> Tuple[Tuple[float,float,float], Tuple[float,float,float]]:
        """Compute tight playable-area bounds from spawn points.
        
        Uses entity origins to determine a padded bounding box, much
        tighter than model 0 which includes skybox brushes.
        """
        spawns = world.get_spawn_points()
        if not spawns:
            return world.get_world_bounds()
        
        # Start from spawn points
        min_x = min(s[0] for s in spawns)
        min_y = min(s[1] for s in spawns)
        min_z = min(s[2] for s in spawns)
        max_x = max(s[0] for s in spawns)
        max_y = max(s[1] for s in spawns)
        max_z = max(s[2] for s in spawns)
        
        # Also include entity origins for a better estimate
        for ent in world.entities:
            origin_str = ent.get('origin', '')
            if not origin_str:
                continue
            try:
                parts = origin_str.split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    min_z = min(min_z, z)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
                    max_z = max(max_z, z)
            except ValueError:
                continue
        
        # Pad generously: players can jump/fall
        PAD_XY = 512.0
        PAD_Z_DOWN = 256.0
        PAD_Z_UP = 512.0
        
        # Clamp to world bounds
        w_mins, w_maxs = world.get_world_bounds()
        min_x = max(w_mins[0], min_x - PAD_XY)
        min_y = max(w_mins[1], min_y - PAD_XY)
        min_z = max(w_mins[2], min_z - PAD_Z_DOWN)
        max_x = min(w_maxs[0], max_x + PAD_XY)
        max_y = min(w_maxs[1], max_y + PAD_XY)
        max_z = min(w_maxs[2], max_z + PAD_Z_UP)
        
        return ((min_x, min_y, min_z), (max_x, max_y, max_z))

    def _cell_index(self, gx: int, gy: int, gz: int) -> int:
        """Convert grid coordinates to flat array index."""
        nx, ny, nz = self.grid.count
        return gz * (nx * ny) + gy * nx + gx

    def _get_state(self, gx: int, gy: int, gz: int) -> int:
        if not self.grid.in_bounds(gx, gy, gz):
            return CELL_SOLID
        return self._states[self._cell_index(gx, gy, gz)]

    def _set_state(self, gx: int, gy: int, gz: int, state: int) -> None:
        if self.grid.in_bounds(gx, gy, gz):
            self._states[self._cell_index(gx, gy, gz)] = state

    def _can_stand(self, wx: float, wy: float, wz: float) -> bool:
        """Test if a player can stand at this position (feet at wz)."""
        # Hull center is at feet + half height + ground clearance
        center = (wx, wy, wz + HULL_STAND[2] + GROUND_CLEARANCE)
        return not self.world.hull_test_thorough(center, HULL_STAND)

    def _can_crouch(self, wx: float, wy: float, wz: float) -> bool:
        """Test if a player can crouch at this position (feet at wz)."""
        center = (wx, wy, wz + HULL_CROUCH[2] + GROUND_CLEARANCE)
        return not self.world.hull_test_thorough(center, HULL_CROUCH)

    def _build_face_floor_map(self, bsp, cell_size: float) -> None:
        """Build a 2D spatial hash of floor Z-values from BSP face geometry.
        
        This catches func_detail floors that are absent from hull-0 brush
        collision, by scanning all renderable BSP faces with upward normals.
        """
        faces = bsp.read_faces()
        planes = bsp.read_planes()
        vertexes = bsp.read_vertexes()
        edges = bsp.read_edges()
        surfedges = bsp.read_surfedges()
        
        floor_map: Dict[Tuple[int,int], List[float]] = {}
        count = 0
        
        for face in faces:
            if face.planenum >= len(planes):
                continue
            plane = planes[face.planenum]
            nz = plane.normal[2]
            if face.side:
                nz = -nz
            # Only upward-facing surfaces (floors/ramps)
            if nz < 0.7:
                continue
            
            verts = bsp.get_face_vertices(face, vertexes, edges, surfedges)
            if len(verts) < 3:
                continue
            
            # Compute face AABB and average Z
            min_x = min(v[0] for v in verts)
            max_x = max(v[0] for v in verts)
            min_y = min(v[1] for v in verts)
            max_y = max(v[1] for v in verts)
            avg_z = sum(v[2] for v in verts) / len(verts)
            
            # Rasterize the XY bounding box into grid cells
            gx0 = int(math.floor(min_x / cell_size))
            gx1 = int(math.floor(max_x / cell_size))
            gy0 = int(math.floor(min_y / cell_size))
            gy1 = int(math.floor(max_y / cell_size))
            
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    key = (gx, gy)
                    if key not in floor_map:
                        floor_map[key] = []
                    floor_map[key].append(avg_z)
            count += 1
        
        # Sort Z values for efficient lookup
        for key in floor_map:
            floor_map[key].sort()
        
        self._face_floor_map = floor_map

    def _face_floor_z(self, wx: float, wy: float, wz: float,
                      max_drop: float) -> Optional[float]:
        """Query the face floor map for the highest floor below wz."""
        gx = int(math.floor(wx / self.grid.res))
        gy = int(math.floor(wy / self.grid.res))
        key = (gx, gy)
        zlist = self._face_floor_map.get(key)
        if not zlist:
            return None
        
        # Find highest Z that is at or below wz + small tolerance
        best = None
        upper = wz + 4.0   # Small tolerance above feet
        lower = wz - max_drop
        for z in reversed(zlist):  # Sorted ascending, iterate descending
            if z <= upper and z >= lower:
                best = z
                break  # Highest valid Z (sorted)
        return best

    def _has_floor(self, wx: float, wy: float, wz: float, 
                   max_drop: float = None) -> Optional[float]:
        """Find the floor Z below a position. Returns floor Z or None.
        
        Uses brush collision traces as primary method, with BSP face
        geometry as fallback (catches func_detail floors).
        
        Args:
            wx, wy, wz: feet position
            max_drop: maximum drop distance (default: step height + small margin)
        """
        if max_drop is None:
            max_drop = STEP_HEIGHT + 2.0
        
        brush_z = None
        hit_z_max = None
        
        # 9-point clustered sweep to catch thin ledges (like fences) that mathematical grid centers miss
        offsets = [(0, 0), (15.0, 15.0), (15.0, -15.0), (-15.0, 15.0), (-15.0, -15.0),
                   (0, 8.0), (0, -8.0), (8.0, 0), (-8.0, 0)]
        for ox, oy in offsets:
            start = (wx + ox, wy + oy, wz + 2.0)  # Start slightly above feet
            end = (wx + ox, wy + oy, wz - max_drop)
            hit = self.world.trace_ray_full(start, end)
            if hit.fraction < 1.0:
                hz = start[2] + hit.fraction * (end[2] - start[2])
                if hit_z_max is None or hz > hit_z_max:
                    hit_z_max = hz
        
        if hit_z_max is not None:
            # Add a 0.1 unit vertical epsilon lift so the hull tests (which use mathematically inclusive overlaps)
            # don't strictly trigger a 'colliding' result from perfectly resting on the AABB proxy mesh surface!
            brush_z = hit_z_max + 0.1
        
        # Fallback: check BSP face floor map (catches func_detail)
        face_z = self._face_floor_z(wx, wy, wz, max_drop)
        if face_z is not None:
            face_z += 0.1  # Same boundary-inclusive lift for func_detail surfaces
            
        # Return the highest floor found (closest to the query position)
        if brush_z is not None and face_z is not None:
            return max(brush_z, face_z)
        return brush_z if brush_z is not None else face_z

    def run(self) -> int:
        """Run the BFS flood-fill from all spawn points.
        
        Returns the number of reachable cells found.
        """
        spawns = self.world.get_spawn_points()
        if not spawns:
            print("  WARNING: No spawn points found in BSP!")
            return 0
        
        if self.verbose:
            print(f"  Found {len(spawns)} spawn point(s)")
        
        # BFS queue: (gx, gy, gz, state)
        queue: deque = deque()
        
        # Seed with spawn points
        for sp in spawns:
            # Try to find actual floor at spawn
            floor_z = self._has_floor(sp[0], sp[1], sp[2], max_drop=128.0)
            if floor_z is not None:
                feet_z = floor_z
            else:
                feet_z = sp[2]
            
            gx, gy, gz = self.grid.world_to_grid(sp[0], sp[1], feet_z)
            if self.grid.in_bounds(gx, gy, gz):
                state = CELL_REACH_STAND
                self._set_state(gx, gy, gz, state)
                self.reachable[(gx, gy, gz)] = state
                # No parent for initial seed points
                queue.append((gx, gy, gz, state))
                if self.verbose:
                    print(f"  Seed: ({sp[0]:.0f},{sp[1]:.0f},{feet_z:.0f}) "
                          f"→ grid ({gx},{gy},{gz})")
        
        # Seed with ladders and dismounts
        for ent in self.world.entities:
            cname = ent.get('classname', '').lower()
            if cname == 'func_useableladder':
                p0 = ent.get('point0')
                p1 = ent.get('point1')
                if p0 and p1:
                    try:
                        x0, y0, z0 = map(float, p0.split())
                        x1, y1, z1 = map(float, p1.split())
                        
                        # Interpolate points along the ladder
                        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
                        length = math.sqrt(dx*dx + dy*dy + dz*dz)
                        steps = int(math.ceil(length / self.grid.res))
                        
                        for i in range(steps + 1):
                            t = i / float(steps) if steps > 0 else 0.0
                            px = x0 + t * dx
                            py = y0 + t * dy
                            pz = z0 + t * dz
                            
                            for lhx in (-1, 0, 1):
                                for lhy in (-1, 0, 1):
                                    gx, gy, gz = self.grid.world_to_grid(px + lhx*8.0, py + lhy*8.0, pz)
                                    if self.grid.in_bounds(gx, gy, gz):
                                        wx, wy, wz = self.grid.grid_to_world(gx, gy, gz)
                                        # Ensure the snapped grid cell isn't physically inside a wall
                                        if not self.world.hull_test((wx, wy, wz + 18.0), HULL_CROUCH):
                                            state = CELL_REACH_STAND
                                            self._set_state(gx, gy, gz, state)
                                            self.reachable[(gx, gy, gz)] = state
                                            # No parent for initial seed points
                                            queue.append((gx, gy, gz, state))
                    except ValueError:
                        pass
            
            elif cname == 'info_ladder_dismount':
                orig = ent.get('origin')
                if orig:
                    try:
                        px, py, pz = map(float, orig.split())
                        gx, gy, gz = self.grid.world_to_grid(px, py, pz)
                        if self.grid.in_bounds(gx, gy, gz):
                            state = CELL_REACH_STAND
                            self._set_state(gx, gy, gz, state)
                            self.reachable[(gx, gy, gz)] = state
                            # No parent for initial seed points
                            queue.append((gx, gy, gz, state))
                    except ValueError:
                        pass
        
        # Neighbor offsets for horizontal movement
        NEIGHBORS_XY = [
            (1, 0), (-1, 0), (0, 1), (0, -1),   # Cardinals
            (1, 1), (1, -1), (-1, 1), (-1, -1),  # Diagonals
        ]
        
        t0 = time.perf_counter()
        cells_tested = 0
        
        while queue:
            gx, gy, gz, cur_state = queue.popleft()
            cells_tested += 1
            
            wx, wy, wz = self.grid.grid_to_world(gx, gy, gz)
            
            # ─── Walk: try each horizontal neighbor ───────────────────────
            for dx, dy in NEIGHBORS_XY:
                nx, ny = gx + dx, gy + dy
                
                # Try walking at same Z, ±1 step, ±2 for ramps
                for dz in (0, 1, -1, 2, -2):
                    nz = gz + dz
                    
                    if not self.grid.in_bounds(nx, ny, nz):
                        continue
                    # Check if already reached with equal or better state
                    # We only want to update if the new state is better (e.g., standing vs crouching)
                    # or if it's a new cell.
                    existing_state = self._get_state(nx, ny, nz)
                    if existing_state >= CELL_REACH_STAND and existing_state >= cur_state:
                        continue
                    
                    nwx, nwy, nwz = self.grid.grid_to_world(nx, ny, nz)
                    # Verify there's a floor first
                    floor_z = self._has_floor(nwx, nwy, nwz)
                    if floor_z is not None:
                        # Snap to floor grid cell
                        _, _, fgz = self.grid.world_to_grid(nwx, nwy, floor_z)
                        if self.grid.in_bounds(nx, ny, fgz):
                            existing = self._get_state(nx, ny, fgz)
                            
                            # Validate landing clearance!
                            new_state = -1
                            if self._can_stand(nwx, nwy, floor_z):
                                new_state = CELL_REACH_STAND
                            elif self._can_crouch(nwx, nwy, floor_z):
                                new_state = CELL_REACH_CROUCH
                                
                            if new_state > 0 and existing < new_state:
                                self._set_state(nx, ny, fgz, new_state)
                                self.reachable[(nx, ny, fgz)] = new_state
                                self.parents[(nx, ny, fgz)] = (gx, gy, gz)
                                queue.append((nx, ny, fgz, new_state))
                                break  # Found a valid walk, skip other dz
            
            # ─── Fall: check cells below ──────────────────────────────────
            fall_z = wz
            fall_floor = self._has_floor(wx, wy, wz, max_drop=4096.0)
            if fall_floor is not None and fall_floor < wz - STEP_HEIGHT:
                _, _, fgz = self.grid.world_to_grid(wx, wy, fall_floor)
                if self.grid.in_bounds(gx, gy, fgz):
                    existing = self._get_state(gx, gy, fgz)
                    # Validate landing clearance!
                    new_state = -1
                    if self._can_stand(wx, wy, fall_floor):
                        new_state = CELL_REACH_STAND
                    elif self._can_crouch(wx, wy, fall_floor):
                        new_state = CELL_REACH_CROUCH
                        
                    if new_state > 0 and existing < new_state:
                        self._set_state(gx, gy, fgz, new_state)
                        self.reachable[(gx, gy, fgz)] = new_state
                        self.parents[(gx, gy, fgz)] = (gx, gy, gz)
                        queue.append((gx, gy, fgz, new_state))
            
            # ─── Jump: check cells above ──────────────────────────────────
            # Players can crouch-jump from a standing or crouching state.
            jump_h = CROUCH_JUMP_HEIGHT
            
            max_jump_cells = int(math.ceil(jump_h / self.grid.res))
            for dz in range(1, max_jump_cells + 1):
                jz = gz + dz
                if not self.grid.in_bounds(gx, gy, jz):
                    break
                
                jwx, jwy, jwz = self.grid.grid_to_world(gx, gy, jz)
                
                # Check if we can fit there (crouched, since we pull legs up)
                if self._can_crouch(jwx, jwy, jwz):
                    # Also check if we can reach it (no ceiling blocking)
                    # Cast ray up from current position using crouched upper bound
                    # (since we crouch jump, the apex is modeled with a crouch hull clearance)
                    up_start = (wx, wy, wz + 36.0)
                    up_end = (wx, wy, jwz + 36.0)
                    up_hit = self.world.trace_ray_full(up_start, up_end)
                    if up_hit.fraction >= 0.99:
                        # We can reach this height — but we need a floor
                        # The jump arc carries us here then we fall
                        
                        # Bug B fix: also check current column (straight-up jump)
                        floor_z = self._has_floor(jwx, jwy, jwz)
                        if floor_z is not None:
                            _, _, fgz = self.grid.world_to_grid(jwx, jwy, floor_z)
                            if self.grid.in_bounds(gx, gy, fgz):
                                existing = self._get_state(gx, gy, fgz)
                                # Validate landing clearance!
                                new_state = -1
                                if self._can_stand(jwx, jwy, floor_z):
                                    new_state = CELL_REACH_STAND
                                elif self._can_crouch(jwx, jwy, floor_z):
                                    new_state = CELL_REACH_CROUCH
                                    
                                if new_state > 0 and existing < new_state:
                                    self._set_state(gx, gy, fgz, new_state)
                                    self.reachable[(gx, gy, fgz)] = new_state
                                    self.parents[(gx, gy, fgz)] = (gx, gy, gz)
                                    queue.append((gx, gy, fgz, new_state))
                        
                        # Also add neighbors horizontally from the apex
                        for dx, dy in NEIGHBORS_XY:
                            for dist in range(1, 4):
                                jnx, jny = gx + dx * dist, gy + dy * dist
                                if not self.grid.in_bounds(jnx, jny, jz):
                                    continue
                                jnwx, jnwy, _ = self.grid.grid_to_world(jnx, jny, jz)
                                
                                # Verify the entire horizontal path is clear for our crouched body
                                path_clear = True
                                for step in range(1, dist + 1):
                                    iwx = wx + dx * self.grid.res * step
                                    iwy = wy + dy * self.grid.res * step
                                    if not self._can_crouch(iwx, iwy, jwz):
                                        path_clear = False
                                        break
                                
                                if not path_clear:
                                    break
                                
                                floor_z = self._has_floor(jnwx, jnwy, jwz, max_drop=4096.0)
                                if floor_z is not None:
                                    _, _, fgz = self.grid.world_to_grid(jnwx, jnwy, floor_z)
                                    if self.grid.in_bounds(jnx, jny, fgz):
                                            existing = self._get_state(jnx, jny, fgz)
                                            # Validate landing clearance!
                                            new_state = -1
                                            if self._can_stand(jnwx, jnwy, floor_z):
                                                new_state = CELL_REACH_STAND
                                            elif self._can_crouch(jnwx, jnwy, floor_z):
                                                new_state = CELL_REACH_CROUCH
                                                
                                            if new_state > 0 and existing < new_state:
                                                self._set_state(jnx, jny, fgz, new_state)
                                                self.reachable[(jnx, jny, fgz)] = new_state
                                                self.parents[(jnx, jny, fgz)] = (gx, gy, gz)
                                                queue.append((jnx, jny, fgz, new_state))
                else:
                    # Bug A fix: continue instead of break — a platform at
                    # this height may block _can_crouch, but higher cells
                    # could be clear (e.g. thin railing at dz=1, open at dz=2)
                    continue
        
        elapsed = time.perf_counter() - t0
        
        stand_count = sum(1 for s in self.reachable.values() if s == CELL_REACH_STAND)
        crouch_count = sum(1 for s in self.reachable.values() if s == CELL_REACH_CROUCH)
        
        if self.verbose:
            print(f"\n  BFS complete in {elapsed:.2f}s")
            print(f"  Cells tested: {cells_tested:,}")
            print(f"  Reachable: {len(self.reachable):,} cells "
                  f"({stand_count} standing, {crouch_count} crouching)")
            
            # Z-level histogram
            from collections import Counter
            z_hist = Counter()
            for (gx, gy, gz), state in self.reachable.items():
                _, _, wz = self.grid.grid_to_world(gx, gy, gz)
                z_hist[int(wz)] += 1
            if z_hist:
                sorted_z = sorted(z_hist.keys())
                print(f"  Z range: {sorted_z[0]} to {sorted_z[-1]}")
                print(f"  Z histogram (top 10 levels):")
                for z, count in z_hist.most_common(10):
                    print(f"    Z={z:>6}: {count:>5} cells")
        
        return len(self.reachable)

    def get_eye_positions(self) -> List[Tuple[float, float, float]]:
        """Get all reachable eye positions in world coordinates."""
        positions = []
        for (gx, gy, gz), state in self.reachable.items():
            wx, wy, wz = self.grid.grid_to_world(gx, gy, gz)
            if state == CELL_REACH_STAND:
                positions.append((wx, wy, wz + EYE_HEIGHT_STAND))
                positions.append((wx, wy, wz + EYE_HEIGHT_CROUCH))
            else:
                positions.append((wx, wy, wz + EYE_HEIGHT_CROUCH))
        return positions

    def save(self, path: str) -> None:
        """Save reachability data to JSON."""
        eye_positions = self.get_eye_positions()
        
        data = {
            "grid_res": self.grid.res,
            "grid_mins": list(self.grid.mins),
            "grid_count": list(self.grid.count),
            "reachable_count": len(self.reachable),
            "standing_count": sum(1 for s in self.reachable.values() if s == CELL_REACH_STAND),
            "crouching_count": sum(1 for s in self.reachable.values() if s == CELL_REACH_CROUCH),
            "eye_positions": [[round(x, 1), round(y, 1), round(z, 1)] for x, y, z in eye_positions],
            "cells": [
                {"g": [gx, gy, gz], "s": state}
                for (gx, gy, gz), state in self.reachable.items()
            ],
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, separators=(',', ':'))
        
        size_kb = os.path.getsize(path) / 1024
        if self.verbose:
            print(f"  Saved {path} ({size_kb:.1f} KB, "
                  f"{len(eye_positions)} eye positions)")

    @staticmethod
    def load_eye_positions(path: str) -> List[Tuple[float, float, float]]:
        """Load eye positions from a saved reachability JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return [tuple(p) for p in data["eye_positions"]]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute player reachability map for a BSP file")
    parser.add_argument("--bsp", required=True, help="Path to BSP file")
    parser.add_argument("--grid-res", type=float, default=32.0,
                        help="Grid cell size in world units (default 32)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON file (default: <bsp>_reach.json)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed progress")
    args = parser.parse_args()

    if not os.path.exists(args.bsp):
        print(f"ERROR: BSP file not found: {args.bsp}")
        sys.exit(1)

    if args.output is None:
        base = os.path.splitext(args.bsp)[0]
        args.output = base + "_reach.json"

    print(f"\n{'='*60}")
    print(f"  Reachability — {os.path.basename(args.bsp)}")
    print(f"{'='*60}\n")

    # Load BSP
    t0 = time.perf_counter()
    bsp = BSPReader(args.bsp)
    bsp.read()
    world = CollisionWorld(bsp, verbose=args.verbose)

    # Run BFS
    rmap = ReachabilityMap(world, grid_res=args.grid_res, verbose=True)
    count = rmap.run()

    if count == 0:
        print("\n  ERROR: No reachable cells found!")
        sys.exit(1)

    # Save output
    rmap.save(args.output)
    
    elapsed = time.perf_counter() - t0
    print(f"\n  Total: {elapsed:.2f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
