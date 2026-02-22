import sys
from pathlib import Path
sys.path.insert(0, 'E:/GitHub/source-extreme-mapping-tools/game/bin/x64')

from bsp_reader import BSPReader
from vpk_reader import VPKReader
from collision import CollisionWorld
from reachability import ReachabilityMap
from vmt_checker import _resolve_search_paths, _find_dir_vpk

bsp = BSPReader('E:/GitHub/source-extreme-mapping-tools/game/bin/x64/bsp_unit_tests/isolated/visibility_sim_test.bsp')
bsp.read()

vpk_readers = []
game_dir = Path("E:/Steam/steamapps/common/Source SDK Base 2013 Multiplayer/sourcetest")
for sp in _resolve_search_paths(game_dir):
    if sp.suffix.lower() == '.vpk':
        d = _find_dir_vpk(sp)
        if d and d.exists(): vpk_readers.append(VPKReader(d))

w = CollisionWorld(bsp, vpk_readers=vpk_readers)

import collections
import math

CELL_REACH_STAND = 2
CELL_REACH_CROUCH = 1
STEP_HEIGHT = 18.0
JUMP_HEIGHT = 56.0
CROUCH_JUMP_HEIGHT = 58.5
NEIGHBORS_XY = [(1, 0), (-1, 0), (0, 1), (0, -1)]

class TracingReachabilityMap(ReachabilityMap):
    def _has_floor(self, x: float, y: float, z: float, max_drop: float = 256.0):
        # We also need to patch _has_floor inside TracingReachabilityMap
        end_z = max(z - max_drop, -10000.0)
        hit = self.world.trace_ray_full((x, y, z), (x, y, end_z))
        if hit.fraction < 1.0:
            return z - (max_drop * hit.fraction)
        return None

    def run(self) -> int:
        queue = collections.deque()
        self.parents = {}
        spawns = self.world.get_spawn_points()
        for sp in spawns:
            floor_z = self._has_floor(sp[0], sp[1], sp[2], max_drop=128.0)
            feet_z = floor_z if floor_z is not None else sp[2]
            gx, gy, gz = self.grid.world_to_grid(sp[0], sp[1], feet_z)
            if self.grid.in_bounds(gx, gy, gz):
                state = CELL_REACH_STAND
                self._set_state(gx, gy, gz, state)
                self.reachable[(gx, gy, gz)] = state
                queue.append((gx, gy, gz, state))
                self.parents[(gx, gy, gz, state)] = "SPAWN"

        max_fall_cells = int(math.ceil(256.0 / self.grid.res))
        max_step_cells = int(math.ceil(STEP_HEIGHT / self.grid.res))

        while queue:
            gx, gy, gz, state = queue.popleft()
            wx, wy, wz = self.grid.grid_to_world(gx, gy, gz)
            current_node = (gx, gy, gz, state)

            for dx, dy in NEIGHBORS_XY:
                nx, ny = gx + dx, gy + dy
                if not self.grid.in_bounds(nx, ny, gz): continue
                for dz in range(max_step_cells, -max_fall_cells - 1, -1):
                    nz = gz + dz
                    if not self.grid.in_bounds(nx, ny, nz): continue
                    nwx, nwy, nwz = self.grid.grid_to_world(nx, ny, nz)
                    floor_z = self._has_floor(nwx, nwy, nwz)
                    if floor_z is not None:
                        _, _, fgz = self.grid.world_to_grid(nwx, nwy, floor_z)
                        if self.grid.in_bounds(nx, ny, fgz):
                            existing = self._get_state(nx, ny, fgz)
                            if existing >= state: break
                            new_state = -1
                            if state == CELL_REACH_STAND and self._can_stand(nwx, nwy, floor_z): new_state = CELL_REACH_STAND
                            elif self._can_crouch(nwx, nwy, floor_z): new_state = CELL_REACH_CROUCH
                            if new_state > 0 and existing < new_state:
                                self._set_state(nx, ny, fgz, new_state)
                                self.reachable[(nx, ny, fgz)] = new_state
                                queue.append((nx, ny, fgz, new_state))
                                self.parents[(nx, ny, fgz, new_state)] = ("WALK", current_node)
                        break

            # ─── Jump ─────────────────────────────────────────────────────
            jump_h = CROUCH_JUMP_HEIGHT
            max_jump_cells = int(math.ceil(jump_h / self.grid.res))
            for dz in range(1, max_jump_cells + 1):
                jz = gz + dz
                if not self.grid.in_bounds(gx, gy, jz): break
                jwx, jwy, jwz = self.grid.grid_to_world(gx, gy, jz)
                if self._can_crouch(jwx, jwy, jwz):
                    up_start = (wx, wy, wz + 36.0)
                    up_end = (wx, wy, jwz + 36.0)
                    up_hit = self.world.trace_ray_full(up_start, up_end)
                    if up_hit.fraction >= 0.99:
                        floor_z = self._has_floor(jwx, jwy, jwz)
                        if floor_z is not None:
                            _, _, fgz = self.grid.world_to_grid(jwx, jwy, floor_z)
                            if self.grid.in_bounds(gx, gy, fgz):
                                existing = self._get_state(gx, gy, fgz)
                                new_state = -1
                                if self._can_stand(jwx, jwy, floor_z): new_state = CELL_REACH_STAND
                                elif self._can_crouch(jwx, jwy, floor_z): new_state = CELL_REACH_CROUCH
                                if new_state > 0 and existing < new_state:
                                    self._set_state(gx, gy, fgz, new_state)
                                    self.reachable[(gx, gy, fgz)] = new_state
                                    queue.append((gx, gy, fgz, new_state))
                                    self.parents[(gx, gy, fgz, new_state)] = ("JUMP_ST", current_node)
                        
                        # Apply new distance logic
                        for dx, dy in NEIGHBORS_XY:
                            for dist in range(1, 4):
                                jnx, jny = gx + dx * dist, gy + dy * dist
                                if not self.grid.in_bounds(jnx, jny, jz): continue
                                jnwx, jnwy, _ = self.grid.grid_to_world(jnx, jny, jz)
                                wall_trace = self.world.trace_ray_full((wx, wy, jwz + 18.0), (jnwx, jnwy, jwz + 18.0))
                                
                                is_debug = wx < -60.0 and wx > -110.0 and wy > -16.0 and wy < 16.0 and wz > 20.0 and wz < 40.0 and dx==-1
                                if is_debug:
                                    print(f"    Tracing wall dx={dx}, dist={dist} from {wx},{wy},{jwz+18.0} to {jnwx},{jnwy},{jwz+18.0}, frac={wall_trace.fraction}")
                                
                                if wall_trace.fraction < 0.99:
                                    if is_debug: print(f"      Wall hit at {wall_trace.end_pos}! Breaking dist loop.")
                                    break
                                    
                                if self._can_crouch(jnwx, jnwy, jwz):
                                    floor_z2 = self._has_floor(jnwx, jnwy, jwz)
                                    if is_debug:
                                        print(f"      Dist {dist} fwd jump reached {jnwx}, {jnwy}, floor_z={floor_z2}")
                                    if floor_z2 is not None:
                                        _, _, fgz = self.grid.world_to_grid(jnwx, jnwy, floor_z2)
                                        if self.grid.in_bounds(jnx, jny, fgz):
                                            existing = self._get_state(jnx, jny, fgz)
                                            new_state = -1
                                            if self._can_stand(jnwx, jnwy, floor_z2): new_state = CELL_REACH_STAND
                                            elif self._can_crouch(jnwx, jnwy, floor_z2): new_state = CELL_REACH_CROUCH
                                            if new_state > 0 and existing < new_state:
                                                self._set_state(jnx, jny, fgz, new_state)
                                                self.reachable[(jnx, jny, fgz)] = new_state
                                                queue.append((jnx, jny, fgz, new_state))
                                                self.parents[(jnx, jny, fgz, new_state)] = ("JUMP_FWD", current_node)
                                                if is_debug: print(f"      JUMP_FWD added for {jnwx}, {jnwy}!!!")
        target_node_1 = None
        max_x_val = 9999
        for node in self.parents:
            gx, gy, gz, st = node
            wx, wy, wz = self.grid.grid_to_world(gx, gy, gz)
            if wx <= -96 and 0 <= wz <= 192 and wy > 0:
                if wx < max_x_val:
                    max_x_val = wx
                    target_node_1 = node

        if target_node_1:
            wx, wy, wz = self.grid.grid_to_world(target_node_1[0], target_node_1[1], target_node_1[2])
            print(f"\nFloor 1 - Jumped the fence! Tracing back from {wx}, {wy}, {wz}")
        else:
            print("\nFloor 1 - Failed to jump fence!")
            
        return len(self.reachable)

trmap = TracingReachabilityMap(w, grid_res=16.0)
trmap.run()

