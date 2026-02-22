import sys
from pathlib import Path
sys.path.insert(0, 'E:/GitHub/source-extreme-mapping-tools/game/bin/x64')

from bsp_reader import BSPReader
from collision import CollisionWorld
from reachability import ReachabilityMap
from visibility import VisibilityOracle, _face_sample_points
from vpk_reader import VPKReader
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

print("Running reachability...")
trmap = ReachabilityMap(w, grid_res=16.0)
trmap.run()

eye_positions = trmap.get_eye_positions()
print("Initializing Visibility Oracle...")
sim = VisibilityOracle(bsp, w, eye_positions, verbose=True)
print(f"Total reachable eye positions: {len(eye_positions)}")

target_faces = [142, 153, 148, 151]

faces = bsp.read_faces()
vertexes = bsp.read_vertexes()
edges = bsp.read_edges()
surfedges = bsp.read_surfedges()
planes = bsp.read_planes()

for face_idx in target_faces:
    print(f"\n--- Testing Face {face_idx} ---")
    if face_idx >= len(faces):
        print("Face index out of bounds!")
        continue
        
    face = faces[face_idx]
    
    # 1. Get sample points for the face using Oracle's native method
    verts = bsp.get_face_vertices(face, vertexes, edges, surfedges)
    normal = planes[face.planenum].normal if face.planenum < len(planes) else None
    if face.side and normal:
        normal = (-normal[0], -normal[1], -normal[2])
    
    samples_data = _face_sample_points(verts, normal)
    if samples_data is None:
        print("  Skipping: No samples (too small or invalid).")
        continue
    
    surf_samples, _ = samples_data
    # For basic testing, just check the surface samples
    samples = surf_samples

    # Attempt to find exactly which eye position can see it
    visible_from = []
    
    for sx, sy, sz in samples:
        for ex, ey, ez in eye_positions:
                dx = sx - ex
                dy = sy - ey
                dz = sz - ez
                dist = (dx*dx+dy*dy+dz*dz)**0.5
                if dist > 16384.0:
                    continue
                    
                hit = w.trace_ray_full((ex, ey, ez), (sx, sy, sz))
                hit_dist = ((hit.end_pos[0]-ex)**2 + (hit.end_pos[1]-ey)**2 + (hit.end_pos[2]-ez)**2)**0.5
                
                # Check if it hit the target or went past it
                if hit.fraction >= 0.999 or (hit_dist >= dist - 1.0):
                    visible_from.append((ex, ey, ez))
                    break # Seen!
        if visible_from:
            break
            
    print(f"  Visible? {len(visible_from) > 0}")
    if visible_from:
        eye = visible_from[0]
        print(f"  First visible from eye: {eye}")
        # Print bounding info around the eye
        print(f"  Eye location info: ")
        
        # Test if it's Floor 2 or Floor 1
        floor_label = "Floor 1" if eye[2] < 200 else "Floor 2"
        print(f"  It is located on {floor_label}.")
