import sys, os
from pathlib import Path
sys.path.insert(0, 'E:/GitHub/source-extreme-mapping-tools/game/bin/x64')

from bsp_reader import BSPReader
from vpk_reader import VPKReader
from collision import CollisionWorld
from reachability import ReachabilityMap
from visibility import VisibilityOracle, _face_sample_points
from vmt_checker import _resolve_search_paths, _find_dir_vpk
from vis_simulator import match_bsp_to_vmf, _canonicalize_plane
from vmf_parser import VMFParser
import math

def run_test():
    bsp = BSPReader('E:/GitHub/source-extreme-mapping-tools/game/bin/x64/bsp_unit_tests/isolated/visibility_sim_test_vistest-debug.bsp')
    bsp.read()

    vmf_path = 'E:/GitHub/source-extreme-mapping-tools/game/bin/x64/bsp_unit_tests/isolated/visibility_sim_test.vmf'
    parser = VMFParser()
    root = parser.parse_file(vmf_path)
    vmf_side_info = {}
    for side_node in root.get_all_recursive('side'):
        sid_str = side_node.get_property('id')
        if not sid_str: continue
        sid = int(sid_str)
        pl_str = side_node.get_property('plane')
        if not pl_str: continue
        g = pl_str.replace('(', '').replace(')', '')
        parts = g.strip().split()
        if len(parts) >= 9:
            p1 = (float(parts[0]), float(parts[1]), float(parts[2]))
            p2 = (float(parts[3]), float(parts[4]), float(parts[5]))
            p3 = (float(parts[6]), float(parts[7]), float(parts[8]))
            e1 = (p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])
            e2 = (p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2])
            nx = e1[1]*e2[2] - e1[2]*e2[1]
            ny = e1[2]*e2[0] - e1[0]*e2[2]
            nz = e1[0]*e2[1] - e1[1]*e2[0]
            length = math.sqrt(nx*nx + ny*ny + nz*nz)
            if length > 1e-10:
                nx /= length; ny /= length; nz /= length
                dist = nx*p1[0] + ny*p1[1] + nz*p1[2]
                nx, ny, nz, dist = _canonicalize_plane(nx, ny, nz, dist)
                mat = (side_node.get_property('material') or '').upper().replace('\\', '/').strip('/')
                vmf_side_info[sid] = {'normal': (nx, ny, nz), 'dist': dist, 'material': mat}

    bsp_faces = bsp.extract_all_face_data(verbose=False)
    face_data = match_bsp_to_vmf(bsp_faces, vmf_side_info, verbose=False)
    bsp_indices = list(set(face_data[31].bsp_face_indices)) if 31 in face_data else []

    vpk_readers = []
    game_dir = Path("E:/Steam/steamapps/common/Source SDK Base 2013 Multiplayer/sourcetest")
    search_paths = _resolve_search_paths(game_dir)
    for sp in search_paths:
        if sp.suffix.lower() == '.vpk':
            dir_vpk = _find_dir_vpk(sp)
            if dir_vpk and dir_vpk.exists():
                try:
                    r = VPKReader(dir_vpk)
                    if r.read():
                        vpk_readers.append(r)
                except Exception:
                    pass

    w = CollisionWorld(bsp, vpk_readers=vpk_readers)
    rmap = ReachabilityMap(w, grid_res=16.0, verbose=False)
    rmap.run()
    eyes = rmap.get_eye_positions()

    oracle = VisibilityOracle(bsp, w, eyes, verbose=True)

    work_items = []
    for f in bsp_indices:
        face = bsp.read_faces()[f]
        plane = bsp.read_planes()[face.planenum]
        verts = bsp.get_face_vertices(face, bsp.read_vertexes(), bsp.read_edges(), bsp.read_surfedges())
        if len(verts) < 3:
            continue
            
        pts = _face_sample_points(verts, plane.normal)
        offset_surface = []
        for s in pts[0]:
            offset_surface.append((
                s[0] + plane.normal[0],
                s[1] + plane.normal[1],
                s[2] + plane.normal[2]))
                
        n_surface = len(offset_surface)
        all_samples = offset_surface + pts[1]
        work_items.append((f, all_samples, n_surface, plane.normal))

    res = oracle.classify_faces(num_workers=0, custom_work_items=work_items)
    vis_count = sum(1 for v in res.values() if v.get("visible"))
    print(f"\nRESULTS:")
    print(f"Total BSP faces comprising Side 31: {len(bsp_indices)}")
    print(f"Total marked VISIBLE: {vis_count}")
    
    if vis_count > 0:
        print("\nList of visible faces:")
        found = False
        for k, v in res.items():
            if v.get("visible"):
                print(f"Face {k}")
                if not found:
                    found = True
                    # Let's print out the exact eye that sees this face to trace it down.
                    print(f"  Debug tracing why Face {k} is visible...")
                    tgt_face = bsp.read_faces()[k]
                    tgt_plane = bsp.read_planes()[tgt_face.planenum]
                    tgt_verts = bsp.get_face_vertices(tgt_face, bsp.read_vertexes(), bsp.read_edges(), bsp.read_surfedges())
                    t_pts = _face_sample_points(tgt_verts, tgt_plane.normal)
                    t_samps = [(s[0]+tgt_plane.normal[0], s[1]+tgt_plane.normal[1], s[2]+tgt_plane.normal[2]) for s in t_pts[0]] + t_pts[1]
                    
                    fc = w.leaf_cluster(t_samps[0])
                    for ei, eye in enumerate(eyes):
                        if w.is_cluster_visible(oracle.eye_clusters[ei], fc):
                            for si, s in enumerate(t_samps):
                                if sum((eye[i]-s[i])**2 for i in range(3))**0.5 > 8000.0: continue
                                if w.trace_ray_full(eye, s, 1).fraction >= 0.99:
                                    print(f"  Eye {eye} can see sample {si} {s}!")
                                    break

if __name__ == '__main__':
    run_test()
