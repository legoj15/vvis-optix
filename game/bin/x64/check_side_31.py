import sys, os
from pathlib import Path
sys.path.insert(0, 'E:/GitHub/source-extreme-mapping-tools/game/bin/x64')

from bsp_reader import BSPReader
from vmf_parser import VMFParser
from vis_simulator import match_bsp_to_vmf, _canonicalize_plane
import math

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
        pts = [
            (float(parts[0]), float(parts[1]), float(parts[2])),
            (float(parts[3]), float(parts[4]), float(parts[5])),
            (float(parts[6]), float(parts[7]), float(parts[8]))
        ]
        p1, p2, p3 = pts
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

if 31 in face_data:
    bsp_indices = face_data[31].bsp_face_indices
    print(f"Side 31 mapped to BSP faces: {bsp_indices}")
else:
    print("Side 31 not mapped!")
