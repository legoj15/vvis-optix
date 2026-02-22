"""
Collision World — BSP-based collision for player movement simulation.

Reconstructs the Source Engine's collision world from BSP lumps (brushes,
nodes, leafs, planes) and static prop .phy collision models. Provides
hull testing (AABB overlap), ray tracing, and PVS (Potentially Visible Set)
queries needed by the reachability flood-fill and visibility oracle.

Usage:
    from bsp_reader import BSPReader
    from collision import CollisionWorld

    bsp = BSPReader("map.bsp")
    bsp.read()
    world = CollisionWorld(bsp)
    
    # Test if a player-sized hull fits at a position
    can_stand = not world.hull_test(position, HULL_STAND)
    
    # Cast a visibility ray
    hit = world.trace_ray(eye_pos, target_pos)
    if hit.fraction < 1.0:
        print(f"Blocked at {hit.fraction:.2f}")
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from bsp_reader import (
    BSPReader, BSPPlane, BSPNode, BSPLeaf, BSPBrush, BSPBrushSide,
    BSPModel, BSPStaticProp,
    MASK_PLAYERSOLID, MASK_VISIBLE, CONTENTS_SOLID, CONTENTS_PLAYERCLIP,
)

Vec3 = Tuple[float, float, float]

# ─── Player hull dimensions ──────────────────────────────────────────────────

#                     (half_width, half_width, half_height)
HULL_STAND  = (16.0, 16.0, 36.0)   # 32×32×72, centered at feet + 36
HULL_CROUCH = (16.0, 16.0, 18.0)   # 32×32×36, centered at feet + 18

EYE_HEIGHT_STAND  = 64.0   # units above feet
EYE_HEIGHT_CROUCH = 28.0

STEP_HEIGHT = 18.0
JUMP_HEIGHT = 56.0
CROUCH_JUMP_HEIGHT = 65.0

# ─── Trace result ─────────────────────────────────────────────────────────────

@dataclass
class TraceResult:
    """Result of a ray or hull trace through the collision world."""
    fraction: float = 1.0       # 0.0 = start is solid, 1.0 = no hit
    end_pos: Vec3 = (0.0, 0.0, 0.0)
    plane_normal: Vec3 = (0.0, 0.0, 0.0)
    plane_dist: float = 0.0
    start_solid: bool = False
    all_solid: bool = False
    contents: int = 0

# ─── AABB-Triangle overlap (SAT) ─────────────────────────────────────────────

def _aabb_tri_overlap(mins: Vec3, maxs: Vec3,
                      tri: Tuple[Vec3, Vec3, Vec3]) -> bool:
    """Test if an AABB overlaps a triangle using SAT.
    
    Fast path: AABB-AABB reject, then full 13-axis SAT.
    """
    v0, v1, v2 = tri
    
    # Quick AABB-AABB reject
    if (min(v0[0], v1[0], v2[0]) >= maxs[0] or
        max(v0[0], v1[0], v2[0]) <= mins[0] or
        min(v0[1], v1[1], v2[1]) >= maxs[1] or
        max(v0[1], v1[1], v2[1]) <= mins[1] or
        min(v0[2], v1[2], v2[2]) >= maxs[2] or
        max(v0[2], v1[2], v2[2]) <= mins[2]):
        return False
    
    # Center and half-extents of AABB
    cx = (mins[0] + maxs[0]) * 0.5
    cy = (mins[1] + maxs[1]) * 0.5
    cz = (mins[2] + maxs[2]) * 0.5
    hx = (maxs[0] - mins[0]) * 0.5
    hy = (maxs[1] - mins[1]) * 0.5
    hz = (maxs[2] - mins[2]) * 0.5
    
    # Translate triangle to AABB center
    t0 = (v0[0] - cx, v0[1] - cy, v0[2] - cz)
    t1 = (v1[0] - cx, v1[1] - cy, v1[2] - cz)
    t2 = (v2[0] - cx, v2[1] - cy, v2[2] - cz)
    
    # Triangle edges
    e0 = (t1[0] - t0[0], t1[1] - t0[1], t1[2] - t0[2])
    e1 = (t2[0] - t1[0], t2[1] - t1[1], t2[2] - t1[2])
    e2 = (t0[0] - t2[0], t0[1] - t2[1], t0[2] - t2[2])
    
    # Test 9 cross product axes (edge × AABB face normal)
    edges = (e0, e1, e2)
    for edge in edges:
        # axis = edge × (1,0,0) = (0, -edge[2], edge[1])
        ax_y, ax_z = -edge[2], edge[1]
        r = hy * abs(ax_y) + hz * abs(ax_z)
        p0 = t0[1] * ax_y + t0[2] * ax_z
        p1 = t1[1] * ax_y + t1[2] * ax_z
        p2 = t2[1] * ax_y + t2[2] * ax_z
        if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
            return False
        
        # axis = edge × (0,1,0) = (edge[2], 0, -edge[0])
        ax_x, ax_z = edge[2], -edge[0]
        r = hx * abs(ax_x) + hz * abs(ax_z)
        p0 = t0[0] * ax_x + t0[2] * ax_z
        p1 = t1[0] * ax_x + t1[2] * ax_z
        p2 = t2[0] * ax_x + t2[2] * ax_z
        if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
            return False
        
        # axis = edge × (0,0,1) = (-edge[1], edge[0], 0)
        ax_x, ax_y = -edge[1], edge[0]
        r = hx * abs(ax_x) + hy * abs(ax_y)
        p0 = t0[0] * ax_x + t0[1] * ax_y
        p1 = t1[0] * ax_x + t1[1] * ax_y
        p2 = t2[0] * ax_x + t2[1] * ax_y
        if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
            return False
    
    # Test triangle normal axis
    nx = e0[1] * e2[2] - e0[2] * e2[1]
    ny = e0[2] * e2[0] - e0[0] * e2[2]
    nz = e0[0] * e2[1] - e0[1] * e2[0]
    r = hx * abs(nx) + hy * abs(ny) + hz * abs(nz)
    d = t0[0] * nx + t0[1] * ny + t0[2] * nz
    if abs(d) > r:
        return False
    
    return True

# ─── Collision World ──────────────────────────────────────────────────────────


class CollisionWorld:
    """BSP-based collision world for hull tracing and visibility queries.
    
    Loads all collision-relevant BSP lumps on construction and provides
    fast queries against brush geometry and static prop collision meshes.
    """

    def __init__(self, bsp: BSPReader, vpk_readers=None, verbose: bool = False):
        """Build collision world from a parsed BSP file.
        
        Args:
            bsp: A BSPReader that has already been .read()
            vpk_readers: Optional list of VPKReader instances for loading
                         static prop .phy files
            verbose: Print loading statistics
        """
        self.vpk_paths = [r._dir_path for r in vpk_readers] if vpk_readers else []
        self.planes: List[BSPPlane] = bsp.read_planes()
        self.nodes: List[BSPNode] = bsp.read_nodes()
        self.leafs: List[BSPLeaf] = bsp.read_leafs()
        self.brushes: List[BSPBrush] = bsp.read_brushes()
        self.brushsides: List[BSPBrushSide] = bsp.read_brushsides()
        self.models: List[BSPModel] = bsp.read_models()
        self.leafbrushes: List[int] = bsp.read_leafbrushes()
        
        # PVS data
        vis_data = bsp.read_visibility()
        if vis_data is not None:
            self._num_clusters, self._cluster_offsets, self._vis_data = vis_data
        else:
            self._num_clusters = 0
            self._cluster_offsets = []
            self._vis_data = b''
        
        # Pre-mark which brushes have been checked (avoid rechecking in traces)
        self._check_count = 0
        self._brush_check_counts = [0] * len(self.brushes)
        
        # Entity brushes: parse entity lump for brush model entities
        self.entities = bsp.read_entities()
        
        # Collect headnodes for solid brush entities (func_brush, func_door, etc)
        self._entity_headnodes = []
        _non_solid_classes = {'func_illusionary', 'func_dustcloud', 'func_smokevolume', 
                              'trigger_multiple', 'trigger_once', 'trigger_teleport',
                              'trigger_hurt', 'func_buyzone', 'func_bomb_target'}
        for ent in self.entities:
            cname = ent.get('classname', '')
            model_str = ent.get('model', '')
            if model_str.startswith('*') and cname not in _non_solid_classes:
                try:
                    model_idx = int(model_str[1:])
                    # if the entity has an origin, we might need a transform, but typically
                    # static visibility ignores origin shifts for doors, or they are spawned at 0,0,0 initially
                    if model_idx < len(self.models):
                        self._entity_headnodes.append(self.models[model_idx].headnode)
                except ValueError:
                    pass
        
        # Static prop collision triangles
        self._prop_triangles: List[Tuple[Vec3, Vec3, Vec3]] = []
        # Spatial hash for fast prop triangle queries: (cx, cy, cz) -> [tri_idx]
        self._prop_grid: dict = {}
        self._prop_cell_size = 64.0
        self._load_static_prop_collisions(bsp, vpk_readers if vpk_readers else [], verbose)

        
        # Store the BSP reader for point_in_leaf
        self._bsp = bsp
        
        if verbose:
            print(f"  CollisionWorld: {len(self.nodes)} nodes, "
                  f"{len(self.leafs)} leafs, {len(self.brushes)} brushes, "
                  f"{len(self.planes)} planes, {self._num_clusters} clusters, "
                  f"{len(self._prop_triangles)} static prop triangles")

    def _load_static_prop_collisions(self, bsp: BSPReader, vpk_readers, 
                                       verbose: bool) -> None:
        """Load .phy collision models for solid static props."""
        from phy_reader import PHYReader, load_phy_from_vpk
        
        static_props = bsp.read_static_props()
        loaded = 0
        total_tris = 0
        skipped_not_solid = 0
        skipped_no_phy = 0
        
        if verbose:
            print(f"  Static props: {len(static_props)} total, "
                  f"{len(vpk_readers)} VPK readers", flush=True)
        
        for prop in static_props:
            if prop.solid < 1:  # Not solid
                skipped_not_solid += 1
                continue
            
            # Try to load .phy from VPKs
            phy_model = None
            for vpk in vpk_readers:
                phy_model = load_phy_from_vpk(vpk, prop.model_name)
                if phy_model is not None:
                    break
            
            # Emulate precise Source Engine continuous hull sweeps by wrapping thin fences 
            # with explicit physical proxy blocks. This solves the discrete grid teleporting bypass.
            
            if 'exterior_fence003b' in prop.model_name.lower():
                # For exterior_fence003b:
                # 1. Provide the physical mesh proxy from floor (-74.37 local Z) up to origin Z (gate handle top)
                # 2. Add an extra AABB for the upper barbed wire (from Z=0 to Z=32.92) to block crouching OVER the fence centrally.
                # Because the fence is X-thickened, it prevents grid tunneling, and provides a ledge at Z=0 for jumping.
                
                # Main body (from -74 down to origin, thick to prevent teleport parsing)
                hull_base_min = (-5.0, -24.0, -74.5)
                # EPSILON FIX: Use -0.1 instead of 0.0 so a player precisely standing at Z=0.0 doesn't mathematically overlap the top face.
                hull_base_max = (5.0, 24.0, -0.1) 
                
                for box_min, box_max in [(hull_base_min, hull_base_max)]:
                    corners = [
                        (box_min[0], box_min[1], box_min[2]),
                        (box_max[0], box_min[1], box_min[2]),
                        (box_max[0], box_max[1], box_min[2]),
                        (box_min[0], box_max[1], box_min[2]),
                        (box_min[0], box_min[1], box_max[2]),
                        (box_max[0], box_min[1], box_max[2]),
                        (box_max[0], box_max[1], box_max[2]),
                        (box_min[0], box_max[1], box_max[2])
                    ]
                    
                    indices = [
                        (0, 1, 2), (0, 2, 3), # Bottom
                        (4, 5, 6), (4, 6, 7), # Top
                        (0, 1, 5), (0, 5, 4), # Front
                        (3, 2, 6), (3, 6, 7), # Back
                        (0, 3, 7), (0, 7, 4), # Left
                        (1, 2, 6), (1, 6, 5)  # Right
                    ]
                    
                    for i0, i1, i2 in indices:
                        v0 = PHYReader.transform_point(corners[i0], prop.origin, prop.angles)
                        v1 = PHYReader.transform_point(corners[i1], prop.origin, prop.angles)
                        v2 = PHYReader.transform_point(corners[i2], prop.origin, prop.angles)
                        self._prop_triangles.append((v0, v1, v2))
                        
                # Notice we skip any broken `.phy` entirely:
                continue
                
            use_aabb_proxy = False
            
            if use_aabb_proxy:
                # Add precise MDL AABB proxy to properly block movement
                import struct
                hull_min, hull_max = None, None
                for vpk in vpk_readers:
                    mdl_data = vpk.read_file(prop.model_name)
                    if mdl_data and mdl_data[0:4] == b'IDST':
                        if len(mdl_data) >= 128:
                            hull_min = struct.unpack_from('<3f', mdl_data, 104)
                            hull_max = struct.unpack_from('<3f', mdl_data, 116)
                        break
                
                if hull_min and hull_max and not (hull_min == (0.0, 0.0, 0.0) and hull_max == (0.0, 0.0, 0.0)):
                    corners = [
                        (hull_min[0], hull_min[1], hull_min[2]),
                        (hull_max[0], hull_min[1], hull_min[2]),
                        (hull_max[0], hull_max[1], hull_min[2]),
                        (hull_min[0], hull_max[1], hull_min[2]),
                        (hull_min[0], hull_min[1], hull_max[2]),
                        (hull_max[0], hull_min[1], hull_max[2]),
                        (hull_max[0], hull_max[1], hull_max[2]),
                        (hull_min[0], hull_max[1], hull_max[2])
                    ]
                    indices = [
                        (0, 2, 1), (0, 3, 2), # Bottom
                        (4, 5, 6), (4, 6, 7), # Top
                        (0, 1, 5), (0, 5, 4), # Front
                        (1, 2, 6), (1, 6, 5), # Right
                        (2, 3, 7), (2, 7, 6), # Back
                        (3, 0, 4), (3, 4, 7)  # Left
                    ]
                    
                    prop_tris = 0
                    for i0, i1, i2 in indices:
                        v0 = PHYReader.transform_point(corners[i0], prop.origin, prop.angles)
                        v1 = PHYReader.transform_point(corners[i1], prop.origin, prop.angles)
                        v2 = PHYReader.transform_point(corners[i2], prop.origin, prop.angles)
                        self._prop_triangles.append((v0, v1, v2))
                        prop_tris += 1
                        
                    total_tris += prop_tris
                    if phy_model is None:
                        loaded += 1
                        continue
                elif phy_model is None:
                    skipped_no_phy += 1
                    continue
            else:
                pass # Only use PHY
            
            if phy_model is None:
                continue
            
            # Get triangles and transform to world space
            # IVP physics stores vertices in meters; Source Engine uses inches
            # Scale factor: 1 meter = 1/0.0254 inches ≈ 39.3701
            IVP_TO_SOURCE = 1.0 / 0.0254
            try:
                # phy_model is already a PHYModel (returned by load_phy_from_vpk)
                prop_tris = 0
                for solid in phy_model.solids:
                    for tri_idx in solid.triangles:
                        i0, i1, i2 = tri_idx
                        if i0 < len(solid.vertices) and i1 < len(solid.vertices) and i2 < len(solid.vertices):
                            # Scale from IVP meters to Source inches first
                            sv0 = solid.vertices[i0]
                            sv1 = solid.vertices[i1]
                            sv2 = solid.vertices[i2]
                            # IVP physics engine stores coordinates in meters. We must scale to Source Engine inches.
                            sv0_s = (sv0[0] * IVP_TO_SOURCE, sv0[1] * IVP_TO_SOURCE, sv0[2] * IVP_TO_SOURCE)
                            sv1_s = (sv1[0] * IVP_TO_SOURCE, sv1[1] * IVP_TO_SOURCE, sv1[2] * IVP_TO_SOURCE)
                            sv2_s = (sv2[0] * IVP_TO_SOURCE, sv2[1] * IVP_TO_SOURCE, sv2[2] * IVP_TO_SOURCE)
                                
                            v0 = PHYReader.transform_point(sv0_s, prop.origin, prop.angles)
                            v1 = PHYReader.transform_point(sv1_s, prop.origin, prop.angles)
                            v2 = PHYReader.transform_point(sv2_s, prop.origin, prop.angles)
                            self._prop_triangles.append((v0, v1, v2))
                            prop_tris += 1

                if prop_tris > 0:
                    loaded += 1
                    total_tris += prop_tris
            except Exception as e:
                if verbose:
                    print(f"    Warning: Failed to process '{prop.model_name}': {e}",
                          flush=True)

        
        # Build spatial hash for fast hull queries
        cs = self._prop_cell_size
        for ti, (v0, v1, v2) in enumerate(self._prop_triangles):
            # Compute triangle AABB and insert into all overlapping cells
            tmin_x = min(v0[0], v1[0], v2[0])
            tmin_y = min(v0[1], v1[1], v2[1])
            tmin_z = min(v0[2], v1[2], v2[2])
            tmax_x = max(v0[0], v1[0], v2[0])
            tmax_y = max(v0[1], v1[1], v2[1])
            tmax_z = max(v0[2], v1[2], v2[2])
            cx0 = int(math.floor(tmin_x / cs))
            cy0 = int(math.floor(tmin_y / cs))
            cz0 = int(math.floor(tmin_z / cs))
            cx1 = int(math.floor(tmax_x / cs))
            cy1 = int(math.floor(tmax_y / cs))
            cz1 = int(math.floor(tmax_z / cs))
            for gx in range(cx0, cx1 + 1):
                for gy in range(cy0, cy1 + 1):
                    for gz in range(cz0, cz1 + 1):
                        key = (gx, gy, gz)
                        if key not in self._prop_grid:
                            self._prop_grid[key] = []
                        self._prop_grid[key].append(ti)
        
        if verbose:
            print(f"  Loaded {loaded} static prop collision models "
                  f"({len(self._prop_triangles)} triangles, "
                  f"{len(self._prop_grid)} grid cells) "
                  f"[skipped: {skipped_not_solid} non-solid, "
                  f"{skipped_no_phy} no .phy]")

    # ─── Point-in-leaf ────────────────────────────────────────────────────────

    def point_in_leaf(self, point: Vec3) -> int:
        """Walk BSP tree to find the leaf containing a point."""
        return self._bsp.point_in_leaf(point, self.nodes, self.planes)

    def leaf_contents(self, point: Vec3) -> int:
        """Get the contents flags of the leaf containing a point."""
        leaf_idx = self.point_in_leaf(point)
        if leaf_idx < len(self.leafs):
            return self.leafs[leaf_idx].contents
        return 0

    def leaf_cluster(self, point: Vec3) -> int:
        """Get the visibility cluster of the leaf containing a point."""
        leaf_idx = self.point_in_leaf(point)
        if leaf_idx < len(self.leafs):
            return self.leafs[leaf_idx].cluster
        return -1

    # ─── Hull test ────────────────────────────────────────────────────────────

    def hull_test(self, center: Vec3, half_extents: Vec3,
                  mask: int = MASK_PLAYERSOLID) -> bool:
        """Test if an AABB overlaps any solid brush.
        
        Args:
            center: Center of the AABB
            half_extents: Half-size in each axis (e.g., (16, 16, 36))
            mask: Content mask to test against (default MASK_PLAYERSOLID)
            
        Returns:
            True if the hull overlaps solid geometry.
        """
        # Find which leaf the center is in
        leaf_idx = self.point_in_leaf(center)
        if leaf_idx >= len(self.leafs):
            return False
        leaf = self.leafs[leaf_idx]
        
        # Quick reject: if the leaf itself has no solid contents, skip
        if not (leaf.contents & mask):
            return False
        
        # Test against brushes in this leaf
        hx, hy, hz = half_extents
        for bi in range(leaf.firstleafbrush, 
                        leaf.firstleafbrush + leaf.numleafbrushes):
            if bi >= len(self.leafbrushes):
                break
            brush_idx = self.leafbrushes[bi]
            if brush_idx >= len(self.brushes):
                continue
            brush = self.brushes[brush_idx]
            
            if not (brush.contents & mask):
                continue
            
            if self._hull_intersects_brush(center, half_extents, brush):
                return True
        
        return False

    def hull_test_thorough(self, center: Vec3, half_extents: Vec3,
                           mask: int = MASK_PLAYERSOLID) -> bool:
        """Test AABB overlap by checking all leaves the AABB touches.
        
        Also checks against static prop collision triangles.
        More accurate than hull_test() for large hulls that span multiple
        leaves — walks the BSP tree to find all overlapping leaves.
        """
        hx, hy, hz = half_extents
        cx, cy, cz = center
        mins = (cx - hx, cy - hy, cz - hz)
        maxs = (cx + hx, cy + hy, cz + hz)
        
        self._check_count += 1
        
        # Test worldspawn
        if self._hull_test_node(0, mins, maxs, mask):
            return True
            
        # Test solid brush entities
        for hn in self._entity_headnodes:
            if self._hull_test_node(hn, mins, maxs, mask):
                return True
                
        # Test static props
        if self._prop_triangles:
            return self._hull_test_props(mins, maxs)
        
        return False
    
    def _hull_test_props(self, mins: Vec3, maxs: Vec3) -> bool:
        """Test AABB overlap against static prop collision triangles.
        
        Uses spatial hash for fast lookup and SAT (Separating Axis Theorem)
        for accurate AABB-triangle overlap detection.
        """
        cs = self._prop_cell_size
        cx0 = int(math.floor(mins[0] / cs))
        cy0 = int(math.floor(mins[1] / cs))
        cz0 = int(math.floor(mins[2] / cs))
        cx1 = int(math.floor(maxs[0] / cs))
        cy1 = int(math.floor(maxs[1] / cs))
        cz1 = int(math.floor(maxs[2] / cs))
        
        tested = set()
        for gx in range(cx0, cx1 + 1):
            for gy in range(cy0, cy1 + 1):
                for gz in range(cz0, cz1 + 1):
                    bucket = self._prop_grid.get((gx, gy, gz))
                    if bucket is None:
                        continue
                    for ti in bucket:
                        if ti in tested:
                            continue
                        tested.add(ti)
                        if _aabb_tri_overlap(mins, maxs,
                                             self._prop_triangles[ti]):
                            return True
        return False

    def _hull_test_node(self, node_idx: int, mins: Vec3, maxs: Vec3,
                        mask: int) -> bool:
        """Recursively walk BSP tree testing AABB against brushes."""
        if node_idx < 0:
            # Leaf node
            leaf_idx = -(node_idx + 1)
            if leaf_idx >= len(self.leafs):
                return False
            leaf = self.leafs[leaf_idx]
            
            if not (leaf.contents & mask):
                return False
            
            for bi in range(leaf.firstleafbrush,
                            leaf.firstleafbrush + leaf.numleafbrushes):
                if bi >= len(self.leafbrushes):
                    break
                brush_idx = self.leafbrushes[bi]
                if brush_idx >= len(self.brushes):
                    continue
                
                # Skip already-checked brushes this query
                if self._brush_check_counts[brush_idx] == self._check_count:
                    continue
                self._brush_check_counts[brush_idx] = self._check_count
                
                brush = self.brushes[brush_idx]
                if not (brush.contents & mask):
                    continue
                
                cx = (mins[0] + maxs[0]) * 0.5
                cy = (mins[1] + maxs[1]) * 0.5
                cz = (mins[2] + maxs[2]) * 0.5
                hx = (maxs[0] - mins[0]) * 0.5
                hy = (maxs[1] - mins[1]) * 0.5
                hz = (maxs[2] - mins[2]) * 0.5
                
                if self._hull_intersects_brush((cx, cy, cz), (hx, hy, hz), brush):
                    return True
            
            return False
        
        # Interior node — check which children the AABB overlaps
        if node_idx >= len(self.nodes):
            return False
        node = self.nodes[node_idx]
        plane = self.planes[node.planenum]
        
        nx, ny, nz = plane.normal
        
        # Compute distance of AABB center to plane
        cx = (mins[0] + maxs[0]) * 0.5
        cy = (mins[1] + maxs[1]) * 0.5
        cz = (mins[2] + maxs[2]) * 0.5
        d = cx * nx + cy * ny + cz * nz - plane.dist
        
        # Compute AABB half-extent projected along plane normal
        hx = (maxs[0] - mins[0]) * 0.5
        hy = (maxs[1] - mins[1]) * 0.5
        hz = (maxs[2] - mins[2]) * 0.5
        extent = abs(nx) * hx + abs(ny) * hy + abs(nz) * hz
        
        if d > extent:
            # Entirely in front
            return self._hull_test_node(node.children[0], mins, maxs, mask)
        elif d < -extent:
            # Entirely behind
            return self._hull_test_node(node.children[1], mins, maxs, mask)
        else:
            # Spanning — check both
            return (self._hull_test_node(node.children[0], mins, maxs, mask) or
                    self._hull_test_node(node.children[1], mins, maxs, mask))

    def _hull_intersects_brush(self, center: Vec3, half_extents: Vec3,
                                brush: BSPBrush) -> bool:
        """Test if an AABB at center intersects a convex brush.
        
        Uses the Minkowski sum approach: expand each brush plane by the
        hull's half-extent projection along the plane normal, then test
        if the center point is inside all expanded planes.
        """
        cx, cy, cz = center
        hx, hy, hz = half_extents
        
        for si in range(brush.firstside, brush.firstside + brush.numsides):
            if si >= len(self.brushsides):
                return False
            bs = self.brushsides[si]
            if bs.bevel:
                continue  # Skip bevel planes
            if bs.planenum >= len(self.planes):
                return False
            
            plane = self.planes[bs.planenum]
            nx, ny, nz = plane.normal
            
            # Expand plane distance by hull half-extent projection
            expand = abs(nx) * hx + abs(ny) * hy + abs(nz) * hz
            
            # Test: if center is in front of expanded plane → outside brush
            d = cx * nx + cy * ny + cz * nz - (plane.dist + expand)
            if d > 0:
                return False  # Center is outside this expanded plane
        
        return True  # Inside all expanded planes → overlapping

    # ─── Ray tracing ──────────────────────────────────────────────────────────

    def trace_ray(self, start: Vec3, end: Vec3,
                  mask: int = MASK_PLAYERSOLID) -> TraceResult:
        """Cast a ray through the BSP and return the nearest hit.
        
        Traces against world brushes (model 0 headnode). Does NOT trace
        against static prop triangles — use trace_ray_full() for that.
        """
        result = TraceResult()
        result.fraction = 1.0
        sx, sy, sz = start
        ex, ey, ez = end
        result.end_pos = end
        
        self._check_count += 1
        self._trace_node(0, 0.0, 1.0, start, end, result, mask)
        
        for hn in self._entity_headnodes:
            if result.fraction <= 0.0: break
            self._trace_node(hn, 0.0, 1.0, start, end, result, mask)
        
        if result.fraction < 1.0:
            fx = sx + result.fraction * (ex - sx)
            fy = sy + result.fraction * (ey - sy)
            fz = sz + result.fraction * (ez - sz)
            result.end_pos = (fx, fy, fz)
        
        return result

    def trace_ray_full(self, start: Vec3, end: Vec3,
                       mask: int = MASK_PLAYERSOLID) -> TraceResult:
        """Trace ray against brushes AND static prop collision triangles."""
        result = self.trace_ray(start, end, mask)
        
        if self._prop_triangles:
            result = self._trace_prop_triangles(start, end, result)
        
        return result

    def _trace_node(self, node_idx: int, start_frac: float, end_frac: float,
                    start: Vec3, end: Vec3, result: TraceResult,
                    mask: int) -> None:
        """Recursively trace a ray through the BSP tree.
        
        The ray (start→end) is always the ORIGINAL full ray. start_frac/end_frac
        define the parametric sub-interval currently being tested. Brush clipping
        always works in original-ray fraction space.
        """
        if result.fraction <= start_frac:
            return  # Already found a closer hit
        
        if node_idx < 0:
            # Leaf
            leaf_idx = -(node_idx + 1)
            if leaf_idx < len(self.leafs):
                leaf = self.leafs[leaf_idx]
                if leaf.contents & mask:
                    # Brush clipping uses original start/end
                    self._trace_leaf_brushes(leaf, start, end, result, mask)
            return
        
        if node_idx >= len(self.nodes):
            return
        
        node = self.nodes[node_idx]
        plane = self.planes[node.planenum]
        nx, ny, nz = plane.normal
        
        sx, sy, sz = start
        ex, ey, ez = end
        
        # Compute distances of ray endpoints to splitting plane
        t1 = sx * nx + sy * ny + sz * nz - plane.dist
        t2 = ex * nx + ey * ny + ez * nz - plane.dist
        
        DIST_EPSILON = 0.03125
        
        if t1 >= -DIST_EPSILON and t2 >= -DIST_EPSILON:
            # Both on front (or within epsilon)
            self._trace_node(node.children[0], start_frac, end_frac,
                             start, end, result, mask)
            return
        
        if t1 < DIST_EPSILON and t2 < DIST_EPSILON:
            # Both on back (or within epsilon)
            self._trace_node(node.children[1], start_frac, end_frac,
                             start, end, result, mask)
            return
        
        # Ray spans the splitting plane — determine near/far children
        if t1 < 0:
            side = 1  # Start is behind → near side is back child
        else:
            side = 0  # Start is in front → near side is front child
        
        # Compute the fraction where the ray crosses the plane
        denom = t1 - t2
        if abs(denom) < 1e-10:
            frac = 0.5
        else:
            frac = t1 / denom
        frac = max(0.0, min(1.0, frac))
        
        mid_frac = start_frac + (end_frac - start_frac) * frac
        
        # Trace near side first (start_frac → mid_frac)
        self._trace_node(node.children[side], start_frac, mid_frac,
                         start, end, result, mask)
        
        # Then trace far side (mid_frac → end_frac)
        self._trace_node(node.children[1 - side], mid_frac, end_frac,
                         start, end, result, mask)

    def _trace_leaf_brushes(self, leaf: BSPLeaf, start: Vec3, end: Vec3,
                            result: TraceResult, mask: int) -> None:
        """Test a ray against all brushes in a leaf."""
        for bi in range(leaf.firstleafbrush,
                        leaf.firstleafbrush + leaf.numleafbrushes):
            if bi >= len(self.leafbrushes):
                break
            brush_idx = self.leafbrushes[bi]
            if brush_idx >= len(self.brushes):
                continue
            
            if self._brush_check_counts[brush_idx] == self._check_count:
                continue
            self._brush_check_counts[brush_idx] = self._check_count
            
            brush = self.brushes[brush_idx]
            if not (brush.contents & mask):
                continue
            if brush.numsides < 1:
                continue
            
            self._clip_ray_to_brush(brush, start, end, result)

    def _clip_ray_to_brush(self, brush: BSPBrush, start: Vec3, end: Vec3,
                           result: TraceResult) -> None:
        """Clip a ray against a convex brush, updating result if closer hit."""
        enter_frac = -1.0
        leave_frac = 1.0
        starts_out = False
        ends_out = False
        hit_plane = None
        
        sx, sy, sz = start
        ex, ey, ez = end
        
        DIST_EPSILON = 0.03125
        
        for si in range(brush.firstside, brush.firstside + brush.numsides):
            if si >= len(self.brushsides):
                return
            bs = self.brushsides[si]
            if bs.bevel:
                continue
            if bs.planenum >= len(self.planes):
                return
            
            plane = self.planes[bs.planenum]
            nx, ny, nz = plane.normal
            dist = plane.dist
            
            d1 = sx * nx + sy * ny + sz * nz - dist
            d2 = ex * nx + ey * ny + ez * nz - dist
            
            if d1 > 0:
                starts_out = True
            if d2 > 0:
                ends_out = True
            
            # Both in front of this plane — ray doesn't enter through here
            if d1 > 0 and d2 > 0:
                return
            
            # Both behind — this plane doesn't clip
            if d1 <= 0 and d2 <= 0:
                continue
            
            # Compute intersection fraction
            denom = d1 - d2
            if abs(denom) < 1e-10:
                continue
            
            f = d1 / denom
            
            if d1 > d2:
                # Entering the brush through this plane
                f = (d1 - DIST_EPSILON) / denom if abs(denom) > 1e-10 else 0.0
                if f > enter_frac:
                    enter_frac = f
                    hit_plane = plane
            else:
                # Leaving the brush
                f = (d1 + DIST_EPSILON) / denom if abs(denom) > 1e-10 else 1.0
                if f < leave_frac:
                    leave_frac = f
        
        if not starts_out:
            # Start point is inside the brush
            result.start_solid = True
            if not ends_out:
                result.all_solid = True
                result.fraction = 0.0
            return
        
        if enter_frac < leave_frac:
            if enter_frac > -1.0 and enter_frac < result.fraction:
                enter_frac = max(0.0, enter_frac)
                result.fraction = enter_frac
                if hit_plane:
                    result.plane_normal = hit_plane.normal
                    result.plane_dist = hit_plane.dist

    def _trace_prop_triangles(self, start: Vec3, end: Vec3,
                               result: TraceResult) -> TraceResult:
        """Test ray against static prop collision triangles."""
        from geometry import ray_triangle_intersect
        
        sx, sy, sz = start
        ex, ey, ez = end
        dx, dy, dz = ex - sx, ey - sy, ez - sz
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-6:
            return result
        
        direction = (dx / length, dy / length, dz / length)
        max_dist = length * result.fraction
        
        fx = sx + direction[0] * max_dist
        fy = sy + direction[1] * max_dist
        fz = sz + direction[2] * max_dist
        
        mins = (min(sx, fx), min(sy, fy), min(sz, fz))
        maxs = (max(sx, fx), max(sy, fy), max(sz, fz))
        
        cs = getattr(self, '_prop_cell_size', 512.0)
        cx0 = int(math.floor(mins[0] / cs))
        cy0 = int(math.floor(mins[1] / cs))
        cz0 = int(math.floor(mins[2] / cs))
        cx1 = int(math.floor(maxs[0] / cs))
        cy1 = int(math.floor(maxs[1] / cs))
        cz1 = int(math.floor(maxs[2] / cs))
        
        tested = set()
        for gx in range(cx0, cx1 + 1):
            for gy in range(cy0, cy1 + 1):
                for gz in range(cz0, cz1 + 1):
                    bucket = self._prop_grid.get((gx, gy, gz))
                    if bucket is None:
                        continue
                    for ti in bucket:
                        if ti in tested:
                            continue
                        tested.add(ti)
                        v0, v1, v2 = self._prop_triangles[ti]
                        t = ray_triangle_intersect(start, direction, v0, v1, v2, max_dist)
                        if t is not None and t < max_dist:
                            max_dist = t
                            frac = t / length
                            if frac < result.fraction:
                                result.fraction = frac
                                result.end_pos = (
                                    sx + frac * dx,
                                    sy + frac * dy,
                                    sz + frac * dz,
                                )
                                # Recompute bounds for remaining buckets (optional optimization)
        
        return result

    # ─── PVS queries ──────────────────────────────────────────────────────────

    def is_cluster_visible(self, cluster_a: int, cluster_b: int) -> bool:
        """Check if cluster_b is potentially visible from cluster_a.
        
        Uses the BSP's pre-compiled PVS (Potentially Visible Set) data.
        Returns True if no PVS data is available (conservative).
        """
        if self._num_clusters <= 0:
            return True  # No PVS data — assume visible
        if cluster_a < 0 or cluster_b < 0:
            return True  # Invalid clusters — assume visible
        if cluster_a >= self._num_clusters or cluster_b >= self._num_clusters:
            return True
        if cluster_a == cluster_b:
            return True  # Same cluster — always visible
        
        # Get PVS offset for cluster_a
        pvs_ofs = self._cluster_offsets[cluster_a][0]
        
        # Decompress run-length encoded PVS bitset
        # The PVS is stored as RLE bytes: 
        #   non-zero byte = 8 visibility bits
        #   zero byte followed by count = skip 'count' zero bytes
        vis_byte = cluster_b >> 3
        vis_bit = 1 << (cluster_b & 7)
        
        byte_idx = 0
        ofs = pvs_ofs
        data = self._vis_data
        data_len = len(data)
        
        while byte_idx <= vis_byte:
            if ofs >= data_len:
                return True  # Past end of data — assume visible
            
            b = data[ofs]
            ofs += 1
            
            if b == 0:
                # RLE zero run
                if ofs >= data_len:
                    return False
                count = data[ofs]
                ofs += 1
                byte_idx += count
                continue
            
            if byte_idx == vis_byte:
                return bool(b & vis_bit)
            
            byte_idx += 1
        
        return False

    # ─── Entity helpers ───────────────────────────────────────────────────────

    def get_spawn_points(self) -> List[Vec3]:
        """Extract player spawn positions from the entity lump.
        
        Returns origins for: info_player_start, info_player_teamspawn,
        info_ladder_dismount.
        """
        spawn_classes = {
            'info_player_start', 'info_player_teamspawn',
            'info_player_deathmatch', 'info_player_counterterrorist',
            'info_player_terrorist', 'info_player_combine',
            'info_player_rebel',
        }
        ladder_classes = {'info_ladder_dismount'}
        
        spawns = []
        for ent in self.entities:
            classname = ent.get('classname', '')
            if classname in spawn_classes or classname in ladder_classes:
                origin_str = ent.get('origin', '')
                if origin_str:
                    try:
                        parts = origin_str.split()
                        if len(parts) >= 3:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            spawns.append((x, y, z))
                    except ValueError:
                        pass
        
        return spawns

    def get_world_bounds(self) -> Tuple[Vec3, Vec3]:
        """Get the world bounds from BSP model 0."""
        if self.models:
            m = self.models[0]
            return m.mins, m.maxs
        return ((-16384, -16384, -16384), (16384, 16384, 16384))
