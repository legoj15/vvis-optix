"""Diagnostic: check if the new side-ID lumps are present in a BSP."""
import struct, sys
from pathlib import Path

def check_bsp_side_id_lumps(bsp_path: str):
    data = Path(bsp_path).read_bytes()
    
    # Parse header
    ident, version = struct.unpack_from('<II', data, 0)
    print(f"BSP ident=0x{ident:08X}  version={version}")
    
    # Lump 22 = LUMP_FACE_SIDEIDS_INDEX
    # Lump 23 = LUMP_FACE_SIDEIDS_DATA
    for lump_id, name in [(11, "LUMP_FACEIDS"), (22, "LUMP_FACE_SIDEIDS_INDEX"), (23, "LUMP_FACE_SIDEIDS_DATA")]:
        ofs = 8 + lump_id * 16
        fileofs, filelen, ver, uncomp = struct.unpack_from('<iiii', data, ofs)
        print(f"\n  Lump {lump_id:2d} ({name}):")
        print(f"    offset={fileofs}  length={filelen}  version={ver}")
        
        if filelen == 0:
            print(f"    ** EMPTY — lump not present **")
            continue
        
        if lump_id == 11:
            # LUMP_FACEIDS: array of uint16
            count = filelen // 2
            print(f"    {count} face IDs")
            # Show first 20
            for i in range(min(count, 20)):
                hid = struct.unpack_from('<H', data, fileofs + i*2)[0]
                print(f"      face[{i}] hammer_id = {hid}")
        
        elif lump_id == 22:
            # Index: array of {int32 firstId, int32 numIds}
            count = filelen // 8
            print(f"    {count} index entries")
            
            # Get lump 23 data for resolving
            ofs23 = 8 + 23 * 16
            fileofs23, filelen23 = struct.unpack_from('<ii', data, ofs23)
            
            # Show entries that have numIds > 1 (merged faces)
            merged_count = 0
            for i in range(count):
                first_id, num_ids = struct.unpack_from('<ii', data, fileofs + i*8)
                if num_ids > 1:
                    merged_count += 1
                    ids = []
                    for j in range(num_ids):
                        sid = struct.unpack_from('<i', data, fileofs23 + (first_id + j)*4)[0]
                        ids.append(sid)
                    print(f"      face[{i}]: {num_ids} IDs = {ids}")
            
            if merged_count == 0:
                print(f"    ** No faces with multiple IDs found — merging not captured **")
                # Show first 10 for debugging
                for i in range(min(count, 10)):
                    first_id, num_ids = struct.unpack_from('<ii', data, fileofs + i*8)
                    ids = []
                    for j in range(num_ids):
                        sid = struct.unpack_from('<i', data, fileofs23 + (first_id + j)*4)[0]
                        ids.append(sid)
                    print(f"      face[{i}]: {num_ids} IDs = {ids}")
            else:
                print(f"    Total faces with merged IDs: {merged_count}")
        
        elif lump_id == 23:
            count = filelen // 4
            print(f"    {count} total side IDs in data array")
    
    # Search for our target IDs
    print(f"\n--- Searching for target VMF side IDs: 1134, 1136, 1142, 1150, 1156 ---")
    target_ids = {1134, 1136, 1142, 1150, 1156}
    
    # Check LUMP_FACEIDS
    ofs11 = 8 + 11 * 16
    fileofs11, filelen11 = struct.unpack_from('<ii', data, ofs11)
    faceids_count = filelen11 // 2
    for i in range(faceids_count):
        hid = struct.unpack_from('<H', data, fileofs11 + i*2)[0]
        if hid in target_ids:
            print(f"  LUMP_FACEIDS: face[{i}] has hammer_id={hid}")
    
    # Check LUMP_FACE_SIDEIDS_DATA
    ofs23 = 8 + 23 * 16
    fileofs23, filelen23 = struct.unpack_from('<ii', data, ofs23)
    if filelen23 > 0:
        sid_count = filelen23 // 4
        for i in range(sid_count):
            sid = struct.unpack_from('<i', data, fileofs23 + i*4)[0]
            if sid in target_ids:
                print(f"  LUMP_FACE_SIDEIDS_DATA: data[{i}] = {sid}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Try to find BSP in common location
        candidates = list(Path('.').glob('**/*.bsp'))
        if not candidates:
            print("Usage: python check_sideids.py <path_to.bsp>")
            sys.exit(1)
        bsp = str(candidates[0])
    else:
        bsp = sys.argv[1]
    
    print(f"Checking: {bsp}\n")
    check_bsp_side_id_lumps(bsp)
