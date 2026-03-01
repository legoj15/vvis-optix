import sys
import struct
import io

def read_lump(path, lump_id):
    with open(path, 'rb') as f:
        data = f.read(1036)
        offset = 8 + lump_id * 16
        fileofs, filelen, vers, uncomp = struct.unpack('<IIII', data[offset:offset+16])
        f.seek(fileofs)
        return f.read(filelen)

def replace_lumps(src_bsp, target_bsp, out_bsp, lump_ids):
    with open(src_bsp, 'rb') as f:
        src_data = f.read()
    with open(target_bsp, 'rb') as f:
        target_data = bytearray(f.read())
        
    for lump_id in lump_ids:
        # read from src
        offset_src = 8 + lump_id * 16
        fileofs_s, filelen_s, vers_s, uncomp_s = struct.unpack('<IIII', src_data[offset_src:offset_src+16])
        
        # We can't just easily resize lumps in the middle of the BSP in python without rewriting the whole directory structure.
        # But wait! If we just want to test, we can APPEND the new lumps at the EOF and update the directory pointer.
        lump_content = src_data[fileofs_s:fileofs_s+filelen_s]
        
        new_fileofs = len(target_data)
        target_data += lump_content
        
        # update target header
        offset_t = 8 + lump_id * 16
        struct.pack_into('<IIII', target_data, offset_t, new_fileofs, len(lump_content), vers_s, uncomp_s)

    with open(out_bsp, 'wb') as f:
        f.write(target_data)

cpu_bsp = "visual_comparison_validation_vrad_gridlines/validation_vrad_gridlines_cpu.bsp"
nxt_bsp = "visual_comparison_validation_vrad_gridlines/validation_vrad_gridlines_nextgen.bsp"
out_bsp = "visual_comparison_validation_vrad_gridlines/validation_vrad_gridlines_nextgen_injected.bsp"

# 15 = WORLDLIGHTS, 30 = VERTNORMALS, 56 = LEAF_AMBIENT_LIGHTING
replace_lumps(cpu_bsp, nxt_bsp, out_bsp, [15, 30, 56])
print("Injected lumps into", out_bsp)
