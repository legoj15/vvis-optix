import sys
import struct

def get_lumps(path):
    with open(path, 'rb') as f:
        data = f.read(1036)
        ident, version = struct.unpack('<II', data[:8])
        if ident != 0x50534256:
            raise Exception("Not a BSP")
        lumps = []
        for i in range(64):
            offset = 8 + i * 16
            fileofs, filelen, vers, uncompressed = struct.unpack('<IIII', data[offset:offset+16])
            lumps.append(filelen)
    return lumps

cpu = get_lumps(sys.argv[1])
nxt = get_lumps(sys.argv[2])

print(f"{'Lump ID':<8} | {'CPU Length':<15} | {'NXT Length':<15}")
print("-" * 45)
for i in range(64):
    if cpu[i] != nxt[i]:
        print(f"{i:<8} | {cpu[i]:<15} | {nxt[i]:<15}")
