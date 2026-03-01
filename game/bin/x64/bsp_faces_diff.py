import sys
import struct

def read_faces(path):
    with open(path, 'rb') as f:
        data = f.read(1036)
        fileofs, filelen, vers, uncomp = struct.unpack('<IIII', data[8+7*16:8+7*16+16])
        f.seek(fileofs)
        face_data = f.read(filelen)
    
    faces = []
    # dface_t is 56 bytes in Source 2013 (check struct size)
    # unsigned short planenum; 2
    # byte side; 1
    # byte onNode; 1
    # int firstedge; 4
    # short numedges; 2
    # short texinfo; 2
    # short dispinfo; 2
    # short surfaceFogVolumeID; 2
    # byte styles[4]; 4
    # int lightofs; 4
    # float area; 4
    # int m_LightmapTextureMinsInLuxels[2]; 8
    # int m_LightmapTextureSizeInLuxels[2]; 8
    # int origFace; 4
    # unsigned short m_NumPrims; 2
    # unsigned short firstPrimID; 2
    # unsigned int smoothingGroups; 4
    # Total: 2+1+1+4+2+2+2+2+4+4+4+8+8+4+2+2+4 = 56 bytes
    face_size = 56
    num_faces = filelen // face_size
    for i in range(num_faces):
        chunk = face_data[i*face_size:(i+1)*face_size]
        faces.append(chunk)
    return faces

cpu = read_faces(sys.argv[1])
nxt = read_faces(sys.argv[2])

print(f"Num faces CPU: {len(cpu)}, NXT: {len(nxt)}")
diff_count = 0
for i in range(len(cpu)):
    if cpu[i] != nxt[i]:
        diff_count += 1
        if diff_count <= 5:
            print(f"Face {i} differs:")
            print(f"CPU: {cpu[i].hex()}")
            print(f"NXT: {nxt[i].hex()}")

print(f"Total different faces: {diff_count}")
