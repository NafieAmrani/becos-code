import os
import json

import trimesh
import numpy as np


def anime_read(filename):
    """
    filename: path of .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: vertice data of the 1st frame (3D positions in x-y-z-order)
        face_data: riangle face data of the 1st frame
        offset_data: 3D offset data from the 2nd to the last frame
    """
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    '''check data consistency'''
    if len(offset_data) != (nf - 1) * nv * 3:
        raise ("data inconsistent error!", filename)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data



def num_vertices(mesh):
    return len(mesh.vertices)


def extract_meshes(file_name, vert_data, face_data, offset_data, frame_number):
    if frame_number == 0:
        mesh = trimesh.Trimesh(vertices=vert_data, faces=face_data, process=False, maintain_order=True)
    else:
        mesh = trimesh.Trimesh(vertices=vert_data + offset_data[frame_number - 1], faces=face_data, process=False,
                               maintain_order=True)

    assert (vert_data + offset_data[frame_number - 1]).shape[0] == mesh.vertices.shape[0], \
        f"Error with {file_name} - original - {(vert_data + offset_data[frame_number - 1]).shape[0]} - trimesh {mesh.vertices.shape[0]} "

    connected_mesh = mesh.split(only_watertight=False)
    connected_mesh = sorted(connected_mesh, key=num_vertices, reverse=True)[0]
    connected_mesh.export(file_name)

def extract_required_frames(type_shapes):
    info_dict = f"data/info_dt4d_{type_shapes}.json"
    original_data = f"data/DeformingThings4D/{type_shapes}"

    with open(info_dict, "r") as file:
        info_dict = json.load(file)

    output_path = f"data/dt4d_{type_shapes}"
    os.makedirs(output_path, exist_ok=True)

    print("Extracting frames from animation files of DT4D ... (this takes a few minutes)")
    for category in info_dict:
        category_path = f"{output_path}/{category}"
        os.makedirs(category_path, exist_ok=True)
        for action in info_dict[category]:
            animation_name = f"{category}_{action}"
            animation_file = f"{original_data}/{animation_name}/{animation_name}.anime"
            nf, _, _, vert_data, face_data, offset_data = anime_read(animation_file)

            for frame_id in info_dict[category][action]:
                mesh_frame_file = f"{category_path}/{action}{frame_id:03d}.obj"
                extract_meshes(mesh_frame_file, vert_data, face_data, offset_data, frame_id)
    print("Done!")

if __name__ == "__main__":
    types = ["animals", "humanoids"]
    for type_shape in types:
        extract_required_frames(type_shape)