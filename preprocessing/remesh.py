import os
import igl
import trimesh
import numpy as np
import fast_simplification
from scipy.spatial.distance import cdist


def remesh(mesh_name, verts, faces, face_count):
    v1 = np.copy(verts)
    f1 = np.copy(faces)

    v1 *= 1000

    if mesh_name == 'kid19':
        v1, f1 = handle_difficult_mesh(v1, f1)

    f1, _ = igl.bfs_orient(f1)  # correct meshes with flipped faces
    s, f2 = igl.loop_subdivision_matrix(v1.shape[0], f1)
    v2 = s * v1

    v2_v1 = np.array(s.argmax(axis=1)).flatten()
    v1_v2 = np.array(s.argmax(axis=0)).flatten()

    v3, f3, collapses = fast_simplification.simplify(v2, f2, target_count=face_count,
                                                     return_collapses=True)
    _, _, v2_v3 = fast_simplification.replay_simplification(v2.astype(np.float32), f2, collapses)

    unique_values = np.unique(v2_v3)

    # Initialize a dictionary to store indices for each unique value
    indices_dict = {}

    # Loop through each unique value and find indices
    for value in unique_values:
        indices = np.where(v2_v3 == value)[0]
        indices_dict[value] = indices

    v3_v2 = np.empty(v3.shape[0], dtype=int)

    for key, value in indices_dict.items():
        dist = cdist(np.array([v3[key]]), v2[value])
        v3_v2[key] = value[np.argmin(dist, axis=1)[0]]

    source_to_remeshed_p2p = v2_v1[v3_v2]
    remeshed_to_source_p2p = v2_v3[v1_v2]

    return (
        source_to_remeshed_p2p,
        remeshed_to_source_p2p,
        v3/1000,
        f3,
    )


def handle_difficult_mesh(v1, f1):
    mesh = trimesh.Trimesh(
        vertices=v1, faces=f1, process=False, maintain_order=True
    )
    vertex_adjacency = mesh.vertex_adjacency_graph
    necessary_verts = np.arange(v1.shape[0])
    unconnected_verts = necessary_verts[
        ~np.isin(necessary_verts, vertex_adjacency.nodes)
    ]
    assert unconnected_verts.shape[0] == 3
    additional_face = np.array([[unconnected_verts[0], unconnected_verts[1], unconnected_verts[2]]])
    f1 = np.concatenate((f1, additional_face), axis=0)
    return v1, f1


def remeshing_wrapper(config, shape_name, vert=None, face=None, face_count=None,
                      remeshed_shape_file=None, file_exists=False):
    if file_exists:
        shape_data = np.load(remeshed_shape_file)
        return (
            shape_data["remeshed_vert"],
            shape_data["remeshed_face"],
            {"corres": shape_data["p2p_source_to_remeshed"], "same_tr": False, "barycentric": False},
            {"corres": shape_data["p2p_remeshed_to_source"], "same_tr": False, "barycentric": False},
        )

    (
        p2p_source_to_remeshed,
        p2p_remeshed_to_source,
        remeshed_vert,
        remeshed_face,
    ) = remesh(shape_name, vert, face, face_count)

    if config["update_precomputed_remeshed"] and config["original_settings"]:
        np.savez(remeshed_shape_file, remeshed_vert=remeshed_vert,
                 remeshed_face=remeshed_face,
                 p2p_source_to_remeshed=p2p_source_to_remeshed,
                 p2p_remeshed_to_source=p2p_remeshed_to_source,
                 face_count=face_count)

    return (
        remeshed_vert,
        remeshed_face,
        {"corres": p2p_source_to_remeshed, "same_tr": False, "barycentric": False},
        {"corres": p2p_remeshed_to_source, "same_tr": False, "barycentric": False},
    )


def adjust_path_remeshing(
    shape_x,
    shape_y,
    path,
    corres_path,
    bary_array_source_to_remeshed_x,
    bary_array_remeshed_to_source_y,
):
    corres_path = [bary_array_source_to_remeshed_x] + corres_path
    shape_x_remeshed = shape_x.copy()
    shape_x_remeshed["name"] = shape_x["name"] + "_remeshed"
    path = [shape_x_remeshed] + path
    corres_path = corres_path + [bary_array_remeshed_to_source_y]
    shape_y_remeshed = shape_y.copy()
    shape_y_remeshed["name"] = shape_y["name"] + "_remeshed"
    path = path + [shape_y_remeshed]
    return path, corres_path
