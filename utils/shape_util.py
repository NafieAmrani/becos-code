import numpy as np
import open3d as o3d
import potpourri3d as pp3d
import trimesh
from scipy.spatial.transform import Rotation as R

from scipy.sparse.csgraph import shortest_path
import scipy
import scipy.spatial
import scipy.sparse.linalg as sla

import robust_laplacian


def laplacian_decomposition(verts, faces, k=150):
    """
    Args:
        verts (np.ndarray): vertices [V, 3].
        faces (np.ndarray): faces [F, 3]
        k (int, optional): number of eigenvalues/vectors to compute. Default 120.

    Returns:
        - evals: (k) list of eigenvalues of the Laplacian matrix.
        - evecs: (V, k) list of eigenvectors of the Laplacian.
        - evecs_trans: (k, V) list of pseudo inverse of eigenvectors of the Laplacian.
    """
    assert k >= 0, f"Number of eigenvalues/vectors should be non-negative, bug get {k}"
    is_cloud = faces is None
    eps = 1e-8

    # Build Laplacian matrix
    if is_cloud:
        L, M = robust_laplacian.point_cloud_laplacian(verts)
        massvec = M.diagonal()
    else:
        L = pp3d.cotan_laplacian(verts, faces, denom_eps=1e-10)
        massvec = pp3d.vertex_areas(verts, faces)
        massvec += eps * np.mean(massvec)

    if np.isnan(L.data).any():
        raise RuntimeError("NaN Laplace matrix")
    if np.isnan(massvec).any():
        raise RuntimeError("NaN mass matrix")

    # Compute the eigenbasis
    # Prepare matrices
    L_eigsh = (L + eps * scipy.sparse.identity(L.shape[0])).tocsc()
    massvec_eigsh = massvec
    Mmat = scipy.sparse.diags(massvec_eigsh)
    eigs_sigma = eps

    fail_cnt = 0
    while True:
        try:
            evals, evecs = sla.eigsh(L_eigsh, k=k, M=Mmat, sigma=eigs_sigma)
            # Clip off any eigenvalues that end up slightly negative due to numerical error
            evals = np.clip(evals, a_min=0.0, a_max=float("inf"))
            evals = evals.reshape(-1, 1)
            break
        except Exception as e:
            if fail_cnt > 3:
                raise ValueError("Failed to compute eigen-decomposition")
            fail_cnt += 1
            print("Decomposition failed; adding eps")
            L_eigsh = L_eigsh + (eps * 10**fail_cnt) * scipy.sparse.identity(L.shape[0])

    evecs = np.array(evecs, ndmin=2)
    evecs_trans = evecs.T @ Mmat

    sqrt_area = np.sqrt(Mmat.diagonal().sum())
    return evals, evecs, evecs_trans, sqrt_area


def rigid_transform_3d(A, B):
    assert A.shape == B.shape

    # Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Center the points
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute the covariance matrix
    H = AA.T @ BB

    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Compute translation vector
    t = centroid_B - R @ centroid_A

    return R, t


def align_shapes(vert_x, face_x, vert_y, face_y, p2p_corres):
    # center shapes
    vert_x = vert_x - np.mean(vert_x, axis=0)
    vert_y = vert_y - np.mean(vert_y, axis=0)

    area_x = get_surface_area(vert_x, face_x)
    area_y = get_surface_area(vert_y, face_y)

    R, t = rigid_transform_3d(vert_x, vert_y[p2p_corres])
    vert_y = vert_y @ R + t

    return vert_x, vert_y, area_x, area_y, R, t

def get_rotation_matrix(rot):
    r = R.from_euler("xyz", rot, degrees=True)
    return r.as_matrix()


def plot_shapes(
    vert_x,
    face_x,
    color_x,
    vert_y,
    face_y,
    color_y,
    vert_x_partial=None,
    face_x_partial=None,
    color_x_partial=None,
    vert_y_partial=None,
    face_y_partial=None,
    color_y_partial=None,
    store_output=False,
    show_output=False,
    show_full_shapes=False,
    file_name=None,
    translation=(1, 0, 0),
    rot_x=None,
    area_x=None,
    area_y=None,
    normals=True,
    cam_pos_x=None,
    cam_pos_y=None
):
    if area_x is not None and area_y is not None:
        vert_x *= area_x
        vert_y *= area_y
        vert_x_partial = (
            area_x * vert_x_partial if vert_x_partial is not None else vert_x_partial
        )
        vert_y_partial = (
            area_y * vert_y_partial if vert_y_partial is not None else vert_y_partial
        )

    mesh_x = get_o3d_mesh(vert_x, face_x)
    mesh_y = get_o3d_mesh(vert_y, face_y)
    mesh_y.translate(translation)
    if normals:
        mesh_x.compute_vertex_normals()
        mesh_y.compute_vertex_normals()

    partial_x = all(
        var is not None for var in (vert_x_partial, face_x_partial, color_x_partial)
    )
    partial_y = all(
        var is not None for var in (vert_y_partial, face_y_partial, color_y_partial)
    )

    if show_full_shapes:
        partial_x = False
        partial_y = False

    if partial_x:
        mesh_x_partial = get_o3d_mesh(vert_x_partial, face_x_partial)
        mesh_x_partial.vertex_colors = o3d.utility.Vector3dVector(color_x_partial)
        if normals:
            mesh_x_partial.compute_vertex_normals()
    else:
        if color_x is not None:
            mesh_x.vertex_colors = o3d.utility.Vector3dVector(color_x)

    if partial_y:
        mesh_y_partial = get_o3d_mesh(vert_y_partial, face_y_partial)
        mesh_y_partial.vertex_colors = o3d.utility.Vector3dVector(color_y_partial)
        mesh_y_partial.translate(translation)
        if normals:
            mesh_y_partial.compute_vertex_normals()
    else:
        if color_y is not None:
            mesh_y.vertex_colors = o3d.utility.Vector3dVector(color_y)
    
    if partial_x and partial_y:
        # scale mesh_x with scale factor 0.9
        mesh_x.scale(0.9, center=mesh_x.get_center())
        mesh_y.scale(0.9, center=mesh_y.get_center())

    if store_output:
        # rotate meshes so that meshes face the camera - does not work on all datasets
        if rot_x is not None:
            rot_matrix = get_rotation_matrix(rot_x)
            mesh_x.rotate(rot_matrix)
            mesh_y.rotate(rot_matrix)
            if partial_x:
                mesh_x_partial.rotate(rot_matrix)
            if partial_y:
                mesh_y_partial.rotate(rot_matrix)

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        opt = vis.get_render_option()
        if partial_x:
            vis.add_geometry(mesh_x_partial)
        else:
            vis.add_geometry(mesh_x)

        if partial_y:
            vis.add_geometry(mesh_y_partial)

        if partial_x or partial_y:
            opt.mesh_show_back_face = True
            # opt.mesh_show_wireframe = True
        else:
            vis.add_geometry(mesh_x)
            vis.add_geometry(mesh_y)

        vis.capture_screen_image(file_name, do_render=True)

    if show_output:
        if partial_x and partial_y:
            geometries = [mesh_x_partial, mesh_y_partial]
            o3d.visualization.draw_geometries(geometries)

        elif partial_y:
            o3d.visualization.draw_geometries(
                [mesh_x, mesh_y_partial],
                mesh_show_wireframe=False,
                mesh_show_back_face=True,
            )
        else:
            o3d.visualization.draw_geometries(
                [mesh_x, mesh_y], mesh_show_wireframe=False, mesh_show_back_face=True
            )


def get_surface_area(verts, faces):
    massvec = pp3d.vertex_areas(verts, faces)
    sqrt_area = np.sqrt(massvec.sum())
    return sqrt_area


def get_o3d_mesh(verts, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh


def get_center(pnts):
    shape_center = [
        (np.max(pnts[:, 0]) + np.min(pnts[:, 0])) / 2,
        (np.max(pnts[:, 1]) + np.min(pnts[:, 1])) / 2,
        (np.max(pnts[:, 2]) + np.min(pnts[:, 2])) / 2,
    ]
    shape_scale = np.max(
        [
            np.max(pnts[:, 0]) - np.min(pnts[:, 0]),
            np.max(pnts[:, 1]) - np.min(pnts[:, 1]),
            np.max(pnts[:, 2]) - np.min(pnts[:, 2]),
        ]
    )
    return shape_center, shape_scale


def num_vertices(trimesh_mesh):
    return len(trimesh_mesh.vertices)


def area(trimesh_mesh):
    return trimesh_mesh.area


def read_o3d_mesh(fielname):
    mesh = o3d.io.read_triangle_mesh(fielname)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    return mesh


def write_off(file, verts, faces):
    with open(file, "w") as f:
        f.write("OFF\n")
        f.write(f"{verts.shape[0]} {faces.shape[0]} {0}\n")
        for x in verts:
            f.write(f"{' '.join(map(str, x))}\n")
        for x in faces:
            f.write(f"{len(x)} {' '.join(map(str, x))}\n")


def create_colormap(contour):
    minx = contour[:, 0].min()
    miny = contour[:, 1].min()
    minz = contour[:, 2].min()
    maxx = contour[:, 0].max()
    maxy = contour[:, 1].max()
    maxz = contour[:, 2].max()
    r = (contour[:, 0] - minx) / (maxx - minx)
    g = (contour[:, 1] - miny) / (maxy - miny)
    b = (contour[:, 2] - minz) / (maxz - minz)
    colors = np.stack((r, g, b), axis=-1)
    assert colors.shape == contour.shape
    return colors


def read_file(filename, verbose=False):
    #try:
    if verbose:
        print(f"Reading {filename}")
    # read mesh without changing
    mesh = trimesh.load(filename, process=False, maintain_order=True)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
        # save mesh to file
    #except ValueError:
        #print(f"Error reading {filename}")
        #verts = np.zeros((0, 3))
        #faces = np.zeros((0, 3))
    return verts, faces


def sample_random_rotation(one_axis=False):
    # Generate random rotation angles around x, y, and z axes
    if one_axis:
        theta_x = 0
        theta_z = 0
    else:
        theta_x = np.random.uniform(0, 2 * np.pi)
        theta_z = np.random.uniform(0, 2 * np.pi)
    theta_y = np.random.uniform(0, 2 * np.pi)

    # Create rotation matrices around x, y, and z axes
    rotation_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)],
        ]
    )

    rotation_y = np.array(
        [
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)],
        ]
    )

    rotation_z = np.array(
        [
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1],
        ]
    )

    # Combine rotation matrices
    rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))

    return rotation_matrix
