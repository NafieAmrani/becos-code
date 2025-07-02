import igl
from scipy.spatial.distance import cdist

from utils.shape_util import *

class RayCaster:
    def __init__(self, width=5000, height=5000):
        self.width = width
        self.height = height
        self.up = [0,1,0]
        self.fov_deg = 90

    def get_rays(self, center, eye):
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=self.fov_deg,
            center=center,
            eye=eye,
            up=self.up,
            width_px=self.width,
            height_px=self.height,
        )
        return rays

    def ray_casting(self, mesh, camera_location):
        pnts = mesh.vertex.positions.numpy()
        diag_extent = igl.bounding_box_diagonal(pnts)
        mesh_updated = False
        if diag_extent >= 2:
            pnts = 0.8 * pnts
            mesh_v2 = o3d.geometry.TriangleMesh()
            mesh_v2.vertices = o3d.utility.Vector3dVector(pnts)
            mesh_v2.triangles = o3d.utility.Vector3iVector(mesh.triangle.indices.numpy())
            mesh_v2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh_v2)
            mesh_updated = True
        center = np.array([0.0, 0.0, 0.0])
        scene = o3d.t.geometry.RaycastingScene()
        if mesh_updated:
            scene.add_triangles(mesh_v2)
        else:
            scene.add_triangles(mesh)
        rays = self.get_rays(center=center, eye=camera_location)
        ans = scene.cast_rays(rays)
        return ans, mesh_updated

    def get_mesh(self, ans, mesh, triangle_ids=None):
        if triangle_ids is None:
            hit = ans["t_hit"].isfinite()
            triangle_ids = np.unique(ans['primitive_ids'][hit].numpy())
        # if triangle ids exist
        vis_tri = mesh.triangle.indices[triangle_ids, :]
        vertex = mesh.vertex.positions
        # extract biggest component
        trimesh_mesh = trimesh.Trimesh(vertices=vertex.numpy(), faces=vis_tri.numpy(), process=False)
        biggest_connected_components = trimesh_mesh.split(only_watertight=False)
        biggest_connected_component = sorted(biggest_connected_components, key=area, reverse=True)[0]
        return np.array(biggest_connected_component.vertices), np.array(biggest_connected_component.faces), triangle_ids

    def get_corres_vert_in_full_shape(self, verts, partial_verts):
        distances = cdist(partial_verts, verts)
        indices = np.argmin(distances, axis=1)
        return indices

    def get_partial_shape(self, verts, faces, camera_location, triangle_ids=None):
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False, maintain_order=True).as_open3d
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        if triangle_ids is None:
            ans, _ = self.ray_casting(mesh, camera_location=camera_location)
        else:
            ans = None
        partial_verts, partial_faces, triangle_ids = self.get_mesh(ans, mesh, triangle_ids=triangle_ids)
        partial2full = self.get_corres_vert_in_full_shape(verts, partial_verts)

        return partial_verts, partial_faces, partial2full, triangle_ids


def random_color():
    import random
    return [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]

def get_o3d_mesh(verts, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh

def add_cam_pos(cam_pos, translation=(0, 0, 0)):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    sphere.translate(cam_pos)
    sphere.translate(translation)
    if translation == (0, 0, 0):
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
    else:
        sphere.paint_uniform_color([0.0, 0.0, 1.0])

    points = [np.array([0.0, 0.0, 0.0]) + translation, cam_pos + translation]
    lines = [[0, 1]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    colors = [[0.0, 1.0, 0.0]]  # Green color
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return sphere, line_set