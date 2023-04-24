from typing import Any, Dict, Sequence

import numpy as np
import open3d as o3d
import pyrender
import trimesh
from scipy.spatial.ckdtree import cKDTree
from tqdm import tqdm

from ..datasets import DatasetBase


class Renderer:
    def __init__(self, height: int = 480, width: int = 640) -> None:
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()

    def __call__(
        self,
        height: int,
        width: int,
        intrinsics: np.ndarray,
        pose: np.ndarray,
        mesh: pyrender.Mesh,
    ) -> Any:
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(
            cx=intrinsics[0, 2],
            cy=intrinsics[1, 2],
            fx=intrinsics[0, 0],
            fy=intrinsics[1, 1],
        )
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)

    def fix_pose(self, pose: np.ndarray) -> np.ndarray:
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh: trimesh.Trimesh) -> pyrender.Mesh:
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self) -> None:
        self.renderer.delete()


def refuse(mesh: trimesh.Trimesh, dataset: DatasetBase, scale: float) -> o3d.geometry.TriangleMesh:
    renderer = Renderer()
    mesh_opengl = renderer.mesh_opengl(mesh)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        # volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=0.04 * scale,
        sdf_trunc=3 * 0.04 * scale,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    img_ids = range(dataset.n_images)
    height = dataset.get_image_size(0)[0]
    width = dataset.get_image_size(0)[1]
    intri = np.eye(4)
    intri[0, 0] = dataset.intrins.fx[0].item()
    intri[1, 1] = dataset.intrins.fy[0].item()
    intri[0, 2] = dataset.intrins.cx[0].item()
    intri[1, 2] = dataset.intrins.cy[0].item()
    for img_id in tqdm(img_ids, total=len(img_ids), desc="Refusing: "):
        pose = dataset.c2w[img_id].cpu().numpy()
        rgb = dataset.gt[img_id].view(height, width, 3).numpy()
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        _, depth_pred = renderer(height, width, intri, pose, mesh_opengl)
        depth_pred = o3d.geometry.Image(depth_pred)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth_pred,
            depth_scale=1.0,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False,
        )
        fx, fy, cx, cy = (
            intri[0, 0],
            intri[1, 1],
            intri[0, 2],
            intri[1, 2],
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy
        )
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)

    return volume.extract_triangle_mesh()


def nn_correspondance(verts1: np.ndarray, verts2: np.ndarray) -> np.ndarray:
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = cKDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances


def transform(
    mesh: o3d.geometry.TriangleMesh, scale: float, offset: Sequence[float]
) -> o3d.geometry.TriangleMesh:
    v = np.asarray(mesh.vertices)
    v /= scale
    v += offset
    mesh.vertices = o3d.utility.Vector3dVector(v)
    return mesh


def transform_normalize(
    mesh: o3d.geometry.TriangleMesh, scale: float, offset: Sequence[float]
) -> o3d.geometry.TriangleMesh:
    v = np.asarray(mesh.vertices)
    v -= offset
    v *= scale
    mesh.vertices = o3d.utility.Vector3dVector(v)
    return mesh


def evaluate_mesh(
    mesh_pred: o3d.geometry.TriangleMesh,
    mesh_trgt: o3d.geometry.TriangleMesh,
    threshold: float = 0.05,
    down_sample: float = 0.02,
) -> Dict:
    pcd_pred = o3d.geometry.PointCloud(mesh_pred.vertices)
    pcd_trgt = o3d.geometry.PointCloud(mesh_trgt.vertices)

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1 = nn_correspondance(verts_pred, verts_trgt)
    dist2 = nn_correspondance(verts_trgt, verts_pred)

    precision = np.mean((dist2 < threshold).astype(np.float32))
    recal = np.mean((dist1 < threshold).astype(np.float32))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {
        "Acc": np.mean(dist2),
        "Comp": np.mean(dist1),
        "Prec": precision,
        "Recal": recal,
        "F-score": fscore,
    }
    return metrics


def remove_isolate_component_by_diameter(
    o3d_mesh,
    diameter_percentage: float = 0.05,
    keep_mesh: bool = False,
    remove_unreferenced_vertices: bool = True,
):
    import copy

    assert diameter_percentage >= 0.0
    assert diameter_percentage <= 1.0
    max_bb = o3d_mesh.get_max_bound()
    min_bb = o3d_mesh.get_min_bound()
    size_bb = np.abs(max_bb - min_bb)
    filter_diag = diameter_percentage * np.linalg.norm(size_bb)

    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)

    triangle_clusters, cluster_n_triangles, _ = o3d_mesh.cluster_connected_triangles()
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    #     triangle_clusters, cluster_n_triangles, _ = (
    #         o3d_mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters) + 1
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    largest_cluster_idx = cluster_n_triangles.argmax() + 1
    for idx in range(1, len(cluster_n_triangles) + 1):  # set label 0 to keep
        if idx == largest_cluster_idx:  # must keep the largest
            triangle_clusters[triangle_clusters == idx] = 0
        else:
            cluster_triangle = triangle_clusters == idx
            cluster_index = np.unique(faces[cluster_triangle])
            cluster_vertices = vertices[cluster_index]
            cluster_bbox = np.abs(
                np.amax(cluster_vertices, axis=0) - np.amin(cluster_vertices, axis=0)
            )
            cluster_bbox_diag = np.linalg.norm(cluster_bbox, ord=2)
            if cluster_bbox_diag >= filter_diag:
                triangle_clusters[triangle_clusters == idx] = 0
    mesh_temp = copy.deepcopy(o3d_mesh) if keep_mesh else o3d_mesh
    mesh_temp.remove_triangles_by_mask(triangle_clusters > 0.5)

    if remove_unreferenced_vertices:
        mesh_temp.remove_unreferenced_vertices()

    print("!!! finish clean the isolated component !!!")
    return mesh_temp


def o3dmesh_to_trimesh(o3d_mesh):
    vertices = o3d_mesh.vertices
    triangles = o3d_mesh.triangles

    mesh = trimesh.Trimesh()
    mesh.vertices = np.asarray(vertices)
    mesh.faces = np.asarray(triangles)

    return mesh


def trimesh_to_o3dmesh(
    tri_mesh: trimesh.Trimesh(),
):
    return tri_mesh.as_open3d()
