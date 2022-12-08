from typing import List
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation
from trimesh import Trimesh
from trimesh.visual import ColorVisuals

from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.types import RGBATuple


def homogeneous_transform_matrix(
    position: Point3DNP = np.array([0, 0, 0]), rotation: Optional[Rotation] = None
) -> np.ndarray:
    if rotation is None:
        rotation = np.eye(3)
    else:
        rotation = rotation.as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def make_color_visuals(mesh: Trimesh, rgba: RGBATuple) -> ColorVisuals:
    return ColorVisuals(mesh, face_colors=np.repeat([rgba], len(mesh.faces), axis=0))


class MeshData:
    def __init__(self) -> None:
        self.vertices: List[List[float]] = []
        self.faces: List[List[float]] = []
        self.face_normals: List[List[float]] = []
        self.face_colors: List[float] = []
        self.index_offset = 0


def unnormalize(mesh: Trimesh) -> Trimesh:
    """Re-create mesh such that none of its faces share vertices"""
    points_per_face = 3
    coords_per_point = 3
    face_count = len(mesh.faces)
    new_vertices = np.empty((face_count * points_per_face, coords_per_point), dtype=np.float32)
    for i, face in enumerate(mesh.faces):
        offset = i * points_per_face
        new_vertices[offset : offset + points_per_face, :] = mesh.vertices[face]
    new_faces = np.array(range(len(new_vertices))).reshape(-1, coords_per_point)
    return Trimesh(vertices=new_vertices, faces=new_faces, process=False)
