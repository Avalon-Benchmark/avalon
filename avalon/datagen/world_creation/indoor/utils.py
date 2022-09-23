from typing import Any

import numpy as np
import scipy.ndimage
from godot_parser import Node as GDNode

from avalon.datagen.world_creation.constants import IDENTITY_BASIS
from avalon.datagen.world_creation.indoor.wall_footprint import WallFootprint
from avalon.datagen.world_creation.types import Basis3DNP
from avalon.datagen.world_creation.types import GodotScene
from avalon.datagen.world_creation.types import Point3DListNP
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.utils import make_transform
from avalon.datagen.world_creation.utils import scale_basis


def get_evenly_spaced_centroids(
    wall_footprint: WallFootprint,
    thing_width: float,
    min_gap: float,
    centroid_y: float,
    max_things: int = np.iinfo(np.int32).max,
) -> Point3DListNP:
    # like CSS flexbox space-evenly. Horizontal only for now. Depth-wise placed in middle of wall.
    thing_count, leftover_space = divmod(wall_footprint.wall_length, thing_width + min_gap)
    if thing_count > max_things:
        thing_count = max_things
        leftover_space = wall_footprint.wall_length - thing_count * (thing_width + min_gap)

    centroids = []
    wall_origin_x, wall_origin_z = wall_footprint.top_left
    for i in range(int(thing_count)):
        extra_gap_per_thing = leftover_space / thing_count
        x = (
            wall_origin_x + (i + 0.5) * (thing_width + min_gap + extra_gap_per_thing)
            if not wall_footprint.is_vertical
            else wall_origin_x + wall_footprint.wall_thickness / 2
        )
        z = (
            wall_origin_z + (i + 0.5) * (thing_width + min_gap + extra_gap_per_thing)
            if wall_footprint.is_vertical
            else wall_origin_z + wall_footprint.wall_thickness / 2
        )
        centroids.append((x, centroid_y, z))
    return np.array(centroids)


MESH_BASE_SIZES = {
    "res://entities/doors/open_button.tscn": np.array([0.2, 0.2, 0.2]),
    "res://entities/doors/rotating_bar.tscn": np.array([1.2, 0.2, 0.2]),
    "res://entities/doors/bar_support.tscn": np.array([0.087, 0.18, 0.3]),
    "res://entities/doors/sliding_bar.tscn": np.array([0.1, 0.6, 0.1]),
    "res://entities/doors/bar_knob.tscn": np.array([0.1, 0.6, 0.1]),
    "res://entities/doors/rail.tscn": np.array([2, 0.135, 0.1]),
    "res://entities/doors/rail_hinge.tscn": np.array([0.12, 0.32, 0.05]),
    "res://entities/doors/bar_latch.tscn": np.array([0.15, 0.09, 0.181]),
    "res://entities/doors/door_body.tscn": np.array([1, 2.7, 0.1]),
    "res://entities/doors/door_handle_vertical.tscn": np.array([0.05, 0.75, 0.075]),
    "res://entities/doors/door_handle_loop.tscn": np.array([0.14, 0.25, 0.05]),
}


def get_scaled_mesh_node(
    scene: GodotScene,
    mesh_resource_path: str,
    mesh_size: Point3DNP,
    mesh_position: Point3DNP = np.array([0, 0, 0]),
    mesh_rotation: Basis3DNP = IDENTITY_BASIS,
    mesh_name: str = "mesh",
) -> GDNode:
    mesh_base_size = MESH_BASE_SIZES[mesh_resource_path]
    mesh_resource = scene.add_ext_resource(mesh_resource_path, "PackedScene")
    scale_factors = mesh_size / mesh_base_size
    return GDNode(
        mesh_name,
        instance=mesh_resource.reference.id,
        properties={
            "transform": make_transform(position=mesh_position, rotation=scale_basis(mesh_rotation, scale_factors))
        },
    )


def rand_integer(rand: np.random.Generator, min_ge: int, max_lt: int) -> int:
    if min_ge >= max_lt:
        return min_ge
    else:
        return rand.integers(min_ge, max_lt)


def inset_borders(
    array: np.ndarray, min_leftover_size: int = 0, void_value: Any = 1, border_value: Any = 0
) -> np.ndarray:
    """
    Moves the external borders of all "voids" in a numpy array one index inward, unless it reduces the void size
    below `min_leftover_size`.
    Examples:
        111       000
        111   ->  010
        111       000

        00000     00000
        01110     00000
        01110     00100
        01110 ->  00000
        01000     00000
        00000     00000

    with min_leftover_size = 2
        111     111
        111  -> 111
        111     111
    """
    assert void_value != border_value
    voids = list(zip(*np.where(array == void_value)))
    if min_leftover_size > 0:
        if len(voids) == 0:
            return array.copy()
        void_width = max([x for z, x in voids]) - min([x for z, x in voids]) + 1
        void_length = max([z for z, x in voids]) - min([z for z, x in voids]) + 1
        if void_length < min_leftover_size + 2 or void_width < min_leftover_size + 2:
            return array.copy()
    inset_array = array.copy()
    convolution = scipy.ndimage.convolve(array.astype(np.int8), np.full((3, 3), void_value), mode="constant")
    inset_array[convolution != 9] = border_value
    return inset_array
