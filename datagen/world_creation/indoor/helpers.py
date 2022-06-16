from typing import List
from typing import Optional
from typing import Union

import godot_parser
import numpy as np
from godot_parser import GDObject
from godot_parser import Node as GDNode
from scipy.spatial.transform import Rotation

from common.errors import SwitchError
from datagen.godot_base_types import Vector3
from datagen.world_creation.geometry import Box
from datagen.world_creation.geometry import Position
from datagen.world_creation.godot_utils import make_transform
from datagen.world_creation.new_godot_scene import ImprovedGodotScene
from datagen.world_creation.types import GDLinkedSection


def visualize_tiles(tiles: np.ndarray) -> None:
    char_by_int = {i + 1: chr(ord("A") + i) for i in range(0, 52)}
    char_by_int[0] = "."
    for line in tiles:
        chars = [char_by_int[-x if x < 0 else x] for x in line]
        print("".join(chars))


def draw_line_in_grid(
    points: List[Position],
    grid: np.ndarray,
    tile_value: Union[int, float],
    include_ends=True,
    filter_values: Optional[Union[int, float]] = 0,
) -> None:
    for i, (from_point, to_point) in enumerate(zip(points[0:-1], points[1:])):
        first_connection = i == 0
        last_connection = i == len(points) - 2
        if from_point.x == to_point.x:
            # todo: improve naming and clarity
            # max() with 0 for negative coordinates (exterior points) to work
            start_z = max(min(from_point.z, to_point.z), 0)
            end_z = max(from_point.z, to_point.z) + 1
            if not include_ends:
                if first_connection and start_z != 0:
                    start_z += 1
                if last_connection:
                    end_z -= 1
            local_grid = grid[start_z:end_z, from_point.x]
        else:
            start_x = max(min(from_point.x, to_point.x), 0)
            end_x = max(from_point.x, to_point.x) + 1
            if not include_ends:
                if first_connection and start_x != 0:
                    start_x += 1
                if last_connection:
                    end_x -= 1
            local_grid = grid[from_point.z, start_x:end_x]
        if filter_values is not None:
            local_grid[local_grid == filter_values] = tile_value
        else:
            local_grid[True] = tile_value


def add_box_to_scene(
    box: Box,
    origin: Vector3,
    scene: ImprovedGodotScene,
    parent_node: GDNode,
    material: GDLinkedSection,
    scaling_factor: float = 1,
    include_mesh: bool = True,
    include_collision_shape: bool = True,
) -> None:
    # todo: clean this up
    assert include_mesh or include_collision_shape
    box_rotation = Rotation.from_euler("xyz", [box.pitch, box.yaw, box.roll], degrees=True)
    box_rotation = box_rotation.as_matrix().flatten()
    if include_mesh:
        cube_mesh = scene.add_sub_resource(
            "CubeMesh",
            size=godot_parser.Vector3(
                box.width * scaling_factor, box.height * scaling_factor, box.length * scaling_factor
            ),
        )
        ref_id = cube_mesh.reference.id
        mesh_instance = GDNode(
            f"Ramp{ref_id}",
            type="MeshInstance",
            properties={
                "transform": GDObject("Transform", 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
                "mesh": cube_mesh.reference,
                "material/0": material.reference,
            },
        )

    if include_collision_shape:
        box_shape = scene.add_sub_resource(
            "BoxShape",
            extents=godot_parser.Vector3(box.width / 2, box.height / 2, box.length / 2),
        )
        ref_id = box_shape.reference.id
        collision_shape = GDNode(
            f"CollisionShape{ref_id}",
            type="CollisionShape",
            properties={
                "transform": make_transform(),
                "shape": box_shape.reference,
            },
        )

    static_body = GDNode(
        f"StaticBody{ref_id}",
        type="StaticBody",
        properties={
            "transform": GDObject(
                "Transform",
                *box_rotation,
                origin.x + box.x * scaling_factor,
                origin.y + box.y * scaling_factor,
                origin.z + box.z * scaling_factor,
            )
        },
    )
    if include_mesh:
        static_body.add_child(mesh_instance)
    if include_collision_shape:
        static_body.add_child(collision_shape)
    parent_node.add_child(static_body)


def rotate_position(position: Position, story_width: int, story_length: int, degrees: int, tile_like=True):
    # tile_like=True behaves like np.rot90; tile_like=False rotates directly
    if degrees == 0:
        new_x = position.x
        new_z = position.z
    elif degrees == 90:
        new_x = story_length - position.z
        if tile_like:
            new_x -= 1
        new_z = position.x
    elif degrees == -90:
        new_x = position.z
        new_z = story_width - position.x
        if tile_like:
            new_z -= 1
    elif degrees == 180:
        new_x = story_width - position.x
        new_z = story_length - position.z
        if tile_like:
            new_x -= 1
            new_z -= 1
    else:
        raise SwitchError(degrees)
    return Position(new_x, new_z)
