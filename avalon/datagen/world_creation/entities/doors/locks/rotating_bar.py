from typing import List

import attr
import numpy as np
from godot_parser import Node as GDNode
from godot_parser import NodePath
from scipy.spatial.transform import Rotation

from avalon.common.utils import only
from avalon.datagen.world_creation.entities.doors.locks.door_lock import DoorLock
from avalon.datagen.world_creation.entities.doors.types import HingeSide
from avalon.datagen.world_creation.geometry import Axis
from avalon.datagen.world_creation.indoor.blocks import new_make_block_nodes
from avalon.datagen.world_creation.indoor.constants import DEFAULT_STORY_HEIGHT
from avalon.datagen.world_creation.indoor.utils import get_scaled_mesh_node
from avalon.datagen.world_creation.types import GodotScene
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.utils import make_transform


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class RotatingBar(DoorLock):
    size: np.ndarray = np.array([1, DEFAULT_STORY_HEIGHT, 0.1])
    rotation: np.ndarray = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    rotation_axis: Axis = Axis.Z
    anchor_side: HingeSide = HingeSide.RIGHT
    unlatch_angle: float = 10.0  # degrees

    def get_node(self, scene: GodotScene) -> GDNode:
        ROTATING_BAR = "rotating_bar"
        ROTATING_BAR_BODY = "bar_body"
        ROTATING_BAR_ANCHOR = "anchor"

        anchor_collision_mesh_node = GDNode(
            "collision_shape",
            type="CollisionShape",
            properties={
                "shape": scene.add_sub_resource("SphereShape", radius=0).reference,
                "disabled": True,
            },
        )
        anchor_node = GDNode(ROTATING_BAR_ANCHOR, "StaticBody")
        anchor_node.add_child(anchor_collision_mesh_node)

        joint_lower_angle = 0 if self.anchor_side == HingeSide.RIGHT else -180
        joint_upper_angle = -180 if self.anchor_side == HingeSide.RIGHT else 0
        hinge_joint_node = GDNode(
            "hinge_joint",
            "HingeJoint",
            properties={
                "nodes/node_a": NodePath(f"../{ROTATING_BAR_ANCHOR}"),
                "nodes/node_b": NodePath(f"../{ROTATING_BAR_BODY}"),
                "angular_limit/enable": True,
                "angular_limit/lower": joint_lower_angle,
                "angular_limit/upper": joint_upper_angle,
                "angular_limit/relaxation": 0.25,
            },
        )

        bar_offset = np.array([self.size[0] / 2, 0, 0])
        bar_body_child_nodes = new_make_block_nodes(scene, bar_offset, self.size, make_parent=False, make_mesh=False)
        bar_body_mesh_node = get_scaled_mesh_node(
            scene, "res://entities/doors/rotating_bar.tscn", self.size, bar_offset
        )
        bar_body_child_nodes.append(bar_body_mesh_node)

        body_script = scene.add_ext_resource("res://entities/doors/bar_body.gd", "Script")
        bar_body_rotation_degrees = -180 if self.anchor_side == HingeSide.RIGHT else 0
        bar_body_rotation = (
            Rotation.from_euler(Axis.Z.value, bar_body_rotation_degrees, degrees=True).as_matrix().flatten()
        )
        body_node = GDNode(
            ROTATING_BAR_BODY,
            type="RigidBody",
            properties={
                "transform": make_transform(rotation=bar_body_rotation),
                "mass": 10.0,
                "script": body_script.reference,
                "entity_id": 0,
            },
        )
        for child_node in bar_body_child_nodes:
            body_node.add_child(child_node)

        support_width, support_height, support_thickness = 0.025, 0.1, 0.29
        support_size = np.array([support_width, support_height, support_thickness])
        bar_support_x = -self.size[0] * 0.8
        if self.anchor_side == HingeSide.RIGHT:
            bar_support_x = -bar_support_x
        bar_support_position = np.array([bar_support_x, 0, -support_thickness / 2.5])
        bar_support_node = only(
            new_make_block_nodes(
                scene,
                bar_support_position,
                support_size,
                parent_name="bar_support",
                make_mesh=False,
                collision_shape_disabled=True,
            )
        )
        bar_support_mesh_node = get_scaled_mesh_node(scene, "res://entities/doors/bar_support.tscn", support_size)
        bar_support_node.add_child(bar_support_mesh_node)

        bar_script = scene.add_ext_resource("res://entities/doors/rotating_bar.gd", "Script")
        group_spatial_node = GDNode(
            f"{ROTATING_BAR}_{self.entity_id}",
            "Spatial",
            properties={
                "transform": make_transform(position=self.position, rotation=self.rotation),
                "script": bar_script.reference,
                "entity_id": self.entity_id,
                "rotation_axis": self.rotation_axis.value,
                "anchor_side": self.anchor_side.value.lower(),
                "unlatch_angle": self.unlatch_angle,
            },
        )
        group_spatial_node.add_child(body_node)
        group_spatial_node.add_child(anchor_node)
        group_spatial_node.add_child(hinge_joint_node)
        group_spatial_node.add_child(bar_support_node)
        return group_spatial_node

    def get_additional_door_body_nodes(
        self, scene: GodotScene, body_size: np.ndarray, body_centroid: Point3DNP = np.array([0, 0, 0])
    ) -> List[GDNode]:
        hook_width, hook_height, hook_thickness = 0.025, 0.1, 0.2
        hook_size = np.array([hook_width, hook_height, hook_thickness])
        bar_x, bar_y, bar_z = self.position
        body_width, body_height, body_thickness = body_size
        hook_span = body_width * 0.4
        hook_distance = hook_span / 5
        hook_x_start = 0 if self.anchor_side == HingeSide.RIGHT else -hook_span
        hook_x_positions = np.arange(hook_x_start, hook_x_start + hook_span + hook_distance, hook_distance)

        hook_nodes = []
        for i, hook_x in enumerate(hook_x_positions):
            bar_hook_offset = np.array([hook_x, 0, 0])
            bar_hook_child_nodes = new_make_block_nodes(
                scene,
                bar_hook_offset,
                hook_size,
                make_parent=False,
                make_mesh=False,
                collision_shape_name=f"collision_shape_{i}",
                collision_shape_disabled=True,
            )
            bar_hook_mesh_node = get_scaled_mesh_node(
                scene,
                "res://entities/doors/bar_support.tscn",
                hook_size,
                mesh_position=bar_hook_offset,
                mesh_name=f"mesh_{i}",
            )
            hook_nodes.extend(bar_hook_child_nodes)
            hook_nodes.append(bar_hook_mesh_node)

        bar_hook_position = body_centroid + np.array([0, bar_y * 0.98, hook_thickness * 0.6])
        hook_spatial_node = GDNode(
            f"bar_hooks",
            "StaticBody",
            properties={"transform": make_transform(position=bar_hook_position)},
        )
        for hook_node in hook_nodes:
            hook_spatial_node.add_child(hook_node)
        return [hook_spatial_node]
