from typing import List
from typing import Tuple

import attr
import numpy as np
from godot_parser import Node as GDNode
from godot_parser import NodePath
from scipy.spatial.transform import Rotation

from avalon.common.utils import only
from avalon.datagen.world_creation.entities.doors.locks.door_lock import DoorLock
from avalon.datagen.world_creation.entities.doors.types import HingeSide
from avalon.datagen.world_creation.entities.doors.types import MountSlot
from avalon.datagen.world_creation.geometry import Axis
from avalon.datagen.world_creation.indoor.blocks import new_make_block_nodes
from avalon.datagen.world_creation.indoor.constants import DEFAULT_STORY_HEIGHT
from avalon.datagen.world_creation.indoor.utils import get_scaled_mesh_node
from avalon.datagen.world_creation.types import GodotScene
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.utils import make_transform


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class SlidingBar(DoorLock):
    size: np.ndarray = np.array([1, DEFAULT_STORY_HEIGHT, 0.1])
    rotation: np.ndarray = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    sliding_axis: Axis = Axis.Y
    door_face_axis: Axis = Axis.Z
    mount_slot: MountSlot = MountSlot.BOTTOM
    mount_side: HingeSide = HingeSide.LEFT
    proportion_to_unlock: float = 0.25

    @property
    def latch_size(self) -> Tuple[float, float, float]:
        bar_width, bar_height, bar_thickness = self.size
        latch_width, latch_height, latch_thickness = bar_width * 1.6, bar_width / 2, bar_thickness * 1.05
        return latch_width, latch_height, latch_thickness

    def get_node(self, scene: GodotScene) -> GDNode:
        SLIDING_BAR = "sliding_bar"
        SLIDING_BAR_BODY = "bar_body"
        SLIDING_BAR_ANCHOR = "anchor"

        anchor_collision_mesh_node = GDNode(
            "collision_shape",
            type="CollisionShape",
            properties={
                "shape": scene.add_sub_resource("SphereShape", radius=0).reference,
                "disabled": True,
            },
        )
        anchor_node = GDNode(SLIDING_BAR_ANCHOR, "StaticBody")
        anchor_node.add_child(anchor_collision_mesh_node)

        bar_width, bar_height, bar_thickness = self.size
        leeway_multiplier = 1.2
        if self.mount_slot == MountSlot.BOTTOM:
            lower_distance = 0
            upper_distance = bar_height * self.proportion_to_unlock * leeway_multiplier
        else:
            lower_distance = -bar_height * self.proportion_to_unlock * leeway_multiplier
            upper_distance = 0

        slider_joint_node = GDNode(
            "slider_joint",
            "SliderJoint",
            properties={
                "transform": make_transform(rotation=Rotation.from_euler("z", 90, degrees=True).as_matrix().flatten()),
                "nodes/node_a": NodePath(f"../{SLIDING_BAR_ANCHOR}"),
                "nodes/node_b": NodePath(f"../{SLIDING_BAR_BODY}"),
                "linear_limit/lower_distance": lower_distance,
                "linear_limit/upper_distance": upper_distance,
                "linear_limit/softness": 0.5,
            },
        )

        x_multiplier = -1 if self.mount_side == HingeSide.LEFT else 1

        latch_width, _latch_height, _latch_thickness = self.latch_size
        # Main body of the bar
        bar_body_offset = np.array([-x_multiplier * latch_width / 2, 0, bar_thickness / 2])
        bar_body_child_nodes = new_make_block_nodes(
            scene,
            bar_body_offset,
            self.size,
            make_parent=False,
            make_mesh=False,
            collision_shape_name="body_collision_shape",
        )
        bar_body_mesh_node = get_scaled_mesh_node(
            scene, "res://entities/doors/sliding_bar.tscn", self.size, bar_body_offset
        )
        bar_body_child_nodes.append(bar_body_mesh_node)

        # Knob
        y_multiplier = 1 if self.mount_slot == MountSlot.BOTTOM else -1
        knob_thickness = 0.1
        knob_size = np.array([0.1, 0.1, knob_thickness])
        knob_offset = bar_body_offset + np.array(
            [0, y_multiplier * bar_height / 2.2, bar_thickness / 2 + knob_thickness / 2]
        )
        knob_child_nodes = new_make_block_nodes(
            scene,
            knob_offset,
            knob_size,
            make_parent=False,
            make_mesh=False,
            collision_shape_name="knob_collision_shape",
        )
        knob_mesh_node = get_scaled_mesh_node(
            scene, "res://entities/doors/bar_knob.tscn", knob_size, knob_offset, mesh_name="knob_mesh"
        )
        knob_child_nodes.append(knob_mesh_node)
        bar_body_child_nodes.extend(knob_child_nodes)

        # Guiderail on the side
        rail_height = bar_height * (1 + self.proportion_to_unlock)
        rail_width = bar_width * 1.5
        rail_thickness = bar_width * 1.5
        # The rail is normally horizontal, so we define its size as if it was going to be placed horizontal
        # and then rotate it
        rail_size = np.array([rail_height, rail_width, rail_thickness])
        rail_position = np.array(
            [
                x_multiplier * rail_width / 2,
                0.5 * self.proportion_to_unlock * bar_height,
                bar_thickness / 1.8,
            ]
        )

        rail_mesh_node = get_scaled_mesh_node(scene, "res://entities/doors/rail.tscn", rail_size)
        rotation_degrees = 90
        rail_node = only(
            new_make_block_nodes(
                scene,
                rail_position,
                rail_size,
                Rotation.from_euler("z", rotation_degrees, degrees=True).as_matrix().flatten(),
                parent_name="rail",
                make_mesh=False,
                collision_shape_disabled=True,
            )
        )
        rail_node.add_child(rail_mesh_node)

        # Guiderail "hinges" (the wheels that slide with the bar up and down)
        hinge_span = (bar_height * (1 - self.proportion_to_unlock)) * 0.7  # 70% of the part of bar above the slot
        hinge_width = bar_width * 2.1
        hinge_height = bar_width
        hinge_thickness = 0.05
        offset_from_bottom = (self.proportion_to_unlock * bar_height) / 2
        # The hinge is normally vertical, so we define its size as if it was going to be placed vertically, then rotate
        hinge_size = np.array([hinge_height, hinge_width, hinge_thickness])
        for i, y_offset in enumerate([-hinge_span / 2 + offset_from_bottom, hinge_span / 2 + offset_from_bottom]):
            hinge_position = np.array([x_multiplier * rail_width / 2, y_offset, bar_thickness / 1.8])
            y_rotation = 0 if self.mount_side == HingeSide.LEFT else 180
            rail_hinge_mesh_node = get_scaled_mesh_node(
                scene, "res://entities/doors/rail_hinge.tscn", hinge_size, mesh_name="hinge_mesh"
            )
            rail_hinge_node = only(
                new_make_block_nodes(
                    scene,
                    hinge_position,
                    hinge_size,
                    Rotation.from_euler("zy", (90, y_rotation), degrees=True).as_matrix().flatten(),
                    make_mesh=False,
                    parent_name=f"rail_hinge_{i}",
                    collision_shape_disabled=True,
                )
            )
            rail_hinge_node.add_child(rail_hinge_mesh_node)
            bar_body_child_nodes.append(rail_hinge_node)

        body_script = scene.add_ext_resource("res://entities/doors/vertical_bar_body.gd", "Script")
        body_node = GDNode(
            SLIDING_BAR_BODY,
            type="RigidBody",
            properties={
                "mass": 10.0,
                "script": body_script.reference,
                "entity_id": 0,
            },
        )
        for child_node in bar_body_child_nodes:
            body_node.add_child(child_node)

        group_spatial_node = GDNode(
            f"{SLIDING_BAR}_{self.entity_id}",
            "Spatial",
            properties={
                "transform": make_transform(position=self.position, rotation=self.rotation),
                "script": scene.add_ext_resource("res://entities/doors/sliding_bar.gd", "Script").reference,
                "sliding_axis": self.sliding_axis.value,
                "proportion_to_unlock": self.proportion_to_unlock,
                "entity_id": self.entity_id,
            },
        )
        group_spatial_node.add_child(body_node)
        group_spatial_node.add_child(anchor_node)
        group_spatial_node.add_child(slider_joint_node)
        group_spatial_node.add_child(rail_node)
        return group_spatial_node

    def get_additional_door_body_nodes(
        self, scene: GodotScene, body_size: np.ndarray, body_centroid: Point3DNP = np.array([0, 0, 0])
    ) -> List[GDNode]:
        # For the sliding bar these are the two big holes on the doors that the bar falls into
        # latch_mesh_base_size = np.array([0.15, 0.09, 0.181])
        bar_width, bar_height, bar_thickness = self.size
        door_body_width, door_body_height, door_body_thickness = body_size
        # latch_width, latch_height, latch_thickness = 0.08, 0.17, 0.29
        latch_width, latch_height, latch_thickness = self.latch_size
        latch_size = np.array([latch_width, latch_height, latch_thickness])
        x_multiplier = -1 if self.mount_side == HingeSide.LEFT else 1
        y_multiplier = -1 if self.mount_slot == MountSlot.BOTTOM else 1
        bar_y_position = self.position[1]

        bar_latch_positions = [
            np.array(
                [
                    body_centroid[0] + x_multiplier * (door_body_width / 2 - latch_width / 2),
                    bar_y_position + y_multiplier * bar_height * self.proportion_to_unlock,
                    door_body_thickness / 2 + bar_thickness / 2,
                ]
            ),
        ]
        latch_nodes = []

        for i, bar_latch_position in enumerate(bar_latch_positions):
            bar_latch_node = only(
                new_make_block_nodes(
                    scene,
                    bar_latch_position,
                    latch_size,
                    make_mesh=False,
                    parent_name=f"bar_latch_{self.entity_id}_{i}",
                    collision_shape_disabled=True,
                )
            )
            bar_latch_mesh_node = get_scaled_mesh_node(scene, "res://entities/doors/bar_latch.tscn", latch_size)
            bar_latch_node.add_child(bar_latch_mesh_node)
            latch_nodes.append(bar_latch_node)
        return latch_nodes
