from typing import List

import attr
import numpy as np
from godot_parser import Node as GDNode
from godot_parser import NodePath

from avalon.common.utils import only
from avalon.datagen.world_creation.entities.doors.door import Door
from avalon.datagen.world_creation.entities.doors.locks.rotating_bar import RotatingBar
from avalon.datagen.world_creation.entities.doors.locks.sliding_bar import SlidingBar
from avalon.datagen.world_creation.entities.doors.types import LatchingMechanics
from avalon.datagen.world_creation.indoor.blocks import new_make_block_nodes
from avalon.datagen.world_creation.indoor.utils import get_scaled_mesh_node
from avalon.datagen.world_creation.types import GodotScene
from avalon.datagen.world_creation.utils import make_transform


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class SlidingDoor(Door):
    slide_right: bool = True
    handle_size_proportion: float = 0.2
    latching_mechanics: LatchingMechanics = LatchingMechanics.NO_LATCH

    def get_node(self, scene: GodotScene) -> GDNode:
        DOOR_BODY = "body"

        body_nodes = self._get_body_nodes(scene, DOOR_BODY)
        rail_nodes = self._get_rail_nodes(scene, DOOR_BODY)

        door_script = scene.add_ext_resource("res://entities/doors/sliding_door.gd", "Script")
        group_spatial_node = GDNode(
            f"sliding_door_{self.entity_id}",
            "Spatial",
            properties={
                "transform": make_transform(self.rotation, self.position),
                "script": door_script.reference,
                "entity_id": self.entity_id,
                "slide_axis": self.face_axis.value,
                "is_latched": self.latching_mechanics != LatchingMechanics.NO_LATCH,
            },
        )
        for body_node in body_nodes:
            group_spatial_node.add_child(body_node)
        for rail_node in rail_nodes:
            group_spatial_node.add_child(rail_node)

        body_node = only(body_nodes)
        for i, lock in enumerate(self.locks):
            # if we have two locks of the same kind, their instance names will otherwise clash
            lock_with_id = attr.evolve(lock, entity_id=i)
            group_spatial_node.add_child(lock_with_id.get_node(scene))
            for body_addition in lock_with_id.get_additional_door_body_nodes(scene, self.size):
                body_node.add_child(body_addition)
        return group_spatial_node

    def _get_body_nodes(self, scene: GodotScene, door_body_node_name: str) -> List[GDNode]:
        body_children_nodes = new_make_block_nodes(
            scene,
            np.array([0, 0, 0]),
            self.size,
            make_parent=False,
            make_mesh=False,
            collision_shape_name=f"{door_body_node_name}_collision_shape",
        )
        door_body_mesh = get_scaled_mesh_node(
            scene, "res://entities/doors/door_body.tscn", self.size, mesh_name="body_mesh"
        )
        body_children_nodes.append(door_body_mesh)
        handle_nodes = self._get_handle_nodes(scene)

        door_width, door_height, door_thickness = self.size
        rail_hinge_distance = door_width * 0.5
        rail_hinge_width, rail_hinge_height, rail_hinge_thickness = 0.12, 0.32, 0.05
        rail_hinge_size = np.array([rail_hinge_width, rail_hinge_height, rail_hinge_thickness])
        rail_hinge_offset = (door_thickness - rail_hinge_thickness) / 2

        left_hinge_mesh_node = get_scaled_mesh_node(
            scene,
            "res://entities/doors/rail_hinge.tscn",
            rail_hinge_size,
            mesh_position=np.array([-rail_hinge_distance / 2, door_height / 2, rail_hinge_offset]),
            mesh_name="left_hinge_mesh",
        )
        body_children_nodes.append(left_hinge_mesh_node)

        right_hinge_mesh_node = get_scaled_mesh_node(
            scene,
            "res://entities/doors/rail_hinge.tscn",
            rail_hinge_size,
            mesh_position=np.array([rail_hinge_distance / 2, door_height / 2, rail_hinge_offset]),
            mesh_name="right_hinge_mesh",
        )
        body_children_nodes.append(right_hinge_mesh_node)

        body_script = scene.add_ext_resource("res://entities/doors/door_body.gd", "Script")
        body_node = GDNode(
            door_body_node_name,
            type="RigidBody",
            properties={
                "transform": make_transform(),
                "mass": 100.0,
                "script": body_script.reference,
                "entity_id": 0,
                "is_auto_latching": self.latching_mechanics == LatchingMechanics.AUTO_LATCH,
            },
        )
        for node in body_children_nodes:
            body_node.add_child(node)
        for node in handle_nodes:
            body_node.add_child(node)
        return [body_node]

    def _get_handle_nodes(self, scene: GodotScene) -> List[GDNode]:
        DOOR_HANDLE = "handle"

        door_width, door_height, door_thickness = self.size
        handle_width = door_width * (self.handle_size_proportion / 2)
        handle_height = door_height * self.handle_size_proportion
        handle_thickness = door_thickness * 0.75

        # We cap the handle height if there are locks that would get obstructed by it
        leeway = 0.05
        bar_locks = [lock for lock in self.locks if isinstance(lock, (RotatingBar, SlidingBar))]
        if len(bar_locks) > 0:
            max_y = np.inf
            min_y = -np.inf
            for lock in bar_locks:
                if not isinstance(lock, (RotatingBar, SlidingBar)):
                    continue
                if lock.position[1] > 0 and (lock_bottom := lock.position[1] - lock.size[1] / 2) < max_y:
                    max_y = lock_bottom
                elif lock.position[1] < 0 and (lock_top := lock.position[1] + lock.size[1] / 2) > min_y:
                    min_y = lock_top
            max_handle_height = min(abs(max_y), abs(min_y)) * 2 - leeway
            if handle_height > max_handle_height:
                handle_height = max_handle_height
        handle_size = np.array([handle_width, handle_height, handle_thickness])
        handle_margin_skew = 25  # left margin / right margin
        handle_side_margin = (door_width - handle_width) / (1 + handle_margin_skew)
        if not self.slide_right:
            handle_side_margin *= handle_margin_skew

        handle_nodes = []
        handle_position = np.array(
            [
                -door_width / 2 + handle_side_margin + handle_width / 2,
                0,
                (handle_thickness / 2 + door_thickness / 2),
            ]
        )
        handle_name = f"{DOOR_HANDLE}_0"
        handle_nodes.extend(
            new_make_block_nodes(
                scene,
                handle_position,
                handle_size,
                make_parent=False,
                make_mesh=False,
                collision_shape_name=f"{handle_name}_collision_shape",
            )
        )
        handle_mesh_node = get_scaled_mesh_node(
            scene,
            "res://entities/doors/door_handle_vertical.tscn",
            handle_size,
            handle_position,
            mesh_name=f"{handle_name}_mesh",
        )
        handle_nodes.append(handle_mesh_node)
        return handle_nodes

    def _get_rail_nodes(self, scene: GodotScene, door_body_node_name: str) -> List[GDNode]:
        DOOR_RAIL = "rail"

        door_width, door_height, door_thickness = self.size
        rail_height = door_height * 0.05
        rail_size = np.array([door_width * 2, rail_height, door_thickness * 1.1])
        direction_multiplier = 1 if self.slide_right else -1
        rail_position = np.array([direction_multiplier * door_width / 2, door_height / 2 + rail_height / 2, 0])

        rail_mesh_node = get_scaled_mesh_node(scene, "res://entities/doors/rail.tscn", rail_size)
        rail_node = only(
            new_make_block_nodes(
                scene,
                rail_position,
                rail_size,
                parent_name=DOOR_RAIL,
                make_mesh=False,
                collision_shape_disabled=True,
            )
        )
        rail_node.add_child(rail_mesh_node)

        slider_joint_position = direction_multiplier * door_width, door_height / 2, 0
        slider_joint_node = GDNode(
            "slider_joint",
            "SliderJoint",
            properties={
                "transform": make_transform(position=slider_joint_position),
                "nodes/node_a": NodePath(f"../{DOOR_RAIL}"),
                "nodes/node_b": NodePath(f"../{door_body_node_name}"),
                "linear_limit/lower_distance": 0 if self.slide_right else -door_width,
                "linear_limit/upper_distance": door_width if self.slide_right else 0,
                "linear_limit/softness": 0.5,
            },
        )
        return [rail_node, slider_joint_node]
