import math
from typing import List

import attr
import numpy as np
from godot_parser import Color
from godot_parser import Node as GDNode
from godot_parser import NodePath
from scipy.spatial.transform import Rotation

from avalon.common.utils import only
from avalon.datagen.world_creation.entities.doors.door import Door
from avalon.datagen.world_creation.entities.doors.locks.rotating_bar import RotatingBar
from avalon.datagen.world_creation.entities.doors.locks.sliding_bar import SlidingBar
from avalon.datagen.world_creation.entities.doors.types import HingeSide
from avalon.datagen.world_creation.entities.doors.types import LatchingMechanics
from avalon.datagen.world_creation.indoor.blocks import new_make_block_nodes
from avalon.datagen.world_creation.indoor.utils import get_scaled_mesh_node
from avalon.datagen.world_creation.types import GodotScene
from avalon.datagen.world_creation.utils import make_transform


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class HingeDoor(Door):
    hinge_side: HingeSide = HingeSide.LEFT
    hinge_radius: float = 0.05
    max_inwards_angle: float = 90.0
    max_outwards_angle: float = 90.0
    handle_size_proportion: float = 0.2
    latching_mechanics: LatchingMechanics = LatchingMechanics.NO_LATCH

    def get_node(self, scene: GodotScene) -> GDNode:
        DOOR_BODY = "body"

        body_nodes = self._get_body_nodes(scene, DOOR_BODY)
        hinge_nodes = self._get_hinge_nodes(scene, DOOR_BODY)

        door_script = scene.add_ext_resource("res://entities/doors/hinge_door.gd", "Script")
        group_spatial_node = GDNode(
            f"hinge_door_{self.entity_id}",
            "Spatial",
            properties={
                "transform": make_transform(self.rotation, self.position),
                "script": door_script.reference,
                "entity_id": self.entity_id,
                "is_latched": self.latching_mechanics != LatchingMechanics.NO_LATCH,
            },
        )
        for body_node in body_nodes:
            group_spatial_node.add_child(body_node)
        for hinge_node in hinge_nodes:
            group_spatial_node.add_child(hinge_node)
        body_node = only(body_nodes)
        door_body_width = self.size[0]
        door_body_size = np.array([door_body_width, self.size[1], self.size[2]])
        body_x_offset = -door_body_width / 2 if self.hinge_side == HingeSide.RIGHT else door_body_width / 2
        for i, lock in enumerate(self.locks):
            # if we have two locks of the same kind, their instance names will otherwise crash
            lock_with_id = attr.evolve(lock, entity_id=i)
            group_spatial_node.add_child(lock_with_id.get_node(scene))
            offset_centroid = np.array([body_x_offset, 0, 0])
            for body_addition in lock_with_id.get_additional_door_body_nodes(scene, door_body_size, offset_centroid):
                body_node.add_child(body_addition)
        return group_spatial_node

    def _get_body_nodes(self, scene: GodotScene, door_body_node_name: str) -> List[GDNode]:
        # The door body here also includes the handle, since PinJoints suck in Godot, so we can't have them separate :/
        door_width, door_height, door_thickness = self.size

        if self.hinge_side == HingeSide.LEFT:
            body_offset = np.array([door_width / 2, 0, 0])
            body_position = np.array([-door_width / 2 + self.hinge_radius, 0, 0])
        else:
            body_offset = np.array([-door_width / 2, 0, 0])
            body_position = np.array([door_width / 2 - self.hinge_radius, 0, 0])

        body_children_nodes = new_make_block_nodes(
            scene,
            body_offset,
            self.size,
            make_parent=False,
            make_mesh=False,
            collision_shape_name=f"{door_body_node_name}_collision_shape",
        )
        body_mesh_node = get_scaled_mesh_node(
            scene, "res://entities/doors/door_body.tscn", self.size, body_offset, mesh_name="body_mesh"
        )
        body_children_nodes.append(body_mesh_node)
        handle_nodes = self._get_handle_nodes(scene)

        body_script = scene.add_ext_resource("res://entities/doors/door_body.gd", "Script")
        body_node = GDNode(
            door_body_node_name,
            type="RigidBody",
            properties={
                "transform": make_transform(position=body_position),
                "mass": 100.0,
                "script": body_script.reference,
                "is_auto_latching": self.latching_mechanics == LatchingMechanics.AUTO_LATCH,
                "entity_id": 0,
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
        handle_width = door_width * self.handle_size_proportion
        handle_height = door_height * (self.handle_size_proportion / 2)
        handle_thickness = 0.15

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

        # We cap the handle width to ensure it doesn't prevent the door from opening inwards
        max_width = math.sqrt(
            (door_width - leeway) ** 2 - handle_thickness**2
        )  # diagonal of handle can't exceed door width
        handle_width = min(max_width, handle_width)
        handle_size = np.array([handle_width, handle_height, handle_thickness])

        handle_margin_skew = 25  # how many right-margins fit in the left margin
        handle_side_margin = (door_width - handle_width) / (1 + handle_margin_skew) * handle_margin_skew
        if self.hinge_side == HingeSide.LEFT:
            handle_x = handle_side_margin + handle_width / 2
        else:
            handle_x = -(handle_side_margin + handle_width / 2)

        handle_offsets = 1, -1
        handle_nodes = []
        for i, handle_offset in enumerate(handle_offsets):
            handle_position = np.array(
                [
                    handle_x,
                    0,
                    handle_offset * (handle_thickness / 2 + door_thickness / 2),
                ]
            )
            handle_name = f"{DOOR_HANDLE}_{i}"
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
                "res://entities/doors/door_handle_loop.tscn",
                handle_size,
                handle_position,
                mesh_name=f"{handle_name}_mesh",
            )
            handle_nodes.append(handle_mesh_node)
        return handle_nodes

    def _get_hinge_nodes(self, scene: GodotScene, door_body_node_name: str) -> List[GDNode]:
        DOOR_HINGE = "hinge"

        door_width, door_height, door_thickness = self.size
        frame_width = door_width + 2 * self.hinge_radius
        hinge_height = door_height
        body_left_side = -frame_width / 2 + 2 * self.hinge_radius
        body_right_side = frame_width / 2 - 2 * self.hinge_radius

        if self.hinge_side == HingeSide.LEFT:
            hinge_x = -frame_width / 2 + self.hinge_radius
        else:
            hinge_x = frame_width / 2 - self.hinge_radius
        hinge_position = hinge_x, 0, 0
        hinge_material = scene.add_sub_resource(
            "SpatialMaterial", albedo_color=Color(0.223529, 0.196078, 0.141176, 1), roughness=0.2, metallic=0.5
        )
        hinge_mesh_node = GDNode(
            "mesh",
            "MeshInstance",
            properties={
                "transform": make_transform(),
                "mesh": scene.add_sub_resource(
                    "CylinderMesh",
                    material=hinge_material.reference,
                    top_radius=self.hinge_radius,
                    bottom_radius=self.hinge_radius,
                    height=hinge_height,
                ).reference,
            },
        )
        hinge_collision_mesh_node = GDNode(
            "collision_shape",
            "CollisionShape",
            properties={
                "shape": scene.add_sub_resource(
                    "CylinderShape", radius=self.hinge_radius, height=hinge_height
                ).reference,
                "disabled": True,
            },
        )
        hinge_node = GDNode(
            DOOR_HINGE,
            "StaticBody",
            properties={
                "transform": make_transform(position=hinge_position),
            },
        )
        hinge_node.add_child(hinge_mesh_node)
        hinge_node.add_child(hinge_collision_mesh_node)

        hinge_rotation_degrees = 90 if self.hinge_side == HingeSide.LEFT else -90
        hinge_joint_position = body_left_side if self.hinge_side == HingeSide.LEFT else body_right_side, 0, 0
        hinge_rotation = Rotation.from_euler("x", hinge_rotation_degrees, degrees=True).as_matrix().flatten()
        hinge_joint_node = GDNode(
            "hinge_joint",
            "HingeJoint",
            properties={
                "transform": make_transform(position=hinge_joint_position, rotation=hinge_rotation),
                "nodes/node_a": NodePath(f"../{DOOR_HINGE}"),
                "nodes/node_b": NodePath(f"../{door_body_node_name}"),
                "params/bias": 0.99,
                "angular_limit/enable": True,
                "angular_limit/lower": -self.max_outwards_angle,
                "angular_limit/upper": self.max_inwards_angle,
                "angular_limit/bias": 0.99,
                "angular_limit/softness": 0.01,
                "angular_limit/relaxation": 0.01,
            },
        )
        return [hinge_node, hinge_joint_node]
