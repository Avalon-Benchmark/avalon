import attr
import numpy as np
from godot_parser import Node as GDNode

from avalon.common.utils import only
from avalon.datagen.world_creation.constants import IDENTITY_BASIS
from avalon.datagen.world_creation.entities.doors.locks.door_lock import DoorLock
from avalon.datagen.world_creation.indoor.blocks import new_make_block_nodes
from avalon.datagen.world_creation.indoor.constants import DEFAULT_STORY_HEIGHT
from avalon.datagen.world_creation.indoor.utils import get_scaled_mesh_node
from avalon.datagen.world_creation.types import GodotScene


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class DoorOpenButton(DoorLock):
    size: np.ndarray = np.array([1, DEFAULT_STORY_HEIGHT, 0.1])
    rotation: np.ndarray = IDENTITY_BASIS

    def get_node(self, scene: GodotScene) -> GDNode:
        button_mesh_node = get_scaled_mesh_node(scene, "res://entities/doors/open_button.tscn", self.size)
        button_node = only(
            new_make_block_nodes(
                scene,
                position=self.position,
                size=self.size,
                rotation=self.rotation,
                make_parent=True,
                parent_type="RigidBody",
                parent_name=f"open_button_{self.entity_id}",
                parent_script=scene.add_ext_resource("res://entities/doors/open_button.gd", "Script").reference,
                parent_extra_props={
                    "entity_id": self.entity_id,
                },
                make_mesh=False,
                make_collision_shape=True,
            )
        )
        button_node.add_child(button_mesh_node)
        return button_node
