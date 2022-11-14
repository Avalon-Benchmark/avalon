from typing import Final

import attr
from godot_parser import GDObject
from godot_parser import Node as GDNode
from scipy.spatial.transform import Rotation

from avalon.datagen.world_creation.entities.item import Item
from avalon.datagen.world_creation.types import GodotScene

PLAYER_SPAWN_POINT: Final = "SpawnPoint"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class SpawnPoint(Item):

    pitch: float
    yaw: float
    is_dynamic: bool = False
    is_visibility_required: bool = True
    entity_id: int = -1

    name: str = PLAYER_SPAWN_POINT

    def get_node(self, scene: GodotScene) -> GDNode:
        spawn_yaw_rotation = Rotation.from_euler("y", self.yaw, degrees=True)
        spawn_pitch_rotation = Rotation.from_euler("x", self.pitch, degrees=True)
        spawn_rotation = (spawn_yaw_rotation * spawn_pitch_rotation).as_matrix().flatten()
        spawn_transform = GDObject("Transform", *spawn_rotation, *self.position)
        return GDNode(self.name, type="Spatial", properties={"transform": spawn_transform})
