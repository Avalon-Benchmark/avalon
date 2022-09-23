from typing import List

import attr
import numpy as np
from godot_parser import Node as GDNode

from avalon.datagen.world_creation.entities.entity import Entity
from avalon.datagen.world_creation.types import GodotScene
from avalon.datagen.world_creation.types import Point3DNP


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class DoorLock(Entity):
    entity_id: int = -1

    def get_additional_door_body_nodes(
        self, scene: GodotScene, body_size: np.ndarray, body_centroid: Point3DNP = np.array([0, 0, 0])
    ) -> List[GDNode]:
        return []
