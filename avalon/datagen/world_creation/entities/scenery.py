import attr
import numpy as np

from avalon.datagen.world_creation.entities.item import InstancedItem


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Scenery(InstancedItem):
    is_dynamic: bool = False
    scale: np.ndarray = np.array([1, 1, 1])
    entity_id: int = -1
