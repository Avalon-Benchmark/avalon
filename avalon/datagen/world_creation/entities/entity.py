from typing import Tuple
from typing import TypeVar

import attr
import numpy as np
from godot_parser import Node as GDNode

from avalon.common.utils import to_immutable_array
from avalon.contrib.serialization import Serializable
from avalon.datagen.world_creation.types import GodotScene


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Entity(Serializable):
    entity_id: int
    is_dynamic: bool
    position: np.ndarray = attr.ib(
        converter=to_immutable_array, eq=attr.cmp_using(eq=np.array_equal)  # type: ignore[attr-defined]
    )

    @property
    def point2d(self) -> Tuple[float, float]:
        return (self.position[0], self.position[2])

    def get_node(self, scene: GodotScene) -> GDNode:
        raise NotImplementedError()


EntityType = TypeVar("EntityType", bound=Entity)
