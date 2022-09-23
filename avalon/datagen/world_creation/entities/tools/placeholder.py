import attr
import numpy as np
from godot_parser import Node as GDNode

from avalon.common.utils import to_immutable_array
from avalon.datagen.world_creation.entities.tools.tool import Tool
from avalon.datagen.world_creation.types import GodotScene


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Placeholder(Tool):
    position: np.ndarray = attr.ib(
        default=np.array([0.0, 0.0, 0.0]),
        converter=to_immutable_array,
        eq=attr.cmp_using(eq=np.array_equal),  # type: ignore[attr-defined]
    )
    entity_id: int = -1
    resource_file: str = "res://items/not_real.tscn"
    offset: float = 0.0

    def get_offset(self) -> float:
        return self.offset

    def get_node(self, scene: GodotScene) -> GDNode:
        raise Exception("Not implemented for a reason--should never be called!")
