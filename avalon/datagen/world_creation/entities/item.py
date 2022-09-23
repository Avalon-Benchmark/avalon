from typing import Optional
from typing import Union
from typing import cast

import attr
import godot_parser
import numpy as np
from godot_parser import GDObject
from godot_parser import Node as GDNode
from scipy.spatial.transform import Rotation

from avalon.common.utils import to_immutable_array
from avalon.datagen.world_creation.constants import IDENTITY_BASIS
from avalon.datagen.world_creation.entities.entity import Entity
from avalon.datagen.world_creation.types import GodotScene

RotationType = Union[np.ndarray, Rotation]


def to_immutable_array_or_rotation(value: RotationType) -> RotationType:
    if isinstance(value, Rotation):
        return value
    return to_immutable_array(value)


def is_array_or_rotation_equal(val_a: RotationType, val_b: RotationType) -> bool:
    if isinstance(val_a, Rotation):
        if not isinstance(val_b, Rotation):
            return False
        val_a = val_a.as_matrix()
        val_b = val_b.as_matrix()
    if isinstance(val_b, Rotation):
        return False
    return np.array_equal(val_a, val_b)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Item(Entity):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class InstancedItem(Item):
    resource_file: str = ""
    rotation: Union[np.ndarray, Rotation] = attr.ib(
        default=IDENTITY_BASIS,
        converter=to_immutable_array_or_rotation,
        eq=attr.cmp_using(eq=is_array_or_rotation_equal),  # type: ignore[attr-defined]
    )
    safe_scale: Optional[np.ndarray] = None
    base_color: Optional[str] = None
    entity_id: int = -1

    @property
    def basis(self) -> np.ndarray:
        if isinstance(self.rotation, Rotation):
            return cast(np.ndarray, self.rotation.as_matrix().flatten())
        else:
            assert isinstance(self.rotation, np.ndarray)
            return self.rotation

    @property
    def node_name(self) -> str:
        assert self.entity_id != -1, (
            f"Attempt to access {self}.node_name before entity_id was set. "
            f"This is likely a bug due to NodePath-related code being called before world.add_item({self})"
        )
        item_name = self.resource_file.split("/")[-1].replace(".tscn", "")
        return f"{item_name}__{self.entity_id}"

    def get_node(self, scene: GodotScene) -> GDNode:
        resource = scene.add_ext_resource(self.resource_file, "PackedScene")
        properties = {
            "entity_id": self.entity_id,
            "transform": GDObject("Transform", *self.basis, *self.position),
        }
        if self.safe_scale is not None:
            properties["safe_scale"] = godot_parser.Vector3(*self.safe_scale)
        if self.base_color is not None:
            properties["base_color"] = self.base_color
        return GDNode(
            self.node_name,
            instance=resource.reference.id,
            properties=properties,
        )


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class InstancedDynamicItem(InstancedItem):
    is_dynamic: bool = True
    mass: float = 1.0

    def get_node(self, scene: GodotScene) -> GDNode:
        node = super().get_node(scene)
        node.properties["mass"] = self.mass
        return node

    def get_offset(self) -> float:
        return 0.5
