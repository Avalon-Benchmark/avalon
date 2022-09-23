import attr
import godot_parser
import numpy as np
from godot_parser import GDObject
from godot_parser import Node as GDNode

from avalon.datagen.world_creation.entities.tools.tool import Tool
from avalon.datagen.world_creation.types import GodotScene
from avalon.datagen.world_creation.utils import make_transform
from avalon.datagen.world_creation.utils import scale_basis


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Stone(Tool):
    size: float = 1.0
    mass: float = 10.0
    resource_file: str = "res://items/stone.tscn"
    script_file: str = "res://items/stone.gd"
    default_mesh_extents: float = 1.0

    def get_node(self, scene: GodotScene) -> GDNode:
        mesh_node = super().get_node(scene)
        item_name = mesh_node.name
        mesh_node.name = "mesh"

        scale_factor = self.size / self.default_mesh_extents
        mesh_node.properties["transform"] = make_transform(
            scale_basis(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), np.array([scale_factor] * 3))
        )

        half_size = self.size / 2
        collision_shape_node = GDNode(
            "collision_shape",
            type="CollisionShape",
            properties={
                "shape": scene.add_sub_resource(
                    "BoxShape",
                    extents=godot_parser.Vector3(half_size, half_size, half_size),
                ).reference,
            },
        )
        item_node = GDNode(
            item_name,
            type="RigidBody",
            properties={
                "transform": GDObject("Transform", *self.basis, *self.position),
                "script": scene.add_ext_resource(self.script_file, "Script").reference,
                "entity_id": self.entity_id,
                "mass": self.mass,
            },
        )
        item_node.add_child(collision_shape_node)
        item_node.add_child(mesh_node)
        return item_node

    def get_offset(self) -> float:
        return (self.size * self.default_mesh_extents) / 2.0
