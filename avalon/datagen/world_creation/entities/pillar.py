import attr
import numpy as np
from godot_parser import GDObject
from godot_parser import Node as GDNode
from godot_parser import Vector3
from scipy.spatial.transform import Rotation

from avalon.datagen.world_creation.entities.item import Item
from avalon.datagen.world_creation.types import GodotScene


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Pillar(Item):
    size: np.ndarray
    is_dynamic: bool = False
    yaw: float = 0.0
    entity_id: int = -1

    def get_node(self, scene: GodotScene) -> GDNode:
        yaw_rotation = Rotation.from_euler("y", self.yaw, degrees=True)
        rotation = yaw_rotation.as_matrix().flatten()
        transform = GDObject("Transform", *rotation, *self.position)
        root = GDNode(
            f"pillar_{self.entity_id}",
            type="StaticBody",
            properties={
                "transform": transform,
            },
        )
        root.add_child(
            GDNode(
                "mesh",
                type="MeshInstance",
                properties={
                    "mesh": scene.add_sub_resource(
                        "CubeMesh",
                        material=scene.add_ext_resource("res://shaders/BasicColor.material", "Material").reference,
                        size=Vector3(*self.size),
                    ).reference,
                    "material/0": None,
                },
            )
        )
        root.add_child(
            GDNode(
                "collision",
                type="CollisionShape",
                properties={
                    "shape": scene.add_sub_resource(
                        "BoxShape", resource_local_to_scene=True, extents=Vector3(*self.size / 2.0)
                    ).reference
                },
            )
        )

        return root
