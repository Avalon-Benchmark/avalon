import attr
import godot_parser
from godot_parser import Node as GDNode

from avalon.datagen.world_creation.entities.tools.tool import Tool
from avalon.datagen.world_creation.types import GodotScene


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ColoredSphere(Tool):
    resource_file: str = "res://items/sphere_marker.tscn"
    color: str = "#0000FF"
    scale: float = 2.0

    def get_node(self, scene: GodotScene) -> GDNode:
        node = super().get_node(scene)
        node.properties["color"] = self.color
        node.properties["scale"] = godot_parser.Vector3(self.scale, self.scale, self.scale)
        return node
