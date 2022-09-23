import attr

from avalon.datagen.world_creation.entities.tools.stone import Stone


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Boulder(Stone):
    mass: float = 60.0
    resource_file: str = "res://items/stone.tscn"
    script_file: str = "res://items/boulder.gd"
    size: float = 1.4
