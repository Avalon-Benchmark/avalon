import attr

from avalon.datagen.world_creation.entities.tools.tool import Tool


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Log(Tool):
    mass: float = 60.0
    resource_file: str = "res://items/log.tscn"

    def get_offset(self) -> float:
        return 0.7  # need to scale this if we scale size
