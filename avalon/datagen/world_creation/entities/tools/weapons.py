from typing import ClassVar

import attr

from avalon.datagen.world_creation.entities.tools.tool import Tool


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Weapon(Tool):
    WEAPON_VALUE: ClassVar = 0


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Stick(Weapon):
    WEAPON_VALUE: ClassVar = 1.0
    mass: float = 5.0
    resource_file: str = "res://items/stick.tscn"

    def get_offset(self) -> float:
        return 0.35


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class LargeStick(Stick):
    WEAPON_VALUE: ClassVar = 1.0
    resource_file: str = "res://items/large_stick.tscn"

    def get_offset(self) -> float:
        return 0.35


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Rock(Weapon):
    WEAPON_VALUE: ClassVar = 0.5

    mass: float = 1.0
    resource_file: str = "res://items/rock.tscn"

    def get_offset(self) -> float:
        return 0.25


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class LargeRock(Rock):
    WEAPON_VALUE: ClassVar = 0.5
    resource_file: str = "res://items/large_rock.tscn"

    def get_offset(self) -> float:
        return 0.5
