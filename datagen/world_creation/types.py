from typing import Union

from godot_parser import GDExtResourceSection
from godot_parser import GDObject
from godot_parser import GDSubResourceSection

GDLinkedSection = Union[GDSubResourceSection, GDExtResourceSection]


class RawGDObject(GDObject):
    def __init__(self, data, *args) -> None:
        super().__init__("", *args)
        self.data = data

    def __str__(self) -> str:
        return self.data
