from typing import Dict

from godot_parser import GDExtResourceSection
from godot_parser import GDScene
from godot_parser import GDSection


class ImprovedGodotScene(GDScene):
    def __init__(self, *sections: GDSection) -> None:
        super().__init__(*sections)
        self._ext_resources: Dict[str, GDExtResourceSection] = {}

    def add_ext_resource(self, path: str, type: str) -> GDExtResourceSection:
        if path not in self._ext_resources:
            self._ext_resources[path] = super().add_ext_resource(path, type)
        return self._ext_resources[path]
