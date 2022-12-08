from enum import Enum
from typing import Dict
from typing import SupportsIndex
from typing import Tuple
from typing import Union

import attr
from godot_parser import GDExtResourceSection
from godot_parser import GDObject
from godot_parser import GDScene
from godot_parser import GDSection
from godot_parser import GDSubResourceSection
from nptyping import Bool
from nptyping import Double
from nptyping import Int32
from nptyping import NDArray
from nptyping import Shape

GDLinkedSection = Union[GDSubResourceSection, GDExtResourceSection]


class RawGDObject(GDObject):
    def __init__(self, data: str) -> None:
        super().__init__("")
        self.data = data

    def __str__(self) -> str:
        return self.data


FloatListNP = NDArray[Shape["ValueIndex"], Double]
Point2DNP = NDArray[Shape["2"], Double]
Point2DListNP = NDArray[Shape["PointIndex, 2"], Double]
Point3DNP = NDArray[Shape["3"], Double]
Basis3DNP = NDArray[Shape["12"], Double]
Point3DListNP = NDArray[Shape["PointIndex, 3"], Double]
MapBoolNP = NDArray[Shape["MapY, MapX"], Bool]
MapFloatNP = NDArray[Shape["MapY, MapX"], Double]
MapIntNP = NDArray[Shape["MapY, MapX"], Int32]

BuildingBoolNP = NDArray[Shape["Z, X"], Bool]
BuildingFloatNP = NDArray[Shape["Z, X"], Double]
BuildingIntNP = NDArray[Shape["Z, X"], Int32]

TileConvKernel = NDArray[Shape["3, 3"], Int32]


def Color(r: float, g: float, b: float, a: float):
    return RawGDObject(f"Color( {r}, {g}, {b}, {a} )")


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Biome:
    id: int
    name: str
    color: str
    color_jitter: float = 0.02
    correlated_color_jitter: float = 0.2


class HeightMode(Enum):
    RELATIVE = "RELATIVE"
    ABSOLUTE = "ABSOLUTE"
    MIDPOINT_RELATIVE = "MIDPOINT_RELATIVE"
    MIDPOINT_ABSOLUTE = "MIDPOINT_ABSOLUTE"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class DebugVisualizationConfig:
    spawn_mode: str = "perspective"  # perspective, normal
    is_2d_graph_drawn: bool = False


class SceneryBorderMode(Enum):
    HARD = "HARD"
    LINEAR = "LINEAR"
    SQUARED = "SQUARED"
    INVERSE_SQUARE = "INVERSE_SQUARE"


class IdGenerator:
    def __init__(self) -> None:
        self.next_id: int = 0

    def get_next_id(self) -> int:
        result: int = self.next_id
        self.next_id += 1
        return result


class GodotScene(GDScene):
    def __init__(self, *sections: GDSection) -> None:
        super().__init__(*sections)
        self._ext_resources: Dict[str, GDExtResourceSection] = {}

    def add_ext_resource(self, path: str, type: str) -> GDExtResourceSection:
        if path not in self._ext_resources:
            self._ext_resources[path] = super().add_ext_resource(path, type)
        return self._ext_resources[path]


class WorldType(Enum):
    PLATONIC = "PLATONIC"
    ARCHIPELAGO = "ARCHIPELAGO"
    CONTINENT = "CONTINENT"
    # this is a trash early one
    JAGGED = "JAGGED"


RGBATuple = Tuple[float, float, float, float]


class SupportsIndexTuple(SupportsIndex, Tuple[int, int]):
    # numpy supports indexing via a tuple, but the type annotations don't know this
    pass
