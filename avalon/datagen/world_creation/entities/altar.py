from typing import cast

import attr
import numpy as np

from avalon.common.utils import to_immutable_array
from avalon.datagen.world_creation.constants import UP_VECTOR
from avalon.datagen.world_creation.entities.food import Food
from avalon.datagen.world_creation.entities.pillar import Pillar
from avalon.datagen.world_creation.entities.utils import get_random_ground_points
from avalon.datagen.world_creation.types import MapBoolNP
from avalon.datagen.world_creation.types import Point3DListNP
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.worlds.height_map import HeightMap


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Altar(Pillar):
    position: np.ndarray = attr.ib(
        default=np.array([0.0, 0.0, 0.0]),
        converter=to_immutable_array,
        eq=attr.cmp_using(eq=np.array_equal),  # type: ignore[attr-defined]
    )
    entity_id: int = -1
    is_tool_helpful: bool = False

    @staticmethod
    def build(height: float, table_dim: float = 1.0) -> "Altar":
        return Altar(size=np.array([table_dim, height, table_dim]))

    def place(self, food_ground_position: Point3DNP) -> "Altar":
        return attr.evolve(self, position=food_ground_position.copy() + UP_VECTOR * self.get_offset())

    def get_food_height(self, food: Food) -> float:
        return cast(float, self.size[1] + food.get_offset())

    def get_offset(self) -> float:
        return cast(float, self.size[1] / 2.0)

    def get_food_locations(
        self,
        rand: np.random.Generator,
        center: Point3DNP,
        count: int,
        map: HeightMap,
        min_radius: float,
        max_radius: float,
        offset: float,
        island_mask: MapBoolNP,
    ) -> Point3DListNP:
        points = get_random_ground_points(rand, center, count, map, min_radius, max_radius, offset, island_mask)
        points[:, 1] = self.size[1] + offset
        return points
