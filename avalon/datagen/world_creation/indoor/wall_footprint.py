from typing import NamedTuple
from typing import Tuple
from typing import cast

import numpy as np

from avalon.datagen.world_creation.indoor.constants import WallType
from avalon.datagen.world_creation.types import Point3DNP


class WallFootprint(NamedTuple):
    top_left: Tuple[int, int]  # x, z
    bottom_right: Tuple[int, int]  # x, z
    wall_type: WallType
    is_vertical: bool

    @property
    def wall_thickness(self) -> int:
        return self.footprint_width if self.is_vertical else self.footprint_length

    @property
    def wall_length(self) -> int:
        return self.footprint_length if self.is_vertical else self.footprint_width

    @property
    def footprint_width(self) -> int:
        return self.bottom_right[0] - self.top_left[0]

    @property
    def footprint_length(self) -> int:
        return self.bottom_right[1] - self.top_left[1]

    @property
    def centroid(self) -> Point3DNP:
        return cast(Point3DNP, np.array(self.top_left) + (np.array(self.bottom_right) - np.array(self.top_left)) / 2)
