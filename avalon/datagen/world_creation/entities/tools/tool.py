from typing import Optional

import attr
import numpy as np

from avalon.common.utils import to_immutable_array
from avalon.datagen.world_creation.entities.item import InstancedDynamicItem
from avalon.datagen.world_creation.types import MapBoolNP
from avalon.datagen.world_creation.types import Point3DNP


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Tool(InstancedDynamicItem):
    position: Point3DNP = attr.ib(
        default=np.array([0.0, 0.0, 0.0]),
        converter=to_immutable_array,
        eq=attr.cmp_using(eq=np.array_equal),  # type: ignore[attr-defined]
    )
    solution_mask: Optional[MapBoolNP] = attr.ib(default=None, eq=False)
