import attr

from datagen.world_creation.heightmap import MapBoolNP
from datagen.world_creation.heightmap import Point3DNP
from datagen.world_creation.new_world import NewWorld
from datagen.world_creation.world_location_data import WorldLocationData


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class CompositionalConstraint:
    locations: WorldLocationData
    world: NewWorld
    center: Point3DNP
    traversal_mask: MapBoolNP
    is_height_inverted: bool
