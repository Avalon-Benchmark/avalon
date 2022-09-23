import attr

from avalon.datagen.world_creation.types import MapBoolNP
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world_locations import WorldLocations


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class CompositionalConstraint:
    locations: WorldLocations
    world: World
    center: Point3DNP
    traversal_mask: MapBoolNP
    is_height_inverted: bool
