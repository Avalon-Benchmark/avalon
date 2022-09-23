import attr

from avalon.datagen.world_creation.types import SceneryBorderMode


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class SceneryConfig:
    # the file that defines the object
    resource_file: str
    # the biome into which to spawn the items
    biome_id: int
    # number of items per square meter (at density=1.0)
    density: float = 0.2
    # how the density varies in the distances between 0 and border_distance
    border_mode: SceneryBorderMode = SceneryBorderMode.LINEAR
    # over what border range to smooth between 0.0 and 1.0
    border_distance: float = 5.0
    # scale of the objects will be +/- this fraction on all axes
    correlated_scale_range: float = 0.4
    # scale of the objects will be +/- this fraction on EACH axis
    # ie, will squish and stretch the object
    skewed_scale_range: float = 0.2
    # if set to True, will align the item with the normal of the surface
    is_oriented_to_surface: bool = False
