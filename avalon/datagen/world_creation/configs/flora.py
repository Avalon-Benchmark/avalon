import attr
import numpy as np


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class FloraConfig:
    # which 3D file to use for the geometry
    resource: str
    # whether or not to use the noise shader with perlin noise
    is_noise_shader_enabled: bool = False
    # whether this scenery should cast shadows
    is_shadowed: bool = True
    # the default scale for this 3D model
    default_scale: float = 1.0
    # how large the collision box is at the default scale. Only applies to trees
    collision_extents: np.ndarray = np.array([1.0, 5.0, 1.0])
    # how much to move the object in the vertical access when placing.
    # helps ensures that things are not floating off of highly sloped terrain
    # only applies to trees
    height_offset: float = 0.0
