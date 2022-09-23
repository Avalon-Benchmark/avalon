import attr

from avalon.contrib.serialization import Serializable


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class WorldConfig(Serializable):
    # random number. Used by some other functions to create a deterministic source of randomness so that we can
    # guarantee that the same worlds are produced every time the function is called with the same seed.
    seed: int
    # how large the world is, in meters, along one edge
    size_in_meters: float
    # how many points will be created for triangulation of the heightmap.
    # controls how detailed the terrain is
    # You probably should not change this unless you plan on re-tuning and changing quite a few other constants, as
    # it affects how reliably distances can be specified in tasks (ex: if there are very few points per meter, it is
    # difficult to control, for example, the exact size of a gap over which we want the agent to jump).
    # This parameter is a trade-off between how fine our control is over the geometry of the world and performance.
    # (at any given world size)
    point_density_in_points_per_square_meter: float
    # terrain construction works by iteratively subdividing a grid.
    # this controls the initial size of the grid.
    # lower numbers are much higher diversity terrain because the iteration works by adding noise, then subdividing.
    # Min is 3 because we need to be able to interpolate
    initial_point_count: int
    # TODO: DEPRECATED:
    # the scale of the noise actually doesn't matter because we reset the height afterwards
    # only really matters relative to mountain_noise_scale
    initial_noise_scale: float
    # how many times to do the add noise + subdivide loop
    fractal_iteration_count: int
    # how quickly the terrain noise decays with each iteration.
    # should be set to < 1.0 to make the noise get smaller each iteration (as it is multiplied by the noise)
    # setting it > 1.0 is allowed, though it will result if crazy jagged worlds that are not particularly interesting.
    noise_scale_decay: float
    # how many "mountains" to create. Mountains, in our terrain generation code, are simply mounds of height that are
    # added to the terrain. This factor controls how many of them to add at each iteration.
    # setting to zero will turn off mountains completely, making the rest of the mountain parameters pointless
    mountain_noise_count: int
    # how large the mound is that is created
    mountain_radius: float
    # how far away from the edge of the map to stay. Helpful for keeping the mountains towards the center to ensure
    # that there will be some land
    mountain_offset: float
    # how the mountain radius is updated with each fractal iteration. Same decay logic as noise_scale_decay
    mountain_radius_decay: float
    # how large the mountain effect is. Note that this only matters relative to the initial_noise_scale, which we
    # canonically set to 1.0.  Setting mountain_noise_scale to larger values means that more of the noise will come
    # from the mountain effect vs the random noise effect.
    mountain_noise_scale: float
    # whether mountain placement is normally or uniformly distributed
    is_mountain_placement_normal_distribution: bool
    # how high to make the highest point of the map (in meters above sea level).
    final_max_altitude_meters: float
    # this is the fraction of the map on each edge that will be linearly interpolated towards zero to ensure that
    # the world is a well-formed island (ie, its edge intersects the ocean)
    final_world_edge_fade_fraction: float = 0.05
    # unlike the above parameter, this controls a sort of fading/lowering of height that does not guarantee that the
    # terrain will reach the ocean. However, it certainly helps in most cases, and it looks much more natural
    # this parameter controls the fraction of the world, on each edge, over which this effect is applied
    fade_fraction: float = 0.2
    # the spatial scale of the noise that is applied to this fading effect. Smaller values make bigger patterns
    fade_noise_scale: float = 0.01
    # how harsh to make the fading effect. Smaller values will make it have a larger effect
    fade_noise_min: float = 0.5
    # whether to apply this fading effect in a circular or square way. The circlse look nicer
    is_fade_circular: bool = True
    # if greater than zero, controls the standard deviation of the blur kernel that is applied
    blur_meters_std_dev: float = 0
    # is set to true on maps where we create a world, but it's pointless, because the only interesting feature is a
    # building, which you spawn inside
    is_indoor_only: bool = False
