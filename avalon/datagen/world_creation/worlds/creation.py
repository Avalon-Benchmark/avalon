from typing import Optional
from typing import Tuple

import attr
import numpy as np
from loguru import logger
from nptyping import assert_isinstance
from scipy import stats
from skimage import morphology

from avalon.common.errors import SwitchError
from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.godot_base_types import FloatRange
from avalon.datagen.world_creation.configs.biome import generate_biome_config
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.world import WorldConfig
from avalon.datagen.world_creation.constants import HALF_AGENT_HEIGHT_VECTOR
from avalon.datagen.world_creation.debug_plots import plot_value_grid
from avalon.datagen.world_creation.entities.constants import CANONICAL_FOOD_HEIGHT_ON_TREE
from avalon.datagen.world_creation.entities.constants import FOOD_TREE_VISIBLE_HEIGHT
from avalon.datagen.world_creation.types import WorldType
from avalon.datagen.world_creation.worlds.difficulty import difficulty_variation
from avalon.datagen.world_creation.worlds.height_map import Point2DNP
from avalon.datagen.world_creation.worlds.height_map import Point3DNP
from avalon.datagen.world_creation.worlds.types import CompositionalConstraint
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world_locations import WorldLocations


def _calculate_fractal_iteration_count(
    initial_count: int, grid_points_per_meter: float, min_size: float
) -> Tuple[int, float]:
    """:returns: how many times we need to double initial_count in order to exceed desired_count"""
    iterations = 0
    count = initial_count
    meters_per_grid_point = 1.0 / grid_points_per_meter
    while True:
        if count * meters_per_grid_point > min_size:
            break
        count *= 2
        iterations += 1
    return iterations, count * meters_per_grid_point


def _range_via_difficulty(min_val: float, max_val: float, difficulty: float, rand: np.random.Generator) -> float:
    delta = max_val - min_val
    return min_val + delta * rand.uniform() * difficulty


def _world_2d_to_3d(point_2d: Point2DNP, world: World) -> Point3DNP:
    assert_isinstance(point_2d, Point2DNP)
    index = world.map.point_to_index(point_2d)
    point_height = world.map.Z[index]
    return np.array([point_2d[0], point_height, point_2d[1]])


def create_world_from_constraint(
    goal_distance: stats.norm,
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint],
    food_height: Optional[float] = None,
    min_size_in_meters: Optional[float] = None,
    ideal_shore_dist_options: Tuple[float, ...] = (4.0, 2.0),
) -> Tuple[World, WorldLocations]:
    if food_height is None:
        visibility_height = FOOD_TREE_VISIBLE_HEIGHT
        food_height = CANONICAL_FOOD_HEIGHT_ON_TREE
    else:
        visibility_height = food_height
    if constraint is None:
        world, locations = create_world_for_skill_scenario(
            rand,
            difficulty,
            food_height,
            goal_distance,
            export_config,
            min_size_in_meters=min_size_in_meters,
            ideal_shore_dist_options=ideal_shore_dist_options,
            visibility_height=visibility_height,
        )
        locations, world = world.begin_height_obstacles(locations)
    else:
        world, locations = constraint.world, constraint.locations
    return world, locations


# these are the default max and min sizes.. move diverse worlds are made larger
BASE_MIN_SIZE_IN_METERS = 30.0
BASE_MAX_SIZE_IN_METERS = 300.0
# how finely the height grid is specified. This corresponds to 0.5m per cell in the HeightMap
GRID_POINTS_PER_METER = 2.0
# how much space to flatten around spawn and goal locations.
# more diverse worlds will have a smaller flattening radius
MAX_FLATTEN_RADIUS = 5.0
MIN_FLATTEN_RADIUS = 1.0
# if visibility is required, how many goal locations to try before giving up and trying a different spawn location
# note that visibility is almost always required, except for some of the more complex and compositional tasks
MAX_VISIBILITY_CHECKS = 5
# how many different spawn locations to try before giving up and making an entirely new world.
# some worlds are simply broken given the constraints specified (ex: imagine asking for a goal and spawn point that
# are 10 meters apart, but there is only a single tiny patch of land)
MAX_SPAWN_TRIES_PER_MAP = 5
# how many different maps to try making before giving up.
# if this is reached, an ImpossibleWorldError is raised
# note that this normally means the caller is asking for constraints that are far too tight
MAX_OVERALL_RETRIES = 10
# prevent things from spawning in your body. No goal point will be selected that is closer than this to the spawn point
MIN_GOAL_DISTANCE = 1.0


def create_world_for_skill_scenario(
    rand: np.random.Generator,
    diversity: float,
    food_height: float,
    # worlds will usually be 2x this size so that you have a good chance of getting the right distance even if you randomly choose a point
    goal_distance: stats.norm,
    export_config: ExportConfig,
    is_logging: bool = False,
    world_type: Optional[WorldType] = None,
    is_visibility_required: bool = True,
    explicit_height_ratio: Optional[float] = None,
    foliage_density_modifier: Optional[float] = None,
    min_size_in_meters: Optional[float] = None,
    max_size_in_meters: Optional[float] = None,
    # first try spawning with us not right next to the water, then try without that
    ideal_shore_dist_options: Tuple[float, ...] = (4.0, 2.0),
    visibility_height: Optional[float] = None,
) -> Tuple[World, WorldLocations]:
    if visibility_height is None:
        visibility_height = food_height

    food_offset = np.array([0, food_height, 0])
    visibility_offset = np.array([0, visibility_height, 0])

    # TODO probably want to remove this
    is_debugging = False

    if max_size_in_meters is None:
        max_size_in_meters = BASE_MAX_SIZE_IN_METERS

    if foliage_density_modifier is None:
        foliage_density_modifier = difficulty_variation(5.0, 30.0, rand, diversity)

    # figure out how big the world should be
    goal_distance_preference = goal_distance.mean()
    # we won't return worlds where the goal is closer than two standard deviations from what you requested
    hard_min_distance = goal_distance_preference - goal_distance.std() * 2.0
    if hard_min_distance < 0.0:
        hard_min_distance = 0.0

    if is_debugging:
        logger.debug(f"Goal and spawn must be at least {hard_min_distance:.1f}m away")

    if min_size_in_meters is None:
        min_size_in_meters = goal_distance_preference * 2.0

    if min_size_in_meters < BASE_MIN_SIZE_IN_METERS:
        min_size_in_meters = BASE_MIN_SIZE_IN_METERS

    if is_debugging:
        logger.debug(f"Min size in meters: {min_size_in_meters}")
        logger.debug(f"Max size in meters: {max_size_in_meters}")

    if max_size_in_meters < min_size_in_meters:
        raise Exception(
            "This task is misconfigured. We should not be giving values that result in mins greater than maxes"
        )

    # we decrease diversity and retry if we fail to find points that are sufficiently far away and have visibility
    # this should generally take zero retries, so not really a performance concern
    num_visibility_check = 0
    for retry_num in range(MAX_OVERALL_RETRIES):

        # lower numbers are much higher diversity terrain. Min is 3 because we need to be able to interpolate
        initial_point_count = round(difficulty_variation(8, 3, rand, diversity))

        # how many points we want in the grid. Controls how many times we're going to iterate
        iteration_min_size_in_meters = min_size_in_meters
        fractal_iteration_count, size_in_meters = _calculate_fractal_iteration_count(
            initial_point_count, GRID_POINTS_PER_METER, iteration_min_size_in_meters
        )
        simplicity_warnings = []
        # if it's too big, just reduce the size and try again
        while size_in_meters > max_size_in_meters:
            iteration_min_size_in_meters /= 2.0
            fractal_iteration_count, size_in_meters = _calculate_fractal_iteration_count(
                initial_point_count, GRID_POINTS_PER_METER, iteration_min_size_in_meters
            )
        if size_in_meters < goal_distance_preference:
            simplicity_warnings.append(
                f"World was somewhat smaller than desired: {size_in_meters} vs {goal_distance_preference}"
            )

        if is_debugging:
            logger.info(f"World size: {size_in_meters}")
        # logger.info(f"World size: {size_in_meters}")

        # how much to flatten the world near the important points
        flatten_radius = (1.0 - diversity) * (MAX_FLATTEN_RADIUS - MIN_FLATTEN_RADIUS) + MIN_FLATTEN_RADIUS

        config = generate_world_config(
            rand,
            diversity,
            export_config,
            initial_point_count,
            fractal_iteration_count,
            size_in_meters,
            world_type,
            explicit_height_ratio,
        )
        biome_config = generate_biome_config(
            rand, export_config, config.final_max_altitude_meters, foliage_density_modifier
        )

        world = World.build(config, export_config, biome_config=biome_config, is_debug_graph_printing_enabled=False)
        map_new = world.map.copy()
        special_biomes_new = map_new.erode_shores(rand, biome_config)
        world = attr.evolve(world, special_biomes=special_biomes_new, map=map_new)

        if is_debugging and False:
            plot_value_grid(world.map.Z, "Possible world")
            world.map.plot()

        # don't spawn within two meters of the edge, that's asking for trouble
        edge_size = 2.0
        coord_range = FloatRange((-size_in_meters / 2.0) + edge_size, (size_in_meters / 2.0) - edge_size)
        for ideal_shore_dist in ideal_shore_dist_options:
            # pick a random point to spawn at
            for i in range(MAX_SPAWN_TRIES_PER_MAP):
                spawn_point_2d = world.map.get_random_land_point(rand, ideal_shore_dist)
                if not coord_range.contains(spawn_point_2d[0]) or not coord_range.contains(spawn_point_2d[1]):
                    if is_debugging:
                        logger.debug("Odd, out of range")
                    continue
                # the extra 1.1 is just to make sure you dont start in the ground :-P
                spawn_point = _world_2d_to_3d(spawn_point_2d, world) + HALF_AGENT_HEIGHT_VECTOR * 1.1

                # filter points to those on the same island
                island_mask, island_mask_without_fresh_water = world.map.get_island(spawn_point_2d)

                restricted_found = False
                if ideal_shore_dist > 0.0:
                    food_mask = island_mask
                    for multiple in (1.0, 0.5):
                        shore_cell_dist = int(multiple * world.map.cells_per_meter * ideal_shore_dist) + 1
                        # shrink the island a little bit so that we dont spawn on the shore because it's annoying
                        smaller_food_mask = np.logical_and(
                            island_mask,
                            np.logical_not(
                                morphology.dilation(np.logical_not(island_mask), morphology.disk(shore_cell_dist))
                            ),
                        )
                        if np.any(smaller_food_mask):
                            restricted_found = True
                            # plot_value_grid(island_mask)
                            # plot_value_grid(smaller_food_mask)
                            food_mask = smaller_food_mask
                            break
                else:
                    food_mask = island_mask

                # couldnt find points sufficiently far from the ocean, not good
                if not restricted_found:
                    if is_debugging:
                        logger.debug("not far enough from ocean")
                    continue

                # also prevent things from being just way too close
                if hard_min_distance > 0.0:
                    sq_min_dist = hard_min_distance**2
                    food_mask = np.logical_and(food_mask, world.map.get_dist_sq_to(spawn_point_2d) > sq_min_dist)

                possible_food_points = np.stack([world.map.X, world.map.Y], axis=2)[food_mask]

                # try a few randomly selected points (weighted by distance preference) until one is visible
                visibility_calculator = None
                if is_visibility_required:
                    visibility_calculator = world.map.generate_visibility_calculator()
                if goal_distance:
                    distances = np.linalg.norm(possible_food_points - spawn_point_2d, axis=1)
                    location_weights = goal_distance.pdf(distances)
                    location_weights[distances < MIN_GOAL_DISTANCE] = 0.0
                    total_weights = location_weights.sum()
                    if total_weights <= 0.0:
                        if is_debugging:
                            logger.debug("No likely food locations")
                        continue
                    location_probabilities = location_weights / total_weights
                    food_point_2ds = rand.choice(possible_food_points, MAX_VISIBILITY_CHECKS, p=location_probabilities)
                else:
                    food_point_2ds = rand.choice(possible_food_points, MAX_VISIBILITY_CHECKS)
                for food_point_2d in food_point_2ds:
                    if not coord_range.contains(food_point_2d[0]) or not coord_range.contains(food_point_2d[1]):
                        if is_debugging:
                            logger.debug("food point is out of range")
                        continue
                    # can't spawn food in literally the same spot as you :-P
                    if np.isclose(food_point_2d, spawn_point_2d).all():
                        if is_debugging:
                            logger.debug("Food is too close to spawn")
                        continue
                    if visibility_calculator is None:
                        # we disabled visibility above, so dont worry about it
                        is_food_visible = True
                    else:
                        # adds a bit of extra height because the spawn point is technically the middle of the agent
                        is_food_visible = visibility_calculator.is_visible_from(
                            spawn_point + HALF_AGENT_HEIGHT_VECTOR,
                            _world_2d_to_3d(food_point_2d, world) + visibility_offset,
                        )
                    if is_food_visible:
                        # how much space to reserve around each object
                        object_radius = 0.75
                        # if they're both within the flattening region, just flatten it all out
                        final_distance = np.linalg.norm(spawn_point_2d - food_point_2d)
                        if final_distance < object_radius * 2.0:
                            world = world.flatten(
                                (spawn_point_2d + food_point_2d) / 2.0, flatten_radius, object_radius * 3.0
                            )
                        else:
                            flatten_radius = min([flatten_radius, final_distance / 2.0])  # type: ignore
                            world = world.flatten(spawn_point_2d, flatten_radius, object_radius)
                            world = world.flatten(food_point_2d, flatten_radius, object_radius)
                        if is_logging:
                            logger.info(f"Visibility checks: {num_visibility_check}")
                        return world, WorldLocations(
                            island=island_mask_without_fresh_water,
                            spawn=spawn_point,
                            goal=_world_2d_to_3d(food_point_2d, world) + food_offset,
                        )
                    else:
                        if is_debugging:
                            logger.debug("food is not visible")
                        num_visibility_check += 1

    raise ImpossibleWorldError("Something is very wrong, or we got very unlucky...")


# worlds smaller than this will be boosted out of the water by the below functions.
# helps ensure there is a sufficiently large land mass on which we can spawn things.
MIN_WORLD_SIZE_FOR_COMPLEXITY = 40.0


def generate_world_config(
    rand: np.random.Generator,
    diversity: float,
    export_config: Optional[ExportConfig],
    # terrain construction works by iteratively subdividing a grid.
    # this controls the initial size of the grid.
    # lower numbers are much higher diversity terrain because the iteration works by adding noise, then subdividing.
    # Min is 3 because we need to be able to interpolate
    initial_point_count: int,
    # how many times to do the add noise + subdivide loop
    fractal_iteration_count: int,
    size_in_meters: float,
    world_type: Optional[WorldType],
    explicit_height_ratio: Optional[float],
) -> WorldConfig:
    # how fine grained the triangulation is
    # 4.0 is ultra high quality. Almost looks like a texture at that point
    # 1.0 is very high quality.
    # 0.1 is medium quality. You can feel that the ground is triangulated, but it doesn't feel offensive
    # 0.05 is about as low as you can go without things feeling broken
    point_density_in_points_per_square_meter = 0.1
    # helps wiggle the edge of the world
    # at 100m, 0.005 is pleasing when looking at the heightmaps
    fade_noise_scale = 0.005 * (100.0 / size_in_meters)
    # there is always a chance, for these simple worlds, that we create a fairly simple world
    # this allows us to practice the skills in a sort of ideal, platonic setting
    platonic_world_probability = np.interp(diversity, [0.0, 0.2, 0.6, 1.0], [1.0, 0.9, 0.4, 0.2])
    is_platonic_world = bool(rand.random() < platonic_world_probability)
    if world_type is not None:
        if world_type == WorldType.PLATONIC:
            is_platonic_world = True
        else:
            is_platonic_world = False
    # what really defines a platonic world is the noise scale of the mountains relative to the other noise
    # more mountains also makes it more platonic... (if they're in the same spot)
    if is_platonic_world:
        is_mountain_placement_normal_distribution = False
        mountain_noise_count = 1
        mountain_radius_decay = 1.0
        noise_scale_decay = pow(10.0, rand.uniform(-0.5, 0.5))
        mountain_radius = _range_via_difficulty(0.6, 0.4, diversity, rand)
        mountain_noise_scale = _range_via_difficulty(5.0, 3.0, diversity, rand)
        mountain_offset = _range_via_difficulty(0.45, 0.3, diversity, rand)
        smoothing_probability = np.interp(diversity, [0.0, 0.2, 0.6, 1.0], [0.5, 0.4, 0.2, 0.1])
        blur_meters_std_dev = 0.0
        is_smoothed = bool(rand.random() < smoothing_probability)
        smooth_min = 0.0
        iterated_noise_scale = pow(noise_scale_decay, fractal_iteration_count)
        if iterated_noise_scale > 15:
            is_smoothed = True
            smooth_min = 1.0
        if is_smoothed:
            blur_meters_std_dev = _range_via_difficulty(1.5, smooth_min, diversity, rand)
        height_ratio = _range_via_difficulty(0.05, 0.2, diversity, rand)
    else:
        if world_type is None:
            # TODO: maybe put archipelagos and stuff back someday, they interfere with the generation too much for now
            # # randomly select the type of world
            # world_type = rand.choice(
            #     [WorldType.CONTINENT, WorldType.ARCHIPELAGO, WorldType.JAGGED],
            #     p=[0.8, 0.2, 0.0],
            # )
            # logger.info(world_type)
            world_type = WorldType.CONTINENT

        # jagged noisy worlds
        if world_type == WorldType.JAGGED:
            # A value of 0.01 here is particularly pleasing
            # This is the scale at which our last iteration ends up
            iterated_noise_scale = pow(0.2, rand.normal(2.0, 0.5))
            noise_scale_decay = iterated_noise_scale ** (1 / (fractal_iteration_count - 1))
            is_mountain_placement_normal_distribution = False
            mountain_noise_count = 0
            mountain_radius = 0.0
            mountain_noise_scale = 0.0
            mountain_offset = 0.0
            mountain_radius_decay = 0.0
            # when small, boost it out of the water
            if size_in_meters < MIN_WORLD_SIZE_FOR_COMPLEXITY:
                mountain_noise_count = 1
                mountain_radius = 0.5
                mountain_noise_scale = 1.0
                mountain_offset = 0.4
                mountain_radius_decay = 1.0
            blur_meters_std_dev = 0.0
            if iterated_noise_scale > 0.1:
                blur_meters_std_dev = _range_via_difficulty(1.0, 0.0, diversity, rand)
            min_height_ratio = 0.02
            height_ratio_mean = (min_height_ratio + diversity * 0.1) + 0.05
            height_ratio_std_dev = (height_ratio_mean - min_height_ratio) * 0.5 * diversity
            height_ratio = rand.normal(height_ratio_mean, height_ratio_std_dev)
            height_ratio = max(height_ratio, min_height_ratio)
        elif world_type == WorldType.ARCHIPELAGO:
            height_variation = rand.normal()
            height_mean = 0.05 + diversity * 0.05
            if height_variation < 0:
                height_variation *= height_mean / 3.0
            else:
                height_variation *= 0.07 * diversity
            height_ratio = height_mean + height_variation
            min_height_ratio = 0.02
            height_ratio = max(height_ratio, min_height_ratio)

            noise_variation = rand.normal()
            if noise_variation < 0:
                noise_variation *= 2
            else:
                noise_variation *= 7 * diversity
            # 20 meters of noise is pretty difficult to traverse. You spend much of your time climbing
            #  roughly corresponds to very rocky outcroppings
            # 2 meters of noise end up with fairly flat terrain in large worlds
            meters_of_noise = max(6.0 + noise_variation, 0.5)
            noise_scale_base = meters_of_noise / (size_in_meters * height_ratio)

            # A value of 0.01 here is particularly pleasing
            # This is the scale at which our last iteration ends up
            iterated_noise_scale = pow(noise_scale_base, rand.normal(2.0, 0.5))
            noise_scale_decay = iterated_noise_scale ** (1 / (fractal_iteration_count - 1))
            is_mountain_placement_normal_distribution = False
            mountain_noise_count = 0
            mountain_radius = 0.0
            mountain_noise_scale = 0.0
            mountain_offset = 0.0
            mountain_radius_decay = 0.0
            # when small, boost it out of the water
            if size_in_meters < MIN_WORLD_SIZE_FOR_COMPLEXITY:
                mountain_noise_count = 1
                mountain_radius = 0.5
                mountain_noise_scale = 1.0
                mountain_offset = 0.4
                mountain_radius_decay = 1.0
            blur_meters_std_dev = 0.0
            if iterated_noise_scale > 0.04:
                blur_meters_std_dev = _range_via_difficulty(2.0, 0.2, diversity, rand)
        elif world_type == WorldType.CONTINENT:
            height_variation = rand.normal()
            height_mean = 0.05 + diversity * 0.1
            if height_variation < 0:
                height_variation *= height_mean / 3.0
            else:
                height_variation *= 0.07 * diversity
            height_ratio = height_mean + height_variation
            min_height_ratio = 0.02
            height_ratio = max(height_ratio, min_height_ratio)

            noise_variation = rand.normal()
            if noise_variation < 0:
                noise_variation *= 2
            else:
                noise_variation *= 7 * diversity
            # 20 meters of noise is pretty difficult to traverse. You spend much of your time climbing
            #  roughly corresponds to very rocky outcroppings
            # 2 meters of noise end up with fairly flat terrain in large worlds
            meters_of_noise = max(6.0 + noise_variation, 0.5)
            noise_scale_base = meters_of_noise / (size_in_meters * height_ratio)

            # A value of 0.01 here is particularly pleasing
            # This is the scale at which our last iteration ends up
            iterated_noise_scale = pow(noise_scale_base, rand.normal(2.0, 0.5))
            noise_scale_decay = iterated_noise_scale ** (1 / (fractal_iteration_count - 1))

            mountain_mode = rand.integers(0, 2)
            if mountain_mode == 0:
                # make one large central mountain
                is_mountain_placement_normal_distribution = rand.uniform() < 0.5
                mountain_noise_count = rand.integers(8, 12)
                mountain_radius_decay = rand.uniform(0.2, 0.7)
                mountain_radius = 0.4
                mountain_noise_scale = 0.1
                mountain_offset = _range_via_difficulty(0.45, 0.35, diversity, rand)
            elif mountain_mode == 1:
                # continually raise the middle section of the map out of the water
                # to make one relatively large continent
                is_mountain_placement_normal_distribution = False
                mountain_noise_count = 2
                mountain_radius_decay = 1.0
                mountain_radius = _range_via_difficulty(0.6, 0.2, diversity, rand)
                # controls how much it ends up being a single landmass
                # if the noise scale is higher, we end up with more of a single mass
                mountain_noise_scale = 0.12 + rand.normal() * 0.03
                mountain_offset = _range_via_difficulty(0.45, 0.2, diversity, rand)
            else:
                raise SwitchError(f"Unhandled mountain mode: {mountain_mode}")

            # when small, boost it out of the water
            if size_in_meters < MIN_WORLD_SIZE_FOR_COMPLEXITY:
                is_mountain_placement_normal_distribution = False
                mountain_noise_count = 1
                mountain_radius = 0.5
                mountain_noise_scale = 1.0
                mountain_offset = 0.4
                mountain_radius_decay = 1.0
            blur_meters_std_dev = 0.0
            if iterated_noise_scale > 0.04:
                blur_meters_std_dev = _range_via_difficulty(2.0, 0.2, diversity, rand)
        else:
            raise SwitchError(f"Unhandled world type: {world_type}")

    # just for debugging:
    if explicit_height_ratio is not None:
        height_ratio = explicit_height_ratio
    final_max_altitude_meters = height_ratio * size_in_meters
    # see the definition of this object for a description of each parameter
    config = WorldConfig(
        seed=rand.integers(0, np.iinfo(np.int64).max),
        point_density_in_points_per_square_meter=point_density_in_points_per_square_meter,
        size_in_meters=size_in_meters,
        initial_point_count=initial_point_count,
        fractal_iteration_count=fractal_iteration_count,
        # the scale of the noise actually doesn't matter because we reset the height afterwards
        # only really matters relative to mountain_noise_scale
        initial_noise_scale=1.0,
        noise_scale_decay=noise_scale_decay,
        is_mountain_placement_normal_distribution=is_mountain_placement_normal_distribution,
        mountain_noise_count=mountain_noise_count,
        mountain_radius=mountain_radius,
        mountain_noise_scale=mountain_noise_scale,
        mountain_offset=mountain_offset,
        mountain_radius_decay=mountain_radius_decay,
        blur_meters_std_dev=blur_meters_std_dev,
        final_max_altitude_meters=final_max_altitude_meters,
        final_world_edge_fade_fraction=0.01,
        fade_fraction=0.2,
        fade_noise_scale=fade_noise_scale,
        fade_noise_min=0.1,
        # made most worlds circular because it's prettier :-P
        is_fade_circular=rand.uniform() < 0.75,
    )
    return config
