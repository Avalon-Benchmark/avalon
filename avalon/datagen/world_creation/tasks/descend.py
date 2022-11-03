from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import cast

import attr
import numpy as np
from scipy import stats

from avalon.common.utils import only
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import TaskConfig
from avalon.datagen.world_creation.constants import MAX_FALL_DISTANCE_TO_DIE
from avalon.datagen.world_creation.constants import MIN_FALL_DISTANCE_TO_CAUSE_DAMAGE
from avalon.datagen.world_creation.constants import UP_VECTOR
from avalon.datagen.world_creation.entities.food import FoodTreeBase
from avalon.datagen.world_creation.entities.tools.placeholder import Placeholder
from avalon.datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from avalon.datagen.world_creation.utils import to_2d_point
from avalon.datagen.world_creation.worlds.creation import create_world_from_constraint
from avalon.datagen.world_creation.worlds.difficulty import normal_distrib_range
from avalon.datagen.world_creation.worlds.difficulty import scale_with_difficulty
from avalon.datagen.world_creation.worlds.difficulty import select_boolean_difficulty
from avalon.datagen.world_creation.worlds.export import export_world
from avalon.datagen.world_creation.worlds.obstacles.configure import make_ring
from avalon.datagen.world_creation.worlds.obstacles.harmonics import EdgeConfig
from avalon.datagen.world_creation.worlds.obstacles.height_path import HeightPath
from avalon.datagen.world_creation.worlds.obstacles.height_solution import HeightSolution
from avalon.datagen.world_creation.worlds.types import CompositionalConstraint
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world_locations import WorldLocations


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class DescendTaskConfig(TaskConfig):
    # how much extra height (above and beyond the amount required to cause some damage) to add for difficulty 0.0 and
    # 1.0 respectively. Result will be normally distributed centered on that value proportional to difficulty, with
    # the given standard deviation
    extra_height_easy: float = 0.0
    extra_height_hard: float = MAX_FALL_DISTANCE_TO_DIE - MIN_FALL_DISTANCE_TO_CAUSE_DAMAGE + 5.0
    extra_height_std_dev: float = 4.0
    # how wide to make the climbable path at difficulty 0.0 and 1.0 respectively
    path_width_easy: float = 10.0
    path_width_hard: float = 1.0
    # how far away the food starts. Not a particularly important parameter for this task probably
    goal_distance_easy: float = 8.0
    goal_distance_hard: float = 16.0


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class SingleTaskDescendTaskConfig(DescendTaskConfig):
    # slightly different settings for the single task case because otherwise agents learn to grab on to trees when they
    # fall. Which, to be far, is a legit strategy for decreasing your likelihood of death when falling, but it's a
    # much worse strategy than just carefully climbing down...
    # make the height even larger in the single task case
    extra_height_hard: float = (MAX_FALL_DISTANCE_TO_DIE - MIN_FALL_DISTANCE_TO_CAUSE_DAMAGE + 5.0) * 2.0
    # make the tree spawn farther away
    goal_distance_hard: float = 32.0


def generate_descend_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: Optional[DescendTaskConfig] = None,
) -> None:
    world, locations, difficulty = create_descend_obstacle(rand, difficulty, export_config, task_config=task_config)
    world, locations = world.end_height_obstacles(
        locations, is_accessible_from_water=False, is_spawn_region_climbable=False
    )
    world = add_food_tree_for_simple_task(world, locations)
    world = world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    export_world(output_path, rand, world)


def create_descend_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
    task_config: Optional[DescendTaskConfig] = None,
) -> Tuple[World, WorldLocations, float]:

    if task_config is None:
        if constraint is None:
            task_config = SingleTaskDescendTaskConfig()
        else:
            task_config = DescendTaskConfig()

    is_platform_mode, difficulty = select_boolean_difficulty(difficulty, rand)
    is_everywhere_climbable, difficulty = select_boolean_difficulty(difficulty, rand)

    desired_goal_dist = scale_with_difficulty(
        difficulty, task_config.goal_distance_easy, task_config.goal_distance_hard
    )
    extra_fall_distance = normal_distrib_range(
        task_config.extra_height_easy,
        task_config.extra_height_hard,
        task_config.extra_height_std_dev,
        rand,
        difficulty,
    )

    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, 0.5), rand, difficulty, export_config, constraint
    )

    height_delta = MIN_FALL_DISTANCE_TO_CAUSE_DAMAGE + extra_fall_distance
    climbable_path_width = normal_distrib_range(
        task_config.path_width_easy, task_config.path_width_hard, 0.5, rand, difficulty
    )

    ring_config = make_ring(
        rand,
        difficulty,
        world,
        locations,
        constraint=constraint,
        gap_distance=0.0,
        height=-height_delta,
        traversal_width=2.0,
        inner_traversal_length=0.0,
        is_single_obstacle=True,
        is_inside_climbable=is_everywhere_climbable,
        is_outside_climbable=is_everywhere_climbable,
        inner_solution=HeightSolution(
            paths=(
                HeightPath(
                    width=climbable_path_width,
                    is_path_climbable=True,
                    is_height_affected=False,
                    is_path_restricted_to_land=False,
                ),
                # this is really here to prevent the task from going over water, but without needing such a huge
                # area to be restricted to land-only (as the above, actually important path with defines where
                # things can be climbed)
                HeightPath(
                    width=1.0,
                    is_path_climbable=False,
                    is_height_affected=False,
                    is_path_restricted_to_land=True,
                ),
            ),
            # this will be replaced with the food below
            inside_items=tuple([Placeholder(position=np.array([0.0, 0.0, 0.0]))]),
            # so that the food ends up away from the edge a little bit
            inside_item_radius=0.75,
            solution_point_brink_distance=1.0,
        ),
        probability_of_centering_on_spawn=1.0,
    )
    ring_config = attr.evolve(ring_config, edge=EdgeConfig(noise=0.25))
    world = world.add_height_obstacle(rand, ring_config, locations.island)

    if constraint is None:
        new_locations = locations
        # # flatten so that you dont fall off where you spawn
        map_new = world.map.copy()
        map_new.radial_flatten(to_2d_point(new_locations.spawn), 2.0 * 2.0, extra_mask=new_locations.island)
        world = attr.evolve(world, map=map_new)
    else:
        # reset the location of the spawn to be where the placeholder ended up
        # height will be reset below
        new_locations = attr.evolve(
            locations, spawn=only([x for x in world.items if isinstance(x, Placeholder)]).position
        )
    new_items = tuple([x for x in world.items if not isinstance(x, Placeholder)])
    world = attr.evolve(world, items=new_items)

    # move the food height down
    down_vector = -1.0 * UP_VECTOR
    new_locations = attr.evolve(new_locations, goal=new_locations.goal + down_vector * height_delta)

    # if this is platform mode, make a platform to fall on so it's easier
    if is_platform_mode:
        # find the cliff edges
        sq_slope = world.map.get_squared_slope()
        cliff_edges = sq_slope > ((height_delta * 0.8) ** 2)
        island_edges = world.map.get_outline(locations.island, 3)
        cliff_edges = np.logical_and(cliff_edges, np.logical_not(island_edges))
        cliff_edges = np.logical_and(cliff_edges, locations.island)
        cliff_movement_distance = scale_with_difficulty(difficulty, 1.0, 6.0)
        for i in range(2):
            nearby = world.map.get_dist_sq_to(to_2d_point(new_locations.spawn)) < cliff_movement_distance**2
            cliff_edges = np.logical_and(cliff_edges, nearby)
            if np.any(cliff_edges):
                cliff_indices = cast(Tuple[int, int], tuple(rand.choice(np.argwhere(cliff_edges))))
                cliff_point_2d = world.map.index_to_point_2d(cliff_indices)
                cliff_position = np.array(
                    [cliff_point_2d[0], new_locations.spawn[1] - height_delta / 2.0, cliff_point_2d[1]]
                )
                cliff = FoodTreeBase(position=cliff_position)
                world = world.add_item(cliff)
                break

    if constraint is None:
        # finally, have to remove trees on harder difficulties so the agent doesn't learn to grab them to break its fall
        new_foliage_density_modifier = (
            scale_with_difficulty(difficulty, 1.0, 0.0) * world.biome_config.foliage_density_modifier
        )
        new_biome_config = attr.evolve(world.biome_config, foliage_density_modifier=new_foliage_density_modifier)
        world = attr.evolve(world, biome_config=new_biome_config)

    return world, new_locations, difficulty
