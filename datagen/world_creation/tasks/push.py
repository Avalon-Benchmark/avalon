from pathlib import Path
from typing import Optional
from typing import Tuple

import numpy as np
from scipy import stats

from common.log_utils import logger
from datagen.world_creation.constants import BOULDER_HEIGHT
from datagen.world_creation.constants import BOULDER_MAX_MASS
from datagen.world_creation.constants import BOULDER_MIN_MASS
from datagen.world_creation.constants import JUMPING_REQUIRED_HEIGHT
from datagen.world_creation.constants import MAX_FLAT_JUMP_DIST
from datagen.world_creation.constants import MIN_BRIDGE_DIST
from datagen.world_creation.constants import AvalonTask
from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.heightmap import HeightMode
from datagen.world_creation.indoor_task_generators import BuildingTask
from datagen.world_creation.indoor_task_generators import create_building_obstacle
from datagen.world_creation.indoor_task_generators import get_radius_for_building_task
from datagen.world_creation.indoor_task_generators import make_indoor_task_world
from datagen.world_creation.items import Boulder
from datagen.world_creation.new_world import HeightPath
from datagen.world_creation.new_world import HeightSolution
from datagen.world_creation.new_world import NewWorld
from datagen.world_creation.tasks.compositional_types import CompositionalConstraint
from datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from datagen.world_creation.tasks.task_worlds import create_world_from_constraint
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import add_offsets
from datagen.world_creation.tasks.utils import export_skill_world
from datagen.world_creation.tasks.utils import make_ring
from datagen.world_creation.tasks.utils import normal_distrib_range
from datagen.world_creation.tasks.utils import scale_with_difficulty
from datagen.world_creation.tasks.utils import select_boolean_difficulty
from datagen.world_creation.tasks.utils import starting_hit_points_from_difficulty
from datagen.world_creation.utils import WorldTooSmall
from datagen.world_creation.world_location_data import WorldLocationData


def generate_push_task(
    rand: np.random.Generator, difficulty: float, output_path: Path, export_config: ExportConfig
) -> TaskGenerationFunctionResult:
    is_indoor = rand.uniform() < 0.2
    # is_indoor = False
    if is_indoor:
        building_radius = get_radius_for_building_task(rand, BuildingTask.PUSH, difficulty)
        building, extra_items, spawn_location, target_location = create_building_obstacle(
            rand, difficulty, BuildingTask.PUSH, building_radius, location=np.array([0, 2, 0]), yaw_radians=0.0
        )
        make_indoor_task_world(
            building, extra_items, difficulty, spawn_location, target_location, output_path, rand, export_config
        )
    else:
        world, locations, difficulty = create_push_obstacle(rand, difficulty, export_config)
        world.end_height_obstacles(locations, is_accessible_from_water=False, is_spawn_region_climbable=False)
        add_food_tree_for_simple_task(world, locations)
        world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
        export_skill_world(output_path, rand, world)

    return TaskGenerationFunctionResult(starting_hit_points_from_difficulty(difficulty))


def create_push_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
) -> Tuple[NewWorld, WorldLocationData, float]:

    is_inside_climbable, difficulty = select_boolean_difficulty(difficulty, rand, final_prob=0.2)

    extra_safety_radius = 3.0
    min_dist = MIN_BRIDGE_DIST + 2 * extra_safety_radius
    desired_goal_dist = (min_dist * 3.0) + (min_dist * 4.0 * difficulty * rand.uniform())

    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, 0.5), rand, difficulty, export_config, constraint
    )

    desired_gap_distance = scale_with_difficulty(difficulty, min_dist, min_dist + MAX_FLAT_JUMP_DIST)
    gap_distance = world.get_critical_distance(locations, min_dist, desired_gap_distance)

    max_height = BOULDER_HEIGHT + JUMPING_REQUIRED_HEIGHT - 0.5
    min_height = JUMPING_REQUIRED_HEIGHT
    height = scale_with_difficulty(difficulty, min_height, max_height)

    if gap_distance is None:
        raise WorldTooSmall(AvalonTask.PUSH, min_dist, locations.get_2d_spawn_goal_distance())
    gap_distance -= 2 * extra_safety_radius
    if gap_distance < MIN_BRIDGE_DIST:
        raise WorldTooSmall(AvalonTask.PUSH, gap_distance, MIN_BRIDGE_DIST)
    logger.trace(f"Creating a {gap_distance} meter gap")
    randomization_dist = scale_with_difficulty(difficulty, 0.0, 10.0)
    boulder_mass = normal_distrib_range(BOULDER_MIN_MASS, BOULDER_MAX_MASS, 10, rand, difficulty)
    boulder = Boulder(entity_id=0, position=np.array([-1.0, 0.0, 0.0]), mass=boulder_mass)
    ring_config = make_ring(
        rand,
        difficulty,
        world,
        locations,
        gap_distance,
        constraint=constraint,
        height=-height,
        traversal_width=normal_distrib_range(10.0, 3.0, 1.0, rand, difficulty),
        is_inside_climbable=True,
        is_outside_climbable=False,
        dual_solution=HeightSolution(
            solution_point_brink_distance=1.0,
            inside_items=tuple(add_offsets([boulder])),
            inside_item_randomization_distance=randomization_dist,
            inside_item_radius=1.0,
            paths=(
                HeightPath(
                    is_solution_flattened=True,
                    is_path_restricted_to_land=False,
                    is_chasm_bottom_flattened=True,
                    is_height_affected=False,
                    width=randomization_dist * 2.0,
                    flattening_mode="min",
                ),
                HeightPath(
                    is_solution_flattened=False,
                    is_path_restricted_to_land=True,
                    is_chasm_bottom_flattened=False,
                    is_height_affected=False,
                    width=2.0,
                ),
            ),
        ),
        probability_of_centering_on_spawn=0.0 if constraint is None else None,
        height_mode=HeightMode.MIDPOINT_ABSOLUTE,
        expansion_meters=extra_safety_radius,
    )
    world.add_height_obstacle(rand, ring_config, locations.island)
    return world, locations, difficulty
