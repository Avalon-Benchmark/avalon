from pathlib import Path
from typing import Optional
from typing import Tuple

import numpy as np
from scipy import stats

from common.errors import SwitchError
from datagen.world_creation.constants import BOX_HEIGHT
from datagen.world_creation.constants import JUMPING_REQUIRED_HEIGHT
from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.indoor_task_generators import BuildingTask
from datagen.world_creation.indoor_task_generators import create_building_obstacle
from datagen.world_creation.indoor_task_generators import get_radius_for_building_task
from datagen.world_creation.indoor_task_generators import make_indoor_task_world
from datagen.world_creation.items import Placeholder
from datagen.world_creation.items import Stone
from datagen.world_creation.items import Tool
from datagen.world_creation.new_world import HeightSolution
from datagen.world_creation.new_world import NewWorld
from datagen.world_creation.tasks.climb import replace_placeholder_with_goal
from datagen.world_creation.tasks.compositional_types import CompositionalConstraint
from datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from datagen.world_creation.tasks.task_worlds import create_world_from_constraint
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import add_offsets
from datagen.world_creation.tasks.utils import difficulty_variation
from datagen.world_creation.tasks.utils import export_skill_world
from datagen.world_creation.tasks.utils import make_ring
from datagen.world_creation.tasks.utils import normal_distrib_range
from datagen.world_creation.tasks.utils import scale_with_difficulty
from datagen.world_creation.tasks.utils import select_categorical_difficulty
from datagen.world_creation.tasks.utils import starting_hit_points_from_difficulty
from datagen.world_creation.world_location_data import WorldLocationData


def generate_stack_task(
    rand: np.random.Generator, difficulty: float, output_path: Path, export_config: ExportConfig
) -> TaskGenerationFunctionResult:
    is_indoor = rand.uniform() < 0.2
    # is_indoor = False
    if is_indoor:
        building_radius = get_radius_for_building_task(rand, BuildingTask.STACK, difficulty)
        building, extra_items, spawn_location, target_location = create_building_obstacle(
            rand, difficulty, BuildingTask.STACK, building_radius, location=np.array([0, 2, 0]), yaw_radians=0.0
        )
        make_indoor_task_world(
            building, extra_items, difficulty, spawn_location, target_location, output_path, rand, export_config
        )
    else:
        world, locations, difficulty = create_stack_obstacle(rand, difficulty, export_config)
        world.end_height_obstacles(locations, is_accessible_from_water=False, is_spawn_region_climbable=False)
        add_food_tree_for_simple_task(world, locations)
        world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
        export_skill_world(output_path, rand, world)

    return TaskGenerationFunctionResult(starting_hit_points_from_difficulty(difficulty))


def create_stack_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
    is_for_carry: bool = False,
) -> Tuple[NewWorld, WorldLocationData, float]:

    stack_height, difficulty = select_categorical_difficulty([1, 2, 3], difficulty, rand)
    spare_boxes, difficulty = select_categorical_difficulty([3, 2, 1, 0], difficulty, rand)

    base_height = JUMPING_REQUIRED_HEIGHT - 0.5
    desired_height = stack_height * BOX_HEIGHT + base_height

    # how far to jiggle the boxes from their "solved" configuration
    randomization_dist = scale_with_difficulty(difficulty, 0.0, 6.0)

    # you aren't technically going to have to move this far--the food gets updated to be close to the cliff edge
    # this mostly just gives a bit of space for the climb paths
    base_dist = 2.0 * desired_height + 2.0 * randomization_dist
    desired_goal_dist = difficulty_variation(base_dist, 2.0 * base_dist, rand, difficulty)

    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, 0.5), rand, difficulty, export_config, constraint
    )

    # TODO: move the food a little ways away from the top of the ridge, raised up?

    cliff_space = 3.0
    # this ensures there is space at the base of the cliff
    if 2 * randomization_dist > locations.get_2d_spawn_goal_distance() - cliff_space:
        randomization_dist = (locations.get_2d_spawn_goal_distance() - cliff_space) / 2.0
    extra_safety_radius = randomization_dist

    ring_config = make_ring(
        rand,
        difficulty,
        world,
        locations,
        gap_distance=0.0,
        height=desired_height,
        constraint=constraint,
        traversal_width=normal_distrib_range(10.0, 1.0, 1.0, rand, difficulty),
        inner_traversal_length=0.0,
        is_single_obstacle=True,
        is_inside_climbable=False,
        inner_solution=HeightSolution(
            inside_items=_get_boxes(stack_height, spare_boxes),
            inside_item_randomization_distance=randomization_dist,
            inside_item_radius=0.5,
            solution_point_brink_distance=1.0,
            outside_items=tuple([Placeholder()]),
            outside_item_radius=1.0,
        ),
        extra_safety_radius=extra_safety_radius,
        probability_of_centering_on_spawn=0.0 if is_for_carry else 0.5,
    )
    world.add_height_obstacle(rand, ring_config, locations.island)

    # move the food to the edge so that it is visible
    new_locations = replace_placeholder_with_goal(locations, world)

    return world, new_locations, difficulty


def _get_boxes(stack_height: int, spare_boxes: int) -> Tuple[Tool, ...]:
    if stack_height == 1:
        boxes = [
            Stone(entity_id=0, position=np.array([0.0, 0.0, 0.0])),
        ]
    elif stack_height == 2:
        boxes = [
            Stone(entity_id=0, position=np.array([0.0, 0.0, 0.0])),
            Stone(entity_id=0, position=np.array([0.0, 1.0, 0.0])),
            Stone(entity_id=0, position=np.array([-1.0, 0.0, 0.0])),
        ]
    elif stack_height == 3:
        boxes = [
            Stone(entity_id=0, position=np.array([0.0, 0.0, 0.0])),
            Stone(entity_id=0, position=np.array([0.0, 1.0, 0.0])),
            Stone(entity_id=0, position=np.array([0.0, 2.0, 0.0])),
            Stone(entity_id=0, position=np.array([-1.0, 0.0, 0.0])),
            Stone(entity_id=0, position=np.array([-1.0, 1.0, 0.0])),
            Stone(entity_id=0, position=np.array([-2.0, 0.0, 0.0])),
        ]
    else:
        raise SwitchError(f"Unhandled number of boxes: {stack_height}")

    if spare_boxes >= 1:
        boxes.append(Stone(entity_id=0, position=np.array([0.0, 0.0, -1.0])))
    if spare_boxes >= 2:
        boxes.append(Stone(entity_id=0, position=np.array([0.0, 0.0, 1.0])))
    if spare_boxes >= 3:
        boxes.append(Stone(entity_id=0, position=np.array([-1.0, 0.0, 1.0])))
    assert spare_boxes <= 3

    return tuple(add_offsets(boxes))
