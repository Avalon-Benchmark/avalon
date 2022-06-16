import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import attr
import numpy as np

from datagen.world_creation.constants import AvalonTask
from datagen.world_creation.heightmap import DebugVisualizationConfig
from datagen.world_creation.heightmap import get_oculus_export_config
from datagen.world_creation.items import ALL_PREDATOR_CLASSES
from datagen.world_creation.items import ALL_PREY_CLASSES
from datagen.world_creation.items import FOODS
from datagen.world_creation.tasks.compositional import ForcedComposition
from datagen.world_creation.tasks.eat import ForcedFood
from datagen.world_creation.tasks.fight import ForceFight
from datagen.world_creation.tasks.hunt import ForceHunt
from datagen.world_creation.utils import ImpossibleWorldError
from datagen.world_creation.world_generator import _GENERATION_FUNCTION_BY_TASK

_BASE_FIGHT_DATA = ForceFight(
    weapon_value=2,
    large_weapon_probability=0,
    rock_probability=1.0,
)
_SPECIAL_FIGHT_PARAMS = [attr.evolve(_BASE_FIGHT_DATA, predators=(x, x, x)) for x in ALL_PREDATOR_CLASSES]
_BASE_HUNT_DATA = ForceHunt(
    weapon_value=2,
    large_weapon_probability=0,
    rock_probability=1.0,
)
_SPECIAL_HUNT_PARAMS = [attr.evolve(_BASE_HUNT_DATA, prey=x) for x in ALL_PREY_CLASSES]
_SPECIAL_FOOD_PARAMS = [ForcedFood(food=x.__class__) for x in FOODS]
_DIFFICULT_WORLD_FRACTION = 0.2


def get_difficulties_for_task(task: AvalonTask, is_practice: bool, min_difficulty: float, num_worlds_per_task: int):
    forced_worlds = []
    if not is_practice:
        if task == AvalonTask.FIGHT:
            forced_worlds = [1.0 for x in range(len(_SPECIAL_FIGHT_PARAMS))]
        if task == AvalonTask.HUNT:
            forced_worlds = [1.0 for x in range(len(_SPECIAL_HUNT_PARAMS))]
        if task == AvalonTask.EAT:
            forced_worlds = [0.5 for x in range(len(_SPECIAL_FOOD_PARAMS))]
    remaining_world_count = num_worlds_per_task - len(forced_worlds)
    if remaining_world_count <= 0:
        return forced_worlds[:num_worlds_per_task]
    difficult_world_count = round(remaining_world_count * _DIFFICULT_WORLD_FRACTION)
    remaining_world_count -= difficult_world_count
    return (
        forced_worlds
        + [round(x.item(), 2) for x in np.linspace(min_difficulty, 1.0, remaining_world_count)]
        + difficult_world_count * [1.0]
    )


def generate_world(
    base_output_path: Path,
    task: AvalonTask,
    difficulty: float,
    seed: int,
    world_id: str,
    # only used by special task generators so that they can know when they're special
    task_idx: int,
    is_practice: bool,
    debug_visualization_config: Optional[DebugVisualizationConfig],
    max_retries: int = 10,
) -> Dict:
    export_config = get_oculus_export_config(world_id)
    if debug_visualization_config:
        export_config = attr.evolve(export_config, num_tiles=1)
        export_config = attr.evolve(export_config, debug_visualization_config=debug_visualization_config)
    start_time = time.time()
    rand = np.random.default_rng(seed)

    output_path = base_output_path / world_id
    output_path.mkdir(parents=True, exist_ok=True)
    generation_function = _GENERATION_FUNCTION_BY_TASK[task]

    impossible_world_errors = []
    unhandled_errors = []

    was_successful = True

    force = None
    if not is_practice:
        if task == AvalonTask.FIGHT:
            if task_idx < len(_SPECIAL_FIGHT_PARAMS):
                force = _SPECIAL_FIGHT_PARAMS[task_idx]
        if task == AvalonTask.HUNT:
            if task_idx < len(_SPECIAL_HUNT_PARAMS):
                force = _SPECIAL_HUNT_PARAMS[task_idx]
        if task == AvalonTask.EAT:
            if task_idx < len(_SPECIAL_FOOD_PARAMS):
                force = _SPECIAL_FOOD_PARAMS[task_idx]
        if task in (AvalonTask.GATHER, AvalonTask.FIND, AvalonTask.NAVIGATE):
            force = ForcedComposition(is_enabled=True)
    kwargs = dict(_FORCED=force) if force else {}

    i = 0
    while i < max_retries:
        try:
            generation_function(rand, difficulty, output_path, export_config, **kwargs)
            break
        except ImpossibleWorldError as e:
            impossible_world_errors.append((world_id, str(e)))
        except Exception as e:
            unhandled_errors.append((world_id, str(e)))
            traceback.print_exc()
            raise
        i += 1

    if i >= max_retries:
        was_successful = False

    end_time = time.time()
    total_time = end_time - start_time
    return dict(
        world_id=world_id,
        task=task,
        difficulty=difficulty,
        time=total_time,
        impossible_world_errors=impossible_world_errors,
        unhandled_errors=unhandled_errors,
        was_successful=was_successful,
    )


def generate_worlds(
    base_output_path: Path,
    tasks: List[AvalonTask],
    num_worlds_per_task: int,
    start_seed: int = 0,
    is_practice: bool = False,
    min_difficulty: float = 0.0,
    is_recreating: bool = True,
    num_workers: int = 10,
    is_using_constraints: bool = False,
    debug_visualization_config: Optional[DebugVisualizationConfig] = None,
    is_async: bool = True,
):
    total_errors = 0
    all_impossible_world_errors = []
    all_unhandled_errors = []
    total_error_counter = [0]

    def on_done(result):
        print(f"Finished generating {result['world_id']}")
        if not result["was_successful"]:
            total_error_counter[0] = total_error_counter[0] + 1
        all_impossible_world_errors.extend(result["impossible_world_errors"])
        all_unhandled_errors.extend(result["unhandled_errors"])

    errors = []

    def on_error(error: BaseException):
        errors.append(error)
        print("BAD: one of the level generators failed")

    start_time = time.time()
    results = []

    with Pool(processes=num_workers) as worker_pool:
        requests = []
        curr_seed = start_seed
        for task in tasks:
            difficulties = get_difficulties_for_task(task, is_practice, min_difficulty, num_worlds_per_task)
            for i in range(num_worlds_per_task):
                difficulty = difficulties[i]
                curr_seed += 1
                seed = curr_seed

                task_name = task.value.lower()
                difficulty_str = str(difficulty).replace(".", "_")
                if is_practice:
                    world_id = f"practice__{task_name}__{seed}__{difficulty_str}"
                else:
                    world_id = f"{task_name}__{seed}__{difficulty_str}"

                output_path = base_output_path / world_id

                if not is_recreating and output_path.exists():
                    continue

                if is_async:
                    request = worker_pool.apply_async(
                        generate_world,
                        args=(
                            base_output_path,
                            task,
                            difficulty,
                            seed,
                            world_id,
                            i,
                            is_practice,
                            debug_visualization_config,
                        ),
                        callback=on_done,
                        error_callback=on_error,
                    )
                    requests.append(request)
                else:
                    generate_world(
                        base_output_path,
                        task,
                        difficulty,
                        seed,
                        world_id,
                        i,
                        is_practice,
                        debug_visualization_config,
                    )
        for request in requests:
            request.wait()
            if request._success:
                results.append(request.get())
    total_num_worlds = num_worlds_per_task * len(tasks)

    if len(errors) > 0:
        raise errors[0]

    print(
        f"Finished generating {total_num_worlds - total_errors} of {total_num_worlds} worlds in {(time.time() - start_time) / 60:.2f} minutes"
    )
    for err in all_impossible_world_errors:
        print("Failed with impossible world error:")
        print(err)

    for err in all_unhandled_errors:
        print("Failed with unhandled error:")
        print(err)

    return results
