import os
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Dict
from typing import Final
from typing import Generic
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar

import attr
import numpy as np
from loguru import logger

from avalon.common.log_utils import configure_parent_logging
from avalon.common.log_utils import logger
from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.world_creation.configs.export import get_eval_agent_export_config
from avalon.datagen.world_creation.configs.export import get_oculus_export_config
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.entities.animals import ALL_PREDATOR_CLASSES
from avalon.datagen.world_creation.entities.animals import ALL_PREY_CLASSES
from avalon.datagen.world_creation.entities.food import FOODS
from avalon.datagen.world_creation.tasks.eat import ForcedFood
from avalon.datagen.world_creation.tasks.fight import ForceFight
from avalon.datagen.world_creation.tasks.hunt import ForceHunt
from avalon.datagen.world_creation.types import DebugVisualizationConfig
from avalon.datagen.world_creation.world_generator import GENERATION_FUNCTION_BY_TASK
from avalon.datagen.world_creation.world_generator import GenerateAvalonWorldParams
from avalon.datagen.world_creation.worlds.export import get_world_slug

ForcedParams = TypeVar("ForcedParams")


@attr.s(auto_attribs=True, collect_by_mro=True)
class _ForceGeneratedWorldCompleteness(Generic[ForcedParams]):
    """Ensures generated worlds include certain forced configurations to guarantee completeness."""

    forced_params: List[ForcedParams]
    difficulty: float

    @property
    def difficulties(self) -> List[float]:
        return [self.difficulty] * len(self.forced_params)


_DIFFICULT_WORLD_FRACTION = 0.2


_SPECIAL_COMPLETENESS_CONFIGS: Final[Dict[AvalonTask, _ForceGeneratedWorldCompleteness]] = {
    AvalonTask.FIGHT: _ForceGeneratedWorldCompleteness(
        difficulty=1.0,
        forced_params=[
            ForceFight(
                weapon_value=2,
                large_weapon_probability=0,
                rock_probability=1.0,
                predators=(predator, predator, predator),
            )
            for predator in ALL_PREDATOR_CLASSES
        ],
    ),
    AvalonTask.HUNT: _ForceGeneratedWorldCompleteness(
        difficulty=1.0,
        forced_params=[
            ForceHunt(
                weapon_value=2,
                large_weapon_probability=0,
                rock_probability=1.0,
                prey=prey,
            )
            for prey in ALL_PREY_CLASSES
        ],
    ),
    AvalonTask.EAT: _ForceGeneratedWorldCompleteness(
        difficulty=0.5,
        forced_params=[ForcedFood(food=x.__class__) for x in FOODS],
    ),
}


def _resolve_forced_difficulties_for_completeness(task: AvalonTask, is_practice: bool) -> List[float]:
    if is_practice or task not in _SPECIAL_COMPLETENESS_CONFIGS:
        return []
    return _SPECIAL_COMPLETENESS_CONFIGS[task].difficulties


def _resolve_forced_kwargs_for_completeness(task: AvalonTask, is_practice: bool, task_index: int) -> dict:
    if is_practice or task not in _SPECIAL_COMPLETENESS_CONFIGS:
        return {}

    forced_params_for_task = _SPECIAL_COMPLETENESS_CONFIGS[task].forced_params
    if task_index < len(forced_params_for_task):
        return {"_FORCED": forced_params_for_task[task_index]}

    return {}


def get_difficulties_for_task(task: AvalonTask, is_practice: bool, min_difficulty: float, num_worlds_per_task: int):
    forced_difficulties_for_completeness = _resolve_forced_difficulties_for_completeness(task, is_practice)
    remaining_world_count = num_worlds_per_task - len(forced_difficulties_for_completeness)
    if remaining_world_count <= 0:
        return forced_difficulties_for_completeness[:num_worlds_per_task]
    difficult_world_count = round(remaining_world_count * _DIFFICULT_WORLD_FRACTION)
    remaining_world_count -= difficult_world_count
    return (
        forced_difficulties_for_completeness
        + [round(x.item(), 2) for x in np.linspace(min_difficulty, 1.0, remaining_world_count)]
        + difficult_world_count * [1.0]
    )


class GeneratedWorld(NamedTuple):
    world_id: str
    task: AvalonTask
    difficulty: float
    time: float
    impossible_world_errors: List[Tuple[str, str]]
    unhandled_errors: List[Tuple[str, str]]
    was_successful: bool


def _generate_world(
    base_output_path: Path,
    task: AvalonTask,
    difficulty: float,
    seed: int,
    world_id: str,
    # only used by special task generators so that they can know when they're special
    task_idx: int,
    is_practice: bool,
    debug_visualization_config: Optional[DebugVisualizationConfig],
    is_generating_for_human: bool,
    max_retries: int = 10,
) -> GeneratedWorld:
    if is_generating_for_human:
        export_config = get_oculus_export_config(world_id)
    else:
        export_config = get_eval_agent_export_config()
    if debug_visualization_config:
        export_config = attr.evolve(export_config, is_tiled=False)
        export_config = attr.evolve(export_config, debug_visualization_config=debug_visualization_config)
    start_time = time.time()
    rand = np.random.default_rng(seed)

    output_path = base_output_path / world_id
    output_path.mkdir(parents=True, exist_ok=True)
    generation_function = GENERATION_FUNCTION_BY_TASK[task]

    impossible_world_errors = []
    unhandled_errors = []

    was_successful = True

    kwargs = _resolve_forced_kwargs_for_completeness(task, is_practice, task_idx)

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
    return GeneratedWorld(
        world_id=world_id,
        task=task,
        difficulty=difficulty,
        time=total_time,
        impossible_world_errors=impossible_world_errors,
        unhandled_errors=unhandled_errors,
        was_successful=was_successful,
    )


def generate_evaluation_worlds(
    base_output_path: Path,
    tasks: Sequence[AvalonTask],
    num_worlds_per_task: int,
    is_generating_for_human: bool,
    start_seed: int = 0,
    is_practice: bool = False,
    min_difficulty: float = 0.0,
    is_recreating: bool = True,
    num_workers: int = 10,
    debug_visualization_config: Optional[DebugVisualizationConfig] = None,
    is_async: bool = True,
    is_verbose: bool = True,
) -> List[GeneratedWorld]:
    """Generates worlds for evaluation and testing, either by models or humans.

    Ensures a special set of configurations are applied to guarantee a completeness of variety in certain tasks (disabled by is_practice).
    For instance, the generated set will include one of each food in EAT, a difficult FIGHT with each predator, etc.
    """
    assert (
        os.getenv("PYTHONHASHSEED", None) is not None
    ), f"PYTHONHASHSEED must be set for worlds to be generated deterministically."
    total_errors = 0
    all_impossible_world_errors = []
    all_unhandled_errors = []
    total_error_counter = [0]

    def on_done(result: GeneratedWorld) -> None:
        if is_verbose:
            logger.info(f"Finished generating {result.world_id}")
        if not result.was_successful:
            total_error_counter[0] = total_error_counter[0] + 1
        all_impossible_world_errors.extend(result.impossible_world_errors)
        all_unhandled_errors.extend(result.unhandled_errors)

    errors = []

    def on_error(error: BaseException) -> None:
        errors.append(error)
        logger.error("BAD: one of the level generators failed")

    start_time = time.time()
    results = []

    params_to_generate: List[GenerateAvalonWorldParams] = []
    current_seed = start_seed
    for task in tasks:
        difficulties = get_difficulties_for_task(task, is_practice, min_difficulty, num_worlds_per_task)
        task_name = task.value.lower()
        for i in range(num_worlds_per_task):
            difficulty = difficulties[i]
            world_id = get_world_slug(task_name, current_seed, difficulty, is_practice)
            params_to_generate.append(
                GenerateAvalonWorldParams(
                    seed=current_seed,
                    index=i,
                    difficulty=difficulty,
                    output=str(base_output_path / world_id),
                    task=task,
                )
            )
            current_seed += 1

    with Pool(processes=num_workers, initializer=configure_parent_logging) as worker_pool:
        requests = []
        for params in params_to_generate:
            world_id = params.output_path.name
            if not is_recreating and params.output_path.exists():
                continue

            args = (
                base_output_path,
                params.task,
                params.difficulty,
                params.seed,
                world_id,
                params.index,
                is_practice,
                debug_visualization_config,
                is_generating_for_human,
            )

            if is_async:
                request = worker_pool.apply_async(
                    _generate_world,
                    args=args,
                    callback=on_done,
                    error_callback=on_error,
                )
                requests.append(request)
            else:
                results.append(_generate_world(*args))

        for request in requests:
            request.wait()
            if request._success:  # type: ignore[attr-defined]
                results.append(request.get())
    total_num_worlds = num_worlds_per_task * len(tasks)

    if len(errors) > 0:
        raise errors[0]

    logger.info(
        f"Finished generating {total_num_worlds - total_errors} of {total_num_worlds} worlds in {(time.time() - start_time) / 60:.2f} minutes"
    )
    for err in all_impossible_world_errors:
        logger.info(f"Failed with impossible world error:\n{err}")

    for err in all_unhandled_errors:
        logger.warning(f"Failed with unhandled error:\n{err}")

    return results
