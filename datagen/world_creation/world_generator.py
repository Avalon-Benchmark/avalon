import shutil
import time
from multiprocessing import Lock
from multiprocessing import Pool
from pathlib import Path
from random import Random
from typing import Final
from typing import Hashable
from typing import List
from typing import Optional
from typing import Tuple

import attr
import numpy as np

from common.even_earlier_hacks import is_notebook
from common.log_utils import logger
from contrib.serialization import Serializable
from datagen.world_creation.constants import TASKS_BY_TASK_GROUP
from datagen.world_creation.constants import AvalonTask
from datagen.world_creation.constants import AvalonTaskGroup
from datagen.world_creation.constants import get_all_tasks_for_task_groups
from datagen.world_creation.heightmap import get_agent_export_config
from datagen.world_creation.tasks.avoid import generate_avoid_task
from datagen.world_creation.tasks.bridge import generate_bridge_task
from datagen.world_creation.tasks.carry import generate_carry_task
from datagen.world_creation.tasks.climb import generate_climb_task
from datagen.world_creation.tasks.descend import generate_descend_task
from datagen.world_creation.tasks.eat import generate_eat_task
from datagen.world_creation.tasks.explore import generate_explore_task
from datagen.world_creation.tasks.fight import generate_fight_task
from datagen.world_creation.tasks.find import generate_find_task
from datagen.world_creation.tasks.gather import generate_gather_task
from datagen.world_creation.tasks.hunt import generate_hunt_task
from datagen.world_creation.tasks.jump import generate_jump_task
from datagen.world_creation.tasks.move import generate_move_task
from datagen.world_creation.tasks.navigate import generate_navigate_task
from datagen.world_creation.tasks.open import generate_open_task
from datagen.world_creation.tasks.push import generate_push_task
from datagen.world_creation.tasks.scramble import generate_scramble_task
from datagen.world_creation.tasks.stack import generate_stack_task
from datagen.world_creation.tasks.survive import generate_survive_task
from datagen.world_creation.tasks.throw import generate_throw_task
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import select_categorical_difficulty
from datagen.world_creation.utils import ImpossibleWorldError


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class GenerateWorldParams(Serializable):
    task: AvalonTask
    difficulty: float
    seed: int
    index: int
    output: str
    # this parameter is a trade-off between how quickly you will hear about a task generator that is
    # totally busted (lower values) vs the probability that your process crashed because you just got unlucky
    # (higher values). Can set to inf if you're sure there are no bugs and you definitely want it to run until it finds a level...
    num_retries: int = 5

    @property
    def output_path(self) -> Path:
        return Path(self.output)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class GeneratedWorldParams(GenerateWorldParams):
    # TODO rename?
    starting_hit_points: float = 1.0

    @classmethod
    def from_input(cls, input_params: GenerateWorldParams, generation_function_result: TaskGenerationFunctionResult):
        return cls(**attr.asdict(input_params), starting_hit_points=generation_function_result.starting_hit_points)


class WorldGenerator:
    def __init__(self, output_path: Path, seed: int):
        self._base_output_path = output_path
        self.output_path = output_path / f"world_seed_{seed}"
        self._seed = seed

    def generate_batch(self, start_world_id: Optional[int], batch_size: int = 1) -> List[GeneratedWorldParams]:
        """
        :returns: a (batch_size)-long list of folders (one folder per scene).
        The folder can be assumed to contained a "main.tscn" file
        """
        raise NotImplementedError()

    def sample_tasks(self, n_tasks: int) -> List[Hashable]:
        """Samples task of the meta-environment

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        raise NotImplementedError()

    def set_task(self, task: Hashable) -> None:
        """Sets the specified task to the current environment

        Args:
            task: task of the meta-learning environment
        """
        raise NotImplementedError()

    def get_task(self) -> Hashable:
        """Gets the task that the agent is performing in the current environment

        Returns:
            task: task of the meta-learning environment
        """
        raise NotImplementedError()

    def set_seed(self, seed):
        self._seed = seed
        self.output_path = self._base_output_path / f"world_seed_{seed}"


class SingleTaskWorldGenerator(WorldGenerator):
    def __init__(self, output_path: Path, seed: int, difficulty: float, task: AvalonTask):
        super().__init__(output_path, seed)
        self.difficulty = difficulty
        self.task = task

    def generate_batch(self, start_world_id: int, batch_size: int = 1) -> List[GeneratedWorldParams]:
        """
        :returns: a (batch_size)-long list of folders (one folder per scene).
        The folder can be assumed to contained a "main.tscn" file
        """
        raise NotImplementedError()

    def sample_tasks(self, n_tasks: int) -> List[int]:
        return [0] * n_tasks

    def set_task(self, task: Hashable) -> None:
        pass

    def get_task(self) -> int:
        return 0


class InsufficientBufferSize(Exception):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class WorldGeneratorTask:
    task: AvalonTask
    difficulty: float


def rand_task_with_difficulty(
    task_groups: Tuple[AvalonTaskGroup, ...], difficulty: float, rand: np.random.Generator
) -> AvalonTask:
    task_group, _difficulty = select_categorical_difficulty(task_groups, difficulty, rand)
    return rand.choice(TASKS_BY_TASK_GROUP[task_group])


class BlockingWorldGenerator(WorldGenerator):
    def __init__(
        self,
        output_path: Path,
        seed: int,
        start_difficulty: float,
        task_groups: Tuple[AvalonTaskGroup, ...],
    ):
        super().__init__(output_path, seed)
        self.difficulties = {x: start_difficulty for x in get_all_tasks_for_task_groups(task_groups)}
        self.task_groups = task_groups
        self.meta_difficulty = start_difficulty

    def _generate_world(self, i: int) -> GeneratedWorldParams:
        rand = np.random.default_rng([self._seed, i])
        task = rand_task_with_difficulty(self.task_groups, self.meta_difficulty, rand)

        return generate_world(
            GenerateWorldParams(
                task=task,
                difficulty=self.difficulties[task],
                seed=self._seed,
                index=i,
                output=str(self.output_path / str(i)),
            )
        )

    def generate_batch(self, start_world_id: int, batch_size: int = 1) -> List[GeneratedWorldParams]:
        return [self._generate_world(i) for i in range(start_world_id, start_world_id + batch_size)]

    def set_task(self, task: Hashable) -> None:
        assert isinstance(task, WorldGeneratorTask)
        self.task = task.task
        self.difficulties[task.task] = task.difficulty

    def set_task_difficulty(self, task, difficulty):
        self.difficulties[task] = difficulty

    def set_meta_difficulty(self, difficulty):
        self.meta_difficulty = difficulty

    def get_task(self) -> WorldGeneratorTask:
        assert self.task is not None
        return WorldGeneratorTask(task=self.task, difficulty=self.difficulties[self.task])

    def set_seed(self, seed):
        super().set_seed(seed)


class FixedWorldGenerator(WorldGenerator):
    """
    Generates num_unique_worlds and copies them to a working dir each time they are requested
    """

    def __init__(
        self,
        output_path: Path,
        seed: int,
        difficulties: Tuple[float, ...],
        task_groups: Tuple[AvalonTaskGroup, ...],
        num_worlds_per_task_difficulty_pair: int,
    ):
        super().__init__(output_path, seed)
        self.worlds: List[GeneratedWorldParams] = []
        index = 0
        for task in get_all_tasks_for_task_groups(task_groups):
            for difficulty in difficulties:
                # assert num_worlds_per_task_difficulty_pair == 1
                for _ in range(num_worlds_per_task_difficulty_pair):
                    self.worlds.append(
                        generate_world(
                            GenerateWorldParams(
                                task=task,
                                difficulty=difficulty,
                                seed=self._seed,
                                index=index,
                                output=str(self.output_path / str(index)),
                            )
                        )
                    )
                    index += 1

    def generate_batch(self, start_world_id: int, batch_size: int = 1) -> List[GeneratedWorldParams]:
        copied_worlds = []
        for i in range(start_world_id, start_world_id + batch_size):
            generated_world = self.worlds[i % len(self.worlds)]
            original_path = generated_world.output_path
            dest_path = original_path.with_stem(original_path.stem + "_working")
            shutil.copytree(str(original_path), str(dest_path), dirs_exist_ok=True)
            copied_worlds.append(attr.evolve(generated_world, output=str(dest_path)))
        return copied_worlds


# TODO: this introduces some non-determinism, fix it by anchoring to ids of levels generated
class LocalProcessWorldGenerator(WorldGenerator):
    def __init__(
        self,
        output_path: Path,
        seed: int,
        start_difficulty: float,
        task_groups: Tuple[AvalonTaskGroup, ...],
        num_workers: int = 4,
        buffer_size: int = 20,
        offset: int = 0,
        is_task_curriculum_used: bool = True,
    ):
        super().__init__(output_path, seed)
        self.task_groups = task_groups
        self.difficulties = {x: start_difficulty for x in get_all_tasks_for_task_groups(task_groups)}
        self.meta_difficulty = start_difficulty
        self.is_task_curriculum_used = is_task_curriculum_used
        self.rand = np.random.default_rng([self._seed])
        self.buffer: List[GeneratedWorldParams] = []
        self.lock = Lock()
        self.worker_pool = Pool(processes=num_workers)
        self.offset = offset
        self._request_batch(buffer_size)

        def signal_handler(sig, frame):
            self.close()
            raise KeyboardInterrupt()

        if is_notebook():
            pass
            # signal.signal(signal.SIGINT, signal_handler)

    def generate_batch(self, start_world_id: Optional[int], batch_size: int = 1) -> List[GeneratedWorldParams]:
        assert start_world_id is None
        self._request_batch(batch_size)
        while True:
            try:
                batch = self._pop_batch(batch_size)
                return batch
            except InsufficientBufferSize:
                time.sleep(0.1)

    def close(self):
        if self.worker_pool is not None:
            self._destroy_pool()

    def _destroy_pool(self):
        self.worker_pool.close()
        self.worker_pool.terminate()
        try:
            self.worker_pool.terminate()
        except AssertionError as e:
            if "cannot have cache with result_hander not alive" in str(e).lower():
                pass
            else:
                raise
        try:
            self.worker_pool.join()
        except RuntimeError as e:
            if "cannot join current thread" in str(e):
                pass
            else:
                raise
        self.worker_pool = None

    def _request_batch(self, batch_size: int):
        with self.lock:
            sampled_tasks = [
                rand_task_with_difficulty(self.task_groups, self.meta_difficulty, self.rand) for _ in range(batch_size)
            ]
            for idx in range(batch_size):
                i = idx + self.offset
                task = sampled_tasks[idx]
                output_dir = self.output_path / str(i)
                assert (
                    not output_dir.exists()
                ), f"{output_dir} already exists! Seed {self._seed} duplicated or path not cleared?"
                if self.is_task_curriculum_used:
                    difficulty = self.difficulties[task]
                else:
                    difficulty = self.rand.uniform()
                self.worker_pool.apply_async(
                    generate_world,
                    args=(
                        GenerateWorldParams(
                            task=task,
                            difficulty=difficulty,
                            seed=self._seed,
                            index=i,
                            output=str(output_dir),
                        ),
                    ),
                    callback=self._on_world_generation_done,
                    error_callback=self._on_world_generation_error,
                )
            self.offset += batch_size

    def _pop_batch(self, batch_size: int) -> List[GeneratedWorldParams]:
        with self.lock:
            if len(self.buffer) >= batch_size:
                result = self.buffer[:batch_size]
                self.buffer = self.buffer[batch_size:]
                return result
            raise InsufficientBufferSize()

    def _on_world_generation_done(self, result: GeneratedWorldParams):
        with self.lock:
            self.buffer.append(result)

    def _on_world_generation_error(self, error: BaseException):
        logger.error("Caught error in _on_world_generation_error. This shouldn't happen!!!:")
        logger.error(error)

    def set_task_difficulty(self, task, difficulty):
        with self.lock:
            self.difficulties[task] = difficulty

    def set_meta_difficulty(self, difficulty):
        with self.lock:
            self.meta_difficulty = difficulty

    def set_seed(self, seed):
        super().set_seed(seed)
        self.rand = Random(f"seed:{self._seed}")


_GENERATION_FUNCTION_BY_TASK: Final = {
    AvalonTask.EAT: generate_eat_task,
    AvalonTask.MOVE: generate_move_task,
    AvalonTask.JUMP: generate_jump_task,
    AvalonTask.CLIMB: generate_climb_task,
    AvalonTask.DESCEND: generate_descend_task,
    AvalonTask.STACK: generate_stack_task,
    AvalonTask.BRIDGE: generate_bridge_task,
    AvalonTask.PUSH: generate_push_task,
    AvalonTask.THROW: generate_throw_task,
    AvalonTask.HUNT: generate_hunt_task,
    AvalonTask.FIGHT: generate_fight_task,
    AvalonTask.AVOID: generate_avoid_task,
    AvalonTask.NAVIGATE: generate_navigate_task,
    AvalonTask.FIND: generate_find_task,
    AvalonTask.EXPLORE: generate_explore_task,
    AvalonTask.SURVIVE: generate_survive_task,
    AvalonTask.OPEN: generate_open_task,
    AvalonTask.CARRY: generate_carry_task,
    AvalonTask.GATHER: generate_gather_task,
    AvalonTask.SCRAMBLE: generate_scramble_task,
}

MAX_NOISE_SCALE = 0.3


def generate_world(params: GenerateWorldParams) -> GeneratedWorldParams:
    output_path = Path(params.output)
    output_path.mkdir(parents=True)
    rand = np.random.default_rng([params.seed, params.index])
    export_config = get_agent_export_config()
    generation_function = _GENERATION_FUNCTION_BY_TASK[params.task]
    for i in range(params.num_retries):
        try:
            generation_function_result = generation_function(rand, params.difficulty, output_path, export_config)
            return GeneratedWorldParams.from_input(params, generation_function_result)
        except ImpossibleWorldError as e:
            if i == params.num_retries - 1:
                logger.error("Ran out of retries to generate a good world!")
                logger.error(f"Params were: {params}")
                raise
            else:
                logger.info(f"Impossible world was generated, trying again... (reason: {e})")
                # clear the output
                shutil.rmtree(output_path)
                output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            if i == params.num_retries - 1:
                logger.error("Ran out of retries to generate a good world!")
                logger.error(f"Params were: {params}")
                raise
            else:
                logger.error(f"Unspecified error in world generation, trying again... (reason: {e})")
                # clear the output
                shutil.rmtree(output_path)
                output_path.mkdir(parents=True, exist_ok=True)
