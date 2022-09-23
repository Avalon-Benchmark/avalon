import multiprocessing as mp
import shutil
import signal
import time
import uuid
import warnings
from multiprocessing import Lock
from multiprocessing import Pool
from pathlib import Path
from random import Random
from typing import Any
from typing import Dict
from typing import Final
from typing import Hashable
from typing import List
from typing import Optional
from typing import Protocol
from typing import Tuple

import attr
import numpy as np

from avalon.common.errors import SwitchError
from avalon.common.log_utils import logger
from avalon.contrib.serialization import Serializable
from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.export import get_agent_export_config
from avalon.datagen.world_creation.configs.task import TaskConfig
from avalon.datagen.world_creation.constants import TASKS_BY_TASK_GROUP
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.constants import AvalonTaskGroup
from avalon.datagen.world_creation.constants import get_all_tasks_for_task_groups
from avalon.datagen.world_creation.tasks.avoid import generate_avoid_task
from avalon.datagen.world_creation.tasks.bridge import generate_bridge_task
from avalon.datagen.world_creation.tasks.carry import generate_carry_task
from avalon.datagen.world_creation.tasks.climb import generate_climb_task
from avalon.datagen.world_creation.tasks.descend import generate_descend_task
from avalon.datagen.world_creation.tasks.eat import generate_eat_task
from avalon.datagen.world_creation.tasks.explore import generate_explore_task
from avalon.datagen.world_creation.tasks.fight import generate_fight_task
from avalon.datagen.world_creation.tasks.find import generate_find_task
from avalon.datagen.world_creation.tasks.gather import generate_gather_task
from avalon.datagen.world_creation.tasks.hunt import generate_hunt_task
from avalon.datagen.world_creation.tasks.jump import generate_jump_task
from avalon.datagen.world_creation.tasks.move import generate_move_task
from avalon.datagen.world_creation.tasks.navigate import generate_navigate_task
from avalon.datagen.world_creation.tasks.open import generate_open_task
from avalon.datagen.world_creation.tasks.push import generate_push_task
from avalon.datagen.world_creation.tasks.scramble import generate_scramble_task
from avalon.datagen.world_creation.tasks.stack import generate_stack_task
from avalon.datagen.world_creation.tasks.survive import generate_survive_task
from avalon.datagen.world_creation.tasks.throw import generate_throw_task
from avalon.datagen.world_creation.worlds.difficulty import select_categorical_difficulty

DEFAULT_WORLD_ID: Final = 0


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
    @classmethod
    def from_input(cls, input_params: GenerateWorldParams) -> "GeneratedWorldParams":
        return cls(**attr.asdict(input_params))


class WorldGenerator:
    def __init__(self, base_path: Path, seed: int):
        self.output_path = base_path / str(uuid.uuid4())
        self.output_path.mkdir(parents=True, exist_ok=False)
        self.set_seed(seed)

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

    def set_seed(self, seed: int):
        self._seed = seed

    def close(self):
        shutil.rmtree(self.output_path)


class SingleTaskWorldGenerator(WorldGenerator):
    def __init__(self, base_path: Path, seed: int, difficulty: float, task: AvalonTask):
        super().__init__(base_path, seed)
        self.difficulty = difficulty
        self.task = task

    def generate_batch(self, start_world_id: Optional[int], batch_size: int = 1) -> List[GeneratedWorldParams]:
        """
        :returns: a (batch_size)-long list of folders (one folder per scene).
        The folder can be assumed to contained a "main.tscn" file
        """
        raise NotImplementedError()

    def sample_tasks(self, n_tasks: int) -> List[Hashable]:
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
    return rand.choice(TASKS_BY_TASK_GROUP[task_group])  # type: ignore


class BlockingWorldGenerator(WorldGenerator):
    def __init__(
        self,
        base_path: Path,
        seed: int,
        start_difficulty: float,
        task_groups: Tuple[AvalonTaskGroup, ...],
    ):
        super().__init__(base_path, seed)
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

    def generate_batch(self, start_world_id: Optional[int], batch_size: int = 1) -> List[GeneratedWorldParams]:
        if start_world_id is None:
            start_world_id = DEFAULT_WORLD_ID
        return [self._generate_world(i) for i in range(start_world_id, start_world_id + batch_size)]

    def set_task(self, task: Hashable) -> None:
        assert isinstance(task, WorldGeneratorTask)
        self.task = task.task
        self.difficulties[task.task] = task.difficulty

    def set_task_difficulty(self, task: AvalonTask, difficulty: float):
        self.difficulties[task] = difficulty

    def set_meta_difficulty(self, difficulty: float):
        self.meta_difficulty = difficulty

    def get_task(self) -> WorldGeneratorTask:
        assert self.task is not None
        return WorldGeneratorTask(task=self.task, difficulty=self.difficulties[self.task])

    def set_seed(self, seed: int):
        super().set_seed(seed)


def get_world_params_for_task_groups(
    task_groups: Tuple[AvalonTaskGroup, ...],
    difficulties: Tuple[float, ...],
    output_path: Path,
    seed: int,
) -> List[GenerateWorldParams]:
    worlds: List[GenerateWorldParams] = []
    index = 0
    for task in get_all_tasks_for_task_groups(task_groups):
        for difficulty in difficulties:
            worlds.append(
                GenerateWorldParams(
                    task=task,
                    difficulty=difficulty,
                    seed=seed,
                    index=index,
                    output=str(output_path / str(index)),
                )
            )
            index += 1

    return worlds


def generate_fixed_worlds(
    world_params: List[GenerateWorldParams],
    num_processes: int = 4,
) -> Dict[int, GeneratedWorldParams]:
    worlds: Dict[int, GeneratedWorldParams] = {}
    index = 0

    ctx = mp.get_context("spawn")

    def on_done(result: GeneratedWorldParams):
        # logger.info(f"Finished generating {result}")
        worlds[result.index] = result

    def on_error(error: BaseException):
        logger.error(f"World generation failed! {error}")
        raise error

    with ctx.Pool(processes=min(len(world_params), num_processes)) as worker_pool:
        for params in world_params:
            worker_pool.apply_async(
                generate_world,
                kwds={"params": params},
                callback=on_done,
                error_callback=on_error,
            )
        worker_pool.close()
        while ctx.active_children():
            # For some reason, when using the PyCharm debugger, it hangs on `join()`, so I added this block.
            # Note: this might break if this process has other children? Not the case currently.
            if len(worlds) == len(world_params):
                worker_pool.terminate()
            time.sleep(1)
        worker_pool.join()

    return worlds


class FixedWorldGenerator(WorldGenerator):
    """
    Generates num_unique_worlds and copies them to a working dir each time they are requested.
    If `load_levels_from_path` is set, load worlds from this path instead of generating them.
    """

    def __init__(
        self,
        base_path: Path,
        seed: int,
        difficulties: Tuple[float, ...],
        task_groups: Tuple[AvalonTaskGroup, ...],
        num_generators: int = 1,
        generator_index: int = 0,
        load_levels_from_path: Optional[Path] = None,
    ):
        super().__init__(base_path, seed)
        self.base_path = base_path
        if load_levels_from_path is None:
            self._init_by_generating(difficulties, task_groups, num_generators, generator_index)
        else:
            self._init_by_loading(load_levels_from_path, num_generators, generator_index)

    def _init_by_loading(self, load_path: Path, num_generators: int = 1, generator_index: int = 0):
        world_paths = sorted(path for path in load_path.iterdir() if not str(path).startswith("practice"))
        selected_list_indices_and_paths = [
            (i, path) for i, path in enumerate(world_paths) if i % num_generators == generator_index
        ]

        assert len(selected_list_indices_and_paths) > 0
        self.worlds: Dict[int, GeneratedWorldParams] = {}
        for list_index, world_path in selected_list_indices_and_paths:
            new_world_path = self.output_path / world_path.name
            shutil.copytree(world_path, new_world_path)
            world_name_parts = new_world_path.name.split("__")
            task = AvalonTask(world_name_parts[0].upper())
            world_index = int(world_name_parts[1])
            if world_index in self.worlds:
                raise Exception("Cannot have two worlds with the same index!")
            difficulty = float(world_name_parts[2].replace("_", "."))

            world = GeneratedWorldParams(
                task=task,
                difficulty=difficulty,
                seed=world_index,
                index=world_index,
                output=str(new_world_path),
            )
            self.worlds[world_index] = world

    def _init_by_generating(
        self,
        difficulties: Tuple[float, ...],
        task_groups: Tuple[AvalonTaskGroup, ...],
        num_generators: int = 1,
        generator_index: int = 0,
    ):
        world_params = get_world_params_for_task_groups(
            task_groups,
            difficulties,
            self.output_path,
            self._seed,
        )
        # each generator will be responsible for worlds where index % num_generators == generator_index
        # TODO: should (deterministically) shuffle these to ensure an unbiased split (in length of episodes)
        world_params = [x for x in world_params if x.index % num_generators == generator_index]
        assert len(world_params) > 0
        self.worlds = generate_fixed_worlds(world_params)

    def generate_batch(self, start_world_id: Optional[int], batch_size: int = 1) -> List[GeneratedWorldParams]:
        assert batch_size == 1, "Not supported for fixed world gen"
        if start_world_id is None:
            start_world_id = DEFAULT_WORLD_ID
        world = self.worlds[start_world_id]
        original_path = world.output_path
        dest_path = original_path.with_stem(original_path.stem + "_working")
        shutil.copytree(str(original_path), str(dest_path), dirs_exist_ok=True)
        copied_world = attr.evolve(world, output=str(dest_path))
        return [copied_world]


def disable_sigint():
    # Disable sigint handler so we can handle cleanup neatly ourselves.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# TODO: this introduces some non-determinism, fix it by anchoring to ids of levels generated
class LocalProcessWorldGenerator(WorldGenerator):
    def __init__(
        self,
        base_path: Path,
        seed: int,
        task_groups: Tuple[AvalonTaskGroup, ...],
        min_difficulty: float = 0,
        start_difficulty: float = 0,
        num_workers: int = 2,
        buffer_size: int = 20,
        offset: int = 0,
        is_task_curriculum_used: bool = True,
    ):
        super().__init__(base_path, seed)
        self.task_groups = task_groups
        self.min_difficulty = min_difficulty
        self.difficulties = {x: start_difficulty for x in get_all_tasks_for_task_groups(task_groups)}
        self.meta_difficulty = start_difficulty
        self.is_task_curriculum_used = is_task_curriculum_used
        self.rand = np.random.default_rng([self._seed])
        self.buffer: List[GeneratedWorldParams] = []
        self.lock = Lock()
        self.worker_pool = Pool(processes=num_workers, initializer=disable_sigint)
        self.offset = offset
        self._request_batch(buffer_size)

    def generate_batch(self, start_world_id: Optional[int], batch_size: int = 1) -> List[GeneratedWorldParams]:
        assert start_world_id is None, "Not supported for local process world gen"
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
        super().close()

    def _destroy_pool(self):
        self.worker_pool.close()
        # These are pretty quick, just let them complete.
        # self.worker_pool.terminate()
        self.worker_pool.join()
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
                    difficulty = self.rand.uniform(low=self.min_difficulty, high=self.difficulties[task])
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

    def set_task_difficulty(self, task: AvalonTask, difficulty: float):
        with self.lock:
            self.difficulties[task] = max(self.min_difficulty, difficulty)

    def set_meta_difficulty(self, difficulty: float):
        with self.lock:
            self.meta_difficulty = difficulty

    def set_seed(self, seed: int):
        super().set_seed(seed)
        # TODO: can this be a np.random.Generator as well?
        self.rand = Random(f"seed:{self._seed}")  # type: ignore


class TaskGenerationFunction(Protocol):
    def __call__(
        self,
        rand: np.random.Generator,
        difficulty: float,
        output_path: Path,
        export_config: ExportConfig,
        task_config: Any = TaskConfig(),
    ) -> None:
        ...


GENERATION_FUNCTION_BY_TASK: Final[Dict[AvalonTask, TaskGenerationFunction]] = {
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
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    output_path = Path(params.output)
    output_path.mkdir(parents=True)
    rand = np.random.default_rng([params.seed, params.index])
    export_config = get_agent_export_config()
    generation_function = GENERATION_FUNCTION_BY_TASK[params.task]
    for i in range(params.num_retries):
        try:
            generation_function(rand, params.difficulty, output_path, export_config)
            return GeneratedWorldParams.from_input(params)
        except ImpossibleWorldError as e:
            if i == params.num_retries - 1:
                logger.error("Ran out of retries to generate a good world!")
                logger.error(f"Params were: {params}")
                raise
            else:
                logger.info(f"Impossible world was generated, this was try {i}... (reason: {e})")
                # clear the output
                shutil.rmtree(output_path)
                output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            if i == params.num_retries - 1:
                logger.error("Ran out of retries to generate a good world!")
                logger.error(f"Params were: {params}")
                raise
            else:
                logger.error(f"Unspecified error in world generation, this was try {i}... (reason: {e})")
                # clear the output
                shutil.rmtree(output_path)
                output_path.mkdir(parents=True, exist_ok=True)
    raise SwitchError("This should never happen")


if __name__ == "__main__":
    generate_world(
        GenerateWorldParams(
            AvalonTask.MOVE,
            1,
            42,
            0,
            str(Path("./standalone/avalon/datagen/godot")),
        )
    )
