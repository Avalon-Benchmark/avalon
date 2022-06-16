import os
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import wandb

from agent.godot_gym import create_base_benchmark_config
from datagen.godot_env import AvalonObservationType
from datagen.godot_env import AvalonScoreEvaluator
from datagen.godot_env import GodotEnv
from datagen.godot_env import VRActionType
from datagen.world_creation.constants import AvalonTask
from datagen.world_creation.world_generator import GenerateWorldParams
from datagen.world_creation.world_generator import GeneratedWorldParams
from datagen.world_creation.world_generator import WorldGenerator

EVAL_TEMP_PATH = "/tmp/avalon_eval"


def get_world_folders(base_path: str):
    return sorted([x for x in os.listdir(base_path) if not x.startswith("practice")])


def world_id_from_world_folder_name(world_folder_name: str) -> int:
    return int(world_folder_name.split("__")[1])


class AvalonFolderWorldGenerator(WorldGenerator):
    def __init__(self, output_path: Path, seed: int = 0):
        self.output_path = output_path
        self.levels = {}
        for world_folder_name in get_world_folders(str(output_path)):
            if world_folder_name.startswith("practice"):
                continue
            dir_parts = world_folder_name.split("__")
            task = AvalonTask[dir_parts[0].upper()]
            index = world_id_from_world_folder_name(world_folder_name)
            if index in self.levels:
                raise Exception("Cannot have two levels with the same index!")
            difficulty = float(dir_parts[2].replace("_", "."))

            if task in (AvalonTask.SURVIVE, AvalonTask.FIND, AvalonTask.GATHER, AvalonTask.NAVIGATE):
                starting_hit_points = 3
            elif task in (AvalonTask.STACK, AvalonTask.CARRY, AvalonTask.EXPLORE):
                starting_hit_points = 2
            else:
                starting_hit_points = 1

            self.levels[index] = GeneratedWorldParams(
                task=task,
                difficulty=difficulty,
                seed=seed,
                index=index,
                starting_hit_points=starting_hit_points,
                output=str(output_path / world_folder_name),
            )

    def generate_batch(self, start_world_id: Optional[int], batch_size: int = 1) -> List[GenerateWorldParams]:
        assert batch_size == 1
        return [self.levels[start_world_id]]


class EvaluationGodotEnv(GodotEnv):
    def __init__(self, eval_world_dir="/mnt/private/avalon_worlds", eval_seed=0, is_symlink_generated: bool = True):
        # if is_symlink_generated and Path(GODOT_PATH).is_dir():
        #     link_path = Path(GODOT_PATH) / "worlds"
        #     if link_path.exists():
        #         os.unlink(link_path)
        #     os.symlink(eval_world_dir, Path(GODOT_PATH) / "worlds")
        base_config = create_base_benchmark_config()
        with base_config.mutable_clone() as config:
            config.player.total_energy_coefficient = 0

        self.eval_world_dir = eval_world_dir
        self.eval_world_ids = tuple(world_id_from_world_folder_name(x) for x in get_world_folders(eval_world_dir))
        self.eval_seed = eval_seed
        self.world_index = 0

        super().__init__(
            config=config,
            observation_type=AvalonObservationType,
            action_type=VRActionType,
            goal_evaluator=AvalonScoreEvaluator(),
        )

    def reset(self):
        world_id = self.eval_world_ids[self.world_index]
        self.world_index = (1 + self.world_index) % len(self.eval_world_ids)
        observation = self.reset_nicely(world_id=world_id)
        lame_observation = self.observation_context.lamify(observation)
        return lame_observation

    def _create_world_generator(self) -> WorldGenerator:
        return AvalonFolderWorldGenerator(output_path=Path(self.eval_world_dir), seed=self.eval_seed)

    def set_eval_world_ids(self, eval_world_ids: Tuple[int, ...]):
        self.eval_world_ids = eval_world_ids
        self.world_index = 0


def get_score(observations: List[AvalonObservationType]):
    initial_hp = observations[0].hit_points.item()
    score_hp: Optional[float] = None
    for obs in observations:
        if obs.is_food_present_in_world.item() == 0 and score_hp is None:
            score_hp = obs.hit_points.item()
    if observations[-1].is_dead or score_hp is None:
        score_hp = observations[-1].hit_points.item()
    return score_hp


def load_checkpoint_from_wandb_run(run_path: str, filename: str) -> str:
    api = wandb.Api()
    run = api.run(run_path)
    run_root = Path(EVAL_TEMP_PATH) / run_path
    os.makedirs(run_root, exist_ok=True)
    bones_checkpoint_path = wandb.restore(filename, run_path=run_path, replace=True, root=str(run_root))
    assert bones_checkpoint_path is not None, "Could not load checkpoint"
    return bones_checkpoint_path.name


def get_latest_checkpoint_filename(run_path: str, prefix="", suffix="") -> str:
    api = wandb.Api()
    run = api.run(run_path)
    checkpoint_filenames = [
        file.name for file in run.files() if file.name.startswith(prefix) and file.name.endswith(suffix)
    ]
    checkpoint_filenames = sorted(checkpoint_filenames, key=lambda x: int(x[len(prefix) : -len(suffix)]))
    return checkpoint_filenames[-1]


def get_wandb_result_key(wandb_run: str, checkpoint_filename: str, data_key: str):
    return "__".join([wandb_run, checkpoint_filename, data_key]).replace("/", "_")
