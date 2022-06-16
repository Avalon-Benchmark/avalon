import copy
import os
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Hashable
from typing import List
from typing import Optional
from typing import Tuple

import attr
import gym
import numpy as np
import numpy.typing as npt
import torch
from einops import rearrange
from torch import Tensor

from common.utils import DATA_FOLDER
from contrib.serialization import Serializable
from datagen.godot_base_types import Vector3
from datagen.godot_env import AvalonObservationType
from datagen.godot_env import DynamicFrameLimitGodotGoalEvaluator
from datagen.godot_env import GodotEnv
from datagen.godot_env import VRActionType
from datagen.godot_generated_types import AvalonSimSpec
from datagen.godot_generated_types import RecordingOptionsSpec
from datagen.godot_generated_types import VRAgentPlayerSpec
from datagen.world_creation.constants import TRAIN_TASK_GROUPS
from datagen.world_creation.constants import AvalonTaskGroup
from datagen.world_creation.constants import get_all_tasks_for_task_groups
from datagen.world_creation.world_generator import FixedWorldGenerator
from datagen.world_creation.world_generator import LocalProcessWorldGenerator
from datagen.world_creation.world_generator import WorldGenerator


def create_base_benchmark_config(data_dir: str = "interactive", random_int: int = 0) -> AvalonSimSpec:
    return AvalonSimSpec(
        dataset_id=0,
        label=0,
        video_min=0,
        video_max=750,
        frame_max=100,
        random_int=random_int,
        random_key="{dir_root}",
        dir_root=os.path.join(DATA_FOLDER, data_dir),
        output_file_name_format="{dir_root}/{video_id}/{feature}__{type_and_dim}.raw",
        scene_path="res://scenes/empty.tscn",
        is_using_shared_caches=False,
        is_generating_paired_videos=False,
        recording_options=RecordingOptionsSpec(
            user_id=None,
            apk_version=None,
            recorder_host=None,
            recorder_port=None,
            is_teleporter_enabled=False,
            resolution_x=96,
            resolution_y=96,
            is_recording_rgb=True,
            is_recording_depth=False,
            is_recording_human_actions=False,
            is_remote_recording_enabled=False,
            is_adding_debugging_views=False,
            is_debugging_output_requested=False,
        ),
        player=VRAgentPlayerSpec(
            max_head_linear_speed=0.3,
            max_head_angular_speed=Vector3(5.0, 10.0, 1.0),
            max_hand_linear_speed=1.0,
            max_hand_angular_speed=15.0,
            jump_height=1.5,
            arm_length=1,
            starting_hit_points=1.0,
            mass=1.0,
            arm_mass_ratio=0.05,
            head_mass_ratio=0.08,
            standup_speed_after_climbing=0.1,
            min_head_position_off_of_floor=0.2,
            push_force_magnitude=5.0,
            throw_force_magnitude=3.0,
            starting_left_hand_position_relative_to_head=Vector3(-0.5, -0.5, -0.5),
            starting_right_hand_position_relative_to_head=Vector3(0.5, -0.5, -0.5),
            minimum_fall_speed=10.0,
            fall_damage_coefficient=1.0,
            total_energy_coefficient=1e-4,
            body_kinetic_energy_coefficient=1.0,
            body_potential_energy_coefficient=1.0,
            head_potential_energy_coefficient=1.0,
            left_hand_kinetic_energy_coefficient=1.0,
            left_hand_potential_energy_coefficient=1.0,
            right_hand_kinetic_energy_coefficient=1.0,
            right_hand_potential_energy_coefficient=1.0,
            num_frames_alive_after_food_is_gone=10,
            eat_area_radius=0.5,
            is_displaying_debug_meshes=False,
        ),
    )


LEVEL_OUTPUT_PATH = "/mnt/private/data/level_gen"


class TrainingProtocolChoice(Enum):
    MULTI_TASK_ALL = "MULTI_TASK_ALL"
    MULTI_TASK_BASIC = "MULTI_TASK_BASIC"
    MULTI_TASK_COMPOSITIONAL = "MULTI_TASK_COMPOSITIONAL"
    SINGLE_TASK_EAT = "SINGLE_TASK_EAT"
    SINGLE_TASK_MOVE = "SINGLE_TASK_MOVE"
    SINGLE_TASK_JUMP = "SINGLE_TASK_JUMP"
    SINGLE_TASK_EXPLORE = "SINGLE_TASK_EXPLORE"
    SINGLE_TASK_SCRAMBLE = "SINGLE_TASK_SCRAMBLE"
    SINGLE_TASK_CLIMB = "SINGLE_TASK_CLIMB"
    SINGLE_TASK_DESCEND = "SINGLE_TASK_DESCEND"
    SINGLE_TASK_THROW = "SINGLE_TASK_THROW"
    SINGLE_TASK_AVOID = "SINGLE_TASK_AVOID"
    SINGLE_TASK_HUNT = "SINGLE_TASK_HUNT"
    SINGLE_TASK_FIGHT = "SINGLE_TASK_FIGHT"
    SINGLE_TASK_PUSH = "SINGLE_TASK_PUSH"
    SINGLE_TASK_STACK = "SINGLE_TASK_STACK"
    SINGLE_TASK_BRIDGE = "SINGLE_TASK_BRIDGE"
    SINGLE_TASK_OPEN = "SINGLE_TASK_OPEN"
    SINGLE_TASK_CARRY = "SINGLE_TASK_CARRY"
    SINGLE_TASK_NAVIGATE = "SINGLE_TASK_NAVIGATE"
    SINGLE_TASK_FIND = "SINGLE_TASK_FIND"
    SINGLE_TASK_GATHER = "SINGLE_TASK_GATHER"
    SINGLE_TASK_SURVIVE = "SINGLE_TASK_SURVIVE"


def task_groups_from_training_protocol(
    training_protocol: TrainingProtocolChoice, is_meta_curriculum_used: bool
) -> Tuple[AvalonTaskGroup]:
    if training_protocol == TrainingProtocolChoice.MULTI_TASK_BASIC:
        if is_meta_curriculum_used:
            return TRAIN_TASK_GROUPS[:4]
        else:
            return (AvalonTaskGroup.SIMPLE,)
    elif training_protocol == TrainingProtocolChoice.MULTI_TASK_ALL:
        if is_meta_curriculum_used:
            return TRAIN_TASK_GROUPS
        else:
            return (AvalonTaskGroup.ALL,)
    elif training_protocol == TrainingProtocolChoice.MULTI_TASK_COMPOSITIONAL:
        return (AvalonTaskGroup.COMPOSITIONAL,)
    else:
        assert training_protocol.value.startswith("SINGLE_TASK")
        task_name = training_protocol.value.split("_")[2]
        return (AvalonTaskGroup[task_name],)


@attr.s(auto_attribs=True, collect_by_mro=True)
class GodotEnvParams(Serializable):
    random_int: int
    gpu_id: int
    max_frames: int = 0
    energy_cost_coefficient: float = 0
    is_fixed_generator: bool = False
    num_fixed_worlds_per_task: int = 6
    fixed_world_min_difficulty: float = 0
    fixed_world_max_difficulty: float = 1
    training_protocol: TrainingProtocolChoice = TrainingProtocolChoice.MULTI_TASK_BASIC
    is_task_curriculum_used: bool = True
    is_meta_curriculum_used: bool = True


class AvalonGodotEnvWrapper(GodotEnv[AvalonObservationType, VRActionType]):
    def __init__(self, params: GodotEnvParams):
        self.params = params

        base_config = create_base_benchmark_config(f"seed{params.random_int}", random_int=params.random_int)

        self.episode_frames: List[np.ndarray] = []
        goal_evaluator = DynamicFrameLimitGodotGoalEvaluator(max_frames=params.max_frames)
        self.current_world_id: int = 0
        self.task_groups = task_groups_from_training_protocol(params.training_protocol, params.is_meta_curriculum_used)
        self.num_tasks = len(get_all_tasks_for_task_groups(self.task_groups))

        if self.params.is_fixed_generator:
            self.unique_envs = self.num_tasks * self.params.num_fixed_worlds_per_task
        else:
            self.unique_envs = 0

        with base_config.mutable_clone() as config:
            config.player.total_energy_coefficient = params.energy_cost_coefficient

        super().__init__(
            config=config,
            observation_type=AvalonObservationType,
            action_type=VRActionType,
            goal_evaluator=goal_evaluator,
            is_error_log_checked_after_each_step=False,
            is_godot_restarted_on_error=True,
            gpu_id=params.gpu_id,
        )

    def reset(self):
        world_id = None
        if self.unique_envs:
            world_id = self.current_world_id
            self.current_world_id = (self.current_world_id + 1) % self.unique_envs

        observation = self.reset_nicely(world_id=world_id)
        lame_observation = self.observation_context.lamify(observation)
        return lame_observation

    def _create_world_generator(self) -> WorldGenerator:
        output_path = Path(LEVEL_OUTPUT_PATH)
        os.makedirs(str(output_path), exist_ok=True)
        if self.params.is_fixed_generator:
            difficulties = tuple(
                np.linspace(
                    self.params.fixed_world_min_difficulty,
                    self.params.fixed_world_max_difficulty,
                    self.params.num_fixed_worlds_per_task,
                )
            )
            world_generator = FixedWorldGenerator(
                output_path=output_path,
                seed=self.params.random_int,
                difficulties=difficulties,
                task_groups=self.task_groups,
                num_worlds_per_task_difficulty_pair=1,
            )
            # if wandb.run:
            #     for world in world_generator.worlds:
            #         wandb.save(world.output + "/*", base_path=LEVEL_OUTPUT_PATH, policy="now")
        else:
            world_generator = LocalProcessWorldGenerator(
                output_path=output_path,
                seed=self.params.random_int,
                start_difficulty=0.0,
                task_groups=self.task_groups,
                is_task_curriculum_used=self.params.is_task_curriculum_used,
            )

        return world_generator

    def __getstate__(self) -> Dict[str, Any]:
        """Make a pickle state for garage to serialize aggressively, not actually the same state, in particular will
        start from the next video

        Returns:
            dict: The pickled state.

        """
        state = self.__dict__.copy()
        # Don't pickle the process
        state["process"] = None
        state["world_generator"] = None
        state["_bridge"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]):
        """See __getstate__ docstring

        Args:
            state (dict): Unpickled state.

        """

        self.__dict__.update(state)
        self._restart_process()
        self.seed_nicely(state["_latest_seed"], -1)

    # task functions from ray.rllib.env.apis.task_settable_env
    def sample_tasks(self, n_tasks: int) -> List[Hashable]:
        if self.unique_envs:
            return [0] * n_tasks
        else:
            return self.world_generator.sample_tasks(n_tasks)

    def set_task(self, task: Hashable) -> None:
        if not self.unique_envs:
            self.world_generator.set_task(task)
        else:
            assert False

    def set_task_difficulty(self, task, difficulty):
        assert self.unique_envs == 0
        self.world_generator.set_task_difficulty(task, difficulty)

    def set_meta_difficulty(self, difficulty):
        assert self.unique_envs == 0
        self.world_generator.set_meta_difficulty(difficulty)

    def get_task(self) -> Hashable:
        if self.unique_envs:
            return 0
        else:
            return self.world_generator.get_task()


def _transform_observation_rgbd(x: npt.NDArray) -> npt.NDArray:
    # flip y
    x = x[::-1]
    # torch ordering
    x = rearrange(x, "h w c -> c h w")
    # -0.5 to 0.5
    return x.astype(np.float32) / 255.0 - 0.5


def _transform_observation_depth(x: Tensor, depth_scale_length: float = 1.0) -> Tensor:
    return torch.clamp(depth_scale_length / (depth_scale_length + x), max=depth_scale_length)


def _transform_observation_scale_bias(x: Tensor, scale: float = 1.0, bias: float = 0.0) -> Tensor:
    return x * scale + bias


def _transform_observation_clamp(
    x: np.ndarray, min: Optional[float] = None, max: Optional[float] = None
) -> np.ndarray:
    return np.clip(x, a_min=min, a_max=max)


def _transform_noop(x: Tensor) -> Tensor:
    return x


def flatten_and_transform_observation_dict(
    observation: Dict[str, Tensor], flattened_observation_keys: List[str]
) -> Tensor:
    return torch.cat([observation[x] for x in flattened_observation_keys], dim=-1)


class GodotObsTransformWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.transforms = {
            "rgbd": _transform_observation_rgbd,
            "physical_body_delta_position": partial(_transform_observation_scale_bias, scale=0.1),
            "physical_body_delta_rotation": partial(_transform_observation_scale_bias, scale=0.005),
            "physical_head_delta_position": partial(_transform_observation_scale_bias, scale=0.1),
            "physical_left_hand_delta_position": partial(_transform_observation_scale_bias, scale=0.1),
            "physical_right_hand_delta_position": partial(_transform_observation_scale_bias, scale=0.1),
            "physical_head_delta_rotation": partial(_transform_observation_scale_bias, scale=0.005),
            "physical_left_hand_delta_rotation": partial(_transform_observation_scale_bias, scale=0.005),
            "physical_right_hand_delta_rotation": partial(_transform_observation_scale_bias, scale=0.005),
            "physical_head_relative_position": partial(_transform_observation_scale_bias, scale=0.1),
            "physical_left_hand_relative_position": partial(_transform_observation_scale_bias, scale=0.1),
            "physical_right_hand_relative_position": partial(_transform_observation_scale_bias, scale=0.1),
            "physical_head_relative_rotation": partial(_transform_observation_scale_bias, scale=0.005),
            "physical_left_hand_relative_rotation": partial(_transform_observation_scale_bias, scale=0.005),
            "physical_right_hand_relative_rotation": partial(_transform_observation_scale_bias, scale=0.005),
            "left_hand_thing_colliding_with_hand": partial(_transform_observation_clamp, min=0, max=1),
            "left_hand_held_thing": partial(_transform_observation_clamp, min=0, max=1),
            "right_hand_thing_colliding_with_hand": partial(_transform_observation_clamp, min=0, max=1),
            "right_hand_held_thing": partial(_transform_observation_clamp, min=0, max=1),
        }
        self.observation_space = copy.deepcopy(env.observation_space)
        if "rgbd" in env.observation_space.spaces:
            shape = env.observation_space.spaces["rgbd"].shape
            shape = (shape[2], shape[0], shape[1])
            self.observation_space.spaces["rgbd"] = gym.spaces.Box(low=-0.5, high=0.5, shape=shape, dtype=np.float32)

        # TODO: do we need to change the other spaces too? are the min/maxes actually used anywhere?

    def observation(self, observation):
        return {k: self.transforms[k](v) if k in self.transforms else v for k, v in observation.items()}


def squash_real(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -4, 4)
    return np.tanh(x)


class ScaleAndSquashAction(gym.ActionWrapper):
    """Constrain to within [-1, 1] with a tanh and then scale the actions."""

    def __init__(self, env, key: str = "real", scale: float = 1):
        super().__init__(env)
        self.key = key
        self.scale = scale

    def action(self, action: Dict[str, np.ndarray]):
        action_dict = copy.deepcopy(action)
        action_dict[self.key] = squash_real(action_dict[self.key] * self.scale)
        return action_dict
