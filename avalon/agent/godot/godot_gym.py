import copy
import json
import os
import uuid
from collections import defaultdict
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import Hashable
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import attr
import gym
import numpy as np
import torch
from PIL import Image
from einops import rearrange
from gym import Wrapper
from numpy.typing import NDArray
from torch import Tensor

from avalon.agent.common.params import EnvironmentParams
from avalon.common.errors import SwitchError
from avalon.common.utils import DATA_FOLDER
from avalon.contrib.utils import FILESYSTEM_ROOT
from avalon.datagen.godot_base_types import Vector3
from avalon.datagen.godot_env.actions import VRAction
from avalon.datagen.godot_env.goals import AvalonGoalEvaluator
from avalon.datagen.godot_env.goals import GoalProgressResult
from avalon.datagen.godot_env.goals import TrainingAvalonGoalEvaluator
from avalon.datagen.godot_env.godot_env import GodotEnv
from avalon.datagen.godot_env.observations import AvalonObservation
from avalon.datagen.godot_generated_types import AvalonSimSpec
from avalon.datagen.godot_generated_types import RecordingOptionsSpec
from avalon.datagen.godot_generated_types import VRAgentPlayerSpec
from avalon.datagen.godot_utils import S3_AVALON_ERROR_BUCKET
from avalon.datagen.world_creation.constants import TRAIN_TASK_GROUPS
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.constants import AvalonTaskGroup
from avalon.datagen.world_creation.constants import get_all_tasks_for_task_groups
from avalon.datagen.world_creation.constants import int_to_avalon_task
from avalon.datagen.world_creation.entities.spawn_point import PLAYER_SPAWN_POINT
from avalon.datagen.world_creation.world_generator import AvalonWorldGenerator
from avalon.datagen.world_creation.world_generator import BlockingWorldGenerator
from avalon.datagen.world_creation.world_generator import FixedWorldGenerator
from avalon.datagen.world_creation.world_generator import FixedWorldLoader
from avalon.datagen.world_creation.world_generator import GeneratedAvalonWorldParams
from avalon.datagen.world_creation.world_generator import LocalProcessWorldGenerator
from avalon.datagen.world_creation.world_generator import is_fixed_world_generator


def create_base_benchmark_config(
    data_dir: str = str(uuid.uuid4()),
    random_int: int = 0,
    num_frames_alive_after_food_is_gone: int = 20,
    resolution: int = 96,
) -> AvalonSimSpec:
    return AvalonSimSpec(
        episode_seed=0,
        dir_root=os.path.join(DATA_FOLDER, data_dir),
        recording_options=RecordingOptionsSpec(
            user_id=None,
            apk_version=None,
            recorder_host=None,
            recorder_port=None,
            is_teleporter_enabled=False,
            resolution_x=resolution,
            resolution_y=resolution,
            is_recording_human_actions=False,
            is_remote_recording_enabled=False,
            is_adding_debugging_views=False,  # Set to True for debugging videos
            is_debugging_output_requested=False,
        ),
        player=VRAgentPlayerSpec(
            spawn_point_name=PLAYER_SPAWN_POINT,
            max_head_linear_speed=0.3,
            max_head_angular_speed=Vector3(5.0, 10.0, 1.0),
            max_hand_linear_speed=1.0,
            max_hand_angular_speed=15.0,
            jump_height=1.5,
            arm_length=1,
            starting_hit_points=1.0,
            mass=60.0,
            arm_mass_ratio=0.05,
            head_mass_ratio=0.08,
            standup_speed_after_climbing=0.1,
            min_head_position_off_of_floor=0.2,
            push_force_magnitude=5.0,
            throw_force_magnitude=3.0,
            starting_left_hand_position_relative_to_head=Vector3(-0.5, -0.5, -0.5),
            starting_right_hand_position_relative_to_head=Vector3(0.5, -0.5, -0.5),
            minimum_fall_speed=10.0,
            fall_damage_coefficient=0.000026,
            num_frames_alive_after_food_is_gone=num_frames_alive_after_food_is_gone,
            eat_area_radius=0.5,
            is_displaying_debug_meshes=False,  # Set to True for debugging videos
            is_human_playback_enabled=False,
            is_slowed_from_crouching=True,
        ),
    )


LEVEL_OUTPUT_PATH = f"{FILESYSTEM_ROOT}/data/level_gen"
CURRICULUM_BASE_PATH = Path(DATA_FOLDER) / "curriculum"


class TrainingProtocolChoice(Enum):
    MULTI_TASK_ALL = "MULTI_TASK_ALL"
    MULTI_TASK_EASY = "MULTI_TASK_EASY"  # Just MOVE and EAT
    MULTI_TASK_BASIC = "MULTI_TASK_BASIC"  # All non-compositional tasks
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
) -> Tuple[AvalonTaskGroup, ...]:
    if training_protocol == TrainingProtocolChoice.MULTI_TASK_BASIC:
        if is_meta_curriculum_used:
            return TRAIN_TASK_GROUPS[:4]
        else:
            return (AvalonTaskGroup.SIMPLE,)
    elif training_protocol == TrainingProtocolChoice.MULTI_TASK_EASY:
        if is_meta_curriculum_used:
            return TRAIN_TASK_GROUPS[:1]
        else:
            return (AvalonTaskGroup.EASY,)
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


@attr.s(auto_attribs=True, frozen=True)
class GodotEnvironmentParams(EnvironmentParams):
    suite: str = "godot"
    # seed: int = 0
    # whether to use per-task curriculum learning. requires a wrapper.
    is_task_curriculum_used: bool = True
    # how much to update the task difficulty per successful or failed episode.
    task_difficulty_update: float = 0.01
    # whether to use a meta-curriculum. requires a wrapper.
    is_meta_curriculum_used: bool = False
    # if using the meta curriculum, the update size
    meta_difficulty_update: float = 0.01
    # initial difficulty value
    initial_difficulty: float = 0
    # can use to give a small per-step penalty for using energy (by moving)
    energy_cost_coefficient: float = 0
    # energy cost for each axis
    head_roll_coefficient: float = 1e-6
    # energy cost for each axis
    head_pitch_coefficient: float = 1e-6
    energy_cost_aggregator: Literal["sum", "max"] = "sum"
    # rgbd image will have resolution (resolution * resolution)
    resolution: int = 96
    greyscale: bool = False
    # dense reward is probably broken; we only use sparse
    is_reward_dense: bool = False
    # the episode will continue for this many steps after eating a food
    num_frames_alive_after_food_is_gone: int = 20
    level_output_base_path: str = f"{FILESYSTEM_ROOT}/data/level_gen"
    # which tasks to train on
    training_protocol: TrainingProtocolChoice = TrainingProtocolChoice.MULTI_TASK_BASIC
    # debug output
    is_debugging_godot: bool = False
    # debug output
    is_video_logged: bool = False
    # debug output
    is_action_logged: bool = False
    # debug output
    is_logging_artifacts_on_error_to_s3: bool = True
    s3_bucket_name: Optional[str] = S3_AVALON_ERROR_BUCKET
    # debug output
    goal_progress_path: Optional[Path] = None

    is_frame_id_transformed_to_frames_remaining: bool = True

    # which for godot to use
    gpu_id: int = 0

    # Evaluation world params
    # fill this in if you want to use fixed worlds for testing
    fixed_worlds_s3_key: Optional[str] = None
    # this will be auto-filled
    fixed_worlds_load_from_path: Optional[Path] = None
    fixed_world_min_difficulty: float = 0
    fixed_world_max_difficulty: float = 1
    val_episodes_per_task: int = 11
    test_episodes_per_task: int = 101
    # difficulty bin size for eval logging histograms
    eval_difficulty_bin_size: float = 0.1

    info_fields: list[str] = [
        "cumulative_episode_return",
        "cumulative_episode_length",
        "task",
        "difficulty",
        "success",
        "score",
        "world_index",
    ]

    @property
    def is_fixed_generator(self) -> bool:
        """Use the `fixed` world genenerator for evaluation."""
        return self.mode != "train"

    @property
    def num_fixed_worlds_per_task(self) -> int:
        """How many evaluation worlds to run per task."""
        if self.mode == "test":
            return self.test_episodes_per_task
        elif self.mode == "val":
            return self.val_episodes_per_task
        else:
            raise SwitchError("mode must be one of {'test', 'val'}")

    @property
    def task_groups(self) -> Tuple[AvalonTaskGroup, ...]:
        return task_groups_from_training_protocol(self.training_protocol, self.is_meta_curriculum_used)

    @property
    def num_tasks(self) -> int:
        return len(get_all_tasks_for_task_groups(self.task_groups))

    @property
    def seed(self) -> int:
        return self.env_index


def write_video_from_np_arrays(video_path: Path, arrays: List[np.ndarray], fps: float = 20) -> None:
    # size = arrays[0].shape[:2]
    # out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size, False)
    # for array in arrays:
    #     out.write(array[::-1, :, :3])
    # out.release()
    images = [Image.fromarray(x) for x in arrays]
    with open(video_path, "wb") as f:
        images[0].save(f, save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)


class AvalonEnv(GodotEnv[AvalonObservation, VRAction, GeneratedAvalonWorldParams]):
    def __init__(self, params: GodotEnvironmentParams) -> None:
        self.params = params

        base_config = create_base_benchmark_config(
            random_int=params.seed,
            num_frames_alive_after_food_is_gone=params.num_frames_alive_after_food_is_gone,
            resolution=params.resolution,
        )
        with base_config.mutable_clone() as config:
            config.recording_options.is_debugging_output_requested = params.is_debugging_godot
            if params.mode == "test" and params.is_video_logged:
                config.player.is_displaying_debug_meshes = True
                config.recording_options.is_adding_debugging_views = True

        self.episode_frames: List[np.ndarray] = []
        if params.mode == "train":
            goal_evaluator: AvalonGoalEvaluator = TrainingAvalonGoalEvaluator(
                energy_cost_coefficient=params.energy_cost_coefficient,
                head_pitch_coefficient=params.head_pitch_coefficient,
                head_roll_coefficient=params.head_roll_coefficient,
            )
        else:
            goal_evaluator = AvalonGoalEvaluator()

        self.current_world_id: int = 0

        self.episode_goal_progress: List[GoalProgressResult] = []
        self.episode_observations: List[AvalonObservation] = []
        self.episode_actions: List[VRAction] = []
        self.eval_world_ids: List[int] = []

        super().__init__(
            config=config,
            observation_type=AvalonObservation,
            action_type=VRAction,
            goal_evaluator=goal_evaluator,
            is_error_log_checked_after_each_step=params.is_debugging_godot,
            is_godot_restarted_on_error=not params.is_debugging_godot,
            gpu_id=params.gpu_id,
            is_logging_artifacts_on_error_to_s3=params.is_logging_artifacts_on_error_to_s3,
            s3_bucket_name=params.s3_bucket_name,
        )
        # We manually indicate to mypy that self.world_generator will be the one we pass above as otherwise it
        # infers the generic type WorldGenerator[GeneratedAvalonWorldParams] and does not locate Avalon-specific methods
        self.world_generator: AvalonWorldGenerator

    def act(self, action: VRAction) -> Tuple[AvalonObservation, GoalProgressResult]:
        observation, goal_progress = super().act(action)
        if self.params.goal_progress_path is not None:
            self.episode_goal_progress.append(goal_progress)
            if self.params.is_video_logged:
                self.episode_observations.append(observation)
            if self.params.is_action_logged:
                self.episode_actions.append(action)
            if goal_progress.is_done:
                assert goal_progress.world_path is not None
                log_goal_progress_path = self.params.goal_progress_path / Path(goal_progress.world_path).relative_to(
                    LEVEL_OUTPUT_PATH
                )
                if not log_goal_progress_path.exists():
                    os.makedirs(log_goal_progress_path)
                    with open(log_goal_progress_path / "goal_progress.json", "w") as f:
                        json.dump([attr.asdict(x) for x in self.episode_goal_progress], f)
                    if self.params.is_video_logged:
                        write_video_from_np_arrays(
                            log_goal_progress_path / "video.png",
                            [
                                np.concatenate([x.rgbd, x.top_down_rgbd, x.isometric_rgbd], axis=1)[::-1, :, :3]
                                for x in self.episode_observations
                            ],
                        )
                        # write_video_from_np_arrays(
                        #     log_goal_progress_path / "top_down_video.png",
                        #     [x.top_down_rgbd[::-1, :, :3] for x in self.episode_observations],
                        # )
                        # write_video_from_np_arrays(
                        #     log_goal_progress_path / "isometric_video.png",
                        #     [x.isometric_rgbd[::-1, :, :3] for x in self.episode_observations],
                        # )
                    if self.params.is_action_logged:
                        with open(log_goal_progress_path / "actions.json", "w") as f:
                            json.dump([attr.asdict(x) for x in self.episode_actions], f)
                self.episode_goal_progress = []
                self.episode_observations = []
                self.episode_actions = []

        return observation, goal_progress

    def update_lame_observation(self, lame_observation: Dict[str, Any]) -> None:
        if self.params.is_frame_id_transformed_to_frames_remaining and "frame_id" in lame_observation:
            assert isinstance(self.goal_evaluator, AvalonGoalEvaluator)
            lame_observation["frame_id"] = self.goal_evaluator.frame_limit - lame_observation["frame_id"] - 1

    def step(self, action: Dict[str, np.ndarray]) -> tuple[AvalonObservation, float, bool, dict]:
        observation, goal_progress = self.act(self.action_type.from_input(action))
        lame_observation = self.observation_context.lamify(observation)
        self.update_lame_observation(lame_observation)
        return lame_observation, goal_progress.reward, goal_progress.is_done, goal_progress.log

    def reset(self):
        world_id = None
        if self.params.is_fixed_generator:
            if len(self.eval_world_ids) == 0:
                assert is_fixed_world_generator(self.world_generator)
                self.eval_world_ids = sorted(self.world_generator.worlds.keys())

            world_id = self.eval_world_ids[self.current_world_id]
            self.current_world_id = (self.current_world_id + 1) % len(self.eval_world_ids)

        observation = self.reset_nicely(world_id=world_id)
        if self.params.is_video_logged:
            self.episode_observations.append(observation)
        lame_observation = self.observation_context.lamify(observation)
        self.update_lame_observation(lame_observation)
        return lame_observation

    def _create_world_generator(self) -> AvalonWorldGenerator:
        base_path = Path(self.params.level_output_base_path)
        if self.params.is_fixed_generator:
            if self.params.fixed_worlds_load_from_path:
                return FixedWorldLoader(
                    base_path=base_path,
                    generator_index=self.params.env_index,
                    num_generators=self.params.env_count,
                    generated_worlds_path=self.params.fixed_worlds_load_from_path,
                )
            difficulties = tuple(
                np.linspace(
                    self.params.fixed_world_min_difficulty,
                    self.params.fixed_world_max_difficulty,
                    self.params.num_fixed_worlds_per_task,
                )
            )
            return FixedWorldGenerator(
                base_path=base_path,
                seed=self.params.seed,
                difficulties=difficulties,
                task_groups=self.params.task_groups,
                generator_index=self.params.env_index,
                num_generators=self.params.env_count,
            )
        return LocalProcessWorldGenerator(
            base_path=base_path,
            seed=self.params.seed,
            min_difficulty=0,
            start_difficulty=self.params.initial_difficulty,
            task_groups=self.params.task_groups,
            is_task_curriculum_used=self.params.is_task_curriculum_used,
        )

    @property
    def curriculum_base_path(self) -> Path:
        # TODO make this not collide
        return CURRICULUM_BASE_PATH

    @property
    def curriculum_save_path(self) -> Path:
        if not self.curriculum_base_path.exists():
            os.makedirs(self.curriculum_base_path, exist_ok=True)
        return self.curriculum_base_path / f"{self.params.env_index}.pkl"

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

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """See __getstate__ docstring

        Args:
            state (dict): Unpickled state.

        """

        self.__dict__.update(state)
        self._restart_process()
        self.seed_nicely(state["_latest_seed"])

    # task functions from ray.rllib.env.apis.task_settable_env
    def sample_tasks(self, n_tasks: int) -> List[Hashable]:
        if self.params.is_fixed_generator:
            return [0] * n_tasks
        else:
            return self.world_generator.sample_tasks(n_tasks)

    def set_task(self, task: Hashable) -> None:
        if not self.params.is_fixed_generator:
            self.world_generator.set_task(task)
        else:
            assert False

    def set_task_difficulty(self, task: AvalonTask, difficulty: float) -> None:
        assert not self.params.is_fixed_generator
        assert isinstance(self.world_generator, (LocalProcessWorldGenerator, BlockingWorldGenerator))
        self.world_generator.set_task_difficulty(task, difficulty)

    def set_meta_difficulty(self, difficulty: float) -> None:
        assert not self.params.is_fixed_generator
        assert isinstance(self.world_generator, (LocalProcessWorldGenerator, BlockingWorldGenerator))
        self.world_generator.set_meta_difficulty(difficulty)

    def get_task(self) -> Hashable:
        if self.params.is_fixed_generator:
            return 0
        else:
            return self.world_generator.get_task()

    def _get_world_params_by_id(self, world_id: Optional[int]) -> GeneratedAvalonWorldParams:
        if world_id is not None:
            already_generated = self.world_generator.load_already_generated_params(world_id)
            if already_generated is not None:
                return already_generated
        return super()._get_world_params_by_id(world_id)


def _transform_observation_rgbd(x: NDArray, greyscale: bool = False) -> NDArray:
    # flip y
    x = x[::-1]
    if greyscale:
        x = x[:, :, 0:1]
    # torch ordering
    x = rearrange(x, "h w c -> c h w")
    # Copy here because torch doesn't like strided arrays
    return x.copy()


def _transform_observation_rgbd_post(x: Tensor) -> Tensor:
    assert x.dtype in (np.uint8, torch.uint8)
    return x / 255.0 - 0.5


def _transform_observation_scale_bias(x: NDArray, scale: float = 1.0, bias: float = 0.0) -> NDArray:
    return x * scale + bias


def _transform_observation_clamp(x: NDArray, min: Optional[float] = None, max: Optional[float] = None) -> NDArray:
    return np.clip(x, a_min=min, a_max=max)  # type: ignore


class GodotObsTransformWrapper(gym.ObservationWrapper):
    def __init__(self, env: GodotEnv, greyscale: bool = False) -> None:
        super().__init__(env)
        self.transforms = {
            "rgbd": partial(_transform_observation_rgbd, greyscale=greyscale),
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
            "frame_id": partial(_transform_observation_scale_bias, scale=0.0001),
        }
        assert isinstance(env.observation_space, gym.spaces.Dict)
        self.observation_space = gym.spaces.Dict()
        if "rgbd" in env.observation_space.spaces:
            shape = env.observation_space.spaces["rgbd"].shape
            shape = (1 if greyscale else shape[2], shape[0], shape[1])
            self.observation_space.spaces["rgbd"] = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

        # really these are vector keys
        self.scalar_keys = {
            k: v.shape[0]
            for k, v in env.observation_space.spaces.items()
            if isinstance(v, gym.spaces.Box) and len(v.shape) == 1
        }
        self.vec_len = sum(self.scalar_keys.values())
        assert len(env.observation_space.spaces) == len(self.observation_space.spaces) + len(self.scalar_keys)
        self.observation_space["scalars"] = gym.spaces.Box(low=-1, high=1, shape=(self.vec_len,), dtype=np.float32)

    def observation(self, observation: Dict[str, Any]):
        transformed = {k: self.transforms[k](v) if k in self.transforms else v for k, v in observation.items()}
        out = {}
        if "rgbd" in transformed:
            out["rgbd"] = transformed["rgbd"]
        scalar_outputs = np.zeros((self.vec_len,), dtype=np.float32)
        i = 0
        for k, v in transformed.items():
            if k == "rgbd":
                continue
            scalar_outputs[i : i + len(v)] = v
            i += len(v)
        # Enforce the bounds of the scalar space
        scalar_outputs = np.clip(
            scalar_outputs, self.observation_space["scalars"].low, self.observation_space["scalars"].high  # type: ignore
        )
        out["scalars"] = scalar_outputs
        return out


def squash_real(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -4, 4)
    return np.tanh(x)


class ScaleAndSquashAction(gym.ActionWrapper):
    """Constrain to within [-1, 1] with a tanh and then scale the actions."""

    def __init__(self, env: gym.Env, key: str = "real", scale: float = 1) -> None:
        super().__init__(env)
        self.key = key
        self.scale = scale

    def action(self, action: Dict[str, np.ndarray]):
        action_dict = copy.deepcopy(action)
        action_dict[self.key] = squash_real(action_dict[self.key] * self.scale)
        return action_dict

    def reverse_action(self, action: Dict[str, np.ndarray]):
        raise NotImplementedError


# Note: the rate of update of difficulty (per global env step) depends on the number of workers.
class CurriculumWrapper(Wrapper):
    def __init__(self, env: AvalonEnv, task_difficulty_update: float, meta_difficulty_update: float) -> None:
        super().__init__(env)
        self._env = env
        self.difficulties: DefaultDict[AvalonTask, float] = defaultdict(float)
        self.task_difficulty_update = task_difficulty_update
        self.meta_difficulty_update = meta_difficulty_update
        self.meta_difficulty = 0.0

    def step(self, action: Dict[str, torch.Tensor]):
        observation, reward, done, info = self.env.step(action)  # type: ignore[misc]
        if done:
            task = AvalonTask[int_to_avalon_task[int(info["task"])]]
            update_step = self.task_difficulty_update  # * np.random.uniform()
            if info["success"] == 1:
                self.difficulties[task] += update_step
                self.meta_difficulty += self.meta_difficulty_update
            elif info["success"] == 0:
                self.difficulties[task] -= update_step
                self.meta_difficulty -= self.meta_difficulty_update
            else:
                assert False, info["success"]
            self.difficulties[task] = max(min(self.difficulties[task], 1.0), 0.0)
            self.meta_difficulty = max(min(self.meta_difficulty, 1.0), 0.0)
            self._env.set_task_difficulty(task, self.difficulties[task])
            self._env.set_meta_difficulty(self.meta_difficulty)
        return observation, reward, done, info
